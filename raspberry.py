#!/usr/bin/env python3
# ================================================================
# THÙNG RÁC THÔNG MINH – Code Raspberry Pi
# Chức năng:
#   1. Camera chụp ảnh rác
#   2. AI nhận diện loại rác (dùng model Teachable Machine)
#   3. Gửi lệnh JSON sang Arduino mở đúng nắp
#   4. Web server để xem kết quả trên điện thoại
# ================================================================
# Cách cài thư viện (chạy 1 lần trong Terminal):
#   pip3 install flask opencv-python numpy tensorflow
#   pip3 install pyserial pillow
# ================================================================

import cv2
import numpy as np
import serial
import time
import json
import threading
from flask import Flask, Response, render_template_string, jsonify, request

# ---------------------------------------------------------------
# ⚙️  CÀI ĐẶT – Chỉnh sửa ở đây nếu cần
# ---------------------------------------------------------------

# Cổng kết nối Arduino (chạy "ls /dev/tty*" trong Terminal để tìm)
ARDUINO_PORT = '/dev/ttyUSB1'   # Thử /dev/ttyACM0 nếu không được
ARDUINO_BAUD = 115200

# Chỉ số camera (0 = camera mặc định, 1 = camera ngoài)
CAMERA_INDEX = 0

# Độ tin cậy tối thiểu để kích hoạt mở nắp (80%)
NGUONG_TIN_CAY = 0.80

# Thời gian chờ giữa 2 lần nhận diện (giây)
THOI_GIAN_CHO = 3

# ---------------------------------------------------------------
# 🤖  LOAD MODEL AI (Teachable Machine export dạng TensorFlow)
# ---------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH  = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'

with open(LABELS_PATH, 'r') as f:
    NHAN_RAC = [line.strip().split(' ', 1)[1] for line in f.readlines()]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

MODEL_SIZE = (224, 224)

# ---------------------------------------------------------------
# ⚙️  CẤU HÌNH SERVO TỪNG THÙNG – Chỉnh ở đây nếu cần
# ---------------------------------------------------------------
# Góc servo và thời gian mở cho mỗi thùng (bin 1, 2, 3)
SERVO_CONFIG = {
    1: {"open": 120, "close": 5, "time": 4000},
    2: {"open": 120, "close": 5, "time": 4000},
    3: {"open": 120, "close": 5, "time": 4000},
}

# ---------------------------------------------------------------
# 🔌  KẾT NỐI ARDUINO
# ---------------------------------------------------------------
arduino = None

def gui_setup_arduino():
    """Gửi các lệnh cài đặt ban đầu cho từng servo"""
    print("⚙️  Đang gửi cấu hình servo...")
    for bin_num, cfg in SERVO_CONFIG.items():
        # 1. Cài thời gian mở
        gui_lenh_arduino({"cmd": "set", "bin": bin_num, "time": cfg["time"]})
        time.sleep(0.2)
        # 2. Cài góc mở / đóng
        gui_lenh_arduino({"cmd": "set", "bin": bin_num,
                          "open": cfg["open"], "close": cfg["close"]})
        time.sleep(0.2)
    print("✅ Đã gửi xong cấu hình servo cho cả 3 thùng")

def ket_noi_arduino():
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(3)  # Chờ Arduino khởi động xong
        print(f"✅ Đã kết nối Arduino tại {ARDUINO_PORT}")
        gui_setup_arduino()  # Gửi cấu hình ngay sau khi kết nối
    except Exception as e:
        print(f"⚠️  Không kết nối được Arduino: {e}")
        arduino = None

# ---- Hàm gửi lệnh JSON chuẩn ----
def gui_lenh_arduino(cmd_dict: dict):
    """Gửi lệnh dạng JSON sang Arduino, VD: {"cmd": "open_bin", "bin": 1}"""
    if arduino and arduino.is_open:
        json_str = json.dumps(cmd_dict) + '\n'
        arduino.write(json_str.encode())
        print(f"📡 Gửi Arduino: {json_str.strip()}")
    else:
        print(f"⚠️  Arduino chưa kết nối – lệnh: {cmd_dict}")

# ---- Các lệnh tiện ích ----
def arduino_scan_start():
    """Bật đèn LED + beep báo hiệu bắt đầu quét"""
    gui_lenh_arduino({"cmd": "scan_start"})

def arduino_scan_end():
    """Tắt đèn LED báo hiệu kết thúc quét"""
    gui_lenh_arduino({"cmd": "scan_end"})

def arduino_open_bin(bin_num: int):
    """Mở nắp thùng số bin_num"""
    gui_lenh_arduino({"cmd": "open_bin", "bin": bin_num})

def arduino_beep():
    """Phát tiếng beep"""
    gui_lenh_arduino({"cmd": "beep"})

def arduino_set_bin(bin_num: int, open_angle=None, close_angle=None, time_ms=None):
    """Cài đặt góc mở/đóng hoặc thời gian mở của thùng"""
    cmd = {"cmd": "set", "bin": bin_num}
    if open_angle is not None:
        cmd["open"] = open_angle
    if close_angle is not None:
        cmd["close"] = close_angle
    if time_ms is not None:
        cmd["time"] = time_ms
    gui_lenh_arduino(cmd)

# ---------------------------------------------------------------
# 📷  XỬ LÝ CAMERA & NHẬN DIỆN AI
# ---------------------------------------------------------------
camera = cv2.VideoCapture(CAMERA_INDEX)

# Biến dùng chung giữa các thread
ket_qua_hien_tai = {
    "nhan":       "Đang chờ...",
    "do_tin_cay": 0.0,
    "thung":      0,
    "frame":      None
}
lock = threading.Lock()

# Trạng thái quét
scan_mode     = False   # True khi đang trong chế độ quét
scan_lock     = threading.Lock()

def tien_xu_ly_anh(frame):
    anh = cv2.resize(frame, MODEL_SIZE)
    anh = np.array(anh, dtype=np.float32)
    anh = (anh / 127.5) - 1.0
    anh = np.expand_dims(anh, axis=0)
    return anh

def nhan_dien_rac(frame):
    anh = tien_xu_ly_anh(frame)
    interpreter.set_tensor(input_details[0]['index'], anh)
    interpreter.invoke()
    ket_qua = interpreter.get_tensor(output_details[0]['index'])[0]

    idx_cao_nhat = int(np.argmax(ket_qua))
    do_tin_cay   = float(ket_qua[idx_cao_nhat])
    nhan         = NHAN_RAC[idx_cao_nhat]

    # Map nhãn → số thùng (int để gửi JSON)
    map_thung = {
        'Huu Co':   1,
        'Rac Nhua': 2,
        'Tai Che':  3,
    }
    so_thung = map_thung.get(nhan, 0)
    return nhan, do_tin_cay, so_thung

def vong_lap_camera():
    """Thread chạy nền: liên tục đọc camera, nhận diện AI khi đang scan"""
    global scan_mode
    thoi_gian_mo_nap = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            print("⚠️  Không đọc được camera")
            time.sleep(1)
            continue

        # Nhận diện AI
        nhan, do_tin_cay, so_thung = nhan_dien_rac(frame)

        # Vẽ kết quả lên ảnh
        mau = (0, 255, 0) if do_tin_cay >= NGUONG_TIN_CAY else (0, 165, 255)
        cv2.putText(frame,
                    f"{nhan}: {do_tin_cay*100:.1f}%",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, mau, 3)
        cv2.putText(frame,
                    f"Thung: {so_thung}" if so_thung != 0 else "Chua xac dinh",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, mau, 2)

        # Hiển thị trạng thái scan
        with scan_lock:
            dang_quet = scan_mode
        if dang_quet:
            cv2.putText(frame, "DANG QUET...", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        with lock:
            ket_qua_hien_tai["nhan"]       = nhan
            ket_qua_hien_tai["do_tin_cay"] = do_tin_cay
            ket_qua_hien_tai["thung"]      = so_thung
            ket_qua_hien_tai["frame"]      = frame.copy()

        # Chỉ mở nắp tự động khi đang trong scan mode
        now = time.time()
        if (dang_quet
                and do_tin_cay >= NGUONG_TIN_CAY
                and so_thung != 0
                and now - thoi_gian_mo_nap > THOI_GIAN_CHO):
            arduino_open_bin(so_thung)
            thoi_gian_mo_nap = now
            # Kết thúc scan sau khi đã nhận diện xong
            with scan_lock:
                scan_mode = False
            arduino_scan_end()
            print(f"✅ Nhận diện xong: {nhan} → Thùng {so_thung}, kết thúc scan")

        time.sleep(0.1)

# ---------------------------------------------------------------
# 🌐  WEB SERVER (Flask)
# ---------------------------------------------------------------
app = Flask(__name__)

TRANG_WEB = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Thùng Rác Thông Minh</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: Arial, sans-serif;
      background: #1a1a2e;
      color: white;
      min-height: 100vh;
      padding: 20px;
    }
    h1 {
      text-align: center;
      color: #4CAF50;
      font-size: 1.4rem;
      margin-bottom: 20px;
    }
    .camera-box {
      width: 100%;
      max-width: 640px;
      margin: 0 auto 20px;
      border-radius: 12px;
      overflow: hidden;
      border: 2px solid #4CAF50;
      position: relative;
    }
    .camera-box img { width: 100%; display: block; }
    .scan-overlay {
      display: none;
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      border: 4px solid red;
      border-radius: 12px;
      animation: pulse 1s infinite;
      pointer-events: none;
    }
    @keyframes pulse {
      0%   { border-color: red; box-shadow: 0 0 10px red; }
      50%  { border-color: #ff6600; box-shadow: 0 0 25px #ff6600; }
      100% { border-color: red; box-shadow: 0 0 10px red; }
    }
    .result-box {
      background: #16213e;
      border-radius: 12px;
      padding: 16px;
      max-width: 640px;
      margin: 0 auto 20px;
      text-align: center;
    }
    .ket-qua { font-size: 1.5rem; font-weight: bold; color: #4CAF50; }
    .do-tin-cay { font-size: 1rem; color: #aaa; margin-top: 6px; }
    .thanh-phan {
      background: #4CAF50;
      height: 12px;
      border-radius: 6px;
      margin-top: 10px;
      transition: width 0.3s;
    }

    /* ---- Nút SCAN ---- */
    .btn-scan-wrap {
      max-width: 640px;
      margin: 0 auto 16px;
    }
    .btn-scan {
      width: 100%;
      padding: 18px;
      border: none;
      border-radius: 14px;
      font-size: 1.2rem;
      font-weight: bold;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, #4CAF50, #2196F3);
      transition: all 0.3s;
      letter-spacing: 1px;
    }
    .btn-scan.scanning {
      background: linear-gradient(135deg, #f44336, #ff9800);
      animation: scanPulse 1.2s infinite;
    }
    @keyframes scanPulse {
      0%   { transform: scale(1);    box-shadow: 0 0 0 rgba(244,67,54,0.4); }
      50%  { transform: scale(1.02); box-shadow: 0 0 20px rgba(244,67,54,0.6); }
      100% { transform: scale(1);    box-shadow: 0 0 0 rgba(244,67,54,0.4); }
    }
    .btn-scan:disabled { opacity: 0.6; cursor: not-allowed; }

    .btn-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      max-width: 640px;
      margin: 0 auto;
    }
    .btn {
      padding: 14px;
      border: none;
      border-radius: 10px;
      font-size: 0.9rem;
      font-weight: bold;
      cursor: pointer;
      color: white;
    }
    .btn1 { background: #4CAF50; }
    .btn2 { background: #2196F3; }
    .btn3 { background: #FF9800; }
    .btn:active { opacity: 0.7; }
    .label-manual {
      max-width: 640px;
      margin: 16px auto 8px;
      font-size: 0.85rem;
      color: #aaa;
      text-align: center;
    }
    .trang-thai {
      text-align: center;
      margin-top: 14px;
      font-size: 0.85rem;
      color: #aaa;
    }
    #thong-bao {
      background: #4CAF50;
      color: white;
      border-radius: 8px;
      padding: 10px;
      text-align: center;
      margin: 10px auto;
      max-width: 640px;
      display: none;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <h1>🗑️ Thùng Rác Thông Minh</h1>

  <!-- Khung camera live -->
  <div class="camera-box">
    <img src="/camera_live" alt="Camera">
    <div class="scan-overlay" id="scan-overlay"></div>
  </div>

  <!-- Kết quả AI -->
  <div class="result-box">
    <div class="ket-qua" id="ten-rac">Đang nhận diện...</div>
    <div class="do-tin-cay" id="phan-tram">--</div>
    <div class="thanh-phan" id="thanh" style="width:0%"></div>
  </div>

  <!-- Thông báo -->
  <div id="thong-bao"></div>

  <!-- NÚT SCAN CHÍNH -->
  <div class="btn-scan-wrap">
    <button class="btn-scan" id="btn-scan" onclick="toggleScan()">
      📷 Bắt Đầu Quét Rác
    </button>
  </div>

  <!-- Nút mở thủ công -->
  <div class="label-manual">— Hoặc mở thủ công —</div>
  <div class="btn-grid">
    <button class="btn btn1" onclick="moThuCong(1)">🌿 Hữu Cơ<br>Thùng 1</button>
    <button class="btn btn2" onclick="moThuCong(2)">🧴 Nhựa<br>Thùng 2</button>
    <button class="btn btn3" onclick="moThuCong(3)">♻️ Tái Chế<br>Thùng 3</button>
  </div>

  <div class="trang-thai" id="trang-thai">🟢 Hệ thống đang hoạt động</div>

<script>
  let dangQuet = false;

  // ---- Bắt đầu / Dừng quét ----
  function toggleScan() {
    const btn = document.getElementById('btn-scan');
    const overlay = document.getElementById('scan-overlay');

    if (!dangQuet) {
      // Bắt đầu quét
      fetch('/bat_dau_quet', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          dangQuet = true;
          btn.textContent = '⏹ Đang Quét... (Bấm để dừng)';
          btn.classList.add('scanning');
          overlay.style.display = 'block';
          hienThongBao(data.thong_bao, '#2196F3');
        })
        .catch(() => hienThongBao('❌ Lỗi kết nối', '#f44336'));
    } else {
      // Dừng quét thủ công
      fetch('/ket_thuc_quet', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
          ketThucScan();
          hienThongBao(data.thong_bao, '#FF9800');
        })
        .catch(() => ketThucScan());
    }
  }

  function ketThucScan() {
    dangQuet = false;
    const btn = document.getElementById('btn-scan');
    btn.textContent = '📷 Bắt Đầu Quét Rác';
    btn.classList.remove('scanning');
    document.getElementById('scan-overlay').style.display = 'none';
  }

  // ---- Cập nhật kết quả AI ----
  function capNhatKetQua() {
    fetch('/ket_qua')
      .then(r => r.json())
      .then(data => {
        document.getElementById('ten-rac').textContent = data.nhan;
        const pct = (data.do_tin_cay * 100).toFixed(1);
        document.getElementById('phan-tram').textContent = `Độ tin cậy: ${pct}%`;
        document.getElementById('thanh').style.width = pct + '%';
        document.getElementById('thanh').style.background =
          data.do_tin_cay >= 0.8 ? '#4CAF50' : '#FF9800';

        // Nếu server báo scan đã kết thúc (AI tự động nhận diện xong)
        if (dangQuet && !data.dang_quet) {
          ketThucScan();
          if (data.thung !== 0) {
            const ten = {1:'Hữu Cơ', 2:'Rác Nhựa', 3:'Tái Chế'}[data.thung] || '';
            hienThongBao(`✅ Đã nhận diện: ${data.nhan} → Mở Thùng ${data.thung} (${ten})`, '#4CAF50');
          }
        }

        document.getElementById('trang-thai').textContent = '🟢 Hệ thống đang hoạt động';
      })
      .catch(() => {
        document.getElementById('trang-thai').textContent = '🔴 Mất kết nối';
      });
  }

  // ---- Mở nắp thủ công ----
  function moThuCong(so) {
    fetch('/mo_nap/' + so, { method: 'POST' })
      .then(r => r.json())
      .then(data => hienThongBao(data.thong_bao, '#4CAF50'));
  }

  // ---- Hiển thị thông báo ----
  function hienThongBao(msg, mau) {
    const tb = document.getElementById('thong-bao');
    tb.textContent = msg;
    tb.style.background = mau || '#4CAF50';
    tb.style.display = 'block';
    setTimeout(() => tb.style.display = 'none', 4000);
  }

  setInterval(capNhatKetQua, 1000);
  capNhatKetQua();
</script>
</body>
</html>
"""

@app.route('/')
def trang_chu():
    return render_template_string(TRANG_WEB)

@app.route('/camera_live')
def camera_live():
    """Stream video trực tiếp từ camera"""
    def generate():
        while True:
            with lock:
                frame = ket_qua_hien_tai.get("frame")
            if frame is None:
                time.sleep(0.1)
                continue
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes()
                   + b'\r\n')
            time.sleep(0.05)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ket_qua')
def lay_ket_qua():
    """API trả về kết quả nhận diện + trạng thái scan"""
    with lock:
        data = {
            "nhan":       ket_qua_hien_tai["nhan"],
            "do_tin_cay": ket_qua_hien_tai["do_tin_cay"],
            "thung":      ket_qua_hien_tai["thung"]
        }
    with scan_lock:
        data["dang_quet"] = scan_mode
    return jsonify(data)

@app.route('/bat_dau_quet', methods=['POST'])
def bat_dau_quet():
    """Bắt đầu chế độ quét: LED sáng + beep, AI sẽ tự nhận diện và mở thùng"""
    global scan_mode
    with scan_lock:
        scan_mode = True
    arduino_scan_start()
    return jsonify({"thong_bao": "📷 Bắt đầu quét – LED sáng, AI đang nhận diện..."})

@app.route('/ket_thuc_quet', methods=['POST'])
def ket_thuc_quet():
    """Kết thúc chế độ quét thủ công"""
    global scan_mode
    with scan_lock:
        scan_mode = False
    arduino_scan_end()
    return jsonify({"thong_bao": "⏹ Đã dừng quét"})

@app.route('/mo_nap/<int:so>', methods=['POST', 'GET'])
def mo_nap_thu_cong(so):
    """Mở nắp thủ công từ website"""
    ten = {1: "Hữu Cơ", 2: "Rác Nhựa", 3: "Tái Chế"}.get(so, "?")
    if so in [1, 2, 3]:
        arduino_open_bin(so)
        return jsonify({"thong_bao": f"✅ Đã mở Thùng {so} – {ten}"})
    return jsonify({"thong_bao": "❌ Số thùng không hợp lệ"})

# Endpoint test tiện ích (theo docs)
@app.route('/test/beep', methods=['POST'])
def test_beep():
    arduino_beep()
    return jsonify({"ok": True, "cmd": "beep"})

@app.route('/test/set_bin', methods=['POST'])
def test_set_bin():
    """VD: POST JSON {"bin":1,"open":120,"close":5} hoặc {"bin":1,"time":4000}"""
    data = request.get_json(force=True)
    arduino_set_bin(
        bin_num     = data.get('bin', 1),
        open_angle  = data.get('open'),
        close_angle = data.get('close'),
        time_ms     = data.get('time')
    )
    return jsonify({"ok": True, "sent": data})

# ---------------------------------------------------------------
# 🚀  KHỞI ĐỘNG CHƯƠNG TRÌNH
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 50)
    print("  THÙNG RÁC THÔNG MINH – Raspberry Pi")
    print("=" * 50)

    ket_noi_arduino()

    thread_camera = threading.Thread(target=vong_lap_camera, daemon=True)
    thread_camera.start()
    print("📷 Camera đang chạy...")

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()

    print(f"\n🌐 Truy cập website tại: http://{ip}:3002")
    print("   (Điện thoại cùng mạng WiFi mở trình duyệt nhập địa chỉ trên)")
    print("\n   Nhấn Ctrl+C để dừng\n")

    app.run(host='0.0.0.0', port=3002, debug=False, threaded=True)
