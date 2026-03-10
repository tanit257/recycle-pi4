#!/usr/bin/env python3
# ================================================================
# THÙNG RÁC THÔNG MINH – Code Raspberry Pi
# Chức năng:
#   1. Camera chụp ảnh rác
#   2. AI nhận diện loại rác (dùng model Teachable Machine)
#   3. Gửi lệnh sang Arduino mở đúng nắp
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
import threading
from flask import Flask, Response, render_template_string, jsonify

# ---------------------------------------------------------------
# ⚙️  CÀI ĐẶT – Chỉnh sửa ở đây nếu cần
# ---------------------------------------------------------------

# Cổng kết nối Arduino (chạy "ls /dev/tty*" trong Terminal để tìm)
ARDUINO_PORT = '/dev/ttyUSB0'   # Thử /dev/ttyACM0 nếu không được
ARDUINO_BAUD = 9600

# Chỉ số camera (0 = camera mặc định, 1 = camera ngoài)
CAMERA_INDEX = 0

# Độ tin cậy tối thiểu để kích hoạt mở nắp (80%)
NGUONG_TIN_CAY = 0.80

# Thời gian chờ giữa 2 lần nhận diện (giây)
THOI_GIAN_CHO = 3

# ---------------------------------------------------------------
# 🤖  LOAD MODEL AI (Teachable Machine export dạng TensorFlow)
# ---------------------------------------------------------------
# Hướng dẫn export model:
#   1. Vào teachablemachine.withgoogle.com
#   2. Train xong → Export Model
#   3. Chọn "TensorFlow" → "TensorFlow Lite" → Download
#   4. Giải nén → copy 2 file vào cùng thư mục code này:
#      - model_unquant.tflite
#      - labels.txt

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH  = 'model_unquant.tflite'
LABELS_PATH = 'labels.txt'

# Đọc nhãn từ file labels.txt
with open(LABELS_PATH, 'r') as f:
    NHAN_RAC = [line.strip().split(' ', 1)[1] for line in f.readlines()]
# NHAN_RAC sẽ là: ['Huu Co', 'Rac Nhua', 'Tai Che']

# Load model TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kích thước ảnh model yêu cầu (Teachable Machine mặc định 224x224)
MODEL_SIZE = (224, 224)

# ---------------------------------------------------------------
# 🔌  KẾT NỐI ARDUINO
# ---------------------------------------------------------------
arduino = None

def ket_noi_arduino():
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(3)  # Chờ Arduino khởi động
        print(f"✅ Đã kết nối Arduino tại {ARDUINO_PORT}")
    except Exception as e:
        print(f"⚠️  Không kết nối được Arduino: {e}")
        print("   → Chạy lệnh: ls /dev/tty* để tìm đúng cổng")
        arduino = None

def gui_lenh_arduino(lenh: str):
    """Gửi lệnh '1', '2', hoặc '3' sang Arduino"""
    if arduino and arduino.is_open:
        arduino.write(lenh.encode())
        print(f"📡 Đã gửi lệnh '{lenh}' sang Arduino")
    else:
        print("⚠️  Arduino chưa kết nối – bỏ qua lệnh")

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

def tien_xu_ly_anh(frame):
    """Resize và chuẩn hóa ảnh để đưa vào model AI"""
    anh = cv2.resize(frame, MODEL_SIZE)
    anh = np.array(anh, dtype=np.float32)
    anh = (anh / 127.5) - 1.0          # Normalize về [-1, 1]
    anh = np.expand_dims(anh, axis=0)  # Thêm batch dimension
    return anh

def nhan_dien_rac(frame):
    """Đưa ảnh vào model AI, trả về (nhãn, độ tin cậy, số thùng)"""
    anh = tien_xu_ly_anh(frame)
    interpreter.set_tensor(input_details[0]['index'], anh)
    interpreter.invoke()
    ket_qua = interpreter.get_tensor(output_details[0]['index'])[0]

    idx_cao_nhat = int(np.argmax(ket_qua))
    do_tin_cay   = float(ket_qua[idx_cao_nhat])
    nhan         = NHAN_RAC[idx_cao_nhat]

    # Map nhãn → số thùng để gửi Arduino
    # Chỉnh lại tên cho khớp với labels.txt của bạn
    map_thung = {
        'Huu Co':   '1',
        'Rac Nhua': '2',
        'Tai Che':  '3',
    }
    so_thung = map_thung.get(nhan, '0')
    return nhan, do_tin_cay, so_thung

def vong_lap_camera():
    """Thread chạy nền: liên tục đọc camera và nhận diện AI"""
    thoi_gian_mo_nap = 0  # Tránh mở nắp liên tục

    while True:
        ret, frame = camera.read()
        if not ret:
            print("⚠️  Không đọc được camera")
            time.sleep(1)
            continue

        # Nhận diện AI
        nhan, do_tin_cay, so_thung = nhan_dien_rac(frame)

        # Vẽ kết quả lên ảnh (để hiển thị trên web)
        mau = (0, 255, 0) if do_tin_cay >= NGUONG_TIN_CAY else (0, 165, 255)
        cv2.putText(frame,
                    f"{nhan}: {do_tin_cay*100:.1f}%",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, mau, 3)
        cv2.putText(frame,
                    f"Thung: {so_thung}" if so_thung != '0' else "Chua xac dinh",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, mau, 2)

        # Cập nhật kết quả dùng chung
        with lock:
            ket_qua_hien_tai["nhan"]       = nhan
            ket_qua_hien_tai["do_tin_cay"] = do_tin_cay
            ket_qua_hien_tai["thung"]      = so_thung
            ket_qua_hien_tai["frame"]      = frame.copy()

        # Gửi lệnh Arduino nếu đủ tin cậy và chưa mở gần đây
        now = time.time()
        if (do_tin_cay >= NGUONG_TIN_CAY
                and so_thung != '0'
                and now - thoi_gian_mo_nap > THOI_GIAN_CHO):
            gui_lenh_arduino(so_thung)
            thoi_gian_mo_nap = now

        time.sleep(0.1)  # ~10 FPS

# ---------------------------------------------------------------
# 🌐  WEB SERVER (Flask)
# ---------------------------------------------------------------
app = Flask(__name__)

# Trang HTML hiển thị trên điện thoại
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
    }
    .camera-box img { width: 100%; display: block; }
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
  </div>

  <!-- Kết quả AI -->
  <div class="result-box">
    <div class="ket-qua" id="ten-rac">Đang nhận diện...</div>
    <div class="do-tin-cay" id="phan-tram">--</div>
    <div class="thanh-phan" id="thanh" style="width:0%"></div>
  </div>

  <!-- Thông báo -->
  <div id="thong-bao"></div>

  <!-- Nút mở thủ công -->
  <div class="btn-grid">
    <button class="btn btn1" onclick="moThuCong('1')">🌿 Hữu Cơ<br>Thùng 1</button>
    <button class="btn btn2" onclick="moThuCong('2')">🧴 Nhựa<br>Thùng 2</button>
    <button class="btn btn3" onclick="moThuCong('3')">♻️ Tái Chế<br>Thùng 3</button>
  </div>

  <div class="trang-thai" id="trang-thai">🟢 Hệ thống đang hoạt động</div>

<script>
  // Cập nhật kết quả AI mỗi 1 giây
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
      })
      .catch(() => {
        document.getElementById('trang-thai').textContent = '🔴 Mất kết nối';
      });
  }

  // Mở nắp thủ công qua nút bấm
  function moThuCong(so) {
    fetch('/mo_nap/' + so)
      .then(r => r.json())
      .then(data => {
        const tb = document.getElementById('thong-bao');
        tb.textContent = data.thong_bao;
        tb.style.display = 'block';
        setTimeout(() => tb.style.display = 'none', 3000);
      });
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
    """API trả về kết quả nhận diện hiện tại (JSON)"""
    with lock:
        return jsonify({
            "nhan":       ket_qua_hien_tai["nhan"],
            "do_tin_cay": ket_qua_hien_tai["do_tin_cay"],
            "thung":      ket_qua_hien_tai["thung"]
        })

@app.route('/mo_nap/<so>')
def mo_nap_thu_cong(so):
    """API mở nắp thủ công từ website"""
    ten = {"1": "Hữu Cơ", "2": "Rác Nhựa", "3": "Tái Chế"}.get(so, "?")
    if so in ['1', '2', '3']:
        gui_lenh_arduino(so)
        return jsonify({"thong_bao": f"✅ Đã mở Thùng {so} – {ten}"})
    return jsonify({"thong_bao": "❌ Số thùng không hợp lệ"})

# ---------------------------------------------------------------
# 🚀  KHỞI ĐỘNG CHƯƠNG TRÌNH
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 50)
    print("  THÙNG RÁC THÔNG MINH – Raspberry Pi")
    print("=" * 50)

    # Kết nối Arduino
    ket_noi_arduino()

    # Khởi động thread camera (chạy nền)
    thread_camera = threading.Thread(target=vong_lap_camera, daemon=True)
    thread_camera.start()
    print("📷 Camera đang chạy...")

    # Lấy địa chỉ IP để truy cập web
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()

    print(f"\n🌐 Truy cập website tại: http://{ip}:3002")
    print("   (Điện thoại cùng mạng WiFi mở trình duyệt nhập địa chỉ trên)")
    print("\n   Nhấn Ctrl+C để dừng\n")

    # Khởi động web server
    app.run(host='0.0.0.0', port=3002, debug=False, threaded=True)
