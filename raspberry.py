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

import os
import cv2
import numpy as np
import serial
import time
import json
import threading

# Thư mục chứa file .py này – dùng để resolve đường dẫn tương đối
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from flask import Flask, Response, render_template_string, jsonify, request

# ---------------------------------------------------------------
# ⚙️  CÀI ĐẶT – Chỉnh sửa ở đây nếu cần
# ---------------------------------------------------------------

# Danh sách cổng Arduino – tự thử từng cổng cho đến khi kết nối được
ARDUINO_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
ARDUINO_BAUD  = 115200

# Danh sách chỉ số camera sẽ thử lần lượt
CAMERA_INDEXES = [0, 1, 2]

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

MODEL_PATH  = os.path.join(BASE_DIR, 'model_unquant.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

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
    1: {"open": 35, "close": 0, "time": 4000},
    2: {"open": 35, "close": 0, "time": 4000},
    3: {"open": 35, "close": 0, "time": 4000},
}

# Remap logical bin → physical servo index trên Arduino
# (do servo được đấu dây lệch so với thứ tự logic)
BIN_MAP = {
    1: 3,   # Vô Cơ      → physical servo 3
    2: 1,   # Hữu Cơ     → physical servo 1
    3: 2,   # Undetermined → physical servo 2
}

# ---------------------------------------------------------------
# 🔌  KẾT NỐI ARDUINO
# ---------------------------------------------------------------
arduino      = None
arduino_port_dang_dung = None   # port đang kết nối thành công
arduino_lock = threading.Lock()

def gui_setup_arduino():
    """Gửi các lệnh cài đặt ban đầu cho từng servo"""
    print("⚙️  Đang gửi cấu hình servo...")
    for bin_num, cfg in SERVO_CONFIG.items():
        gui_lenh_arduino({"cmd": "set", "bin": bin_num, "time": cfg["time"]})
        time.sleep(0.2)
        gui_lenh_arduino({"cmd": "set", "bin": bin_num,
                          "open": cfg["open"], "close": cfg["close"]})
        time.sleep(0.2)
    print("✅ Đã gửi xong cấu hình servo cho cả 3 thùng")

def _thu_ket_noi(port: str) -> bool:
    """Thử kết nối 1 port, trả về True nếu thành công"""
    global arduino, arduino_port_dang_dung
    try:
        ser = serial.Serial(port, ARDUINO_BAUD, timeout=1)
        time.sleep(3)          # Chờ Arduino boot
        with arduino_lock:
            arduino = ser
            arduino_port_dang_dung = port
        print(f"✅ Đã kết nối Arduino tại {port}")
        gui_setup_arduino()
        return True
    except Exception as e:
        print(f"   ✗ {port}: {e}")
        return False

def ket_noi_arduino():
    """Thử lần lượt từng port trong ARDUINO_PORTS"""
    print(f"🔍 Tìm Arduino trên: {ARDUINO_PORTS}")
    for port in ARDUINO_PORTS:
        if _thu_ket_noi(port):
            return
    print("⚠️  Không tìm thấy Arduino – sẽ thử lại ở background")

def vong_lap_ket_noi_lai():
    """Thread chạy nền: nếu Arduino mất kết nối thì tự tìm và kết nối lại"""
    global arduino
    while True:
        time.sleep(5)
        with arduino_lock:
            ket_noi = arduino is not None and arduino.is_open
        if not ket_noi:
            print("🔄 Mất kết nối Arduino – đang thử kết nối lại...")
            # Đóng port cũ nếu còn
            with arduino_lock:
                try:
                    if arduino:
                        arduino.close()
                except Exception:
                    pass
                arduino = None
                arduino_port_dang_dung = None
            # Thử lại từng port
            for port in ARDUINO_PORTS:
                if _thu_ket_noi(port):
                    break
            else:
                print("⚠️  Vẫn chưa kết nối được – thử lại sau 5 giây...")

# ---- Hàm gửi lệnh JSON chuẩn ----
def gui_lenh_arduino(cmd_dict: dict):
    """Gửi lệnh dạng JSON sang Arduino, VD: {"cmd": "open_bin", "bin": 1}"""
    with arduino_lock:
        ser = arduino
    if ser and ser.is_open:
        try:
            json_str = json.dumps(cmd_dict, separators=(',', ':')) + '\n'
            ser.write(json_str.encode())
            print(f"📡 Gửi Arduino [{arduino_port_dang_dung}]: {json_str.strip()}")
        except Exception as e:
            print(f"⚠️  Lỗi gửi lệnh: {e} – sẽ reconnect...")
            with arduino_lock:
                globals()['arduino'] = None
    else:
        print(f"⚠️  Arduino chưa kết nối – lệnh bị bỏ qua: {cmd_dict}")

# ---- Các lệnh tiện ích ----
def arduino_scan_start():
    """Bật đèn LED + beep báo hiệu bắt đầu quét"""
    gui_lenh_arduino({"cmd": "scan_start"})

def arduino_scan_end():
    """Tắt đèn LED báo hiệu kết thúc quét"""
    gui_lenh_arduino({"cmd": "scan_end"})

def arduino_open_bin(bin_num: int):
    """Mở nắp thùng số bin_num (tự động remap sang physical servo)"""
    physical = BIN_MAP.get(bin_num, bin_num)
    gui_lenh_arduino({"cmd": "open_bin", "bin": physical})

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
def tim_camera():
    """Thử lần lượt từng index trong CAMERA_INDEXES, trả về VideoCapture đầu tiên mở được"""
    for idx in CAMERA_INDEXES:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"📷 Tìm thấy camera tại index {idx}")
                return cap
        cap.release()
    print("⚠️  Không tìm thấy camera nào – sẽ thử lại khi đọc frame")
    return None

camera = None  # Không mở camera lúc khởi động – chỉ mở khi bắt đầu quét

# Biến dùng chung giữa các thread
ket_qua_hien_tai = {
    "nhan":       "Đang chờ...",
    "do_tin_cay": 0.0,
    "thung":      0,
    "frame":      None
}
lock = threading.Lock()

# Trạng thái quét
scan_mode       = False   # True khi đang trong chế độ quét
scan_start_time = 0.0     # Thời điểm bắt đầu quét
scan_lock       = threading.Lock()

# Thống kê số lần phân loại
thong_ke = {"Vo Co": 0, "Huu Co": 0, "Khong Xac Dinh": 0, "Undetermined": 0}
thong_ke_lock = threading.Lock()

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
    nhan         = NHAN_RAC[idx_cao_nhat] if idx_cao_nhat < len(NHAN_RAC) else "Undetermined"

    # Map nhãn → số thùng (int để gửi JSON)
    map_thung = {
        'Vo Co':          1,
        'Huu Co':         2,
        'Khong Xac Dinh': 3,
    }
    so_thung = map_thung.get(nhan, 0)
    return nhan, do_tin_cay, so_thung

def vong_lap_camera():
    """Thread chạy nền: chỉ mở camera khi scan_mode=True, đóng ngay sau khi xong"""
    global scan_mode

    while True:
        # Ngủ khi không quét – tiết kiệm CPU/nhiệt
        with scan_lock:
            dang_quet = scan_mode
        if not dang_quet:
            time.sleep(0.2)
            continue

        # Scan vừa bắt đầu – mở camera
        print("📷 Đang mở camera để quét...")
        cap = tim_camera()
        if cap is None:
            print("⚠️  Không mở được camera – huỷ quét")
            with scan_lock:
                scan_mode = False
            arduino_scan_end()
            time.sleep(2)
            continue

        # Reset kết quả cũ trước khi quét mới
        with lock:
            ket_qua_hien_tai["nhan"]       = "Đang nhận diện..."
            ket_qua_hien_tai["do_tin_cay"] = 0.0
            ket_qua_hien_tai["thung"]      = 0

        print("✅ Camera sẵn sàng – bắt đầu nhận diện...")
        da_xu_ly = False

        try:
            while True:
                with scan_lock:
                    dang_quet         = scan_mode
                    thoi_gian_bat_dau = scan_start_time

                # Dừng sớm nếu bị huỷ từ bên ngoài (ví dụ /ket_thuc_quet)
                if not dang_quet:
                    break

                now = time.time()

                # Sau 4 giây → quyết định dựa trên kết quả AI mới nhất
                if not da_xu_ly and (now - thoi_gian_bat_dau) >= 4.0:
                    da_xu_ly = True
                    with lock:
                        nhan_cu       = ket_qua_hien_tai["nhan"]
                        do_tin_cay_cu = ket_qua_hien_tai["do_tin_cay"]
                        so_thung_cu   = ket_qua_hien_tai["thung"]

                    if do_tin_cay_cu >= NGUONG_TIN_CAY and so_thung_cu != 0:
                        bin_mo  = so_thung_cu
                        ten_rac = nhan_cu
                    else:
                        bin_mo  = 3
                        ten_rac = "Undetermined"
                        with lock:
                            ket_qua_hien_tai["nhan"]  = "Undetermined"
                            ket_qua_hien_tai["thung"] = 3

                    with scan_lock:
                        scan_mode = False
                    arduino_scan_end()
                    arduino_open_bin(bin_mo)
                    with thong_ke_lock:
                        thong_ke[ten_rac] = thong_ke.get(ten_rac, 0) + 1
                    print(f"✅ Kết quả: {ten_rac} ({do_tin_cay_cu*100:.1f}%) → Mở Thùng {bin_mo}")
                    break  # Thoát vòng scan, chuẩn bị tắt camera

                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Không đọc được frame – bỏ qua...")
                    time.sleep(0.1)
                    continue

                # Nhận diện AI
                nhan, do_tin_cay, so_thung = nhan_dien_rac(frame)

                # Vẽ kết quả lên ảnh
                mau = (0, 255, 0) if do_tin_cay >= NGUONG_TIN_CAY else (0, 165, 255)
                cv2.putText(frame,
                            f"{nhan}: {do_tin_cay*100:.1f}%",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, mau, 3)
                cv2.putText(frame,
                            f"Thung: {so_thung}" if so_thung != 0 else "Undetermined",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, mau, 2)

                con_lai = max(0.0, 4.0 - (now - thoi_gian_bat_dau))
                cv2.putText(frame, f"DANG QUET... {con_lai:.1f}s", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                with lock:
                    ket_qua_hien_tai["nhan"]       = nhan
                    ket_qua_hien_tai["do_tin_cay"] = do_tin_cay
                    ket_qua_hien_tai["thung"]      = so_thung
                    ket_qua_hien_tai["frame"]      = frame.copy()

                time.sleep(0.1)

        except Exception as e:
            print(f"⚠️  Lỗi camera thread: {e}")
        finally:
            cap.release()
            with lock:
                ket_qua_hien_tai["frame"] = None  # Xoá frame – camera đã tắt
            print("📷 Camera đã tắt")

# ---------------------------------------------------------------
# 🌐  WEB SERVER (Flask)
# ---------------------------------------------------------------

def _tao_frame_placeholder():
    """Tạo ảnh tĩnh hiển thị khi camera đang tắt"""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)
    cv2.putText(img, "Camera dang tat", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.putText(img, "Nhan nut de bat dau quet", (15, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    return img
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
      padding-bottom: 30px;
    }

    /* ===== NAVBAR ===== */
    .navbar {
      background: #0f0f23;
      border-bottom: 2px solid #4CAF50;
      display: flex;
      flex-direction: column;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .navbar-title {
      color: #4CAF50;
      font-size: 1rem;
      font-weight: bold;
      text-align: center;
      padding: 10px 16px 4px;
      width: 100%;
    }
    .nav-tabs {
      display: flex;
      width: 100%;
      justify-content: center;
    }
    .nav-tab {
      padding: 8px 24px;
      cursor: pointer;
      font-size: 0.9rem;
      font-weight: bold;
      color: #aaa;
      border-bottom: 3px solid transparent;
      transition: all 0.2s;
      user-select: none;
    }
    .nav-tab.active { color: #4CAF50; border-bottom-color: #4CAF50; }
    .nav-tab:hover { color: white; }

    /* ===== SCREENS ===== */
    .screen { display: none; padding: 20px; }
    .screen.active { display: block; }

    /* ===== TRANG CHỦ ===== */
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
      0%   { border-color: red;    box-shadow: 0 0 10px red; }
      50%  { border-color: #ff6600; box-shadow: 0 0 25px #ff6600; }
      100% { border-color: red;    box-shadow: 0 0 10px red; }
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
    .btn-scan-wrap { max-width: 640px; margin: 0 auto 16px; }
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
      0%   { transform: scale(1);    box-shadow: 0 0 0 rgba(244,67,54,.4); }
      50%  { transform: scale(1.02); box-shadow: 0 0 20px rgba(244,67,54,.6); }
      100% { transform: scale(1);    box-shadow: 0 0 0 rgba(244,67,54,.4); }
    }
    /* ===== THỐNG KÊ ===== */
    .stat-wrap {
      max-width: 640px;
      margin: 0 auto 20px;
    }
    .stat-title {
      text-align: center;
      font-size: 0.85rem;
      color: #aaa;
      margin-bottom: 10px;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    .stat-total {
      text-align: center;
      font-size: 0.9rem;
      color: #4CAF50;
      margin-bottom: 12px;
      font-weight: bold;
    }
    .stat-cards {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }
    .stat-card {
      background: #16213e;
      border-radius: 14px;
      padding: 14px 10px;
      text-align: center;
      border: 2px solid transparent;
      transition: border-color 0.3s;
    }
    .stat-card.voco   { border-color: #FFC107; }
    .stat-card.huuco  { border-color: #4CAF50; }
    .stat-card.other  { border-color: #e0e0e0; }
    .stat-icon  { font-size: 1.6rem; margin-bottom: 4px; }
    .stat-label { font-size: 0.7rem; color: #aaa; margin-bottom: 8px; }
    .stat-count { font-size: 2rem; font-weight: bold; line-height: 1; margin-bottom: 6px; }
    .stat-card.voco  .stat-count { color: #FFC107; }
    .stat-card.huuco .stat-count { color: #4CAF50; }
    .stat-card.other .stat-count { color: #e0e0e0; }
    .stat-bar-bg {
      background: #0f0f23;
      border-radius: 4px;
      height: 6px;
      margin-bottom: 4px;
      overflow: hidden;
    }
    .stat-bar { height: 100%; border-radius: 4px; transition: width 0.5s; width: 0%; }
    .stat-card.voco  .stat-bar { background: #FFC107; }
    .stat-card.huuco .stat-bar { background: #4CAF50; }
    .stat-card.other .stat-bar { background: #e0e0e0; }
    .stat-pct { font-size: 0.72rem; color: #888; }
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

    /* ===== CẤU HÌNH ===== */
    .cfg-wrap { max-width: 640px; margin: 0 auto; }
    .cfg-section {
      background: #16213e;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }
    .cfg-section-title {
      font-size: 0.95rem;
      font-weight: bold;
      color: #4CAF50;
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    /* Simple button row (Scan/LED & Mở Thùng) */
    .btn-cmd-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn-cmd {
      flex: 1;
      min-width: 80px;
      padding: 12px 8px;
      border: none;
      border-radius: 10px;
      background: #4CAF50;
      color: white;
      font-size: 0.88rem;
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s;
      text-align: center;
    }
    .btn-cmd:active { background: #388e3c; }
    .btn-cmd.blue   { background: #2196F3; }
    .btn-cmd.blue:active { background: #1565c0; }
    .btn-cmd.orange { background: #FF9800; }
    .btn-cmd.orange:active { background: #e65100; }
    .btn-cmd.ok { background: #43a047; }
    .btn-cmd.yellow { background: #FFC107; color: #333; }
    .btn-cmd.yellow:active { background: #f9a825; }
    .btn-cmd.white  { background: #f5f5f5; color: #333; }
    .btn-cmd.white:active  { background: #e0e0e0; }
    /* Input rows (Servo & Time) */
    .input-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
    }
    .input-row:last-of-type { margin-bottom: 0; }
    .input-label {
      font-size: 0.8rem;
      color: #aaa;
      min-width: 60px;
      flex-shrink: 0;
    }
    .input-group {
      display: flex;
      align-items: center;
      gap: 6px;
      flex: 1;
      flex-wrap: wrap;
    }
    .input-group label {
      font-size: 0.75rem;
      color: #888;
      white-space: nowrap;
    }
    .num-input {
      width: 70px;
      background: #0f0f23;
      border: 1px solid #333;
      border-radius: 8px;
      color: #7ec8e3;
      font-size: 0.9rem;
      font-weight: bold;
      padding: 7px 8px;
      text-align: center;
      transition: border-color 0.2s;
    }
    .num-input:focus { outline: none; border-color: #4CAF50; }
    .btn-send {
      background: #4CAF50;
      border: none;
      border-radius: 8px;
      color: white;
      font-size: 0.8rem;
      font-weight: bold;
      padding: 8px 16px;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
      transition: background 0.2s;
    }
    .btn-send:hover  { background: #43a047; }
    .btn-send:active { background: #388e3c; }
    .btn-send.ok     { background: #1565c0; }
    /* Custom command textarea */
    .cmd-textarea {
      width: 100%;
      background: #0f0f23;
      border: 1px solid #333;
      border-radius: 8px;
      color: #fff;
      font-family: monospace;
      font-size: 0.88rem;
      padding: 10px;
      resize: vertical;
      min-height: 70px;
      line-height: 1.4;
      transition: border-color 0.2s;
      box-sizing: border-box;
    }
    .cmd-textarea:focus { outline: none; border-color: #4CAF50; }
    .cmd-textarea.error { border-color: #f44336; }
    .custom-send-row {
      display: flex;
      justify-content: flex-end;
      margin-top: 8px;
    }
    .cfg-log {
      background: #0f0f23;
      border-radius: 8px;
      padding: 10px;
      font-family: monospace;
      font-size: 0.78rem;
      color: #aaa;
      max-height: 160px;
      overflow-y: auto;
      margin-top: 10px;
    }
    .cfg-log .log-ok  { color: #4CAF50; }
    .cfg-log .log-err { color: #f44336; }
  </style>
</head>
<body>

  <!-- NAVBAR -->
  <nav class="navbar">
    <div class="navbar-title">🗑️ Thùng Rác Thông Minh</div>
    <div class="nav-tabs">
      <div class="nav-tab active" id="tab-home" onclick="switchTab('home')">Trang Chủ</div>
      <div class="nav-tab"       id="tab-cfg"  onclick="switchTab('cfg')">⚙️ Cấu Hình</div>
    </div>
  </nav>

  <!-- ============ TRANG CHỦ ============ -->
  <div class="screen active" id="screen-home">

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
      <button class="btn-scan" id="btn-scan" onclick="batDauScan()">
        📷 Bắt Đầu Quét Rác
      </button>
    </div>

    <!-- Thống kê -->
    <div class="stat-wrap">
      <div class="stat-title">📊 Thống Kê Phân Loại</div>
      <div class="stat-total" id="stat-total">Tổng: 0 lần</div>
      <div class="stat-cards">
        <div class="stat-card voco">
          <div class="stat-icon">🪨</div>
          <div class="stat-label">Vô Cơ</div>
          <div class="stat-count" id="cnt-voco">0</div>
          <div class="stat-bar-bg"><div class="stat-bar" id="bar-voco"></div></div>
          <div class="stat-pct" id="pct-voco">0%</div>
        </div>
        <div class="stat-card huuco">
          <div class="stat-icon">🌿</div>
          <div class="stat-label">Hữu Cơ</div>
          <div class="stat-count" id="cnt-huuco">0</div>
          <div class="stat-bar-bg"><div class="stat-bar" id="bar-huuco"></div></div>
          <div class="stat-pct" id="pct-huuco">0%</div>
        </div>
        <div class="stat-card other">
          <div class="stat-icon">❓</div>
          <div class="stat-label">Undetermined</div>
          <div class="stat-count" id="cnt-other">0</div>
          <div class="stat-bar-bg"><div class="stat-bar" id="bar-other"></div></div>
          <div class="stat-pct" id="pct-other">0%</div>
        </div>
      </div>
    </div>

    <div class="trang-thai" id="trang-thai">🟢 Hệ thống đang hoạt động</div>
  </div>

  <!-- ============ CẤU HÌNH ============ -->
  <div class="screen" id="screen-cfg">
    <div class="cfg-wrap">

      <!-- Scan / LED -->
      <div class="cfg-section">
        <div class="cfg-section-title">💡 Scan / LED</div>
        <div class="btn-cmd-row">
          <button class="btn-cmd"        onclick="guiCmd({cmd:'scan_start'})">▶ Bắt đầu quét</button>
          <button class="btn-cmd orange" onclick="guiCmd({cmd:'scan_end'})">⏹ Kết thúc quét</button>
          <button class="btn-cmd blue"   onclick="guiCmd({cmd:'beep'})">🔔 Beep</button>
        </div>
      </div>

      <!-- Mở thùng -->
      <div class="cfg-section">
        <div class="cfg-section-title">🗑️ Mở Thùng</div>
        <div class="btn-cmd-row">
          <button class="btn-cmd yellow" onclick="guiCmd({cmd:'open_bin',bin:1})">🪨 Thùng 1<br><small>Vô Cơ</small></button>
          <button class="btn-cmd"       onclick="guiCmd({cmd:'open_bin',bin:2})">🌿 Thùng 2<br><small>Hữu Cơ</small></button>
          <button class="btn-cmd white"  onclick="guiCmd({cmd:'open_bin',bin:3})">❓ Thùng 3<br><small>Undetermined</small></button>
        </div>
      </div>

      <!-- Chỉnh Góc Servo -->
      <div class="cfg-section">
        <div class="cfg-section-title">🔧 Chỉnh Góc Servo</div>
        <div class="input-row">
          <span class="input-label">Thùng 1</span>
          <div class="input-group">
            <label>Open</label>
            <input class="num-input" type="number" id="open-1" value="35" min="0" max="180">
            <label>Close</label>
            <input class="num-input" type="number" id="close-1" value="0" min="0" max="180">
          </div>
          <button class="btn-send" onclick="guiServo(1)">Gửi</button>
        </div>
        <div class="input-row">
          <span class="input-label">Thùng 2</span>
          <div class="input-group">
            <label>Open</label>
            <input class="num-input" type="number" id="open-2" value="35" min="0" max="180">
            <label>Close</label>
            <input class="num-input" type="number" id="close-2" value="0" min="0" max="180">
          </div>
          <button class="btn-send" onclick="guiServo(2)">Gửi</button>
        </div>
        <div class="input-row">
          <span class="input-label">Thùng 3</span>
          <div class="input-group">
            <label>Open</label>
            <input class="num-input" type="number" id="open-3" value="35" min="0" max="180">
            <label>Close</label>
            <input class="num-input" type="number" id="close-3" value="0" min="0" max="180">
          </div>
          <button class="btn-send" onclick="guiServo(3)">Gửi</button>
        </div>
      </div>

      <!-- Chỉnh Thời Gian -->
      <div class="cfg-section">
        <div class="cfg-section-title">⏱️ Thời Gian Mở (ms)</div>
        <div class="input-row">
          <span class="input-label">Thùng 1</span>
          <div class="input-group">
            <input class="num-input" type="number" id="time-1" value="4000" min="0" max="10000" style="width:90px">
            <label>ms</label>
          </div>
          <button class="btn-send" onclick="guiTime(1)">Gửi</button>
        </div>
        <div class="input-row">
          <span class="input-label">Thùng 2</span>
          <div class="input-group">
            <input class="num-input" type="number" id="time-2" value="4000" min="0" max="10000" style="width:90px">
            <label>ms</label>
          </div>
          <button class="btn-send" onclick="guiTime(2)">Gửi</button>
        </div>
        <div class="input-row">
          <span class="input-label">Thùng 3</span>
          <div class="input-group">
            <input class="num-input" type="number" id="time-3" value="4000" min="0" max="10000" style="width:90px">
            <label>ms</label>
          </div>
          <button class="btn-send" onclick="guiTime(3)">Gửi</button>
        </div>
      </div>

      <!-- Reset thống kê -->
      <div class="cfg-section">
        <div class="cfg-section-title">🔄 Thống Kê</div>
        <button class="btn-cmd orange" style="width:100%;padding:14px;font-size:1rem;" onclick="resetThongKe()">
          🗑️ Reset Thống Kê về 0
        </button>
      </div>

      <!-- Pull Code -->
      <div class="cfg-section">
        <div class="cfg-section-title">🔄 Cập Nhật Code</div>
        <button class="btn-cmd blue" id="btn-pull" style="width:100%;padding:14px;font-size:1rem;" onclick="pullCode()">
          ⬇️ Pull Code Mới Nhất & Restart
        </button>
        <div id="pull-log" style="margin-top:10px;background:#0f0f23;border-radius:8px;padding:10px;font-family:monospace;font-size:0.8rem;color:#aaa;display:none;white-space:pre-wrap;word-break:break-all;"></div>
      </div>

      <!-- Custom command -->
      <div class="cfg-section">
        <div class="cfg-section-title">✏️ Lệnh Tuỳ Chỉnh</div>
        <textarea class="cmd-textarea" id="cmd-custom" placeholder='Nhập JSON, VD: {"cmd": "beep"}'></textarea>
        <div class="custom-send-row">
          <button class="btn-send" onclick="guiCustom()">Gửi</button>
        </div>
        <div class="cfg-log" id="cfg-log"><span style="color:#555">— Log lệnh sẽ hiện ở đây —</span></div>
      </div>

    </div><!-- cfg-wrap -->
  </div><!-- screen-cfg -->

<script>
  /* ========= NAV ========= */
  function switchTab(tab) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('screen-' + tab).classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
  }

  /* ========= TRANG CHỦ ========= */
  let dangQuet = false;
  let scanTimer = null;

  function batDauScan() {
    if (dangQuet) return;
    fetch('/bat_dau_quet', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        dangQuet = true;
        const btn = document.getElementById('btn-scan');
        const overlay = document.getElementById('scan-overlay');
        btn.disabled = true;
        btn.classList.add('scanning');
        overlay.style.display = 'block';
        hienThongBao(data.thong_bao, '#2196F3');

        // Countdown on button
        let dem = 4;
        btn.textContent = `⏳ Đang quét... ${dem}s`;
        scanTimer = setInterval(() => {
          dem--;
          if (dem > 0) {
            btn.textContent = `⏳ Đang quét... ${dem}s`;
          } else {
            btn.textContent = '🔄 Đang xử lý...';
            clearInterval(scanTimer);
            scanTimer = null;
          }
        }, 1000);
      })
      .catch(() => hienThongBao('❌ Lỗi kết nối', '#f44336'));
  }

  function ketThucScan() {
    dangQuet = false;
    if (scanTimer) { clearInterval(scanTimer); scanTimer = null; }
    const btn = document.getElementById('btn-scan');
    btn.textContent = '📷 Bắt Đầu Quét Rác';
    btn.classList.remove('scanning');
    btn.disabled = false;
    document.getElementById('scan-overlay').style.display = 'none';
  }

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
        if (dangQuet && !data.dang_quet) {
          ketThucScan();
          if (data.thung !== 0) {
            const ten = {1:'Vô Cơ', 2:'Hữu Cơ', 3:'Undetermined'}[data.thung] || '';
            hienThongBao(`✅ Đã nhận diện: ${data.nhan} → Mở Thùng ${data.thung} (${ten})`, '#4CAF50');
          }
        }
        document.getElementById('trang-thai').textContent = '🟢 Hệ thống đang hoạt động';
      })
      .catch(() => { document.getElementById('trang-thai').textContent = '🔴 Mất kết nối'; });
  }

  function moThuCong(so) {
    fetch('/mo_nap/' + so, { method: 'POST' })
      .then(r => r.json())
      .then(data => hienThongBao(data.thong_bao, '#4CAF50'));
  }

  function hienThongBao(msg, mau) {
    const tb = document.getElementById('thong-bao');
    tb.textContent = msg;
    tb.style.background = mau || '#4CAF50';
    tb.style.display = 'block';
    setTimeout(() => tb.style.display = 'none', 4000);
  }

  /* ========= CẤU HÌNH – Gửi lệnh ========= */
  function guiCmd(cmdObj) {
    fetch('/gui_lenh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cmdObj)
    })
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        logCfg(`✅ Đã gửi: ${JSON.stringify(cmdObj)}`, 'log-ok');
      } else {
        logCfg(`❌ ${data.error || 'Lỗi không xác định'}: ${JSON.stringify(cmdObj)}`, 'log-err');
      }
    })
    .catch(err => logCfg(`❌ Lỗi: ${err}`, 'log-err'));
  }

  function guiServo(bin) {
    const open  = parseInt(document.getElementById('open-'  + bin).value);
    const close = parseInt(document.getElementById('close-' + bin).value);
    guiCmd({ cmd: 'set', bin: bin, open: open, close: close });
  }

  function guiTime(bin) {
    const t = parseInt(document.getElementById('time-' + bin).value);
    guiCmd({ cmd: 'set', bin: bin, time: t });
  }

  function guiCustom() {
    const ta  = document.getElementById('cmd-custom');
    const raw = ta.value.trim();
    let parsed;
    try {
      parsed = JSON.parse(raw);
      ta.classList.remove('error');
    } catch(e) {
      ta.classList.add('error');
      logCfg(`❌ JSON không hợp lệ: ${e.message}`, 'log-err');
      return;
    }
    guiCmd(parsed);
  }

  function logCfg(msg, cls) {
    const log = document.getElementById('cfg-log');
    const now = new Date().toLocaleTimeString('vi-VN');
    const line = document.createElement('div');
    line.className = cls || '';
    line.textContent = `[${now}] ${msg}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
    // Giữ tối đa 60 dòng
    while (log.children.length > 60) log.removeChild(log.firstChild);
  }

  function capNhatThongKe() {
    fetch('/thong_ke')
      .then(r => r.json())
      .then(d => {
        const tong = d.tong || 0;
        document.getElementById('stat-total').textContent = `Tổng: ${tong} lần`;
        const items = [
          { id: 'voco',  val: d['Vo Co']       || 0 },
          { id: 'huuco', val: d['Huu Co']       || 0 },
          { id: 'other', val: (d['Khong Xac Dinh'] || 0) + (d['Undetermined'] || 0) },
        ];
        items.forEach(({id, val}) => {
          const pct = tong > 0 ? (val / tong * 100).toFixed(1) : 0;
          document.getElementById('cnt-' + id).textContent = val;
          document.getElementById('bar-' + id).style.width  = pct + '%';
          document.getElementById('pct-' + id).textContent  = pct + '%';
        });
      });
  }

  function resetThongKe() {
    if (!confirm('Reset toàn bộ thống kê về 0?')) return;
    fetch('/reset_thong_ke', { method: 'POST' })
      .then(r => r.json())
      .then(() => {
        capNhatThongKe();
        logCfg('✅ Đã reset thống kê', 'log-ok');
      });
  }

  function pullCode() {
    if (!confirm('Pull code mới nhất từ GitHub và restart hệ thống?')) return;
    const btn = document.getElementById('btn-pull');
    const log = document.getElementById('pull-log');
    btn.disabled = true;
    btn.textContent = '⏳ Đang pull...';
    log.style.display = 'block';
    log.textContent = 'Đang kết nối GitHub...';
    fetch('/pull_and_restart', { method: 'POST' })
      .then(r => r.json())
      .then(data => {
        log.textContent = data.message;
        log.style.color = data.ok ? '#4CAF50' : '#f44336';
        if (data.ok) {
          btn.textContent = '🔄 Đang restart... (chờ ~10s)';
          setTimeout(() => {
            btn.disabled = false;
            btn.textContent = '⬇️ Pull Code Mới Nhất & Restart';
          }, 12000);
        } else {
          btn.disabled = false;
          btn.textContent = '⬇️ Pull Code Mới Nhất & Restart';
        }
      })
      .catch(() => {
        log.textContent = '❌ Mất kết nối (có thể đang restart...)';
        log.style.color = '#FF9800';
        btn.textContent = '🔄 Đang restart... (chờ ~10s)';
        setTimeout(() => {
          btn.disabled = false;
          btn.textContent = '⬇️ Pull Code Mới Nhất & Restart';
        }, 12000);
      });
  }

  function taiCauHinhServo() {
    fetch('/servo_config')
      .then(r => r.json())
      .then(cfg => {
        [1, 2, 3].forEach(bin => {
          if (cfg[bin]) {
            document.getElementById('open-'  + bin).value = cfg[bin].open;
            document.getElementById('close-' + bin).value = cfg[bin].close;
            document.getElementById('time-'  + bin).value = cfg[bin].time;
          }
        });
      });
  }

  setInterval(capNhatKetQua, 1000);
  setInterval(capNhatThongKe, 3000);
  capNhatKetQua();
  capNhatThongKe();
  taiCauHinhServo();
</script>
</body>
</html>
"""

@app.route('/')
def trang_chu():
    return render_template_string(TRANG_WEB)

@app.route('/camera_live')
def camera_live():
    """Stream video trực tiếp từ camera; hiển thị placeholder khi camera tắt"""
    def generate():
        placeholder = _tao_frame_placeholder()
        while True:
            with lock:
                frame = ket_qua_hien_tai.get("frame")
            if frame is None:
                # Camera đang tắt – gửi placeholder với tốc độ chậm (2fps)
                _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + buffer.tobytes()
                       + b'\r\n')
                time.sleep(0.5)
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
    """Bắt đầu chế độ quét: LED sáng + beep, sau 2 giây lấy frame cuối để quyết định"""
    global scan_mode, scan_start_time
    with scan_lock:
        scan_mode = True
        scan_start_time = time.time()
    arduino_scan_start()
    return jsonify({"thong_bao": "📷 Bắt đầu quét – đang nhận diện trong 4 giây..."})

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
    ten = {1: "Vô Cơ", 2: "Hữu Cơ", 3: "Undetermined"}.get(so, "?")
    if so in [1, 2, 3]:
        arduino_open_bin(so)
        return jsonify({"thong_bao": f"✅ Đã mở Thùng {so} – {ten}"})
    return jsonify({"thong_bao": "❌ Số thùng không hợp lệ"})

@app.route('/thong_ke')
def lay_thong_ke():
    with thong_ke_lock:
        data = dict(thong_ke)
    data["tong"] = sum(data.values())
    return jsonify(data)

@app.route('/reset_thong_ke', methods=['POST'])
def reset_thong_ke():
    with thong_ke_lock:
        for key in thong_ke:
            thong_ke[key] = 0
    return jsonify({"ok": True})

@app.route('/servo_config')
def lay_servo_config():
    return jsonify(SERVO_CONFIG)

@app.route('/gui_lenh', methods=['POST'])
def gui_lenh_raw():
    """Nhận JSON tuỳ ý từ trang Cấu Hình và forward thẳng sang Arduino"""
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"ok": False, "error": "JSON không hợp lệ"}), 400
    with arduino_lock:
        connected = arduino is not None and arduino.is_open
    if not connected:
        return jsonify({"ok": False, "error": "⚠️ Arduino chưa kết nối"}), 503
    # Remap open_bin qua BIN_MAP giống như arduino_open_bin()
    if data.get("cmd") == "open_bin" and "bin" in data:
        data = dict(data, bin=BIN_MAP.get(data["bin"], data["bin"]))
    # Mirror set commands into SERVO_CONFIG so angles survive reconnects
    if data.get("cmd") == "set":
        bin_num = data.get("bin")
        if bin_num in SERVO_CONFIG:
            if "open"  in data: SERVO_CONFIG[bin_num]["open"]  = data["open"]
            if "close" in data: SERVO_CONFIG[bin_num]["close"] = data["close"]
            if "time"  in data: SERVO_CONFIG[bin_num]["time"]  = data["time"]
    gui_lenh_arduino(data)
    return jsonify({"ok": True, "sent": data})

@app.route('/pull_and_restart', methods=['POST'])
def pull_and_restart():
    """Pull code mới nhất từ GitHub rồi restart service"""
    import subprocess
    try:
        result = subprocess.run(
            ['git', '-C', BASE_DIR, 'pull', 'origin', 'main'],
            capture_output=True, text=True, timeout=30
        )
        git_output = result.stdout.strip() + result.stderr.strip()
        if result.returncode != 0:
            return jsonify({"ok": False, "message": f"Git pull lỗi: {git_output}"})

        # Restart service sau 1 giây (để Flask kịp trả response về client)
        def do_restart():
            time.sleep(1)
            subprocess.run(['sudo', 'systemctl', 'restart', 'thungrac'])
        threading.Thread(target=do_restart, daemon=True).start()

        return jsonify({"ok": True, "message": f"✅ Pull thành công!\n{git_output}\n🔄 Đang restart..."})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "message": "❌ Git pull timeout (>30s)"})
    except Exception as e:
        return jsonify({"ok": False, "message": f"❌ Lỗi: {str(e)}"})

# ---------------------------------------------------------------
# 🚀  KHỞI ĐỘNG CHƯƠNG TRÌNH
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 50)
    print("  THÙNG RÁC THÔNG MINH – Raspberry Pi")
    print("=" * 50)

    ket_noi_arduino()

    # Thread tự động kết nối lại Arduino khi mất kết nối
    thread_reconnect = threading.Thread(target=vong_lap_ket_noi_lai, daemon=True)
    thread_reconnect.start()

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
