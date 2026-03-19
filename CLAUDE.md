# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Thùng Rác Thông Minh** (Smart Waste Bin) — a Raspberry Pi 4 + Arduino system that:
1. Captures camera frames of trash
2. Classifies the waste type using a TFLite AI model (exported from Teachable Machine)
3. Sends JSON commands over USB serial to an Arduino to open the correct bin lid
4. Serves a Vietnamese-language web UI on port **3002**

GitHub repo: `https://github.com/tanit257/recycle-pi4.git`

## File Structure

| File | Purpose |
|---|---|
| `raspberry.py` | Main Python app — Flask server, camera loop, AI inference, Arduino serial control |
| `arduino/thungrac/thungrac.ino` | Arduino firmware — servo control, NeoPixel LED, buzzer |
| `model_unquant.tflite` | Teachable Machine TFLite model |
| `labels.txt` | Class labels: `0 Vo Co`, `1 Huu Co` |
| `thungrac.service` | systemd service file (production version with venv) |
| `setup_autostart.sh` | One-time setup script — installs and enables the systemd service |
| `config.txt` | Static IP config instructions for `/etc/dhcpcd.conf` |

## Running the Service

```bash
# Manual run (activate venv first)
cd /home/pi/Documents/bin/recycle-pi4
source venv/bin/activate
python raspberry.py

# Service management
sudo systemctl status thungrac
sudo systemctl restart thungrac
sudo systemctl stop thungrac

# View live logs
sudo journalctl -u thungrac -f -n 50
```

## One-time Setup (after fresh clone)

```bash
cd /home/pi/Documents/bin/recycle-pi4
python3 -m venv venv
source venv/bin/activate
pip install flask opencv-python numpy pyserial pillow tflite-runtime

# Install and enable systemd autostart
chmod +x setup_autostart.sh
sudo ./setup_autostart.sh
```

> **Note:** The production `thungrac.service` uses `source venv/bin/activate && python ...` (not the bare `python3` that `setup_autostart.sh` generates). After running the setup script, verify the service file still uses the venv path, or manually copy `thungrac.service` to `/etc/systemd/system/` and run `sudo systemctl daemon-reload && sudo systemctl restart thungrac`.

## Updating Code

From the web UI (Settings tab → "Pull Code & Restart") or manually:
```bash
cd /home/pi/Documents/bin/recycle-pi4
git pull
sudo systemctl restart thungrac
```

## Architecture

### raspberry.py — Thread Model

Three concurrent threads:
- **Main thread**: Flask web server on `0.0.0.0:3002`
- **`vong_lap_camera`**: Reads camera frames continuously, runs AI inference every frame, triggers `arduino_open_bin()` after 2-second scan window
- **`vong_lap_ket_noi_lai`**: Reconnect watchdog — polls every 5 seconds, re-attempts Arduino serial connection if lost

Shared state protected by locks:
- `lock` — `ket_qua_hien_tai` dict (current AI result + latest frame)
- `scan_lock` — `scan_mode` bool + `scan_start_time`
- `arduino_lock` — `arduino` serial object
- `thong_ke_lock` — classification statistics dict

### Scan Flow

1. Web UI POST `/bat_dau_quet` → sets `scan_mode=True`, sends `scan_start` to Arduino (LED on + beep)
2. Camera thread detects `scan_mode`, waits 2 seconds, takes the latest AI result
3. If confidence ≥ 80%: open matching bin (1=Vo Co, 2=Huu Co); else open bin 3 (Undetermined)
4. Sends `open_bin` JSON to Arduino, clears `scan_mode`, sends `scan_end`

### Arduino Serial Protocol

Commands are JSON strings terminated by `\n` at 115200 baud:

```json
{"cmd": "scan_start"}
{"cmd": "scan_end"}
{"cmd": "open_bin", "bin": 1}
{"cmd": "beep"}
{"cmd": "set", "bin": 1, "open": 120, "close": 5, "time": 4000}
```

Arduino auto-tries ports: `/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyACM0`, `/dev/ttyACM1`

### Web API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/camera_live` | GET | MJPEG stream |
| `/ket_qua` | GET | Current AI result + scan status (JSON) |
| `/bat_dau_quet` | POST | Start scan |
| `/ket_thuc_quet` | POST | Stop scan manually |
| `/mo_nap/<1\|2\|3>` | POST | Manually open a bin |
| `/thong_ke` | GET | Classification statistics |
| `/reset_thong_ke` | POST | Reset statistics |
| `/gui_lenh` | POST | Forward raw JSON to Arduino |
| `/pull_and_restart` | POST | `git pull` + `systemctl restart thungrac` |

### AI Model

- Input: 224×224 RGB, normalized to `[-1, 1]`
- Output: 2-class softmax — index 0 = `Vo Co` (inorganic → bin 1), index 1 = `Huu Co` (organic → bin 2)
- Falls back to `tflite_runtime`, then `tensorflow.lite` if not installed

### Network

Static IP configured on `wlan0`: `10.133.248.100/24` (gateway `10.133.248.1`).
Web UI accessible at `http://10.133.248.100:3002` or `http://raspberrypi.local:3002` (via Avahi mDNS).

## Key Configuration Constants (in raspberry.py)

```python
ARDUINO_PORTS   = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyACM0', '/dev/ttyACM1']
ARDUINO_BAUD    = 115200
CAMERA_INDEXES  = [0, 1, 2, 3]
NGUONG_TIN_CAY  = 0.80   # 80% confidence threshold
THOI_GIAN_CHO   = 3      # seconds between scans (UI-side, not enforced server-side)
SERVO_CONFIG    = {1: {"open":120,"close":5,"time":4000}, ...}  # same for all 3 bins
```
