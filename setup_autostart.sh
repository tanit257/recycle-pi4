#!/bin/bash
# ================================================================
# Script cài đặt tự động khởi động Thùng Rác Thông Minh
# Chạy 1 lần trên Raspberry Pi:
#   chmod +x setup_autostart.sh
#   sudo ./setup_autostart.sh
# ================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="thungrac"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
CURRENT_USER="${SUDO_USER:-$USER}"

echo "================================================"
echo "  Cài đặt tự động khởi động Thùng Rác"
echo "================================================"
echo "  User    : $CURRENT_USER"
echo "  Thư mục : $PROJECT_DIR"
echo ""

# 1. Ghi file service với đúng path và user
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Thung Rac Thong Minh
After=network-online.target
Wants=network-online.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/python3 $PROJECT_DIR/raspberry.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Đã tạo $SERVICE_FILE"

# 2. Bật mDNS (Avahi) để truy cập qua raspberrypi.local
if ! dpkg -s avahi-daemon &>/dev/null; then
    echo "📦 Cài avahi-daemon (mDNS)..."
    apt-get install -y avahi-daemon
fi
systemctl enable avahi-daemon
systemctl start avahi-daemon
echo "✅ mDNS (Avahi) đã bật – truy cập qua hostname.local"

# 3. Bật và khởi động service
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo ""
echo "================================================"
echo "  XONG! Thùng rác sẽ tự chạy khi Pi khởi động."
echo ""

# Lấy hostname và IP để hiển thị
HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')

echo "  Truy cập website:"
echo "    http://${HOSTNAME}.local:3002   ← LUÔN DÙNG CÁI NÀY"
echo "    http://${IP}:3002               ← IP hiện tại (có thể đổi)"
echo ""
echo "  Lệnh hữu ích:"
echo "    sudo systemctl status thungrac   # Xem trạng thái"
echo "    sudo systemctl stop thungrac     # Dừng"
echo "    sudo systemctl restart thungrac  # Khởi động lại"
echo "    sudo journalctl -u thungrac -f   # Xem log realtime"
echo "================================================"
