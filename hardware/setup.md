# Hardware Setup

## Components

| Component | Model | Role |
|-----------|-------|------|
| Single-board computer | Raspberry Pi 4 Model B (4 GB) | Main compute unit aboard AUV |
| Front camera | SJ4000 Action Camera | Forward-looking video |
| Downward camera | SJ4000 Action Camera | Nadir-looking (defect inspection) |
| WiFi router | TP-Link TL-703N | Access point, AUV ↔ shore link |

## Network Configuration

```
AUV (Raspberry Pi 4)          Shore station (laptop)
   192.168.0.10        ←WiFi→      192.168.0.100
         |                               |
    [transmitter]               [receiver + detector]
    port 9000 →                     ← port 9000
```

Configure the TP-Link TL-703N in **access-point mode** with:
- SSID: `glaideron-net`
- Security: WPA2-PSK
- Channel: 6 (2.4 GHz, better range in water-adjacent environments)

## Camera Connection (Raspberry Pi)

```bash
# List available V4L2 devices
v4l2-ctl --list-devices

# Front camera  → /dev/video0
# Downward cam  → /dev/video1

# Test capture
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test_front.jpg
```

## Raspberry Pi Setup

```bash
# Install dependencies
sudo apt update && sudo apt install -y python3-pip libopencv-dev

pip install torch torchvision opencv-python --index-url https://download.pytorch.org/whl/cpu

# Clone and run
git clone https://github.com/b0gdaan/auv-defect-detection.git
cd auv-defect-detection
pip install -r requirements.txt

python -m src.main auv --host 192.168.0.100
```

## Wiring Overview

```
SJ4000 (front)    ─USB─┐
                        ├─ Raspberry Pi 4 ─── TP-Link TL-703N
SJ4000 (downward) ─USB─┘       |
                           12V battery pack
                           (waterproof enclosure)
```

## Measured Performance

| Metric | Value |
|--------|-------|
| Video latency (WiFi) | ~120 ms |
| Compression ratio (Huffman) | 61–74 % of original size |
| CNN inference time (RPi 4, CPU) | ~110–130 ms / frame |
| Effective detection frame rate | ~7 fps |
| WiFi range (surface) | ~80 m |
| Max dive depth (AUV) | 5 m |
