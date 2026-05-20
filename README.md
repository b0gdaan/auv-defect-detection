# Glaideron AUV — Underwater Defect Detection System

> Bachelor's thesis project · SamGTU · 2023  
> Developed in cooperation with [NPK "Network-Centric Platforms"](https://network-centric.ru/)  
> RZD Grant No. 2/22-РЖД/2022

Real-time vision system for detecting structural defects in underwater hydraulic infrastructure (bridges, dams, piers), deployed on the **Glaideron** autonomous underwater vehicle.

---

## What it does

The system streams live video from two cameras mounted on the AUV to a shore station, where a convolutional neural network classifies each frame for structural damage:

| Defect class | Description |
|---|---|
| `crack` | Linear cracks in concrete or masonry |
| `concrete_damage` | Spalling, crumbling, section loss |
| `empty_joint` | Missing mortar in brick/stone joints |
| `normal` | No visible damage |

**Detection accuracy: 84 % top-1** on a 200-frame held-out test set.

---

## System Architecture

```
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│         Glaideron AUV           │        │         Shore Station            │
│                                 │        │                                 │
│  SJ4000 (front)  ─┐             │  WiFi  │  ┌─ Receiver                   │
│                   ├─ RPi 4 ─────┼────────┼──┤                             │
│  SJ4000 (nadir)  ─┘  │          │        │  └─ DefectDetector (CNN)        │
│                      │          │        │       ↓                         │
│                   Huffman       │        │  Live annotated feed            │
│                   compress      │        │  + CSV defect log               │
└─────────────────────────────────┘        └─────────────────────────────────┘
         TP-Link TL-703N access point (192.168.0.1)
```

---

## AUV Specifications

| Parameter | Value |
|---|---|
| Vehicle | Glaideron ANPA |
| Max depth | 5 m |
| Endurance | up to 5 nautical miles |
| Weight | ≤ 150 kg |
| Cruise speed | 2 knots |
| Compute (vision) | Raspberry Pi 4 Model B (4 GB) |
| Cameras | 2 × SJ4000 Action Camera |
| Wireless link | TP-Link TL-703N (2.4 GHz, ~80 m surface range) |

---

## Repository Structure

```
src/
├── compression.py          # Huffman coding for video transmission
├── stream/
│   ├── capture.py          # Dual-camera capture (OpenCV / V4L2)
│   ├── transmitter.py      # AUV-side stream sender
│   └── receiver.py         # Shore-side stream receiver
└── detection/
    ├── classes.py           # Defect class definitions
    ├── model.py             # MobileNetV2-based CNN
    ├── detector.py          # Inference pipeline + OpenCV overlay
    └── main.py             # CLI entry point

notebooks/
└── defect_detection_training.ipynb   # Model training walkthrough

hardware/
└── setup.md                # Wiring, network config, measured performance

docs/
└── thesis_ru.pdf           # Full bachelor's thesis (Russian)
```

---

## Quick Start

```bash
git clone https://github.com/b0gdaan/auv-defect-detection.git
cd auv-defect-detection
pip install -r requirements.txt
```

**Run defect detector on a local video file (no AUV needed):**
```bash
python -m src.main demo --video sample_footage.mp4
```

**AUV side (Raspberry Pi 4):**
```bash
python -m src.main auv --host 192.168.0.100
```

**Shore station:**
```bash
python -m src.main shore --weights weights/defectoscope.pth
```

---

## Model

- **Backbone:** MobileNetV2 (ImageNet pretrained)  
- **Head:** Dropout → Linear(1280→128) → ReLU → Dropout → Linear(128→4)  
- **Training data:** ~1 200 labelled frames from AUV footage  
- **Augmentation:** horizontal flip, brightness ±30 %, CLAHE (turbid water contrast)  
- **Inference time:** ~115 ms on Raspberry Pi 4 (CPU only)

---

## Compression

Frames are JPEG-encoded then further compressed with **Huffman coding** before transmission, reducing payload size to **61–74 %** of the original JPEG.  
See [`src/compression.py`](src/compression.py) for the full implementation.

---

## References

- [Glaideron AUV — sea trials report](https://network-centric.ru/news/завершены-испытания-глайдерона/) (Russian)  
- [Intelligent Defectoscope product page](https://network-centric.ru/products/интеллектуальный-дефектоскоп/) (Russian)  
- Field demo videos: [trial 1](https://www.youtube.com/watch?v=iciuMopU_TA) · [trial 2](https://www.youtube.com/watch?v=s7XKP502s5U)  
- Full thesis (Russian): [`docs/thesis_ru.pdf`](docs/thesis_ru.pdf)

---

## Tech Stack

`Python 3.10` · `PyTorch 2.0` · `OpenCV` · `torchvision` · `Raspberry Pi 4` · `MobileNetV2`
