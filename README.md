# Glaideron AUV — Underwater Defect Detection System

> Bachelor's thesis project · SamGTU · 2023
> Developed in cooperation with [NPK "Network-Centric Platforms"](https://network-centric.ru/)
> RZD Grant No. 2/22-РЖД/2022

Research prototype associated with a bachelor’s thesis on an optoelectronic system for the **Glaideron** AUV. The public code combines video streaming with four-class, frame-level image classification; it does not localize defects with bounding boxes or masks.

---

## What it does

The system streams live video from two cameras mounted on the AUV to a shore station, where a convolutional neural network classifies each frame for structural damage:

| Defect class | Description |
|---|---|
| `crack` | Linear cracks in concrete or masonry |
| `concrete_damage` | Spalling, crumbling, section loss |
| `empty_joint` | Missing mortar in brick/stone joints |
| `normal` | No visible damage |

Earlier project documentation reports **84% top-1 accuracy on 200 frames**. The public repository does not include the trained defect checkpoint, dataset split or evaluation outputs needed to reproduce that figure. Treat it as a reported result, not a benchmark of a fresh checkout.

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
    └── detector.py          # Inference pipeline + OpenCV overlay

# CLI entry point: src/main.py

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
python -m src.main demo --video sample_footage.mp4 --weights weights/defectoscope.pth
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

The video and trained checkpoint in the commands are user-supplied; neither is
included. Without defect-specific weights, the replacement classifier head is
untrained and its labels are not meaningful. The code runs inference on the shore
station in `shore` mode; the Raspberry Pi captures and transmits video.

## Model

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Head:** Dropout → Linear(1280→128) → ReLU → Dropout → Linear(128→4)
- **Training data:** ~1 200 labelled frames from AUV footage
- **Augmentation:** horizontal flip, brightness ±30 %, CLAHE (turbid water contrast)
- **Reported inference time:** ~115 ms on Raspberry Pi 4; no reproducible hardware benchmark is bundled

---

## Compression

Frames are JPEG-encoded and passed through **Huffman coding** before transmission. Measure the complete packet (including the code table) on representative footage: additional compression of JPEG is not guaranteed to reduce network traffic.
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


## Security limits

The streaming protocol is a research prototype without authentication or encryption.
Use only on an isolated, trusted network. It is not hardened against oversized or
malformed packets. Load only trusted model files; checkpoint loading uses PyTorch's
restricted `weights_only=True` mode. This repository is not a safety-certified inspection system.
