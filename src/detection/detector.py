"""
Real-time defect detection pipeline.

Accepts frames from the stream receiver and runs the CNN classifier.
Results are overlaid on the frame and optionally saved to a log file.
"""

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.detection.classes import CLASS_NAMES, get_class, get_color
from src.detection.model import build_model, load_weights

logger = logging.getLogger(__name__)

# Input size expected by MobileNetV2
INPUT_SIZE = 224

PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    timestamp:  float


class DefectDetector:
    """
    Wraps the CNN model and provides frame-level inference.

    Usage:
        det = DefectDetector(weights_path="weights/defectoscope.pth")
        result = det.predict(frame)       # -> Detection
        frame_out = det.draw(frame, result)
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device:       str = "cpu",
        confidence_threshold: float = 0.60,
        log_path: str | None = "detections.csv",
    ):
        self.device = torch.device(device)
        self.threshold = confidence_threshold

        self.model = build_model(pretrained=(weights_path is None))
        if weights_path:
            self.model = load_weights(self.model, weights_path, device)
        self.model.to(self.device)
        self.model.eval()

        self._log_path = log_path
        self._log_file = None
        self._csv_writer = None
        if log_path:
            self._open_log(log_path)

        self._inference_times: list[float] = []

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Detection:
        """Run classifier on a single BGR frame."""
        tensor = PREPROCESS(frame).unsqueeze(0).to(self.device)

        t0  = time.monotonic()
        out = self.model(tensor)
        dt  = time.monotonic() - t0
        self._inference_times.append(dt)

        probs = F.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
        det = Detection(
            class_id=int(idx),
            class_name=CLASS_NAMES[int(idx)],
            confidence=float(conf),
            timestamp=time.time(),
        )

        if self._csv_writer and det.confidence >= self.threshold:
            self._csv_writer.writerow([
                det.timestamp,
                det.class_name,
                f"{det.confidence:.4f}",
            ])

        return det

    # ── overlay ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        """Draw detection result on a copy of the frame."""
        out   = frame.copy()
        cls   = get_class(det.class_id)
        color = get_color(det.class_id)

        label = f"{cls.label_ru}  {det.confidence*100:.1f}%"
        cv2.rectangle(out, (10, 10), (10 + 400, 50), (20, 20, 20), -1)
        cv2.putText(out, label, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        if cls.severity > 0:
            # Red border for defects
            h, w = out.shape[:2]
            thickness = 4 if cls.severity == 2 else 2
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)

        return out

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def avg_inference_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times) * 1000

    # ── log ───────────────────────────────────────────────────────────────────

    def _open_log(self, path: str):
        self._log_file  = open(path, "a", newline="")
        self._csv_writer = csv.writer(self._log_file)
        if Path(path).stat().st_size == 0:
            self._csv_writer.writerow(["timestamp", "class", "confidence"])

    def close(self):
        if self._log_file:
            self._log_file.close()
