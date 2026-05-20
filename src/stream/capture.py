"""
Dual-camera capture module for the Glaideron AUV.

Hardware:
    - 2× SJ4000 action cameras (front-facing + downward-facing)
    - Connected to Raspberry Pi 4 via USB / V4L2
    - Resolution: 1080p @ 30 fps (configurable)

The captured frames are handed off to the transmitter which applies
Huffman compression before sending over the TP-Link TL-703 WiFi link.
"""

import time
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Camera indices on Raspberry Pi (V4L2 /dev/video*)
CAM_FRONT   = 0
CAM_DOWNWARD = 1


@dataclass
class CameraConfig:
    index:  int   = 0
    width:  int   = 1280
    height: int   = 720
    fps:    int   = 30
    name:   str   = "camera"


class CameraCapture:
    """
    Thread-safe wrapper around a single OpenCV VideoCapture.

    Usage:
        cam = CameraCapture(CameraConfig(index=0, name='front'))
        cam.start()
        frame = cam.read()   # latest frame, non-blocking
        cam.stop()
    """

    def __init__(self, config: CameraConfig):
        self.cfg   = config
        self._cap  = None
        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "CameraCapture":
        self._cap = cv2.VideoCapture(self.cfg.index, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.cfg.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.cfg.name} (index={self.cfg.index})")

        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.cfg.name}")
        self._thread.start()
        logger.info("Camera '%s' started  %dx%d@%dfps",
                    self.cfg.name, self.cfg.width, self.cfg.height, self.cfg.fps)
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info("Camera '%s' stopped", self.cfg.name)

    # ── internal grab loop ─────────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Camera '%s': missed frame", self.cfg.name)
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    # ── public API ─────────────────────────────────────────────────────────────

    def read(self) -> np.ndarray | None:
        """Return the latest captured frame (BGR, or None if not yet available)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


class DualCameraCapture:
    """
    Manages both AUV cameras simultaneously.

        front    – forward-looking (obstacle avoidance / navigation)
        downward – nadir-looking  (defect detection on structures below)
    """

    def __init__(
        self,
        front_cfg:    CameraConfig = CameraConfig(CAM_FRONT,    name="front"),
        downward_cfg: CameraConfig = CameraConfig(CAM_DOWNWARD, name="downward"),
    ):
        self.front    = CameraCapture(front_cfg)
        self.downward = CameraCapture(downward_cfg)

    def start(self):
        self.front.start()
        self.downward.start()
        return self

    def stop(self):
        self.front.stop()
        self.downward.stop()

    def read_both(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (front_frame, downward_frame) — either may be None."""
        return self.front.read(), self.downward.read()

    def encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        """JPEG-encode a frame before Huffman compression."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return buf.tobytes()
