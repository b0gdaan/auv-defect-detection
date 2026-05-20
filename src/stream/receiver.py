"""
Video stream receiver — shore station side.

Receives compressed frames from the AUV, decompresses them and passes
each frame to registered callbacks (display, defect detector, recorder).
"""

import io
import json
import logging
import socket
import struct
import threading
from typing import Callable

import cv2
import numpy as np

from src.compression import decompress

logger = logging.getLogger(__name__)

FrameCallback = Callable[[str, np.ndarray], None]  # (channel, frame) -> None

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9000


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError("AUV disconnected")
        buf += chunk
    return buf


def _parse_packet(sock: socket.socket) -> tuple[str, np.ndarray]:
    """
    Read one packet from socket, decompress, decode JPEG → ndarray.
    Returns (channel_name, frame).
    """
    header     = _recv_exact(sock, 4)
    body_len   = struct.unpack(">I", header)[0]
    body       = _recv_exact(sock, body_len)

    channel    = body[0:1].decode()          # 'F' or 'D'
    table_len  = struct.unpack(">I", body[1:5])[0]
    table_json = body[5 : 5 + table_len]
    payload    = body[5 + table_len :]

    codes      = {k: v for k, v in json.loads(table_json).items()}
    # JSON serialises int keys as strings — restore int keys
    codes      = {int(k): v for k, v in codes.items()}

    raw_jpeg   = decompress(payload, codes)
    arr        = np.frombuffer(raw_jpeg, dtype=np.uint8)
    frame      = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return ("front" if channel == "F" else "downward"), frame


class StreamReceiver:
    """
    Listens for incoming AUV connections and dispatches frames to callbacks.

    Usage:
        def on_frame(channel, frame):
            cv2.imshow(channel, frame)

        rx = StreamReceiver(callbacks=[on_frame])
        rx.start()
    """

    def __init__(
        self,
        callbacks: list[FrameCallback] | None = None,
        host: str = LISTEN_HOST,
        port: int = LISTEN_PORT,
    ):
        self.callbacks = callbacks or []
        self.host = host
        self.port = port
        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats  = {"received": 0, "bytes": 0, "errors": 0}

    def start(self):
        self._thread = threading.Thread(target=self._listen, daemon=True, name="receiver")
        self._thread.start()
        logger.info("Receiver listening on %s:%d", self.host, self.port)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def add_callback(self, cb: FrameCallback):
        self.callbacks.append(cb)

    # ── internal ──────────────────────────────────────────────────────────────

    def _listen(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            srv.settimeout(1.0)
            logger.info("Waiting for AUV connection …")

            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except TimeoutError:
                    continue
                logger.info("AUV connected from %s", addr)
                self._handle(conn)

    def _handle(self, conn: socket.socket):
        with conn:
            while not self._stop.is_set():
                try:
                    channel, frame = _parse_packet(conn)
                    self._stats["received"] += 1
                    for cb in self.callbacks:
                        cb(channel, frame)
                except ConnectionResetError:
                    logger.info("AUV disconnected")
                    break
                except Exception as exc:
                    logger.error("Packet error: %s", exc)
                    self._stats["errors"] += 1
