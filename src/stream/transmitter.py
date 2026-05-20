"""
Video stream transmitter — AUV side (Raspberry Pi 4).

Pipeline:
    Camera frame → JPEG encode → Huffman compress → TCP socket → shore station

Network topology:
    Raspberry Pi 4  ──WiFi──  TP-Link TL-703 (access point)  ──WiFi──  Shore laptop
    192.168.0.10                  192.168.0.1                           192.168.0.100
"""

import io
import json
import logging
import socket
import struct
import time
import threading

import numpy as np

from src.compression import compress
from src.stream.capture import DualCameraCapture

logger = logging.getLogger(__name__)

DEFAULT_HOST = "192.168.0.100"
DEFAULT_PORT = 9000
JPEG_QUALITY = 75          # balance quality vs bandwidth
STREAM_FPS   = 15          # target transmit rate (limited by WiFi throughput)


def _send_packet(sock: socket.socket, channel: str, payload: bytes, codes: dict):
    """
    Packet format:
        4 bytes  – total packet length (big-endian uint32)
        1 byte   – channel tag  ('F'=front, 'D'=downward)
        4 bytes  – code-table length
        N bytes  – code-table (JSON)
        M bytes  – compressed payload
    """
    tag        = channel[0].upper().encode()
    table_json = json.dumps(codes).encode()
    table_len  = struct.pack(">I", len(table_json))

    body   = tag + table_len + table_json + payload
    header = struct.pack(">I", len(body))

    sock.sendall(header + body)


class StreamTransmitter:
    """Continuously captures and streams both camera channels."""

    def __init__(
        self,
        cameras:   DualCameraCapture,
        host:      str = DEFAULT_HOST,
        port:      int = DEFAULT_PORT,
        fps:       int = STREAM_FPS,
    ):
        self.cameras = cameras
        self.host    = host
        self.port    = port
        self.interval = 1.0 / fps

        self._stop   = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats  = {"sent": 0, "bytes": 0, "errors": 0}

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="transmitter")
        self._thread.start()
        logger.info("Transmitter started → %s:%d", self.host, self.port)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    logger.info("Connected to shore station at %s:%d", self.host, self.port)
                    self._stream_loop(sock)
            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                logger.warning("Connection error: %s — retrying in 2 s", exc)
                self._stats["errors"] += 1
                time.sleep(2.0)

    def _stream_loop(self, sock: socket.socket):
        t_next = time.monotonic()
        while not self._stop.is_set():
            front, down = self.cameras.read_both()
            t_next += self.interval
            now = time.monotonic()

            for channel, frame in (("F", front), ("D", down)):
                if frame is None:
                    continue
                try:
                    raw      = self.cameras.encode_jpeg(frame, JPEG_QUALITY)
                    comp, tbl = compress(raw)
                    _send_packet(sock, channel, comp, tbl)
                    self._stats["sent"]  += 1
                    self._stats["bytes"] += len(comp)
                except OSError:
                    raise                     # triggers reconnect
                except Exception as exc:
                    logger.error("Encode/send error: %s", exc)
                    self._stats["errors"] += 1

            sleep_for = t_next - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
