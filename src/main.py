"""
Glaideron AUV — Opto-Electronic Vision System
Entry point.

Run modes:
    auv       – runs on Raspberry Pi 4 aboard the AUV
                (captures video, compresses, streams to shore)

    shore     – runs on the operator's laptop
                (receives stream, runs defect detector, shows live feed)

    demo      – shore mode with a local video file instead of live stream
                (no AUV connection required)

Usage:
    python -m src.main auv   --host 192.168.0.100
    python -m src.main shore --weights weights/defectoscope.pth
    python -m src.main demo  --video sample_footage.mp4 --weights weights/defectoscope.pth
"""

import argparse
import logging
import sys
import time

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── AUV mode ──────────────────────────────────────────────────────────────────

def run_auv(args):
    from src.stream.capture import DualCameraCapture, CameraConfig
    from src.stream.transmitter import StreamTransmitter

    logger.info("=== AUV MODE — starting cameras and transmitter ===")
    cameras = DualCameraCapture(
        front_cfg=CameraConfig(index=0, width=1280, height=720, fps=30, name="front"),
        downward_cfg=CameraConfig(index=1, width=1280, height=720, fps=30, name="downward"),
    ).start()

    tx = StreamTransmitter(cameras, host=args.host, port=args.port, fps=args.fps)
    tx.start()

    try:
        while True:
            time.sleep(5)
            s = tx.stats
            logger.info("TX stats — sent=%d  bytes=%d  errors=%d",
                        s["sent"], s["bytes"], s["errors"])
    except KeyboardInterrupt:
        logger.info("Shutting down …")
    finally:
        tx.stop()
        cameras.stop()


# ── Shore mode ────────────────────────────────────────────────────────────────

def run_shore(args):
    from src.detection.detector import DefectDetector
    from src.stream.receiver import StreamReceiver

    logger.info("=== SHORE MODE — waiting for AUV stream ===")

    detector = DefectDetector(
        weights_path=args.weights,
        confidence_threshold=args.threshold,
        log_path=args.log,
    )

    def on_frame(channel: str, frame):
        if channel != "downward":
            cv2.imshow(f"AUV — {channel}", frame)
            return

        det        = detector.predict(frame)
        annotated  = detector.draw(frame, det)
        cv2.imshow("Defect Detector — downward", annotated)

        if det.class_name != "normal":
            logger.warning("DEFECT  class=%s  conf=%.2f", det.class_name, det.confidence)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    rx = StreamReceiver(callbacks=[on_frame], port=args.port)
    rx.start()

    try:
        while True:
            time.sleep(10)
            s = rx.stats
            logger.info("RX stats — received=%d  errors=%d  avg_infer=%.1f ms",
                        s["received"], s["errors"], detector.avg_inference_ms)
    except KeyboardInterrupt:
        logger.info("Shutting down …")
    finally:
        rx.stop()
        detector.close()
        cv2.destroyAllWindows()


# ── Demo mode (local video file) ──────────────────────────────────────────────

def run_demo(args):
    from src.detection.detector import DefectDetector

    logger.info("=== DEMO MODE — file: %s ===", args.video)

    detector = DefectDetector(
        weights_path=args.weights,
        confidence_threshold=args.threshold,
        log_path=None,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay = int(1000 / fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det      = detector.predict(frame)
        annotated = detector.draw(frame, det)
        cv2.imshow("Defect Detector — demo", annotated)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo finished.  Avg inference: %.1f ms", detector.avg_inference_ms)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Glaideron AUV Vision System")
    sub = p.add_subparsers(dest="mode", required=True)

    # auv
    a = sub.add_parser("auv", help="Run on Raspberry Pi 4 (transmit)")
    a.add_argument("--host",  default="192.168.0.100")
    a.add_argument("--port",  type=int, default=9000)
    a.add_argument("--fps",   type=int, default=15)

    # shore
    s = sub.add_parser("shore", help="Run on shore station (receive + detect)")
    s.add_argument("--port",      type=int,   default=9000)
    s.add_argument("--weights",   default=None)
    s.add_argument("--threshold", type=float, default=0.60)
    s.add_argument("--log",       default="detections.csv")

    # demo
    d = sub.add_parser("demo", help="Run defect detector on a local video file")
    d.add_argument("--video",     required=True)
    d.add_argument("--weights",   default=None)
    d.add_argument("--threshold", type=float, default=0.60)

    args = p.parse_args()
    {"auv": run_auv, "shore": run_shore, "demo": run_demo}[args.mode](args)


if __name__ == "__main__":
    main()
