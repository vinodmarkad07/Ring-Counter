"""Central configuration for RingCount AI.

All values can be overridden with environment variables.  The application is
usable without a YOLO weight file: it falls back to the classical multi-strip
detector and reports that model status honestly.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    APP_VERSION = os.getenv("RINGCOUNT_VERSION", "5.0.0")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = _int("PORT", 5000)
    DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

    MAX_UPLOAD_MB = _float("MAX_UPLOAD_MB", 4.0)
    MAX_CONTENT_LENGTH = int(MAX_UPLOAD_MB * 1024 * 1024)
    MAX_IMAGE_WIDTH = _int("MAX_IMAGE_WIDTH", 1600)
    JPEG_QUALITY = _int("JPEG_QUALITY", 88)

    MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "weights" / "ring_best.pt")))
    MODEL_CONFIDENCE = _float("MODEL_CONFIDENCE", 0.35)
    MODEL_IOU = _float("MODEL_IOU", 0.50)
    MODEL_IMAGE_SIZE = _int("MODEL_IMAGE_SIZE", 1024)
    DEVICE = os.getenv("DEVICE", "auto")

    N_STRIPS = _int("N_STRIPS", 9)
    STRIP_WIDTH_FRAC = _float("STRIP_WIDTH_FRAC", 0.08)
    AXIS_WIDTH_FRAC = _float("AXIS_WIDTH_FRAC", 0.72)
    MIN_ROI_HEIGHT_FRAC = _float("MIN_ROI_HEIGHT_FRAC", 0.12)
    MIN_RING_GAP_PX = _int("MIN_RING_GAP_PX", 5)
    MAX_RING_GAP_FRAC = _float("MAX_RING_GAP_FRAC", 0.20)
    PEAK_PROMINENCE = _float("PEAK_PROMINENCE", 0.55)
    STRIP_AGREEMENT_MIN = _float("STRIP_AGREEMENT_MIN", 0.28)
    MISSING_GAP_RATIO = _float("MISSING_GAP_RATIO", 1.75)
    MISSING_EVIDENCE = _float("MISSING_EVIDENCE", 0.62)
    CONFIDENCE_HIGH = _int("CONFIDENCE_HIGH", 85)
    CONFIDENCE_MEDIUM = _int("CONFIDENCE_MEDIUM", 70)

    BLUR_WARN = _float("BLUR_WARN", 60.0)
    BRIGHTNESS_LOW = _float("BRIGHTNESS_LOW", 35.0)
    BRIGHTNESS_HIGH = _float("BRIGHTNESS_HIGH", 225.0)
    GLARE_THRESHOLD = _int("GLARE_THRESHOLD", 245)

    STABILITY_WINDOW = _int("STABILITY_WINDOW", 7)
    MIN_STABLE_FRAMES = _int("MIN_STABLE_FRAMES", 5)
    TARGET_ACCURACY = _float("TARGET_ACCURACY", 0.90)

    DEFAULT_DATABASE = "/tmp/ringcount.db" if os.getenv("VERCEL") else str(
        ROOT_DIR / "database" / "ringcount.db"
    )
    DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DEFAULT_DATABASE))
    HISTORY_LIMIT = _int("HISTORY_LIMIT", 100)
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


SETTINGS = Settings()
