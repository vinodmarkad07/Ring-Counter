"""Image decoding, resizing and quality measurements."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import SETTINGS


@dataclass(frozen=True)
class ImageQuality:
    sharpness: float
    brightness: float
    contrast: float
    glare_percent: float
    width: int
    height: int

    @property
    def usable(self) -> bool:
        return (
            self.width >= 240
            and self.height >= 240
            and self.sharpness >= SETTINGS.BLUR_WARN
            and SETTINGS.BRIGHTNESS_LOW <= self.brightness <= SETTINGS.BRIGHTNESS_HIGH
            and self.glare_percent < 35.0
        )


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("The uploaded image is empty.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the image. Upload a valid JPG, PNG, or WEBP file.")
    return image


def resize_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= SETTINGS.MAX_IMAGE_WIDTH:
        return image, 1.0
    scale = SETTINGS.MAX_IMAGE_WIDTH / float(width)
    resized = cv2.resize(
        image,
        (SETTINGS.MAX_IMAGE_WIDTH, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def quality_metrics(gray: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> ImageQuality:
    region = gray
    if roi is not None:
        x, y, width, height = roi
        region = gray[y : y + height, x : x + width]
    if region.size == 0:
        region = gray
    glare = float(np.mean(region >= SETTINGS.GLARE_THRESHOLD) * 100.0)
    return ImageQuality(
        sharpness=float(cv2.Laplacian(region, cv2.CV_64F).var()),
        brightness=float(np.mean(region)),
        contrast=float(np.std(region)),
        glare_percent=glare,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
    )


def enhanced_gray(gray: np.ndarray) -> np.ndarray:
    """Improve local contrast without erasing thin ring boundaries."""
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def normalize_signal(signal: np.ndarray, window: int = 41) -> np.ndarray:
    window = max(5, int(window) | 1)
    pad = window // 2
    padded = np.pad(signal.astype(np.float32), pad, mode="reflect")
    kernel = np.ones(window, dtype=np.float32) / window
    mean = np.convolve(padded, kernel, mode="valid")
    second = np.convolve(padded * padded, kernel, mode="valid")
    std = np.sqrt(np.maximum(second - mean * mean, 1e-4))
    return (signal - mean) / std


def row_edge_profile(gray_roi: np.ndarray) -> np.ndarray:
    """Return a robust horizontal-boundary signal for a vertical stack.

    Median/percentile aggregation makes a line count only when it appears
    across a meaningful portion of the selected stack width, which reduces
    scratches and isolated glare compared with a single centre line.
    """
    sobel_y = np.abs(cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3))
    edge_signal = np.percentile(sobel_y, 60, axis=1)
    intensity_signal = np.abs(np.gradient(np.median(gray_roi, axis=1).astype(np.float32)))
    edge_signal = cv2.GaussianBlur(edge_signal.reshape(-1, 1), (1, 7), 0).ravel()
    intensity_signal = cv2.GaussianBlur(intensity_signal.reshape(-1, 1), (1, 7), 0).ravel()
    edge_signal = normalize_signal(edge_signal)
    intensity_signal = normalize_signal(intensity_signal)
    combined = 0.70 * edge_signal + 0.30 * intensity_signal
    return np.maximum(combined, 0.0)


def encode_jpeg(image: np.ndarray, quality: int | None = None) -> str:
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality or SETTINGS.JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Could not encode the result image.")
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")