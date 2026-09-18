"""Hybrid ring counting engine.

The classical path is deliberately independent of Flask so it can later be
used for video frames or RTSP input.  A YOLO model is optional; when a valid
weight file and ultralytics are available, its vertical detections are fused
with the multi-strip signal rather than blindly counted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import SETTINGS
from .preprocessing import (
    ImageQuality,
    decode_image,
    encode_jpeg,
    enhanced_gray,
    quality_metrics,
    resize_image,
    row_edge_profile,
)
from .validation import ROI, build_roi, consistency_from_gaps, spacing_statistics

logger = logging.getLogger(__name__)


class RingDetectionError(ValueError):
    """A user-correctable detection error."""


@dataclass
class StripCandidate:
    position: float
    strength: float


def _local_maxima(signal: np.ndarray, minimum_distance: int, threshold: float) -> list[int]:
    """Simple scipy-free peak finder with distance-aware suppression."""
    if len(signal) < 3:
        return []
    raw = [
        index
        for index in range(1, len(signal) - 1)
        if signal[index] >= signal[index - 1]
        and signal[index] > signal[index + 1]
        and signal[index] >= threshold
    ]
    raw.sort(key=lambda index: float(signal[index]), reverse=True)
    selected: list[int] = []
    for index in raw:
        if all(abs(index - other) >= minimum_distance for other in selected):
            selected.append(index)
    return sorted(selected)


def _cluster_candidates(
    candidates: list[tuple[float, int, float]],
    tolerance: float,
    minimum_agreement: float,
    strip_count: int,
) -> tuple[list[float], list[float], int]:
    """Align peaks from different horizontal strips by their Y coordinate."""
    if not candidates:
        return [], [], 0
    clusters: list[list[tuple[float, int, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if not clusters or candidate[0] - np.median([item[0] for item in clusters[-1]]) > tolerance:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    positions: list[float] = []
    agreements: list[float] = []
    rejected = 0
    for cluster in clusters:
        hit_strips = len({item[1] for item in cluster})
        agreement = hit_strips / max(1, strip_count)
        if agreement >= minimum_agreement:
            weights = np.asarray([max(item[2], 0.01) for item in cluster])
            values = np.asarray([item[0] for item in cluster])
            positions.append(float(np.average(values, weights=weights)))
            agreements.append(float(agreement))
        else:
            rejected += 1
    return positions, agreements, rejected


def _quality_score(quality: ImageQuality) -> float:
    sharpness = min(1.0, quality.sharpness / max(SETTINGS.BLUR_WARN * 3.0, 1.0))
    brightness = 1.0 - min(1.0, abs(quality.brightness - 128.0) / 128.0)
    glare = max(0.0, 1.0 - quality.glare_percent / 25.0)
    resolution = min(1.0, min(quality.width, quality.height) / 900.0)
    return float(np.mean([sharpness, brightness, glare, resolution]))


class RingCounter:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_error: str | None = None
        self.device = "cpu"
        self._load_optional_model()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def _load_optional_model(self) -> None:
        model_path = Path(SETTINGS.MODEL_PATH)
        if not model_path.exists():
            self.model_error = f"Model not found: {model_path}"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(str(model_path))
            self.device = self._detect_device()
            logger.info("Loaded ring model from %s on %s", model_path, self.device)
        except Exception as exc:  # optional dependency/model must not break CPU fallback
            self.model_error = str(exc)
            self.model = None
            logger.warning("YOLO model unavailable; using classical fallback: %s", exc)

    def _detect_device(self) -> str:
        if SETTINGS.DEVICE != "auto":
            return SETTINGS.DEVICE
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _model_positions(self, image: np.ndarray, roi: ROI) -> tuple[list[float], float]:
        if self.model is None:
            return [], 0.0
        try:
            result = self.model.predict(
                source=image,
                imgsz=SETTINGS.MODEL_IMAGE_SIZE,
                conf=SETTINGS.MODEL_CONFIDENCE,
                iou=SETTINGS.MODEL_IOU,
                device=self.device,
                verbose=False,
            )[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return [], 0.0
            positions: list[float] = []
            confidences: list[float] = []
            for box, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                center_y = float((box[1] + box[3]) / 2.0)
                if roi.y <= center_y <= roi.y + roi.height:
                    positions.append(center_y - roi.y)
                    confidences.append(float(confidence))
            return sorted(positions), float(np.mean(confidences)) if confidences else 0.0
        except Exception as exc:
            logger.warning("YOLO inference failed; request continues with CV fallback: %s", exc)
            return [], 0.0

    def _classical_positions(self, gray: np.ndarray, roi: ROI) -> tuple[list[float], list[float], int]:
        region = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        strip_count = max(3, SETTINGS.N_STRIPS)
        strip_width = max(8, int(roi.width * SETTINGS.STRIP_WIDTH_FRAC))
        centers = np.linspace(
            int(roi.width * 0.08),
            int(roi.width * 0.92),
            strip_count,
            dtype=int,
        )
        candidates: list[tuple[float, int, float]] = []
        for strip_index, center in enumerate(centers):
            left = max(0, center - strip_width // 2)
            right = min(roi.width, left + strip_width)
            profile = row_edge_profile(region[:, left:right])
            minimum_distance = max(
                SETTINGS.MIN_RING_GAP_PX, int(round(roi.height * 0.006))
            )
            dynamic_threshold = max(
                float(np.percentile(profile, 68)),
                float(np.mean(profile) + SETTINGS.PEAK_PROMINENCE * np.std(profile)),
            )
            for position in _local_maxima(profile, minimum_distance, dynamic_threshold):
                candidates.append((float(position), strip_index, float(profile[position])))
        tolerance = max(3.0, roi.height * 0.012)
        return _cluster_candidates(
            candidates,
            tolerance=tolerance,
            minimum_agreement=SETTINGS.STRIP_AGREEMENT_MIN,
            strip_count=strip_count,
        )

    def _missing_gap_hypotheses(
        self,
        positions: list[float],
        profile: np.ndarray,
    ) -> tuple[list[float], int]:
        if len(positions) < 4:
            return positions, 0
        gaps = np.diff(positions)
        median_gap = float(np.median(gaps))
        if median_gap <= 0:
            return positions, 0
        additions: list[float] = []
        for start, gap in zip(positions[:-1], gaps):
            if gap < SETTINGS.MISSING_GAP_RATIO * median_gap:
                continue
            midpoint = int(round(start + gap / 2.0))
            window = profile[max(0, midpoint - 3) : min(len(profile), midpoint + 4)]
            evidence = float(np.mean(window)) if len(window) else 0.0
            if evidence >= SETTINGS.MISSING_EVIDENCE:
                additions.append(float(midpoint))
        merged = sorted(positions + additions)
        return merged, len(additions)

    def _annotate(
        self,
        image: np.ndarray,
        roi: ROI,
        positions: list[float],
        count: int,
        confidence_score: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        annotated = image.copy()
        cv2.rectangle(
            annotated,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            (0, 210, 150),
            2,
        )
        for number, position in enumerate(positions, start=1):
            y = roi.y + int(round(position))
            cv2.line(annotated, (roi.x, y), (roi.x + roi.width, y), (0, 255, 80), 2)
            if confidence_score >= SETTINGS.CONFIDENCE_MEDIUM:
                cv2.putText(
                    annotated,
                    str(number),
                    (max(5, roi.x - 32), max(16, y + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 80),
                    1,
                    cv2.LINE_AA,
                )
        label = f"{count} rings | {confidence_score:.0f}%"
        cv2.rectangle(annotated, (8, 8), (260, 48), (10, 20, 30), -1)
        cv2.putText(annotated, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 220), 2)

        pad = max(12, int(roi.width * 0.10))
        x0 = max(0, roi.x - pad)
        x1 = min(image.shape[1], roi.x + roi.width + pad)
        crop = image[roi.y : roi.y + roi.height, x0:x1].copy()
        for position in positions:
            cv2.line(crop, (0, int(round(position))), (crop.shape[1], int(round(position))), (0, 255, 80), 2)
        return annotated, crop

    def analyze(
        self,
        image: np.ndarray,
        top: float,
        bottom: float,
        x_fraction: float,
        *,
        auto_roi: bool = False,
    ) -> dict:
        started = time.perf_counter()
        image, scale = resize_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = build_roi(gray.shape, top, bottom, x_fraction)
        quality = quality_metrics(gray, roi.as_tuple())
        if quality.width < 240 or quality.height < 240:
            raise RingDetectionError("Image resolution is too low for reliable counting.")

        enhanced = enhanced_gray(gray)
        positions, agreements, rejected = self._classical_positions(enhanced, roi)
        region = enhanced[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        profile = row_edge_profile(region)
        positions, missing = self._missing_gap_hypotheses(positions, profile)
        model_positions, model_confidence = self._model_positions(image, roi)

        if model_positions and len(model_positions) >= 3:
            # The model's one-instance-per-ring result is preferred only when
            # it is materially supported by the image; this avoids raw YOLO
            # duplicate boxes becoming the displayed count.
            model_count = len(model_positions)
            cv_count = max(0, len(positions) - 1)
            if abs(model_count - cv_count) <= max(2, round(cv_count * 0.12)):
                count = model_count
                positions_for_draw = model_positions
            else:
                count = cv_count
                positions_for_draw = positions
        else:
            count = max(0, len(positions) - 1)
            positions_for_draw = positions

        spacing = spacing_statistics(positions_for_draw)
        spacing_score = consistency_from_gaps(positions_for_draw)
        agreement_score = float(np.mean(agreements)) if agreements else 0.0
        quality_score = _quality_score(quality)
        evidence_score = min(1.0, len(positions_for_draw) / 12.0)
        confidence_score = 100.0 * (
            0.28 * agreement_score
            + 0.24 * spacing_score
            + 0.23 * quality_score
            + 0.17 * evidence_score
            + (0.08 * model_confidence if model_positions else 0.08 * agreement_score)
        )
        confidence_score -= min(18.0, missing * 6.0)
        if auto_roi:
            confidence_score -= 12.0
        confidence_score = float(max(0.0, min(100.0, confidence_score)))
        if confidence_score >= SETTINGS.CONFIDENCE_HIGH:
            confidence = "High"
        elif confidence_score >= SETTINGS.CONFIDENCE_MEDIUM:
            confidence = "Medium"
        else:
            confidence = "Low"

        annotated, crop = self._annotate(
            image, roi, positions_for_draw, count, confidence_score
        )
        elapsed = round(time.perf_counter() - started, 3)
        result = {
            "success": True,
            "ring_count": int(count),
            "confidence": confidence,
            "confidence_score": round(confidence_score, 1),
            "consistency": round(float(spacing_score), 3),
            "sharpness": round(quality.sharpness, 1),
            "brightness": round(quality.brightness, 1),
            "contrast": round(quality.contrast, 1),
            "glare": round(quality.glare_percent, 2),
            "detected_rings": int(max(0, len(positions_for_draw) - 1)),
            "rejected_detections": int(rejected),
            "possible_missing_rings": int(missing),
            "processing_time": elapsed,
            "median_gap": spacing["median_gap"],
            "mean_gap": spacing["mean_gap"],
            "gap_std": spacing["std_gap"],
            "coefficient_variation": spacing["coefficient_variation"],
            "roi": {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height},
            "quality_warning": not quality.usable,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "image_data_url": encode_jpeg(annotated),
            "crop_data_url": encode_jpeg(crop),
        }
        logger.info(
            "count=%s confidence=%s score=%.1f candidates=%s rejected=%s missing=%s time=%.3fs",
            count,
            confidence,
            confidence_score,
            len(positions_for_draw),
            rejected,
            missing,
            elapsed,
        )
        return result

    def analyze_bytes(
        self, data: bytes, top: float, bottom: float, x_fraction: float, *, auto_roi: bool = False
    ) -> dict:
        return self.analyze(
            decode_image(data), top, bottom, x_fraction, auto_roi=auto_roi
        )