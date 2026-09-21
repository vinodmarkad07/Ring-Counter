"""ROI and candidate validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SETTINGS


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def validate_fractions(top: float, bottom: float, x: float) -> None:
    values = (top, bottom, x)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("ROI coordinates must be finite numbers.")
    if not (0.0 <= top < bottom <= 1.0):
        raise ValueError("Tap the top of the stack first and the bottom below it.")
    if not (0.0 <= x <= 1.0):
        raise ValueError("The horizontal tap must stay inside the image.")
    if bottom - top < SETTINGS.MIN_ROI_HEIGHT_FRAC:
        raise ValueError("The selected stack region is too short. Retap the top and bottom.")


def build_roi(
    image_shape: tuple[int, ...],
    top: float,
    bottom: float,
    x_fraction: float,
    width_fraction: float | None = None,
) -> ROI:
    height, width = image_shape[:2]
    validate_fractions(top, bottom, x_fraction)
    y = max(0, min(height - 1, int(round(top * height))))
    bottom_y = max(y + 1, min(height, int(round(bottom * height))))
    roi_height = bottom_y - y
    frac = width_fraction if width_fraction is not None else SETTINGS.AXIS_WIDTH_FRAC
    roi_width = max(40, int(round(width * frac)))
    roi_width = min(width, roi_width)
    center_x = int(round(x_fraction * max(0, width - 1)))
    x = max(0, min(width - roi_width, center_x - roi_width // 2))
    return ROI(x=x, y=y, width=roi_width, height=roi_height)


def spacing_statistics(positions: list[float] | np.ndarray) -> dict[str, float]:
    values = np.asarray(positions, dtype=float)
    gaps = np.diff(values) if len(values) > 1 else np.asarray([], dtype=float)
    if not len(gaps):
        return {
            "median_gap": 0.0,
            "mean_gap": 0.0,
            "std_gap": 0.0,
            "coefficient_variation": 1.0,
        }
    median = float(np.median(gaps))
    mean = float(np.mean(gaps))
    std = float(np.std(gaps))
    return {
        "median_gap": round(median, 3),
        "mean_gap": round(mean, 3),
        "std_gap": round(std, 3),
        "coefficient_variation": round(std / max(mean, 1e-6), 4),
    }


def consistency_from_gaps(positions: list[float] | np.ndarray) -> float:
    stats = spacing_statistics(positions)
    return float(max(0.0, min(1.0, 1.0 - stats["coefficient_variation"])))