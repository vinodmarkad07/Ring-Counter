"""Automatic horizontal localization of a ring stack in a photo.

Replaces the previous behaviour (app.py hardcoding x_fraction=0.5 with a
fixed-width ROI covering 72% of the image) for requests that don't supply
a manual tap. That fixed-center assumption only worked when the target
stack happened to sit near the middle of the frame; on any photo with
multiple stacks, or the target off-center, the wide fixed box spilled
into neighbouring stacks and background -- confirmed against real photos
where the box visibly spanned two separate stacks.

This scans candidate vertical strips across the full image width and
scores each by how strong AND how regularly-spaced its ring-boundary
signal is (real ring stacks produce many evenly-spaced sharp edges;
background, other equipment, or a second out-of-focus stack usually
doesn't score as well). Ties are broken toward the horizontal center of
the frame, since product photos are usually shot with the intended stack
roughly centered -- but unlike the previous code, this is a tiebreaker
on real evidence, not an assumption used blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import row_edge_profile


@dataclass(frozen=True)
class StackLocation:
    x_fraction: float
    width_fraction: float
    reliable: bool
    n_peaks: int
    periodicity: float


def _local_maxima_count(signal: np.ndarray, min_distance: int, threshold: float) -> list[int]:
    if len(signal) < 5:
        return []
    raw = [
        i for i in range(1, len(signal) - 1)
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1] and signal[i] >= threshold
    ]
    raw.sort(key=lambda i: float(signal[i]), reverse=True)
    selected: list[int] = []
    for i in raw:
        if all(abs(i - other) >= min_distance for other in selected):
            selected.append(i)
    return sorted(selected)


def _periodicity(peaks: list[int]) -> float:
    if len(peaks) < 3:
        return 0.0
    gaps = np.diff(peaks)
    median = float(np.median(gaps))
    if median <= 0:
        return 0.0
    std = float(np.std(gaps))
    return max(0.0, 1.0 - min(1.0, std / median))


def locate_stack_column(
    gray: np.ndarray,
    *,
    col_width_frac: float = 0.14,
    step_frac: float = 0.02,
    margin_frac: float = 0.03,
    center_bias: float = 0.35,
    min_peaks: int = 6,
    min_periodicity: float = 0.30,
) -> StackLocation:
    height, width = gray.shape[:2]
    y0 = int(height * margin_frac)
    y1 = height - y0
    col_w = max(12, int(width * col_width_frac))
    step = max(4, int(width * step_frac))
    frame_center = width / 2.0

    best: tuple[float, int, int, int, float] | None = None  # score, x0, x1, npeaks, periodicity
    for x0 in range(0, max(1, width - col_w), step):
        x1 = x0 + col_w
        region = gray[y0:y1, x0:x1]
        profile = row_edge_profile(region)
        rng = float(profile.max() - profile.min()) if len(profile) else 0.0
        if rng < 1e-6:
            continue
        threshold = float(np.mean(profile) + 0.6 * np.std(profile))
        min_distance = max(2, int((y1 - y0) * 0.008))
        peaks = _local_maxima_count(profile, min_distance, threshold)
        periodicity = _periodicity(peaks)

        col_center = (x0 + x1) / 2.0
        center_dist_frac = abs(col_center - frame_center) / (width / 2.0)
        center_weight = 1.0 - center_bias * min(1.0, center_dist_frac)
        score = len(peaks) * periodicity * center_weight

        if best is None or score > best[0]:
            best = (score, x0, x1, len(peaks), periodicity)

    if best is None:
        return StackLocation(0.5, 0.72, False, 0, 0.0)

    _, x0, x1, n_peaks, periodicity = best
    reliable = n_peaks >= min_peaks and periodicity >= min_periodicity

    # Widen a bit from the winning strip toward a sensible stack-width
    # box, but only ever as wide as necessary -- never the old fixed 72%
    # of the whole frame, which is what caused multi-stack spillover.
    center = (x0 + x1) / 2.0
    fitted_width = min(width * 0.55, col_w * 2.6)
    x_fraction = center / width
    width_fraction = fitted_width / width

    return StackLocation(
        x_fraction=float(x_fraction),
        width_fraction=float(width_fraction),
        reliable=reliable,
        n_peaks=n_peaks,
        periodicity=float(periodicity),
    )
