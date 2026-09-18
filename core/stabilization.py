"""Consensus helpers for repeated image measurements and future video frames."""

from __future__ import annotations

from collections import Counter, deque


class CountStabilizer:
    def __init__(self, window: int = 7, minimum_stable_frames: int = 5) -> None:
        self.values: deque[tuple[int, float]] = deque(maxlen=max(1, window))
        self.minimum_stable_frames = max(1, minimum_stable_frames)
        self.stable_count: int | None = None

    def update(self, count: int, confidence_score: float) -> int | None:
        self.values.append((int(count), float(confidence_score)))
        counts = Counter(value for value, _ in self.values)
        value, frequency = counts.most_common(1)[0]
        if frequency >= min(self.minimum_stable_frames, len(self.values)):
            self.stable_count = value
        return self.stable_count


def robust_consensus(measurements: list[tuple[int, float]]) -> tuple[int, float]:
    """Choose the confidence-weighted mode, falling back to a median."""
    if not measurements:
        return 0, 0.0
    scores: dict[int, float] = {}
    for count, confidence in measurements:
        scores[count] = scores.get(count, 0.0) + max(0.0, float(confidence))
    best_count = max(scores, key=scores.get)
    total = sum(scores.values())
    agreement = scores[best_count] / total if total else 0.0
    if agreement < 0.34:
        ordered = sorted(count for count, _ in measurements)
        best_count = ordered[len(ordered) // 2]
    return int(best_count), round(float(agreement), 4)