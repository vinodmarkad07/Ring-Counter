"""Regression and API smoke tests.

The synthetic test verifies the pipeline mechanics without pretending that it
is a production accuracy test.  Real accuracy belongs in evaluate.py with
held-out, manually verified images.
"""

from __future__ import annotations

import cv2
import numpy as np

from app import app
from core.detector import RingCounter
from core.validation import build_roi


def synthetic_stack(rings: int = 12) -> bytes:
    image = np.full((700, 500, 3), 185, dtype=np.uint8)
    top, gap = 80, 42
    for index in range(rings + 1):
        y = top + index * gap
        cv2.line(image, (70, y), (430, y), (25, 25, 25), 4)
        cv2.line(image, (70, y + 9), (430, y + 9), (225, 225, 225), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_roi_validation():
    roi = build_roi((700, 500), 0.1, 0.9, 0.5)
    assert roi.height > 0
    assert 0 <= roi.x < 500


def test_synthetic_counter_returns_a_result():
    result = RingCounter().analyze_bytes(synthetic_stack(), 0.1, 0.85, 0.5)
    assert result["success"] is True
    assert isinstance(result["ring_count"], int)
    assert 0 <= result["confidence_score"] <= 100


def synthetic_two_stacks() -> bytes:
    """One stack off-center-left, another off-center-right, background
    between them. Regression test for the bug where auto_roi assumed the
    stack was always horizontally centered (x_fraction=0.5, fixed 72%
    width) -- which spilled into both stacks and the background gap on
    any photo like this."""
    image = np.full((700, 900, 3), 90, dtype=np.uint8)
    # left stack: x 60-260
    top, gap = 80, 42
    for index in range(13):
        y = top + index * gap
        cv2.line(image, (60, y), (260, y), (25, 25, 25), 4)
        cv2.line(image, (60, y + 9), (260, y + 9), (200, 200, 200), 2)
    # right stack: x 640-840 (clearly separate, own periodic pattern)
    for index in range(9):
        y = top + index * gap
        cv2.line(image, (640, y), (840, y), (25, 25, 25), 4)
        cv2.line(image, (640, y + 9), (840, y + 9), (200, 200, 200), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_auto_roi_does_not_span_both_stacks():
    """The auto-located ROI must land inside ONE stack's x-range, never
    spanning the gap between two separate stacks in frame."""
    result = RingCounter().analyze_bytes(
        synthetic_two_stacks(), 0.04, 0.96, 0.5, auto_roi=True
    )
    if not result["success"]:
        # Refusing on an ambiguous synthetic image is an acceptable
        # outcome (better than guessing) -- but silently spanning both
        # stacks is not, so only check the geometry when it DID return
        # a result.
        return
    roi = result["roi"]
    left_stack_range = (60, 260)
    right_stack_range = (640, 840)
    roi_left, roi_right = roi["x"], roi["x"] + roi["width"]
    spans_left = roi_left < left_stack_range[1] and roi_right > left_stack_range[0]
    spans_right = roi_left < right_stack_range[1] and roi_right > right_stack_range[0]
    assert not (spans_left and spans_right), (
        f"ROI {roi} spans both synthetic stacks -- the old "
        f"fixed-center-72%-width bug has regressed."
    )
    assert result["image_data_url"].startswith("data:image/jpeg;base64,")


def test_status_and_json_error():
    client = app.test_client()
    status = client.get("/status")
    assert status.status_code == 200
    assert status.get_json()["success"] is True
    error = client.post("/count")
    assert error.status_code == 400
    assert error.get_json()["success"] is False