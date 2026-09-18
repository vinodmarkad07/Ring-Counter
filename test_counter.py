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
    assert result["image_data_url"].startswith("data:image/jpeg;base64,")


def test_status_and_json_error():
    client = app.test_client()
    status = client.get("/status")
    assert status.status_code == 200
    assert status.get_json()["success"] is True
    error = client.post("/count")
    assert error.status_code == 400
    assert error.get_json()["success"] is False