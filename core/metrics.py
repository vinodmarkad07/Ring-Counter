"""Evaluation metrics with no hidden or image-specific correction rules."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class EvaluationRow:
    image: str
    actual: int
    predicted: int
    error: int
    absolute_error: int
    confidence: str
    confidence_score: float


def evaluate_rows(rows: Iterable[EvaluationRow]) -> dict:
    rows = list(rows)
    total = len(rows)
    if not total:
        return {
            "test_images": 0,
            "exact_accuracy": None,
            "mae": None,
            "mean_error": None,
            "overcount_rate": None,
            "undercount_rate": None,
        }
    exact = sum(row.error == 0 for row in rows)
    return {
        "test_images": total,
        "exact_accuracy": round(exact / total, 4),
        "mae": round(sum(row.absolute_error for row in rows) / total, 4),
        "mean_error": round(sum(row.error for row in rows) / total, 4),
        "overcount_rate": round(sum(row.error > 0 for row in rows) / total, 4),
        "undercount_rate": round(sum(row.error < 0 for row in rows) / total, 4),
        "average_confidence_score": round(
            sum(row.confidence_score for row in rows) / total, 4
        ),
    }


def write_evaluation_csv(path: Path, rows: Iterable[EvaluationRow]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else [
            "image", "actual", "predicted", "error", "absolute_error",
            "confidence", "confidence_score"
        ])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)