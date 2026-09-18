"""Evaluate exact ring counts from a labelled CSV manifest.

Manifest columns:
    image,actual,top_frac,bottom_frac,x_frac

The ROI columns are optional only when auto_roi=1 is passed.  They should be
recorded from the same workflow used by the operator; they are not inferred
from the ground-truth count.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from core.detector import RingCounter
from core.metrics import EvaluationRow, evaluate_rows, write_evaluation_csv
from core.preprocessing import decode_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RingCount AI on held-out images.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default="results/evaluation_results.csv", type=Path)
    parser.add_argument("--summary", default="results/evaluation_summary.json", type=Path)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--fail-below-target", action="store_true")
    args = parser.parse_args()

    counter = RingCounter()
    rows: list[EvaluationRow] = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            image_path = Path(record["image"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            actual = int(record["actual"])
            auto = record.get("auto_roi", "0").lower() in {"1", "true", "yes"}
            top = float(record.get("top_frac") or (0.04 if auto else ""))
            bottom = float(record.get("bottom_frac") or (0.96 if auto else ""))
            x_fraction = float(record.get("x_frac") or 0.5)
            result = counter.analyze_bytes(
                image_path.read_bytes(), top, bottom, x_fraction, auto_roi=auto
            )
            predicted = int(result["ring_count"])
            rows.append(
                EvaluationRow(
                    image=str(image_path),
                    actual=actual,
                    predicted=predicted,
                    error=predicted - actual,
                    absolute_error=abs(predicted - actual),
                    confidence=result["confidence"],
                    confidence_score=float(result["confidence_score"]),
                )
            )

    summary = evaluate_rows(rows)
    write_evaluation_csv(args.output, rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_below_target and (
        summary["exact_accuracy"] is None or summary["exact_accuracy"] < args.target
    ):
        print(f"FAILED: exact accuracy is below target {args.target:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())