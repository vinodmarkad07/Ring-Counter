"""Train an optional YOLO ring detector.

This script never runs during normal web requests.  It requires the optional
Ultralytics package and a real annotated dataset in YOLO detection or
segmentation format.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO ring detector.")
    parser.add_argument("--model", default=os.getenv("TRAIN_MODEL", "yolo11n.pt"))
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "150")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("IMAGE_SIZE", "1024")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "4")))
    parser.add_argument("--device", default=os.getenv("DEVICE", "auto"))
    parser.add_argument("--project", default=os.getenv("PROJECT", "runs/ringcount"))
    parser.add_argument("--name", default=os.getenv("NAME", "ring-detector"))
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"Dataset file not found: {data_path}")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        parser.error(
            "Ultralytics is not installed. Install optional ML dependencies with "
            "pip install -r requirements-ml.txt"
        )
    device = None if args.device == "auto" else args.device
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=max(1, args.epochs),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=30,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())