"""Train a YOLOv8 detector for in-cabin safety objects.

Requires a labelled dataset in YOLO format (see ``training/data.yaml``) and,
realistically, a GPU. After training, copy the best weights into the project and
run the monitor with them::

    python training/train.py --data training/data.yaml --epochs 100
    python main.py            # after pointing object_model_path at best.pt

The resulting ``runs/detect/train/weights/best.pt`` can be passed to
``ObjectDetector`` (set ``Config.object_model_path``).
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 for DMS objects")
    parser.add_argument("--data", default="training/data.yaml", help="dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="base weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    return parser.parse_args()


def main() -> None:
    """Run YOLOv8 training with the given arguments."""
    from ultralytics import YOLO  # imported lazily so --help works without it

    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
