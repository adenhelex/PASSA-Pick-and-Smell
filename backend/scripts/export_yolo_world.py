"""Bake the tabletop class list into a YOLO-World ONNX file.

Install the exporter once, from backend/:

    .venv/bin/pip install ultralytics

Then:

    .venv/bin/python scripts/export_yolo_world.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.detect import CLASSES, MODEL_PATH  # noqa: E402


def main() -> None:
    from ultralytics import YOLO

    model = YOLO("yolov8s-worldv2.pt")
    model.set_classes(list(CLASSES))
    try:
        exported = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=True))
    except Exception:
        exported = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=False))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), MODEL_PATH)
    print(f"Wrote {MODEL_PATH} with {len(CLASSES)} classes.")


if __name__ == "__main__":
    main()
