"""Tight object boxes from a local detector.

The language model can name a scene, but its boxes land on a region of the
photo instead of the object. This detector returns one box per object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8s-worldv2.onnx"
INPUT_SIZE = 640

# Prompts baked into the YOLO-World ONNX file. Order must match the export.
# Color is read from the box later, so these names stay generic.
CLASSES = (
    "person",
    "bottle",
    "water bottle",
    "wine glass",
    "cup",
    "mug",
    "bowl",
    "jar",
    "can",
    "box",
    "plate",
    "fork",
    "knife",
    "spoon",
    "banana",
    "apple",
    "orange",
    "mouse",
    "remote",
    "cell phone",
    "book",
    "pen",
    "pencil",
    "marker",
    "scissors",
    "keys",
    "glasses",
    "watch",
    "wallet",
    "card",
    "tissue",
    "lid",
    "cap",
    "vase",
    "clock",
    "toothbrush",
    "handbag",
    "backpack",
    "laptop",
    "keyboard",
    "tv",
    "potted plant",
    "chair",
    "couch",
    "sink",
)

# The table is the surface under the objects, not something to pick.
SKIP = {"dining table"}

DISPLAY = {
    "tv": "Monitor",
    "cell phone": "Phone",
    "handbag": "Bag",
    "potted plant": "Plant",
    "wine glass": "Glass",
}

_session = None
_lock = Lock()


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: list[int]
    confidence: float


def detect_objects(image: Image.Image, conf: float = 0.2, iou: float = 0.45) -> list[Detection]:
    session = _model()
    canvas, scale, pad_x, pad_y = _letterbox(image, INPUT_SIZE)
    blob = np.asarray(canvas, dtype=np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None]
    name = session.get_inputs()[0].name
    output = session.run(None, {name: blob})[0]
    return _decode(output, scale, pad_x, pad_y, image.width, image.height, conf, iou)


def _model():
    global _session
    with _lock:
        if _session is None:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Object detector model is missing: {MODEL_PATH.name}")
            import onnxruntime as ort

            _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        return _session


def _letterbox(image: Image.Image, size: int) -> tuple[Image.Image, float, float, float]:
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))), Image.Resampling.BILINEAR)
    pad_x = (size - resized.width) / 2
    pad_y = (size - resized.height) / 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, scale, pad_x, pad_y


def _decode(output, scale: float, pad_x: float, pad_y: float, width: int, height: int, conf: float, iou: float) -> list[Detection]:
    pred = np.squeeze(np.asarray(output))
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    boxes = pred[:, :4]
    scores = pred[:, 4:]
    if scores.shape[1] != len(CLASSES):
        return []
    class_ids = scores.argmax(axis=1)
    class_scores = scores[np.arange(scores.shape[0]), class_ids]
    keep = class_scores >= conf
    boxes = boxes[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]
    if len(boxes) == 0:
        return []
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, max(0, width - 1))
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, max(0, height - 1))
    order = _nms(xyxy, class_scores, class_ids, iou)
    found: list[Detection] = []
    area_limit = width * height * 0.7
    for index in order:
        label = CLASSES[int(class_ids[index])]
        if label in SKIP:
            continue
        x1, y1, x2, y2 = xyxy[index]
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        if w < 8 or h < 8 or w * h > area_limit:
            continue
        found.append(
            Detection(
                label=DISPLAY.get(label, label.title()),
                bbox=[int(round(x1)), int(round(y1)), w, h],
                confidence=float(class_scores[index]),
            )
        )
    found.sort(key=lambda item: item.confidence, reverse=True)
    return found


def _nms(boxes, scores, class_ids, iou: float) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        same = class_ids[rest] == class_ids[current]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_c = max(0, boxes[current, 2] - boxes[current, 0]) * max(0, boxes[current, 3] - boxes[current, 1])
        area_r = np.maximum(0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0, boxes[rest, 3] - boxes[rest, 1])
        overlap = inter / (area_c + area_r - inter + 1e-6)
        drop = same & (overlap > iou)
        order = rest[~drop]
    return kept
