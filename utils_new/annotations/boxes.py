"""Shared bounding-box geometry for ensemble suppression and fusion. No Qt/ML."""

import numpy as np


def yolo_to_xyxy(cx, cy, w, h):
    """
    Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
    Возвращает координаты также в диапазоне 0..1.
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.clip([x1, y1, x2, y2], 0, 1)


def xyxy_to_yolo(x1, y1, x2, y2):
    """
    Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
    """
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


def iou_xyxy(box1, box2):
    """
    Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
    Возвращает значение в диапазоне [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union
