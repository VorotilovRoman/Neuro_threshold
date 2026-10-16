# utils/annotation_converter.py
"""
Конвертация canonical-аннотаций между типами:
    detect ↔ obb ↔ segment

Используется при экспорте YOLO-датасета в выбранный subtype
(например, всё в detect или всё в segment).
"""
from import_libs_external import *
import numpy as np

from utils.image_io import canonicalize_obb_corners


SUBTYPE_KEEP    = 'keep'
SUBTYPE_DETECT  = 'detect'
SUBTYPE_OBB     = 'obb'
SUBTYPE_SEGMENT = 'segment'


def to_detect(ann):
    """Любой тип → detect (axis-aligned bbox)."""
    if ann[0] == 'detect':
        return ann
    if ann[0] == 'obb':
        _, cls, corners = ann
        xs = corners[0::2]
        ys = corners[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return ('detect', cls, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                x2 - x1, y2 - y1)
    if ann[0] == 'segment':
        _, cls, pts = ann
        xs = pts[0::2]
        ys = pts[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        return ('detect', cls, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                x2 - x1, y2 - y1)
    return ann


def to_obb(ann):
    """Любой тип → OBB (4 канонических угла)."""
    if ann[0] == 'obb':
        return ann
    if ann[0] == 'detect':
        _, cls, cx, cy, w, h = ann
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        # Порядок углов: BR, BL, TL, TR (совпадает с canonicalize)
        corners = [x2, y2, x1, y2, x1, y1, x2, y1]
        corners = canonicalize_obb_corners(corners)
        return ('obb', cls, corners)
    if ann[0] == 'segment':
        _, cls, pts = ann
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
        if len(pts_arr) < 3:
            return ann
        try:
            rect = cv2.minAreaRect(pts_arr)
            corners = cv2.boxPoints(rect).flatten().tolist()
            corners = [max(0.0, min(1.0, float(p))) for p in corners]
            corners = canonicalize_obb_corners(corners)
            return ('obb', cls, corners)
        except Exception:
            return to_detect(ann)
    return ann


def to_segment(ann):
    """Любой тип → segment (полигон)."""
    if ann[0] == 'segment':
        return ann
    if ann[0] == 'detect':
        _, cls, cx, cy, w, h = ann
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
        return ('segment', cls, [x1, y1, x2, y1, x2, y2, x1, y2])
    if ann[0] == 'obb':
        _, cls, corners = ann
        return ('segment', cls, list(corners))
    return ann


_CONVERTERS = {
    SUBTYPE_KEEP:    lambda a: a,
    SUBTYPE_DETECT:  to_detect,
    SUBTYPE_OBB:     to_obb,
    SUBTYPE_SEGMENT: to_segment,
}


def convert_annotations(annotations, subtype):
    """
    Применяет конвертер subtype ко всем аннотациям.
    subtype: 'keep' | 'detect' | 'obb' | 'segment'
    """
    fn = _CONVERTERS.get(subtype)
    if fn is None:
        return list(annotations)
    return [fn(a) for a in annotations]