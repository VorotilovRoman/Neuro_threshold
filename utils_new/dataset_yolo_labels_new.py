# -*- coding: utf-8 -*-
"""YOLO TXT decode/encode at the boundary of the pixel-space annotation core."""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
from utils_new.annotations import Annotation, ImageSize, from_legacy, to_legacy
LABEL_HINTS = {'auto','mixed','detect','obb_corners','obb_xywhr','segment'}

def _xywhr_points(vals):
    cx, cy, w, h, radians = map(float, vals)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    v1 = np.array((w*cos_a/2, w*sin_a/2), dtype=float)
    v2 = np.array((-h*sin_a/2, h*cos_a/2), dtype=float)
    c = np.array((cx, cy), dtype=float)
    return np.array([c+v1+v2, c+v1-v2, c-v1-v2, c-v1+v2]).flatten().tolist()



def read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None):
    """Read old YOLO txt variants into canonical pixel-space annotations.

    9 tokens are OBB in auto mode. Explicit 'segment' disambiguates a 4-point
    polygon. Invalid rows are logged and skipped as in legacy image_io.
    """
    if format_hint not in LABEL_HINTS:
        raise ValueError(f'Unsupported YOLO input format {format_hint!r}')
    if path is None or not Path(path).exists():
        return []
    result = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            parts = line.split()
            cls = int(parts[0])
            numbers = list(map(float, parts[1:]))
            count = len(parts)
            if format_hint == 'detect':
                kind = 'detect' if count == 5 else None
            elif format_hint == 'obb_corners':
                kind = 'obb' if count == 9 else None
            elif format_hint == 'obb_xywhr':
                kind = 'obb_xywhr' if count == 6 else None
            elif format_hint == 'segment':
                kind = 'segment' if count >= 7 and (count-1)%2 == 0 else None
            else:
                kind = ('detect' if count == 5 else 'obb_xywhr' if count == 6
                        else 'obb' if count == 9 else 'segment' if count >= 7 and (count-1)%2 == 0
                        else None)
            if kind is None:
                raise ValueError(f'{count} tokens incompatible with format_hint={format_hint}')
            if kind == 'obb_xywhr':
                kind, numbers = 'obb', _xywhr_points(numbers)
            value = ((kind, cls, *numbers) if kind == 'detect' else (kind, cls, numbers))
            result.append(from_legacy(value, size))
        except (TypeError, ValueError, IndexError) as exc:
            if log:
                log(f'[labels] {path}:{line_no}: {exc}; запись пропущена')
    return result



def _normalized_xywhr(ann, size):
    """Legacy-compatible xywhr: angle in radians of normalized corner edges."""
    normalized = np.asarray(to_legacy(ann, size)[2], dtype=np.float64).reshape(4, 2)
    center = normalized.mean(axis=0)
    edges = np.roll(normalized, -1, axis=0) - normalized
    lengths = np.linalg.norm(edges, axis=1)
    idx = int(np.argmax(lengths))
    width = float(lengths[idx])
    angle = math.atan2(float(edges[idx, 1]), float(edges[idx, 0]))
    height = (float(lengths[(idx + 1) % 4]) + float(lengths[(idx + 3) % 4])) / 2
    if height > width:
        width, height = height, width
        angle += math.pi / 2
    angle = angle % math.pi
    if angle >= 3*math.pi/4:
        angle -= math.pi
    return [float(center[0]), float(center[1]), width, height, angle]



def write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners'):
    if obb_format not in ('corners', 'xywhr'):
        raise ValueError('obb_format must be corners/xywhr')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ann in annotations:
        if not isinstance(ann, Annotation):
            raise TypeError('Expected canonical Annotation objects')
        legacy = to_legacy(ann, size)
        kind = legacy[0]
        if kind == 'detect':
            coords = legacy[2:]
        elif kind == 'obb' and obb_format == 'xywhr':
            coords = _normalized_xywhr(ann, size)
        else:
            coords = legacy[2]
        # Four-point Segment has 9 tokens (class+8), ambiguous with OBB.
        # Insert a collinear midpoint to preserve the exact polygon shape and
        # make AUTO / mixed readers identify Segment unambiguously.
        if kind == 'segment' and len(coords) == 8:
            coords = list(coords)
            midpoint = [(float(coords[0])+float(coords[2]))/2,
                        (float(coords[1])+float(coords[3]))/2]
            coords = coords[:2] + midpoint + coords[2:]
        rows.append(f'{ann.class_id} ' + ' '.join(f'{float(v):.6f}' for v in coords))
    path.write_text(('\n'.join(rows) + '\n') if rows else '', encoding='utf-8')

