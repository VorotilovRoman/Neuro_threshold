"""Adapters at external boundaries. Existing normalized YOLO tuple format retained."""
from __future__ import annotations

import numpy as np

from .model import Annotation, ImageSize


def from_legacy(value, size: ImageSize, *, allow_classless=False) -> Annotation:
    """Legacy tuple -> pixel Annotation; strict by default for dataset export.

    detect: ('detect', cls, cx, cy, w, h)
    obb/segment: ('obb'/'segment', cls, [x1,y1,...])
    For display only, allow_classless admits ('detect',cx,cy,w,h) or (typ,flat).
    """
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError('annotation is not a nonempty tuple/list')
    kind = value[0]
    if kind not in ('detect', 'obb', 'segment'):
        raise ValueError(f'Unknown kind: {kind!r}')
    expected = 6 if kind == 'detect' else 3
    if len(value) == expected:
        cls = value[1]
        coords = value[2:] if kind == 'detect' else value[2]
    elif allow_classless and len(value) == expected - 1:
        cls = 0
        coords = value[1:] if kind == 'detect' else value[1]
    else:
        raise ValueError(f'{kind} expects {expected} tuple elements (classless requires opt-in)')
    if isinstance(cls, float) and not cls.is_integer():
        raise ValueError('class must be an integer')
    try:
        class_id = int(cls)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid class ID') from exc
    if class_id < 0:
        raise ValueError('Invalid negative class ID')
    if kind == 'detect':
        cx, cy, bw, bh = np.asarray(coords, dtype=np.float64).tolist()
        if bw <= 0 or bh <= 0:
            raise ValueError('Detect width/height must be > 0')
        points = np.array([[cx-bw/2, cy-bh/2], [cx+bw/2, cy-bh/2],
                           [cx+bw/2, cy+bh/2], [cx-bw/2, cy+bh/2]], dtype=np.float64)
    else:
        flat = np.asarray(coords, dtype=np.float64).reshape(-1)
        if len(flat) % 2:
            raise ValueError('Polygon must have an even number of coordinates')
        points = flat.reshape(-1, 2)
    points = points.copy()
    points[:, 0] *= size.width
    points[:, 1] *= size.height
    return Annotation(kind, class_id, points)


def to_legacy(ann: Annotation, size: ImageSize):
    """Canonical Annotation -> original normalized tuple; does not round/clamp."""
    p = ann.points_px
    if ann.kind == 'detect':
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        return ('detect', ann.class_id, (x0+x1)/(2*size.width),
                (y0+y1)/(2*size.height), (x1-x0)/size.width,
                (y1-y0)/size.height)
    q = p.copy()
    q[:, 0] /= size.width
    q[:, 1] /= size.height
    return (ann.kind, ann.class_id, q.flatten().tolist())


def from_legacy_many(values, size, *, allow_classless=False):
    return [from_legacy(v, size, allow_classless=allow_classless) for v in values]


def to_legacy_many(values, size):
    return [to_legacy(v, size) for v in values]


def from_detection_dict(detection: dict, size: ImageSize) -> Annotation:
    """YOLO+AE detection dictionary (pixel xyxy, corners or mask_xy) -> Annotation.

    Ignores UI-only confidence/name fields. No torch/Ultralytics import.
    """
    if not isinstance(detection, dict):
        raise TypeError('detection must be a dictionary')
    kind = str(detection.get('kind', 'detect')).lower()
    cls = detection.get('cls_id', 0)
    if kind == 'detect':
        x1, y1, x2, y2 = map(float, detection['xyxy'])
        return Annotation('detect', int(cls), [[x1, y1], [x2, y1],
                                               [x2, y2], [x1, y2]])
    if kind in ('segment', 'seg'):
        pts = detection.get('mask_xy')
        if pts is None:
            pts = detection.get('corners')
        return Annotation('segment', int(cls), np.asarray(pts).reshape(-1, 2))
    if kind == 'obb':
        return Annotation('obb', int(cls), np.asarray(detection['corners']).reshape(-1, 2))
    raise ValueError(f'unsupported detection kind {kind!r}')


def legacy_convert_annotations(values, size: ImageSize, target: str, *,
                               allow_lossy=False, obb_policy='refit'):
    """Explicit bridge for the old converter when migrating DatasetGenerator.

    Does not silently return Detect for an OBB conversion failure. Callers must
    explicitly allow lossy conversions and provide source image dimensions.
    """
    from .geometry import convert_annotation
    return [to_legacy(convert_annotation(from_legacy(v, size), target,
                                          allow_lossy=allow_lossy, obb_policy=obb_policy), size)
            for v in values]


def to_detection_dict(ann: Annotation, *, class_name=None, confidence=None):
    """Canonical annotation -> YOLO+AE/VAE detection dictionary in pixels."""
    import cv2

    from .geometry import bounds
    x0, y0, x1, y1 = bounds(ann.points_px)
    if ann.kind == 'segment':
        rect = cv2.minAreaRect(np.asarray(ann.points_px, dtype=np.float32))
        corners = cv2.boxPoints(rect).astype(float).tolist()
    else:
        corners = ann.points_px.tolist()
    edges = np.asarray(corners, dtype=np.float64)
    vectors = np.roll(edges, -1, axis=0) - edges
    longest = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
    angle = float(np.degrees(np.arctan2(longest[1], longest[0])))
    result = {
        'kind': 'seg' if ann.kind == 'segment' else ann.kind,
        'cls_id': ann.class_id,
        'cls_name': str(ann.class_id) if class_name is None else str(class_name),
        'xyxy': [x0, y0, x1, y1],
        'corners': corners,
        'center': ((x0+x1)/2, (y0+y1)/2),
        'angle_deg': angle,
    }
    if confidence is not None:
        result['conf'] = float(confidence)
    if ann.kind == 'segment':
        result['mask_xy'] = ann.points_px.tolist()
    return result


def from_ultralytics_result(result, size: ImageSize):
    """Duck-typed Ultralytics result -> canonical annotations; no torch import.

    Uses OBB first, then segmentation polygons, then axis-aligned boxes.
    Returns annotations sorted by descending confidence, matching the viewer.
    """
    def arr(value):
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return np.asarray(value)

    if result is None:
        return []
    candidates = []
    obb = getattr(result, 'obb', None)
    if obb is not None and getattr(obb, 'xyxyxyxy', None) is not None and len(obb):
        for pts, cls, confidence in zip(arr(obb.xyxyxyxy), arr(obb.cls), arr(obb.conf)):
            candidates.append((float(confidence), Annotation('obb', int(cls), np.asarray(pts).reshape(-1, 2))))
    elif getattr(getattr(result, 'masks', None), 'xy', None) is not None and len(result.masks.xy):
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return []
        for pts, cls, confidence in zip(result.masks.xy, arr(boxes.cls), arr(boxes.conf)):
            pts = arr(pts).reshape(-1, 2)
            if len(pts) >= 3:
                candidates.append((float(confidence), Annotation('segment', int(cls), pts)))
    else:
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return []
        for xyxy, cls, confidence in zip(arr(boxes.xyxy), arr(boxes.cls), arr(boxes.conf)):
            x0,y0,x1,y1 = map(float, xyxy)
            if x1 <= x0 or y1 <= y0:
                continue
            candidates.append((float(confidence), Annotation('detect', int(cls),
                [[x0,y0],[x1,y0],[x1,y1],[x0,y1]])))
    return [ann for confidence, ann in sorted(candidates, key=lambda x: -x[0])]
