"""Optional NumPy/OpenCV rendering for exports; never replaces Qt overlays."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import bounds
from .masks import annotation_binary_mask
from .model import Annotation, ImageSize, MaskEncoding


def _bgr_color(color_for_class, cls):
    value = tuple(int(x) for x in color_for_class(int(cls)))
    if len(value) != 3 or any(x < 0 or x > 255 for x in value):
        raise ValueError('color_for_class must return 3 BGR values in [0,255]')
    return np.asarray(value, dtype=np.float32)


def render_annotations(image, annotations: list[Annotation], color_for_class, *,
                       selected_index=None, selected_color=(0, 255, 255),
                       thickness=2, draw_labels=True, text_color=(255, 255, 255)):
    """BGR raster preview; does not modify image."""
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        out = image.copy()
    elif image.ndim == 3 and image.shape[2] == 1:
        out = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError('Only gray/BGR images supported by raster preview')
    for idx, a in enumerate(annotations):
        pixels = np.rint(a.points_px).astype(np.int32)
        color = (selected_color if idx == selected_index else tuple(_bgr_color(color_for_class, a.class_id).astype(int)))
        cv2.polylines(out, [pixels], True, tuple(int(v) for v in color), max(1, int(thickness)))
        if draw_labels:
            x0, y0, _, _ = bounds(a.points_px)
            cv2.putText(out, str(a.class_id), (int(round(x0)), max(12, int(round(y0))-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tuple(text_color), 1, cv2.LINE_AA)
    return out


def overlay_class_mask(image, mask, color_for_class, *, encoding: MaskEncoding = MaskEncoding(),
                       opacity=0.5, annotations=None, restrict_to_annotations=False):
    """Blend indexed mask with global class palette (BGR), preserving background.

    If geometries are supplied, their class IDs color mask pixels inside them;
    restrict_to_annotations prevents leftover mask values coloring empty areas.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('overlay_class_mask requires BGR HxWx3 image')
    if mask.ndim != 2:
        raise ValueError('mask must be 2D')
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = float(np.clip(opacity, 0., 1.))
    result = image.copy()
    if alpha <= 0:
        return result
    valid = mask != encoding.background_value
    covered = np.zeros((h, w), dtype=bool)
    if annotations:
        # Similar to existing dataset viewer policy: detect < obb < segment,
        # within same kind smaller objects drawn last to own the overlap.
        order = {'detect': 0, 'obb': 1, 'segment': 2}
        ranked = sorted(annotations, key=lambda a: (order[a.kind], -cv2.contourArea(a.points_px.astype(np.float32))))
        for ann in ranked:
            selection = (annotation_binary_mask(ann, ImageSize(w, h)) > 0) & valid
            if selection.any():
                bgr = _bgr_color(color_for_class, ann.class_id)
                result[selection] = np.round((1-alpha)*image[selection].astype(np.float32) + alpha*bgr).clip(0,255).astype(np.uint8)
                covered |= selection
    if restrict_to_annotations:
        return result
    for pix in np.unique(mask):
        val = int(pix)
        if val == encoding.background_value:
            continue
        cls = val - encoding.offset
        if cls < 0:
            raise ValueError('Mask contains an invalid negative class ID')
        select = (mask == pix) & ~covered
        if select.any():
            bgr = _bgr_color(color_for_class, cls)
            result[select] = np.round((1-alpha)*image[select].astype(np.float32) + alpha*bgr).clip(0,255).astype(np.uint8)
    return result
