"""Indexed masks: explicit background/offset; no interpolation of class IDs."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import (
    GeometryError,
    bounds,
    canonical_obb_points,
    is_axis_aligned,
    rectangle_points,
)
from .model import Annotation, ImageSize, MaskEncoding


def _dtype_for_values(values):
    max_value = max(values) if values else 0
    return np.uint8 if max_value <= 255 else np.uint16 if max_value <= 65535 else np.int32


def annotation_binary_mask(ann: Annotation, size: ImageSize) -> np.ndarray:
    """Raster footprint with half-open pixel-center semantics for rectangles.

    A Detect and the identical contour converted to a four-corner Segment or
    axis-aligned OBB MUST rasterize identically. For irregular/rotated polygons
    OpenCV fillPoly is retained; its boundary convention can differ by at most
    a narrow edge band from the continuous polygon area.
    """
    selected = np.zeros((size.height, size.width), dtype=np.uint8)
    points = ann.points_px
    if ann.kind == 'detect' or (len(points) == 4 and is_axis_aligned(points)):
        x0, y0, x1, y1 = bounds(points)
        # A pixel at integer index x has physical center x+.5.
        left = max(0, int(np.ceil(x0 - .5)))
        top = max(0, int(np.ceil(y0 - .5)))
        right = min(size.width, int(np.ceil(x1 - .5)))
        bottom = min(size.height, int(np.ceil(y1 - .5)))
        if right > left and bottom > top:
            selected[top:bottom, left:right] = 1
    else:
        cv2.fillPoly(selected, [np.rint(points).astype(np.int32)], 1)
    return selected


def rasterize_annotations(annotations: list[Annotation], size: ImageSize, *,
                          encoding: MaskEncoding = MaskEncoding(), overlap='last', dtype=None):
    """Rasterize detect/obb/segment to one label per pixel.

    Overlap: 'last' draws later objects on top, 'first' protects earlier ones,
    'error' rejects overlap between differently labelled objects.
    """
    if overlap not in ('last', 'first', 'error'):
        raise ValueError('overlap must be last/first/error')
    max_values = [encoding.background_value] + [a.class_id + encoding.offset for a in annotations]
    if (not encoding.allow_ambiguous_zero and
            any(v == encoding.background_value for v in max_values[1:])):
        raise ValueError('A class pixel value collides with background_value')
    chosen_dtype = np.dtype(dtype or _dtype_for_values(max_values))
    if not np.issubdtype(chosen_dtype, np.integer):
        raise TypeError('mask dtype must be integer')
    info = np.iinfo(chosen_dtype)
    if max(max_values) > info.max or min(max_values) < info.min:
        raise ValueError(f'Mask class values do not fit {chosen_dtype}')
    result = np.full((size.height, size.width), encoding.background_value, dtype=chosen_dtype)
    for ann in annotations:
        selected = annotation_binary_mask(ann, size) > 0
        if overlap == 'first':
            selected &= result == encoding.background_value
        if overlap == 'error':
            invalid = selected & (result != encoding.background_value) & (result != ann.class_id + encoding.offset)
            if invalid.any():
                raise GeometryError('Overlapping annotations have different classes')
        result[selected] = ann.class_id + encoding.offset
    return result


def remap_mask_indices(mask: np.ndarray, class_map: dict[int, int], *,
                       source_encoding: MaskEncoding = MaskEncoding(),
                       target_encoding: MaskEncoding = MaskEncoding(),
                       unmapped='background', dtype=None):
    """Remap index image without intermediate value collisions or input mutation.

    class_map keys/values are class IDs, not mask pixel values. Out-of-map
    pixels become background (or raise if unmapped='error').
    """
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError('mask must be a 2D integer array')
    if unmapped not in ('background', 'error'):
        raise ValueError('unmapped must be background/error')
    if any(int(k) < 0 or int(v) < 0 for k, v in class_map.items()):
        raise ValueError('Class IDs must be nonnegative')
    highest = max([target_encoding.background_value] + [v + target_encoding.offset for v in class_map.values()])
    out_dtype = np.dtype(dtype or _dtype_for_values([highest]))
    if highest > np.iinfo(out_dtype).max:
        raise ValueError('Target class values do not fit output dtype')
    out = np.full(mask.shape, target_encoding.background_value, dtype=out_dtype)
    covered = mask == source_encoding.background_value
    for old_cls, new_cls in class_map.items():
        pix_old = int(old_cls) + source_encoding.offset
        pix_new = int(new_cls) + target_encoding.offset
        if pix_new == target_encoding.background_value and not target_encoding.allow_ambiguous_zero:
            raise ValueError('Remapped class collides with target background')
        selection = (mask == pix_old) & (mask != source_encoding.background_value)
        out[selection] = pix_new
        covered |= selection
    if unmapped == 'error' and (~covered).any():
        raise ValueError('Unmapped foreground mask pixel encountered')
    return out


def mask_roi_at_image_scale(mask: np.ndarray, size: ImageSize, bounds_xyxy):
    """Nearest-neighbor ROI from a mask whose resolution differs from image."""
    if mask.ndim != 2:
        raise ValueError('Expected a 2D mask')
    x0, y0, x1, y1 = map(int, bounds_xyxy)
    if not (0 <= x0 < x1 <= size.width and 0 <= y0 < y1 <= size.height):
        raise ValueError('Invalid ROI boundaries')
    mh, mw = mask.shape
    if (mw, mh) == (size.width, size.height):
        return mask[y0:y1, x0:x1].copy()
    xs = np.minimum(mw-1, np.floor(np.arange(x0, x1)*mw/size.width).astype(np.intp))
    ys = np.minimum(mh-1, np.floor(np.arange(y0, y1)*mh/size.height).astype(np.intp))
    return mask[np.ix_(ys, xs)].copy()


def mask_to_annotations(mask: np.ndarray, *, encoding: MaskEncoding = MaskEncoding(),
                        kind='segment', min_area=1., holes='error'):
    """Lossy contour extraction from class mask; explicit policy for holes.

    Returns one annotation per exterior connected component of every class.
    Pixel contours cannot round-trip arbitrary holes or thin features exactly.
    """
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError('Expected 2D indexed integer mask')
    if kind not in ('detect', 'obb', 'segment'):
        raise ValueError('Unsupported output annotation kind')
    result = []
    for pix in np.unique(mask):
        value = int(pix)
        if value == encoding.background_value:
            continue
        cls = value - encoding.offset
        if cls < 0:
            raise ValueError('Mask contains pixel value below class offset')
        blob = (mask == pix).astype(np.uint8)
        contours, hierarchy = cv2.findContours(blob, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                continue
            if any(hierarchy[0][child][3] == i for child in range(len(contours))):
                if holes == 'error':
                    raise GeometryError('Mask contains holes: choose holes=drop explicitly')
                if holes != 'drop':
                    raise ValueError('holes must be error or drop')
            points = contour.reshape(-1, 2).astype(np.float64)
            area = cv2.contourArea(contour)
            if area < min_area or len(points) < 3:
                continue
            if kind == 'detect':
                # findContours vertices are pixel centers. Convert to a bounding
                # rectangle in pixel *edge* coordinates (inclusive final pixel).
                bx, by, bw, bh = cv2.boundingRect(contour)
                points = rectangle_points(bx, by, bx+bw, by+bh)
            elif kind == 'obb':
                points = canonical_obb_points(points)
            result.append(Annotation(kind, cls, points))
    return result
