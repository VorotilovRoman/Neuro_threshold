"""Pure geometry in continuous pixel-edge coordinates. No image mutation.

Clipping uses exact polygon intersection when Shapely is available. Without
Shapely, convex clipping works; non-convex boundary cuts fail explicitly.
"""
from __future__ import annotations

import cv2
import numpy as np
from .model import Annotation, ImageSize

try:
    from shapely.geometry import Polygon as _Polygon, box as _box
except ImportError:
    _Polygon = _box = None


class GeometryError(ValueError):
    pass


class LossyConversionError(GeometryError):
    pass


def polygon_area(points) -> float:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2)


def bounds(points) -> tuple[float, float, float, float]:
    p = np.asarray(points, dtype=float)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def rectangle_points(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def canonical_obb_points(points):
    """Best-fitting minimum-area rectangle in *pixel* coordinates.

    Canonical order: first uppermost vertex (then leftmost), clockwise in image
    coordinates. Refitting an arbitrary polygon can add background area.
    """
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        raise GeometryError('OBB fit requires at least 3 vertices')
    rect = cv2.minAreaRect(p)
    out = cv2.boxPoints(rect).astype(np.float64)
    if polygon_area(out) <= 1e-10:
        raise GeometryError('Degenerate OBB')
    # cv2.boxPoints is cyclical; do not change winding. Make starting index deterministic.
    first = min(range(4), key=lambda i: (round(float(out[i, 1]), 9), round(float(out[i, 0]), 9)))
    return np.roll(out, -first, axis=0)


def is_rectangle(points, tol=1e-6):
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 2):
        return False
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    if (lengths < 1e-9).any():
        return False
    dot = np.sum(edges * np.roll(edges, -1, axis=0), axis=1)
    return bool(np.all(np.abs(dot) <= tol * lengths * np.roll(lengths, -1)))


def is_axis_aligned(points, atol=1e-7):
    x0, y0, x1, y1 = bounds(points)
    return bool(np.allclose(np.sort(np.asarray(points)[:, 0]), [x0, x0, x1, x1], atol=atol) and
                np.allclose(np.sort(np.asarray(points)[:, 1]), [y0, y0, y1, y1], atol=atol))


def convert_annotation(ann: Annotation, target: str, *, allow_lossy=False,
                       obb_policy='refit') -> Annotation:
    """Detect/OBB/Segment. Never silently return a different target subtype.

    Segment -> OBB and any non-Detect -> Detect normally discard geometry;
    allow_lossy=True is required. 'keep' returns the input annotation.
    """
    if target == 'keep' or target == ann.kind:
        return ann
    if target not in ('detect', 'obb', 'segment'):
        raise ValueError('target must be keep/detect/obb/segment')
    if target == 'segment':
        return ann.with_points(ann.points_px, kind='segment')
    if target == 'detect':
        if not allow_lossy and not (ann.kind == 'obb' and is_axis_aligned(ann.points_px)):
            raise LossyConversionError('Converting to axis-aligned Detect may lose geometry')
        return Annotation('detect', ann.class_id, rectangle_points(*bounds(ann.points_px)))
    if target == 'obb':
        if ann.kind == 'detect':
            return Annotation('obb', ann.class_id, ann.points_px)
        if not allow_lossy and not is_rectangle(ann.points_px):
            raise LossyConversionError('Segment -> OBB requires polygon refit; allow_lossy=True')
        if obb_policy != 'refit':
            raise ValueError('Only obb_policy=refit is supported when converting to OBB')
        return Annotation('obb', ann.class_id, canonical_obb_points(ann.points_px))
    raise AssertionError('unreachable')


def _clip_convex_polygon(points, rect):
    """Sutherland–Hodgman with float64 coordinates, convex polygons only."""
    x0, y0, x1, y1 = rect
    p = [np.asarray(pt, dtype=float) for pt in points]
    for axis, value, keep_greater in ((0, x0, True), (0, x1, False),
                                      (1, y0, True), (1, y1, False)):
        if not p:
            break
        def inside(t):
            return t[axis] >= value - 1e-10 if keep_greater else t[axis] <= value + 1e-10
        def cross(a, b):
            denom = b[axis] - a[axis]
            t = (value - a[axis]) / denom if abs(denom) > 1e-15 else 0.0
            return a + np.clip(t, 0.0, 1.0) * (b-a)
        out = []
        previous = p[-1]
        for current in p:
            if inside(current):
                if not inside(previous):
                    out.append(cross(previous, current))
                out.append(current)
            elif inside(previous):
                out.append(cross(previous, current))
            previous = current
        p = out
    return [np.asarray(p, dtype=float)] if len(p) >= 3 and polygon_area(p) > 1e-10 else []


def _iter_polygons(geom):
    if geom.is_empty:
        return
    if geom.geom_type == 'Polygon':
        yield geom
    elif hasattr(geom, 'geoms'):
        for g in geom.geoms:
            yield from _iter_polygons(g)


def clip_polygon_to_rect(points, rect, *, hole_policy='error', min_area=1e-9):
    """Exact polygon/ROI intersection; may return multiple components.

    For output Segment (which cannot encode holes), hole_policy='error' raises
    instead of silently losing holes; 'drop' explicitly discards interior rings.
    """
    x0, y0, x1, y1 = map(float, rect)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid crop rectangle')
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3 or polygon_area(p) < min_area:
        return []
    if _Polygon is None:
        if not cv2.isContourConvex(p.astype(np.float32)):
            # Whole shape contained needs no topology processing.
            if np.all((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)):
                return [p.copy()]
            raise GeometryError('Shapely required to clip nonconvex polygons correctly')
        return [q for q in _clip_convex_polygon(p, (x0, y0, x1, y1)) if polygon_area(q) >= min_area]
    poly = _Polygon(p)
    if not poly.is_valid:
        raise GeometryError('Self-intersecting polygon; repair explicitly before clipping')
    intersection = poly.intersection(_box(x0, y0, x1, y1))
    result = []
    for component in _iter_polygons(intersection):
        if component.area < min_area:
            continue
        if len(component.interiors) and hole_policy != 'drop':
            raise GeometryError('Polygon with holes cannot be stored as one YOLO Segment; use hole_policy=drop')
        q = np.asarray(component.exterior.coords[:-1], dtype=np.float64)
        if len(q) >= 3:
            result.append(q)
    return result


def clip_annotation(ann: Annotation, rect, *, min_area=1.0,
                    multi_policy='all', obb_policy='promote_to_segment',
                    hole_policy='error') -> list[Annotation]:
    """Clip against pixel-edge ROI. Coordinate origin is NOT shifted.

    multi_policy: all/largest/error for disconnected Segment components.
    obb_policy: promote_to_segment (exact), refit (lossy), error.
    """
    x0, y0, x1, y1 = map(float, rect)
    if ann.kind == 'detect':
        ax0, ay0, ax1, ay1 = bounds(ann.points_px)
        bx0, by0, bx1, by1 = max(ax0, x0), max(ay0, y0), min(ax1, x1), min(ay1, y1)
        if bx1 <= bx0 or by1 <= by0 or (bx1 - bx0)*(by1 - by0) < min_area:
            return []
        return [Annotation('detect', ann.class_id, rectangle_points(bx0, by0, bx1, by1))]
    poly = ann.points_px
    pieces = clip_polygon_to_rect(poly, rect, hole_policy=hole_policy, min_area=min_area)
    if not pieces:
        return []
    if len(pieces) > 1:
        if multi_policy == 'largest':
            pieces = [max(pieces, key=polygon_area)]
        elif multi_policy == 'error':
            raise GeometryError('Clipping produced multiple components')
        elif multi_policy != 'all':
            raise ValueError('multi_policy must be all/largest/error')
    fully_inside = bool(np.all((poly[:, 0] >= x0 - 1e-8) &
                               (poly[:, 0] <= x1 + 1e-8) &
                               (poly[:, 1] >= y0 - 1e-8) &
                               (poly[:, 1] <= y1 + 1e-8)))
    if fully_inside:
        return [ann]
    if ann.kind == 'obb':
        if obb_policy == 'error':
            raise LossyConversionError('Clipped OBB is not a rectangle')
        if obb_policy == 'refit':
            return [Annotation('obb', ann.class_id, canonical_obb_points(p)) for p in pieces]
        if obb_policy != 'promote_to_segment':
            raise ValueError('Unknown OBB clipping policy')
    return [Annotation('segment', ann.class_id, p) for p in pieces]


def overlap_area(a: Annotation, b: Annotation):
    if _Polygon is None:
        if cv2.isContourConvex(a.points_px.astype(np.float32)) and cv2.isContourConvex(b.points_px.astype(np.float32)):
            return float(cv2.intersectConvexConvex(a.points_px.astype(np.float32), b.points_px.astype(np.float32))[0])
        raise GeometryError('Shapely needed for nonconvex polygon overlap')
    return float(_Polygon(a.points_px).intersection(_Polygon(b.points_px)).area)


def overlap_ratio_of_smaller(a: Annotation, b: Annotation):
    smaller = min(polygon_area(a.points_px), polygon_area(b.points_px))
    return min(1.0, max(0.0, overlap_area(a, b) / smaller)) if smaller > 0 else 0.0
