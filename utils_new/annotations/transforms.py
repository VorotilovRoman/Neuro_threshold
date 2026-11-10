"""One deterministic transform path for pixels, annotations and indexed mask.

Geometry coordinates are continuous pixel *edges*. OpenCV warpAffine samples
pixel centers x+0.5, y+0.5: M_index=T(-.5) @ M_edges @ T(+.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from .geometry import (
    GeometryError,
    bounds,
    canonical_obb_points,
    clip_annotation,
    is_axis_aligned,
    is_rectangle,
    rectangle_points,
)
from .model import Annotation, AnnotationSample, ImageSize


@dataclass(frozen=True)
class Flip:
    horizontal: bool = False
    vertical: bool = False


@dataclass(frozen=True)
class Rotate90:
    k: int = 1                         # Number of clockwise quarter-turns


@dataclass(frozen=True)
class Rotate:
    angle_deg: float                   # Positive = CCW, per OpenCV convention
    expand_canvas: bool = True


@dataclass(frozen=True)
class Shear:
    x_deg: float = 0.0
    y_deg: float = 0.0
    expand_canvas: bool = True


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Resize:
    target: tuple[int, int]            # (output width, output height)
    mode: Literal['letterbox', 'stretch'] = 'letterbox'
    fill_color: tuple[int, ...] | int = (0, 0, 0)


@dataclass(frozen=True)
class Translate:
    dx: float
    dy: float
    expand_canvas: bool = False


@dataclass(frozen=True)
class TransformPlan:
    steps: tuple = ()

    def __init__(self, steps=()):
        object.__setattr__(self, 'steps', tuple(steps))


@dataclass
class TransformReport:
    """Non-fatal data-loss diagnostics (e.g., Detect AABB, OBB -> Segment)."""
    notes: list[str] = field(default_factory=list)


def _translate_matrix(x, y):
    return np.array([[1., 0., float(x)], [0., 1., float(y)], [0., 0., 1.]], dtype=np.float64)


def apply_affine_to_points(points, matrix):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape not in ((2, 3), (3, 3)):
        raise ValueError('matrix must be 2x3 or 3x3')
    result = np.c_[p, np.ones((len(p), 1))] @ M[:2].T
    return result


def compose_affine_matrices(matrices):
    """Input order == application order: [M1,M2] -> M2 @ M1."""
    result = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (2, 3):
            m = np.vstack([m, [0., 0., 1.]])
        if m.shape != (3, 3):
            raise ValueError('Expected 3x3 affine matrices')
        result = m @ result
    return result


def _map_annotations(annotations, M, new_size, *, detect_policy, obb_policy,
                     multi_policy, hole_policy, min_area, report, step_name):
    output = []
    rect = (0, 0, new_size.width, new_size.height)
    for i, a in enumerate(annotations):
        pts = apply_affine_to_points(a.points_px, M)
        kind = a.kind
        if kind == 'detect' and not is_axis_aligned(pts):
            if detect_policy == 'aabb':
                pts = rectangle_points(*bounds(pts))
                report.notes.append(f'{step_name}: Detect #{i} approximated as AABB')
            elif detect_policy == 'promote_to_segment':
                kind = 'segment'
                report.notes.append(f'{step_name}: Detect #{i} promoted to Segment')
            else:
                raise ValueError('detect_policy must be aabb or promote_to_segment')
        if kind == 'obb' and not is_rectangle(pts):
            if obb_policy == 'promote_to_segment':
                kind = 'segment'
                report.notes.append(f'{step_name}: OBB #{i} promoted to Segment after non-rectangular transform')
            elif obb_policy == 'refit':
                pts = canonical_obb_points(pts)
                report.notes.append(f'{step_name}: OBB #{i} refitted (lossy)')
            elif obb_policy == 'error':
                raise GeometryError(f'{step_name}: OBB #{i} not rectangular after affine transform')
            else:
                raise ValueError('obb_policy must be promote_to_segment/refit/error')
        transformed = Annotation(kind, a.class_id, pts)
        clipped = clip_annotation(transformed, rect, min_area=min_area,
                                  multi_policy=multi_policy, obb_policy=obb_policy,
                                  hole_policy=hole_policy)
        if len(clipped) != 1 or (clipped and clipped[0].kind != kind):
            report.notes.append(f'{step_name}: annotation #{i} clipped/split/dropped ({len(clipped)} output)')
        output.extend(clipped)
    return output


def _build_canvas_matrix(M, size: ImageSize, expand_canvas):
    if not expand_canvas:
        return M, size
    corners = rectangle_points(0, 0, size.width, size.height)
    q = apply_affine_to_points(corners, M)
    x0, y0, x1, y1 = bounds(q)
    out_w, out_h = max(1, int(np.ceil(x1-x0 - 1e-9))), max(1, int(np.ceil(y1-y0 - 1e-9)))
    return _translate_matrix(-x0, -y0) @ M, ImageSize(out_w, out_h)


def _pixel_center_matrix(M):
    return _translate_matrix(-.5, -.5) @ M @ _translate_matrix(.5, .5)


def _border_value_for_image(fill, img):
    if img.ndim == 2:
        return int(fill[0]) if isinstance(fill, (tuple, list)) else int(fill)
    if isinstance(fill, (tuple, list)):
        if len(fill) == img.shape[2]:
            return tuple(int(x) for x in fill)
        return tuple([int(fill[0])] * img.shape[2])
    return tuple([int(fill)] * img.shape[2])


def _warp(sample, M, new_size, fill_color):
    A = _pixel_center_matrix(M)[:2]
    image = cv2.warpAffine(sample.image, A, (new_size.width, new_size.height),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=_border_value_for_image(fill_color, sample.image))
    # OpenCV collapses (H,W,1), preserve caller's channel rank.
    if sample.image.ndim == 3 and sample.image.shape[2] == 1 and image.ndim == 2:
        image = image[..., None]
    mask = None
    if sample.mask is not None:
        mask = cv2.warpAffine(sample.mask, A, (new_size.width, new_size.height),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=sample.mask_encoding.background_value)
    return image, mask


def _resize(sample, step):
    out_w, out_h = map(int, step.target)
    if out_w < 1 or out_h < 1:
        raise ValueError('Resize target must be positive')
    h, w = sample.image.shape[:2]
    if step.mode == 'stretch':
        img = cv2.resize(sample.image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mask = (cv2.resize(sample.mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                if sample.mask is not None else None)
        M = np.diag([out_w/w, out_h/h, 1.])
    elif step.mode == 'letterbox':
        scale = min(out_w/w, out_h/h)
        nw, nh = max(1, min(out_w, int(round(w*scale)))), max(1, min(out_h, int(round(h*scale))))
        left, top = (out_w-nw)//2, (out_h-nh)//2
        right, bottom = out_w - nw - left, out_h - nh - top
        img = cv2.resize(sample.image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=_border_value_for_image(step.fill_color, sample.image))
        if sample.image.ndim == 3 and sample.image.shape[2] == 1 and img.ndim == 2:
            img = img[..., None]
        mask = None
        if sample.mask is not None:
            mask = cv2.resize(sample.mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=sample.mask_encoding.background_value)
        # Use actual rounded resize factor, not idealized scale.
        M = _translate_matrix(left, top) @ np.diag([nw/w, nh/h, 1.])
    else:
        raise ValueError('Resize mode must be letterbox or stretch')
    return img, mask, M, ImageSize(out_w, out_h)


def _apply_step(sample, step, *, report, detect_policy, obb_policy,
                multi_policy, hole_policy, min_area, fill_color):
    size = sample.size
    name = type(step).__name__
    if isinstance(step, Flip):
        sx, sy = -1 if step.horizontal else 1, -1 if step.vertical else 1
        M = np.array([[sx, 0., size.width if step.horizontal else 0.],
                      [0., sy, size.height if step.vertical else 0.],
                      [0., 0., 1.]], dtype=np.float64)
        flip_code = (-1 if step.horizontal and step.vertical else
                     1 if step.horizontal else 0 if step.vertical else None)
        img = cv2.flip(sample.image, flip_code) if flip_code is not None else sample.image.copy()
        mask = cv2.flip(sample.mask, flip_code) if sample.mask is not None and flip_code is not None else (sample.mask.copy() if sample.mask is not None else None)
        new_size = size
    elif isinstance(step, Rotate90):
        k = int(step.k) % 4
        mapping = [np.eye(3), np.array([[0., -1., size.height], [1., 0., 0.], [0., 0., 1.]]),
                   np.array([[-1., 0., size.width], [0., -1., size.height], [0., 0., 1.]]),
                   np.array([[0., 1., 0.], [-1., 0., size.width], [0., 0., 1.]])]
        M = mapping[k]
        cv_codes = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        img = cv2.rotate(sample.image, cv_codes[k]) if k else sample.image.copy()
        mask = cv2.rotate(sample.mask, cv_codes[k]) if k and sample.mask is not None else (sample.mask.copy() if sample.mask is not None else None)
        new_size = ImageSize(size.height, size.width) if k % 2 else size
    elif isinstance(step, Crop):
        x, y, w, h = step.x, step.y, step.width, step.height
        if not all(isinstance(v, int) for v in (x, y, w, h)) or x < 0 or y < 0 or w < 1 or h < 1 or x+w > size.width or y+h > size.height:
            raise ValueError('Crop must be an integer rectangle inside the current image')
        M = _translate_matrix(-x, -y)
        img = sample.image[y:y+h, x:x+w].copy()
        mask = sample.mask[y:y+h, x:x+w].copy() if sample.mask is not None else None
        new_size = ImageSize(w, h)
    elif isinstance(step, Resize):
        img, mask, M, new_size = _resize(sample, step)
    elif isinstance(step, (Rotate, Shear, Translate)):
        center = _translate_matrix(size.width/2, size.height/2)
        inv_center = _translate_matrix(-size.width/2, -size.height/2)
        if isinstance(step, Rotate):
            angle = np.deg2rad(float(step.angle_deg))
            c, s = float(np.cos(angle)), float(np.sin(angle))
            # OpenCV positive rotates counter-clockwise in displayed image.
            linear = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
            M, new_size = _build_canvas_matrix(center @ linear @ inv_center, size, step.expand_canvas)
        elif isinstance(step, Shear):
            tx, ty = float(np.tan(np.deg2rad(step.x_deg))), float(np.tan(np.deg2rad(step.y_deg)))
            linear = np.array([[1., tx, 0.], [ty, 1., 0.], [0., 0., 1.]])
            M, new_size = _build_canvas_matrix(center @ linear @ inv_center, size, step.expand_canvas)
        else:
            M, new_size = _build_canvas_matrix(_translate_matrix(step.dx, step.dy), size, step.expand_canvas)
        img, mask = _warp(sample, M, new_size, fill_color)
    else:
        raise TypeError(f'Unrecognized transform step: {step!r}')
    if sample.image.ndim == 3 and sample.image.shape[2] == 1 and img.ndim == 2:
        img = img[..., None]
    result_anns = _map_annotations(sample.annotations, M, new_size,
                                    detect_policy=detect_policy, obb_policy=obb_policy,
                                    multi_policy=multi_policy, hole_policy=hole_policy,
                                    min_area=min_area, report=report, step_name=name)
    return AnnotationSample(img, result_anns, mask, sample.mask_encoding)


def transform_sample(sample: AnnotationSample, plan, *,
                     detect_policy='aabb', obb_policy='promote_to_segment',
                     multi_policy='all', hole_policy='error',
                     min_area=1e-9, fill_color=(0, 0, 0), return_report=False):
    """Transform image, canonical annotations and optional mask identically.

    Safe defaults preserve precise OBB/Segment geometry by promoting a cut or
    skewed OBB to Segment, while Detect remains an AABB (reported as lossy).
    """
    if not isinstance(sample, AnnotationSample):
        raise TypeError('sample must be AnnotationSample')
    steps = plan.steps if isinstance(plan, TransformPlan) else tuple(plan)
    report = TransformReport()
    result = sample.detached()
    for step in steps:
        result = _apply_step(result, step, report=report,
                             detect_policy=detect_policy, obb_policy=obb_policy,
                             multi_policy=multi_policy, hole_policy=hole_policy,
                             min_area=min_area, fill_color=fill_color)
    return (result, report) if return_report else result


def extract_roi(sample: AnnotationSample, bounds_xyxy, **kwargs):
    x0, y0, x1, y1 = bounds_xyxy
    if any(int(v) != v for v in bounds_xyxy):
        raise ValueError('ROI boundaries must be integers')
    return transform_sample(sample, [Crop(int(x0), int(y0), int(x1-x0), int(y1-y0))], **kwargs)
