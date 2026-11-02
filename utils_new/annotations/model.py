"""Core types: pixel-edge coordinates, explicit sizes and mask encoding.

Coordinates are continuous pixel *edges*: (0, 0) is the upper-left frame
boundary and (width, height) the lower-right. No Qt, YOLO, filesystem imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

AnnotationKind = Literal['detect', 'obb', 'segment']


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or self.width < 1 or self.height < 1:
            raise ValueError('ImageSize: width/height must be positive integers')


@dataclass(frozen=True, eq=False)
class Annotation:
    """One canonical annotation. points_px is an independent read-only Nx2 float64 array.

    Detect: 4 corners of an axis-aligned rectangle (in arbitrary perimeter order).
    OBB: 4 rectangle corners; Segment: N >= 3 polygon corners.
    """
    kind: AnnotationKind
    class_id: int
    points_px: np.ndarray

    def __post_init__(self):
        if self.kind not in ('detect', 'obb', 'segment'):
            raise ValueError(f'Unsupported annotation kind: {self.kind!r}')
        if not isinstance(self.class_id, (int, np.integer)) or int(self.class_id) < 0:
            raise ValueError('class_id must be a nonnegative integer')
        pts = np.array(self.points_px, dtype=np.float64, copy=True)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError('points_px must be Nx2 with N >= 3')
        if self.kind in ('detect', 'obb') and len(pts) != 4:
            raise ValueError(f'{self.kind} must have 4 points')
        if not np.isfinite(pts).all():
            raise ValueError('points_px contains NaN or infinity')
        x, y = pts[:, 0], pts[:, 1]
        area2 = float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area2 < 1e-10:
            raise ValueError('Annotation polygon has zero area')
        pts.setflags(write=False)
        object.__setattr__(self, 'points_px', pts)
        object.__setattr__(self, 'class_id', int(self.class_id))

    def with_points(self, pts, *, kind=None) -> 'Annotation':
        return Annotation(kind or self.kind, self.class_id, pts)


@dataclass(frozen=True)
class MaskEncoding:
    """Class-to-pixel: value = class_id + offset; background is independent.

    offset=0/background=0 makes class 0 indistinguishable from background.
    Explicitly opt into that ambiguous format with allow_ambiguous_zero=True.
    """
    offset: int = 1
    background_value: int = 0
    allow_ambiguous_zero: bool = False

    def __post_init__(self):
        if not isinstance(self.offset, int) or not isinstance(self.background_value, int):
            raise TypeError('offset and background_value must be integers')
        if self.offset < 0 or self.background_value < 0:
            raise ValueError('mask offset/background_value must be nonnegative')
        if self.offset == self.background_value and not self.allow_ambiguous_zero:
            raise ValueError('class 0 would collide with background; use offset=1 or explicit override')


@dataclass
class AnnotationSample:
    image: np.ndarray
    annotations: list[Annotation] = field(default_factory=list)
    mask: np.ndarray | None = None
    mask_encoding: MaskEncoding | None = None

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray) or self.image.ndim not in (2, 3):
            raise ValueError('image must be a 2D/3D numpy array')
        if self.image.shape[0] < 1 or self.image.shape[1] < 1:
            raise ValueError('image cannot be empty')
        if any(not isinstance(a, Annotation) for a in self.annotations):
            raise TypeError('annotations must contain Annotation instances')
        if self.mask is not None:
            if self.mask.ndim != 2 or self.mask.shape != self.image.shape[:2]:
                raise ValueError('indexed mask must match image H,W and be single-channel')
            if not np.issubdtype(self.mask.dtype, np.integer):
                raise TypeError('indexed mask must have integer dtype')
            if self.mask_encoding is None:
                raise ValueError('mask_encoding is required for an indexed mask')

    @property
    def size(self) -> ImageSize:
        h, w = self.image.shape[:2]
        return ImageSize(w, h)

    def detached(self) -> 'AnnotationSample':
        return AnnotationSample(self.image.copy(), list(self.annotations),
                                self.mask.copy() if self.mask is not None else None,
                                self.mask_encoding)
