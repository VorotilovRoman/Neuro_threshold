"""Read-only geometric information for an annotation inspector; no Qt or settings."""

from dataclasses import dataclass

import numpy as np

from .adapters import from_legacy
from .geometry import bounds, polygon_area
from .model import Annotation, ImageSize


@dataclass(frozen=True)
class ObjectGeometry:
    points: np.ndarray
    bounds: tuple
    center: tuple
    area: float

    @property
    def width(self):
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self):
        return self.bounds[3] - self.bounds[1]


@dataclass(frozen=True)
class ObjectInfo:
    """ID is the zero-based index in the complete extraction result, never a row."""

    index: int
    normalized: Annotation

    @classmethod
    def from_tuple(cls, index, value):
        return cls(index, from_legacy(value, ImageSize(1, 1)))

    def geometry(self, size):
        points = self.normalized.points_px * np.asarray(size)
        box = bounds(points)
        return ObjectGeometry(
            points,
            box,
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            polygon_area(points),
        )

    @property
    def area_fraction(self):
        return polygon_area(self.normalized.points_px)

    def source_annotation(self, source_size):
        return self.normalized.with_points(self.geometry(source_size).points)
