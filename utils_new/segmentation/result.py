"""Segmentation values and display math, independent of Qt and model backends."""

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class ResultStatus(str, Enum):
    NO_IMAGE = "no_image"
    NOT_RUN = "not_run"
    RUNNING = "running"
    READY = "ready"
    ERROR = "error"


@dataclass
class SegmentationResult:
    status: ResultStatus = ResultStatus.NOT_RUN
    mask: np.ndarray | None = None
    error: str | None = None


@dataclass(frozen=True)
class InferenceObject:
    class_id: int
    mask: np.ndarray
    native: object = None


@dataclass(frozen=True)
class ModelPrediction:
    """Keep native instances/classes beside the binary foreground preview.

    Semantic pixel ID 0 is background; class offset defaults to 1.
    Native YOLO class IDs already use annotation numbering and stay unchanged.
    A rasterized box is geometry, never a claimed pixel segmentation.
    """
    shape: tuple
    objects: tuple = ()
    label_map: np.ndarray | None = None
    mask_kind: str = 'segmentation'

    @classmethod
    def semantic(cls, mask, metadata):
        mask = np.asarray(mask)
        union = binary_mask(mask, mask.shape)
        offset = metadata.get('annotation_class_offset', metadata.get('mask_class_offset', 1))
        if type(offset) is not int or offset < 0:
            raise ValueError('Смещение классов маски должно быть неотрицательным целым.')
        indexed = (metadata.get('num_classes', 0) > 2 or metadata.get('output_kind') == 'labels'
                   or (not metadata.get('num_classes') and mask.dtype.kind in 'ui' and not np.isin(mask, [0, 255]).all()))
        if indexed:
            if mask.dtype.kind not in 'ui' or (mask < 0).any():
                raise ValueError('Карта классов должна содержать неотрицательные целые ID.')
            entries = tuple(InferenceObject(int(value) - offset, (mask == value).astype(np.uint8) * 255)
                            for value in np.unique(mask) if value > 0 and value >= offset)
            return cls(mask.shape, entries, mask.copy())
        foreground = metadata.get('foreground_class_id', max(0, 1-offset) if metadata.get('num_classes') == 2 else 0)
        return cls(mask.shape, (InferenceObject(foreground, union),) if union.any() else ())

    def combined_mask(self):
        result = np.zeros(self.shape, np.uint8)
        for item in self.objects:
            result |= binary_mask(item.mask, self.shape)
        return result

    def postprocessed(self, steps, shape):
        from utils_new.images.morphology import apply_morphology

        entries = []
        for item in self.objects:
            mask = apply_morphology(item.mask, steps, shape=shape)
            native = item.native if np.array_equal(mask, item.mask) else None
            if mask.any():
                entries.append(InferenceObject(item.class_id, mask, native))
        return ModelPrediction(self.shape, tuple(entries), mask_kind=self.mask_kind)

    def annotations(self, image, kind, use_hull=False):
        from utils_new.annotations.adapters import to_legacy
        from utils_new.annotations.geometry import convert_annotation
        from utils_new.annotations.model import ImageSize
        from utils_new.annotations.modes import supports_hull
        from utils_new.segmentation.postprocessing import extract_objects

        if kind == 'none':
            return []
        use_hull = bool(use_hull and supports_hull(kind))
        result = []
        for item in self.objects:
            # Reuse the global area filter, contour policy and hull implementation.
            shapes = extract_objects(image, item.mask, kind, use_hull)[1]
            # Native rectangles/quads are convex already; a hull must not
            # replace their subpixel coordinates by a raster-derived refit.
            if shapes and item.native is not None and (not use_hull or item.native.kind in ('detect', 'obb')):
                shapes = [to_legacy(convert_annotation(item.native, kind, allow_lossy=True),
                                    ImageSize(image.shape[1], image.shape[0]))]
            result.extend((value[0], item.class_id, *value[2:]) for value in shapes)
        return result


@dataclass(frozen=True)
class ImageFrame:
    path: str
    image: np.ndarray
    gray: np.ndarray
    source_size: tuple[int, int]
    working_size: tuple[int, int]
    scale: tuple[float, float]
    key: tuple


def binary_mask(mask, image_shape):
    """Convert binary/probability/label-union output to uint8 0/255.

    Integer labels: every positive class belongs to the foreground union.
    Float probabilities: threshold 0.5. Exact float 0/255 is also supported.
    Wrong sizes and invalid/nonfinite values are errors, never empty success.
    """
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != tuple(image_shape[:2]):
        raise ValueError(
            f"Размер маски {array.shape} не совпадает с кадром {tuple(image_shape[:2])}."
        )
    if (
        array.dtype.kind not in "buif"
        or not np.isfinite(array).all()
        or (array < 0).any()
    ):
        raise ValueError(
            "Маска должна содержать конечные неотрицательные числовые значения."
        )
    if array.dtype.kind == "f":
        if array.max(initial=0) <= 1:
            foreground = array >= 0.5
        elif np.isin(array, [0, 255]).all():
            foreground = array > 0
        else:
            raise ValueError(
                "Ожидаются вероятности 0..1 или бинарная float-маска 0/255."
            )
    else:
        foreground = array > 0
    return foreground.astype(np.uint8) * 255


def masked_preview(image, mask):
    if image.ndim == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        result = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        result = image.copy()
    else:
        raise ValueError("Кадр должен быть grayscale, BGR или BGRA.")
    if mask is not None:
        result[binary_mask(mask, image.shape) == 0] = 255
    return result
