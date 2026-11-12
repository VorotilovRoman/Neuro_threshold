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
