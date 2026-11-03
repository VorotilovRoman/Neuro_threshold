# -*- coding: utf-8 -*-
"""Shared, Qt-free indexed-mask preview for dataset viewer and pair validator.

Input masks are *labels* (0, 1, 2, ...) rather than gray-scale intensities.
Never normalize mask pixel values for display. Instead blend categorical colors
into a copy of the original BGR image; the mask is not mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskPreviewInfo:
    values: tuple[int, ...]
    foreground_pixels: int
    total_pixels: int
    rendered_pixels: int

    @property
    def is_empty(self) -> bool:
        return self.foreground_pixels == 0

    @property
    def percentage(self) -> float:
        return 100.0 * self.foreground_pixels / max(1, self.total_pixels)

    def description(self) -> str:
        preview = ', '.join(map(str, self.values[:12]))
        if len(self.values) > 12:
            preview += ', ...'
        return (f'Значения: [{preview}]; ненулевых/объектных пикселей: '
                f'{self.foreground_pixels}/{self.total_pixels} '
                f'({self.percentage:.2f}%), окрашено: {self.rendered_pixels}')


def _validate_and_resize(mask: np.ndarray, width: int, height: int):
    if mask is None or mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError('Нужна одноканальная целочисленная индексная маска (H, W)')
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def blend_index_mask(image_bgr: np.ndarray, mask: np.ndarray, color_for_class, *,
                     opacity: float = 0.5, pixel_offset: int = 1,
                     background_value: Optional[int] = 0,
                     pixel_to_class: Optional[Mapping[int, int]] = None,
                     annotations=None,
                     prefer_annotation_classes: bool = False):
    """Return ``(BGR image, MaskPreviewInfo)`` without Qt, Ultralytics or writes.

    ``pixel_offset`` maps raw pixel -> class ID: class = pixel - offset.
    ``pixel_to_class`` if supplied *overrides* that mapping (source pixel ->
    displayed class); classes excluded from this mapping are not colored.
    ``background_value=None`` allows legacy offset=0 where pixel 0 is class 0.

    If ``prefer_annotation_classes`` is true, annotation class IDs color mask
    regions intersecting matching polygons; elsewhere the indexed mask is
    rendered normally. This supports instance/binary (255) masks together with
    labels without suppressing valid mask pixels outside the polygons.
    """
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Отображаемый снимок должен иметь тип BGR uint8 HxWx3')
    h, w = image.shape[:2]
    source = _validate_and_resize(np.asarray(mask), w, h)
    unique = np.unique(source)
    values = tuple(int(x) for x in unique)
    alpha = float(np.clip(opacity, 0.0, 1.0))
    output = image.copy()

    foreground = (np.ones((h, w), dtype=bool) if background_value is None
                  else source != int(background_value))
    if pixel_to_class is not None:
        mapping = {int(k): int(v) for k, v in pixel_to_class.items()
                   if background_value is None or int(k) != int(background_value)}
        allowed = np.zeros((h, w), dtype=bool)
        for value in mapping:
            allowed |= source == value
        foreground &= allowed
    else:
        mapping = {v: v - int(pixel_offset) for v in values
                   if (background_value is None or v != background_value)
                   and v >= int(pixel_offset)}
        foreground &= np.isin(source, list(mapping))

    affected = np.zeros((h, w), dtype=bool)
    if alpha > 0:
        # Qt annotations are stored in legacy normalized tuples, whereas the
        # canonical core uses pixel edges. Convert only when needed.
        if prefer_annotation_classes and annotations:
            from utils_new.annotations import ImageSize, from_legacy
            from utils_new.annotations.masks import annotation_binary_mask
            for value in annotations:
                try:
                    ann = (value if hasattr(value, 'points_px')
                           else from_legacy(value, ImageSize(w, h), allow_classless=True))
                    covered = annotation_binary_mask(ann, ImageSize(w, h)).astype(bool)
                    covered &= foreground
                    if not covered.any():
                        continue
                    color = _bgr(color_for_class, ann.class_id)
                    output[covered] = np.rint(
                        (1.0 - alpha) * image[covered].astype(np.float32) + alpha * color
                    ).clip(0, 255).astype(np.uint8)
                    affected |= covered
                except (ValueError, TypeError, AttributeError, cv2.error):
                    # Invalid display-only annotation cannot hide an otherwise
                    # valid mask; uncaptured pixels are rendered by pixel index.
                    continue
        for pixel, class_id in mapping.items():
            selected = (source == pixel) & foreground & ~affected
            if not selected.any():
                continue
            color = _bgr(color_for_class, class_id)
            output[selected] = np.rint(
                (1.0 - alpha) * image[selected].astype(np.float32) + alpha * color
            ).clip(0, 255).astype(np.uint8)
            affected |= selected

    return output, MaskPreviewInfo(values, int(foreground.sum()), h * w, int(affected.sum()))


def _bgr(color_for_class, class_id):
    value = tuple(int(c) for c in color_for_class(int(class_id)))
    if len(value) != 3 or not all(0 <= c <= 255 for c in value):
        raise ValueError(f'Неверный цвет класса {class_id}: {value!r}')
    return np.asarray(value, dtype=np.float32)
