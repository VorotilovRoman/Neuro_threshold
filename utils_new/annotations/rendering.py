"""Optional NumPy/OpenCV rendering for exports; never replaces Qt overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import cv2
import numpy as np

from utils_new.annotations.adapters import from_legacy
from utils_new.annotations.geometry import bounds, polygon_area
from utils_new.annotations.masks import (
    annotation_binary_mask,
    mask_roi_at_image_scale,
    normalize_index_mask_shape,
)
from utils_new.annotations.model import Annotation, ImageSize, MaskEncoding


def render_annotations(
    image,
    annotations: list[Annotation],
    color_for_class,
    *,
    selected_index=None,
    selected_color=(0, 255, 255),
    thickness=2,
    draw_labels=True,
    text_color=(255, 255, 255),
):
    """BGR raster preview; does not modify image."""
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        out = image.copy()
    elif image.ndim == 3 and image.shape[2] == 1:
        out = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError("Only gray/BGR images supported by raster preview")
    for idx, a in enumerate(annotations):
        pixels = np.rint(a.points_px).astype(np.int32)
        color = (
            selected_color
            if idx == selected_index
            else tuple(_bgr_color(color_for_class, a.class_id).astype(int))
        )
        cv2.polylines(
            out, [pixels], True, tuple(int(v) for v in color), max(1, int(thickness))
        )
        if draw_labels:
            x0, y0, _, _ = bounds(a.points_px)
            cv2.putText(
                out,
                str(a.class_id),
                (int(round(x0)), max(12, int(round(y0)) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple(text_color),
                1,
                cv2.LINE_AA,
            )
    return out


def overlay_class_mask(
    image,
    mask,
    color_for_class,
    *,
    encoding: MaskEncoding = MaskEncoding(),
    opacity=0.5,
    annotations=None,
    restrict_to_annotations=False,
):
    """Canonical indexed-mask overlay; specific/small annotation owns overlaps."""
    return blend_index_mask(
        image,
        mask,
        color_for_class,
        opacity=opacity,
        pixel_offset=encoding.offset,
        background_value=encoding.background_value,
        annotations=annotations,
        prefer_annotation_classes=True,
        annotation_order="specific",
        invalid_indices="error",
        restrict_to_annotations=restrict_to_annotations,
    )[0]


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
        preview = ", ".join(map(str, self.values[:12]))
        if len(self.values) > 12:
            preview += ", ..."
        return (
            f"Значения: [{preview}]; ненулевых/объектных пикселей: "
            f"{self.foreground_pixels}/{self.total_pixels} "
            f"({self.percentage:.2f}%), окрашено: {self.rendered_pixels}"
        )


def _validate_and_resize(mask: np.ndarray, width: int, height: int):
    mask = normalize_index_mask_shape(mask)
    if mask.shape != (height, width):
        # Index resampling preserves even uint32/int64 IDs unsupported by cv2.resize.
        mask = mask_roi_at_image_scale(
            mask, ImageSize(width, height), (0, 0, width, height)
        )
    return mask


def blend_index_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_for_class,
    *,
    opacity: float = 0.5,
    pixel_offset: int = 1,
    background_value: Optional[int] = 0,
    pixel_to_class: Optional[Mapping[int, int]] = None,
    annotations=None,
    prefer_annotation_classes: bool = False,
    annotation_order="input",
    restrict_to_annotations=False,
    invalid_indices="ignore",
):
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
    if annotation_order not in ("input", "specific"):
        raise ValueError("annotation_order must be input or specific")
    if invalid_indices not in ("ignore", "error", "clamp"):
        raise ValueError("invalid_indices must be ignore, error or clamp")
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Отображаемый снимок должен иметь тип BGR uint8 HxWx3")
    h, w = image.shape[:2]
    source = _validate_and_resize(np.asarray(mask), w, h)
    unique = np.unique(source)
    values = tuple(int(x) for x in unique)
    if not np.isfinite(opacity):
        raise ValueError("Mask opacity must be finite")
    alpha = float(np.clip(opacity, 0.0, 1.0))
    output = image.copy()

    foreground = (
        np.ones((h, w), dtype=bool)
        if background_value is None
        else source != int(background_value)
    )
    if pixel_to_class is not None:
        mapping = {
            int(k): int(v)
            for k, v in pixel_to_class.items()
            if background_value is None or int(k) != int(background_value)
        }
        allowed = np.zeros((h, w), dtype=bool)
        for value in mapping:
            allowed |= source == value
        foreground &= allowed
    else:
        mapping = {}
        for value in values:
            if background_value is not None and value == background_value:
                continue
            class_id = value - int(pixel_offset)
            if class_id < 0:
                if invalid_indices == "error":
                    raise ValueError("Mask contains pixel below class offset")
                if invalid_indices == "ignore":
                    continue
                class_id = 0
            mapping[value] = class_id
        foreground &= np.isin(source, list(mapping))

    affected = np.zeros((h, w), dtype=bool)
    if alpha > 0:
        # Qt annotations are stored in legacy normalized tuples, whereas the
        # canonical core uses pixel edges. Convert only when needed.
        if prefer_annotation_classes and annotations:
            converted = []
            for value in annotations:
                try:
                    converted.append(
                        value
                        if isinstance(value, Annotation)
                        else from_legacy(value, ImageSize(w, h), allow_classless=True)
                    )
                except (ValueError, TypeError, AttributeError):
                    continue
            if annotation_order == "specific":
                priority = {"detect": 0, "obb": 1, "segment": 2}
                converted.sort(
                    key=lambda a: (priority[a.kind], -polygon_area(a.points_px))
                )
            for ann in converted:
                try:
                    covered = annotation_binary_mask(ann, ImageSize(w, h)).astype(bool)
                    covered &= foreground
                    if not covered.any():
                        continue
                    color = _bgr_color(color_for_class, ann.class_id)
                    output[covered] = (
                        np.rint(
                            (1.0 - alpha) * image[covered].astype(np.float32)
                            + alpha * color
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    affected |= covered
                except (ValueError, TypeError, AttributeError, cv2.error):
                    # Invalid display-only annotation cannot hide an otherwise
                    # valid mask; uncaptured pixels are rendered by pixel index.
                    continue
        for pixel, class_id in () if restrict_to_annotations else mapping.items():
            selected = (source == pixel) & foreground & ~affected
            if not selected.any():
                continue
            color = _bgr_color(color_for_class, class_id)
            output[selected] = (
                np.rint(
                    (1.0 - alpha) * image[selected].astype(np.float32) + alpha * color
                )
                .clip(0, 255)
                .astype(np.uint8)
            )
            affected |= selected

    return output, MaskPreviewInfo(
        values, int(foreground.sum()), h * w, int(affected.sum())
    )


def _bgr_color(color_for_class, class_id):
    value = tuple(int(c) for c in color_for_class(int(class_id)))
    if len(value) != 3 or not all(0 <= c <= 255 for c in value):
        raise ValueError(f"Неверный цвет класса {class_id}: {value!r}")
    return np.asarray(value, dtype=np.float32)
