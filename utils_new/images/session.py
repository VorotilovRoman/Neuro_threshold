"""One image-record pipeline with explicit display/model numerical policies."""

from functools import partial
from pathlib import Path
import cv2
import numpy as np
from utils_new.import_export import images as image_io
from utils_new.import_export.yolo_labels import load_annotations
from utils_new.images.processing import (
    normalize_to_uint8,
    segmentation_uint8,
    resize_to_max_side,
)
from utils_new.images.cache import LazyImageCache

HARD_MAX_SIDE = 16384


def load_one_image_item(
    image_path,
    resize_enabled=True,
    max_side=1024,
    safe_max_side=None,
    load_annotations_flag=True,
    format_hint="auto",
    hard_max_side=HARD_MAX_SIDE,
    normalization_policy="image-display-v1",
):
    if normalization_policy not in ("image-display-v1", "legacy-segmentation-v1"):
        raise ValueError("Unknown normalization policy")
    image = image_io.read_image_with_fallback(image_path)
    if image is None or image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
        return None
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return None
    if not np.isfinite(image).all():
        return None
    source_h, source_w = image.shape[:2]
    segmentation = normalization_policy == "legacy-segmentation-v1"
    image = segmentation_uint8(image) if segmentation else normalize_to_uint8(image)
    rounding = "floor" if segmentation else "round"
    for limit in (hard_max_side, max_side if resize_enabled else None, safe_max_side):
        image = resize_to_max_side(image, limit, rounding=rounding)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    annotations = []
    label = Path(image_path).with_suffix(".txt")
    if load_annotations_flag and label.is_file():
        annotations = load_annotations(
            label, source_w, source_h, format_hint=format_hint
        )
    h, w = image.shape[:2]
    return dict(
        path=str(image_path),
        image=image,
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        annotations=annotations,
        source_size=(source_w, source_h),
        working_size=(w, h),
        scale=(w / source_w, h / source_h),
        normalization_policy=normalization_policy,
    )


load_segmentation_image = partial(
    load_one_image_item, normalization_policy="legacy-segmentation-v1"
)


def create_segmentation_cache(source, **loader_options):
    return LazyImageCache(
        image_io.collect_image_paths(source),
        partial(load_segmentation_image, **loader_options),
        window=8,
        prefetch_ahead=4,
        max_resident=16,
        max_workers=2,
    )
