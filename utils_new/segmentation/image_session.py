"""Segmentation image records with explicit legacy numeric/resize compatibility.

Reading and caching are owned by dataset_files_new. No Qt, model loading, or
shared mutable image collections across tabs.
"""
from pathlib import Path

import cv2
import numpy as np

from utils_new.dataset_files_new import (
    HARD_MAX_SIDE,
    LazyImageCache,
    read_image_with_fallback,
)
from utils_new.dataset_yolo_labels_new import load_annotations

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def image_paths(source):
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        return sorted(str(p) for p in Path(source).iterdir() if p.suffix.lower() in EXTENSIONS)
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]
    return []


def legacy_uint8(image):
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    if image.dtype in (np.float32, np.float64) and image.size and image.max() <= 1:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def resize_legacy(image, max_side):
    h, w = image.shape[:2]
    if max_side is None or max(h, w) <= max_side:
        return image
    factor = max_side / max(h, w)
    return cv2.resize(image, (max(1, int(w*factor)), max(1, int(h*factor))),
                      interpolation=cv2.INTER_AREA)


def load_one_image_item(image_path, resize_enabled=True, max_side=1024,
                        safe_max_side=None, load_annotations_flag=True, format_hint='auto'):
    image = read_image_with_fallback(image_path)
    if image is None or image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
        return None
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        return None
    source_h, source_w = image.shape[:2]
    image = resize_legacy(legacy_uint8(image), HARD_MAX_SIDE)
    if resize_enabled:
        image = resize_legacy(image, max_side)
    image = resize_legacy(image, safe_max_side)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] != 3:
        image = image[:, :, :3]
    annotations = []
    label = Path(image_path).with_suffix('.txt')
    if load_annotations_flag and label.exists():
        annotations = load_annotations(label, source_w, source_h, format_hint=format_hint)
    h, w = image.shape[:2]
    return {'path': str(image_path), 'image': image,
            'gray': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 'annotations': annotations,
            'source_size': (source_w, source_h), 'working_size': (w, h),
            'scale': (w/source_w, h/source_h), 'normalization_policy': 'legacy-segmentation-v1'}


def create_cache(source, **loader_options):
    paths = image_paths(source)
    return LazyImageCache(paths, lambda path: load_one_image_item(path, **loader_options),
                         window=8, prefetch_ahead=4, max_resident=16, max_workers=2)
