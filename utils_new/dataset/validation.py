# -*- coding: utf-8 -*-
"""Strict YOLO-label and image/mask pair validation. Qt-free."""

import os

import cv2

from utils_new.annotations import ImageSize
from utils_new.import_export.images import _read_with_fallback, read_mask_with_fallback
from utils_new.images.processing import ImageStats
from utils_new.import_export.yolo_labels import read_yolo_labels, decode_yolo_row

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
INPUT_FORMAT_AUTO = "auto"
INPUT_FORMAT_MIXED = "mixed"
INPUT_FORMAT_DETECT = "detect"
INPUT_FORMAT_OBB_CORNERS = "obb_corners"
INPUT_FORMAT_OBB_XYWHR = "obb_xywhr"
INPUT_FORMAT_SEGMENT = "segment"
_VALID_INPUT_FORMATS = {
    INPUT_FORMAT_AUTO,
    INPUT_FORMAT_MIXED,
    INPUT_FORMAT_DETECT,
    INPUT_FORMAT_OBB_CORNERS,
    INPUT_FORMAT_OBB_XYWHR,
    INPUT_FORMAT_SEGMENT,
}


class PairValidator:
    """
    dataset_type:
        0 — labels (.txt);
        1 — masks (.png);
        2 — images-only.

    input_format (только для dataset_type=0):
        'auto'/'mixed' — эвристика;
        'detect' / 'obb_corners' / 'obb_xywhr' / 'segment' — жёстко.
    """

    def __init__(self, dataset_type=0, input_format=INPUT_FORMAT_AUTO):
        self.dataset_type = dataset_type
        self.input_format = (
            input_format if input_format in _VALID_INPUT_FORMATS else INPUT_FORMAT_AUTO
        )
        self._cache = {}

    def set_dataset_type(self, dataset_type):
        self.dataset_type = dataset_type
        self._cache.clear()

    def set_input_format(self, fmt):
        self.input_format = fmt if fmt in _VALID_INPUT_FORMATS else INPUT_FORMAT_AUTO
        self._cache.clear()

    def clear_cache(self):
        self._cache.clear()

    def validate_full(self, pair):
        key = (
            self.dataset_type,
            self.input_format,
            pair[0],
            pair[1] if len(pair) > 1 else None,
        )

        def signature(path):
            try:
                stat = os.stat(path)
                return stat.st_mtime_ns, stat.st_size
            except (OSError, TypeError):
                return None

        versions = signature(pair[0]), signature(pair[1] if len(pair) > 1 else None)
        if key in self._cache and self._cache[key][0] == versions:
            return self._cache[key][1]
        result = self._validate_impl(pair)
        self._cache[key] = versions, result
        return result

    def validate(self, pair):
        ok, err, _, _ = self.validate_full(pair)
        return ok, err

    def validate_all(self, pairs, progress_cb=None):
        self._cache.clear()
        total = len(pairs)
        if total == 0:
            return 0, 0
        valid = 0
        invalid = 0
        step = max(1, total // 100)
        for idx, pair in enumerate(pairs):
            ok, _ = self.validate(pair)
            if ok:
                valid += 1
            else:
                invalid += 1
            if progress_cb and ((idx + 1) % step == 0 or idx == total - 1):
                progress_cb(idx + 1, total)
        return valid, invalid

    def recompute_stats(self, pairs):
        color_stats = ImageStats.empty_color_stats()
        bit_stats = ImageStats.empty_bit_stats()
        for pair in pairs:
            _, _, ctype, btype = self.validate_full(pair)
            color_stats[ctype] = color_stats.get(ctype, 0) + 1
            bit_stats[btype] = bit_stats.get(btype, 0) + 1
        return color_stats, bit_stats

    def _validate_impl(self, pair):
        img_path = pair[0]
        aux_path = pair[1] if len(pair) > 1 else None

        if not os.path.exists(img_path):
            return False, "Image file not found", "unknown", "unknown"
        try:
            img = _read_with_fallback(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Cannot decode image", "unknown", "unknown"
            img_h, img_w = img.shape[:2]
            if img_h == 0 or img_w == 0:
                return False, "Zero-size image", "unknown", "unknown"
            color_type = ImageStats.detect_color_type(img)
            bit_type = ImageStats.detect_bit_type(img, img_path)
        except Exception as e:
            return False, f"Image read error: {e}", "unknown", "unknown"

        if self.dataset_type == 2:
            return True, "", color_type, bit_type
        if aux_path is None:
            return False, "Annotation file is missing", color_type, bit_type

        if self.dataset_type == 1:
            if not os.path.exists(aux_path):
                return False, "Mask file not found", color_type, bit_type
            try:
                mask = read_mask_with_fallback(str(aux_path))
                if mask is None:
                    return False, "Cannot decode mask", color_type, bit_type
                if mask.ndim != 2:
                    return (
                        False,
                        f"Mask expected 2D, got shape {mask.shape}",
                        color_type,
                        bit_type,
                    )
                if mask.shape[:2] != (img_h, img_w):
                    return (
                        False,
                        f"Mask size {mask.shape[1]}x{mask.shape[0]} != "
                        f"image {img_w}x{img_h}",
                        color_type,
                        bit_type,
                    )
            except Exception as e:
                return False, f"Mask read error: {e}", color_type, bit_type
            return True, "", color_type, bit_type

        try:
            annotations = read_yolo_labels(
                aux_path,
                ImageSize(img_w, img_h),
                format_hint=self.input_format,
                strict=True,
            )
            types = {a.kind for a in annotations}
            if len(types) > 1 and self.input_format != INPUT_FORMAT_MIXED:
                return (
                    False,
                    "Mixed types in file: select mixed format",
                    color_type,
                    bit_type,
                )
            return True, "", color_type, bit_type
        except (OSError, ValueError, cv2.error) as exc:
            return False, str(exc), color_type, bit_type

    @staticmethod
    def detect_annotation_type(
        label_path, format_hint=INPUT_FORMAT_AUTO, image_size=None
    ):
        try:
            from pathlib import Path

            size = ImageSize(*image_size) if image_size is not None else None
            values = [
                decode_yolo_row(line, size, format_hint=format_hint)[0]
                for line in Path(label_path)
                .read_text(encoding="utf-8-sig")
                .splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            kinds = set(values)
            return (
                "empty"
                if not kinds
                else next(iter(kinds))
                if len(kinds) == 1
                else "mixed"
            )
        except (OSError, ValueError, TypeError, IndexError, cv2.error):
            return "error"
