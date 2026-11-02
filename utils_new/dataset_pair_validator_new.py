# -*- coding: utf-8 -*-
"""Strict YOLO-label and image/mask pair validation. Qt-free."""
import os
import cv2
import numpy as np
from utils_new.annotations import ImageSize, to_legacy
from utils_new.dataset_files_new import _read_with_fallback, read_index_mask
from utils_new.dataset_yolo_labels_new import read_yolo_labels
from utils_new.dataset_image_stats_new import ImageStats
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
INPUT_FORMAT_AUTO = 'auto'
INPUT_FORMAT_MIXED = 'mixed'
INPUT_FORMAT_DETECT = 'detect'
INPUT_FORMAT_OBB_CORNERS = 'obb_corners'
INPUT_FORMAT_OBB_XYWHR = 'obb_xywhr'
INPUT_FORMAT_SEGMENT = 'segment'
_VALID_INPUT_FORMATS = {INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED,
    INPUT_FORMAT_DETECT, INPUT_FORMAT_OBB_CORNERS, INPUT_FORMAT_OBB_XYWHR, INPUT_FORMAT_SEGMENT}


def read_mask_with_fallback(path):
    try:
        return read_index_mask(path)
    except (OSError, ValueError):
        return None



def load_annotations(path, image_w, image_h, format_hint=INPUT_FORMAT_AUTO):
    """Legacy tuple API backed by the canonical parser (no old image_io)."""
    size = ImageSize(max(1, int(image_w)), max(1, int(image_h)))
    return [to_legacy(a, size) for a in read_yolo_labels(path, size, format_hint=format_hint)]



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
        self.input_format = input_format if input_format in _VALID_INPUT_FORMATS \
            else INPUT_FORMAT_AUTO
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
        key = (self.dataset_type, self.input_format, pair[0],
               pair[1] if len(pair) > 1 else None)
        if key in self._cache:
            return self._cache[key]
        result = self._validate_impl(pair)
        self._cache[key] = result
        return result

    def validate(self, pair):
        ok, err, _, _ = self.validate_full(pair)
        return ok, err

    def validate_all(self, pairs, progress_cb=None):
        self._cache.clear()
        total = len(pairs)
        if total == 0:
            return 0, 0
        valid = 0; invalid = 0
        step = max(1, total // 100)
        for idx, pair in enumerate(pairs):
            ok, _ = self.validate(pair)
            if ok: valid += 1
            else: invalid += 1
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
            return False, "Image file not found", 'unknown', 'unknown'
        try:
            img = _read_with_fallback(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Cannot decode image", 'unknown', 'unknown'
            img_h, img_w = img.shape[:2]
            if img_h == 0 or img_w == 0:
                return False, "Zero-size image", 'unknown', 'unknown'
            color_type = ImageStats.detect_color_type(img)
            bit_type = ImageStats.detect_bit_type(img, img_path)
        except Exception as e:
            return False, f"Image read error: {e}", 'unknown', 'unknown'

        if aux_path is None:
            return True, "", color_type, bit_type

        if self.dataset_type == 1:
            if not os.path.exists(aux_path):
                return False, "Mask file not found", color_type, bit_type
            try:
                mask = read_mask_with_fallback(str(aux_path))
                if mask is None:
                    return False, "Cannot decode mask", color_type, bit_type
                if mask.ndim != 2:
                    return (False,
                            f"Mask expected 2D, got shape {mask.shape}",
                            color_type, bit_type)
                if mask.dtype not in (np.uint8, np.uint16):
                    return (False,
                            f"Mask dtype {mask.dtype} (expected uint8 or uint16)",
                            color_type, bit_type)
                if mask.shape[:2] != (img_h, img_w):
                    return (False,
                            f"Mask size {mask.shape[1]}x{mask.shape[0]} != "
                            f"image {img_w}x{img_h}",
                            color_type, bit_type)
            except Exception as e:
                return False, f"Mask read error: {e}", color_type, bit_type
            return True, "", color_type, bit_type

        if not os.path.exists(aux_path):
            return False, "Label file not found", color_type, bit_type
        try:
            with open(aux_path, 'r', encoding='utf-8') as f:
                raw_lines = f.readlines()
        except Exception as e:
            return False, f"Label read error: {e}", color_type, bit_type

        lines = []
        for ln, raw in enumerate(raw_lines, 1):
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            lines.append((ln, s))

        if not lines:
            return True, "", color_type, bit_type

        types_in_file = set()
        for ln, line in lines:
            parts = line.split()
            if len(parts) < 5:
                return (False, f"Line {ln}: too few values ({len(parts)})",
                        color_type, bit_type)
            try:
                cls = int(parts[0])
                if cls < 0:
                    return (False, f"Line {ln}: negative class id {cls}",
                            color_type, bit_type)
            except ValueError:
                return (False, f"Line {ln}: invalid class id '{parts[0]}'",
                        color_type, bit_type)
            try:
                coords = [float(p) for p in parts[1:]]
            except ValueError as e:
                return (False, f"Line {ln}: invalid coordinate ({e})",
                        color_type, bit_type)

            def in_range(v):
                return bool(np.isfinite(v)) and 0.0 <= v <= 1.0

            n = len(parts)

            # Учитываем input_format: жёсткое правило или эвристика.
            fmt = self.input_format
            if fmt in (INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED):
                if n == 5:
                    kind = 'detect'
                elif n == 6:
                    kind = 'obb_xywhr'
                elif n == 9:
                    kind = 'obb_corners'
                elif n >= 7 and (n - 1) % 2 == 0:
                    kind = 'segment'
                else:
                    return (False,
                            f"Line {ln}: unknown format ({n} values)",
                            color_type, bit_type)
            elif fmt == INPUT_FORMAT_DETECT:
                if n != 5:
                    return (False,
                            f"Line {ln}: expected detect (5 values), got {n}",
                            color_type, bit_type)
                kind = 'detect'
            elif fmt == INPUT_FORMAT_OBB_CORNERS:
                if n != 9:
                    return (False,
                            f"Line {ln}: expected obb_corners (9 values), got {n}",
                            color_type, bit_type)
                kind = 'obb_corners'
            elif fmt == INPUT_FORMAT_OBB_XYWHR:
                if n != 6:
                    return (False,
                            f"Line {ln}: expected obb_xywhr (6 values), got {n}",
                            color_type, bit_type)
                kind = 'obb_xywhr'
            elif fmt == INPUT_FORMAT_SEGMENT:
                if not (n >= 7 and (n - 1) % 2 == 0):
                    return (False,
                            f"Line {ln}: expected segment (even >= 6 coords), got {n}",
                            color_type, bit_type)
                kind = 'segment'
            else:
                return (False, f"Line {ln}: unknown input_format {fmt}",
                        color_type, bit_type)

            if not np.isfinite(coords).all():
                return (False, f"Line {ln}: non-finite coordinate", color_type, bit_type)

            if kind == 'detect':
                types_in_file.add('detect')
                cx, cy, bw, bh = coords
                if not in_range(cx) or not in_range(cy):
                    return (False, f"Line {ln}: center out of [0,1]",
                            color_type, bit_type)
                if bw <= 0 or bh <= 0:
                    return (False, f"Line {ln}: non-positive bbox size",
                            color_type, bit_type)
                if bw > 1 or bh > 1:
                    return (False, f"Line {ln}: bbox size > 1",
                            color_type, bit_type)
            elif kind == 'obb_corners':
                types_in_file.add('obb')
                for i in range(0, 8, 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return (False,
                                f"Line {ln}: OBB point {i // 2 + 1} out of [0,1]",
                                color_type, bit_type)
            elif kind == 'obb_xywhr':
                types_in_file.add('obb')
                cx, cy, bw, bh, angle = coords
                if not in_range(cx) or not in_range(cy):
                    return (False, f"Line {ln}: center out of [0,1]",
                            color_type, bit_type)
                if bw <= 0 or bh <= 0:
                    return (False, f"Line {ln}: non-positive OBB size",
                            color_type, bit_type)
            else:  # segment
                types_in_file.add('segment')
                num_pts = (n - 1) // 2
                if num_pts < 3:
                    return (False,
                            f"Line {ln}: polygon needs >=3 points, got {num_pts}",
                            color_type, bit_type)
                for i in range(0, len(coords), 2):
                    if not in_range(coords[i]) or not in_range(coords[i + 1]):
                        return (False,
                                f"Line {ln}: polygon point out of [0,1]",
                                color_type, bit_type)

        if len(types_in_file) > 1:
            return (False,
                    f"Mixed types in file: {', '.join(sorted(types_in_file))}",
                    color_type, bit_type)

        return True, "", color_type, bit_type

    @staticmethod
    def detect_annotation_type(label_path, format_hint=INPUT_FORMAT_AUTO):
        types = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    l = len(parts)
                    if format_hint in (INPUT_FORMAT_AUTO, INPUT_FORMAT_MIXED):
                        if l == 5:
                            types.add('detect')
                        elif l == 6:
                            types.add('obb')
                        elif l == 9:
                            types.add('obb')
                        elif l >= 7 and (l - 1) % 2 == 0:
                            types.add('segment')
                        else:
                            types.add('unknown')
                    elif format_hint == INPUT_FORMAT_DETECT:
                        types.add('detect' if l == 5 else 'unknown')
                    elif format_hint == INPUT_FORMAT_OBB_CORNERS:
                        types.add('obb' if l == 9 else 'unknown')
                    elif format_hint == INPUT_FORMAT_OBB_XYWHR:
                        types.add('obb' if l == 6 else 'unknown')
                    elif format_hint == INPUT_FORMAT_SEGMENT:
                        types.add('segment'
                                  if (l >= 7 and (l - 1) % 2 == 0)
                                  else 'unknown')
        except Exception:
            return 'error'
        if len(types) == 0:
            return 'empty'
        if len(types) == 1:
            return next(iter(types))
        return 'mixed'

    @staticmethod
    def check_pair_type_consistency():
        return None

