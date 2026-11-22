"""Explicit image file conversions, preserving masks and reporting new paths."""

import os
import cv2
from utils_new.import_export.images import _read_with_fallback, write_image
from utils_new.images.processing import ImageStats


def select_pairs_for_conversion(pairs, validator, target):
    to_convert = []
    for pair in pairs:
        img_path = pair[0]
        _, _, ctype, _ = validator.validate_full(pair)
        if target == "gray":
            if ctype in ("color", "gray_visual"):
                to_convert.append((pair, img_path, ctype))
        elif target == "color":
            if ctype == "gray_real":
                to_convert.append((pair, img_path, ctype))
    return to_convert


def convert_pairs_color(pairs, target, progress_cb=None, log_cb=None):
    converted = 0
    errors = 0
    total = len(pairs)
    step = max(1, total // 100) if total else 1
    for idx, item in enumerate(pairs):
        img_path = item[1] if len(item) > 1 else item[0][0]
        try:
            img = _read_with_fallback(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors += 1
            else:
                out = (
                    ImageStats.to_gray(img)
                    if target == "gray"
                    else ImageStats.to_bgr3(img)
                )
                if out is None:
                    errors += 1
                else:
                    write_image(str(img_path), out)
                    converted += 1
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"Ошибка конвертации {img_path}: {e}")
        if progress_cb and total and ((idx + 1) % step == 0 or idx == total - 1):
            progress_cb(idx + 1, total)
    return converted, errors


def select_pairs_for_8bit_png(pairs, validator):
    """
    Возвращает (all_pairs, need_convert).
    ВАЖНО: для dataset_type==1 (masks) маски НЕ включаются —
    они должны оставаться в исходном dtype.
    """
    need = []
    for pair in pairs:
        _, _, _, btype = validator.validate_full(pair)
        if btype != "uint8_png":
            need.append(pair)
    return list(pairs), need


def convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None, skip_masks=False):
    """
    skip_masks: если True — pairs трактуются как pairs из (img, mask),
                но конвертируем только изображения (pair[0]), не маски.
    """
    converted = 0
    errors = 0
    path_map = {}
    total = len(pairs)
    step = max(1, total // 100) if total else 1
    for idx, pair in enumerate(pairs):
        img_path = pair[0]
        try:
            img = _read_with_fallback(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors += 1
            else:
                img8 = ImageStats.to_uint8(img)
                base, ext = os.path.splitext(str(img_path))
                target_path = str(img_path) if ext.lower() == ".png" else base + ".png"
                if target_path != str(img_path) and os.path.exists(target_path):
                    raise FileExistsError(f"Target PNG already exists: {target_path}")
                write_image(target_path, img8)
                if target_path != str(img_path):
                    # Do not delete the source if its removal fails: the user
                    # still has the original, and the new path is reported.
                    try:
                        os.remove(str(img_path))
                    except OSError as exc:
                        if log_cb:
                            log_cb(f"Не удалось удалить {img_path}: {exc}")
                    path_map[str(img_path)] = target_path
                converted += 1
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"Ошибка конвертации {img_path}: {e}")
        if progress_cb and total and ((idx + 1) % step == 0 or idx == total - 1):
            progress_cb(idx + 1, total)
    return converted, errors, path_map
