# -*- coding: utf-8 -*-
"""Folder scanning, image-file conversion and pair-display presentation metadata. Qt-free."""
import os

import cv2
import numpy as np

from utils_new.dataset_files_new import _read_with_fallback, write_image
from utils_new.dataset_image_stats_new import ImageStats
from utils_new.dataset_pair_validator_new import (
    IMG_EXTENSIONS,
    INPUT_FORMAT_AUTO,
    PairValidator,
    read_mask_with_fallback,
)


def _list_images(folder):
    result = {}
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTENSIONS:
            result[name] = os.path.join(folder, f)
    return result



def scan_detection_pairs(images_folder, labels_folder, log_cb=None,
                          format_hint=INPUT_FORMAT_AUTO):
    types_stats = {}
    pairs = []
    img_files = _list_images(images_folder)
    for name, img_path in img_files.items():
        label_path = os.path.join(labels_folder, name + '.txt')
        if not os.path.exists(label_path):
            continue
        try:
            ann_type = PairValidator.detect_annotation_type(label_path, format_hint)
            types_stats[ann_type] = types_stats.get(ann_type, 0) + 1
            pairs.append((img_path, label_path))
        except Exception as e:
            if log_cb:
                log_cb(f"Ошибка обработки пары {name}: {e}")
    return pairs, types_stats



def scan_segmentation_pairs(images_folder, masks_folder, log_cb=None):
    pairs = []
    img_files = _list_images(images_folder)
    for name, img_path in img_files.items():
        mask_path = None
        for ext in IMG_EXTENSIONS:
            cand = os.path.join(masks_folder, name + ext)
            if os.path.exists(cand):
                if os.path.abspath(cand) == os.path.abspath(img_path):
                    if log_cb:
                        log_cb(f"Предупреждение: маска совпадает с изображением {img_path}")
                    mask_path = None
                    break
                mask_path = cand
                break
        if mask_path is None:
            for f in os.listdir(masks_folder):
                mask_name, mask_ext = os.path.splitext(f)
                if mask_name == name and mask_ext.lower() in IMG_EXTENSIONS:
                    cand = os.path.join(masks_folder, f)
                    if os.path.abspath(cand) != os.path.abspath(img_path):
                        mask_path = cand
                    break
        if mask_path is None:
            continue
        test = read_mask_with_fallback(str(mask_path))
        if test is None:
            if log_cb:
                log_cb(f"Предупреждение: не удалось прочитать маску {mask_path}")
            continue
        pairs.append((img_path, mask_path))
    return pairs



def scan_image_only_pairs(images_folder):
    return [(p, None) for p in _list_images(images_folder).values()]



def select_pairs_for_conversion(pairs, validator, target):
    to_convert = []
    for pair in pairs:
        img_path = pair[0]
        _, _, ctype, _ = validator.validate_full(pair)
        if target == 'gray':
            if ctype in ('color', 'gray_visual'):
                to_convert.append((pair, img_path, ctype))
        elif target == 'color':
            if ctype == 'gray_real':
                to_convert.append((pair, img_path, ctype))
    return to_convert



def convert_pairs_color(pairs, target, progress_cb=None, log_cb=None):
    converted = 0; errors = 0
    total = len(pairs)
    step = max(1, total // 100) if total else 1
    for idx, item in enumerate(pairs):
        img_path = item[1] if len(item) > 1 else item[0][0]
        try:
            img = _read_with_fallback(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                errors += 1
            else:
                out = ImageStats.to_gray(img) if target == 'gray' \
                    else ImageStats.to_bgr3(img)
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
        if btype != 'uint8_png':
            need.append(pair)
    return list(pairs), need



def convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None,
                              skip_masks=False):
    """
    skip_masks: если True — pairs трактуются как pairs из (img, mask),
                но конвертируем только изображения (pair[0]), не маски.
    """
    converted = 0; errors = 0
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
                target_path = str(img_path) if ext.lower() == '.png' else base + '.png'
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



def build_pair_display(pair, validator, dataset_type, include_empty):
    img_path = pair[0]
    aux_path = pair[1] if len(pair) > 1 else None
    name = os.path.splitext(os.path.basename(img_path))[0]

    is_valid, err, color_type, bit_type = validator.validate_full(pair)

    if dataset_type == 0 and aux_path:
        try:
            ann_type = PairValidator.detect_annotation_type(
                aux_path, validator.input_format
            )
        except Exception:
            ann_type = 'error'
    elif dataset_type == 1:
        ann_type = 'segment'
    else:
        ann_type = None

    color_marker = {
        'gray_real':   " [ч/б 1ch]",
        'gray_visual': " [визуально ч/б]",
        'color':       " [цветное]",
    }.get(color_type, "")

    bit_marker = {
        'uint8_other': " [8b не-PNG]",
        'uint16':      " [16bit]",
        'int16':       " [int16]",
        'float':       " [float]",
        'other':       " [??bit]",
    }.get(bit_type, "")

    if not is_valid:
        tag = 'invalid'
        kind = 'label' if dataset_type == 0 else ('mask' if dataset_type == 1 else '')
        text = f"{name}{color_marker}{bit_marker} (image + {kind}) [INVALID: {err}]"
    elif ann_type == 'empty':
        tag = 'empty'
        marker = "empty → include" if include_empty else "empty → skip"
        text = f"{name}{color_marker}{bit_marker} (image + label) [{marker}]"
    elif ann_type in ('detect', 'obb', 'segment', 'error', 'mixed', 'unknown'):
        tag = ann_type
        text = f"{name}{color_marker}{bit_marker} (image + label) [{ann_type}]"
    else:
        tag = None
        suffix = "image + mask" if dataset_type == 1 else "image only"
        text = f"{name}{color_marker}{bit_marker} ({suffix})"

    return {
        'name': name, 'is_valid': is_valid, 'err': err,
        'color_type': color_type, 'bit_type': bit_type,
        'ann_type': ann_type, 'tag': tag, 'text': text,
    }

