# -*- coding: utf-8 -*-
"""Folder scanning, image-file conversion and pair-display presentation metadata. Qt-free."""

import os


from utils_new.import_export.images import collect_image_paths, image_size_from_file
from utils_new.dataset.validation import (
    IMG_EXTENSIONS,
    INPUT_FORMAT_AUTO,
    PairValidator,
)
from utils_new.import_export.images import read_mask_with_fallback


def _list_images(folder):
    return {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in collect_image_paths(folder)
    }


def scan_detection_pairs(
    images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO
):
    types_stats = {}
    pairs = []
    img_files = _list_images(images_folder)
    for name, img_path in img_files.items():
        label_path = os.path.join(labels_folder, name + ".txt")
        if not os.path.exists(label_path):
            continue
        try:
            ann_type = PairValidator.detect_annotation_type(
                label_path, format_hint, image_size_from_file(img_path)
            )
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
                        log_cb(
                            f"Предупреждение: маска совпадает с изображением {img_path}"
                        )
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


def build_pair_display(pair, validator, dataset_type, include_empty):
    img_path = pair[0]
    aux_path = pair[1] if len(pair) > 1 else None
    name = os.path.splitext(os.path.basename(img_path))[0]

    is_valid, err, color_type, bit_type = validator.validate_full(pair)

    if dataset_type == 0 and aux_path:
        try:
            ann_type = PairValidator.detect_annotation_type(
                aux_path, validator.input_format, image_size_from_file(img_path)
            )
        except Exception:
            ann_type = "error"
    elif dataset_type == 1:
        ann_type = "segment"
    else:
        ann_type = None

    color_marker = {
        "gray_real": " [ч/б 1ch]",
        "gray_visual": " [визуально ч/б]",
        "color": " [цветное]",
    }.get(color_type, "")

    bit_marker = {
        "uint8_other": " [8b не-PNG]",
        "uint16": " [16bit]",
        "int16": " [int16]",
        "float": " [float]",
        "other": " [??bit]",
    }.get(bit_type, "")

    if not is_valid:
        tag = "invalid"
        kind = "label" if dataset_type == 0 else ("mask" if dataset_type == 1 else "")
        text = f"{name}{color_marker}{bit_marker} (image + {kind}) [INVALID: {err}]"
    elif ann_type == "empty":
        tag = "empty"
        marker = "empty → include" if include_empty else "empty → skip"
        text = f"{name}{color_marker}{bit_marker} (image + label) [{marker}]"
    elif ann_type in ("detect", "obb", "segment", "error", "mixed", "unknown"):
        tag = ann_type
        text = f"{name}{color_marker}{bit_marker} (image + label) [{ann_type}]"
    else:
        tag = None
        suffix = "image + mask" if dataset_type == 1 else "image only"
        text = f"{name}{color_marker}{bit_marker} ({suffix})"

    return {
        "name": name,
        "is_valid": is_valid,
        "err": err,
        "color_type": color_type,
        "bit_type": bit_type,
        "ann_type": ann_type,
        "tag": tag,
        "text": text,
    }
