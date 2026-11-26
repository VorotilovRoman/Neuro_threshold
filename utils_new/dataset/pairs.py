# -*- coding: utf-8 -*-
"""Folder scanning, image-file conversion and pair-display presentation metadata. Qt-free."""

import os


from utils_new.import_export.images import (
    collect_image_paths,
    image_size_from_file,
    index_mask_files,
)
from utils_new.dataset.validation import (
    INPUT_FORMAT_AUTO,
    PairValidator,
)
from utils_new.import_export.images import read_mask_with_fallback


def scan_detection_pairs(
    images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO
):
    types_stats = {}
    pairs = []
    for img_path in collect_image_paths(images_folder):
        name = os.path.splitext(os.path.basename(img_path))[0]
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
    masks = index_mask_files(masks_folder)
    for img_path in collect_image_paths(images_folder):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = masks.get(name.casefold())
        if mask_path is None:
            continue
        mask_path = str(mask_path)
        if os.path.realpath(mask_path) == os.path.realpath(img_path):
            if log_cb:
                log_cb(f"Предупреждение: маска совпадает с изображением {img_path}")
            continue
        test = read_mask_with_fallback(str(mask_path))
        if test is None:
            if log_cb:
                log_cb(f"Предупреждение: не удалось прочитать маску {mask_path}")
            continue
        pairs.append((img_path, mask_path))
    return pairs


def scan_image_only_pairs(images_folder):
    return [(p, None) for p in collect_image_paths(images_folder)]


def build_pair_display(
    pair, validator, dataset_type, include_empty, *, validated=None, details=None
):
    img_path = pair[0]
    aux_path = pair[1] if len(pair) > 1 else None
    name = os.path.splitext(os.path.basename(img_path))[0]

    is_valid, err, color_type, bit_type = (
        validated if validated is not None else validator.validate_full(pair)
    )

    if details is not None:
        ann_type = details.get("ann_type", "error" if not is_valid else None)
        if is_valid and details.get('is_background'):
            ann_type = 'empty'
    elif dataset_type == 0 and aux_path:
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

    if is_valid and dataset_type != 2 and aux_path is None:
        tag = "unlabelled"
        text = f"{name}{color_marker}{bit_marker} [Без разметки]"
    elif not is_valid:
        tag = "invalid"
        kind = "label" if dataset_type == 0 else ("mask" if dataset_type == 1 else "")
        text = f"{name}{color_marker}{bit_marker} (image + {kind}) [INVALID: {err}]"
    elif ann_type == "empty":
        tag = "empty"
        marker = "empty → include" if include_empty else "empty → skip"
        kind = 'mask' if dataset_type == 1 else 'label'
        text = f"{name}{color_marker}{bit_marker} (image + {kind}) [{marker}]"
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


class PairInspectionJob:
    """File-by-file validation/statistics snapshot; never retain decoded images."""

    def __init__(self, pairs, dataset_type=0, input_format="auto", include_empty=True):
        self.pairs = list(pairs)
        self.dataset_type, self.input_format = dataset_type, input_format
        self.include_empty = include_empty
        self.cancelled = False
        self.progress = lambda *_: None
        self.log = lambda *_: None

    def cancel(self):
        self.cancelled = True

    def run(self):
        from collections import Counter

        validator = PairValidator(self.dataset_type, self.input_format)
        rows, classes = [], set()
        colors, bits, types = Counter(), Counter(), Counter()
        for index, pair in enumerate(self.pairs):
            if self.cancelled:
                return {"cancelled": True}
            validated = validator.validate_full(pair)
            details = validator.details.get(tuple(pair[:2]), {})
            info = build_pair_display(
                pair,
                validator,
                self.dataset_type,
                self.include_empty,
                validated=validated,
                details=details,
            )
            rows.append(info)
            info['annotation_found'] = bool(self.dataset_type != 2 and len(pair) > 1
                                            and pair[1] is not None and os.path.isfile(pair[1]))
            colors[info["color_type"]] += 1
            bits[info["bit_type"]] += 1
            if info.get("ann_type"):
                types[info["ann_type"]] += 1
            if info["is_valid"]:
                classes.update(details.get("classes", []))
            self.progress(index + 1, len(self.pairs))
        if self.cancelled:
            return {"cancelled": True}
        valid = sum(row['is_valid'] for row in rows)
        labelled = sum(row['annotation_found'] for row in rows)
        summary = dict(images=len(self.pairs), checked=len(rows), labelled=labelled,
                       unlabelled=len(rows) - labelled, errors=len(rows) - valid,
                       with_objects=sum(row['is_valid'] and row['annotation_found'] and row['ann_type'] != 'empty' for row in rows),
                       background=sum(row['is_valid'] and row['annotation_found'] and row['ann_type'] == 'empty' for row in rows),
                       classes=len(classes - {0} if self.dataset_type == 1 else classes))
        return dict(
            rows=rows,
            classes=sorted(classes),
            colors=dict(colors),
            bits=dict(bits),
            types=dict(types),
            validation_cache=validator._cache,
            pairs=self.pairs,
            valid=valid,
            summary=summary,
        )
