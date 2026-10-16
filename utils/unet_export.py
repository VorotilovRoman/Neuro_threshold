# utils/unet_export.py
"""
Экспорт U-Net-датасета.

Два сценария:
    1) input=labels → export_masks_from_annotations:
       читает labels/{split}/*.txt, рисует маски (пиксель = cls + offset)
       в masks/{split}/*.png.

    2) input=masks → маски уже лежат в masks/{split} (после _split_and_move),
       здесь — только export_unet_yaml.

Дополнительно — write_unet_yaml пишет masks_data.yaml в корень датасета.

Структура masks_data.yaml:
    path: /abs/path
    train_images: images/train
    train_masks: masks/train
    val_images:   images/val
    val_masks:    masks/val
    test_images:  images/test
    test_masks:   masks/test
    num_classes: N + 1        # включая фон
    class_names: ["background", ...]
    mask_class_offset: 1
"""
from import_libs_external import *
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.image_io import load_annotations, FORMAT_HINT_AUTO


def _choose_dtype(max_pixel_value):
    return np.uint16 if max_pixel_value > 255 else np.uint8


def export_masks_from_annotations(output_path, split, class_names,
                                   mask_class_offset=1,
                                   progress_cb=None, log_cb=None):
    """
    Читает labels/{split}/*.txt, пишет masks/{split}/*.png.
    Пиксель = yolo_cls + mask_class_offset (по умолчанию 1 — U-Net-стиль).
    Фон = 0.

    class_names: {yolo_cls_id: name} — плотная нумерация.
    Возвращает число записанных масок.
    """
    output_path = Path(output_path)
    img_dir = output_path / 'images' / split
    lbl_dir = output_path / 'labels' / split
    mask_dir = output_path / 'masks' / split
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not lbl_dir.is_dir():
        if log_cb:
            log_cb(f"[U-Net] {lbl_dir} не существует, пропуск")
        return 0

    # dtype: если максимальное значение пикселя > 255 → uint16
    max_cls = max(class_names.keys()) if class_names else 0
    max_pixel = int(max_cls) + int(mask_class_offset)
    dtype = _choose_dtype(max_pixel)

    txt_files = sorted(lbl_dir.glob('*.txt'))
    total = len(txt_files)
    written = 0
    errors = 0
    step = max(1, total // 100) if total else 1

    for i, lbl_path in enumerate(txt_files):
        stem = lbl_path.stem
        # Ищем соответствующее изображение
        img_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            errors += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue
        h, w = img.shape[:2]

        mask = np.zeros((h, w), dtype=dtype)
        try:
            anns = load_annotations(str(lbl_path), w, h,
                                     format_hint=FORMAT_HINT_AUTO)
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"[U-Net] ошибка чтения {lbl_path}: {e}")
            continue

        for ann in anns:
            typ = ann[0]
            cls = int(ann[1])
            value = cls + int(mask_class_offset)
            try:
                if typ == 'detect':
                    _, _, cx, cy, bw, bh = ann
                    x1 = int(round((cx - bw / 2.0) * w))
                    y1 = int(round((cy - bh / 2.0) * h))
                    x2 = int(round((cx + bw / 2.0) * w))
                    y2 = int(round((cy + bh / 2.0) * h))
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    cv2.rectangle(mask, (x1, y1), (x2, y2), value, -1)
                elif typ in ('obb', 'segment'):
                    pts_norm = ann[2]
                    pts = []
                    for j in range(0, len(pts_norm) - 1, 2):
                        px = int(round(pts_norm[j] * w))
                        py = int(round(pts_norm[j + 1] * h))
                        px = max(0, min(px, w - 1))
                        py = max(0, min(py, h - 1))
                        pts.append([px, py])
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], value)
            except Exception as e:
                if log_cb:
                    log_cb(f"[U-Net] ошибка отрисовки в {lbl_path.name}: {e}")

        out_path = mask_dir / (stem + '.png')
        try:
            if not cv2.imwrite(str(out_path), mask):
                errors += 1
                continue
            written += 1
        except Exception as e:
            errors += 1
            if log_cb:
                log_cb(f"[U-Net] ошибка записи {out_path}: {e}")

        if progress_cb and ((i + 1) % step == 0 or i == total - 1):
            progress_cb(i + 1, total)

    if log_cb:
        log_cb(f"[U-Net] masks/{split}: записано={written}, ошибок={errors} "
               f"(dtype={'uint16' if dtype == np.uint16 else 'uint8'}, "
               f"offset={mask_class_offset})")
    return written


def write_unet_yaml(output_path, class_names, mask_class_offset=1,
                     log_cb=None):
    """
    Пишет masks_data.yaml в корень output_path.
    class_names: {yolo_cls_id: name} — плотная нумерация.
    """
    output_path = Path(output_path)

    classes_list = ["background"]
    for cid in sorted(class_names.keys()):
        classes_list.append(str(class_names[cid]))

    payload = {
        'path': str(output_path.absolute()).replace('\\', '/'),
        'train_images': 'images/train',
        'train_masks': 'masks/train',
        'val_images':   'images/val',
        'val_masks':    'masks/val',
        'test_images':  'images/test',
        'test_masks':   'masks/test',
        'num_classes': len(classes_list),
        'class_names': classes_list,
        'mask_class_offset': int(mask_class_offset),
    }

    out = output_path / 'masks_data.yaml'
    with open(out, 'w', encoding='utf-8') as f:
        yaml.dump(payload, f, default_flow_style=False, allow_unicode=True)

    if log_cb:
        log_cb(f"[U-Net] masks_data.yaml создан: num_classes={len(classes_list)}, "
               f"offset={mask_class_offset}")
    return str(out)