# utils/coco_export.py
"""
Экспорт датасета в COCO JSON.

Два источника:
    • export_coco_from_labels  — читает labels/{split}/*.txt (lossless);
    • export_coco_from_masks   — читает masks/{split}/*.png (lossy, с warning).

Пишет annotations_{split}.json в корень датасета:
    {
      "info": {...}, "licenses": [...],
      "images": [...],
      "annotations": [...],
      "categories": [...]
    }

Правила:
    • bbox — абсолютные пиксели [x, y, w, h], где x,y — левый-верх;
    • segmentation — список полигонов (для segment и obb), абсолютные пиксели;
    • category_id = yolo_class_id + 1 (0 зарезервирован);
    • file_name — относительный путь от корня датасета
      (images/{split}/name.png).
"""
from import_libs_external import *
import json
import os
from pathlib import Path

import cv2
import numpy as np

from utils.image_io import (
    load_annotations, read_mask_with_fallback, FORMAT_HINT_AUTO,
)


def _polygon_area_abs(pts_flat):
    """Площадь полигона (шнуровка), pts_flat — [x1,y1,x2,y2,...] в абс. пикселях."""
    n = len(pts_flat) // 2
    if n < 3:
        return 0.0
    xs = pts_flat[0::2]
    ys = pts_flat[1::2]
    s = 0.0
    for i in range(n):
        s += xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
    return abs(s) * 0.5


def _ann_to_coco_entry(ann, image_id, ann_id, w, h):
    """canonical-аннотация → словарь COCO. Возвращает dict или None."""
    typ = ann[0]
    cls = int(ann[1])
    cat_id = cls + 1

    if typ == 'detect':
        _, _, cx, cy, bw, bh = ann
        x_abs = (cx - bw / 2.0) * w
        y_abs = (cy - bh / 2.0) * h
        w_abs = bw * w
        h_abs = bh * h
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x_abs), float(y_abs), float(w_abs), float(h_abs)],
            "area": float(w_abs * h_abs),
            "iscrowd": 0,
            "segmentation": [],
        }

    if typ == 'segment':
        _, _, pts = ann
        if len(pts) < 6:
            return None
        pts_abs = []
        for i in range(0, len(pts), 2):
            pts_abs.append(float(pts[i] * w))
            pts_abs.append(float(pts[i + 1] * h))
        xs = pts_abs[0::2]
        ys = pts_abs[1::2]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "area": float(_polygon_area_abs(pts_abs)),
            "iscrowd": 0,
            "segmentation": [pts_abs],
        }

    if typ == 'obb':
        _, _, corners = ann
        if len(corners) != 8:
            return None
        pts_abs = []
        for i in range(0, 8, 2):
            pts_abs.append(float(corners[i] * w))
            pts_abs.append(float(corners[i + 1] * h))
        xs = pts_abs[0::2]
        ys = pts_abs[1::2]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "area": float(_polygon_area_abs(pts_abs)),
            "iscrowd": 0,
            "segmentation": [pts_abs],
        }

    return None


def _categories_from_class_names(class_names):
    """
    class_names: {yolo_cls_id: name} — плотная нумерация 0..N-1.
    Возвращает список COCO-категорий: id начиная с 1.
    """
    cats = []
    for cid in sorted(class_names.keys()):
        cats.append({
            "id": int(cid) + 1,
            "name": str(class_names[cid]),
            "supercategory": "none",
        })
    return cats


def _images_for_split(output_path, split):
    """Список изображений в images/{split}. Возвращает [(img_path, w, h)]."""
    img_dir = output_path / 'images' / split
    out = []
    if not img_dir.is_dir():
        return out
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        out.append((f, w, h))
    return out


def export_coco_from_labels(output_path, split, class_names,
                             log_cb=None):
    """
    Читает labels/{split}/*.txt, пишет annotations_{split}.json.

    class_names: {yolo_cls_id: name} — плотная нумерация.
    Возвращает число записанных аннотаций.
    """
    output_path = Path(output_path)
    lbl_dir = output_path / 'labels' / split
    images = _images_for_split(output_path, split)

    coco_images = []
    coco_annotations = []
    ann_id = 0

    for img_id, (img_path, w, h) in enumerate(images):
        rel_name = f"images/{split}/{img_path.name}"
        coco_images.append({
            "id": img_id,
            "file_name": rel_name,
            "width": int(w),
            "height": int(h),
            "license": 1,
        })

        lbl_path = lbl_dir / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        try:
            anns = load_annotations(str(lbl_path), w, h,
                                     format_hint=FORMAT_HINT_AUTO)
        except Exception as e:
            if log_cb:
                log_cb(f"[COCO] ошибка чтения {lbl_path}: {e}")
            continue

        for ann in anns:
            entry = _ann_to_coco_entry(ann, img_id, ann_id, w, h)
            if entry is None:
                continue
            coco_annotations.append(entry)
            ann_id += 1

    payload = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "Dataset generated by Dataset Preparation Tool",
            "contributor": "Dataset Preparation Tool",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": _categories_from_class_names(class_names),
    }

    out = output_path / f'annotations_{split}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if log_cb:
        log_cb(f"[COCO] {out.name}: images={len(coco_images)}, "
               f"annotations={len(coco_annotations)}, "
               f"categories={len(payload['categories'])}")
    return len(coco_annotations)


def _mask_to_segment_polygons(mask, pixel_value, w, h):
    """
    Возвращает список полигонов (каждый — плоский список абс. координат)
    для всех контуров с данным pixel_value.
    Lossy: теряются дырки, мелкие детали.
    """
    binary = (mask == pixel_value).astype(np.uint8)
    if binary.sum() == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for contour in contours:
        if len(contour) < 3:
            continue
        c = contour.squeeze()
        if c.ndim != 2 or c.shape[0] < 3:
            continue
        flat = []
        for pt in c:
            flat.append(float(pt[0]))
            flat.append(float(pt[1]))
        polys.append(flat)
    return polys


def export_coco_from_masks(output_path, split, class_names,
                            mask_pixel_offset=1, log_cb=None):
    """
    Читает masks/{split}/*.png, делает findContours, пишет
    annotations_{split}.json. Lossy.

    mask_pixel_offset: pixel_value = yolo_class + offset.
        offset=1 — U-Net-стиль (фон=0, класс i → i+1);
        offset=0 — pixel = yolo_class.
    """
    output_path = Path(output_path)
    mask_dir = output_path / 'masks' / split
    images = _images_for_split(output_path, split)

    coco_images = []
    coco_annotations = []
    ann_id = 0

    # pixel_value для каждого yolo-класса
    pixel_values = {int(cid): int(cid) + int(mask_pixel_offset)
                    for cid in class_names.keys()}

    for img_id, (img_path, w, h) in enumerate(images):
        rel_name = f"images/{split}/{img_path.name}"
        coco_images.append({
            "id": img_id,
            "file_name": rel_name,
            "width": int(w),
            "height": int(h),
            "license": 1,
        })

        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'):
            cand = mask_dir / (img_path.stem + ext)
            if cand.exists():
                mask_path = cand
                break
        if mask_path is None:
            continue

        mask = read_mask_with_fallback(str(mask_path))
        if mask is None:
            if log_cb:
                log_cb(f"[COCO из масок] не читается {mask_path}")
            continue
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for cls_id, pixel_val in pixel_values.items():
            polys = _mask_to_segment_polygons(mask, pixel_val, w, h)
            for poly in polys:
                if len(poly) < 6:
                    continue
                xs = poly[0::2]
                ys = poly[1::2]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1),
                             float(x2 - x1), float(y2 - y1)],
                    "area": float(_polygon_area_abs(poly)),
                    "iscrowd": 0,
                    "segmentation": [poly],
                })
                ann_id += 1

    payload = {
        "info": {
            "year": 2026, "version": "1.0",
            "description": "Dataset generated from masks (lossy contours)",
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": _categories_from_class_names(class_names),
    }

    out = output_path / f'annotations_{split}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if log_cb:
        log_cb(f"[COCO из масок] {out.name}: images={len(coco_images)}, "
               f"annotations={len(coco_annotations)} "
               f"(⚠️ lossy: контуры из масок)")
    return len(coco_annotations)