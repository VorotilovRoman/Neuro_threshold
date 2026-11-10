# -*- coding: utf-8 -*-
"""COCO JSON ingestion using canonical geometry and the common YOLO writer."""
import json
import os
from pathlib import Path

import numpy as np

from utils_new.annotations import (
    ImageSize,
    canonical_obb_points,
    from_legacy,
    to_legacy,
)
from utils_new.dataset_io_new import read_image, write_yolo_labels


def canonicalize_obb_corners(norm_points, width, height):
    """Return deterministic corners after fitting in *pixel* coordinates."""
    size = ImageSize(int(width), int(height))
    value = from_legacy(('obb', 0, list(norm_points)), size)
    corners = canonical_obb_points(value.points_px)
    return to_legacy(value.with_points(corners, kind='obb'), size)[2]

def _bbox_to_detect(bbox, w, h):
    """COCO bbox [x, y, bw, bh] в абс. пикселях → canonical detect."""
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    bw_n = bw / w
    bh_n = bh / h
    return ('detect', 0, float(cx), float(cy), float(bw_n), float(bh_n))


def _segmentation_to_segment(seg, w, h):
    """
    COCO segmentation: [poly1, poly2, ...], каждый poly — плоский список абс. координат.
    Возвращает список canonical segment-аннотаций.
    """
    out = []
    for poly in seg or []:
        if not isinstance(poly, (list, tuple)) or len(poly) < 6:
            continue
        norm = []
        for i in range(0, len(poly), 2):
            if i + 1 >= len(poly):
                break
            norm.append(float(poly[i]) / w)
            norm.append(float(poly[i + 1]) / h)
        if len(norm) >= 6:
            out.append(norm)
    return out


def _is_polygon_rectangle(poly_abs, tol=2.0):
    """
    Эвристика: 4 точки, стороны примерно параллельны друг другу,
    углы ~90°. Используется, чтобы отличить OBB от произвольного
    сегмента с 4 точками.
    """
    if len(poly_abs) != 8:
        return False
    pts = np.array(poly_abs, dtype=np.float64).reshape(4, 2)
    # Стороны
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    lengths = [np.linalg.norm(e) for e in edges]
    # Противоположные стороны примерно равны
    if abs(lengths[0] - lengths[2]) > tol or abs(lengths[1] - lengths[3]) > tol:
        return False
    # Углы ~90°: dot(e_i, e_{i+1}) / (|e_i||e_{i+1}|) ~ 0
    for i in range(4):
        e1 = edges[i]
        e2 = edges[(i + 1) % 4]
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cos_a = abs(np.dot(e1, e2) / (n1 * n2))
        if cos_a > 0.15:
            return False
    return True


def load_coco(json_path, log_cb=None, *, preserve_category_ids=False):
    """
    Загружает COCO JSON.

    Возвращает:
        items: list of dict:
            {
                'img_path':   abs path,
                'width':      int,
                'height':     int,
                'annotations': [(canonical), ...],
            }
        categories: {coco_cat_id: name}
        coco_root:  Path — корень COCO-датасета (папка с json).
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO JSON not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    root = Path(json_path).parent

    categories = {}
    for c in payload.get('categories', []):
        cid = int(c.get('id', -1))
        if cid >= 0:
            categories[cid] = str(c.get('name', str(cid)))

    images = payload.get('images', [])
    anns = payload.get('annotations', [])

    # Индексируем аннотации по image_id
    anns_by_image = {}
    for a in anns:
        iid = a.get('image_id')
        anns_by_image.setdefault(iid, []).append(a)

    items = []
    for im in images:
        iid = im.get('id')
        file_name = im.get('file_name', '')
        if not file_name:
            continue
        w = int(im.get('width', 0))
        h = int(im.get('height', 0))

        # file_name может быть относительным (images/train/xxx.png)
        # или абсолютным.
        if os.path.isabs(file_name):
            img_path = file_name
        else:
            img_path = str(root / file_name)
            if not os.path.isfile(img_path):
                # Попробуем как имя файла в корне
                cand = root / os.path.basename(file_name)
                if cand.is_file():
                    img_path = str(cand)

        if not os.path.isfile(img_path):
            if log_cb:
                log_cb(f"[COCO] изображение не найдено: {file_name}")
            continue

        # Если width/height не указаны — прочитаем из файла
        if w <= 0 or h <= 0:
            try:
                img = read_image(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
            except Exception:
                continue

        canonical = []
        for a in anns_by_image.get(iid, []):
            # category_id 1..N → yolo class 0..N-1
            coco_cat = int(a.get('category_id', 0))
            yolo_cls = (coco_cat if preserve_category_ids else max(0, coco_cat - 1))

            bbox = a.get('bbox')
            seg = a.get('segmentation')
            iscrowd = int(a.get('iscrowd', 0))

            if iscrowd:
                # crowd-регион: конвертируем bbox в detect (приблизительно)
                if bbox and len(bbox) == 4:
                    d = _bbox_to_detect(bbox, w, h)
                    canonical.append(('detect', yolo_cls, *d[2:]))
                continue

            # 1) segmentation есть
            if seg:
                polys = _segmentation_to_segment(seg, w, h)
                for poly_norm in polys:
                    # Проверяем эвристику OBB для 4-точечного полигона
                    poly_abs = []
                    for i in range(0, len(poly_norm), 2):
                        poly_abs.append(poly_norm[i] * w)
                        poly_abs.append(poly_norm[i + 1] * h)
                    if len(poly_abs) == 8 and _is_polygon_rectangle(poly_abs):
                        corners = canonicalize_obb_corners(poly_norm, w, h)
                        canonical.append(('obb', yolo_cls, corners))
                    else:
                        canonical.append(('segment', yolo_cls, poly_norm))
                continue

            # 2) только bbox → detect
            if bbox and len(bbox) == 4:
                d = _bbox_to_detect(bbox, w, h)
                canonical.append(('detect', yolo_cls, *d[2:]))
                continue

        items.append({
            'img_path': img_path,
            'width': w,
            'height': h,
            'annotations': canonical,
        })

    if log_cb:
        total_anns = sum(len(it['annotations']) for it in items)
        log_cb(f"[COCO] загружено: images={len(items)}, "
               f"annotations={total_anns}, "
               f"categories={len(categories)}")

    return items, categories, root


def write_yolo_txt_for_coco_items(items, output_labels_dir, class_remap=None,
                                  log_cb=None, name_prefix=""):
    """COCO canonical tuples -> shared YOLO writer, preserving source image pairing."""
    outdir = Path(output_labels_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for item in items:
        image_path = item['img_path']
        label_path = outdir / f"{name_prefix}{Path(image_path).stem}.txt"
        size = ImageSize(int(item['width']), int(item['height']))
        annotations = []
        for value in item['annotations']:
            try:
                if class_remap is not None:
                    value = (value[0], class_remap.get(value[1], value[1]), *value[2:])
                annotations.append(from_legacy(value, size))
            except (TypeError, ValueError) as exc:
                if log_cb:
                    log_cb(f"[COCO] skipped malformed annotation: {exc}")
        try:
            write_yolo_labels(label_path, annotations, size, obb_format='corners')
            pairs.append((image_path, str(label_path)))
        except (OSError, ValueError) as exc:
            if log_cb:
                log_cb(f"[COCO] failed to write {label_path}: {exc}")
    if log_cb:
        log_cb(f"[COCO -> YOLO] {len(pairs)} pairs")
    return pairs
