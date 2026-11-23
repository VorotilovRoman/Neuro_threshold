from utils_new.annotations.geometry import fit_normalized_obb_corners

# -*- coding: utf-8 -*-
"""COCO JSON ingestion using canonical geometry and the common YOLO writer."""
import json
import os
from pathlib import Path

import numpy as np

from utils_new.annotations import ImageSize, from_legacy
from utils_new.import_export.images import read_image
from utils_new.import_export.yolo_labels import write_yolo_labels, unique_label_path


def _bbox_to_detect(bbox, w, h):
    """COCO bbox [x, y, bw, bh] в абс. пикселях → canonical detect."""
    x, y, bw, bh = bbox
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    bw_n = bw / w
    bh_n = bh / h
    return ("detect", 0, float(cx), float(cy), float(bw_n), float(bh_n))


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


def load_coco(json_path, log_cb=None, *, preserve_category_ids=False, infer_obb=True):
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

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    root = Path(json_path).parent

    categories = {}
    for c in payload.get("categories", []):
        cid = int(c.get("id", -1))
        if cid >= 0:
            categories[cid] = str(c.get("name", str(cid)))

    images = payload.get("images", [])
    anns = payload.get("annotations", [])

    # Индексируем аннотации по image_id
    anns_by_image = {}
    for a in anns:
        iid = a.get("image_id")
        anns_by_image.setdefault(iid, []).append(a)

    items = []
    for im in images:
        iid = im.get("id")
        file_name = im.get("file_name", "")
        if not file_name:
            continue
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))

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
            coco_cat = int(a.get("category_id", 0))
            yolo_cls = coco_cat if preserve_category_ids else max(0, coco_cat - 1)

            bbox = a.get("bbox")
            seg = a.get("segmentation")
            iscrowd = int(a.get("iscrowd", 0))

            if isinstance(seg, dict):
                try:
                    binary = decode_rle(seg, expected_shape=(h, w))
                except (ValueError, UnicodeError) as exc:
                    if not bbox or len(bbox) != 4:
                        raise ValueError(
                            f"[COCO] Invalid RLE annotation {a.get('id')}: {exc}"
                        ) from exc
                    d = _bbox_to_detect(bbox, w, h)
                    canonical.append(("detect", yolo_cls, *d[2:]))
                    if log_cb:
                        log_cb(
                            f"[COCO] Invalid RLE {a.get('id')}: BBox fallback (lossy): {exc}"
                        )
                    continue
                from utils_new.annotations import mask_to_annotations

                polygons = mask_to_annotations(binary, holes="drop", min_area=0)
                for ann in polygons:
                    coords = (ann.points_px / [w, h]).reshape(-1).tolist()
                    canonical.append(("segment", yolo_cls, coords))
                if not polygons and binary.any():
                    ys, xs = np.nonzero(binary)
                    d = _bbox_to_detect(
                        [
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max() - xs.min() + 1),
                            int(ys.max() - ys.min() + 1),
                        ],
                        w,
                        h,
                    )
                    canonical.append(("detect", yolo_cls, *d[2:]))
                if log_cb:
                    log_cb(
                        f"[COCO] RLE {a.get('id')}: editable contours (lossy: holes/thin pixels); source JSON unchanged"
                    )
                continue

            if iscrowd:
                # crowd-регион: конвертируем bbox в detect (приблизительно)
                if bbox and len(bbox) == 4:
                    d = _bbox_to_detect(bbox, w, h)
                    canonical.append(("detect", yolo_cls, *d[2:]))
                    if log_cb:
                        log_cb(
                            f"[COCO] crowd annotation {a.get('id')}: converted to BBox (lossy)"
                        )
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
                    if (
                        infer_obb
                        and len(poly_abs) == 8
                        and _is_polygon_rectangle(poly_abs)
                    ):
                        corners = fit_normalized_obb_corners(poly_norm, w, h)
                        canonical.append(("obb", yolo_cls, corners))
                    else:
                        canonical.append(("segment", yolo_cls, poly_norm))
                continue

            # 2) только bbox → detect
            if bbox and len(bbox) == 4:
                d = _bbox_to_detect(bbox, w, h)
                canonical.append(("detect", yolo_cls, *d[2:]))
                continue

        items.append(
            {
                "img_path": img_path,
                "width": w,
                "height": h,
                "annotations": canonical,
                "annotation_status": im.get("annotation_status", "labelled"),
            }
        )

    if log_cb:
        total_anns = sum(len(it["annotations"]) for it in items)
        log_cb(
            f"[COCO] загружено: images={len(items)}, "
            f"annotations={total_anns}, "
            f"categories={len(categories)}"
        )

    if payload.get("_dataset_sources") and log_cb:
        from utils_new.import_export.dataset_sources import log_manifest

        log_manifest(json_path, log_cb)
    return items, categories, root


def write_yolo_txt_for_coco_items(
    items, output_labels_dir, class_remap=None, log_cb=None, name_prefix=""
):
    """COCO canonical tuples -> shared YOLO writer, preserving source image pairing."""
    outdir = Path(output_labels_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = []
    used = {p.name.casefold() for p in outdir.iterdir()}
    for item in items:
        image_path = item["img_path"]
        if item.get("annotation_status") == "unlabelled":
            pairs.append((image_path, None))
            continue
        label_path = unique_label_path(
            outdir, image_path, prefix=name_prefix, used=used
        )
        size = ImageSize(int(item["width"]), int(item["height"]))
        annotations = []
        for value in item["annotations"]:
            try:
                if class_remap is not None:
                    value = (value[0], class_remap.get(value[1], value[1]), *value[2:])
                annotations.append(from_legacy(value, size))
            except (TypeError, ValueError) as exc:
                if log_cb:
                    log_cb(f"[COCO] skipped malformed annotation: {exc}")
        try:
            write_yolo_labels(label_path, annotations, size, obb_format="corners")
            pairs.append((image_path, str(label_path)))
        except (OSError, ValueError) as exc:
            if log_cb:
                log_cb(f"[COCO] failed to write {label_path}: {exc}")
    if log_cb:
        log_cb(f"[COCO -> YOLO] {len(pairs)} pairs")
    return pairs


def encode_rle(mask):
    """COCO uncompressed RLE in column-major order, including the initial zero run."""
    values = np.asarray(mask)
    if values.ndim != 2 or not values.size:
        raise ValueError("RLE requires a nonempty 2D mask")
    flat = (values > 0).astype(np.uint8).ravel(order="F")
    changes = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, changes, flat.size]).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return dict(size=list(values.shape), counts=counts)


def decode_rle(segmentation, *, expected_shape=None):
    """Decode standard compressed or uncompressed COCO RLE without optional packages."""
    size = segmentation.get("size")
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or any(type(v) is not int or v < 1 for v in size)
    ):
        raise ValueError("Invalid COCO RLE size")
    if expected_shape is not None and tuple(size) != tuple(expected_shape):
        raise ValueError("COCO RLE size differs from image dimensions")
    counts = segmentation.get("counts")
    if isinstance(counts, (str, bytes)):
        codes = counts.decode("ascii") if isinstance(counts, bytes) else counts
        counts, pos = [], 0
        while pos < len(codes):
            value, shift, more = 0, 0, True
            while more:
                if pos >= len(codes) or shift > 60:
                    raise ValueError("Invalid compressed RLE counts")
                code = ord(codes[pos]) - 48
                if not 0 <= code <= 63:
                    raise ValueError("Invalid compressed RLE character")
                pos += 1
                value |= (code & 31) << shift
                shift += 5
                more = bool(code & 32)
                if not more and code & 16:
                    value |= -1 << shift
            if len(counts) > 2:
                value += counts[-2]
            counts.append(value)
    total = size[0] * size[1]
    if (
        not isinstance(counts, list)
        or not counts
        or any(type(v) is not int or v < 0 for v in counts)
        or sum(counts) != total
    ):
        raise ValueError("COCO RLE counts do not cover the image")
    result = np.zeros(total, np.uint8)
    start = 0
    for i, length in enumerate(counts):
        if i % 2:
            result[start : start + length] = 1
        start += length
    return result.reshape(tuple(size), order="F")


def export_coco_yolo(
    json_path, output_dir, *, class_names=None, subtype="detect", log=None
):
    """Export one homogeneous YOLO task with dense IDs and nested image paths.

    COCO RLE/multipolygons use the common editor conversion. Mask holes cannot
    be represented in YOLO polygons; conversion warnings are sent to ``log``.
    """
    from utils_new.annotations import convert_annotation

    if subtype not in ("detect", "segment"):
        raise ValueError("YOLO export subtype must be detect/segment")
    payload = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
    categories = sorted(payload.get("categories", []), key=lambda c: int(c["id"]))
    names = (
        [str(c.get("name", c["id"])) for c in categories]
        if class_names is None
        else list(class_names)
    )
    if len(set(names)) != len(names):
        raise ValueError("Category names must be unique for name-based remapping")
    mapping = {}
    for category in categories:
        name = str(category.get("name", category["id"]))
        if name not in names:
            raise ValueError(f"Unknown COCO category name: {name}")
        mapping[int(category["id"])] = names.index(name)
    items, _, _ = load_coco(
        json_path, log_cb=log, preserve_category_ids=True, infer_obb=False
    )
    if len(items) != len(payload.get("images", [])):
        raise ValueError("Some COCO images are missing; export refused")
    planned = []
    used = set()
    for meta, item in zip(payload["images"], items):
        relative = Path(str(meta["file_name"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                "COCO → YOLO requires relative image file names without .."
            )
        if relative.parts and relative.parts[0] == "images":
            if len(relative.parts) < 3 or relative.parts[1] not in (
                "train",
                "val",
                "test",
            ):
                raise ValueError(f"Invalid image split: {relative}")
            relative = Path(*relative.parts[1:])
        else:
            relative = Path("train") / relative
        target = Path(output_dir) / "labels" / relative.with_suffix(".txt")
        key = str(target).casefold()
        if key in used or target.exists():
            raise ValueError(f"YOLO label collision: {target}")
        used.add(key)
        size = ImageSize(item["width"], item["height"])
        annotations = []
        for value in item["annotations"]:
            if value[1] not in mapping:
                raise ValueError(f"Unknown category ID: {value[1]}")
            ann = from_legacy((value[0], mapping[value[1]], *value[2:]), size)
            annotations.append(convert_annotation(ann, subtype, allow_lossy=True))
        planned.append((target, annotations, size, item["annotation_status"]))
    for target, annotations, size, status in planned:
        if status != "unlabelled":
            write_yolo_labels(target, annotations, size)
    return names
