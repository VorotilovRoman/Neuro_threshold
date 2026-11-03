# -*- coding: utf-8 -*-
"""Единая обработка аннотаций Detect/OBB/Segment для вкладок датасета и YOLO.

Источник аннотаций не важен: записи YOLO из файлов и результаты
Ultralytics конвертируются к единой схеме ('detect'/'obb'/'segment', class, ...).
Отвечает за кроп, поворот, смещения, наложение масок, морфологию,
проверку наложений и преобразования координат. Не использует PyQt5 и
не загружает модели. Высокоуровневый конвейер не меняет входные массивы;
низкоуровневая fill_outside_object сохраняет исходную in-place семантику.

Цвета BGR; координаты унифицированных аннотаций нормализованы.
YOLO detection dictionaries содержат пиксельные координаты.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils_new.morphology import (
    MorphologyStep,
    apply_morphology as apply_shared_morphology,
)

try:  # необязательный ускоренный и точный backend для невыпуклых полигонов
    from shapely.geometry import Polygon as _Polygon
except ImportError:
    _Polygon = None


def class_id_for_annotation(annotation) -> int:
    """Класс отсутствует/None/отрицательный -> 0 (общая политика V4)."""
    if not isinstance(annotation, (tuple, list)) or not annotation:
        return 0
    typ = annotation[0]
    if typ == "detect" and len(annotation) == 5:
        return 0
    if typ in ("obb", "segment") and len(annotation) == 2:
        return 0
    try:
        value = int(annotation[1])
    except (IndexError, TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


def normalized_annotation(annotation):
    """Добавляет класс 0 к устаревшим записям без класса, не меняя геометрию."""
    if not isinstance(annotation, (tuple, list)) or not annotation:
        return annotation
    typ = annotation[0]
    if typ == "detect":
        if len(annotation) == 5:
            return ("detect", 0, *annotation[1:])
        if len(annotation) == 6:
            try:
                if annotation[1] is None or int(annotation[1]) < 0:
                    return ("detect", 0, *annotation[2:])
            except (TypeError, ValueError, OverflowError):
                return ("detect", 0, *annotation[2:])
    if typ in ("obb", "segment"):
        if len(annotation) == 2:
            return (typ, 0, annotation[1])
        if len(annotation) == 3:
            try:
                if annotation[1] is None or int(annotation[1]) < 0:
                    return (typ, 0, annotation[2])
            except (TypeError, ValueError, OverflowError):
                return (typ, 0, annotation[2])
    return annotation


def polygon_for_annotation(annotation, width: int, height: int) -> np.ndarray:
    """N×2 float64 вершины в пикселях исходного изображения, без округления."""
    if not annotation or width <= 0 or height <= 0:
        return np.empty((0, 2), dtype=np.float64)
    ann = normalized_annotation(annotation)
    try:
        typ = ann[0]
        if typ == "detect":
            _, _, cx, cy, bw, bh = ann
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            points = [
                (cx - bw / 2, cy - bh / 2),
                (cx + bw / 2, cy - bh / 2),
                (cx + bw / 2, cy + bh / 2),
                (cx - bw / 2, cy + bh / 2),
            ]
        elif typ in ("obb", "segment"):
            flat = ann[2]
            if len(flat) < 6 or len(flat) % 2:
                return np.empty((0, 2), dtype=np.float64)
            points = [(float(flat[i]), float(flat[i + 1]))
                      for i in range(0, len(flat), 2)]
        else:
            return np.empty((0, 2), dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64)
        pts[:, 0] *= width
        pts[:, 1] *= height
        if not np.isfinite(pts).all() or cv2.contourArea(pts.astype(np.float32)) <= 0:
            return np.empty((0, 2), dtype=np.float64)
        return pts
    except (ValueError, TypeError, IndexError, OverflowError):
        return np.empty((0, 2), dtype=np.float64)


def crop_bounds_for_annotation(annotation, image_w: int, image_h: int,
                               padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """Pixel-exclusive (x1,y1,x2,y2), обрезанные по границам снимка."""
    poly = polygon_for_annotation(annotation, image_w, image_h)
    if len(poly) < 3:
        return None
    pad = max(0, int(padding))
    x1 = max(0, int(math.floor(float(poly[:, 0].min()))) - pad)
    y1 = max(0, int(math.floor(float(poly[:, 1].min()))) - pad)
    x2 = min(image_w, int(math.ceil(float(poly[:, 0].max()))) + pad)
    y2 = min(image_h, int(math.ceil(float(poly[:, 1].max()))) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_for_annotation(image: np.ndarray, annotation, padding: int = 10):
    """Возвращает (copy ROI без ресайза, bounds); не меняет входной массив."""
    if image is None or image.size == 0:
        return None, None
    h, w = image.shape[:2]
    bounds = crop_bounds_for_annotation(annotation, w, h, padding)
    if bounds is None:
        return None, None
    x1, y1, x2, y2 = bounds
    return image[y1:y2, x1:x2].copy(), bounds


def annotation_in_crop(annotation, original_w: int, original_h: int,
                       bounds: Tuple[int, int, int, int]):
    """Пересчитывает аннотацию в координаты кропа для векторного Qt-слоя.

    OBB сохраняет 4 угла, segment — все вершины, detect — cx/cy/w/h.
    Геометрию не поворачивает, не масштабирует и не сдвигает относительно ROI.
    """
    ann = normalized_annotation(annotation)
    x1, y1, x2, y2 = bounds
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        raise ValueError("Пустая область кропа")
    cls_id = class_id_for_annotation(ann)
    if ann[0] == "detect":
        _, _, cx, cy, bw, bh = ann
        return ("detect", cls_id,
                (float(cx) * original_w - x1) / cw,
                (float(cy) * original_h - y1) / ch,
                float(bw) * original_w / cw,
                float(bh) * original_h / ch)
    if ann[0] in ("obb", "segment"):
        flat = ann[2]
        mapped = []
        for i in range(0, len(flat), 2):
            mapped.extend(((float(flat[i]) * original_w - x1) / cw,
                           (float(flat[i + 1]) * original_h - y1) / ch))
        return (ann[0], cls_id, mapped)
    raise ValueError("Неподдерживаемый тип аннотации")


def mask_roi_at_image_scale(mask: np.ndarray, image_w: int, image_h: int,
                            bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """Извлекает маску точно для ROI без ресайза всей маски к размеру снимка.

    Если маска меньше снимка, INTER_NEAREST соответствует выборке
    floor(x * mask_width / image_width) по каждой оси.
    """
    if mask is None or mask.ndim != 2 or image_w < 1 or image_h < 1:
        raise ValueError("Некорректные входные данные маски")
    x1, y1, x2, y2 = bounds
    if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h or x2 <= x1 or y2 <= y1:
        raise ValueError("Некорректные границы ROI")
    mh, mw = mask.shape
    if (mw, mh) == (image_w, image_h):
        return mask[y1:y2, x1:x2]
    xx = np.minimum(mw - 1, np.floor(np.arange(x1, x2) * mw / image_w).astype(np.intp))
    yy = np.minimum(mh - 1, np.floor(np.arange(y1, y2) * mh / image_h).astype(np.intp))
    return mask[np.ix_(yy, xx)]


def apply_class_mask_overlay(image: np.ndarray, mask: Optional[np.ndarray],
                             opacity: float, color_for_class,
                             *, pixel_offset: int = 1, background_value: int = 0,
                             annotations=None, restrict_to_annotations=False):
    """Показывает индексную маску цветами палитры классов (BGR).

    Без ``annotations`` действует прежний режим: class_id = value - pixel_offset.
    Если геометрия аннотаций известна, её class_id имеет приоритет над
    значением пикселя маски ВНУТРИ полигона объекта. Это исправляет
    binary/instance masks (255 = объект, а не class 254) и гарантирует
    соответствие цвета маски и рамки данного объекта.

    Фон (background_value) всегда прозрачен. При перекрытии объектов
    приоритет получает более специфическая форма и меньший объект.
    При restrict_to_annotations=True маска рисуется ТОЛЬКО внутри
    геометрии переданных аннотаций. Остальные пиксели не окрашиваются,
    независимо от их значения в исходном файле маски. Если аннотаций нет,
    можно сохранить прежнее индексное наложение для mask-only датасетов,
    передав restrict_to_annotations=False.

    Исходные image/mask не изменяются; alpha применяется единожды.
    """
    if image is None or mask is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    if mask.ndim != 2:
        raise ValueError("Маска должна быть одномерным классом на пиксель (H,W)")
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    alpha = max(0.0, min(1.0, float(opacity)))
    if alpha == 0.0:
        return image.copy()
    result = image.copy()
    valid_mask = mask != background_value
    covered = None
    if annotations:
        # Сначала переносим на маску class_id аннотаций. Изображение
        # смешивается с цветом только один раз; позиции без геометрической
        # привязки ниже используют стандартное значение mask label.
        shapes = []
        priority = {'detect': 0, 'obb': 1, 'segment': 2}
        for ann in annotations:
            if not ann:
                continue
            points = polygon_for_annotation(ann, w, h)
            if len(points) < 3:
                continue
            area = abs(float(cv2.contourArea(points.astype(np.float32))))
            shapes.append((priority.get(ann[0], -1), -area, ann, points))
        shapes.sort(key=lambda item: (item[0], item[1]))
        if shapes:
            covered = np.zeros((h, w), dtype=bool)
        for _priority, _neg_area, ann, points in shapes:
            x1 = max(0, int(math.floor(float(points[:, 0].min()))))
            y1 = max(0, int(math.floor(float(points[:, 1].min()))))
            x2 = min(w, int(math.ceil(float(points[:, 0].max()))) + 1)
            y2 = min(h, int(math.ceil(float(points[:, 1].max()))) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            pts = np.rint(points - np.array([x1, y1])).astype(np.int32)
            cv2.fillPoly(local, [pts], 1)
            region = local.astype(bool) & valid_mask[y1:y2, x1:x2]
            if not region.any():
                continue
            src_roi = image[y1:y2, x1:x2]
            dst_roi = result[y1:y2, x1:x2]
            bgr = np.asarray(color_for_class(class_id_for_annotation(ann)),
                             dtype=np.float32)
            dst_roi[region] = ((1 - alpha) * src_roi[region].astype(np.float32)
                               + alpha * bgr).round().clip(0, 255).astype(np.uint8)
            covered[y1:y2, x1:x2][region] = True

    # При просмотре размеченного датасета пиксели вне геометрии аннотаций
    # должны оставаться исходными, даже если маска ошибочно содержит
    # ненулевой индекс (например 4/5 -> класс 4).
    if restrict_to_annotations:
        return result

    # Legacy indexed/mask-only mode (no geometry to restrict to).
    for pixel_value in np.unique(mask):
        label = int(pixel_value)
        if label == background_value:
            continue
        region = mask == pixel_value
        if covered is not None:
            region &= ~covered
        if not region.any():
            continue
        cls = max(0, label - int(pixel_offset))
        bgr = np.asarray(color_for_class(cls), dtype=np.float32)
        result[region] = ((1 - alpha) * image[region].astype(np.float32)
                          + alpha * bgr).round().clip(0, 255).astype(np.uint8)
    return result


def _shapely_polygon(points):
    if _Polygon is None:
        return None
    try:
        poly = _Polygon(points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty and poly.area > 0 else None
    except Exception:
        return None


def _raster_intersection_area(a: np.ndarray, b: np.ndarray,
                              rect: Tuple[float, float, float, float],
                              max_pixels=2_000_000) -> float:
    """Fallback для невыпуклых полигонов без shapely.

    Для больших областей используется уменьшенный временный растр —
    это влияет только на оценку пересечения, не на качество снимка.
    """
    x1, y1, x2, y2 = rect
    ww = max(1.0, x2 - x1)
    hh = max(1.0, y2 - y1)
    scale = min(1.0, math.sqrt(float(max_pixels) / (ww * hh)))
    width = max(1, int(math.ceil(ww * scale)) + 3)
    height = max(1, int(math.ceil(hh * scale)) + 3)
    # scale+shift: одна система координат для обоих объектов
    aa = np.round((a - np.array([x1, y1])) * scale).astype(np.int32)
    bb = np.round((b - np.array([x1, y1])) * scale).astype(np.int32)
    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros_like(mask_a)
    cv2.fillPoly(mask_a, [aa], 1)
    cv2.fillPoly(mask_b, [bb], 1)
    return float(np.count_nonzero(mask_a & mask_b)) / (scale * scale)


def overlap_ratio_of_smaller(annotation_a, annotation_b,
                             width: int, height: int) -> float:
    """Пересечение / площадь МЕНЬШЕГО объекта, в интервале [0,1].

    Такой показатель выявляет полностью вложенные дубликаты: их IoU может
    быть маленьким при сильном несоответствии размеров.
    """
    a = polygon_for_annotation(annotation_a, width, height)
    b = polygon_for_annotation(annotation_b, width, height)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    abox = (a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max())
    bbox = (b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max())
    ix1 = max(float(abox[0]), float(bbox[0]))
    iy1 = max(float(abox[1]), float(bbox[1]))
    ix2 = min(float(abox[2]), float(bbox[2]))
    iy2 = min(float(abox[3]), float(bbox[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    pa = _shapely_polygon(a)
    pb = _shapely_polygon(b)
    if pa is not None and pb is not None:
        small = min(pa.area, pb.area)
        return max(0.0, min(1.0, float(pa.intersection(pb).area) / small)) if small else 0.0

    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aarea = float(abs(cv2.contourArea(aa)))
    barea = float(abs(cv2.contourArea(bb)))
    small = min(aarea, barea)
    if small <= 1e-8:
        return 0.0
    # Для прямоугольников и любых выпуклых полигонов — точная геометрия OpenCV.
    if cv2.isContourConvex(aa) and cv2.isContourConvex(bb):
        try:
            area, _ = cv2.intersectConvexConvex(aa, bb)
            return max(0.0, min(1.0, float(area) / small))
        except cv2.error:
            pass
    # Вырезаем только bounding box пересечения.
    area = _raster_intersection_area(a, b, (ix1, iy1, ix2, iy2))
    return max(0.0, min(1.0, area / small))


def find_strong_overlaps(annotations, width: int, height: int,
                         threshold: float = 0.70) -> Dict[int, List[Tuple[int, float]]]:
    """Сопоставляет индексы аннотаций с перекрывающими их соседями.

    Красным подсвечиваются ОБА объекта пары, класс не влияет на проверку.
    """
    result = {}
    threshold = max(0.0, min(1.0, float(threshold)))
    # Sweep по горизонтальным габаритам исключает почти все удалённые пары,
    # особенно при большом числе небольших объектов на крупном снимке.
    rectangles = []
    for idx, ann in enumerate(annotations):
        p = polygon_for_annotation(ann, width, height)
        if len(p) >= 3:
            rectangles.append((float(p[:, 0].min()), float(p[:, 1].min()),
                               float(p[:, 0].max()), float(p[:, 1].max()), idx))
    rectangles.sort(key=lambda e: e[0])
    for n, (x1, y1, x2, y2, i) in enumerate(rectangles):
        for ax1, ay1, ax2, ay2, j in rectangles[n + 1:]:
            if ax1 >= x2:
                break
            if ay1 >= y2 or ay2 <= y1:
                continue
            score = overlap_ratio_of_smaller(annotations[i], annotations[j], width, height)
            if score + 1e-8 >= threshold:
                result.setdefault(i, []).append((j, score))
                result.setdefault(j, []).append((i, score))
    return result


# ============================================================================
# Shared YOLO + dataset viewer primitives (V6)
# ============================================================================

def detection_to_unified(detection, image_w: int, image_h: int):
    """Return one normalized annotation for a YOLO+AE/VAE detection dict.

    Unlike the old YOLO controller conversion, class IDs stay numeric. This
    permits one global editable dataset class palette in both Qt viewers.
    """
    if not detection or image_w <= 0 or image_h <= 0:
        return None
    kind = str(detection.get('kind', 'detect')).lower()
    try:
        cls_id = max(0, int(detection.get('cls_id', 0)))
    except (TypeError, ValueError, OverflowError):
        cls_id = 0
    if kind == 'detect':
        try:
            x1, y1, x2, y2 = map(float, detection['xyxy'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite((x1, y1, x2, y2)).all() or x2 <= x1 or y2 <= y1:
            return None
        return ('detect', cls_id, (x1 + x2) / (2 * image_w),
                (y1 + y2) / (2 * image_h),
                (x2 - x1) / image_w, (y2 - y1) / image_h)
    if kind == 'obb':
        points = detection.get('corners', [])
        typ = 'obb'
    elif kind in ('seg', 'segment'):
        points = detection.get('mask_xy') or detection.get('corners', [])
        typ = 'segment'
    else:
        return None
    try:
        pts = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    except (ValueError, TypeError):
        return None
    if not np.isfinite(pts).all() or len(pts) < (4 if typ == 'obb' else 3):
        return None
    if typ == 'obb' and len(pts) != 4:
        return None
    pts = pts.copy()
    pts[:, 0] /= image_w
    pts[:, 1] /= image_h
    return (typ, cls_id, pts.reshape(-1).tolist())


def detections_to_unified(detections, image_w: int, image_h: int):
    """Convert all detections, keeping list positions stable for Qt selection.

    Invalid polygons fall back to their AABB when possible; entries which
    cannot be represented remain None (caller should skip those indices).
    """
    result = []
    for det in detections:
        annotation = detection_to_unified(det, image_w, image_h)
        if annotation is None and isinstance(det, dict) and 'xyxy' in det:
            annotation = detection_to_unified(
                {**det, 'kind': 'detect'}, image_w, image_h)
        result.append(annotation)
    return result


def _name_for_class(names, class_id):
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _as_numpy(value):
    """Accept Ultralytics tensors and plain numpy arrays for standalone tests."""
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def yolo_result_to_detections(result, names=None, *, angle_fn=None):
    """Decode YOLO detect/OBB/segment into the existing AE/VAE dict contract.

    Preserves `kind`, `cls_id`, `cls_name`, `conf`, `xyxy`, `corners`,
    `mask_xy`, `angle_deg`, and `center` used by yolo_test_aeb.py.
    """
    if result is None:
        return []
    detections = []
    obb = getattr(result, 'obb', None)
    if (obb is not None and getattr(obb, 'xyxyxyxy', None) is not None
            and len(obb) > 0):
        corners_all = _as_numpy(obb.xyxyxyxy)
        cls_all = _as_numpy(obb.cls).astype(int)
        conf_all = _as_numpy(obb.conf)
        for corners, cls_id, confidence in zip(corners_all, cls_all, conf_all):
            pts = np.asarray(corners, dtype=float).reshape((-1, 2))
            if pts.shape != (4, 2):
                continue
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            center = pts.mean(axis=0)
            pts_list = pts.tolist()
            angle = (angle_fn(pts_list) if angle_fn is not None else None)
            detections.append({
                'kind': 'obb', 'cls_id': int(cls_id),
                'cls_name': _name_for_class(names, int(cls_id)),
                'conf': float(confidence),
                'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                'corners': pts_list, 'angle_deg': angle,
                'center': (float(center[0]), float(center[1])),
            })
    elif (getattr(result, 'masks', None) is not None
          and getattr(result.masks, 'xy', None) is not None
          and len(result.masks.xy) > 0):
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy_all = _as_numpy(boxes.xyxy)
        cls_all = _as_numpy(boxes.cls).astype(int)
        conf_all = _as_numpy(boxes.conf)
        for poly, xyxy, cls_id, confidence in zip(
                result.masks.xy, xyxy_all, cls_all, conf_all):
            if poly is None or len(poly) < 3:
                continue
            pts = np.asarray(_as_numpy(poly), dtype=np.float32).reshape((-1, 1, 2))
            rect = cv2.minAreaRect(pts)
            corner_list = cv2.boxPoints(rect).tolist()
            angle = (angle_fn(corner_list) if angle_fn is not None else None)
            detections.append({
                'kind': 'seg', 'cls_id': int(cls_id),
                'cls_name': _name_for_class(names, int(cls_id)),
                'conf': float(confidence),
                'xyxy': [float(v) for v in xyxy],
                'corners': corner_list,
                'mask_xy': pts.reshape(-1, 2).tolist(),
                'angle_deg': angle,
                'center': (float(rect[0][0]), float(rect[0][1])),
            })
    else:
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy_all = _as_numpy(boxes.xyxy)
        cls_all = _as_numpy(boxes.cls).astype(int)
        conf_all = _as_numpy(boxes.conf)
        for xyxy, cls_id, confidence in zip(xyxy_all, cls_all, conf_all):
            x1, y1, x2, y2 = map(float, xyxy)
            if x2 <= x1 or y2 <= y1:
                continue
            angle = 0.0 if (x2 - x1) >= (y2 - y1) else 90.0
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            detections.append({
                'kind': 'detect', 'cls_id': int(cls_id),
                'cls_name': _name_for_class(names, int(cls_id)),
                'conf': float(confidence),
                'xyxy': [x1, y1, x2, y2],
                'corners': corners, 'angle_deg': angle,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
            })
    return sorted(detections, key=lambda d: -d['conf'])


def prepare_annotation_preview(image, annotation, padding=10, *, mask=None,
                               opacity=0.5, color_for_class=None,
                               pixel_offset=1, background_value=0):
    """Single crop pipeline used by the dataset viewer and YOLO-side previews.

    No resize, angle correction, or positional offsets. The return tuple is
    (crop BGR, localized annotation, xyxy bounds). Any mask is sampled only
    over the crop's original-pixel ROI.
    """
    crop, bounds = crop_for_annotation(image, annotation, padding=padding)
    if crop is None:
        return None, None, None
    h, w = image.shape[:2]
    crop_annotation = annotation_in_crop(annotation, w, h, bounds)
    if mask is not None and color_for_class is not None:
        mask_roi = mask_roi_at_image_scale(mask, w, h, bounds)
        crop = apply_class_mask_overlay(
            crop, mask_roi, opacity, color_for_class,
            pixel_offset=pixel_offset, background_value=background_value,
            annotations=[crop_annotation],
            restrict_to_annotations=True,
        )
    return crop, crop_annotation, bounds


def render_unified_annotations(image, annotations, color_for_class, *,
                               selected_index=-1, selected_color=(0, 255, 255),
                               text_color=(255, 255, 255), thickness=1,
                               font_scale=0.5):
    """Optional OpenCV renderer for exporting detection imagery.

    Interactive Qt views should use SmartGraphicsViewNew overlays instead.
    Input colors are BGR, as in SettingsManager.
    """
    if image is None:
        return None
    out = image.copy()
    h, w = out.shape[:2]
    for idx, ann in enumerate(annotations):
        if ann is None:
            continue
        pts = polygon_for_annotation(ann, w, h)
        if len(pts) < 3:
            continue
        color = (selected_color if idx == selected_index else
                 color_for_class(class_id_for_annotation(ann)))
        pixels = np.round(pts).astype(np.int32)
        cv2.polylines(out, [pixels], True, tuple(map(int, color)),
                      max(1, int(thickness)), cv2.LINE_AA)
        pos = pixels[np.argmin(pixels[:, 1])]
        cv2.putText(out, str(class_id_for_annotation(ann)),
                    (int(pos[0]), max(12, int(pos[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, float(font_scale),
                    tuple(map(int, text_color)), 1, cv2.LINE_AA)
    return out


def render_ae_error_map(error, base, *, valid_region=None, overlay=True,
                        alpha=0.5, show_mask=False, threshold=float('nan')):
    """Render an AE/VAE error heatmap with the original percentile stretch.

    Neural inference and torch pre-processing are kept in the YOLO+AE tab;
    this routine only handles NumPy/OpenCV post-processing and visualization.
    """
    if error is None or base is None:
        return None
    h, w = base.shape[:2]
    if valid_region is not None:
        x1, y1, x2, y2 = valid_region
        roi = error[y1:y2, x1:x2]
        if roi.size == 0:
            roi = error
    else:
        roi = error
    full = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(np.percentile(full, 1)), float(np.percentile(full, 99))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    norm = np.clip((full - lo) / (hi - lo), 0., 1.)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if overlay:
        a = max(0.0, min(1.0, float(alpha)))
        image = cv2.addWeighted(base, 1 - a, heat, a, 0)
    else:
        image = heat
    if show_mask and np.isfinite(threshold):
        bw = (roi > threshold).astype(np.uint8)
        full_bw = cv2.resize(bw, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            full_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
    return image


# ===========================================================================
# Unified geometry + image manipulation for YOLO detections and saved labels
# ===========================================================================

# ============================================================================
# Проверенные алгоритмы из utils_aeb/yolo_aeb_utils.py (перенос без изменения)
# Внимание: публичные сигнатуры и пиксельная семантика сохранены.
# ============================================================================

def longest_edge_angle_deg(corners):
    n = len(corners)
    best_len = -1.0
    best_angle = 0.0
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L > best_len:
            best_len = L
            best_angle = math.degrees(math.atan2(dy, dx))
    return best_angle


def build_detection_mask(det, W, H):
    """
    Бинарная маска (uint8, 0/255) для детекции.
    Для seg используется mask_xy (точный контур), иначе — corners.
    Возвращает None, если контур некорректен.
    """
    if det is None:
        return None
    if det.get("kind") == "seg" and det.get("mask_xy"):
        poly = np.asarray(det["mask_xy"], dtype=np.float32).reshape(-1, 2)
    else:
        poly = np.asarray(det["corners"], dtype=np.float32).reshape(-1, 2)

    if poly.shape[0] < 3:
        return None

    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def morph_image_in_mask(img, mask, erosion_px, dilation_px,
                        shape_name="Rectangle", contour_only=True):
    """
    Применяет эрозию и/или дилатацию к изображению, сохраняя результат
    только внутри маски (вне маски — исходный img).

    Если contour_only=True — результат сохраняется не по всей площади
    маски, а только в граничной полосе (ring), толщина полосы привязана
    к размеру ядра операции.
    """
    if shape_name == "Rectangle":
        shape = cv2.MORPH_RECT
    elif shape_name == "Ellipse":
        shape = cv2.MORPH_ELLIPSE
    else:
        shape = cv2.MORPH_CROSS

    # -------- выбор эффективной маски: полная площадь либо контур --------
    effective_mask = mask
    if contour_only:
        ring_px = max(int(erosion_px), int(dilation_px), 3)
        if ring_px % 2 == 0:
            ring_px += 1
        ring_px = max(3, ring_px)
        ring_kernel = cv2.getStructuringElement(shape, (ring_px, ring_px))
        eroded_mask = cv2.erode(mask, ring_kernel, iterations=1)
        ring_mask = cv2.subtract(mask, eroded_mask)
        if ring_mask.any():
            effective_mask = ring_mask
        # если маска слишком тонкая и кольцо пустое — работаем по всей маске

    if img.ndim == 3:
        mask_bool = effective_mask[..., None] > 0
    else:
        mask_bool = effective_mask > 0

    if erosion_px > 1:
        k = erosion_px if erosion_px % 2 == 1 else erosion_px + 1
        k = max(3, k)
        kernel = cv2.getStructuringElement(shape, (k, k))
        eroded = cv2.erode(img, kernel, iterations=1)
        img = np.where(mask_bool, eroded, img)

    if dilation_px > 1:
        k = dilation_px if dilation_px % 2 == 1 else dilation_px + 1
        k = max(3, k)
        kernel = cv2.getStructuringElement(shape, (k, k))
        dilated = cv2.dilate(img, kernel, iterations=1)
        img = np.where(mask_bool, dilated, img)

    return img


def apply_object_morphology(img, det, erosion_px=1, dilation_px=1,
                            shape_name="Rectangle", contour_only=True,
                            *, operations=None, contour_width_px=None):
    """Морфология изображения внутри выбранной YOLO-детекции.

    ``operations`` — список MorphologyStep/dict из общего виджета,
    порядок списка строго соблюдается. Старые erosion_px/dilation_px
    сохранены для совместимости API со старыми вызывающими модулями.
    Ограничение по объекту происходит *до* кропа/доворота, как и ранее.
    """
    if operations is None:
        steps = [MorphologyStep("erode", int(erosion_px)),
                 MorphologyStep("dilate", int(dilation_px))]
    else:
        steps = list(operations)

    # Выключенные шаги не должны менять исходное изображение.
    enabled = [x for x in steps if int(x.kernel_px if isinstance(x, MorphologyStep)
                                      else x["kernel_px"]) > 1]
    if not enabled or det is None:
        return img, None

    H, W = img.shape[:2]
    mask = build_detection_mask(det, W, H)
    if mask is None or not mask.any():
        return img, None

    output = apply_shared_morphology(
        img, steps, shape=shape_name, mask=mask,
        contour_only=bool(contour_only), contour_width_px=contour_width_px,
    )
    info = {
        "kind": det.get("kind"),
        "cls_name": det.get("cls_name"),
        "contour_mode": "contour" if contour_only else "full",
        "erosion": next((int(x.kernel_px if isinstance(x, MorphologyStep)
                              else x["kernel_px"]) for x in steps
                         if (x.operation if isinstance(x, MorphologyStep)
                             else x["operation"]) == "erode"), 0),
        "dilation": next((int(x.kernel_px if isinstance(x, MorphologyStep)
                               else x["kernel_px"]) for x in steps
                          if (x.operation if isinstance(x, MorphologyStep)
                              else x["operation"]) == "dilate"), 0),
        "operations": [
            f"{x.operation if isinstance(x, MorphologyStep) else x['operation']}="
            f"{x.kernel_px if isinstance(x, MorphologyStep) else x['kernel_px']}px"
            for x in steps if int(x.kernel_px if isinstance(x, MorphologyStep)
                                  else x["kernel_px"]) > 1
        ],
        "mask_px": int((mask > 0).sum()),
    }
    return output, info


def fill_outside_object(img, det, fill_mode,
                        rotation_matrix=None, crop_offset=(0, 0)):
    """
    Заливает пиксели вне маски объекта (fill_mode: 'none' / 'black' / 'white').
    rotation_matrix и crop_offset позволяют применить маску к уже
    повёрнутому/обрезанному изображению.
    """
    if fill_mode == "none":
        return img
    color_val = 0 if fill_mode == "black" else 255

    H, W = img.shape[:2]

    if det.get("kind") == "seg" and det.get("mask_xy"):
        poly = np.asarray(det["mask_xy"], dtype=np.float32).reshape(-1, 2)
    else:
        poly = np.asarray(det["corners"], dtype=np.float32).reshape(-1, 2)

    if poly.shape[0] < 3:
        return img

    if rotation_matrix is not None:
        ones = np.ones((poly.shape[0], 1), dtype=np.float32)
        poly_h = np.hstack([poly, ones])
        poly = (rotation_matrix @ poly_h.T).T

    ox, oy = crop_offset
    poly = poly.copy()
    poly[:, 0] -= float(ox)
    poly[:, 1] -= float(oy)

    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)

    outside = (mask == 0)
    if img.ndim == 2:
        img[outside] = color_val
    else:
        img[outside] = (color_val,) * img.shape[2]
    return img


def crop_to_detection(img, det, padding, off_x, off_y, fill_mode):
    """Обрезает изображение по xyxy объекта с учётом padding и смещений."""
    x1, y1, x2, y2 = det["xyxy"]
    H, W = img.shape[:2]
    xi1 = max(0, int(round(x1)) - padding + off_x)
    yi1 = max(0, int(round(y1)) - padding + off_y)
    xi2 = min(W, int(round(x2)) + padding + off_x)
    yi2 = min(H, int(round(y2)) + padding + off_y)
    if xi2 <= xi1 or yi2 <= yi1:
        return img
    cropped = img[yi1:yi2, xi1:xi2].copy()
    cropped = fill_outside_object(
        cropped, det, fill_mode, rotation_matrix=None, crop_offset=(xi1, yi1)
    )
    return cropped


def rotate_with_optional_crop(img, det, auto_crop, padding, off_x, off_y, fill_mode):
    """
    Доворачивает изображение так, чтобы длинная сторона объекта стала
    горизонтальной. Если auto_crop=True — обрезает по границам объекта
    с учётом padding и смещений.
    """
    angle_deg = det["angle_deg"]
    cx, cy = det["center"]
    H, W = img.shape[:2]

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(H * sin_a + W * cos_a)
    new_h = int(H * cos_a + W * sin_a)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    if auto_crop:
        corners = np.array(det["corners"], dtype=np.float32)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        new_corners = (M @ corners_h.T).T
        x1n = int(np.floor(new_corners[:, 0].min())) - padding + off_x
        y1n = int(np.floor(new_corners[:, 1].min())) - padding + off_y
        x2n = int(np.ceil(new_corners[:, 0].max())) + padding + off_x
        y2n = int(np.ceil(new_corners[:, 1].max())) + padding + off_y
        xi1 = max(0, x1n); yi1 = max(0, y1n)
        xi2 = min(rotated.shape[1], x2n)
        yi2 = min(rotated.shape[0], y2n)
        if xi2 > xi1 and yi2 > yi1:
            rotated = rotated[yi1:yi2, xi1:xi2].copy()
            rotated = fill_outside_object(
                rotated, det, fill_mode, rotation_matrix=M, crop_offset=(xi1, yi1)
            )
    return rotated


def cls_suffix_from_detection(det, index=None):
    """
    Безопасный суффикс имени файла по классу объекта и его номеру в списке.

    index — 1-based позиция объекта в текущем списке детекций (может быть None).
    Итоговый вид: "ClassName_N" или просто "ClassName", если index не задан
    либо <= 0. Пустая строка, если объект/имя класса не заданы.
    """
    if det is None:
        return ""
    cls_name = det.get("cls_name")
    if not cls_name:
        return ""
    safe = "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in str(cls_name)
    ).strip("_")
    if not safe:
        return ""
    if index is not None and int(index) > 0:
        return f"{safe}_{int(index)}"
    return safe


def limit_image_size(img, max_side):
    """Пропорционально уменьшает изображение, если длинная сторона > max_side."""
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def aeb_transform_image(image, detection, *, erode=1, dilate=1,
                        morphology_steps=None, contour_width_px=None,
                        shape_name='Rectangle', contour_only=True,
                        auto_crop=False, auto_rotate=False, padding=0,
                        off_x=0, off_y=0, fine_angle=0.0, fill_mode='none',
                        manual_ops=(), invert=False):
    """Common sequential image pipeline for live preview and save-all.

    Every annotation operation is implemented in this module. Output is
    (processed_image, mode, morphology_info).
    """
    if image is None:
        return None, 'empty', None
    img = image.copy()
    mode = 'original'
    img, morph_info = apply_object_morphology(
        img, detection, erode, dilate,
        shape_name=shape_name, contour_only=contour_only,
        operations=morphology_steps, contour_width_px=contour_width_px,
    )
    if detection is not None and (auto_rotate or auto_crop):
        if auto_rotate and detection.get('angle_deg') is not None:
            img = rotate_with_optional_crop(img, detection, auto_crop=bool(auto_crop),
                         padding=int(padding), off_x=int(off_x),
                         off_y=int(off_y), fill_mode=fill_mode)
            mode = 'rotated'
        elif auto_crop:
            img = crop_to_detection(img, detection, padding=int(padding),
                       off_x=int(off_x), off_y=int(off_y), fill_mode=fill_mode)
            mode = 'cropped'
    if abs(float(fine_angle)) > 1e-6:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(fine_angle), 1.)
        border = (cv2.BORDER_CONSTANT if fill_mode in ('black', 'white')
                  else cv2.BORDER_REPLICATE)
        val = 255 if fill_mode == 'white' else 0
        border_color = ((val,) * img.shape[2] if img.ndim == 3 else val)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=border, borderValue=border_color)
        if mode == 'original':
            mode = 'fine_rotated'
    for op in manual_ops:
        if op == 'rot_cw':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif op == 'rot_ccw':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif op == 'flip_v':
            img = cv2.flip(img, 0)
        elif op == 'flip_h':
            img = cv2.flip(img, 1)
        else:
            continue
        mode = 'manual'
    if invert:
        img = cv2.bitwise_not(img)
    return img, mode, morph_info



# ============================================================================
# Adapter: file labels <-> YOLO detection dictionaries
# ============================================================================

def annotation_to_detection(annotation, image_w, image_h, *, class_name=None):
    """Build an AEB-compatible detection dict from normalized YOLO annotation.

    Supports detect, obb, segment; missing class is class 0. All geometry
    uses full image pixels. The legacy AEB algorithms can therefore also
    operate on a saved dataset label without requiring a YOLO model.
    """
    ann = normalized_annotation(annotation)
    points = polygon_for_annotation(ann, image_w, image_h)
    if len(points) < 3:
        return None
    kind = ann[0]
    cls_id = class_id_for_annotation(ann)
    if kind == 'segment':
        pts_cv = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts_cv)
        corners = cv2.boxPoints(rect).tolist()
        center = (float(rect[0][0]), float(rect[0][1]))
        angle = longest_edge_angle_deg(corners)
    else:
        corners = points.tolist()
        center = tuple(map(float, np.asarray(points).mean(axis=0)))
        if kind == 'obb':
            angle = longest_edge_angle_deg(corners)
        else:
            edges = np.asarray(points)
            bbox_w = float(edges[:, 0].max() - edges[:, 0].min())
            bbox_h = float(edges[:, 1].max() - edges[:, 1].min())
            angle = 0.0 if bbox_w >= bbox_h else 90.0
    xs,ys = points[:,0], points[:,1]
    result = {
        'kind': 'seg' if kind == 'segment' else kind,
        'cls_id': cls_id,
        'cls_name': str(cls_id) if class_name is None else str(class_name),
        'xyxy': [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        'corners': corners,
        'center': center,
        'angle_deg': angle,
    }
    if kind == 'segment':
        result['mask_xy'] = points.tolist()
    return result


def transform_annotation_image(image, annotation, **kwargs):
    """Run the same AEB pipeline for a label from a saved dataset.

    Returns the same tuple as aeb_transform_image: image, mode, info.
    The underlying processing functions are shared with YOLO+AE/VAE.
    """
    if image is None:
        return None, 'empty', None
    h, w = image.shape[:2]
    detection = annotation_to_detection(annotation, w, h)
    return aeb_transform_image(image, detection, **kwargs)


# ---------------------------------------------------------------------------
# Извлечение YOLO-аннотаций из бинарного изображения (Threshold Methods).
# ---------------------------------------------------------------------------

def _mask_extraction_dimensions(binary: np.ndarray, image_shape=None, min_area=None):
    """Площадь отсечения — та же, что в старом get_display_params().

    img_shape используется для нормализации/отсечения и обычно совпадает
    с binary.shape. Для миграции допускается явное значение min_area.
    """
    if binary is None or binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("Требуется одноканальная бинарная маска uint8")
    height, width = binary.shape[:2]
    if image_shape is not None:
        img_h, img_w = int(image_shape[0]), int(image_shape[1])
        if (img_h, img_w) != (height, width):
            raise ValueError("Размеры бинарной маски и изображения должны совпадать")
    if min_area is None:
        min_area = max(1, int(0.0005 * height * width))
    return height, width, min_area


def extract_segments_from_mask(binary: np.ndarray, *, image_shape=None,
                               use_hull=False, min_area=None, class_id=0):
    """Segment: внешние контуры CHAIN_APPROX_SIMPLE, нормализованные вершины.

    Сохраняется исходный порядок OpenCV findContours. Отсечение — по
    площади исходного контура ДО cv2.convexHull (как в старой версии).
    """
    height, width, area_limit = _mask_extraction_dimensions(
        binary, image_shape=image_shape, min_area=min_area)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < area_limit:
            continue
        if use_hull:
            cnt = cv2.convexHull(cnt)
        pts = cnt.squeeze().astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        points = []
        for pt in pts:
            points.append(pt[0] / width)
            points.append(pt[1] / height)
        objects.append(('segment', class_id, points))
    return objects


def extract_boxes_from_mask(binary: np.ndarray, *, image_shape=None,
                            min_area=None, class_id=0):
    """Detect: connectedComponentsWithStats, 8-связность, как в старом коде."""
    height, width, area_limit = _mask_extraction_dimensions(
        binary, image_shape=image_shape, min_area=min_area)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    objects = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < area_limit:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        cx = (x + w_box / 2) / width
        cy = (y + h_box / 2) / height
        wn = w_box / width
        hn = h_box / height
        objects.append(('detect', class_id, cx, cy, wn, hn))
    return objects


def extract_obb_from_mask(binary: np.ndarray, *, image_shape=None,
                          use_hull=False, min_area=None, class_id=0):
    """OBB: minAreaRect + boxPoints, с исходным усечением до int32."""
    height, width, area_limit = _mask_extraction_dimensions(
        binary, image_shape=image_shape, min_area=min_area)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < area_limit:
            continue
        if use_hull:
            cnt = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        points = []
        for pt in box:
            points.append(pt[0] / width)
            points.append(pt[1] / height)
        objects.append(('obb', class_id, points))
    return objects


def extract_annotations_from_mask(binary: np.ndarray, mode='detect', *,
                                  image_shape=None, use_hull=False,
                                  min_area=None, class_id=0):
    """Единая точка получения Detect/OBB/Segment из бинарной маски.

    Принимает значения mode из интерфейса Threshold Methods или короткие
    названия detect/obb/segment. Режим None возвращает пустой список.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in ('none', ''):
        return []
    common = dict(image_shape=image_shape, min_area=min_area, class_id=class_id)
    if normalized_mode in ('detect', 'bounding box (detect)'):
        return extract_boxes_from_mask(binary, **common)
    if normalized_mode in ('obb', 'obb (oriented box)'):
        return extract_obb_from_mask(binary, use_hull=use_hull, **common)
    if normalized_mode in ('segment', 'segmentation (polygon)'):
        return extract_segments_from_mask(binary, use_hull=use_hull, **common)
    # В оригинальной вкладке неизвестный draw_mode возвращает [], без ошибки.
    return []


def format_annotation_display(index, ann, img_w, img_h):
    """Форматирует аннотацию для отображения в QListWidget.
       Поддерживает форматы ('detect',...), ('obb',...), ('segment',...)."""
    typ = ann[0]
    if typ == 'detect':
        _, cls, cx, cy, w, h = ann
        x_px = int((cx - w / 2) * img_w)
        y_px = int((cy - h / 2) * img_h)
        width_px = int(w * img_w)
        height_px = int(h * img_h)
        markup = f"Obj {index + 1}: rect x={x_px} y={y_px} w={width_px} h={height_px}"
        yolo = f"YOLO: {cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        return f"{markup} | {yolo}"
    elif typ == 'obb':
        _, cls, points = ann  # 8 координат
        pts_str = ' '.join(f"{p:.6f}" for p in points)
        return f"Obj {index + 1}: OBB | YOLO-OBB: {cls} {pts_str}"
    elif typ == 'segment':
        _, cls, points = ann
        num_pts = len(points) // 2
        if len(points) > 6:
            pts_preview = points[:6]
            preview = ' '.join(f"{p:.6f}" for p in pts_preview) + ' ...'
        else:
            preview = ' '.join(f"{p:.6f}" for p in points)
        return f"Obj {index + 1}: polygon {num_pts} pts | YOLO-seg: {cls} {preview}"
    else:
        return f"Obj {index + 1}: unknown type"

def format_object_for_list(index, obj, img_w, img_h):
    """Универсальное форматирование для старого стиля (используется в threshold_methods.py)."""
    if isinstance(obj, tuple) and len(obj) > 0 and obj[0] in ('detect', 'obb', 'segment'):
        return format_annotation_display(index, obj, img_w, img_h)
    elif len(obj) == 4:
        x, y, w, h = obj
        cx_norm = (x + w / 2) / img_w
        cy_norm = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        markup = f"obj {index + 1}: x={x}, y={y}, w={w}, h={h}"
        yolo = f"YOLO: 0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        return f"{markup} | {yolo}"
    elif len(obj) == 5:
        cx, cy, w, h, angle = obj
        return f"obj {index + 1}: center=({cx:.1f},{cy:.1f}), size=({w:.1f}x{h:.1f}), angle={angle:.1f}°"
    else:
        return f"obj {index + 1}: unknown"
