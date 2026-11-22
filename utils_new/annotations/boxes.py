"""Shared bounding-box geometry for ensemble suppression and fusion. No Qt/ML."""

import numpy as np


def yolo_to_xyxy(cx, cy, w, h):
    """
    Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
    Возвращает координаты также в диапазоне 0..1.
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.clip([x1, y1, x2, y2], 0, 1)


def xyxy_to_yolo(x1, y1, x2, y2):
    """
    Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
    """
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


def iou_xyxy(box1, box2):
    """
    Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
    Возвращает значение в диапазоне [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union


def _thresholds(iou_threshold, conf_threshold):
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (iou_threshold, conf_threshold)):
        raise ValueError("Пороги IoU и оценки должны быть конечными числами 0..1.")


def non_max_suppression(boxes, scores, iou_threshold=0.5, conf_threshold=0.1):
    """Stable single-class NMS over xyxy boxes; callers separate class IDs."""
    _thresholds(iou_threshold, conf_threshold)
    if len(boxes) != len(scores):
        raise ValueError("Количество боксов и оценок различается.")
    order = sorted(
        (i for i, s in enumerate(scores) if s >= conf_threshold),
        key=lambda i: (-scores[i], i),
    )
    kept = []
    for index in order:
        if not any(
            (overlap := iou_xyxy(boxes[index], boxes[j])) > 0
            and overlap >= iou_threshold
            for j in kept
        ):
            kept.append(index)
    return [list(boxes[i]) for i in kept], [float(scores[i]) for i in kept]


def _contains_box(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def ensemble_detections(
    detections_list,
    weights,
    iou_threshold=0.5,
    conf_threshold=0.1,
    use_containment=False,
    method="wbf",
    verbose=False,
):
    """Fuse normalized Detect tuples per class, with explicit source weights.

    NMS ranks scores * (source weight / maximum active source weight).
    WBF coordinates use score*weight; confidence uses one vote per source and
    divides by the sum of *all* active source weights, including empty sources.
    Repeated detections from a source cannot inflate its vote. Outputs remain
    Detect (polygons must be converted to enclosing boxes by the caller).
    """
    _thresholds(iou_threshold, conf_threshold)
    if method not in ("nms", "wbf"):
        raise ValueError("Метод объединения: nms или wbf.")
    if len(weights) != len(detections_list) or any(
        not np.isfinite(w) or w < 0 for w in weights
    ):
        raise ValueError("Нужен один конечный неотрицательный вес на источник.")
    if not weights or max(weights) == 0:
        return []
    by_class = {}
    for source, (detections, weight) in enumerate(zip(detections_list, weights)):
        if weight == 0:
            continue
        for det in detections:
            if len(det) not in (5, 6) or not np.isfinite(det).all():
                raise ValueError(
                    "Детекция: (class,cx,cy,w,h[,score]), только конечные числа."
                )
            cls, cx, cy, w, h = det[:5]
            score = float(det[5]) if len(det) == 6 else 1.0
            if int(cls) != cls or cls < 0 or w <= 0 or h <= 0 or not 0 <= score <= 1:
                raise ValueError(
                    "Некорректный класс, размер бокса или оценка детекции."
                )
            box = np.asarray(yolo_to_xyxy(cx, cy, w, h), dtype=float)
            if score <= 0 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            by_class.setdefault(int(cls), []).append(
                (box, score, float(weight), source)
            )
    output = []
    for cls, records in sorted(by_class.items()):
        if method == "nms":
            boxes, scores = non_max_suppression(
                [r[0] for r in records],
                [r[1] * r[2] / max(weights) for r in records],
                iou_threshold,
                conf_threshold,
            )
        else:
            # First suppress duplicate votes from each source, retaining distinct objects.
            records = sorted(records, key=lambda r: -r[1] * r[2])
            unique = []
            for record in records:
                if any(
                    r[3] == record[3]
                    and (v := iou_xyxy(r[0], record[0])) > 0
                    and v >= iou_threshold
                    for r in unique
                ):
                    continue
                unique.append(record)
            groups = []
            for record in unique:
                box, score, weight, source = record
                best = None
                best_iou = -1.0
                for group in groups:
                    if source in group["sources"]:
                        continue
                    overlap = iou_xyxy(box, group["box"])
                    contained = use_containment and (
                        _contains_box(box, group["box"])
                        or _contains_box(group["box"], box)
                    )
                    if (
                        (overlap > 0 and overlap >= iou_threshold) or contained
                    ) and overlap > best_iou:
                        best, best_iou = group, overlap
                contribution = score * weight
                if best is None:
                    groups.append(
                        dict(
                            box=box.copy(),
                            sum_box=box * contribution,
                            mass=contribution,
                            sources={source},
                        )
                    )
                else:
                    best["sum_box"] += box * contribution
                    best["mass"] += contribution
                    best["sources"].add(source)
                    best["box"] = best["sum_box"] / best["mass"]
            boxes = []
            scores = []
            for group in groups:
                score = group["mass"] / sum(weights)
                if score >= conf_threshold:
                    boxes.append(group["box"])
                    scores.append(min(1.0, score))
        for box, score in zip(boxes, scores):
            output.append((cls, *map(float, xyxy_to_yolo(*box)), float(score)))
    output.sort(key=lambda det: (-det[-1], det[0], det[1], det[2]))
    if verbose:
        print(f"Ensemble {method}: sources={len(weights)}, output={len(output)}")
    return output
