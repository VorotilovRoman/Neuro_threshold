"""Strict object geometry and maximum-cardinality one-to-one evaluation."""

import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from utils_new.annotations import ImageSize, from_legacy

QUALITY_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def parse_sequence(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("Некорректный JSON геометрии объектов.") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "Объекты должны быть списком; отсутствующая разметка не является пустой."
        )
    return value


def normalize_annotations(values, size):
    """Return class + normalized polygon, preserving physical source geometry."""
    result = []
    frame = ImageSize(*map(int, size))
    for value in parse_sequence(values):
        ann = from_legacy(value, frame)
        points = ann.points_px / [frame.width, frame.height]
        polygon = Polygon(points)
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Некорректная топология или нулевая площадь объекта.")
        result.append((ann.class_id, polygon))
    return result


def overlap_matrix(predictions, truth, *, class_aware=False):
    overlaps = np.zeros((len(predictions), len(truth)))
    for i, (cls, polygon) in enumerate(predictions):
        for j, (gt_cls, ideal) in enumerate(truth):
            if class_aware and cls != gt_cls:
                continue
            intersection = polygon.intersection(ideal).area
            union = polygon.area + ideal.area - intersection
            overlaps[i, j] = min(1.0, max(0.0, intersection / union)) if union else 0.0
    return overlaps


def maximum_iou_assignment(overlaps, threshold=0.5):
    """Maximize match count first, then total IoU; zero overlap is never a match."""
    if not 0 <= threshold <= 1:
        raise ValueError('IoU threshold must be 0..1')
    overlaps=np.asarray(overlaps,dtype=float)
    if overlaps.ndim!=2 or not np.isfinite(overlaps).all() or (overlaps<0).any() or (overlaps>1).any():
        raise ValueError('IoU matrix must be finite, 2D and in [0,1]')
    n_pred,n_true=overlaps.shape
    if not n_pred or not n_true:
        return np.array([],dtype=int),np.array([],dtype=int)
    valid=(overlaps>=threshold-1e-12)&(overlaps>0)
    rows,cols=linear_sum_assignment(valid*(min(n_pred,n_true)+1)+overlaps*valid,maximize=True)
    matched=valid[rows,cols]
    return rows[matched],cols[matched]


def match_overlaps(overlaps, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU должен быть больше 0 и не больше 1.')
    matrix=np.asarray(overlaps,dtype=float)
    rows,cols=maximum_iou_assignment(matrix,threshold)
    values=matrix[rows,cols]
    tp=len(values)
    return dict(tp=tp,fp=matrix.shape[0]-tp,fn=matrix.shape[1]-tp,
                matched_iou_sum=float(values.sum()),mean_iou=float(values.mean()) if tp else 0.)


def count_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    return dict(
        precision=precision,
        recall=recall,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    )


def evaluate_image(predictions, truth, *, threshold=0.5, class_aware=False):
    overlaps = overlap_matrix(predictions, truth, class_aware=class_aware)
    result = match_overlaps(overlaps, threshold)
    result.update(count_metrics(result["tp"], result["fp"], result["fn"]))
    curve = []
    for value in QUALITY_THRESHOLDS:
        counts = match_overlaps(overlaps, value)
        curve.append(count_metrics(counts["tp"], counts["fp"], counts["fn"])["f1"])
    result["mean_f1"] = float(np.mean(curve))
    result["true_objects"] = len(truth)
    result["predicted_objects"] = len(predictions)
    return result
