"""Class-aware one-to-one matching and small-sample embedding projection."""

import numpy as np
from utils_new.annotations.metrics import maximum_iou_assignment
from utils_new.annotations.boxes import iou_xyxy


def match_detections(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold=0.3):
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    if len(gt_boxes) != len(gt_classes) or len(pred_boxes) != len(pred_classes):
        raise ValueError("Число классов не совпадает с числом боксов.")
    if not np.isfinite(gt_boxes).all() or not np.isfinite(pred_boxes).all():
        raise ValueError("Неконечные координаты боксов.")
    if not 0 <= threshold <= 1:
        raise ValueError("Порог IoU должен быть 0..1.")
    overlaps = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, a in enumerate(gt_boxes):
        for j, b in enumerate(pred_boxes):
            if int(gt_classes[i]) != int(pred_classes[j]):
                continue
            overlaps[i, j] = iou_xyxy(a, b)
    rows, cols = maximum_iou_assignment(overlaps, threshold)
    ious = np.zeros(len(gt_boxes), float)
    ious[rows] = overlaps[rows, cols]
    return {
        "ious": ious,
        "matched": len(rows),
        "false_negatives": len(gt_boxes) - len(rows),
        "false_positives": len(pred_boxes) - len(rows),
    }


def project_embeddings(values, prefer_umap=False, log=lambda text: None):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or not len(values)
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Эмбеддинги должны быть конечным непустым массивом N×D.")
    if len(values) == 1 or np.allclose(values, values[0]):
        return np.zeros((len(values), 2), float)
    if prefer_umap and len(values) >= 4:
        try:
            import umap

            return umap.UMAP(
                n_neighbors=min(15, len(values) - 1),
                min_dist=0.1,
                n_components=2,
                metric="cosine",
                random_state=42,
                init="random",
            ).fit_transform(values)
        except Exception as exc:
            log(f"UMAP: {exc}; используется PCA")
    from sklearn.decomposition import PCA

    projected = PCA(
        n_components=min(2, len(values), values.shape[1]), random_state=42
    ).fit_transform(values)
    return np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
