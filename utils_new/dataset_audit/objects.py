"""Strict, size-aware sample reading shared by filters, audits and export."""

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from utils_new.import_export.images import image_size_from_file
from utils_new.annotations import ImageSize, to_legacy
from utils_new.annotations.geometry import polygon_area
from utils_new.import_export.images import read_index_mask
from utils_new.import_export.yolo_labels import read_yolo_labels


@dataclass(frozen=True)
class DatasetObject:
    class_id: int
    area_px: float
    center: tuple
    annotation: tuple | None = None
    bounds: tuple | None = None


def image_size(path):
    return image_size_from_file(path)


def load_pair_annotations(image_path, label_path, format_hint="auto"):
    if not label_path or not Path(label_path).is_file():
        raise FileNotFoundError(f"Нет разметки: {label_path}")
    size = ImageSize(*image_size(image_path))
    messages = []
    annotations = read_yolo_labels(
        label_path, size, format_hint=format_hint, log=messages.append, strict=True
    )
    return [to_legacy(ann, size) for ann in annotations]


def read_objects(pair, dataset_type=0, input_format="auto"):
    w, h = image_size(pair[0])
    aux = pair[1] if len(pair) > 1 else None
    if dataset_type == 2:
        return (w, h), []
    if not aux or not Path(aux).is_file():
        raise FileNotFoundError(f"Нет разметки: {aux}")
    if dataset_type == 0:
        size = ImageSize(w, h)
        messages = []
        annotations = read_yolo_labels(
            aux, size, format_hint=input_format, log=messages.append, strict=True
        )
        result = []
        for ann in annotations:
            pts = ann.points_px
            moments = cv2.moments(pts.astype(np.float32))
            center = (
                np.array(
                    [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
                )
                / [w, h]
            ).tolist()
            result.append(
                DatasetObject(
                    ann.class_id,
                    polygon_area(pts),
                    tuple(center),
                    to_legacy(ann, size),
                    tuple(
                        np.concatenate((pts.min(axis=0), pts.max(axis=0)))
                        / [w, h, w, h]
                    ),
                )
            )
        return (w, h), result
    if dataset_type != 1:
        raise ValueError("Неизвестный вид датасета.")
    mask = read_index_mask(aux)
    if mask.shape != (h, w):
        raise ValueError(
            f"Размер маски {mask.shape} не совпадает с изображением {(h, w)}"
        )
    result = []
    # The loader class table uses raw positive mask values. Zero is background.
    for value in np.unique(mask):
        if value == 0:
            continue
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (mask == value).astype(np.uint8), connectivity=8
        )
        for i in range(1, count):
            result.append(
                DatasetObject(
                    int(value),
                    int(stats[i, cv2.CC_STAT_AREA]),
                    ((centers[i, 0] + 0.5) / w, (centers[i, 1] + 0.5) / h),
                    None,
                    (
                        stats[i, 0] / w,
                        stats[i, 1] / h,
                        (stats[i, 0] + stats[i, 2]) / w,
                        (stats[i, 1] + stats[i, 3]) / h,
                    ),
                )
            )
    return (w, h), result
