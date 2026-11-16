"""Explicit binary targets for traditional classifiers, never implicit pseudo-labels."""

from pathlib import Path

import numpy as np
from PIL import Image

from utils_new.annotations import ImageSize
from utils_new.annotations.masks import annotation_binary_mask
from utils_new.dataset_yolo_labels_new import read_yolo_labels


def manual_target(image_path, shape):
    """YOLO TXT next to image, or images/... -> labels/...; union of all objects.

    Empty TXT is valid background. Missing/ambiguous/malformed labels are errors.
    BBox/OBB are filled regions, not precise segmentation ground truth.
    """
    path = Path(image_path)
    candidates = [path.with_suffix(".txt")]
    for index, part in enumerate(path.parts):
        if part == "images":
            candidates.append(
                Path(
                    *path.parts[:index], "labels", *path.parts[index + 1 :]
                ).with_suffix(".txt")
            )
    found = {p.resolve() for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError(
            f"Для {path.name} нужен один YOLO TXT: найдено {len(found)}. Пустой TXT допустим; отсутствующий — нет."
        )
    with Image.open(path) as image:
        source = ImageSize(*image.size)
    messages = []
    annotations = read_yolo_labels(found.pop(), source, log=messages.append)
    if any("запись пропущена" in message for message in messages):
        raise ValueError(f"Некорректная ручная разметка {path.name}: {messages}")
    target = ImageSize(shape[1], shape[0])
    scale = np.array([target.width / source.width, target.height / source.height])
    result = np.zeros(shape, np.uint8)
    for annotation in annotations:
        scaled = annotation.with_points(annotation.points_px * scale)
        result |= annotation_binary_mask(scaled, target)
    return result
