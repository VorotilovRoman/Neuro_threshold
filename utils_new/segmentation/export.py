"""Atomic export through the canonical YOLO serializer."""
import os
import tempfile
from pathlib import Path

from utils_new.dataset_yolo_labels_new import save_annotations


def write_annotations_atomic(annotations, destination, width, height, obb_as_xywhr=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.txt', delete=False) as stream:
            temporary = Path(stream.name)
        if not save_annotations(annotations, temporary, width, height, obb_as_xywhr):
            raise ValueError('Некорректные аннотации; исходный TXT не изменён')
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
