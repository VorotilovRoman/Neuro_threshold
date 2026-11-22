"""Atomic export through the canonical YOLO serializer."""
from utils_new.import_export.atomic import atomic_path

from utils_new.import_export.yolo_labels import save_annotations


def write_annotations_atomic(annotations, destination, width, height, obb_as_xywhr=False):
    with atomic_path(destination, suffix='.txt') as temporary:
        if not save_annotations(annotations, temporary, width, height, obb_as_xywhr):
            raise ValueError('Некорректные аннотации; исходный TXT не изменён')
