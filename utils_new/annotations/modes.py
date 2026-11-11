"""Annotation modes shared by segmentation and dataset annotation tools."""
from typing import Literal

AnnotationMode = Literal['none', 'detect', 'obb', 'segment']
MODE_LABELS = {'none': 'Нет', 'detect': 'BBox', 'obb': 'OBB', 'segment': 'Segment'}


def validate_mode(mode):
    if mode not in MODE_LABELS:
        raise ValueError(f'Неизвестный режим разметки: {mode!r}')
    return mode


def supports_hull(mode):
    return validate_mode(mode) in ('obb', 'segment')
