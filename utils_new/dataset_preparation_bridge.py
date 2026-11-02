# -*- coding: utf-8 -*-
"""Pure adapters between Dataset Preparation UI and DatasetGenerator_new.

The class with source mask pixel 0 is always background, never a trainable
class. Positive source pixels selected in the GUI become dense class ids.
This module deliberately has no Qt, filesystem or legacy utility imports.
"""
from __future__ import annotations

from typing import Iterable


def mask_classes_from_rows(rows: Iterable[tuple[int, str, bool]]):
    """Return (pixel->class_id, class_id->name) for selected positive pixels.

    ``rows`` consists of (source pixel value, class name, included).
    Dense IDs follow ascending source-pixel order, matching the legacy UI.
    Background pixel 0 is ignored regardless of its checkbox state.
    """
    selected = {}
    for pixel, name, included in rows:
        pixel = int(pixel)
        if pixel < 0:
            raise ValueError(f'Negative mask pixel value: {pixel}')
        if pixel == 0 or not included:
            continue
        if pixel in selected:
            raise ValueError(f'Duplicate source mask pixel: {pixel}')
        selected[pixel] = str(name).strip() or str(pixel)
    ordered = sorted(selected.items())
    mapping = {source_pixel: dense_id for dense_id, (source_pixel, _) in enumerate(ordered)}
    names = {dense_id: name for dense_id, (_, name) in enumerate(ordered)}
    return mapping, names


def normalize_generation_pairs(pairs, input_type):
    """Convert legacy (image, label-or-mask) to (image, label, mask)."""
    if input_type not in (0, 1, 2):
        raise ValueError(f'Invalid input type: {input_type}')
    result = []
    for pair in pairs:
        if not pair or not pair[0]:
            raise ValueError('Missing source image path')
        image = pair[0]
        aux = pair[1] if len(pair) > 1 else None
        if input_type == 0:
            result.append((image, aux, None))
        elif input_type == 1:
            result.append((image, None, aux))
        else:
            result.append((image, None, None))
    return result
