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
            raise ValueError(f"Negative mask pixel value: {pixel}")
        if pixel == 0 or not included:
            continue
        if pixel in selected:
            raise ValueError(f"Duplicate source mask pixel: {pixel}")
        selected[pixel] = str(name).strip() or str(pixel)
    ordered = sorted(selected.items())
    mapping = {
        source_pixel: dense_id for dense_id, (source_pixel, _) in enumerate(ordered)
    }
    names = {dense_id: name for dense_id, (_, name) in enumerate(ordered)}
    return mapping, names


def normalize_generation_pairs(pairs, input_type):
    """Convert legacy (image, label-or-mask) to (image, label, mask)."""
    if input_type not in (0, 1, 2):
        raise ValueError(f"Invalid input type: {input_type}")
    result = []
    for pair in pairs:
        if not pair or not pair[0]:
            raise ValueError("Missing source image path")
        image = pair[0]
        aux = pair[1] if len(pair) > 1 else None
        if input_type == 0:
            result.append((image, aux, None))
        elif input_type == 1:
            result.append((image, None, aux))
        else:
            result.append((image, None, None))
    return result


def has_generation_annotation(pair, input_type):
    """Path presence for both UI pairs and normalized generator triples."""
    if input_type == 2:
        return False
    if input_type == 1 and len(pair) > 2 and pair[2]:
        return True
    return bool(len(pair) > 1 and pair[1])


def select_generation_pairs(pairs, inspection, input_type, *, include_empty=False, include_unlabelled=False):
    """Select from cached validation metadata, without filesystem or pixel reads."""
    selected = []
    skipped = dict(invalid=0, unlabelled=0, empty=0)
    for pair in pairs:
        info = inspection.get(tuple(pair), {})
        if not info.get('is_valid', False):
            skipped['invalid'] += 1
        elif input_type != 2 and not has_generation_annotation(pair, input_type) and not include_unlabelled:
            skipped['unlabelled'] += 1
        elif input_type != 2 and info.get('tag') == 'empty' and not include_empty:
            skipped['empty'] += 1
        else:
            selected.append(pair)
    return selected, skipped


def allocate_split_counts(total, fractions):
    """Largest-remainder allocation; zero-share splits never receive samples."""
    import math

    values = tuple(float(value) for value in fractions)
    if total < 0 or int(total) != total or not values:
        raise ValueError("Invalid sample count or empty split")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Split fractions must be finite and in [0, 1]")
    if not math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Train/val/test split must total 100%")
    raw = [int(total) * value for value in values]
    counts = [math.floor(value) for value in raw]
    order = sorted(
        (i for i, v in enumerate(values) if v > 0),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for i in order[: int(total) - sum(counts)]:
        counts[i] += 1
    return tuple(counts)
