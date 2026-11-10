# -*- coding: utf-8 -*-
"""Qt-free catalog services for all dataset tabs.

One source of truth for YOLO/U-Net catalog loading, COCO multi-file class
merging, source detection, and class collection. No widgets, GUI dialogs,
temporary output, or mutation of a controller's state.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from utils_new.annotations import ImageSize
from utils_new.dataset_coco_new import load_coco
from utils_new.dataset_files_new import read_index_mask
from utils_new.dataset_unet_new import load_unet_yaml
from utils_new.dataset_yaml_new import load_yolo_yaml
from utils_new.dataset_yolo_labels_new import read_yolo_labels


@dataclass
class DatasetCatalog:
    """Normalized file-pair catalog. Values are source class/pixel IDs."""
    pairs: list
    dataset_type: int
    root: str
    class_names: dict = field(default_factory=dict)
    input_format: str = 'auto'
    metadata: dict = field(default_factory=dict)


@dataclass
class CocoSource:
    path: str
    items: list
    root: str


@dataclass
class CocoCatalog:
    sources: list
    class_names: dict
    name_to_id: dict


class CatalogSourceError(ValueError):
    """Invalid catalog source; source file is included in error text."""


def load_yolo_catalog(path, log_cb=None, *, require_labels=False):
    """Read existing YOLO data.yaml into UI-independent pairs and class names."""
    pairs, names, root, splits_info = load_yolo_yaml(path, log_cb=log_cb)
    missing_count = sum(1 for _img, lbl in pairs if lbl is None)
    if require_labels:
        pairs = [(img, lbl) for img, lbl in pairs if lbl is not None]
    return DatasetCatalog(list(pairs), 0, str(root), dict(names or {}), 'auto',
                          {'yaml_path': str(path), 'split_counts': dict(splits_info),
                           'missing_labels': missing_count})


def load_unet_catalog(path, log_cb=None):
    """Source mask pixel values map to names; preserve actual input offset."""
    data = load_unet_yaml(path, log_cb=log_cb)
    offset = int(data['mask_class_offset'])
    classes = {int(cid) + offset: str(name)
               for cid, name in data['class_names'].items()}
    return DatasetCatalog(list(data['pairs']), 1, str(data['root']), classes,
                          metadata={'unet_path': str(path), 'mask_class_offset': offset})


def discover_coco_json(folder):
    """Find only JSON files containing COCO images and annotations fields."""
    result = []
    for root, _dirs, files in os.walk(folder):
        for filename in files:
            if not filename.lower().endswith('.json'):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and 'images' in data and 'annotations' in data:
                    result.append(path)
            except (OSError, ValueError, TypeError):
                continue
    return sorted(result)


def import_coco_catalog(json_paths, existing_name_to_id=None, *, log_cb=None):
    """Merge COCO sources by category NAME and remap objects to global IDs.

    Returns separate source groups for consumers wishing to write labels,
    or to keep their own in-memory catalog. Does not touch input items/files.
    """
    name_to_id = dict(existing_name_to_id or {})
    next_id = max(name_to_id.values(), default=-1) + 1
    sources = []
    for json_path in json_paths:
        try:
            items, categories, root = load_coco(json_path, log_cb=log_cb,
                                               preserve_category_ids=True)
        except Exception as exc:
            raise CatalogSourceError(f'{json_path}: {exc}') from exc
        if not items:
            if log_cb:
                log_cb(f'⚠️ {Path(json_path).name}: нет изображений — пропуск.')
            continue
        local_to_global = {}
        for coco_id, category_name in categories.items():
            name = str(category_name) if category_name is not None else str(coco_id)
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1
            local_to_global[int(coco_id)] = name_to_id[name]
        remapped_items = []
        for item in items:
            mapped = []
            for ann in item['annotations']:
                # Catalog requests original COCO IDs, including category_id=0.
                src_category_id = int(ann[1])
                global_id = local_to_global.get(src_category_id)
                if global_id is None:
                    name = str(src_category_id)
                    if name not in name_to_id:
                        name_to_id[name] = next_id
                        next_id += 1
                    global_id = name_to_id[name]
                    local_to_global[src_category_id] = global_id
                mapped.append((ann[0], global_id, *ann[2:]))
            copied = dict(item)
            copied['annotations'] = mapped
            remapped_items.append(copied)
        sources.append(CocoSource(str(json_path), remapped_items, str(root)))
    return CocoCatalog(sources, {gid: name for name, gid in name_to_id.items()}, name_to_id)


def collect_label_classes(pairs, format_hint='auto', log_cb=None):
    """Read class IDs from YOLO labels without a QApplication or label rendering."""
    found = set()
    unit = ImageSize(1, 1)  # class IDs independent of image dimensions
    for pair in pairs:
        path = pair[1] if len(pair) > 1 else None
        if not path:
            continue
        try:
            anns = read_yolo_labels(path, unit, format_hint=format_hint)
            found.update(int(a.class_id) for a in anns)
        except Exception as exc:
            if log_cb:
                log_cb(f'Ошибка классов {path}: {exc}')
    return sorted(found)


def collect_mask_classes(pairs, log_cb=None):
    """Return (source pixel IDs, decodable pairs); masks are never modified."""
    unique = set()
    valid_pairs = []
    for pair in pairs:
        path = pair[1] if len(pair) > 1 else None
        if not path:
            continue
        try:
            mask = read_index_mask(path)
            if mask.size == 0:
                continue
            if mask.dtype not in (np.uint8, np.uint16):
                mask = mask.astype(np.uint8)
            unique.update(int(v) for v in np.unique(mask))
            valid_pairs.append((pair[0], path))
        except (OSError, ValueError, TypeError) as exc:
            if log_cb:
                log_cb(f'Ошибка маски {path}: {exc}')
    return sorted(unique), valid_pairs


def resolve_class_name(names, cid):
    if not names:
        return None
    try:
        if isinstance(names, dict):
            if cid in names:
                return str(names[cid])
            if str(cid) in names:
                return str(names[str(cid)])
        elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
            return str(names[cid])
    except (ValueError, TypeError, IndexError):
        pass
    return None


def dense_class_remap(ids):
    """Map selected SOURCE identifiers to zero-based dense IDs, sorted."""
    return {old: new for new, old in enumerate(sorted({int(i) for i in ids}))}
