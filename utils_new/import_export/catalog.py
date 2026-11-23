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

import numpy as np

from utils_new.annotations import ImageSize
from utils_new.import_export.coco import load_coco
from utils_new.import_export.images import read_index_mask, image_size_from_file
from utils_new.import_export.mask_yaml import load_unet_yaml
from utils_new.import_export.yolo_yaml import load_yolo_yaml
from utils_new.import_export.yolo_labels import read_yolo_labels


@dataclass
class DatasetCatalog:
    """Normalized file-pair catalog. Values are source class/pixel IDs."""

    pairs: list
    dataset_type: int
    root: str
    class_names: dict = field(default_factory=dict)
    input_format: str = "auto"
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
    return DatasetCatalog(
        list(pairs),
        0,
        str(root),
        dict(names or {}),
        "auto",
        {
            "yaml_path": str(path),
            "split_counts": dict(splits_info),
            "missing_labels": missing_count,
        },
    )


def load_unet_catalog(path, log_cb=None):
    """Source mask pixel values map to names; preserve actual input offset."""
    data = load_unet_yaml(path, log_cb=log_cb)
    offset = int(data["mask_class_offset"])
    classes = {
        int(cid) + offset: str(name) for cid, name in data["class_names"].items()
    }
    return DatasetCatalog(
        list(data["pairs"]),
        1,
        str(data["root"]),
        classes,
        metadata={"unet_path": str(path), "mask_class_offset": offset},
    )


def discover_coco_json(folder):
    """Find only JSON files containing COCO images and annotations fields."""
    result = []
    for root, _dirs, files in os.walk(folder):
        for filename in files:
            if not filename.lower().endswith(".json"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if (
                    isinstance(data, dict)
                    and "images" in data
                    and "annotations" in data
                ):
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
            items, categories, root = load_coco(
                json_path, log_cb=log_cb, preserve_category_ids=True
            )
        except Exception as exc:
            raise CatalogSourceError(f"{json_path}: {exc}") from exc
        if not items:
            if log_cb:
                log_cb(f"⚠️ {Path(json_path).name}: нет изображений — пропуск.")
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
            for ann in item["annotations"]:
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
            copied["annotations"] = mapped
            remapped_items.append(copied)
        sources.append(CocoSource(str(json_path), remapped_items, str(root)))
    return CocoCatalog(
        sources, {gid: name for name, gid in name_to_id.items()}, name_to_id
    )


def collect_label_classes(pairs, format_hint="auto", log_cb=None):
    """Read class IDs from YOLO labels without a QApplication or label rendering."""
    found = set()
    for pair in pairs:
        path = pair[1] if len(pair) > 1 else None
        if not path:
            continue
        try:
            anns = read_yolo_labels(
                path, ImageSize(*image_size_from_file(pair[0])), format_hint=format_hint
            )
            found.update(int(a.class_id) for a in anns)
        except Exception as exc:
            if log_cb:
                log_cb(f"Ошибка классов {path}: {exc}")
    return sorted(found)


def collect_mask_classes(pairs, log_cb=None):
    """Return (source pixel IDs, decodable pairs); masks are never modified."""
    unique = set()
    valid_pairs = []
    for pair in pairs:
        path = pair[1] if len(pair) > 1 else None
        if not path:
            valid_pairs.append((pair[0], None))
            continue
        try:
            mask = read_index_mask(path)
            if mask.size == 0:
                continue
            unique.update(int(v) for v in np.unique(mask))
            valid_pairs.append((pair[0], path))
        except (OSError, ValueError, TypeError) as exc:
            if log_cb:
                log_cb(f"Ошибка маски {path}: {exc}")
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


class DatasetImportJob:
    """Cancellable discovery/conversion for shared loaders and lazy dataset editors."""

    def __init__(self, kind, paths, options=None):
        self.kind, self.paths = kind, list(paths)
        self.options = dict(options or {})
        self.cancelled = False
        self._log_count = 0
        self.progress = lambda *_: None
        self.log = lambda *_: None

    def _report(self, message):
        if str(message).startswith(("[U-Net → YOLO]", "[COCO -> YOLO]")):
            return
        self._log_count += 1
        if self._log_count <= 40:
            self.log(message)

    def cancel(self):
        self.cancelled = True

    def run(self):
        from tempfile import TemporaryDirectory
        from utils_new.import_export.images import (
            collect_image_paths,
            index_mask_files,
            sidecar_matches,
        )
        from utils_new.import_export.coco import write_yolo_txt_for_coco_items
        from utils_new.import_export.mask_yaml import write_yolo_txt_for_unet_pairs
        from utils_new.import_export.dataset_sources import (
            import_save_targets,
            metadata,
        )
        from utils_new.import_export.yolo_yaml import (
            build_yolo_save_target_map,
            find_mask_for_image,
        )

        if self.cancelled:
            return {"cancelled": True}
        if self.kind == "yolo" and self.options.get("editor"):
            from utils_new.import_export.mask_yaml import is_unet_yaml

            if is_unet_yaml(self.paths[0]):
                self.kind = "unet_editor"
        resource = None
        masks = {}
        if self.kind in ("folder", "images"):
            files = collect_image_paths(
                self.paths[0] if len(self.paths) == 1 else self.paths
            )
            pairs, cache = [], {}
            dtype = self.options.get("dataset_type", 2)
            aux = self.options.get("annotations")
            mask_index = (
                index_mask_files(aux)
                if dtype == 1 and aux and Path(aux).is_dir()
                else {}
            )
            for index, image in enumerate(files):
                if self.cancelled:
                    return {"cancelled": True}
                target = None
                if dtype == 0 and aux:
                    matches = sidecar_matches(aux, Path(image).name, cache)
                    if len(matches) > 1:
                        raise ValueError(f"Несколько меток для {image}")
                    target = str(matches[0]) if matches else None
                elif dtype == 1:
                    target = mask_index.get(Path(image).stem.casefold())
                    target = str(target) if target is not None else None
                pairs.append((image, target))
                if index % 32 == 0:
                    self.progress(index + 1, len(files))
            catalog = DatasetCatalog(pairs, dtype, self.paths[0])
        elif self.kind == "yolo":
            catalog = load_yolo_catalog(self.paths[0], log_cb=self._report)
        elif self.kind in ("unet", "unet_editor"):
            catalog = load_unet_catalog(self.paths[0], log_cb=self._report)
            if self.kind == "unet_editor":
                offset = catalog.metadata["mask_class_offset"]
                names = {
                    pixel - offset: name for pixel, name in catalog.class_names.items()
                }
                self.log(
                    "[U-Net] Подготовка контуров из масок: отверстия и тонкие детали могут теряться; оригинальные маски не изменяются."
                )
                resource = TemporaryDirectory(prefix="neuro-unet-labels-")
                pairs = write_yolo_txt_for_unet_pairs(
                    catalog.pairs,
                    resource.name,
                    class_names=names,
                    mask_offset=offset,
                    include_empty=True,
                    cancel_check=lambda: self.cancelled,
                    progress_cb=self.progress,
                    log_cb=self._report,
                )
                masks = {
                    image: mask for image, mask in catalog.pairs if mask is not None
                }
                catalog = DatasetCatalog(
                    pairs,
                    0,
                    catalog.root,
                    names,
                    metadata=dict(mask_class_offset=offset, mask_map=masks),
                )
        elif self.kind == "coco":
            source = import_coco_catalog(
                self.paths, self.options.get("existing_names"), log_cb=self._report
            )
            resource = TemporaryDirectory(prefix="neuro-coco-labels-")
            pairs = []
            total = sum(len(part.items) for part in source.sources)
            for part_index, part in enumerate(source.sources):
                offset = len(pairs)
                pairs.extend(
                    write_yolo_txt_for_coco_items(
                        part.items,
                        resource.name,
                        name_prefix=f"j{part_index:03d}_",
                        cancel_check=lambda: self.cancelled,
                        progress_cb=lambda done, _, offset=offset: self.progress(
                            offset + done, total
                        ),
                    )
                )
            catalog = DatasetCatalog(
                pairs, 0, str(Path(self.paths[0]).parent), source.class_names
            )
        else:
            raise ValueError(f"Неизвестный источник: {self.kind}")
        if self.cancelled:
            return {"cancelled": True}
        if self._log_count > 40:
            self.log(f"Дополнительных сообщений о разметке: {self._log_count - 40}")
        result = dict(
            catalog=catalog, resource=resource, paths=self.paths, kind=self.kind
        )
        if self.options.get("editor") and self.kind not in ("folder", "images"):
            images = [pair[0] for pair in catalog.pairs]
            result["save_targets"] = (
                build_yolo_save_target_map(self.paths[0], images, root=catalog.root)
                if self.kind == "yolo"
                else import_save_targets(self.paths, images)
            )
            if (
                self.kind == "yolo"
                and self.options.get("overlay")
                and not metadata(self.paths[0])
            ):
                for index, image in enumerate(images):
                    if self.cancelled:
                        return {"cancelled": True}
                    mask = find_mask_for_image(image, base_path=catalog.root)
                    if mask:
                        masks[image] = mask
                    if index % 32 == 0:
                        self.progress(index + 1, len(images))
                catalog.metadata["mask_map"] = masks
        return result
