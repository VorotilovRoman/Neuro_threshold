"""Explicit, inspectable dataset import plans and immutable approved manifests. No Qt/ML."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import os
import yaml

from utils_new.import_export.images import collect_image_paths, index_mask_files
from utils_new.import_export.yolo_yaml import resolve_split_sources, _local_image_path

MODES = {
    "declared": "По YAML",
    "beside": "Рядом со снимками",
    "labels": "Папка labels",
    "other": "Другие пути",
    "none": "Без разметки",
    "json": "Аннотации COCO JSON",
}
KEY = "_dataset_sources"


def metadata(path):
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    cfg = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return cfg.get(KEY, {}) if isinstance(cfg, dict) else {}


def original_source(path):
    return metadata(path).get("source", str(path))


def _paths(value, root):
    values = value if isinstance(value, list) else [value]
    return [
        str((Path(v) if Path(v).is_absolute() else root / v).resolve())
        for v in values
        if v
    ]


def defaults(paths, kind):
    paths = [str(Path(p).resolve()) for p in paths]
    if kind == "coco":
        root = Path(os.path.commonpath([str(Path(p).parent) for p in paths]))
        rows = [
            dict(split=Path(p).stem, images=[str(root)], annotations=[p]) for p in paths
        ]
        return dict(kind=kind, sources=paths, root=str(root), config={}, rows=rows)
    cfg = yaml.safe_load(Path(paths[0]).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError("YAML должен содержать словарь.")
    # A saved view remains usable even if its original YAML has been moved.
    if KEY in cfg:
        info = cfg[KEY]
        return dict(
            kind=info["kind"],
            sources=[info["source"]],
            root=info["root"],
            config=cfg,
            rows=deepcopy(info["rows"]),
            initial_mode=info["mode"],
        )
    if any(
        k in cfg for k in ("train_masks", "val_masks", "mask_class_offset", "masks")
    ):
        kind = "unet"
    root = Path(cfg.get("path") or Path(paths[0]).parent)
    if not root.is_absolute():
        root = Path(paths[0]).parent / root
    root = root.resolve()
    rows = []
    if cfg.get("train_txt") or (root / "ImageSets/Segmentation/train.txt").is_file():
        for split in ("train", "val", "test"):
            listing = _paths(
                cfg.get(split + "_txt", f"ImageSets/Segmentation/{split}.txt"), root
            )[0]
            if not Path(listing).is_file():
                continue
            masks = _paths(cfg.get("seg_dir", "SegmentationClass"), root)
            rows.append(
                dict(
                    split=split,
                    images=[listing],
                    annotations=masks,
                    declared=masks,
                    voc_images=_paths(cfg.get("jpeg_dir", "JPEGImages"), root)[0],
                )
            )
        return dict(kind="unet", sources=paths, root=str(root), config=cfg, rows=rows)
    for split in ("train", "val", "test"):
        aliases = ("val", "valid", "validation") if split == "val" else (split,)
        key = next((k for k in aliases if k in cfg or k + "_images" in cfg), split)
        raw = cfg.get(key + "_images", cfg.get(key))
        if raw is None:
            continue
        masks = cfg.get("masks") or {}
        aux = (
            cfg.get(key + "_masks", masks.get(key, f"masks/{key}"))
            if kind == "unet"
            else f"labels/{key}"
        )
        rows.append(
            dict(
                split=split,
                images=_paths(raw, root),
                annotations=_paths(aux, root),
                declared=_paths(aux, root),
            )
        )
    return dict(kind=kind, sources=paths, root=str(root), config=cfg, rows=rows)


def relocate(spec, root):
    result = deepcopy(spec)
    old, new = Path(spec["root"]), Path(root).resolve()
    for row in result["rows"]:
        for key in ("images", "annotations", "declared"):
            if key not in row:
                continue
            if spec["kind"] == "coco" and key == "annotations":
                continue
            row[key] = [
                str(new / Path(p).relative_to(old))
                if Path(p).is_relative_to(old)
                else p
                for p in row[key]
            ]
        if row.get("voc_images") and Path(row["voc_images"]).is_relative_to(old):
            row["voc_images"] = str(new / Path(row["voc_images"]).relative_to(old))
    result["root"] = str(new)
    return result


def proposed_annotations(spec, row, mode):
    if mode in ("declared", "json", "other"):
        return list(row["annotations"])
    if mode == "none":
        return []
    if mode == "beside":
        if row.get("voc_images"):
            return [row["voc_images"]]
        return [
            str(Path(p).parent if Path(p).is_file() else Path(p)) for p in row["images"]
        ]
    return [
        str(
            Path(spec["root"])
            / ("masks" if spec["kind"] == "unet" else "labels")
            / row["split"]
        )
    ]


def preview(spec, mode, cancel=None):
    """Count every image, matched sidecar, missing sidecar and conflicting source."""
    spec = deepcopy(spec)
    for row in spec["rows"]:
        for key in ("images", "annotations"):
            row[key] = _paths(row[key], Path(spec["root"]))
    if spec["kind"] == "coco":
        spec["sources"] = [row["annotations"][0] for row in spec["rows"]]

    def checkpoint():
        if cancel is not None and cancel.is_set():
            from concurrent.futures import CancelledError

            raise CancelledError()

    entries, summaries, coco_payloads = [], [], []
    roots = {}
    for row in spec["rows"]:
        checkpoint()
        row_entries = []
        if spec["kind"] == "coco":
            source = Path(row["annotations"][0])
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
            counts = Counter(a.get("image_id") for a in payload.get("annotations", []))
            images = []
            for record in payload.get("images", []):
                checkpoint()
                name = Path(record["file_name"])
                candidates = (
                    [name]
                    if name.is_absolute()
                    else [Path(p) / name for p in row["images"]]
                    + [Path(p) / name.name for p in row["images"]]
                )
                found = next((p.resolve() for p in candidates if p.is_file()), None)
                if found is None:
                    continue
                copy = dict(record, file_name=str(found))
                images.append(copy)
                row_entries.append(
                    dict(
                        image=str(found),
                        annotation=None if record.get('annotation_status') == 'unlabelled' else str(source),
                        split=row["split"],
                        alternatives={},
                        objects=counts[record["id"]],
                    )
                )
            coco_payloads.append(
                dict(source=str(source), payload=dict(payload, images=images))
            )
            file_count = len(payload.get("annotations", []))
        else:
            folders = proposed_annotations(spec, row, mode)
            file_count = set()
            if spec["kind"] == "yolo" and mode != "beside":
                for folder in folders:
                    key = ("label_files", folder)
                    if key not in roots:
                        roots[key] = (
                            {str(p.resolve()) for p in Path(folder).rglob("*.txt")}
                            if Path(folder).is_dir()
                            else set()
                        )
                    file_count.update(roots[key])
            seen = set()
            for value in row["images"]:
                if row.get("voc_images") and Path(value).suffix.lower() == ".txt":
                    from utils_new.import_export.voc import read_voc_images

                    resolved_sources = read_voc_images(value, row["voc_images"])
                else:
                    resolved_sources = resolve_split_sources(
                        value, spec["root"], strict=True
                    )
                for source in resolved_sources:
                    for image in collect_image_paths(
                        source, recursive=Path(source).is_dir()
                    ):
                        checkpoint()
                        image = str(Path(image).resolve())
                        if image in seen:
                            continue
                        seen.add(image)
                        local = _local_image_path(
                            image,
                            row.get("voc_images")
                            or (source if Path(source).is_dir() else None),
                            spec["root"],
                        )
                        if spec["kind"] == "yolo":
                            from utils_new.import_export.yolo_yaml import (
                                label_candidates,
                            )

                            alternatives = label_candidates(
                                image,
                                spec["root"],
                                row["split"],
                                source if Path(source).is_dir() else None,
                            )
                        else:
                            alternatives = {}
                            for name, dirs in [
                                ("declared", row.get("declared", row["annotations"])),
                                ("beside", [str(Path(image).parent)]),
                                (
                                    "labels",
                                    [str(Path(spec["root"]) / "masks" / row["split"])],
                                ),
                            ]:
                                found = []
                                for folder in dirs:
                                    key = ("mask", folder)
                                    if key not in roots:
                                        roots[key] = (
                                            index_mask_files(folder)
                                            if Path(folder).is_dir()
                                            else {}
                                        )
                                    item = roots[key].get(
                                        local.with_suffix("").as_posix().casefold()
                                    )
                                    if name == "beside":
                                        item = roots[key].get(
                                            Path(image).stem.casefold()
                                        )
                                    if item and item.resolve() != Path(image):
                                        found.append(str(item.resolve()))
                                alternatives[name] = list(dict.fromkeys(found))
                        if mode == "none":
                            selected = []
                        elif mode in ("beside", "labels"):
                            selected = alternatives.get(mode, [])
                        else:
                            selected = []
                            for folder in folders:
                                folder = Path(folder)
                                if spec["kind"] == "yolo":
                                    candidates = [
                                        folder / local.with_suffix(".txt"),
                                        folder / (Path(image).stem + ".txt"),
                                    ]
                                    selected.extend(
                                        str(p.resolve())
                                        for p in candidates
                                        if p.is_file()
                                    )
                                    key = ("label_files", str(folder))
                                    if key not in roots:
                                        roots[key] = (
                                            {
                                                str(p.resolve())
                                                for p in folder.rglob("*.txt")
                                            }
                                            if folder.is_dir()
                                            else set()
                                        )
                                else:
                                    key = ("mask", str(folder))
                                    if key not in roots:
                                        roots[key] = (
                                            index_mask_files(folder)
                                            if folder.is_dir()
                                            else {}
                                        )
                                    item = roots[key].get(
                                        local.with_suffix("").as_posix().casefold()
                                    )
                                    if item and item.resolve() != Path(image):
                                        selected.append(str(item.resolve()))
                                    file_count.update(map(str, roots[key].values()))
                        selected = list(dict.fromkeys(selected))
                        alternatives[mode] = selected
                        if len(selected) > 1:
                            raise ValueError(
                                f"В выбранном источнике несколько меток для {image}: {selected}. Уточните папку."
                            )
                        row_entries.append(
                            dict(
                                image=image,
                                annotation=selected[0] if selected else None,
                                split=row["split"],
                                alternatives=alternatives,
                                target=(
                                    selected[0]
                                    if selected
                                    else str(Path(image).with_suffix(".txt"))
                                    if mode == "beside"
                                    else str(
                                        Path(folders[0]) / local.with_suffix(".txt")
                                    )
                                    if folders
                                    else str(
                                        Path(spec["root"])
                                        / "labels"
                                        / row["split"]
                                        / local.with_suffix(".txt")
                                    )
                                ),
                            )
                        )
            if mode in ("beside", "labels"):
                row["actual_annotations"] = (
                    sorted({str(Path(e["image"]).parent) for e in row_entries})
                    if mode == "beside"
                    else sorted(
                        {
                            str(Path(e["annotation"]).parent)
                            for e in row_entries
                            if e["annotation"]
                        }
                    )
                    or folders
                )
                if spec["kind"] == "yolo":
                    for folder in row["actual_annotations"]:
                        key = ("label_files", folder)
                        if key not in roots:
                            roots[key] = {
                                str(p.resolve()) for p in Path(folder).glob("*.txt")
                            }
                        file_count.update(roots[key])
            if mode in ("beside", "labels"):
                file_count.update(
                    e["annotation"] for e in row_entries if e["annotation"]
                )
            file_count = len(file_count)
        conflicts = sum(
            len({p for values in e["alternatives"].values() for p in values}) > 1
            for e in row_entries
        )
        summaries.append(
            dict(
                split=row["split"],
                row_index=row.get("row_index", len(summaries)),
                images=len(row_entries),
                matched=sum(bool(e["annotation"]) for e in row_entries),
                files=file_count,
                missing=sum(not e["annotation"] for e in row_entries),
                conflicts=conflicts,
            )
        )
        entries.extend(row_entries)
    if spec["kind"] == "coco":
        by_image = {}
        for entry in entries:
            by_image.setdefault(entry["image"], set()).add(entry["annotation"])
        for summary, row in zip(summaries, spec["rows"]):
            summary["conflicts"] = sum(
                len(by_image[e["image"]]) > 1
                for e in entries
                if e["split"] == row["split"]
            )
    # A repeated image may belong to multiple splits; retain membership for audit.
    return dict(
        spec=deepcopy(spec),
        mode=mode,
        entries=entries,
        summaries=summaries,
        coco=coco_payloads,
    )


def write_manifest(plan, directory):
    """Persist only the approved paths. Never copy images or change input datasets."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = plan["spec"]
    info = dict(
        version=1,
        source=spec["sources"][0],
        kind=spec["kind"],
        root=spec["root"],
        mode=plan["mode"],
        annotations={e["image"]: e["annotation"] for e in plan["entries"]},
        targets={e["image"]: e.get("target") for e in plan["entries"]},
        summaries=plan["summaries"],
        rows=spec["rows"],
        missing_dir=str(directory / "unlabelled"),
    )
    if spec["kind"] == "coco":
        result = []
        for i, row in enumerate(plan["coco"]):
            payload = dict(row["payload"], **{KEY: dict(info, source=row["source"])})
            path = directory / f"coco_{i}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            result.append(str(path))
        return result
    cfg = deepcopy(spec["config"])
    for key in (
        "train",
        "val",
        "valid",
        "validation",
        "test",
        "train_images",
        "val_images",
        "valid_images",
        "test_images",
    ):
        cfg.pop(key, None)
    cfg["path"], cfg[KEY] = spec["root"], info
    for row in spec["rows"]:
        split = row["split"]
        paths = list(
            dict.fromkeys(e["image"] for e in plan["entries"] if e["split"] == split)
        )
        listing = directory / f"{split}.txt"
        listing.write_text("".join(p + "\n" for p in paths), encoding="utf-8")
        cfg[split] = str(listing)
    path = directory / "data.yaml"
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return [str(path)]


def log_plan(plan, log):
    spec = plan["spec"]
    log(
        f"[ИСТОЧНИК] {', '.join(spec['sources'])}; path={spec['root']}; разметка: {MODES[plan['mode']]}"
    )
    for row, count in zip(spec["rows"], plan["summaries"]):
        paths = row.get(
            "actual_annotations", proposed_annotations(spec, row, plan["mode"])
        )
        log(
            f"[ИСТОЧНИК] {row['split']}: снимки={row['images']}; метки/маски={paths}; "
            f"снимков={count['images']}, с разметкой={count['matched']}, файлов разметки={count['files']}, без разметки={count['missing']}"
        )


def log_manifest(path, log):
    info = metadata(path)
    if info and log:
        spec = dict(
            sources=[info["source"]],
            root=info["root"],
            kind=info["kind"],
            rows=info["rows"],
        )
        log_plan(dict(spec=spec, mode=info["mode"], summaries=info["summaries"]), log)


def persist_source(path, directory):
    """Keep the approved view with a training run so resume survives GUI closure."""
    info = metadata(path)
    if not info or info["kind"] == "coco":
        return str(path)
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    entries = []
    for row in info["rows"]:
        for source in resolve_split_sources(
            cfg.get(row["split"]), info["root"], strict=True
        ):
            for image in collect_image_paths(source, recursive=Path(source).is_dir()):
                image = str(Path(image).resolve())
                entries.append(
                    dict(
                        image=image,
                        annotation=info["annotations"].get(image),
                        target=info.get("targets", {}).get(image),
                        split=row["split"],
                    )
                )
    spec = dict(
        kind=info["kind"],
        sources=[info["source"]],
        root=info["root"],
        config=cfg,
        rows=info["rows"],
    )
    return write_manifest(
        dict(
            spec=spec, entries=entries, summaries=info["summaries"], mode=info["mode"]
        ),
        directory,
    )[0]


def enable_ultralytics_sources():
    """Install a transparent factory adapter; each dataset keeps its own selection."""
    from runtime_policy import configure_ultralytics
    from runtime_torch import require_torch

    configure_ultralytics()
    require_torch()
    import ultralytics.data.build as build
    from utils_new.training.dataset_sources import SelectedYOLODataset

    if build.YOLODataset is not SelectedYOLODataset:
        build.YOLODataset = SelectedYOLODataset


def import_save_targets(source_paths, image_paths):
    """Editor writes go to durable labels/, never the session's conversion cache."""
    roots = []
    for source in source_paths:
        info = metadata(source)
        if info:
            roots.append(Path(info['root']).resolve())
        elif Path(source).suffix.lower() in ('.yaml', '.yml'):
            cfg = yaml.safe_load(Path(source).read_text(encoding='utf-8-sig')) or {}
            root = Path(cfg.get('path') or Path(source).resolve().parent)
            roots.append(root if root.is_absolute() else Path(source).resolve().parent / root)
        else:
            roots.append(Path(source).resolve().parent)
    result, used = {}, set()
    for raw in image_paths:
        image = Path(raw).resolve()
        root = next((p for p in roots if image.is_relative_to(p)), roots[0])
        if image.is_relative_to(root / 'images'):
            local = image.relative_to(root / 'images').with_suffix('.txt')
        elif image.is_relative_to(root):
            local = image.relative_to(root).with_suffix('.txt')
        else:
            import hashlib
            suffix = hashlib.sha256(str(image).encode()).hexdigest()[:12]
            local = Path(image.stem + '_' + suffix + '.txt')
        target = root / 'labels' / local
        key = str(target).casefold()
        if key in used:
            raise ValueError(f'Несколько снимков используют один путь сохранения: {target}')
        used.add(key)
        result[str(raw)] = str(target)
    return result
