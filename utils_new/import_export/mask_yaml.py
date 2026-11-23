# -*- coding: utf-8 -*-
"""U-Net masks_data.yaml discovery without wildcard/legacy imports."""

from pathlib import Path

import yaml


def _mask_config(source, default_num_classes=2):
    source = Path(source).resolve()
    if source.is_file():
        cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("Mask YAML must contain a mapping")
        root = Path(cfg.get("path") or source.parent)
        if not root.is_absolute():
            root = source.parent / root
    elif source.is_dir():
        root, cfg = source, {}
    else:
        raise FileNotFoundError(f"Не найден датасет: {source}")
    names = cfg.get("class_names") or []
    if not isinstance(names, list):
        raise ValueError("class_names должен быть списком имён по pixel ID.")
    offset = cfg.get("mask_class_offset", 1)
    nc = cfg.get("num_classes", len(names) or default_num_classes)
    if type(offset) is not int or not 0 <= offset <= 65535:
        raise ValueError("mask_class_offset должен быть целым 0..65535.")
    if (
        type(nc) is not int
        or nc < 1
        or (names and len(names) != nc and not (nc == 1 and len(names) == 2))
    ):
        raise ValueError("num_classes должен совпадать с числом имён pixel IDs.")
    return cfg, root.resolve(), nc, offset, names


def _mask_split_pairs(cfg, root, split, *, strict, log=None):
    from utils_new.import_export.images import collect_image_paths, index_mask_files
    from utils_new.import_export.yolo_yaml import (
        resolve_split_sources,
        _local_image_path,
    )

    if cfg.get('_dataset_sources'):
        selection = cfg['_dataset_sources']
        result = []
        for value in resolve_split_sources(cfg.get(split), root, strict=True):
            for image in collect_image_paths(value, recursive=Path(value).is_dir()):
                image = str(Path(image).resolve())
                mask = selection['annotations'].get(image)
                if strict and not mask:
                    continue  # No mask is not a background target for semantic training.
                result.append((image, mask))
        return list(dict.fromkeys(result))

    aliases = ("val", "valid", "validation") if split == "val" else (split,)
    key = next((key for key in aliases if key in cfg or key + "_images" in cfg), split)
    raw = cfg.get(key + "_images", cfg.get(key))
    default_images, default_masks = root / "images" / key, root / "masks" / key
    if raw is None:
        if not default_images.exists() and not default_masks.exists():
            return []
        raw = str(default_images)
    image_entries = raw if isinstance(raw, list) else [raw]
    masks = cfg.get("masks") or {}
    if not isinstance(masks, dict):
        raise ValueError("masks должен содержать пути по split.")
    masks_raw = cfg.get(
        key + "_masks",
        cfg.get(split + "_masks", masks.get(key, masks.get(split, str(default_masks)))),
    )
    mask_entries = masks_raw if isinstance(masks_raw, list) else [masks_raw]
    if (
        not image_entries
        or not mask_entries
        or len(mask_entries) not in (1, len(image_entries))
    ):
        raise ValueError("Число image/mask источников не совпадает.")
    result, seen, indices = [], set(), {}
    for i, entry in enumerate(image_entries):
        mask_root = Path(mask_entries[min(i, len(mask_entries) - 1)])
        if not mask_root.is_absolute():
            mask_root = root / mask_root
        try:
            if mask_root not in indices:
                indices[mask_root] = index_mask_files(mask_root)
            mask_index = indices[mask_root]
        except FileNotFoundError as exc:
            if strict:
                raise
            if log:
                log(f"[U-Net] {exc}")
            mask_index = {}
        sources = resolve_split_sources(entry, root, strict=strict)
        for resolved in sources:
            for value in collect_image_paths(
                resolved, recursive=Path(resolved).is_dir()
            ):
                image = Path(value).resolve()
                if not image.is_file():
                    if strict:
                        raise FileNotFoundError(f"Нет изображения: {image}")
                    if log:
                        log(f"[U-Net] нет изображения: {image}")
                    continue
                if image in seen:
                    continue
                local = _local_image_path(
                    image, resolved if Path(resolved).is_dir() else None, root
                )
                mask = mask_index.get(local.with_suffix("").as_posix().casefold())
                if mask is None or mask.resolve() == image:
                    message = f"Нет отдельной маски для {image} в {mask_root}"
                    if strict:
                        raise FileNotFoundError(message)
                    if log:
                        log(f"[U-Net] {message}")
                    result.append((str(image), None))
                    seen.add(image)
                    continue
                result.append((str(image), str(mask.resolve())))
                seen.add(image)
    return result


def load_unet_yaml(yaml_path, log_cb=None):
    """Read the same nested/list/mask sources as training, skipping missing pairs.

    Names returned to the UI use object IDs; class_names in YAML uses raw pixel
    IDs, including background and any reserved IDs below mask_class_offset.
    """
    if not Path(yaml_path).is_file():
        raise FileNotFoundError(f"U-Net YAML not found: {yaml_path}")
    cfg, root, nc, offset, names = _mask_config(yaml_path)
    pairs, seen = [], set()
    for split in ("train", "val", "test"):
        for pair in _mask_split_pairs(cfg, root, split, strict=False, log=log_cb):
            if pair[0] not in seen:
                pairs.append(pair)
                seen.add(pair[0])
    classes = {
        pixel - offset: str(name) for pixel, name in enumerate(names) if pixel >= offset
    }
    if log_cb:
        log_cb(
            f"[U-Net] загружено: pairs={len(pairs)}, classes={len(classes)}, offset={offset}, root={root}"
        )
    if cfg.get('_dataset_sources') and log_cb:
        from utils_new.import_export.dataset_sources import log_manifest
        log_manifest(yaml_path, log_cb)
    return dict(
        pairs=pairs,
        class_names=classes,
        mask_class_offset=offset,
        num_classes=nc,
        root=str(root),
    )


def is_unet_yaml(yaml_path):
    """
    Эвристика: является ли YAML U-Net-структурой.
    Признак — наличие 'train_masks' / 'val_masks' / 'mask_class_offset'
    или 'class_names' со строкой 'background'.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return False

    if not isinstance(cfg, dict):
        return False
    if any(
        k in cfg
        for k in (
            "train_masks",
            "val_masks",
            "test_masks",
            "mask_class_offset",
            "masks",
        )
    ):
        return True
    cn = cfg.get("class_names")
    if isinstance(cn, list) and cn and str(cn[0]).lower() == "background":
        return True
    return False


def load_unet_yaml_pairs(yaml_path, log_cb=None):
    """Compatibility tuple for tabs, delegating all YAML logic to load_unet_yaml."""
    data = load_unet_yaml(yaml_path, log_cb=log_cb)
    return (
        data["pairs"],
        data["class_names"],
        int(data["mask_class_offset"]),
        data["root"],
        int(data["num_classes"]),
    )


def write_yolo_txt_for_unet_pairs(
    pairs,
    output_labels_dir,
    *,
    class_names,
    mask_offset=1,
    include_empty=True,
    log_cb=None,
    cancel_check=None,
    progress_cb=None,
):
    """Convert indexed masks to editable YOLO-Segment labels (lossy contours).

    Keeps each original image paired with its original mask.  Only labels are
    temporary; mask files are never rewritten.  Uses shared canonical geometry.
    """
    import cv2

    from utils_new.annotations import ImageSize, MaskEncoding, mask_to_annotations
    from utils_new.import_export.images import read_index_mask
    from utils_new.import_export.yolo_labels import write_yolo_labels, unique_label_path

    output = Path(output_labels_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    used = {p.name.casefold() for p in output.iterdir()}
    allowed = set(map(int, class_names.keys()))
    for index, (image_path, mask_path) in enumerate(pairs):
        if cancel_check and cancel_check():
            break
        if progress_cb and (index % 16 == 0 or index + 1 == len(pairs)):
            progress_cb(index + 1, len(pairs))
        if mask_path is None:
            result.append((str(image_path), None))
            continue
        try:
            mask = read_index_mask(mask_path)
            height, width = mask.shape
            if mask_offset == 0 and log_cb:
                log_cb(
                    "[U-Net] offset=0: pixel 0 considered background and cannot represent class 0 unambiguously"
                )
            encoding = MaskEncoding(
                offset=int(mask_offset),
                background_value=0,
                allow_ambiguous_zero=(int(mask_offset) == 0),
            )
            anns = mask_to_annotations(
                mask, encoding=encoding, kind="segment", holes="drop"
            )
            if class_names:
                anns = [a for a in anns if a.class_id in allowed]
            if not anns and not include_empty:
                continue
            target = unique_label_path(output, image_path, used=used)
            write_yolo_labels(target, anns, ImageSize(width, height))
            result.append((str(image_path), str(target)))
            if log_cb:
                log_cb(
                    f"[U-Net → YOLO] {Path(mask_path).name}: {len(anns)} contours (lossy)"
                )
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            if log_cb:
                log_cb(f"[U-Net] failed mask {mask_path}: {exc}")
    return result


def load_training_sources(source, default_num_classes=2):
    """Exact train/val/test mask pairs from a root, mask YAML or VOC YAML.

    Unlike catalog previews, explicitly named missing sources are errors.
    Split membership, class count and nested paths are preserved.
    """
    from utils_new.import_export.voc import read_voc_pairs

    cfg, root, nc, _offset, _names = _mask_config(source, default_num_classes)
    result = {key: [] for key in ("train", "val", "test")}
    voc = not cfg.get('_dataset_sources') and (
        bool(cfg.get("train_txt"))
        or (root / "ImageSets/Segmentation/train.txt").exists()
    )
    for split in result:
        if voc:
            split_file = cfg.get(
                split + "_txt", str(Path("ImageSets/Segmentation") / (split + ".txt"))
            )
            path = Path(split_file)
            path = path if path.is_absolute() else root / path
            if not path.exists() and split != "train" and split + "_txt" not in cfg:
                continue
            images = Path(cfg.get("jpeg_dir", "JPEGImages"))
            masks = Path(cfg.get("seg_dir", "SegmentationClass"))
            result[split] = read_voc_pairs(
                path,
                images if images.is_absolute() else root / images,
                masks if masks.is_absolute() else root / masks,
            )
            continue
        result[split] = _mask_split_pairs(cfg, root, split, strict=True)
    if not result["train"]:
        raise ValueError("Обучающая выборка пуста.")
    assigned = {}
    for split, pairs in result.items():
        for image, mask in pairs:
            if image in assigned:
                raise ValueError(
                    f"Снимок находится в нескольких split: {image}: {assigned[image]}/{split}"
                )
            assigned[image] = split
    return dict(
        splits=result,
        num_classes=nc,
        root=str(root),
        class_names=cfg.get("class_names", []),
    )
