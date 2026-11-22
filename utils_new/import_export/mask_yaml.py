# -*- coding: utf-8 -*-
"""U-Net masks_data.yaml discovery without wildcard/legacy imports."""

import os
from pathlib import Path

import yaml


def load_unet_yaml(yaml_path, log_cb=None):
    """
    Загружает masks_data.yaml (или data.yaml с masks-полями).

    Возвращает dict:
        {
            'pairs':            [(img_path, mask_path), ...],
            'class_names':      {yolo_cls_id: name},   # плотная 0..N-1
            'mask_class_offset': int,
            'num_classes':      int,     # включая фон
            'root':             str,
        }
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"U-Net YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Mask YAML must contain a mapping")
    yaml_path = os.path.abspath(yaml_path)
    root = cfg.get("path", "")
    if not root:
        root = os.path.dirname(yaml_path)
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(yaml_path), root))
    root = os.path.abspath(root)

    # Ключи могут быть разными: train_images/train_masks
    # или images/train + masks/train через общий train.
    splits_cfg = []
    for split in ("train", "val", "test"):
        img_rel = cfg.get(f"{split}_images")
        mask_rel = cfg.get(f"{split}_masks")

        # Fallback на общий формат YOLO: images/{split} + masks/{split}
        if not img_rel:
            split_val = cfg.get(split)
            if isinstance(split_val, str):
                img_rel = split_val
            elif isinstance(split_val, list) and split_val:
                img_rel = split_val[0]
        if not mask_rel:
            mask_rel = f"masks/{split}"

        if img_rel:
            splits_cfg.append((split, img_rel, mask_rel))

    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    mask_exts = (".png", ".tif", ".tiff", ".bmp")

    pairs = []
    for split, img_rel, mask_rel in splits_cfg:
        img_dir = os.path.join(root, img_rel)
        mask_dir = os.path.join(root, mask_rel)
        if not os.path.isdir(img_dir):
            if log_cb:
                log_cb(f"[U-Net] нет папки изображений: {img_dir}")
            continue
        if not os.path.isdir(mask_dir):
            if log_cb:
                log_cb(f"[U-Net] нет папки масок: {mask_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in img_exts:
                continue
            img_path = os.path.join(img_dir, fname)
            mask_path = None
            for mext in mask_exts:
                cand = os.path.join(mask_dir, stem + mext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                if log_cb:
                    log_cb(f"[U-Net] нет маски для {fname}")
                continue
            pairs.append((img_path, mask_path))

    # Классы
    class_names_list = cfg.get("class_names", [])
    mask_class_offset = int(cfg.get("mask_class_offset", 1))
    num_classes = int(cfg.get("num_classes", len(class_names_list)))

    # class_names_list[0] обычно 'background'.
    # Для приложения yolo_cls_id начинается с 0.
    # Если offset=1 → yolo_cls i соответствует pixel i+1, а name — class_names_list[i+1]
    # Если offset=0 → yolo_cls i соответствует pixel i, name — class_names_list[i]
    class_names = {}
    if class_names_list:
        if mask_class_offset >= 1:
            for i, name in enumerate(class_names_list):
                if i == 0:
                    continue  # background
                class_names[i - 1] = str(name)
        else:
            for i, name in enumerate(class_names_list):
                class_names[i] = str(name)

    if log_cb:
        log_cb(
            f"[U-Net] загружено: pairs={len(pairs)}, "
            f"classes={len(class_names)}, "
            f"offset={mask_class_offset}, root={root}"
        )

    return {
        "pairs": pairs,
        "class_names": class_names,
        "mask_class_offset": mask_class_offset,
        "num_classes": num_classes,
        "root": root,
    }


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
        for k in ("train_masks", "val_masks", "test_masks", "mask_class_offset")
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
):
    """Convert indexed masks to editable YOLO-Segment labels (lossy contours).

    Keeps each original image paired with its original mask.  Only labels are
    temporary; mask files are never rewritten.  Uses shared canonical geometry.
    """
    import cv2

    from utils_new.annotations import ImageSize, MaskEncoding, mask_to_annotations
    from utils_new.import_export.images import read_index_mask
    from utils_new.import_export.yolo_labels import write_yolo_labels

    output = Path(output_labels_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = []
    used = set()
    allowed = set(map(int, class_names.keys()))
    for image_path, mask_path in pairs:
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
            anns = [a for a in anns if a.class_id in allowed]
            if not anns and not include_empty:
                continue
            stem = Path(image_path).stem
            candidate = stem
            serial = 2
            while candidate in used:
                candidate = f"{stem}__{serial}"
                serial += 1
            used.add(candidate)
            target = output / f"{candidate}.txt"
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
    from utils_new.import_export.images import collect_image_paths, index_mask_files
    from utils_new.import_export.yolo_yaml import (
        resolve_split_sources,
        _local_image_path,
    )
    from utils_new.import_export.voc import read_voc_pairs

    source = Path(source).resolve()
    if source.is_file():
        cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
        if not isinstance(cfg, dict):
            raise ValueError("YAML должен содержать объект.")
        root = Path(cfg.get("path") or source.parent)
        if not root.is_absolute():
            root = source.parent / root
        root = root.resolve()
    elif source.is_dir():
        root = source
        cfg = {}
    else:
        raise FileNotFoundError(f"Не найден датасет: {source}")
    nc = cfg.get(
        "num_classes",
        len(cfg["class_names"])
        if isinstance(cfg.get("class_names"), list) and cfg["class_names"]
        else default_num_classes,
    )
    if type(nc) is not int or nc < 1:
        raise ValueError("YAML num_classes должен быть положительным целым.")
    result = {key: [] for key in ("train", "val", "test")}
    voc = (
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
        raw = cfg.get(split + "_images", cfg.get(split))
        default_images = root / "images" / split
        default_masks = root / "masks" / split
        if raw is None:
            if not default_images.exists() and not default_masks.exists():
                if split == "train":
                    raise ValueError(f"Нет train-источника: {root}")
                continue
            raw = str(default_images)
        image_entries = raw if isinstance(raw, list) else [raw]
        masks_raw = cfg.get(split + "_masks", str(default_masks))
        mask_entries = masks_raw if isinstance(masks_raw, list) else [masks_raw]
        if len(mask_entries) not in (1, len(image_entries)):
            raise ValueError("Число image/mask источников не совпадает.")
        seen = set()
        for i, entry in enumerate(image_entries):
            mask_root = Path(mask_entries[min(i, len(mask_entries) - 1)])
            if not mask_root.is_absolute():
                mask_root = root / mask_root
            mask_index = index_mask_files(mask_root)
            for resolved in resolve_split_sources(entry, root, strict=True):
                for image in collect_image_paths(
                    resolved, recursive=Path(resolved).is_dir()
                ):
                    image = Path(image).resolve()
                    if image in seen:
                        continue
                    local = _local_image_path(
                        image, resolved if Path(resolved).is_dir() else None, root
                    )
                    mask = mask_index.get(local.with_suffix("").as_posix().casefold())
                    if mask is None:
                        raise FileNotFoundError(f"Нет маски для {image} в {mask_root}")
                    result[split].append((str(image), str(mask.resolve())))
                    seen.add(image)
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
