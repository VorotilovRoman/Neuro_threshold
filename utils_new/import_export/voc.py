"""Loss-aware YOLO/mask dataset export to VOC and strict VOC split reading."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from utils_new.annotations import ImageSize, bounds
from utils_new.import_export.atomic import atomic_path, write_text_atomic
from utils_new.import_export.images import (
    IMAGE_EXTENSIONS,
    collect_image_paths,
    image_size_from_file,
    read_index_mask,
)
from utils_new.import_export.dataset_output import write_index_mask
from utils_new.import_export.yolo_labels import read_yolo_labels
from utils_new.import_export.yolo_yaml import (
    resolve_split_sources,
    _find_label_for_image,
)


def read_voc_images(txt_path, jpeg_dir):
    path = Path(txt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден VOC список: {path}")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="cp1251").splitlines()
    pairs = []
    seen = set()
    for line in lines:
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Некорректный VOC ID: {name}")
        images = [
            p
            for p in (Path(jpeg_dir) / name).parent.glob(Path(name).name + ".*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) != 1:
            raise ValueError(
                f"Для VOC ID {name} нужно ровно одно изображение: {images}"
            )
        image = images[0].resolve()
        if image not in seen:
            pairs.append(str(image))
            seen.add(image)
    return pairs


def read_voc_pairs(txt_path, jpeg_dir, seg_dir):
    pairs = []
    for image in read_voc_images(txt_path, jpeg_dir):
        relative = Path(image).relative_to(Path(jpeg_dir).resolve()).with_suffix(".png")
        mask = Path(seg_dir) / relative
        if not mask.is_file():
            raise FileNotFoundError(f"Нет маски: {mask}")
        pairs.append((image, str(mask.resolve())))
    return pairs


def convert_mask_to_voc_format(source, destination, *, binary=False):
    mask = read_index_mask(source)
    if binary:
        mask = (mask > 0).astype(np.uint8)
    write_index_mask(destination, mask)


def convert_yolo_txt_to_voc_xml(
    txt_path,
    img_path,
    class_names,
    output_xml_path,
    *,
    image_filename=None,
    format_hint="auto",
):
    width, height = image_size_from_file(img_path)
    annotations = read_yolo_labels(
        txt_path, ImageSize(width, height), strict=True, format_hint=format_hint
    )
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = image_filename or Path(img_path).name
    size = ET.SubElement(root, "size")
    for name, value in (("width", width), ("height", height), ("depth", 3)):
        ET.SubElement(size, name).text = str(value)
    ET.SubElement(root, "segmented").text = "0"
    for annotation in annotations:
        x0, y0, x1, y1 = bounds(annotation.points_px)
        left, top = max(1, int(np.floor(x0)) + 1), max(1, int(np.floor(y0)) + 1)
        right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if right < left or bottom < top:
            raise ValueError(f"Объект вне изображения: {txt_path}")
        obj = ET.SubElement(root, "object")
        name = (
            class_names.get(annotation.class_id, str(annotation.class_id))
            if isinstance(class_names, dict)
            else class_names[annotation.class_id]
            if annotation.class_id < len(class_names)
            else str(annotation.class_id)
        )
        ET.SubElement(obj, "name").text = str(name)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        for key, value in zip(
            ("xmin", "ymin", "xmax", "ymax"), (left, top, right, bottom)
        ):
            ET.SubElement(box, key).text = str(value)
    with atomic_path(output_xml_path) as temporary:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)


def build_voc_from_yolo(
    yaml_file, output_root, *, binary_masks=False, format_hint="auto"
):
    source = Path(yaml_file).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("YAML должен содержать объект.")
    root = Path(config.get("path") or source.parent)
    if not root.is_absolute():
        root = source.parent / root
    destination = Path(output_root).resolve() / "VOC"
    if destination.exists():
        raise FileExistsError(f"Выходная папка уже существует: {destination}")
    sources = {}
    for split in ("train", "val", "test"):
        raw = config.get(split + "_images", config.get(split))
        if split == "val":
            raw = raw or config.get("valid") or config.get("validation")
        sources[split] = [
            (Path(image), resolved)
            for resolved in resolve_split_sources(raw, root, strict=True)
            for image in collect_image_paths(
                resolved, recursive=Path(resolved).is_dir()
            )
        ]
    if not sources["train"]:
        raise ValueError("Не найдены train-изображения.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = config.get("names") or config.get("class_names") or {}
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    from utils_new.import_export.mask_yaml import _mask_split_pairs

    selection = config.get("_dataset_sources")
    selected = selection.get("annotations", {}) if selection else None
    kind = selection.get("kind") if selection else None
    mask_pairs = {}
    if kind == "unet" or (
        not selection
        and (
            config.get("masks")
            or any(k.endswith("_masks") for k in config)
            or (root / "masks").is_dir()
        )
    ):
        for split in sources:
            mask_pairs.update(_mask_split_pairs(config, root, split, strict=False))
    memberships = {}
    for split, entries in sources.items():
        for image, _ in entries:
            image = image.resolve()
            if image in memberships and memberships[image] != split:
                raise ValueError(f"Изображение входит в несколько split: {image}")
            memberships[image] = split
    identities, used = {}, set()
    max_pixel = 0
    pixel_names = list(config.get("class_names") or [])
    xml_names = names or {
        i: name
        for i, name in enumerate(pixel_names[config.get("mask_class_offset", 1) :])
    }
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".voc-"
    ) as temporary:
        stage = Path(temporary) / "VOC"
        for folder in (
            "JPEGImages",
            "SegmentationClass",
            "Annotations",
            "ImageSets/Segmentation",
            "ImageSets/Main",
            "ImageSets/All",
        ):
            (stage / folder).mkdir(parents=True)
        for split, entries in sources.items():
            split_ids, segmentation_ids, all_ids = [], [], []
            for image, resolved in entries:
                image = image.resolve()
                if image not in identities:
                    identity = image.stem
                    if identity.casefold() in used:
                        identity += (
                            "_" + hashlib.sha256(str(image).encode()).hexdigest()[:12]
                        )
                    if identity.casefold() in used:
                        raise ValueError("Коллизия VOC ID")
                    identities[image] = identity
                    used.add(identity.casefold())
                    # Preserve image bytes and bit depth; do not silently JPEG-encode.
                    filename = identity + image.suffix
                    shutil.copy2(image, stage / "JPEGImages" / filename)
                    if selected is not None:
                        label = selected.get(str(image)) if kind == "yolo" else None
                    else:
                        label = _find_label_for_image(
                            str(image),
                            str(root),
                            split,
                            resolved if Path(resolved).is_dir() else None,
                        )
                    if label:
                        convert_yolo_txt_to_voc_xml(
                            label,
                            image,
                            xml_names,
                            stage / "Annotations" / (identity + ".xml"),
                            image_filename=filename,
                            format_hint=format_hint,
                        )
                    mask_path = mask_pairs.get(str(image))
                    if mask_path:
                        mask = read_index_mask(mask_path)
                        width, height = image_size_from_file(image)
                        if mask.shape != (height, width):
                            raise ValueError(f"Размер маски не совпадает: {mask_path}")
                        if binary_masks:
                            mask = (mask > 0).astype(np.uint8)
                        max_pixel = max(max_pixel, int(mask.max()))
                        write_index_mask(
                            stage / "SegmentationClass" / (identity + ".png"), mask
                        )
                identity = identities[image]
                if identity not in all_ids:
                    all_ids.append(identity)
                if (
                    stage / "Annotations" / (identity + ".xml")
                ).is_file() and identity not in split_ids:
                    split_ids.append(identity)
                if (
                    stage / "SegmentationClass" / (identity + ".png")
                ).is_file() and identity not in segmentation_ids:
                    segmentation_ids.append(identity)
            for folder, ids in (
                ("Main", split_ids),
                ("Segmentation", segmentation_ids),
                ("All", all_ids),
            ):
                write_text_atomic(
                    stage / "ImageSets" / folder / (split + ".txt"),
                    "\n".join(ids) + ("\n" if ids else ""),
                )
        count = max_pixel + 1
        if binary_masks:
            count, pixel_names = 2, ["background", "object"]
        else:
            count = max(count, len(pixel_names), int(config.get("num_classes", 0)))
            if not pixel_names:
                offset = int(config.get("mask_class_offset", 1))
                mapping = names if isinstance(names, dict) else dict(enumerate(names))
                count = max(count, max(mapping, default=-1) + offset + 1)
                pixel_names = ["background"] + [
                    str(mapping.get(i - offset, f"class_{i}")) for i in range(1, count)
                ]
            elif len(pixel_names) < count:
                raise ValueError("Имена масок не покрывают обнаруженные pixel IDs")
        output_config = dict(
            jpeg_dir="JPEGImages",
            seg_dir="SegmentationClass",
            num_classes=count,
            class_names=pixel_names,
            names=names,
            mask_class_offset=1 if binary_masks else config.get("mask_class_offset", 1),
        )
        for split in sources:
            output_config[split + "_txt"] = f"ImageSets/Segmentation/{split}.txt"
        write_text_atomic(
            stage / "voc.yaml",
            yaml.safe_dump(output_config, allow_unicode=True, sort_keys=False),
        )
        stage.rename(destination)
    return str(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="YOLO/индексные маски → VOC. OBB/Segment в XML становятся охватывающими BBox."
    )
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--output", default="./VOC_output")
    parser.add_argument(
        "--binary-masks",
        action="store_true",
        help="Явно объединить все положительные ID в класс 1",
    )
    parser.add_argument(
        "--format-hint",
        default="auto",
        choices=["auto", "detect", "obb_corners", "obb_xywhr", "segment"],
    )
    args = parser.parse_args(argv)
    print(
        build_voc_from_yolo(
            args.yaml,
            args.output,
            binary_masks=args.binary_masks,
            format_hint=args.format_hint,
        )
    )
    return 0


if __name__ == "__main__":
    from utils_new.training.protocol import configure_worker_streams

    configure_worker_streams()
    raise SystemExit(main())
