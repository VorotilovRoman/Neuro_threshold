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
    index_mask_files,
)
from utils_new.import_export.dataset_output import write_index_mask
from utils_new.import_export.yolo_labels import read_yolo_labels
from utils_new.import_export.yolo_yaml import (
    resolve_split_sources,
    _find_label_for_image,
    _local_image_path,
)


def read_voc_pairs(txt_path, jpeg_dir, seg_dir):
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
        mask = Path(seg_dir) / (name + ".png")
        if not mask.is_file():
            raise FileNotFoundError(f"Нет маски: {mask}")
        image = images[0].resolve()
        if image not in seen:
            pairs.append((str(image), str(mask.resolve())))
            seen.add(image)
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
    identities = {}
    used = {}
    mask_indices = {}
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
        ):
            (stage / folder).mkdir(parents=True)
        for split, entries in sources.items():
            split_ids = []
            segmentation_ids = []
            for image, resolved in entries:
                image = image.resolve()
                if image not in identities:
                    identity = image.stem
                    if identity in used:
                        identity += (
                            "_" + hashlib.sha256(str(image).encode()).hexdigest()[:12]
                        )
                    if identity in used:
                        raise ValueError("Коллизия VOC ID.")
                    identities[image] = identity
                    used[identity] = image
                    filename = identity + image.suffix
                    shutil.copy2(image, stage / "JPEGImages" / filename)
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
                            names,
                            stage / "Annotations" / (identity + ".xml"),
                            image_filename=filename,
                            format_hint=format_hint,
                        )
                    mask_dirs = config.get(split + "_masks", str(Path("masks") / split))
                    if not isinstance(mask_dirs, str):
                        raise ValueError(
                            "VOC экспорт требует один mask-каталог на split."
                        )
                    mask_root = Path(mask_dirs)
                    mask_root = (
                        mask_root if mask_root.is_absolute() else root / mask_root
                    )
                    if mask_root not in mask_indices:
                        mask_indices[mask_root] = (
                            index_mask_files(mask_root) if mask_root.is_dir() else {}
                        )
                    local = _local_image_path(
                        image, resolved if Path(resolved).is_dir() else None, root
                    )
                    match = mask_indices[mask_root].get(
                        local.with_suffix("").as_posix().casefold()
                    )
                    found = [match] if match is not None else []
                    if found:
                        convert_mask_to_voc_format(
                            found[0],
                            stage / "SegmentationClass" / (identity + ".png"),
                            binary=binary_masks,
                        )
                    elif split + "_masks" in config:
                        raise FileNotFoundError(f"Нет маски: {image}")
                if (
                    stage / "Annotations" / (identities[image] + ".xml")
                ).is_file() and identities[image] not in split_ids:
                    split_ids.append(identities[image])
                if (
                    stage / "SegmentationClass" / (identities[image] + ".png")
                ).is_file() and identities[image] not in segmentation_ids:
                    segmentation_ids.append(identities[image])
            write_text_atomic(
                stage / "ImageSets/Segmentation" / (split + ".txt"),
                "\n".join(segmentation_ids) + "\n",
            )
            write_text_atomic(
                stage / "ImageSets/Main" / (split + ".txt"), "\n".join(split_ids) + "\n"
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
    raise SystemExit(main())
