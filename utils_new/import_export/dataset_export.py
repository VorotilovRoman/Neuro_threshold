"""Transactional filtered-subset export with explicit type/remap/split contracts."""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import yaml
from utils_new.annotations import ImageSize, Annotation, from_legacy
from utils_new.import_export.images import read_index_mask, write_image
from utils_new.import_export.yolo_labels import write_yolo_labels
from utils_new.dataset_audit.objects import load_pair_annotations, image_size


def infer_split(path):
    # Match directory components, never filenames like 'contest.jpg' or 'training'.
    parts = str(path).replace("\\", "/").split("/")[:-1]
    for part in reversed(parts):
        part = part.lower()
        if part in ("val", "valid", "validation"):
            return "val"
        if part in ("train", "test"):
            return part
    return "unknown"


def export_subset(
    filtered_pairs,
    out_dir,
    class_names,
    progress_cb=None,
    log_cb=None,
    included_class_ids=None,
    *,
    dataset_type=0,
    input_format="auto",
    cancel_check=None,
    split_by_image=None,
):
    target = Path(out_dir).resolve()
    if dataset_type not in (0, 1, 2):
        raise ValueError('Неизвестный вид датасета.')
    existed = target.exists()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(
            "Выберите отдельную пустую папку: существующий датасет не перезаписывается."
        )
    if not filtered_pairs:
        raise ValueError("Нет пар для экспорта.")
    names = {int(key): str(value) for key, value in class_names.items()}
    ids = sorted(
        names if included_class_ids is None else (set(names) & set(included_class_ids))
    )
    if dataset_type != 2 and not ids:
        raise ValueError("Не выбраны классы для экспорта.")
    if dataset_type == 1 and (len(ids) > 65535 or any(i <= 0 for i in ids)):
        raise ValueError(
            "Для масок нужны положительные pixel IDs; 0 — фон, максимум 65535 классов."
        )
    mapping = {old: new for new, old in enumerate(ids)}
    entries = []
    outputs = set()
    for pair in filtered_pairs:
        image = Path(pair[0])
        aux = pair[1] if len(pair) > 1 else None
        size = image_size(image)
        membership = set((split_by_image or {}).get(str(image), []))
        if len(membership) > 1:
            raise ValueError(f"Изображение входит в несколько split: {image}")
        split = next(iter(membership)) if membership else infer_split(image)
        if split == "unknown":
            split = "train"
        image_rel = Path("images") / split / image.name
        label_rel = (
            Path("masks" if dataset_type == 1 else "labels")
            / split
            / (image.stem + (".png" if dataset_type == 1 else ".txt"))
        )
        for rel in [image_rel] if dataset_type == 2 or aux is None else [image_rel, label_rel]:
            key = os.path.normcase(str(rel))
            if key in outputs:
                raise ValueError(f"Коллизия выходных имён: {rel}")
            outputs.add(key)
        if dataset_type != 2 and aux is not None and not Path(aux).is_file():
            raise FileNotFoundError(f"Нет разметки: {aux}")
        entries.append((image, aux, size, image_rel, label_rel))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=target.parent, prefix=".dataset-export-"))
    try:
        for split in ("train", "val", "test"):
            (stage / "images" / split).mkdir(parents=True)
            if dataset_type != 2:
                (stage / ("masks" if dataset_type == 1 else "labels") / split).mkdir(
                    parents=True
                )
        for index, (image, aux, size, image_rel, label_rel) in enumerate(entries):
            if cancel_check and cancel_check():
                from concurrent.futures import CancelledError

                raise CancelledError("Экспорт отменён.")
            shutil.copy2(image, stage / image_rel)
            if aux is None:
                pass  # Preserve an unlabelled image; do not invent a background label/mask.
            elif dataset_type == 0:
                annotations = []
                for value in load_pair_annotations(image, aux, input_format):
                    ann = from_legacy(value, ImageSize(*size))
                    if ann.class_id in mapping:
                        annotations.append(
                            Annotation(ann.kind, mapping[ann.class_id], ann.points_px)
                        )
                write_yolo_labels(stage / label_rel, annotations, ImageSize(*size))
            elif dataset_type == 1:
                mask = read_index_mask(aux)
                if mask.shape != (size[1], size[0]):
                    raise ValueError(f"Размер маски не совпадает: {aux}")
                mapped = np.zeros(
                    mask.shape, np.uint8 if len(ids) <= 255 else np.uint16
                )
                for old, new in mapping.items():
                    mapped[mask == old] = new + 1
                write_image(stage / label_rel, mapped)
            elif dataset_type != 2:
                raise ValueError("Неизвестный вид датасета.")
            if progress_cb:
                progress_cb(index + 1, len(entries))
        config = {
            "path": str(target),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(ids),
            "names": {mapping[old]: names[old] for old in ids},
        }
        if dataset_type == 1:
            config.update(
                mask_class_offset=1,
                class_names=["background", *[names[old] for old in ids]],
                num_classes=len(ids) + 1,
                masks={
                    "train": "masks/train",
                    "val": "masks/val",
                    "test": "masks/test",
                },
            )
        (stage / "data.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        if cancel_check and cancel_check():
            from concurrent.futures import CancelledError

            raise CancelledError("Экспорт отменён.")
        if target.exists():
            target.rmdir()
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        if existed and not target.exists():
            target.mkdir()
        raise
    if log_cb:
        log_cb(f"Экспортировано {len(entries)} снимков: {target}")
    return str(target / "data.yaml")
