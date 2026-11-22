# -*- coding: utf-8 -*-
"""YOLO data.yaml discovery without wildcard/legacy imports."""

import os

import yaml
from utils_new.import_export.images import collect_image_paths, IMAGE_EXTENSIONS


def _resolve_relative_to_yaml(raw_path, yaml_path):
    """Абсолютный путь: если raw_path абсолютный — берём как есть,
    иначе — относительно директории YAML."""
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(yaml_path)), raw_path)
    )


def resolve_split_sources(split_val, base_path, *, strict=False):
    """Resolve dataset dirs/images/TXT lists without widening list membership.

    Bare relative entries prefer the dataset root, then the enclosing list.
    Explicit ./ and ../ entries are relative to their enclosing TXT list.
    Nested lists use the same rule and cycles are always rejected.
    """
    from pathlib import Path

    root = Path(base_path).resolve()
    result, active = [], set()

    def visit(value, parent=None):
        if not isinstance(value, str):
            raise ValueError("Split entries must be paths")
        path = Path(value)
        if not path.is_absolute():
            if parent is not None and value.startswith(("./", "../")):
                path = parent / path
            else:
                candidate = root / path
                path = (
                    parent / path
                    if parent is not None and not candidate.exists()
                    else candidate
                )
        path = path.resolve()
        if path.is_dir() or path.suffix.lower() in IMAGE_EXTENSIONS:
            if strict and not path.exists():
                raise ValueError(f"Не найден источник датасета: {path}")
            result.append(str(path))
        elif path.is_file() and path.suffix.lower() == ".txt":
            if path in active:
                raise ValueError(f"Циклический TXT-список датасета: {path}")
            active.add(path)
            try:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    value = line.strip()
                    if value and not value.startswith("#"):
                        visit(value, path.parent)
            finally:
                active.remove(path)
        elif strict:
            raise ValueError(
                f"Не найден или не поддерживается источник датасета: {path}"
            )

    if split_val:
        for value in split_val if isinstance(split_val, (list, tuple)) else [split_val]:
            visit(value)
    return list(dict.fromkeys(result))


def _local_image_path(image, source, root):
    from pathlib import Path

    image = Path(image).resolve()
    if source is not None and Path(source).is_dir():
        return image.relative_to(Path(source).resolve())
    try:
        local = image.relative_to(Path(root).resolve() / "images")
        if local.parts[0] in ("train", "val", "test", "valid", "validation"):
            local = Path(*local.parts[1:])
        return local
    except ValueError:
        return Path(image.name)


def _find_label_for_image(img_path, base_path, split, rel_dir):
    """
    Ищет .txt для изображения в стандартных местах YOLO-структуры.
    Возвращает путь или None.
    """
    img_path_abs = os.path.abspath(img_path)
    d = os.path.dirname(img_path_abs)
    base = os.path.splitext(os.path.basename(img_path_abs))[0]

    # 1) рядом с изображением
    same = os.path.join(d, base + ".txt")
    if os.path.exists(same):
        return same

    # 2) подстановка images → labels в пути
    norm = img_path_abs.replace("\\", "/")
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p == "images":
            new_parts = parts[:i] + ["labels"] + parts[i + 1 :]
            new_parts[-1] = base + ".txt"
            cand = os.sep.join(new_parts)
            if os.path.exists(cand):
                return cand

    local = _local_image_path(img_path, rel_dir, base_path)
    canonical_split = "val" if split in ("valid", "validation") else split
    canonical = os.path.join(
        base_path, "labels", canonical_split, str(local.with_suffix(".txt"))
    )
    if os.path.isfile(canonical):
        return canonical

    # 3) стандартные относительные пути от base_path
    for lbl_dir in [
        os.path.join(base_path, "labels", split),
        os.path.join(base_path, "labels", os.path.basename(rel_dir))
        if rel_dir
        else None,
        os.path.join(base_path, "labels"),
        os.path.join(os.path.dirname(d), "labels"),
    ]:
        if not lbl_dir:
            continue
        cand = os.path.join(lbl_dir, base + ".txt")
        if os.path.exists(cand):
            return cand
    return None


def find_mask_for_image(img_path, base_path=None):
    """Prefer lossless masks/ sidecars over alternate image encodings beside a photo."""
    from pathlib import Path

    image = Path(img_path).resolve()
    folders = []
    for i, part in enumerate(image.parts[:-1]):
        if part == "images":
            folders.append(Path(*image.parts[:i], "masks", *image.parts[i + 1:-1]))
    if base_path:
        folders.append(Path(base_path) / "masks")
    folders.extend([image.parent.parent / "masks", image.parent])
    seen = set()
    for folder in folders:
        if folder in seen:
            continue
        seen.add(folder)
        candidates = set()
        for extension in (".png", ".tif", ".tiff", ".bmp"):
            for suffix in (extension, extension.upper()):
                candidate = (folder / (image.stem + suffix)).resolve()
                if candidate.is_file() and candidate != image:
                    candidates.add(candidate)
        if len(candidates) > 1:
            raise ValueError(f"Неоднозначная маска для {image}: {sorted(map(str, candidates))}")
        if candidates:
            return str(candidates.pop())
    return None


def load_yolo_yaml(yaml_path, log_cb=None):
    """
    Парсит YOLO data.yaml.

    Возвращает:
        pairs:      [(img_path, label_path_or_None), ...]
        names:      {cls_id: name} | None
        base_path:  str — абсолютный корень датасета
        splits_info: {'train': n, 'val': n, 'test': n} — сколько найдено в каждом
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML не содержит словаря")

    root_raw = cfg.get("path", "")
    if root_raw:
        base_path = _resolve_relative_to_yaml(root_raw, yaml_path)
    else:
        base_path = os.path.dirname(os.path.abspath(yaml_path))

    names = cfg.get("names")
    # names может быть списком ['cat','dog'] или словарём {0:'cat',1:'dog'}
    names_dict = None
    if isinstance(names, dict):
        try:
            names_dict = {int(k): str(v) for k, v in names.items()}
        except Exception:
            names_dict = None
    elif isinstance(names, (list, tuple)):
        names_dict = {i: str(v) for i, v in enumerate(names)}

    # Секции split'ов, которые поддерживаем
    split_keys = [
        ("train", cfg.get("train")),
        ("val", cfg.get("val")),
        ("valid", cfg.get("valid")),
        ("validation", cfg.get("validation")),
        ("test", cfg.get("test")),
    ]

    pairs = []
    splits_info = {}
    seen_paths = set()

    for split, split_val in split_keys:
        if not split_val:
            continue
        sources = resolve_split_sources(split_val, base_path)
        n_in_split = 0
        for source in sources:
            for img_path in collect_image_paths(
                source, recursive=os.path.isdir(source)
            ):
                if not os.path.isfile(img_path):
                    if log_cb:
                        log_cb(f"Image file not found: {img_path}")
                    continue
                key = os.path.normcase(os.path.realpath(img_path))
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                lbl = _find_label_for_image(
                    img_path,
                    base_path,
                    split,
                    source if os.path.isdir(source) else None,
                )
                pairs.append((img_path, lbl))
                n_in_split += 1
        splits_info[split] = n_in_split

    return pairs, names_dict, base_path, splits_info


# --- Shared save destinations for editor/viewer (read locations may differ). ---
def build_yolo_save_target_map(yaml_path, image_paths, *, root=None):
    """Resolve canonical labels/<split>/<image>.txt destinations for YOLO YAML.

    Does not use the old .txt location found during loading. This is important
    when some labels are missing or were stored beside images. Files are not
    created until an explicit Save/Save All confirmation.
    """
    from pathlib import Path

    yaml_path = str(yaml_path)
    with open(yaml_path, "r", encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YOLO YAML mapping: {yaml_path}")
    if root is None:
        root = (
            _resolve_relative_to_yaml(cfg.get("path"), yaml_path)
            if cfg.get("path")
            else os.path.dirname(os.path.abspath(yaml_path))
        )
    root_path = Path(root).resolve()
    images = {
        os.path.normcase(os.path.abspath(str(image))): str(image)
        for image in image_paths
    }
    targets = {}
    # User-facing output uses three canonical splits even for the synonyms
    # 'valid' and 'validation'. Input directories themselves may use aliases.
    split_values = [
        ("train", "train"),
        ("val", "val"),
        ("valid", "val"),
        ("validation", "val"),
        ("test", "test"),
    ]
    for split_key, output_split in split_values:
        sources = resolve_split_sources(cfg.get(split_key), str(root_path))
        for source in sources:
            source_path = Path(source).resolve()
            for normalized, original in images.items():
                if original in targets:
                    continue
                image_file = Path(original).resolve()
                if source_path.is_dir():
                    try:
                        image_file.relative_to(source_path)
                    except ValueError:
                        continue
                elif image_file != source_path:
                    continue
                local = _local_image_path(image_file, source_path, str(root_path))
                targets[original] = str(
                    root_path / "labels" / output_split / local.with_suffix(".txt")
                )
    if len(targets) != len(images):
        missing = [image for image in images.values() if image not in targets]
        raise ValueError(
            "Cannot determine YAML split for saving labels: " + ", ".join(missing[:5])
        )
    reversed_targets = {}
    for image, target in targets.items():
        identity = os.path.normcase(os.path.abspath(target))
        if identity in reversed_targets:
            raise ValueError(
                "Two images would overwrite one YOLO label: "
                f"{reversed_targets[identity]} and {image} -> {target}"
            )
        reversed_targets[identity] = image
    return targets
