"""Explicit YAML split membership, including overlapping paths and text lists."""

from pathlib import Path
import yaml
from utils_new.import_export.yolo_yaml import resolve_split_sources


def split_memberships(yaml_path, pairs):
    if not yaml_path:
        return {}
    source = Path(yaml_path).resolve()
    try:
        config = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Некорректный YAML split: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("YAML должен содержать объект.")
    root = Path(config.get("path") or source.parent)
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    images = {str(pair[0]): Path(pair[0]).resolve() for pair in pairs}
    result = {}

    def entries(value):
        values = value if isinstance(value, (list, tuple)) else [value]
        values = [
            item.get("images") if isinstance(item, dict) else item for item in values
        ]
        return [
            Path(path) for path in resolve_split_sources([v for v in values if v], root)
        ]

    for key, split in [
        ("train", "train"),
        ("val", "val"),
        ("valid", "val"),
        ("validation", "val"),
        ("test", "test"),
    ]:
        for entry in entries(config.get(key) or config.get(key + "_images")):
            for original, path in images.items():
                if path == entry or (entry.is_dir() and path.is_relative_to(entry)):
                    result.setdefault(original, set()).add(split)
    return result
