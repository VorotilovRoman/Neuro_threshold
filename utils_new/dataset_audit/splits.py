"""Explicit YAML split membership, including overlapping paths and text lists."""

from pathlib import Path
import yaml


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
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, dict):
                item = item.get("images")
            if not item:
                continue
            if not isinstance(item, str):
                raise ValueError("Путь split должен быть строкой.")
            path = Path(item)
            path = path if path.is_absolute() else root / path
            if path.is_file() and path.suffix.lower() == ".txt":
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        listed = Path(line)
                        yield (
                            listed if listed.is_absolute() else root / listed
                        ).resolve()
            else:
                yield path.resolve()

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
