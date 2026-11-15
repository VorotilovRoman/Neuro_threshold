"""Read an immutable manual YOLO reference independently of report locations."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import yaml
from PIL import Image
from utils_new.annotations import ImageSize, to_legacy
from utils_new.dataset_yolo_labels_new import read_yolo_labels
from .geometry_metrics import normalize_annotations

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def path_parts(value):
    return tuple(
        part
        for part in str(value).replace("\\", "/").split("/")
        if part not in ("", ".")
    )


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


@dataclass(frozen=True)
class TruthSample:
    key: str
    image_path: str | None
    label_path: str
    size: tuple
    annotations: tuple


@dataclass(frozen=True)
class GroundTruth:
    samples: tuple
    source: str
    format_hint: str = "auto"


def _read_sample(key, image, label, size, format_hint):
    if not label.is_file():
        raise ValueError(
            f"Нет эталонной разметки: {label}. Пустой TXT допустим, отсутствующий — нет."
        )
    messages = []
    annotations = read_yolo_labels(
        label, ImageSize(*size), format_hint=format_hint, log=messages.append
    )
    if any("запись пропущена" in message for message in messages):
        raise ValueError(f"Ошибка эталонной разметки {label}: {messages}")
    legacy = tuple(to_legacy(ann, ImageSize(*size)) for ann in annotations)
    normalize_annotations(legacy, size)
    return TruthSample(key, str(image) if image else None, str(label), size, legacy)


def _images(entry, root, visited=None):
    visited = set() if visited is None else visited
    for value in entry if isinstance(entry, list) else [entry]:
        if not isinstance(value, str):
            raise ValueError("Путь изображения/split в YAML должен быть строкой.")
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise ValueError(f"Не найден источник эталона: {path}")
        if path.is_dir():
            yield from sorted(
                p
                for p in path.rglob("*")
                if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
            )
        elif path.suffix.lower() == ".txt":
            identity = path.resolve()
            if identity in visited:
                raise ValueError(f"Циклический TXT-список эталона: {path}")
            visited.add(identity)
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    yield from _images(line.strip(), root, visited)
            visited.remove(identity)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
        else:
            raise ValueError(f"Неизвестный источник эталона: {path}")


def _label_for(image, root):
    candidates = [image.with_suffix(".txt")]
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part == "images":
            candidates.append(
                Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
            )
    try:
        relative = image.relative_to(root / "images")
    except ValueError:
        relative = (
            image.relative_to(root) if image.is_relative_to(root) else Path(image.name)
        )
    candidates += [(root / "labels" / relative).with_suffix(".txt")]
    found = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(found) > 1:
        raise ValueError(f"Неоднозначная разметка для {image}: {found}")
    return found[0] if found else candidates[0]


def _labels_only(root, reports, format_hint, cancel_check):
    if reports is None or reports.empty:
        raise ValueError(
            "В папке только TXT. Сначала загрузите отчёты: нужны размеры и имена изображений."
        )
    references = {}
    for _, row in reports.iterrows():
        if cancel_check():
            raise InterruptedError("Проверка эталона отменена.")
        path = str(row["image_path"])
        size = tuple(json.loads(row["source_size"]))
        if path in references and references[path] != size:
            raise ValueError(f"Отчёты расходятся в размерах: {path}")
        references[path] = size
    samples = []
    labels = sorted(
        p
        for p in root.rglob("*.txt")
        if not p.name.endswith(("_report.txt", ".partial.txt"))
    )
    for label in labels:
        if cancel_check():
            raise InterruptedError("Проверка эталона отменена.")
        key = label.relative_to(root).with_suffix("").as_posix()
        target = path_parts(key)
        matches = []
        for path, size in references.items():
            parts = path_parts(path)
            stem = (*parts[:-1], Path(parts[-1]).stem)
            if stem[-len(target) :] == target:
                matches.append((path, size))
        if not matches:
            matches = [
                (path, size)
                for path, size in references.items()
                if Path(path_parts(path)[-1]).stem == label.stem
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Для {label} нужны ровно один снимок/размер из отчётов; найдено {len(matches)}."
            )
        path, size = matches[0]
        sample_key = (
            label.relative_to(root)
            .with_suffix(Path(path_parts(path)[-1]).suffix)
            .as_posix()
        )
        samples.append(_read_sample(sample_key, None, label, size, format_hint))
    return samples


def load_ground_truth(
    source, *, reports=None, format_hint="auto", cancel_check=lambda: False
):
    path = Path(source).resolve()
    if not path.exists():
        raise ValueError(f"Эталон не найден: {path}")
    if path.is_file():
        if path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Выберите YOLO YAML или папку с изображениями/TXT.")
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Некорректный YAML: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("YAML эталона должен содержать объект.")
        root = Path(config.get("path") or path.parent)
        if not root.is_absolute():
            root = path.parent / root
        root = root.resolve()
        files = []
        for split in ("train", "val", "valid", "validation", "test"):
            if config.get(split):
                files.extend(_images(config[split], root))
    else:
        root = path
        image_root = root / "images" if (root / "images").is_dir() else root
        files = [
            p
            for p in image_root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_SUFFIXES
            and not {"labels", "masks"} & set(p.relative_to(image_root).parts[:-1])
        ]
    samples = []
    for image in sorted({p.resolve() for p in files}):
        if cancel_check():
            raise InterruptedError("Оценка отменена.")
        with Image.open(image) as source_image:
            size = tuple(source_image.size)
        base = root / "images" if image.is_relative_to(root / "images") else root
        key = (
            image.relative_to(base).as_posix()
            if image.is_relative_to(base)
            else image.as_posix()
        )
        samples.append(
            _read_sample(key, image, _label_for(image, root), size, format_hint)
        )
    if not samples and path.is_dir():
        samples = _labels_only(root, reports, format_hint, cancel_check)
    if not samples:
        raise ValueError("Эталонный набор пуст.")
    if len({sample.key for sample in samples}) != len(samples):
        raise ValueError("Имена изображений эталона неоднозначны.")
    return GroundTruth(tuple(samples), str(path), format_hint)


def match_sample(image_path, truth):
    """Exact path, then longest unique portable suffix; never guess duplicate basenames."""
    parts = path_parts(image_path)
    exact = [
        sample
        for sample in truth.samples
        if sample.image_path and path_parts(sample.image_path) == parts
    ]
    if len(exact) == 1:
        return exact[0]
    ranked = []
    for sample in truth.samples:
        target = path_parts(sample.image_path or sample.key)
        count = 0
        for left, right in zip(reversed(parts), reversed(target)):
            if left != right:
                break
            count += 1
        if count:
            ranked.append((count, sample))
    if not ranked:
        return None
    best = max(count for count, _ in ranked)
    matches = [sample for count, sample in ranked if count == best]
    if len(matches) > 1:
        raise ValueError(
            f"Неоднозначное имя снимка отчёта: {image_path}. Сохраните структуру каталогов."
        )
    return matches[0]
