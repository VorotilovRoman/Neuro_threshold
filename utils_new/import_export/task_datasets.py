"""Pose / image classification sessions, standard YOLO/COCO boundaries; Qt-free."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import CancelledError
import json
import shutil
import tempfile
import numpy as np
import yaml

from utils_new.annotations.model import (
    Annotation,
    PoseAnnotation,
    PoseSchema,
    ImageClassification,
    ImageSize,
)
from utils_new.annotations.geometry import rectangle_points, bounds
from utils_new.import_export.images import collect_image_paths, image_size_from_file
from utils_new.import_export.json_objects import read_json_object, write_json_atomic
from utils_new.import_export.atomic import write_text_atomic


def check(cancel=None):
    if cancel is not None and cancel.is_set():
        raise CancelledError()


def classes_dict(value):
    result = (
        {i: str(n).strip() for i, n in enumerate(value)}
        if isinstance(value, list)
        else {int(i): str(n).strip() for i, n in value.items()}
    )
    if (
        not result
        or min(result) < 0
        or any(
            not n or n in (".", "..") or any(c in n for c in "/\\\0")
            for n in result.values()
        )
        or len(set(result.values())) != len(result)
    ):
        raise ValueError("Нужны уникальные безопасные имена классов и ID >=0.")
    return result


def schema_from_yaml(cfg):
    shape = cfg.get("kpt_shape")
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or type(shape[0]) is not int
        or shape[0] < 1
    ):
        raise ValueError("Pose YAML требует kpt_shape: [число_точек, 2 или 3].")
    names = cfg.get("kpt_names")
    if isinstance(names, dict):
        choices = list(names.values())
        if choices and any(v != choices[0] for v in choices):
            raise ValueError(
                "Редактор требует общую схему имен точек для классов; импортируйте классы с разными схемами отдельно."
            )
        names = choices[0] if choices else None
    if names is None:
        names = [f"point_{i+1}" for i in range(shape[0])]
    if len(names) != shape[0]:
        raise ValueError("Число имен точек не равно kpt_shape.")
    # Project extension uses zero-based edges. COCO skeleton is converted at its boundary.
    return PoseSchema(
        tuple(names),
        tuple(map(tuple, cfg.get("neuro_pose_edges", []))),
        cfg.get("flip_idx"),
        shape[1],
    )


def decode_pose(text, size, schema):
    poses = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            tokens = line.split()
            cls = int(tokens[0])
            values = np.asarray(list(map(float, tokens[1:])))
            if (
                len(tokens) != 5 + len(schema.names) * schema.dims
                or cls < 0
                or not np.isfinite(values).all()
            ):
                raise ValueError("число значений / класс / NaN")
            cx, cy, w, h = values[:4]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("некорректная рамка")
            pts = values[4:].reshape(-1, schema.dims)
            v = pts[:, 2] if schema.dims == 3 else np.full(len(pts), 2)
            if (
                not np.isin(v, [0, 1, 2]).all()
                or ((pts[v > 0, :2] < 0) | (pts[v > 0, :2] > 1)).any()
            ):
                raise ValueError("координаты / видимость точек")
            xy = pts[:, :2] * [size.width, size.height]
            xy[v == 0] = 0
            bbox = Annotation(
                "detect",
                cls,
                rectangle_points(
                    (cx - w / 2) * size.width,
                    (cy - h / 2) * size.height,
                    (cx + w / 2) * size.width,
                    (cy + h / 2) * size.height,
                ),
            )
            poses.append(PoseAnnotation(bbox, np.column_stack((xy, v))))
        except (ValueError, IndexError) as error:
            raise ValueError(f"Pose TXT, строка {line_no}: {error}") from error
    return poses


def encode_pose(poses, size, schema):
    rows = []
    for pose in poses:
        if len(pose.keypoints) != len(schema.names):
            raise ValueError("Число точек не соответствует схеме.")
        x0, y0, x1, y1 = bounds(pose.bbox.points_px)
        bbox = np.array(
            [
                (x0 + x1) / 2 / size.width,
                (y0 + y1) / 2 / size.height,
                (x1 - x0) / size.width,
                (y1 - y0) / size.height,
            ]
        )
        pts = pose.keypoints.copy()
        valid = pts[:, 2] > 0
        if ((pts[valid, :2] < 0) | (pts[valid, :2] > [size.width, size.height])).any():
            raise ValueError("Точка за пределами снимка.")
        pts[:, :2] /= [size.width, size.height]
        pts[~valid, :2] = 0
        if schema.dims == 2 and not valid.all():
            raise ValueError("Для пропущенных точек сохраните схему с dims=3.")
        values = list(bbox) + list(pts[:, : schema.dims].ravel())
        rows.append(
            str(pose.bbox.class_id) + " " + " ".join(f"{float(v):.10g}" for v in values)
        )
    text = "\n".join(rows) + ("\n" if rows else "")
    decode_pose(text, size, schema)  # Same validation in both directions.
    return text


@dataclass
class TaskSample:
    path: str
    split: str = "train"
    classification: ImageClassification = field(default_factory=ImageClassification)
    label_path: str | None = None
    label_dims: int = 3
    poses: list | None = None
    size: ImageSize | None = None


@dataclass
class TaskDataset:
    task: str
    classes: dict
    samples: list = field(default_factory=list)
    schema: PoseSchema | None = None
    source: str = ""

    def __post_init__(self):
        if self.task not in ("pose", "classify"):
            raise ValueError("Задача pose или classify.")
        self.classes = classes_dict(self.classes)
        if self.task == "pose" and not isinstance(self.schema, PoseSchema):
            raise ValueError("Pose требует схему.")
        paths = set()
        for sample in self.samples:
            sample.path = str(Path(sample.path).resolve())
            if sample.path in paths:
                raise ValueError("Один снимок повторяется в датасете / split.")
            paths.add(sample.path)
            if sample.split not in ("train", "val", "test"):
                raise ValueError("Неизвестный split.")
            if (
                sample.classification.class_id is not None
                and sample.classification.class_id not in self.classes
            ):
                raise ValueError("Неизвестный класс снимка.")
            if self.task == "pose" and sample.poses is not None:
                if any(
                    p.bbox.class_id not in self.classes
                    or len(p.keypoints) != len(self.schema.names)
                    for p in sample.poses
                ):
                    raise ValueError("Pose проекта не соответствует классам / схеме.")
                if sample.poses and sample.size is None:
                    raise ValueError("Проект Pose должен хранить размер снимка.")
                if sample.size:
                    encode_pose(
                        sample.poses,
                        sample.size,
                        PoseSchema(
                            self.schema.names,
                            self.schema.edges,
                            self.schema.flip_idx,
                            3,
                        ),
                    )

    def load_poses(self, sample):
        if sample.size is None:
            w, h = image_size_from_file(sample.path)
            sample.size = ImageSize(w, h)
        if sample.poses is None:
            parser_schema = PoseSchema(
                self.schema.names,
                self.schema.edges,
                self.schema.flip_idx,
                sample.label_dims,
            )
            sample.poses = (
                decode_pose(
                    Path(sample.label_path).read_text(encoding="utf-8-sig"),
                    sample.size,
                    parser_schema,
                )
                if sample.label_path
                else []
            )
            if any(p.bbox.class_id not in self.classes for p in sample.poses):
                raise ValueError("В TXT есть неизвестный класс.")
        return sample.poses


def load_task_dataset(path, task, *, cancel=None, progress=None):
    path = Path(path).resolve()
    check(cancel)
    if path.suffix.lower() == ".json":
        cfg = read_json_object(path)
        if cfg.get("format") == "neuro-task-dataset-v1":
            if cfg["task"] != task:
                raise ValueError("Задача проекта не соответствует выбранному режиму.")
            sch = PoseSchema(**cfg["schema"]) if task == "pose" else None
            samples = []
            for row in cfg["samples"]:
                check(cancel)
                image = Path(row["path"])
                image = image if image.is_absolute() else path.parent / image
                size = ImageSize(**row["size"]) if row.get("size") else None
                poses = (
                    None
                    if row.get("poses") is None
                    else [
                        PoseAnnotation(
                            Annotation("detect", v["class_id"], v["bbox"]),
                            np.array(v["keypoints"]),
                        )
                        for v in row["poses"]
                    ]
                )
                label = row.get("label_path")
                label = (
                    str((path.parent / label).resolve())
                    if label and not Path(label).is_absolute()
                    else label
                )
                samples.append(
                    TaskSample(
                        str(image),
                        row["split"],
                        ImageClassification(**row.get("classification", {})),
                        label,
                        row.get("label_dims", 3),
                        poses,
                        size,
                    )
                )
            return TaskDataset(task, cfg["classes"], samples, sch, str(path))
        if task == "pose" and "images" in cfg and "annotations" in cfg:
            return load_coco_pose(cfg, path, cancel)
        raise ValueError("Нужен проект разметки или COCO Keypoints.")
    if task == "classify":
        if not path.is_dir():
            raise ValueError(
                "Classification: выберите корневую папку train/val/test либо JSON проекта."
            )
        training = path / "train" if (path / "train").is_dir() else path
        names = sorted(p.name for p in training.iterdir() if p.is_dir())
        classes = classes_dict(names)
        lookup = {n: i for i, n in classes.items()}
        samples = []
        split_folders = (
            [(s, path / s) for s in ("train", "val", "test")]
            if training != path
            else [("train", path)]
        )
        for split, folder in split_folders:
            if not folder.is_dir():
                continue
            for child in sorted(folder.iterdir()):
                check(cancel)
                if not child.is_dir():
                    continue
                if child.name not in lookup:
                    raise ValueError(f"Класс {child.name} отсутствует в train.")
                for image in collect_image_paths(child, recursive=True):
                    check(cancel)
                    samples.append(
                        TaskSample(
                            image, split, ImageClassification(lookup[child.name], False)
                        )
                    )
            if progress:
                progress(len(samples), 0)
        return TaskDataset(task, classes, samples, source=str(path))
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Для Pose нужны YAML со схемой или COCO/проект JSON.")
    from utils_new.import_export.yolo_yaml import load_yolo_yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    schema = schema_from_yaml(cfg)
    pairs, names, root, _ = load_yolo_yaml(str(path), allow_pose=True)
    from utils_new.import_export.yolo_yaml import resolve_split_sources
    from utils_new.import_export.dataset_sources import metadata

    info = metadata(path)
    splits = {}
    for split in ("train", "val", "test"):
        for source in resolve_split_sources(cfg.get(split), root, strict=True):
            for image in collect_image_paths(source, recursive=True):
                check(cancel)
                if image in splits and splits[image] != split:
                    raise ValueError("Снимок повторяется между split.")
                splits[image] = split
    samples = []
    for i, (image, label) in enumerate(pairs):
        check(cancel)
        samples.append(
            TaskSample(
                image,
                splits.get(image, "train"),
                label_path=label,
                label_dims=schema.dims,
            )
        )
        if progress:
            progress(i + 1, len(pairs))
    return TaskDataset(task, names, samples, schema, str(path))


def save_task_project(dataset, path, *, cancel=None, progress=None):
    rows = []
    for i, s in enumerate(dataset.samples):
        check(cancel)
        poses = dataset.load_poses(s) if dataset.task == "pose" else None
        rows.append(
            dict(
                path=s.path,
                split=s.split,
                label_path=s.label_path,
                label_dims=s.label_dims,
                classification=asdict(s.classification),
                size=asdict(s.size) if s.size else None,
                poses=(
                    [
                        dict(
                            class_id=p.bbox.class_id,
                            bbox=p.bbox.points_px.tolist(),
                            keypoints=p.keypoints.tolist(),
                        )
                        for p in poses
                    ]
                    if poses is not None
                    else None
                ),
            )
        )
        if progress:
            progress(i + 1, len(dataset.samples))
    check(cancel)
    write_json_atomic(
        path,
        dict(
            format="neuro-task-dataset-v1",
            task=dataset.task,
            classes=dataset.classes,
            schema=asdict(dataset.schema) if dataset.schema else None,
            samples=rows,
        ),
    )


def export_task_dataset(
    dataset, destination, *, cancel=None, progress=None, coco=False
):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(
            "Укажите новую подпапку экспорта: исходные данные не перезаписываются."
        )
    if not dataset.samples:
        raise ValueError("Нет снимков.")
    usable = (
        [
            s
            for s in dataset.samples
            if (s.label_path is not None or bool(s.poses) or s.classification.reviewed)
        ]
        if dataset.task == "pose"
        else [s for s in dataset.samples if s.classification.class_id is not None]
    )
    if not all(any(s.split == split for s in usable) for split in ("train", "val")):
        raise ValueError(
            "Для экспорта обучающего датасета назначьте train и val. Неразмеченные cls не экспортируются."
        )
    if dataset.task == "classify" and len(
        {n.casefold() for n in dataset.classes.values()}
    ) != len(dataset.classes):
        raise ValueError("Имена папок классов должны различаться без учёта регистра.")
    if dataset.task == "classify" and coco:
        raise ValueError("COCO Keypoints доступен только для Pose.")
    order = (
        sorted(dataset.classes, key=lambda i: dataset.classes[i])
        if dataset.task == "classify"
        else sorted(dataset.classes)
    )
    mapping = {old: new for new, old in enumerate(order)}
    names = {mapping[i]: dataset.classes[i] for i in order}
    if dataset.task == "classify" and any(
        not any(s.split == "train" and s.classification.class_id == cls for s in usable)
        for cls in order
    ):
        raise ValueError("Для каждого класса нужны обучающие снимки train.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".task-export-", dir=destination.parent
    ) as tmp:
        out = Path(tmp) / "dataset"
        out.mkdir()
        images = []
        annotations = []
        ann_id = 1
        for split in ("train", "val", "test"):
            if dataset.task == "classify":
                for name in names.values():
                    (out / split / name).mkdir(parents=True)
        content_splits = {}
        for i, s in enumerate(usable):
            check(cancel)
            if dataset.task == "classify":
                image_size_from_file(s.path)
            import hashlib

            digest = hashlib.sha256()
            with open(s.path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    check(cancel)
                    digest.update(chunk)
            key = digest.hexdigest()
            if key in content_splits and content_splits[key] != s.split:
                raise ValueError(
                    "Одинаковые файлы снимков попали в разные split. Исправьте разделение данных."
                )
            content_splits[key] = s.split
            filename = f"{i:07d}_{Path(s.path).name}"
            folder = (
                (Path(s.split) / dataset.classes[s.classification.class_id])
                if dataset.task == "classify"
                else Path("images") / s.split
            )
            target = out / folder / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s.path, target)
            if dataset.task == "pose":
                poses = dataset.load_poses(s)
                mapped = [
                    PoseAnnotation(
                        Annotation(
                            "detect", mapping[p.bbox.class_id], p.bbox.points_px
                        ),
                        p.keypoints,
                    )
                    for p in poses
                ]
                encode_pose(mapped, s.size, dataset.schema)
                if coco:
                    images.append(
                        dict(
                            id=i,
                            file_name=str(folder / filename).replace("\\", "/"),
                            width=s.size.width,
                            height=s.size.height,
                            split=s.split,
                        )
                    )
                    for p in mapped:
                        x0, y0, x1, y1 = bounds(p.bbox.points_px)
                        kp = p.keypoints.copy()
                        kp[kp[:, 2] == 0, :2] = 0
                        annotations.append(
                            dict(
                                id=ann_id,
                                image_id=i,
                                category_id=p.bbox.class_id + 1,
                                bbox=[x0, y0, x1 - x0, y1 - y0],
                                area=(x1 - x0) * (y1 - y0),
                                iscrowd=0,
                                keypoints=kp.ravel().tolist(),
                                num_keypoints=int((kp[:, 2] > 0).sum()),
                            )
                        )
                        ann_id += 1
                else:
                    write_text_atomic(
                        out / "labels" / s.split / Path(filename).with_suffix(".txt"),
                        encode_pose(mapped, s.size, dataset.schema),
                    )
            if progress:
                progress(i + 1, len(usable))
        if dataset.task == "pose":
            if coco:
                categories = [
                    dict(
                        id=i + 1,
                        name=n,
                        keypoints=list(dataset.schema.names),
                        skeleton=[[a + 1, b + 1] for a, b in dataset.schema.edges],
                    )
                    for i, n in names.items()
                ]
                write_json_atomic(
                    out / "annotations.json",
                    dict(
                        images=images,
                        annotations=annotations,
                        categories=categories,
                        neuro_pose_schema=asdict(dataset.schema),
                    ),
                )
            else:
                cfg = dict(
                    train="images/train",
                    val="images/val",
                    names=names,
                    kpt_shape=[len(dataset.schema.names), dataset.schema.dims],
                    kpt_names={i: list(dataset.schema.names) for i in names},
                    neuro_pose_edges=dataset.schema.edges,
                )
                if any(s.split == "test" for s in usable):
                    cfg["test"] = "images/test"
                if dataset.schema.flip_idx is not None:
                    cfg["flip_idx"] = list(dataset.schema.flip_idx)
                write_text_atomic(
                    out / "data.yaml", yaml.safe_dump(cfg, allow_unicode=True)
                )
        write_json_atomic(
            out / "class_mapping.json",
            dict(
                source_to_export=mapping,
                names=names,
                unlabelled_excluded=len(dataset.samples) - len(usable),
            ),
        )
        check(cancel)
        out.rename(destination)
    return str(destination)


def load_coco_pose(cfg, path, cancel=None):
    categories = cfg["categories"]
    schema = None
    classes = {}
    mapping = {}
    for i, c in enumerate(categories):
        names = c.get("keypoints")
        if not names:
            raise ValueError("В COCO категории отсутствуют keypoints.")
        candidate = PoseSchema(
            tuple(names),
            tuple((int(a) - 1, int(b) - 1) for a, b in c.get("skeleton", [])),
        )
        if schema is not None and candidate != schema:
            raise ValueError(
                "COCO содержит несовместимые схемы: загрузите их отдельно."
            )
        schema = candidate
        classes[i] = c["name"]
        mapping[c["id"]] = i
    if cfg.get("neuro_pose_schema"):
        schema = PoseSchema(**cfg["neuro_pose_schema"])
    samples = {}
    for img in cfg["images"]:
        check(cancel)
        filename = Path(img["file_name"])
        image = filename if filename.is_absolute() else path.parent / filename
        samples[img["id"]] = TaskSample(
            str(image),
            img.get("split", "train"),
            poses=[],
            size=ImageSize(img["width"], img["height"]),
        )
    for a in cfg["annotations"]:
        check(cancel)
        sample = samples[a["image_id"]]
        x, y, w, h = a["bbox"]
        pts = np.array(a["keypoints"], dtype=float).reshape(-1, 3)
        if len(pts) != len(schema.names):
            raise ValueError("COCO: число точек не совпадает со схемой.")
        sample.poses.append(
            PoseAnnotation(
                Annotation(
                    "detect",
                    mapping[a["category_id"]],
                    rectangle_points(x, y, x + w, y + h),
                ),
                pts,
            )
        )
    return TaskDataset("pose", classes, list(samples.values()), schema, str(path))


def merge_task_datasets(current, incoming, *, cancel=None):
    """Keep IDs stable by class name, reject conflicting samples and schemata."""
    from copy import deepcopy
    from dataclasses import replace

    if current.task != incoming.task or current.schema != incoming.schema:
        raise ValueError(
            "Для объединения нужны одинаковая задача и схема Pose (имена/порядок/связи/flip_idx)."
        )
    result = deepcopy(current)
    new = deepcopy(incoming)
    lookup = {name: i for i, name in result.classes.items()}
    mapping = {}
    for cls, name in new.classes.items():
        if name not in lookup:
            idx = max(result.classes, default=-1) + 1
            result.classes[idx] = name
            lookup[name] = idx
        mapping[cls] = lookup[name]
    existing = {s.path: s for s in result.samples}
    for s in new.samples:
        check(cancel)
        if new.task == "pose":
            poses = new.load_poses(s)
            s.poses = [
                PoseAnnotation(
                    Annotation("detect", mapping[p.bbox.class_id], p.bbox.points_px),
                    p.keypoints,
                )
                for p in poses
            ]
        elif s.classification.class_id is not None:
            s.classification = replace(
                s.classification, class_id=mapping[s.classification.class_id]
            )
        old = existing.get(s.path)
        if old is None:
            result.samples.append(s)
            existing[s.path] = s
            continue
        if old.split != s.split:
            raise ValueError(f"Снимок уже находится в другом split: {s.path}")
        if new.task == "classify":
            if (
                old.classification.class_id is not None
                and s.classification.class_id is not None
                and old.classification.class_id != s.classification.class_id
            ):
                raise ValueError(f"Конфликт классов: {s.path}")
            if old.classification.class_id is None:
                old.classification = s.classification
        else:
            old_poses = result.load_poses(old)
            old_known = bool(old.label_path or old_poses or old.classification.reviewed)
            new_known = bool(s.label_path or s.poses or s.classification.reviewed)
            if old_known and new_known:
                if len(old_poses) != len(s.poses) or any(
                    a.bbox.class_id != b.bbox.class_id
                    or not np.array_equal(a.bbox.points_px, b.bbox.points_px)
                    or not np.array_equal(a.keypoints, b.keypoints)
                    for a, b in zip(old_poses, s.poses)
                ):
                    raise ValueError(f"Конфликт Pose: {s.path}")
            elif not old_known and new_known:
                old.poses = s.poses
                old.size = s.size
                old.label_path = s.label_path
                old.label_dims = s.label_dims
    return result
