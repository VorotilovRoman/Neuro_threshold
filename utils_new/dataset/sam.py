"""Lazy per-instance SAM targets from the shared, reviewed dataset sources."""

from collections import defaultdict
from concurrent.futures import CancelledError
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from utils_new.annotations import ImageSize
from utils_new.annotations.masks import annotation_binary_mask
from utils_new.images.processing import ImageStats
from utils_new.import_export.coco import decode_rle
from utils_new.import_export.dataset_sources import defaults, metadata, preview
from utils_new.import_export.ground_truth import digest_file
from utils_new.import_export.images import image_size_from_file, read_image, read_index_mask
from utils_new.import_export.yolo_labels import read_yolo_labels


def check_stop(stop):
    if stop is not None and stop.is_set():
        raise CancelledError()


@dataclass(frozen=True)
class SamSample:
    image: str
    kind: str
    target: object
    split: str = "train"

    def masks(self, cfg):
        width, height = image_size_from_file(self.image)
        valid = np.ones((height, width), bool)
        if self.kind == "yolo":
            # YAML is explicitly selected as YOLO Segment: four-corner masks are valid too.
            annotations = read_yolo_labels(self.target, ImageSize(width, height), format_hint="segment", strict=True)
            if any(ann.kind != "segment" for ann in annotations):
                raise ValueError(f"SAM требует Segment, а не BBox/OBB: {self.target}")
            masks = [annotation_binary_mask(ann, ImageSize(width, height)).astype(bool)
                     for ann in annotations]
        elif self.kind == "unet":
            labels = read_index_mask(self.target, ignore_index=cfg["ignore_index"])
            if labels.shape != valid.shape:
                raise ValueError(f"Размер маски отличается от снимка: {self.target}")
            if cfg["ignore_index"] is not None:
                valid = labels != cfg["ignore_index"]
            masks = []
            for cid in np.unique(labels[valid]):
                if cid == cfg["background_id"]:
                    continue
                count, components = cv2.connectedComponents((labels == cid).astype(np.uint8), connectivity=8)
                masks.extend(components == index for index in range(1, count))
        else:
            masks = []
            for ann in self.target:
                if ann.get("iscrowd", 0):
                    # Crowd regions are not individual targets and must not become negative labels.
                    raise ValueError(f"COCO iscrowd нельзя использовать как отдельный объект SAM: {self.image}")
                segmentation = ann.get("segmentation")
                if isinstance(segmentation, dict):
                    mask = decode_rle(segmentation, expected_shape=(height, width)).astype(bool)
                elif isinstance(segmentation, list) and segmentation:
                    mask = np.zeros((height, width), np.uint8)
                    polygons = []
                    for polygon in segmentation:
                        points = np.asarray(polygon, dtype=np.float64)
                        if points.ndim != 1 or len(points) < 6 or len(points) % 2 or not np.isfinite(points).all():
                            raise ValueError(f"Некорректный полигон COCO: {self.image}")
                        polygons.append(np.rint(points.reshape(-1, 2)).astype(np.int32))
                    cv2.fillPoly(mask, polygons, 1)
                    mask = mask.astype(bool)
                else:
                    raise ValueError(f"У объекта COCO нет маски; BBox недостаточно: {self.image}")
                masks.append(mask)
        if any(not mask.any() for mask in masks):
            raise ValueError(f"Пустая маска объекта после растеризации: {self.image}")
        return masks, valid

    def rgb(self):
        image = read_image(self.image)
        if image is None:
            raise ValueError(f"Не удалось прочитать снимок: {self.image}")
        return ImageStats.to_rgb3(image)


def discover_samples(source, stop=None, log=print):
    path = Path(source)
    if path.is_dir():
        candidate = path / "data.yaml"
        if not candidate.is_file():
            raise ValueError("Выберите YOLO/маски YAML или COCO JSON; для папки нужен data.yaml.")
        path = candidate
    kind = "coco" if path.suffix.lower() == ".json" else "yolo"
    selection = metadata(path)
    if kind == "coco":
        spec = defaults([path], kind)
        # Reviewed JSON has absolute, approved image paths; preserve its RLE/polygons directly.
        plan = preview(spec, "json", cancel=stop)
        payload = plan["coco"][0]["payload"]
        by_id = defaultdict(list)
        for ann in payload.get("annotations", []):
            by_id[ann["image_id"]].append(ann)
        return [SamSample(str(Path(im["file_name"]).resolve()), "coco", by_id[im["id"]])
                for im in payload["images"] if im.get("annotation_status") != "unlabelled"]
    spec = defaults([path], kind)
    kind = spec["kind"]
    plan = preview(spec, spec.get("initial_mode", "declared"), cancel=stop)
    samples, missing = [], 0
    for entry in plan["entries"]:
        check_stop(stop)
        image = entry["image"]
        target = selection["annotations"].get(image) if selection else entry["annotation"]
        if not selection and sum(bool(value) for value in entry.get("alternatives", {}).values()) > 1:
            candidates = {p for values in entry["alternatives"].values() for p in values}
            if len(candidates) > 1:
                raise ValueError(f"Неоднозначная разметка {image}. Подтвердите источник в окне импорта.")
        if target:
            samples.append(SamSample(image, kind, target, entry["split"]))
        else:
            missing += 1
            if missing <= 10:
                log(f"[SAM] Без разметки — пропуск: {image}")
    if missing:
        log(f"[SAM] Снимков без разметки исключено: {missing}.")
    return samples


def training_samples(cfg, stop=None, log=print):
    samples = discover_samples(cfg["source"], stop, log)
    train = [s for s in samples if s.split == "train"]
    val = [s for s in samples if s.split in ("val", "valid", "validation")]
    if cfg["val_source"]:
        candidates = discover_samples(cfg["val_source"], stop, log)
        val = [s for s in candidates if s.split in ("val", "valid", "validation")]
        if not val:
            val = [s for s in candidates if s.split == "train"]
        if not val:
            raise ValueError("В отдельном источнике валидации нет размеченных train/val снимков; test не используется.")
    if not train:
        raise ValueError("Нет обучающих снимков (split train).")
    if not val:
        # Split by image content so copies cannot leak across train/validation.
        groups = defaultdict(list)
        for sample in train:
            check_stop(stop)
            groups[digest_file(sample.image)].append(sample)
        keys = sorted(groups)
        np.random.default_rng(cfg["seed"]).shuffle(keys)
        count = max(1, round(len(keys) * cfg["validation_fraction"]))
        if len(keys) < 2:
            raise ValueError("Для отдельной валидации нужны минимум два разных снимка.")
        val = [s for key in keys[:count] for s in groups[key]]
        train = [s for key in keys[count:] for s in groups[key]]
        log(f"[SAM] Нет val: отдельные {len(val)} снимков выделены по seed={cfg['seed']}; test не используется.")
    digests = []
    for name, group in (("train", train), ("val", val)):
        seen, files, objects, empty = set(), [], 0, 0
        for index, sample in enumerate(group):
            check_stop(stop)
            digest = digest_file(sample.image)
            if digest in seen:
                log(f"[SAM] Повтор снимка в {name} — пропуск: {sample.image}")
                continue
            seen.add(digest)
            masks, _ = sample.masks(cfg)
            if not masks:
                empty += 1
                continue  # No positive instance to generate a valid training prompt.
            objects += len(masks)
            files.append(sample)
            if index % 32 == 0 or index + 1 == len(group):
                log(f"[SAM] Проверка {name}: {index + 1}/{len(group)}; объектов: {objects}")
        if not files:
            raise ValueError(f"Нет объектов с масками в {name}.")
        if name == "train":
            train = files
        else:
            val = files
        digests.append(seen)
        log(f"[SAM] {name}: {len(files)} снимков, {objects} объектов; пустых снимков исключено: {empty}.")
    if digests[0] & digests[1]:
        raise ValueError("Утечка данных: одинаковые снимки найдены в train и val (проверены SHA-256).")
    return train, val


def dataset_signature(groups, stop=None):
    """Resume must not silently continue with edited images or edited masks."""
    digest = hashlib.sha256()
    for name, samples in groups.items():
        digest.update(name.encode())
        for sample in samples:
            check_stop(stop)
            digest.update(sample.image.encode())
            digest.update(digest_file(sample.image).encode())
            target = (json.dumps(sample.target, sort_keys=True).encode()
                      if sample.kind == "coco" else Path(sample.target).read_bytes())
            digest.update(hashlib.sha256(target).digest())
    return digest.hexdigest()


def sample_prompt(mask, valid, cfg, rng, *, training):
    """Pixel coordinates; validation always uses an exact box and reproducible clicks."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Нельзя создать подсказку для пустого объекта.")
    height, width = mask.shape
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    if training and cfg["box_jitter"]:
        scale = np.tile(box[2:] - box[:2], 2) * cfg["box_jitter"]
        box += rng.uniform(-1, 1, 4) * scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    mode = cfg["prompt_type"]
    if mode == "mixed":
        mode = rng.choice(["box", "points", "both"]) if training else "both"
    points, labels = [], []
    if mode in ("points", "both"):
        for region, label, count in ((mask, 1, cfg["positive_points"]),
                                     (~mask & valid, 0, cfg["negative_points"])):
            yy, xx = np.nonzero(region)
            if len(xx):
                for i in rng.choice(len(xx), count, replace=len(xx) < count):
                    points.append([xx[i] + 0.5, yy[i] + 0.5])
                    labels.append(label)
    return (box if mode in ("box", "both") else None,
            np.asarray(points, np.float32).reshape(-1, 2), np.asarray(labels, np.int32))
