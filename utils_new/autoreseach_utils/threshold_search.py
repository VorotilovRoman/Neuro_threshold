"""Numerical search, bounded image memory and atomic streamed JSONL reports."""

import json
import os
import tempfile
from pathlib import Path

import cv2

from utils_new.annotations import ImageSize, from_legacy
from utils_new.annotations.geometry import bounds
from utils_new.morphology import apply_morphology
from utils_new.segmentation.image_session import load_one_image_item
from utils_new.segmentation.postprocessing import extract_objects
from utils_new.threshold_processing import apply_threshold_method
from .search_space import combination_count, iter_presets


def evaluate(gray, image, preset):
    mask, _ = apply_threshold_method(gray, preset["method"], preset["params"])
    mask = cv2.bitwise_not(mask)
    morphology = preset["morphology"]
    mask = apply_morphology(mask, morphology["operations"], shape=morphology["shape"])
    if preset["invert"]:
        mask = cv2.bitwise_not(mask)
    objects = extract_objects(
        image, mask, preset["annotation_mode"], preset["use_hull"]
    )[1]
    return mask, objects


def metric_boxes(annotations, size):
    result = []
    for obj in annotations:
        ann = from_legacy(obj, ImageSize(*size))
        x0, y0, x1, y1 = bounds(ann.points_px)
        result.append(
            (
                ann.class_id,
                (x0 + x1) / (2 * size[0]),
                (y0 + y1) / (2 * size[1]),
                (x1 - x0) / size[0],
                (y1 - y0) / size[1],
            )
        )
    return result


def run_search(
    paths,
    settings,
    cancel,
    progress=lambda *args: None,
    *,
    resize_enabled=True,
    max_side=1024,
):
    count = combination_count(settings)
    if not count:
        raise ValueError("Нет включённых методов.")
    targets = [
        str(Path(p).with_name(Path(p).stem + "_full_report.txt").resolve())
        for p in paths
    ]
    if len(set(map(os.path.normcase, targets))) != len(targets):
        raise ValueError(
            "У изображений совпадают пути отчётов. Используйте разные имена снимков."
        )
    completed = 0
    outputs = []
    errors = []
    failed = 0
    for path in paths:
        if cancel.is_set():
            break
        destination = Path(path).with_name(Path(path).stem + "_full_report.txt")
        partial = destination.with_name(destination.stem + ".partial.txt")
        temporary = None
        try:
            record = load_one_image_item(
                path,
                resize_enabled=resize_enabled,
                max_side=max_side,
                load_annotations_flag=False,
            )
            if record is None:
                raise ValueError("Не удалось прочитать изображение.")
            image, gray = record["image"], record["gray"]
            h, w = gray.shape
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                delete=False,
                suffix=".tmp",
            ) as stream:
                temporary = Path(stream.name)

                def write(value):
                    stream.write(
                        json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
                    )

                write(
                    {
                        "record": "header",
                        "schema_version": 1,
                        "kind": "threshold_search",
                        "image_path": str(path),
                        "source_size": record["source_size"],
                        "working_size": [w, h],
                        "planned": count,
                    }
                )
                processed = 0
                for preset in iter_presets(settings):
                    if cancel.is_set():
                        break
                    try:
                        _, annotations = evaluate(gray, image, preset)
                        write(
                            {
                                "record": "result",
                                "preset": preset,
                                "annotations": annotations,
                                "objects": metric_boxes(annotations, (w, h)),
                                "status": "ok",
                            }
                        )
                    except Exception as exc:
                        failed += 1
                        write(
                            {
                                "record": "result",
                                "preset": preset,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                    processed += 1
                    completed += 1
                    progress(completed, count * len(paths), Path(path).name)
                canceled = cancel.is_set()
                write(
                    {
                        "record": "footer",
                        "complete": not canceled,
                        "processed": processed,
                    }
                )
            target = partial if canceled else destination
            os.replace(temporary, target)
            temporary = None
            outputs.append(str(target))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {
        "reports": outputs,
        "errors": errors,
        "canceled": cancel.is_set(),
        "processed": completed,
        "failed": failed,
    }
