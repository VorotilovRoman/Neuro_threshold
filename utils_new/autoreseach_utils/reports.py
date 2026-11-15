"""Strict streamed report reader and lossless preset/processing reconstruction."""

import json
from pathlib import Path
import pandas as pd
from .search_space import validate_search_preset, method_params
from .search import metric_boxes
from .geometry_metrics import normalize_annotations


def preset_from_row(row):
    value = row["preset_json"]
    preset = json.loads(value) if isinstance(value, str) else value
    validate_search_preset(preset)
    return preset


def _size(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(v) is not int or v < 1 for v in value)
    ):
        raise ValueError("Размеры отчёта должны быть положительными целыми W/H.")
    return tuple(value)


def read_report(
    path,
    *,
    allow_partial=False,
    include_failures=False,
    load_adjacent_truth=True,
    log=lambda message: None,
):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        header = json.loads(next(stream))
        if (
            header.get("record") != "header"
            or header.get("schema_version") != 1
            or header.get("kind") not in ("threshold_search", "gradient_search")
        ):
            raise ValueError("Неподдерживаемый формат автоотчёта.")
        size = _size(header["source_size"])
        working = _size(header["working_size"])
        if type(header.get("planned")) is not int or header["planned"] < 1:
            raise ValueError("Неверное число запланированных вычислений.")
        processing = header.get(
            "processing",
            {"resize_enabled": "unknown", "max_side": None, "pipeline_version": 1},
        )
        if "processing" in header and (
            type(processing.get("resize_enabled")) is not bool
            or type(processing.get("max_side")) is not int
            or processing["max_side"] < 1
            or processing.get("pipeline_version") != 1
        ):
            raise ValueError("Неверные условия обработки в отчёте.")
        truth = None
        truth_annotations = None
        label = Path(header["image_path"]).with_suffix(".txt")
        if load_adjacent_truth and label.is_file():
            from utils_new.dataset_yolo_labels_new import read_yolo_labels
            from utils_new.annotations import ImageSize, to_legacy

            messages = []
            values = read_yolo_labels(label, ImageSize(*size), log=messages.append)
            if any("запись пропущена" in message for message in messages):
                raise ValueError(f"Некорректная истинная разметка {label}: {messages}")
            truth_annotations = [to_legacy(ann, ImageSize(*size)) for ann in values]
            normalize_annotations(truth_annotations, size)
            truth = metric_boxes(truth_annotations, size)
        processed = 0
        failed = 0
        footer = None
        seen = set()
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if footer is not None:
                raise ValueError("Данные после завершающей записи отчёта.")
            if record.get("record") == "footer":
                footer = record
                continue
            if record.get("record") != "result" or record.get("status") not in (
                "ok",
                "error",
            ):
                raise ValueError("Неизвестная запись/статус отчёта.")
            processed += 1
            preset = record["preset"]
            validate_search_preset(preset)
            if header["kind"] != preset["kind"] + "_search":
                raise ValueError("Тип пресета не совпадает с отчётом.")
            if (
                processing.get("resize_enabled") in (True, False)
                and processing.get("max_side") == 1024
            ):
                preset = dict(preset, preprocessing=processing)
            preset_json = json.dumps(preset, sort_keys=True)
            if preset_json in seen:
                raise ValueError("Повторный пресет в одном отчёте.")
            seen.add(preset_json)
            success = record["status"] == "ok"
            if success:
                normalize_annotations(record["annotations"], working)
                if any(
                    ann[0] != preset["annotation_mode"] for ann in record["annotations"]
                ):
                    raise ValueError(
                        "Геометрия объектов не совпадает с режимом пресета."
                    )
                # Do not trust an independently edited bounding-box cache.
                objects = metric_boxes(record["annotations"], working)
            else:
                failed += 1
                objects = None
                if not include_failures:
                    continue
            params = dict(method_params(preset))
            if preset["kind"] == "gradient":
                params.update(
                    threshold=preset["threshold"],
                    fill_contours=preset["fill_contours"],
                    min_area_percent=preset["min_area_percent"],
                )
            ops = {
                op["operation"]: op["kernel_px"]
                for op in preset["morphology"]["operations"]
            }
            rows.append(
                dict(
                    image_path=header["image_path"],
                    report_path=str(Path(path).resolve()),
                    kind=preset["kind"],
                    method=preset["method"],
                    params=json.dumps(params, sort_keys=True),
                    invert=preset["invert"],
                    **{op + "_px": v for op, v in ops.items()},
                    kernel_shape=preset["morphology"]["shape"],
                    operation_order=" → ".join(
                        op["operation"] for op in preset["morphology"]["operations"]
                    ),
                    annotation_mode=preset["annotation_mode"],
                    use_hull=preset["use_hull"],
                    morphology_key=json.dumps(preset["morphology"], sort_keys=True),
                    preset_json=preset_json,
                    num_objects=len(objects) if success else None,
                    objects=json.dumps(objects) if success else None,
                    annotations_json=json.dumps(record["annotations"])
                    if success
                    else None,
                    true_objects=json.dumps(truth) if truth is not None else None,
                    truth_annotations_json=json.dumps(truth_annotations)
                    if truth is not None
                    else None,
                    ground_truth_available=truth is not None,
                    status=record["status"],
                    error=record.get("error", ""),
                    source_size=json.dumps(size),
                    working_size=json.dumps(working),
                    processing_json=json.dumps(processing, sort_keys=True),
                    source_sha256=header.get("source_sha256", ""),
                )
            )
        if footer is None or type(footer.get("complete")) is not bool:
            raise ValueError("Отчёт не завершён корректно.")
        if footer.get("processed") != processed:
            raise ValueError("Неверное число записей отчёта.")
        if footer["complete"] and header["planned"] != processed:
            raise ValueError("Готовый отчёт содержит не все вычисления.")
        if processed > header["planned"]:
            raise ValueError("Лишние вычисления в отчёте.")
        if not footer["complete"] and not allow_partial:
            raise ValueError("Неполный отчёт: используйте готовый отчёт.")
    if failed:
        log(
            f"{path}: ошибочных вычислений: {failed}; они не являются успешными пустыми результатами."
        )
    return rows


def parse_reports(
    paths,
    log=lambda message: None,
    *,
    strict=False,
    include_failures=False,
    load_adjacent_truth=True,
):
    rows = []
    errors = []
    for path in sorted(set(map(str, paths))):
        try:
            rows.extend(
                read_report(
                    path,
                    log=log,
                    include_failures=include_failures,
                    load_adjacent_truth=load_adjacent_truth,
                )
            )
        except Exception as exc:
            message = f"{path}: {exc}"
            errors.append(message)
            log(message)
    if errors and strict:
        raise ValueError("\n".join(errors))
    return pd.DataFrame(rows) if rows else None
