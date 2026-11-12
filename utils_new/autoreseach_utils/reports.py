"""Strict report reader and lossless preset reconstruction, independent of Qt."""

import json
from pathlib import Path
import pandas as pd
from .search_space import validate_search_preset, method_params
from .search import metric_boxes


def preset_from_row(row):
    value = row["preset_json"]
    preset = json.loads(value) if isinstance(value, str) else value
    validate_search_preset(preset)
    return preset


def read_report(path, *, allow_partial=False, log=lambda message: None):
    with Path(path).open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        records = [json.loads(line) for line in stream if line.strip()]
    if header.get("schema_version") != 1 or header.get("kind") not in (
        "threshold_search",
        "gradient_search",
    ):
        raise ValueError("Неподдерживаемый формат автоотчёта.")
    footer = records[-1] if records else {}
    if footer.get("record") != "footer":
        raise ValueError("Отчёт не завершён корректно.")
    if not footer["complete"] and not allow_partial:
        raise ValueError("Неполный отчёт: используйте готовый отчёт.")
    results = [r for r in records if r.get("record") == "result"]
    if footer["complete"] and header["planned"] != len(results):
        raise ValueError("Готовый отчёт содержит не все запланированные вычисления.")
    if footer["processed"] != len(results):
        raise ValueError("Неверное число записей отчёта.")
    label = Path(header["image_path"]).with_suffix(".txt")
    size = tuple(header["source_size"])
    truth = None
    if label.exists():
        # Empty labels are valid negatives. A malformed nonempty label must not become one.
        from utils_new.dataset_yolo_labels_new import read_yolo_labels
        from utils_new.annotations import ImageSize, to_legacy

        messages = []
        annotations = read_yolo_labels(label, ImageSize(*size), log=messages.append)
        if messages:
            raise ValueError(f"Некорректная истинная разметка {label}: {messages}")
        truth = metric_boxes(
            [to_legacy(a, ImageSize(*size)) for a in annotations], size
        )
    failed = sum(record["status"] != "ok" for record in results)
    if failed:
        log(f"{path}: пропущено ошибочных вычислений: {failed}")
    rows = []
    for record in results:
        if record["status"] != "ok":
            continue
        preset = record["preset"]
        validate_search_preset(preset)
        if header["kind"] != preset["kind"] + "_search":
            raise ValueError("Тип пресета не совпадает с отчётом.")
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
            {
                "image_path": header["image_path"],
                "report_path": str(path),
                "kind": preset["kind"],
                "method": preset["method"],
                "params": json.dumps(params, sort_keys=True),
                "invert": preset["invert"],
                **{op + "_px": v for op, v in ops.items()},
                "kernel_shape": preset["morphology"]["shape"],
                "operation_order": " → ".join(
                    op["operation"] for op in preset["morphology"]["operations"]
                ),
                "annotation_mode": preset["annotation_mode"],
                "use_hull": preset["use_hull"],
                "morphology_key": json.dumps(preset["morphology"], sort_keys=True),
                "preset_json": json.dumps(preset, sort_keys=True),
                "num_objects": len(record["objects"]),
                "objects": str(record["objects"]),
                "true_objects": str(truth) if truth is not None else None,
                "ground_truth_available": truth is not None,
            }
        )
    return rows


def parse_reports(paths, log=lambda message: None):
    rows = []
    for path in paths:
        try:
            rows.extend(read_report(path, log=log))
        except Exception as exc:
            log(f"{path}: {exc}")
    return pd.DataFrame(rows) if rows else None
