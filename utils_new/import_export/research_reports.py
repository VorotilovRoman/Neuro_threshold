"""Strict streamed report reader and lossless preset/processing reconstruction."""

import json
from pathlib import Path
import pandas as pd
from utils_new.autoreseach_utils.search_space import validate_search_preset, method_params
from utils_new.annotations.metrics import normalize_annotations


def preset_from_row(row):
    value = row["preset_json"]
    preset = json.loads(value) if isinstance(value, str) else value
    validate_search_preset(preset)
    return preset


def describe_preset(preset):
    """Derive display/export parameters from the actual recipe, never a sample row."""
    validate_search_preset(preset)
    params = dict(method_params(preset))
    if preset["kind"] == "gradient":
        params.update(
            threshold=preset["threshold"],
            fill_contours=preset["fill_contours"],
            min_area_percent=preset["min_area_percent"],
        )
    operations = preset["morphology"]["operations"]
    return dict(
        kind=preset["kind"],
        method=preset["method"],
        params=json.dumps(params, sort_keys=True),
        invert=preset["invert"],
        **{op["operation"] + "_px": op["kernel_px"] for op in operations},
        kernel_shape=preset["morphology"]["shape"],
        operation_order=" → ".join(op["operation"] for op in operations),
        annotation_mode=preset["annotation_mode"],
        use_hull=preset["use_hull"],
        morphology_key=json.dumps(preset["morphology"], sort_keys=True),
        preset_json=json.dumps(preset, sort_keys=True),
    )


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
    cancel_check=lambda: False,
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
        processed = 0
        failed = 0
        footer = None
        seen = set()
        for line in stream:
            if cancel_check():
                raise InterruptedError("Загрузка отчётов отменена.")
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
            else:
                failed += 1
            rows.append(
                dict(
                    image_path=header["image_path"],
                    report_path=str(Path(path).resolve()),
                    **describe_preset(preset),
                    num_objects=len(record["annotations"]) if success else None,
                    annotations_json=json.dumps(record["annotations"])
                    if success
                    else None,
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
    cancel_check=lambda: False,
    progress=lambda *args: None,
):
    rows = []
    errors = []
    selected = sorted(set(map(str, paths)))
    for index, path in enumerate(selected):
        if cancel_check():
            raise InterruptedError("Загрузка отчётов отменена.")
        try:
            rows.extend(
                read_report(
                    path,
                    log=log,
                    cancel_check=cancel_check,
                )
            )
        except InterruptedError:
            raise
        except Exception as exc:
            if isinstance(
                exc, (json.JSONDecodeError, StopIteration, UnicodeDecodeError)
            ):
                detail = "Файл не является готовым отчётом автоподбора UTF-8/JSONL. Выполните автоподбор заново и выберите созданный отчёт."
            elif isinstance(exc, FileNotFoundError):
                detail = "Файл не найден. Проверьте путь или выберите отчёт заново."
            elif isinstance(exc, (KeyError, AttributeError, TypeError)):
                detail = f"В отчёте отсутствуют обязательные данные или повреждена структура ({exc}). Создайте отчёт заново."
            else:
                detail = str(exc)
            message = f"{path}: {detail}"
            errors.append(message)
            log(message)
        progress(index + 1, len(selected), Path(path).name)
    if errors:
        raise ValueError("\n".join(errors))
    return pd.DataFrame(rows) if rows else None
