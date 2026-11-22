"""Readable recommendation and transactional export of all measured results."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from utils_new.import_export.research_reports import preset_from_row
from utils_new.annotations.modes import MODE_LABELS


PARAMETER_LABELS = {
    "threshold": "Порог",
    "window": "Размер окна",
    "c": "Смещение C",
    "k": "Коэффициент k",
    "r": "Диапазон R",
    "init": "Начальный порог",
    "excess": "Избыток фона",
    "ksize": "Размер ядра",
    "scale": "Масштаб",
    "threshold1": "Нижний порог Canny",
    "threshold2": "Верхний порог Canny",
    "aperture": "Апертурa Canny",
    "fill_contours": "Заливка контуров",
    "min_area_percent": "Минимальная площадь, %",
}


def preset_settings_text(row):
    """Complete reproducible settings, for both UI selection and saved summary."""
    state = preset_from_row(row)
    from utils_new.import_export.research_reports import describe_preset
    import math

    description = describe_preset(state)
    lines = [f"Метод: {state['method']}", "Параметры:"]
    values = json.loads(description["params"])
    for key, value in values.items():
        if type(value) is bool:
            value = "да" if value else "нет"
        lines.append(f"  {PARAMETER_LABELS.get(key, key)}: {value}")
    if not values:
        lines.append("  У метода нет настраиваемых параметров.")
    lines += [
        f"Инверсия: {'да' if state['invert'] else 'нет'}",
        f"Разметка: {MODE_LABELS[state['annotation_mode']]}",
        f"Выпуклая оболочка: {'да' if state['use_hull'] else 'нет'}",
        "",
        "Морфология, в порядке применения:",
    ]
    names = {
        "close": "Закрытие",
        "open": "Открытие",
        "erode": "Эрозия",
        "dilate": "Дилатация",
    }
    for operation in state["morphology"]["operations"]:
        value = operation["kernel_px"]
        lines.append(
            f"  {names[operation['operation']]}: {value} px"
            + (" (не применяется)" if value <= 1 else "")
        )
    shape = {"Rectangle": "Прямоугольник", "Ellipse": "Эллипс", "Cross": "Крест"}.get(
        state["morphology"]["shape"], state["morphology"]["shape"]
    )
    lines.append("Форма ядра: " + shape)
    processing = json.loads(row.get("processing_json", "{}"))
    if type(processing.get("resize_enabled")) is bool:
        lines.append(
            "Размер обработки: "
            + (
                f"уменьшение до {processing['max_side']} px"
                if processing["resize_enabled"]
                else "исходное разрешение"
            )
        )
    else:
        lines.append(
            "Размер обработки: в старом отчёте не указан; перед применением уточните режим уменьшения."
        )
    if "expected_images" in row:
        lines += [
            "",
            f"Проверено снимков: {int(row['evaluated_images'])} / {int(row['expected_images'])}",
        ]
        if row.get("exclusion_reason"):
            lines.append(str(row["exclusion_reason"]))
        for key, label in [
            ("score", "F1 всего датасета"),
            ("precision", "Точность"),
            ("recall", "Полнота"),
            ("avg_iou_tp", "IoU совпавших объектов"),
        ]:
            value = row.get(key)
            lines.append(
                f"{label}: {float(value):.1%}"
                if value is not None and math.isfinite(value)
                else f"{label}: —"
            )
    return "\n".join(lines)


def recommendation_text(result):
    from utils_new.autoreseach_utils.workflow import result_guidance

    title, body, next_action = result_guidance(result)
    lines = [
        title,
        body,
        next_action,
        "",
        f"Эталон: {result.reference_images} снимков, {result.reference_objects} объектов.",
        f"Порог совпадения IoU: {result.threshold:g}.",
        "Учёт классов: "
        + (
            "включён"
            if result.class_aware
            else "выключен — сравнивается геометрия объектов"
        ),
        "Рейтинг — измеренный F1 по суммарным TP/FP/FN всего эталона.",
        "Средний F1 по IoU 0.50–0.95 не является mAP.",
        "",
        "Лучшие настройки каждого метода:",
    ]
    for _, row in result.methods.iterrows():
        lines += ["", preset_settings_text(row)]
    lines += [
        "",
        "Оценка описывает выбранный эталон. Проверьте выбранные настройки на новых снимках, не участвовавших в подборе.",
    ]
    lines.extend(result.diagnostics)
    return "\n".join(lines)


def export_evaluation(result, destination):
    target = Path(destination).resolve()
    existed = target.exists()
    if existed and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Выберите отдельную пустую папку для оценки.")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".report-evaluation-", dir=target.parent))
    try:
        for filename, frame in [
            ("rankings.csv", result.rankings),
            ("image_details.csv", result.details),
            ("best_methods.csv", result.methods),
        ]:
            frame.to_csv(stage / filename, index=False, encoding="utf-8-sig")
        metadata = dict(
            schema_version=1,
            reference_source=result.reference_source,
            images=result.reference_images,
            objects=result.reference_objects,
            iou_threshold=result.threshold,
            class_aware=result.class_aware,
            ranking="micro F1; successful coverage 100%",
            diagnostics=list(result.diagnostics),
            presets=json.loads(result.rankings.to_json(orient="records")),
        )
        (stage / "evaluation.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (stage / "summary.md").write_text(
            "# Выбор методов по ручной разметке\n\n" + recommendation_text(result),
            encoding="utf-8",
        )
        if target.exists():
            target.rmdir()
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        if existed and not target.exists():
            target.mkdir()
        raise
    return str(target)
