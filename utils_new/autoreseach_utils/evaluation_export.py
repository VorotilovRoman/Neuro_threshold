"""Readable recommendation and transactional export of all measured results."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from .reports import preset_from_row
from utils_new.annotations.modes import MODE_LABELS


def recommendation_text(result):
    lines = [
        f"Эталон: {result.reference_images} снимков, {result.reference_objects} объектов.",
        f"Сравнение: IoU ≥ {result.threshold:g}; классы "
        + (
            "учитываются."
            if result.class_aware
            else "объединены (поиск геометрии объектов)."
        ),
        "Рейтинг: F1 по суммарным TP/FP/FN всего эталона; только успешное покрытие 100%.",
        "Средний F1 по IoU 0.50–0.95 — дополнительная метрика, не mAP.",
        "",
    ]
    valid = result.rankings[result.rankings.eligible]
    if valid.empty:
        lines.append(
            "Победитель не определён: ни один пресет не проверен на всём эталоне."
        )
    else:
        best = valid.iloc[0]
        lines += [
            f"Лучший проверенный пресет: {best['kind']} / {best['method']}.",
            f"F1={best.micro_f1:.4f}; Precision={best.precision:.4f}; Recall={best.recall:.4f}.",
            f"TP={best.tp}, лишних объектов FP={best.fp}, пропущенных FN={best.fn}.",
            "",
            "Лучшие настройки каждого метода:",
        ]
        if (valid["score"] == best["score"]).sum() > 1:
            lines.insert(
                -2,
                "Есть несколько пресетов с одинаковым F1; порядок уточнён средним F1 по снимкам и IoU, затем качеством локализации.",
            )
        for _, row in result.methods.iterrows():
            state = preset_from_row(row)
            names = {"threshold": "Пороговые", "gradient": "Градиентные"}
            operations = " → ".join(
                f"{op['operation']}: {op['kernel_px']} px"
                for op in state["morphology"]["operations"]
            )
            parameters = (
                "; ".join(
                    f"{key}={value}" for key, value in json.loads(row["params"]).items()
                )
                or "без параметров метода"
            )
            processing = json.loads(row.get("processing_json", "{}"))
            if type(processing.get("resize_enabled")) is bool:
                resize = (
                    f"до {processing['max_side']} px"
                    if processing["resize_enabled"]
                    else "исходное разрешение"
                )
            else:
                resize = "режим уменьшения в отчёте не указан"
            lines += [
                f"• {names[row['kind']]} / {row['method']}: F1={row.micro_f1:.4f}; Precision={row.precision:.4f}; Recall={row.recall:.4f}; FP={row.fp}; FN={row.fn}",
                f"  Средний F1 по IoU 0.50–0.95: {row.mean_f1:.4f}; средний IoU совпавших объектов: {row.avg_iou_tp:.4f}",
                f"  {parameters}; инверсия: {'да' if state['invert'] else 'нет'}; разметка: {MODE_LABELS[state['annotation_mode']]}; оболочка: {'да' if state['use_hull'] else 'нет'}",
                f"  Морфология: {operations}; форма: {state['morphology']['shape']}",
                f"  Обработка: {resize}.",
            ]
    lines += [
        "",
        f"Успешно на всём эталоне: {len(valid)} из {len(result.rankings)} пресетов.",
        "Оценка описывает выбранный эталон. Для проверки переноса на новые снимки нужен отдельный контрольный датасет.",
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
