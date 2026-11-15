"""Evaluate every exact recipe on one fixed reference set, without learned scores."""

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import math
import pandas as pd
from .geometry_metrics import normalize_annotations, evaluate_image, count_metrics
from .ground_truth import load_ground_truth, match_sample, digest_file
from .reports import preset_from_row, describe_preset


@dataclass
class EvaluationResult:
    rankings: pd.DataFrame
    details: pd.DataFrame
    methods: pd.DataFrame
    diagnostics: tuple
    reference_images: int
    reference_objects: int
    reference_source: str
    threshold: float
    class_aware: bool


def evaluate_dataset(
    reports,
    reference,
    *,
    format_hint="auto",
    threshold=0.5,
    class_aware=False,
    cancel_check=lambda: False,
    progress=lambda *args: None,
):
    if reports is None or reports.empty:
        raise ValueError("Загрузите результаты автоподбора.")
    if not 0 < threshold <= 1:
        raise ValueError("IoU должен быть больше 0 и не больше 1.")
    truth = (
        load_ground_truth(
            reference,
            reports=reports,
            format_hint=format_hint,
            cancel_check=cancel_check,
        )
        if isinstance(reference, (str, bytes, Path))
        else reference
    )
    ground = {
        sample.key: normalize_annotations(sample.annotations, sample.size)
        for sample in truth.samples
    }
    sample_lookup = {sample.key: sample for sample in truth.samples}
    mapped = {}
    digests = {}
    diagnostics = []
    groups = {}
    extra_images = set()
    duplicates = 0
    for _, row in reports.iterrows():
        if cancel_check():
            raise InterruptedError("Оценка отменена.")
        preset = preset_from_row(row)
        preset_json = json.dumps(preset, sort_keys=True)
        processing = json.loads(row.get("processing_json", "{}"))
        if (
            preset.get("preprocessing") is not None
            and preset["preprocessing"] != processing
        ):
            raise ValueError(
                "Пресет и условия обработки расходятся. Загрузите исходный отчёт."
            )
        recipe = hashlib.sha256(
            json.dumps([preset, processing], sort_keys=True).encode()
        ).hexdigest()
        if recipe not in groups:
            groups[recipe] = {"row": row.to_dict(), "records": {}}
        image_path = str(row["image_path"])
        if image_path not in mapped:
            mapped[image_path] = match_sample(image_path, truth)
        sample = mapped[image_path]
        if sample is None:
            extra_images.add(image_path)
            continue
        status = row.get("status", "ok")
        error = str(row.get("error", ""))
        size = tuple(json.loads(row["source_size"]))
        if size != sample.size:
            status = "error"
            error = f"Размер источника {size} не совпадает с эталоном {sample.size}."
        fingerprint = row.get("source_sha256", "")
        if fingerprint and sample.image_path:
            if sample.key not in digests:
                digests[sample.key] = digest_file(sample.image_path)
            if fingerprint != digests[sample.key]:
                status = "error"
                error = "Содержимое изображения отличается от источника отчёта."
        annotations = row.get("annotations_json")
        if status == "ok" and (not isinstance(annotations, str) or not annotations):
            status = "error"
            error = "В отчёте нет исходной геометрии объектов. Повторите автоподбор."
        record = dict(
            status=status,
            error=error,
            annotations=annotations,
            working_size=row["working_size"],
            report_path=row.get("report_path", ""),
            preset_json=preset_json,
        )
        existing = groups[recipe]["records"].get(sample.key)
        if existing:
            duplicates += 1
            keys = ("status", "error", "annotations", "working_size")
            if any(existing[key] != record[key] for key in keys):
                raise ValueError(
                    f"Конфликтующие результаты одного пресета для {sample.key}. Уберите устаревшие отчёты."
                )
        else:
            groups[recipe]["records"][sample.key] = record
    if duplicates:
        diagnostics.append(
            f"Одинаковых повторных вычислений: {duplicates}; повторно не учитываются."
        )
    if extra_images:
        diagnostics.append(
            f"Вне эталона: {len(extra_images)} изображений отчётов; в оценку не входят."
        )
    if not any(group["records"] for group in groups.values()):
        raise ValueError("Ни один снимок отчётов не сопоставлен с эталоном.")
    details = []
    rankings = []
    total = len(groups) * len(truth.samples)
    done = 0
    for recipe, group in groups.items():
        items = []
        description = describe_preset(preset_from_row(group["row"]))
        for key, sample in sample_lookup.items():
            if cancel_check():
                raise InterruptedError("Оценка отменена.")
            record = group["records"].get(key)
            item = dict(
                kind=description["kind"],
                method=description["method"],
                recipe_key=recipe,
                image_id=key,
                image_path=sample.image_path,
                label_path=sample.label_path,
                report_path=record["report_path"] if record else "",
                status=record["status"] if record else "missing",
                error=record["error"]
                if record
                else "Нет вычисления этого пресета для снимка.",
                true_objects=len(ground[key]),
            )
            if record and record["status"] == "ok":
                try:
                    predictions = normalize_annotations(
                        record["annotations"], json.loads(record["working_size"])
                    )
                    item.update(
                        evaluate_image(
                            predictions,
                            ground[key],
                            threshold=threshold,
                            class_aware=class_aware,
                        )
                    )
                except (ValueError, TypeError, KeyError) as exc:
                    item.update(status="error", error=str(exc))
            items.append(item)
            details.append(item)
            done += 1
            progress(done, total, key)
        success = [item for item in items if item["status"] == "ok"]
        tp = sum(item["tp"] for item in success)
        fp = sum(item["fp"] for item in success)
        fn = sum(item["fn"] for item in success)
        measured = (
            count_metrics(tp, fp, fn)
            if success
            else dict(precision=math.nan, recall=math.nan, f1=math.nan)
        )
        eligible = len(success) == len(items)
        row = describe_preset(preset_from_row(group["row"]))
        row["processing_json"] = group["row"].get("processing_json", "{}")
        negative = [item for item in success if item["true_objects"] == 0]
        all_negative = sum(not ground[sample.key] for sample in truth.samples)
        clean = sum(item["predicted_objects"] == 0 for item in negative)
        row.update(
            recipe_key=recipe,
            preset_json=json.dumps(preset_from_row(group["row"]), sort_keys=True),
            eligible=eligible,
            coverage=len(success) / len(items),
            evaluated_images=len(success),
            expected_images=len(items),
            missing_images=sum(item["status"] == "missing" for item in items),
            failed_images=sum(item["status"] == "error" for item in items),
            tp=tp,
            fp=fp,
            fn=fn,
            precision=measured["precision"],
            recall=measured["recall"],
            micro_f1=measured["f1"],
            actual_f1=measured["f1"] if eligible else math.nan,
            score=measured["f1"] if eligible else math.nan,
            macro_f1=sum(item["f1"] for item in success) / len(success)
            if success
            else math.nan,
            mean_f1=sum(item["mean_f1"] for item in success) / len(success)
            if success
            else math.nan,
            avg_iou_tp=sum(item["matched_iou_sum"] for item in success) / tp
            if tp
            else math.nan,
            negative_images=all_negative,
            clean_negative_images=clean,
            negative_accuracy=clean / all_negative if all_negative else math.nan,
            iou_threshold=threshold,
            class_aware=class_aware,
            exclusion_reason=""
            if eligible
            else "Неполное покрытие эталона: не участвует в выборе победителя.",
        )
        rankings.append(row)
    ranked = (
        pd.DataFrame(rankings)
        .sort_values(
            ["eligible", "score", "macro_f1", "mean_f1", "avg_iou_tp", "recipe_key"],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )
    objects = sum(len(values) for values in ground.values())
    ranked["recommendable"] = ranked["eligible"] & (objects > 0)
    if not objects:
        ranked["score"] = math.nan
        ranked["actual_f1"] = math.nan
        ranked["exclusion_reason"] = (
            "В эталоне нет положительных примеров: выбрать метод обнаружения объектов нельзя."
        )
    ranked["rank"] = pd.array(
        [i + 1 if row.recommendable else None for i, row in ranked.iterrows()],
        dtype="Int64",
    )
    ranked["method_rank"] = (
        ranked.groupby(["kind", "method"])["recommendable"]
        .cumsum()
        .where(ranked["recommendable"])
        .astype("Int64")
    )
    methods = ranked[ranked.recommendable].drop_duplicates(["kind", "method"]).copy()
    if not objects:
        diagnostics.append(
            "В эталоне нет объектов: оцениваются только ложные срабатывания, полнота и локализация объектов не проверяются."
        )
    if not ranked.eligible.any():
        diagnostics.append(
            "Нет ни одного пресета, успешно проверенного на всём эталоне. Победитель не определён; дополните автоподбор."
        )
    if any("unknown" in row.get("processing_json", "") for row in rankings):
        diagnostics.append(
            "В старых отчётах не указан режим resize. Сравниваются сохранённые результаты; для воспроизведения уточните режим или повторите автоподбор."
        )
    if any(not sample.image_path for sample in truth.samples):
        diagnostics.append(
            "Эталон содержит только TXT: размеры взяты из отчётов; идентичность пикселей снимков проверить невозможно."
        )
    return EvaluationResult(
        ranked,
        pd.DataFrame(details),
        methods,
        tuple(diagnostics),
        len(truth.samples),
        objects,
        truth.source,
        threshold,
        class_aware,
    )
