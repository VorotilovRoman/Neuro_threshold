"""Input collection and user-facing readiness for reference-based comparison."""

from dataclasses import dataclass
from pathlib import Path
from utils_new.import_export.research_reports import parse_reports


@dataclass(frozen=True)
class ReportCollection:
    paths: tuple
    frame: object
    image_count: int
    threshold_files: int
    gradient_files: int
    failed_results: int


def discover_reports(folder):
    root = Path(folder)
    if not root.is_dir():
        raise ValueError("Выберите папку с готовыми отчётами автоподбора.")
    paths = sorted(
        str(path.resolve())
        for path in root.rglob("*.txt")
        if path.name.endswith(("_full_report.txt", "_gradient_report.txt"))
    )
    if not paths:
        raise ValueError(
            "Готовые отчёты не найдены. Запустите автоподбор на вкладке методов или выберите другую папку. Отчёты заканчиваются на _full_report.txt или _gradient_report.txt."
        )
    return paths


def load_collection(paths, *, cancel_check=lambda: False, progress=lambda *args: None):
    selected = tuple(sorted({str(Path(path).resolve()) for path in paths}))
    if not selected:
        raise ValueError("Добавьте хотя бы один готовый отчёт автоподбора.")
    frame = parse_reports(selected, cancel_check=cancel_check, progress=progress)
    if frame is None or frame.empty:
        raise ValueError(
            "В отчётах нет вычислений. Проверьте диапазоны автоподбора и создайте отчёты заново."
        )
    counts = frame.groupby("kind")["report_path"].nunique()
    return ReportCollection(
        selected,
        frame,
        int(frame["image_path"].nunique()),
        int(counts.get("threshold", 0)),
        int(counts.get("gradient", 0)),
        int((frame["status"] != "ok").sum()),
    )


def input_guidance(reference, collection):
    """Basic readiness; geometry and correspondence are checked by comparison."""
    if not reference:
        return (
            False,
            "Шаг 1: выберите датасет с правильной ручной разметкой — YOLO YAML или папку с изображениями и TXT.",
        )
    if not Path(reference).exists():
        return (
            False,
            "Эталон не найден по выбранному пути. Выберите существующий YAML или папку.",
        )
    if collection is None:
        return (
            False,
            "Шаг 2: добавьте результаты автоподбора для этих изображений. Можно загрузить пороговые, градиентные отчёты или оба вида вместе.",
        )
    return (
        True,
        "Шаг 3: нажмите «Сравнить методы». Будут проверены эталон, соответствие снимков и качество каждого пресета на всём датасете.",
    )


def result_guidance(result):
    """Explain the conclusion and the next useful action without hiding failures."""
    if not result.reference_objects:
        return (
            "Для выбора метода нужны снимки с объектами",
            f"В эталоне {result.reference_images} снимков, но ни одного размеченного объекта. Можно проверить ложные срабатывания, но нельзя определить, какой метод находит объекты.",
            "Добавьте правильно размеченные положительные примеры в эталон и запустите сравнение заново.",
        )
    valid = result.rankings[result.rankings["eligible"]]
    if valid.empty:
        return (
            "Победитель пока не определён",
            f"Ни один из {len(result.rankings)} пресетов не проверен успешно на всех {result.reference_images} снимках. Неполные результаты не участвуют в выборе.",
            "Выберите настройку и откройте «Снимки выбранной настройки»: там указаны отсутствующие вычисления и ошибки. Повторите автоподбор для нужных снимков с одинаковыми диапазонами.",
        )
    best = valid.iloc[0]
    if best["score"] == 0:
        return (
            "Подходящие настройки не найдены",
            "Полные результаты есть, но ни один пресет не дал совпавших объектов при выбранных условиях.",
            "Проверьте формат ручной разметки, инверсию и режим объектов; расширьте диапазоны автоподбора. При учёте классов проверьте соответствие class ID.",
        )
    family = {"threshold": "Пороговые", "gradient": "Градиентные"}[best["kind"]]
    return (
        f"Лучший из проверенных: {family} / {best['method']}",
        f"F1 {best['score']:.1%} · точность {best['precision']:.1%} · полнота {best['recall']:.1%}. Проверены все {result.reference_images} снимков: лишних объектов {int(best['fp'])}, пропущенных {int(best['fn'])}.",
        "Выберите метод в таблице: справа будут его полные настройки, а во вкладке снимков — ошибки по изображениям. Добавьте нужные настройки в пресеты соответствующей вкладки и проверьте их на новых снимках.",
    )


def load_folder_collection(
    folder, *, existing=(), cancel_check=lambda: False, progress=lambda *args: None
):
    return load_collection(
        [*existing, *discover_reports(folder)],
        cancel_check=cancel_check,
        progress=progress,
    )


def selected_presets(frame):
    from utils_new.import_export.research_reports import preset_from_row

    if frame is None or frame.empty:
        raise ValueError("Выберите хотя бы одну настройку в таблице результата.")
    column = "recommendable" if "recommendable" in frame else "eligible"
    if column not in frame or not frame[column].all():
        raise ValueError(
            "Для сохранения выберите настройки, проверенные на всём эталоне с положительными примерами."
        )
    return [preset_from_row(row) for _, row in frame.iterrows()]
