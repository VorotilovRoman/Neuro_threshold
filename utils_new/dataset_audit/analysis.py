"""Numerical dataset statistics, filtering and training-run inventory."""

"""
Быстрая логика анализа датасета (без ML, без длительных операций):
  • фильтрация пар по размеру / яркости / aspect / объектам /
    площади (px² и % от изображения);
  • распределение площадей (общее — в px² и %, и по классам);
  • распределение по классам (таблица + круговая + столбчатая диаграммы);
  • дисбаланс классов (ratio, Gini, Entropy);
  • разрешение изображений (W/H/aspect);
  • яркость (mean, тёмные/пересвеченные);
  • сканирование папки run/ с результатами обучения YOLO;
  • экспорт подмножества с ремапом классов в .txt и data.yaml.

Продвинутые проверки (дубликаты, утечка, тепловая карта,
Loss Inspection, кластеризация эмбеддингов) —
в utils_new/dataset_audit/checks.py и jobs.py.
"""
import os
import cv2
import numpy as np
from PIL import Image

_brightness_cache = {}
from utils_new.dataset_audit.objects import read_objects
from utils_new.import_export.dataset_export import export_subset
from utils_new.import_export.images import read_image_with_fallback
from utils_new.images.processing import ImageStats


def histogram_edges(values, max_bins=50):
    """Finite, distinct edges even for areas differing only by rounding noise."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not data.size:
        return np.array([-0.5, 0.5])
    low, high = float(data.min()), float(data.max())
    if np.isclose(low, high, rtol=1e-12, atol=1e-12):
        radius = max(0.5, abs(low) * 1e-6)
        return np.array([low - radius, high + radius])
    return np.unique(np.linspace(low, high, min(max_bins, data.size) + 1))


def get_image_brightness(path):
    try:
        path = os.path.realpath(path)
        stat = os.stat(path)
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None
    cached = _brightness_cache.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    image = read_image_with_fallback(path)
    if image is None or not np.isfinite(image).all():
        return None
    gray = ImageStats.to_gray(image)
    if gray is None or not np.isfinite(gray).all():
        return None
    mean = float(gray.mean())
    _brightness_cache[path] = (signature, mean)
    return mean


def clear_brightness_cache():
    """Сброс кэша яркости (например, после удаления файлов)."""
    _brightness_cache.clear()


def _safe_progress(progress_cb, i, total, step=25):
    if progress_cb and ((i + 1) % step == 0 or i == total - 1):
        progress_cb(i + 1, total)


def _check_cancel(cancel_check):
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


class FilterParams:
    """Параметры фильтрации.

    Соглашения о «не задано»:
      • min_w=0, min_h=0, min_objs=0, min_area=0,
        min_area_pct=0.0, min_aspect=0.0 — без ограничения;
      • max_w=0, max_h=0, max_objs=0, max_area=0,
        max_area_pct=0.0, max_aspect=0.0 — без ограничения сверху;
      • min_brightness=-1, max_brightness=-1 — без ограничения
        (0 — валидное значение яркости, поэтому sentinel = -1).
    """

    __slots__ = (
        "min_w",
        "max_w",
        "min_h",
        "max_h",
        "min_brightness",
        "max_brightness",
        "min_objs",
        "max_objs",
        "min_area",
        "max_area",
        "min_area_pct",
        "max_area_pct",
        "min_aspect",
        "max_aspect",
    )

    def __init__(
        self,
        min_w=0,
        max_w=0,
        min_h=0,
        max_h=0,
        min_brightness=-1,
        max_brightness=-1,
        min_objs=0,
        max_objs=0,
        min_area=0,
        max_area=0,
        min_area_pct=0.0,
        max_area_pct=0.0,
        min_aspect=0.0,
        max_aspect=0.0,
    ):
        self.min_w = min_w
        self.max_w = max_w
        self.min_h = min_h
        self.max_h = max_h
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_objs = min_objs
        self.max_objs = max_objs
        self.min_area = min_area
        self.max_area = max_area
        self.min_area_pct = min_area_pct
        self.max_area_pct = max_area_pct
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect


class DatasetAnalyzer:
    def __init__(self, dataset_type=0, input_format="auto"):
        self.dataset_type = dataset_type
        self.input_format = input_format

    def set_dataset_type(self, dt):
        self.dataset_type = dt

    def set_input_format(self, fmt):
        self.input_format = fmt

    def read_objects(self, pair):
        return read_objects(pair, self.dataset_type, self.input_format)

    def filter_pairs(
        self,
        pairs,
        included_class_ids,
        total_classes,
        params,
        progress_cb=None,
        include_empty=False,
        cancel_check=None,
    ):
        result = dict(
            filtered_pairs=[],
            class_counts={},
            class_areas={},
            total_objects=0,
            object_areas=[],
            object_areas_pct=[],
            empty_images=0,
            unlabelled_images=0,
            error=None,
            errors=[],
            cancelled=False,
            dominant_classes={},
        )
        if not pairs:
            return result
        if total_classes > 0 and not included_class_ids and self.dataset_type != 2:
            result["error"] = "Не выбрано ни одного класса — фильтрация невозможна."
            return result
        for lower, upper in [
            ("min_w", "max_w"),
            ("min_h", "max_h"),
            ("min_objs", "max_objs"),
            ("min_area", "max_area"),
            ("min_area_pct", "max_area_pct"),
            ("min_aspect", "max_aspect"),
        ]:
            if getattr(params, upper) > 0 and getattr(params, lower) > getattr(
                params, upper
            ):
                result["error"] = f"Нижняя граница {lower} выше {upper}."
                return result
        if params.max_brightness >= 0 and params.min_brightness > params.max_brightness:
            result["error"] = "Нижняя граница яркости выше верхней."
            return result

        def within(value, lo, hi):
            return value >= lo and (hi <= 0 or value <= hi)

        for i, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                result["cancelled"] = True
                break
            try:
                (w, h), objects = self.read_objects(pair)
                from collections import Counter
                counts = Counter(obj.class_id for obj in objects)
                result["dominant_classes"][tuple(pair)] = counts.most_common(1)[0][0] if counts else None
                if (
                    not within(w, params.min_w, params.max_w)
                    or not within(h, params.min_h, params.max_h)
                    or not within(w / h, params.min_aspect, params.max_aspect)
                ):
                    continue
                if params.min_brightness >= 0 or params.max_brightness >= 0:
                    brightness = get_image_brightness(pair[0])
                    if brightness is None:
                        raise ValueError("Не удалось прочитать яркость.")
                    if (
                        params.min_brightness >= 0
                        and brightness < params.min_brightness
                    ) or (
                        params.max_brightness >= 0
                        and brightness > params.max_brightness
                    ):
                        continue
                if self.dataset_type != 2 and (len(pair) < 2 or pair[1] is None):
                    if params.min_objs == 0:
                        result['filtered_pairs'].append(pair)
                        result['unlabelled_images'] += 1
                    continue
                selected = [
                    obj
                    for obj in objects
                    if not total_classes or obj.class_id in included_class_ids
                ]
                if not within(len(selected), params.min_objs, params.max_objs):
                    continue
                valid = [
                    obj
                    for obj in selected
                    if within(obj.area_px, params.min_area, params.max_area)
                    and within(
                        obj.area_px / (w * h) * 100,
                        params.min_area_pct,
                        params.max_area_pct,
                    )
                ]
                # A class filter cannot turn a nonempty annotation into a negative example.
                empty = not objects
                if (
                    not valid
                    and not (empty and include_empty and params.min_objs == 0)
                    and self.dataset_type != 2
                ):
                    continue
                result["filtered_pairs"].append(pair)
                if empty and self.dataset_type != 2:
                    result["empty_images"] += 1
                for obj in valid:
                    cls = obj.class_id
                    result["class_counts"][cls] = result["class_counts"].get(cls, 0) + 1
                    result["class_areas"].setdefault(cls, []).append(obj.area_px)
                    result["total_objects"] += 1
                    result["object_areas"].append(obj.area_px)
                    result["object_areas_pct"].append(obj.area_px / (w * h) * 100)
            except (OSError, ValueError, TypeError, cv2.error) as exc:
                result["errors"].append(f"{pair[0]}: {exc}")
            finally:
                if progress_cb:
                    progress_cb(i + 1, len(pairs))
        return result

    export_subset = staticmethod(export_subset)


class ClassImbalanceAnalyzer:
    @staticmethod
    def analyze(class_counts):
        if not class_counts or len(class_counts) < 2:
            return {
                "imbalance_ratio": 1.0,
                "gini": 0.0,
                "entropy": 0.0,
                "majority_class": None,
                "minority_class": None,
                "warning": "Недостаточно классов для оценки дисбаланса.",
            }
        counts = np.array([c for c in class_counts.values() if c > 0], dtype=np.float64)
        if len(counts) < 2:
            return {
                "imbalance_ratio": 1.0,
                "gini": 0.0,
                "entropy": 0.0,
                "majority_class": None,
                "minority_class": None,
                "warning": "Недостаточно классов для оценки дисбаланса.",
            }
        ratio = float(counts.max() / counts.min())
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        sorted_c = np.sort(counts)
        n = len(sorted_c)
        gini = float(
            2 * np.sum(np.arange(1, n + 1) * sorted_c) / (n * sorted_c.sum())
            - (n + 1) / n
        )
        ids = [key for key, value in class_counts.items() if value > 0]
        majority = ids[int(np.argmax(counts))]
        minority = ids[int(np.argmin(counts))]
        if ratio >= 100:
            warn = (
                "🚨 КРИТИЧЕСКИЙ дисбаланс (≥100:1). Редкие классы могут игнорироваться."
            )
        elif ratio >= 10:
            warn = "⚠️ Сильный дисбаланс (≥10:1). Рекомендуется oversampling / weighted loss."
        elif ratio >= 3:
            warn = "ℹ️ Умеренный дисбаланс (≥3:1). Стоит проверить распределение."
        else:
            warn = "✅ Дисбаланс в пределах нормы."
        return {
            "imbalance_ratio": ratio,
            "gini": gini,
            "entropy": entropy,
            "majority_class": majority,
            "minority_class": minority,
            "warning": warn,
        }


class ResolutionAnalyzer:
    @staticmethod
    def analyze(pairs, progress_cb=None, cancel_check=None):
        widths, heights, aspects = ([], [], [])
        total = len(pairs)
        cancelled = False
        for i, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            try:
                with Image.open(pair[0]) as img:
                    w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h if h else 0)
            except Exception:
                pass
            _safe_progress(progress_cb, i, total)
        if not widths:
            return None
        return {
            "widths": widths,
            "heights": heights,
            "aspects": aspects,
            "median_w": float(np.median(widths)),
            "median_h": float(np.median(heights)),
            "recommended_imgsz": int(max(np.median(widths), np.median(heights))),
            "cancelled": cancelled,
        }


class BrightnessAnalyzer:
    DARK_BOUND = 40.0
    BRIGHT_BOUND = 215.0

    @staticmethod
    def analyze(pairs, progress_cb=None, cancel_check=None):
        means = []
        total = len(pairs)
        cancelled = False
        for i, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            b = get_image_brightness(pair[0])
            if b is not None:
                means.append(b)
            _safe_progress(progress_cb, i, total)
        if not means:
            return None
        arr = np.array(means)
        return {
            "means": means,
            "dark_count": int((arr < BrightnessAnalyzer.DARK_BOUND).sum()),
            "bright_count": int((arr > BrightnessAnalyzer.BRIGHT_BOUND).sum()),
            "mean_brightness": float(arr.mean()),
            "cancelled": cancelled,
        }


class TrainingResultsLoader:
    """Сканирует папку run/ с результатами обучения Ultralytics YOLO.

    Ожидаемые файлы (номера батчей могут быть любыми):
        results.png, labels.jpg, confusion_matrix_normalized.png,
        confusion_matrix.png, Box*/Mask*_curve.png,
        results.csv, args.yaml,
        train_batch*.jpg, val_batch*_labels.jpg, val_batch*_pred.jpg,
        weights/best.pt, weights/best_meta.json, weights/best.onnx,
        weights/last.pt, weights/last_meta.json, weights/best_onnx_meta.json.

    Категоризация:
        main     — 3 ключевых файла для предпросмотра в UI;
        curves   — остальные *_curve.png и confusion_matrix.png;
        batches  — train_batch* и val_batch*;
        metrics  — results.csv и args.yaml;
        weights  — содержимое подпапки weights/;
        best_model, best_model_meta — пути к weights/best.pt и
        weights/best_meta.json (если есть).
    """

    MAIN_FILES = ("results.png", "labels.jpg", "confusion_matrix_normalized.png")

    @staticmethod
    def scan(folder):
        result = {
            "folder": folder,
            "main": {},
            "curves": {},
            "batches": [],
            "metrics": {},
            "weights": {},
            "best_model": None,
            "best_model_meta": None,
            "error": None,
        }
        if not folder or not os.path.isdir(folder):
            result["error"] = f"Папка не найдена: {folder}"
            return result
        try:
            entries = sorted(os.listdir(folder))
        except Exception as e:
            result["error"] = f"Не удалось прочитать папку: {e}"
            return result
        main_set = set(TrainingResultsLoader.MAIN_FILES)
        for name in entries:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if name in main_set:
                result["main"][name] = path
                continue
            lower = name.lower()
            if name == "confusion_matrix.png" or lower.endswith("_curve.png"):
                result["curves"][name] = path
            elif name in ("results.csv", "args.yaml"):
                result["metrics"][name] = path
            elif name.startswith("train_batch") and lower.endswith(".jpg"):
                result["batches"].append((name, path))
            elif name.startswith("val_batch") and lower.endswith(".jpg"):
                result["batches"].append((name, path))
        wdir = os.path.join(folder, "weights")
        if os.path.isdir(wdir):
            try:
                for name in sorted(os.listdir(wdir)):
                    path = os.path.join(wdir, name)
                    if os.path.isfile(path):
                        result["weights"][name] = path
            except Exception:
                pass
        result["best_model"] = result["weights"].get("best.pt")
        result["best_model_meta"] = result["weights"].get("best_meta.json")
        return result

    @staticmethod
    def summary_text(data):
        """Текстовое описание содержимого."""
        if not data:
            return "Нет данных."
        if data.get("error"):
            return f"❌ {data['error']}"
        lines = [f"Папка: {data['folder']}"]
        if data["main"]:
            lines.append(f"Основные: {sorted(data['main'].keys())}")
        else:
            lines.append(
                "Основные: не найдено (results.png / labels.jpg / confusion_matrix_normalized.png)"
            )
        if data["curves"]:
            lines.append(f"Кривые: {len(data['curves'])} файлов")
        if data["batches"]:
            lines.append(f"Батчи: {len(data['batches'])} файлов")
        if data["metrics"]:
            lines.append(f"Метрики/args: {sorted(data['metrics'].keys())}")
        if data["weights"]:
            lines.append(f"Weights: {sorted(data['weights'].keys())}")
        if data["best_model"]:
            lines.append(f"Best model: {os.path.basename(data['best_model'])}")
        else:
            lines.append("Best model: не найден (weights/best.pt)")
        if data["best_model_meta"]:
            lines.append("Best meta: best_meta.json")
        return "\n".join(lines)


class DatasetFilterJob:
    """Filter/statistics snapshot independent of widgets and mutable UI analyzers."""
    def __init__(self, pairs, dataset_type, input_format, included_ids, total_classes, params, include_empty):
        self.pairs, self.dataset_type, self.input_format = list(pairs), dataset_type, input_format
        self.included_ids, self.total_classes, self.params = set(included_ids), total_classes, params
        self.include_empty = include_empty
        self.cancelled = False
        self.progress = lambda *_: None
        self.log = lambda *_: None

    def cancel(self):
        self.cancelled = True

    def run(self):
        analyzer = DatasetAnalyzer(self.dataset_type, self.input_format)
        return analyzer.filter_pairs(self.pairs, self.included_ids, self.total_classes, self.params,
            progress_cb=self.progress, include_empty=self.include_empty,
            cancel_check=lambda: self.cancelled)
