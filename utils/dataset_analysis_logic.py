# utils/dataset_analysis_logic.py
"""
Быстрая логика анализа датасета (без ML, без длительных операций):
  • фильтрация пар по площади / объектам / размеру (с учётом пустых аннотаций);
  • распределение площадей (общее и по классам);
  • распределение по классам (таблица + круговая + столбчатая диаграммы);
  • дисбаланс классов (ratio, Gini, Entropy);
  • разрешение изображений (W/H/aspect);
  • яркость (mean, тёмные/пересвеченные);
  • экспорт подмножества с ремапом классов в .txt и data.yaml.

Продвинутые проверки (дубликаты, утечка, тепловая карта,
Loss Inspection, кластеризация эмбеддингов) —
в utils/dataset_audit_logic.py.
"""
from import_libs_internal import *
from PIL import Image
import cv2
import numpy as np


# ============================================================
# Утилиты (локальные — не тянем зависимость от audit_logic)
# ============================================================
def _safe_progress(progress_cb, i, total, step=25):
    if progress_cb and ((i + 1) % step == 0 or i == total - 1):
        progress_cb(i + 1, total)


def _check_cancel(cancel_check):
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


# ============================================================
# DTO фильтрации
# ============================================================
class FilterParams:
    __slots__ = ('min_w', 'min_h', 'min_objs', 'max_objs',
                 'min_area', 'max_area')

    def __init__(self, min_w=0, min_h=0, min_objs=0, max_objs=0,
                 min_area=0, max_area=0):
        self.min_w = min_w
        self.min_h = min_h
        self.min_objs = min_objs
        self.max_objs = max_objs
        self.min_area = min_area
        self.max_area = max_area


# ============================================================
# Основной анализатор: фильтрация + отрисовка + экспорт
# ============================================================
class DatasetAnalyzer:
    def __init__(self, dataset_type=0, input_format='auto'):
        self.dataset_type = dataset_type
        self.input_format = input_format

    def set_dataset_type(self, dt):
        self.dataset_type = dt

    def set_input_format(self, fmt):
        self.input_format = fmt

    # ==================================================================
    # Данные из class_table
    # ==================================================================
    @staticmethod
    def get_included_class_ids(class_table):
        included = set()
        for row in range(class_table.rowCount()):
            cb = class_table.cellWidget(row, 4)
            if cb and cb.isChecked():
                try:
                    included.add(int(class_table.item(row, 1).text()))
                except (ValueError, TypeError, AttributeError):
                    continue
        return included

    @staticmethod
    def get_class_names_by_orig_id(class_table):
        names = {}
        for row in range(class_table.rowCount()):
            try:
                orig = int(class_table.item(row, 1).text())
            except (ValueError, TypeError, AttributeError):
                continue
            name_item = class_table.item(row, 2)
            names[orig] = name_item.text().strip() if name_item else str(orig)
        return names

    @staticmethod
    def get_class_names_included(class_table):
        names = {}
        for row in range(class_table.rowCount()):
            cb = class_table.cellWidget(row, 4)
            if not (cb and cb.isChecked()):
                continue
            try:
                orig_id = int(class_table.item(row, 1).text())
            except (ValueError, TypeError, AttributeError):
                continue
            name_item = class_table.item(row, 2)
            name = name_item.text().strip() if name_item else str(orig_id)
            names[orig_id] = name or str(orig_id)
        return names

    # ==================================================================
    # Фильтрация
    # ==================================================================
    def filter_pairs(self, pairs, included_class_ids, total_classes, params,
                     progress_cb=None, include_empty=False):
        """Фильтрует пары по площади / объектам / размеру.

        include_empty — включать в результат изображения с пустой
        (0 объектов) разметкой. Они попадают в filtered_pairs и в
        total_objects не увеличивают счётчик, но считаются в
        empty_images.
        """
        empty = {
            'filtered_pairs': [], 'class_counts': {}, 'class_areas': {},
            'total_objects': 0, 'object_areas': [], 'empty_images': 0,
            'error': None,
        }
        if not pairs:
            return empty
        if total_classes > 0 and len(included_class_ids) == 0:
            empty['error'] = "Не выбрано ни одного класса — фильтрация невозможна."
            return empty

        class_filter_active = (0 < len(included_class_ids) < total_classes)
        min_objs = params.min_objs
        max_objs = params.max_objs if params.max_objs > 0 else float('inf')
        min_area = params.min_area
        max_area = params.max_area if params.max_area > 0 else float('inf')

        valid_pairs = []
        class_counts = {}
        class_areas = {}
        total_objects = 0
        object_areas = []
        empty_images = 0
        total = len(pairs)

        for i, pair in enumerate(pairs):
            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                if progress_cb:
                    progress_cb(i + 1, total)
                continue
            if w < params.min_w or h < params.min_h:
                if progress_cb:
                    progress_cb(i + 1, total)
                continue

            anns = self._load_annotations_for_pair(
                lbl_path, included_class_ids if class_filter_active else None)
            num_objs = len(anns)
            if num_objs < min_objs or num_objs > max_objs:
                if progress_cb:
                    progress_cb(i + 1, total)
                continue

            valid_objs_in_img = 0
            for ann in anns:
                try:
                    cls_id = int(ann[1])
                except (ValueError, TypeError, IndexError):
                    continue
                area = self._annotation_area(ann, w, h)
                if min_area <= area <= max_area:
                    valid_objs_in_img += 1
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    class_areas.setdefault(cls_id, []).append(area)
                    total_objects += 1
                    object_areas.append(area)

            # Пустые аннотации: num_objs == 0 и min_objs == 0
            # (нет смысла включать пустые, если явно запрошено «мин. 1 объект»).
            is_empty_image = (num_objs == 0)
            include_this = False
            if valid_objs_in_img > 0:
                include_this = True
            elif include_empty and is_empty_image and min_objs == 0:
                include_this = True
                empty_images += 1
                # Пустые изображения также попадают в file_list как «прошли»
                # но НЕ считаются как объекты в metrics.

            if include_this:
                valid_pairs.append(pair)

            if progress_cb:
                progress_cb(i + 1, total)

        return {
            'filtered_pairs': valid_pairs,
            'class_counts': class_counts,
            'class_areas': class_areas,
            'total_objects': total_objects,
            'object_areas': object_areas,
            'empty_images': empty_images,
            'error': None,
        }

    def _load_annotations_for_pair(self, lbl_path, included_class_ids):
        if not lbl_path:
            return []
        anns = []
        if self.dataset_type == 0:
            try:
                anns = load_annotations(lbl_path, 1, 1,
                                        format_hint=self.input_format)
            except Exception:
                anns = []
        elif self.dataset_type == 1:
            try:
                mask = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    unique_cls = np.unique(mask)
                    anns = [(0, int(c), 0, 0, 0, 0)
                            for c in unique_cls if c != 0]
            except Exception:
                anns = []
        if included_class_ids is not None:
            anns = [a for a in anns if int(a[1]) in included_class_ids]
        return anns

    def _annotation_area(self, ann, w, h):
        """Площадь в px²: bbox → w·bw·h·bh; segment/OBB → shoelace."""
        if not ann:
            return 0
        typ = ann[0]

        if typ == 'detect' and len(ann) >= 6:
            try:
                bw, bh = float(ann[4]), float(ann[5])
                return (bw * w) * (bh * h)
            except (ValueError, TypeError):
                return 0

        if typ in ('segment', 'obb') and len(ann) >= 3:
            try:
                pts = ann[2]
                if not pts or len(pts) < 6:
                    return 0
                n = len(pts) // 2
                area = 0.0
                for i in range(n):
                    x1 = float(pts[2 * i]) * w
                    y1 = float(pts[2 * i + 1]) * h
                    j = (i + 1) % n
                    x2 = float(pts[2 * j]) * w
                    y2 = float(pts[2 * j + 1]) * h
                    area += x1 * y2 - x2 * y1
                return abs(area) / 2.0
            except (ValueError, TypeError, IndexError):
                return 0

        return 0

    # ==================================================================
    # Отрисовка метрик
    #   • 2 гистограммы площадей (общая + по классам);
    #   • таблица классов;
    #   • опционально: круговая + столбчатая диаграммы классов
    #     (передаются через ax_pie/ax_bar; если не переданы —
    #     старое поведение полностью сохраняется).
    # ==================================================================
    @staticmethod
    def render_metrics(metrics_summary_label, class_stats_table,
                       ax_all, ax_classes, canvas,
                       filtered_pairs, all_pairs, class_counts, total_objects,
                       object_areas, class_areas, name_by_orig,
                       ax_pie=None, ax_bar=None):
        metrics_summary_label.setText(
            f"<b>Снимков после фильтрации:</b> "
            f"{len(filtered_pairs)} из {len(all_pairs)}<br>"
            f"<b>Всего объектов (по выбранным классам):</b> {total_objects}<br>"
            f"<b>Уникальных классов:</b> {len(class_counts)}"
        )

        # ---- Таблица классов ----
        class_stats_table.setRowCount(0)
        sorted_classes = sorted(class_counts.items(),
                                key=lambda x: x[1], reverse=True)
        class_stats_table.setRowCount(len(sorted_classes))
        for row, (cls_id, count) in enumerate(sorted_classes):
            name = name_by_orig.get(cls_id, str(cls_id))
            item0 = QTableWidgetItem(f"{cls_id}: {name}")
            item0.setFlags(item0.flags() & ~Qt.ItemIsEditable)
            item1 = QTableWidgetItem(str(count))
            item1.setFlags(item1.flags() & ~Qt.ItemIsEditable)
            item2 = QTableWidgetItem(
                f"{(count / total_objects) * 100:.1f}%"
                if total_objects else "0%")
            item2.setFlags(item2.flags() & ~Qt.ItemIsEditable)
            class_stats_table.setItem(row, 0, item0)
            class_stats_table.setItem(row, 1, item1)
            class_stats_table.setItem(row, 2, item2)

        # ---- Гистограммы площадей ----
        ax_all.clear()
        if object_areas:
            ax_all.hist(object_areas, bins=50, color='skyblue',
                        edgecolor='black', log=True)
            ax_all.set_title("Площади: все объекты")
            ax_all.set_xlabel("Площадь (px²)")
            ax_all.set_ylabel("Кол-во (Log)")
        else:
            ax_all.text(0.5, 0.5, "Нет данных", ha='center', va='center')
            ax_all.set_title("Площади: все объекты")

        ax_classes.clear()
        if class_areas:
            top = sorted(class_areas.items(),
                         key=lambda kv: len(kv[1]), reverse=True)[:10]
            colors = ['C0', 'C1', 'C2', 'C3', 'C4',
                      'C5', 'C6', 'C7', 'C8', 'C9']
            for i, (cls_id, areas) in enumerate(top):
                name = name_by_orig.get(cls_id, str(cls_id))
                ax_classes.hist(areas, bins=40, alpha=0.55,
                                color=colors[i % len(colors)],
                                edgecolor='black', log=True,
                                label=f"{cls_id}: {name} (n={len(areas)})")
            ax_classes.legend(fontsize=6, loc='upper right')
            ax_classes.set_title("Площади по классам (топ-10)")
            ax_classes.set_xlabel("Площадь (px²)")
            ax_classes.set_ylabel("Log")
        else:
            ax_classes.text(0.5, 0.5, "Нет данных",
                            ha='center', va='center')
            ax_classes.set_title("Площади по классам")

        canvas.draw()

        # ---- Круговая + столбчатая диаграммы распределения по классам ----
        if ax_pie is None or ax_bar is None:
            return

        ax_pie.clear()
        ax_bar.clear()

        if not class_counts:
            ax_pie.text(0.5, 0.5, "Нет данных",
                        ha='center', va='center')
            ax_pie.set_title("Распределение по классам", fontsize=9)
            ax_bar.text(0.5, 0.5, "Нет данных",
                        ha='center', va='center')
            ax_bar.set_title("Кол-во по классам", fontsize=9)
            try:
                ax_pie.figure.tight_layout()
                ax_pie.figure.canvas.draw()
            except Exception:
                pass
            return

        # Круговая: топ-9 + «Прочие» для читаемости
        if len(sorted_classes) > 10:
            top_pie = sorted_classes[:9]
            other_count = sum(c for _, c in sorted_classes[9:])
            labels_pie = [
                f"{cid}: {name_by_orig.get(cid, cid)}"
                for cid, _ in top_pie
            ] + [f"Прочие ({len(sorted_classes) - 9} кл.)"]
            sizes_pie = [c for _, c in top_pie] + [other_count]
        else:
            labels_pie = [
                f"{cid}: {name_by_orig.get(cid, cid)}"
                for cid, _ in sorted_classes
            ]
            sizes_pie = [c for _, c in sorted_classes]

        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('tab20')
            colors = [cmap(i % 20) for i in range(len(sizes_pie))]
        except Exception:
            colors = None

        try:
            ax_pie.pie(
                sizes_pie,
                labels=labels_pie,
                autopct='%1.0f%%',
                startangle=90,
                textprops={'fontsize': 6},
                colors=colors,
            )
            ax_pie.set_title(
                f"Распределение по классам (объектов: {total_objects})",
                fontsize=9)
        except Exception:
            ax_pie.text(0.5, 0.5, "Ошибка отрисовки",
                        ha='center', va='center')
            ax_pie.set_title("Распределение по классам", fontsize=9)

        # Столбчатая (горизонтальная — удобнее длинные имена)
        try:
            names_bar = [
                f"{cid}: {name_by_orig.get(cid, cid)}"
                for cid, _ in sorted_classes
            ]
            counts_bar = [c for _, c in sorted_classes]
            y_pos = list(range(len(counts_bar)))
            ax_bar.barh(y_pos, counts_bar,
                        color='skyblue', edgecolor='black')
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(names_bar, fontsize=6)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Кол-во объектов", fontsize=7)
            ax_bar.set_title(
                f"Кол-во по классам (классов: {len(counts_bar)})",
                fontsize=9)
            for i, c in enumerate(counts_bar):
                ax_bar.text(c, i, f"  {c}", va='center', fontsize=6)
        except Exception:
            ax_bar.text(0.5, 0.5, "Ошибка отрисовки",
                        ha='center', va='center')
            ax_bar.set_title("Кол-во по классам", fontsize=9)

        try:
            ax_pie.figure.tight_layout()
            ax_pie.figure.canvas.draw()
        except Exception:
            pass

    # ==================================================================
    # Экспорт
    # ==================================================================
    @staticmethod
    def export_subset(filtered_pairs, out_dir, class_names,
                      progress_cb=None, log_cb=None,
                      included_class_ids=None):
        for d in ('images/train', 'images/val',
                  'labels/train', 'labels/val'):
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

        sorted_included = sorted(class_names.keys())
        remap = {old: new for new, old in enumerate(sorted_included)}

        total = len(filtered_pairs)
        skipped_lines = 0
        skipped_pairs_labels = 0
        empty_pairs = 0

        for i, pair in enumerate(filtered_pairs):
            img_path = pair[0]
            lbl_path = pair[1] if len(pair) > 1 else None

            shutil.copy(img_path, os.path.join(
                out_dir, 'images', 'train', os.path.basename(img_path)))

            # Пустые аннотации (нет файла метки или он пуст) — просто
            # копируем изображение; .txt создаём пустым, чтобы YOLO
            # корректно распознавал «background image».
            if not lbl_path or not os.path.exists(lbl_path):
                empty_pairs += 1
                if lbl_path:
                    out_lbl = os.path.join(
                        out_dir, 'labels', 'train',
                        os.path.basename(lbl_path))
                    try:
                        open(out_lbl, 'w', encoding='utf-8').close()
                    except Exception:
                        pass
            else:
                out_lbl = os.path.join(
                    out_dir, 'labels', 'train',
                    os.path.basename(lbl_path))
                try:
                    with open(lbl_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    kept = []
                    for line in lines:
                        s = line.strip()
                        if not s or s.startswith('#'):
                            kept.append(line)
                            continue
                        parts = s.split(None, 1)
                        try:
                            cls = int(parts[0])
                        except (ValueError, IndexError):
                            skipped_lines += 1
                            continue
                        if (included_class_ids is not None
                                and cls not in included_class_ids):
                            continue
                        if cls not in remap:
                            skipped_lines += 1
                            continue
                        new_cls = remap[cls]
                        rest = parts[1] if len(parts) > 1 else ''
                        new_line = (f"{new_cls} {rest}\n" if rest
                                    else f"{new_cls}\n")
                        kept.append(new_line)
                    with open(out_lbl, 'w', encoding='utf-8') as f:
                        f.writelines(kept)
                    if not kept:
                        empty_pairs += 1
                except Exception as e:
                    skipped_pairs_labels += 1
                    if log_cb:
                        log_cb(f"⚠️ Не удалось обработать {lbl_path}: {e}")
                    try:
                        shutil.copy(lbl_path, out_lbl)
                    except Exception:
                        pass

            if progress_cb:
                progress_cb(i + 1, total)

        yaml_path = os.path.join(out_dir, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {out_dir}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write(f"nc: {len(class_names)}\n")
            f.write("names:\n")
            for old_id in sorted_included:
                f.write(f"  {remap[old_id]}: {class_names[old_id]}\n")

        if log_cb:
            log_cb(f"Экспорт завершен. Путь: {out_dir}")
            if empty_pairs:
                log_cb(f"ℹ️ Изображений без аннотаций: {empty_pairs}")
            if skipped_lines:
                log_cb(f"⚠️ Пропущено строк: {skipped_lines}")
            if skipped_pairs_labels:
                log_cb(f"⚠️ .txt не перезаписаны: {skipped_pairs_labels}")
        return yaml_path


# ============================================================
# Дисбаланс классов
# ============================================================
class ClassImbalanceAnalyzer:
    @staticmethod
    def analyze(class_counts):
        if not class_counts or len(class_counts) < 2:
            return {
                'imbalance_ratio': 1.0, 'gini': 0.0, 'entropy': 0.0,
                'majority_class': None, 'minority_class': None,
                'warning': "Недостаточно классов для оценки дисбаланса.",
            }

        counts = np.array([c for c in class_counts.values() if c > 0],
                          dtype=np.float64)
        if len(counts) < 2:
            return {
                'imbalance_ratio': 1.0, 'gini': 0.0, 'entropy': 0.0,
                'majority_class': None, 'minority_class': None,
                'warning': "Недостаточно классов для оценки дисбаланса.",
            }

        ratio = float(counts.max() / counts.min())
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())

        sorted_c = np.sort(counts)
        n = len(sorted_c)
        gini = float(
            (2 * np.sum(np.arange(1, n + 1) * sorted_c) /
             (n * sorted_c.sum())) - (n + 1) / n
        )

        ids = list(class_counts.keys())
        majority = ids[int(np.argmax(counts))]
        minority = ids[int(np.argmin(counts))]

        if ratio >= 100:
            warn = ("🚨 КРИТИЧЕСКИЙ дисбаланс (≥100:1). "
                    "Редкие классы могут игнорироваться.")
        elif ratio >= 10:
            warn = ("⚠️ Сильный дисбаланс (≥10:1). "
                    "Рекомендуется oversampling / weighted loss.")
        elif ratio >= 3:
            warn = ("ℹ️ Умеренный дисбаланс (≥3:1). "
                    "Стоит проверить распределение.")
        else:
            warn = "✅ Дисбаланс в пределах нормы."

        return {
            'imbalance_ratio': ratio, 'gini': gini, 'entropy': entropy,
            'majority_class': majority, 'minority_class': minority,
            'warning': warn,
        }


# ============================================================
# Разрешение изображений
# ============================================================
class ResolutionAnalyzer:
    @staticmethod
    def analyze(pairs, progress_cb=None, cancel_check=None):
        widths, heights, aspects = [], [], []
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
            'widths': widths,
            'heights': heights,
            'aspects': aspects,
            'median_w': float(np.median(widths)),
            'median_h': float(np.median(heights)),
            'recommended_imgsz':
                int(max(np.median(widths), np.median(heights))),
            'cancelled': cancelled,
        }


# ============================================================
# Яркость
# ============================================================
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
            try:
                img = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    means.append(float(img.mean()))
            except Exception:
                pass
            _safe_progress(progress_cb, i, total)

        if not means:
            return None

        arr = np.array(means)
        return {
            'means': means,
            'dark_count': int((arr < BrightnessAnalyzer.DARK_BOUND).sum()),
            'bright_count':
                int((arr > BrightnessAnalyzer.BRIGHT_BOUND).sum()),
            'mean_brightness': float(arr.mean()),
            'cancelled': cancelled,
        }