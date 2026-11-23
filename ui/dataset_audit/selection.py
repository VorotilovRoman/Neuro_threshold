"""Dataset analysis: selection Qt controller."""

import os
import subprocess
import sys as _sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QMessageBox,
    QTableWidgetItem,
)

from ui.dataset_audit import presentation as audit_presentation
from ui.dataset_audit.lifecycle import serialized_scan
from ui.settings_service import settings
from utils_new.dataset_audit.analysis import (
    ClassImbalanceAnalyzer,
    DatasetAnalyzer,
    FilterParams,
    clear_brightness_cache,
)

get_class_color = settings.get_dataset_class_color


class DatasetSelectionController:
    def apply_filters_and_update_metrics(self):
        self._sync_analyzer()
        self._invalidate_analysis_results()
        self._filters_applied = False
        if not self.pairs:
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()
            self.pair_count_label.setText(f"Всего пар: 0{self._build_extra_info()}")
            return

        included_class_ids = audit_presentation.get_included_class_ids(self.class_table)
        total_classes = self.class_table.rowCount()

        # ---- Читаем ВСЕ поля фильтра ----
        params = FilterParams(
            min_w=self.filter_min_w.value(),
            max_w=self.filter_max_w.value(),
            min_h=self.filter_min_h.value(),
            max_h=self.filter_max_h.value(),
            min_brightness=self.filter_min_brightness.value(),  # -1 = нет
            max_brightness=self.filter_max_brightness.value(),  # -1 = нет
            min_objs=self.filter_min_objs.value(),
            max_objs=self.filter_max_objs.value(),
            min_area=self.filter_min_area.value(),
            max_area=self.filter_max_area.value(),
            min_area_pct=self.filter_min_area_pct.value(),
            max_area_pct=self.filter_max_area_pct.value(),
            min_aspect=self.filter_min_aspect.value(),
            max_aspect=self.filter_max_aspect.value(),
        )
        include_empty = bool(self.include_empty_annotations.isChecked())
        self._applied_filter_settings = self._filter_settings()

        self._audit_log(
            f"Старт фильтрации: pairs={len(self.pairs)}, "
            f"classes={len(included_class_ids)}/{total_classes}, "
            f"include_empty={include_empty}, "
            f"size=[{params.min_w}..{params.max_w}]×"
            f"[{params.min_h}..{params.max_h}], "
            f"brightness=[{params.min_brightness}..{params.max_brightness}], "
            f"objs=[{params.min_objs}..{params.max_objs}], "
            f"area=[{params.min_area}..{params.max_area}]px², "
            f"area%=[{params.min_area_pct:.2f}..{params.max_area_pct:.2f}], "
            f"aspect=[{params.min_aspect:.2f}..{params.max_aspect:.2f}]",
            tag="ANALYSIS",
        )

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.pairs))
        self.progress_bar.setValue(0)

        from ui.dataset_loading import dataset_tasks
        from utils_new.dataset_audit.analysis import DatasetFilterJob
        self.btn_export_filtered.setEnabled(False)
        dataset_tasks(self).start('filters', DatasetFilterJob, list(self.pairs), self.dataset_type,
            self.input_format, included_class_ids, total_classes, params, include_empty,
            label='Фильтрация и расчёт статистики…', finished=self._filters_finished)

    def _filters_finished(self, result):
        if self._closing:
            return
        self.progress_bar.setVisible(False)
        if result.get('cancelled'):
            return
        self._filters_applied = True
        self._pair_class_cache = result.get('dominant_classes', {})
        self.btn_export_filtered.setEnabled(bool(result.get('filtered_pairs')))
        if result["error"]:
            self._audit_log(
                f"Ошибка фильтрации: {result['error']}", tag="ANALYSIS", level="ERROR"
            )
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()
            self.pair_count_label.setText(
                f"Всего пар: {len(self.pairs)} | после фильтрации: 0"
            )
            return

        for message in result.get("errors", []):
            self.log(message)
        self.filtered_pairs = result["filtered_pairs"]
        empty_imgs = result.get("empty_images", 0)

        # --- сохраняем итоги для отчёта ---
        self._last_filter_stats = {
            "filtered_pairs": len(self.filtered_pairs),
            "empty_images": empty_imgs,
            "total_objects": result["total_objects"],
            "classes_with_objects": len(result["class_counts"]),
        }
        self._audit_log(
            f"Фильтрация: {len(self.pairs)} → "
            f"{len(self.filtered_pairs)} пар, "
            f"объектов={result['total_objects']}, "
            f"пустых={empty_imgs}, "
            f"классов={len(result['class_counts'])}",
            tag="ANALYSIS",
            level="OK",
        )

        label = (
            f"Всего пар: {len(self.pairs)} | после фильтрации: "
            f"{len(self.filtered_pairs)}"
        )
        if empty_imgs:
            label += f" (пустых: {empty_imgs})"
        label += self._build_extra_info()
        self.pair_count_label.setText(label)

        self._render_metrics(
            result["class_counts"],
            result["total_objects"],
            result["object_areas"],
            result["class_areas"],
            object_areas_pct=result.get("object_areas_pct", []),
        )
        self._update_imbalance_table(result["class_counts"])
        self._update_class_legend(result["class_counts"])
        self._restyle_and_sort_file_list()
        self.refresh_workspace_summary()

    def _update_imbalance_table(self, class_counts):
        self.imbalance_table.setRowCount(0)
        if not class_counts:
            self.imbalance_summary_label.setText("Нет классов.")
            self._last_imbalance = None
            return
        res = ClassImbalanceAnalyzer.analyze(class_counts)
        rows = [
            ("Классов", str(len(class_counts))),
            ("Imbalance ratio", f"{res['imbalance_ratio']:.2f}:1"),
            ("Gini", f"{res['gini']:.4f}"),
            ("Entropy", f"{res['entropy']:.4f}"),
            ("Majority class", str(res["majority_class"])),
            ("Minority class", str(res["minority_class"])),
        ]
        TIPS = {
            "Классов": "Число классов, реально присутствующих в отфильтрованном "
            "наборе.",
            "Imbalance ratio": "Отношение max/min количества объектов между классами.\n"
            "≥3:1 — умеренный; ≥10:1 — сильный; ≥100:1 — критический.",
            "Gini": "Коэффициент Джини (0..1). 0 — равномерно, 1 — всё в одном классе.",
            "Entropy": "Энтропия Шеннона. Чем выше — тем равномернее. "
            "Максимум = ln(N).",
            "Majority class": "Класс с наибольшим числом объектов.",
            "Minority class": "Класс с наименьшим числом объектов.",
        }
        self.imbalance_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ik = QTableWidgetItem(k)
            ik.setToolTip(TIPS.get(k, ""))
            iv = QTableWidgetItem(v)
            iv.setToolTip(TIPS.get(k, ""))
            self.imbalance_table.setItem(i, 0, ik)
            self.imbalance_table.setItem(i, 1, iv)
        self.imbalance_summary_label.setText(res["warning"])
        self._last_imbalance = res
        self._audit_log(
            f"Дисбаланс: ratio={res['imbalance_ratio']:.2f}, "
            f"gini={res['gini']:.4f}, entropy={res['entropy']:.4f}",
            tag="IMBALANCE",
            level="OK",
        )

    def _render_metrics(
        self,
        class_counts,
        total_objects,
        object_areas,
        class_areas=None,
        object_areas_pct=None,
    ):
        if class_areas is None:
            class_areas = {}
        name_by_orig = audit_presentation.get_class_names_by_orig_id(self.class_table)
        audit_presentation.render_metrics(
            self.metrics_summary_label,
            self.class_stats_table,
            self.ax,
            self.ax_classes,
            self.canvas,
            self.filtered_pairs,
            self.pairs,
            class_counts,
            total_objects,
            object_areas,
            class_areas,
            name_by_orig,
            ax_pie=getattr(self, "ax_pie", None),
            ax_bar=getattr(self, "ax_bar", None),
            ax_area_pct=getattr(self, "ax_area_pct", None),
            object_areas_pct=object_areas_pct or [],
        )

    def _get_pair_dominant_class(self, pair):
        return self._pair_class_cache.get(tuple(pair))

    def _restyle_and_sort_file_list(self):
        if not self.pairs or not hasattr(self, "file_list"):
            return

        filtered_set = set()
        for p in self.filtered_pairs or []:
            img = p[0] if isinstance(p, (tuple, list)) and p else p
            aux = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else None
            filtered_set.add((img, aux))

        self.file_list.blockSignals(True)
        try:
            items = [self.file_list.takeItem(0) for _ in range(self.file_list.count())]

            kept, rejected = [], []
            for item in items:
                if item is None:
                    continue
                pair = item.data(Qt.UserRole + 1)
                if not pair:
                    item.setForeground(QColor(0, 0, 0))
                    kept.append(item)
                    continue

                if pair in filtered_set:
                    cls_id = self._get_pair_dominant_class(pair)
                    if cls_id is not None:
                        try:
                            bgr = get_class_color(cls_id)
                            if isinstance(bgr, (tuple, list)) and len(bgr) >= 3:
                                b, g, r = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                                item.setForeground(QColor(r, g, b))
                            else:
                                item.setForeground(QColor(0, 0, 0))
                        except Exception:
                            item.setForeground(QColor(0, 0, 0))
                    else:
                        item.setForeground(QColor(0, 0, 0))
                    kept.append(item)
                else:
                    item.setForeground(QColor(128, 128, 128))
                    rejected.append(item)

            for it in kept:
                self.file_list.addItem(it)
            for it in rejected:
                self.file_list.addItem(it)
        finally:
            self.file_list.blockSignals(False)

    def _on_file_list_context_menu(self, pos):
        """Правый клик по элементу списка пар."""
        item = self.file_list.itemAt(pos)
        if item is None:
            return
        pair = item.data(Qt.UserRole + 1)
        if not pair:
            return

        menu = QMenu(self)
        act_show = menu.addAction("📂 Показать в проводнике")
        menu.addSeparator()
        act_delete = menu.addAction("🗑 Удалить файлы (изображение + разметка)")

        global_pos = self.file_list.viewport().mapToGlobal(pos)
        chosen = menu.exec_(global_pos)

        if chosen == act_show:
            self._show_pair_in_explorer(pair)
        elif chosen == act_delete:
            self._delete_pair_files(pair)

    def _show_pair_in_explorer(self, pair):
        """Открывает папку с изображением пары в проводнике и выделяет файл."""
        img_path = pair[0] if isinstance(pair, (tuple, list)) and pair else None
        if not img_path:
            return
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Файл не найден", img_path)
            return

        try:
            norm_path = os.path.normpath(os.path.abspath(img_path))

            if _sys.platform.startswith("win"):
                subprocess.Popen(["explorer.exe", "/select,", norm_path])
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", "-R", norm_path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(norm_path)])

            self._audit_log(f"Открываю в проводнике: {norm_path}", tag="FS", level="OK")
        except Exception as e:
            self._audit_log(
                f"Не удалось открыть {img_path}: {e}", tag="FS", level="ERROR"
            )
            QMessageBox.warning(self, "Не удалось открыть", f"{img_path}\n{e}")

    def _delete_pair_files(self, pair, refresh_metrics=True):
        """Удаляет файлы пары (изображение + разметка/маска).

        Возвращает True, если хотя бы один файл был удалён.
        """
        img_path = pair[0] if isinstance(pair, (tuple, list)) and pair else None
        lbl_path = (
            pair[1] if isinstance(pair, (tuple, list)) and len(pair) > 1 else None
        )

        if not img_path:
            return False

        paths_to_delete = []
        if os.path.exists(img_path):
            paths_to_delete.append(img_path)
        if lbl_path and os.path.exists(lbl_path):
            paths_to_delete.append(lbl_path)

        if not paths_to_delete:
            QMessageBox.warning(
                self, "Нечего удалять", "Файлы уже удалены или не существуют."
            )
            return False

        files_list = "\n".join(paths_to_delete)
        reply = QMessageBox.question(
            self,
            "Удалить файлы?",
            f"Удалить без возможности восстановления:\n\n{files_list}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False

        errors = []
        deleted_any = False
        for p in paths_to_delete:
            try:
                os.remove(p)
                deleted_any = True
                self._audit_log(f"Удалён: {p}", tag="FS", level="OK")
            except Exception as e:
                errors.append(f"{p}: {e}")
                self._audit_log(f"Ошибка удаления {p}: {e}", tag="FS", level="ERROR")

        if not deleted_any:
            QMessageBox.warning(
                self,
                "Ошибка удаления",
                "Не удалось удалить ни один файл:\n" + "\n".join(errors),
            )
            return False

        # Инвалидируем кэш яркости для удалённых файлов,
        # чтобы они не участвовали в следующих фильтрациях.
        try:
            clear_brightness_cache()
        except Exception:
            pass

        # Удаляем из состояния
        self.pairs = [p for p in self.pairs if p[0] != img_path]
        self.filtered_pairs = [p for p in self.filtered_pairs if p[0] != img_path]
        self._pair_class_cache.pop((img_path, lbl_path), None)

        # Удаляем из списка пар
        for i in range(self.file_list.count() - 1, -1, -1):
            it = self.file_list.item(i)
            if it is None:
                continue
            p = it.data(Qt.UserRole + 1)
            if p and p[0] == img_path:
                self.file_list.takeItem(i)

        # Обновляем сохранённые результаты дубликатов
        self._on_pair_deleted_from_state(img_path)

        if refresh_metrics:
            try:
                self.apply_filters_and_update_metrics()
            except Exception as e:
                self._audit_log(
                    f"Ошибка обновления метрик: {e}", tag="ANALYSIS", level="WARN"
                )

        if errors:
            QMessageBox.warning(self, "Частичные ошибки", "\n".join(errors))
        return True

    def _on_pair_deleted_from_state(self, img_path):
        """Убирает путь из сохранённых результатов анализа дубликатов."""
        if not self._last_dup:
            return
        for key in ("exact", "near"):
            groups = self._last_dup.get(key, [])
            new_groups = []
            for g in groups:
                new_g = [p for p in g if p != img_path]
                if len(new_g) > 1:
                    new_groups.append(new_g)
            self._last_dup[key] = new_groups
        try:
            from utils_new.dataset_audit.checks import LeakageChecker

            self._last_dup["leak"] = LeakageChecker.check(
                self._last_dup["exact"] + self._last_dup["near"],
                self.pairs,
                self._analysis_splits,
            )
        except Exception:
            pass
        try:
            self._rebuild_dup_table()
        except Exception as e:
            self._audit_log(
                f"Ошибка перестройки таблицы дубликатов: {e}", tag="DUP", level="WARN"
            )

    def _rebuild_dup_table(self):
        """Перестраивает dup_table из self._last_dup."""
        if not self._last_dup:
            self.dup_table.setRowCount(0)
            self._dup_groups_by_row = []
            return

        from utils_new.dataset_audit.checks import LeakageChecker

        exact = self._last_dup.get("exact", [])
        near = self._last_dup.get("near", [])

        rows = []
        self._dup_groups_by_row = []
        overlaps = [
            row["paths"]
            for row in self._last_dup.get("leak", {}).get("groups", [])
            if len(row["paths"]) == 1 and row["leak"]
        ]
        for kind, groups in (("MD5", exact), ("pHash", near), ("YAML", overlaps)):
            for g_idx, group in enumerate(groups, 1):
                if not group:
                    continue
                membership = LeakageChecker.check(
                    [group], self.pairs, self._analysis_splits
                )["groups"][0]
                splits = membership["splits"]
                leak = membership["leak"]
                files_short = ", ".join(os.path.basename(p) for p in group[:3])
                if len(group) > 3:
                    files_short += f" … (+{len(group) - 3})"
                rows.append(
                    (
                        f"{kind} #{g_idx}",
                        str(len(group)),
                        ", ".join(f"{k}:{v}" for k, v in splits.items()),
                        files_short,
                        "⚠️" if leak else "✅",
                        leak,
                    )
                )
                self._dup_groups_by_row.append(list(group))

        self.dup_table.setRowCount(len(rows))
        for r, (kind, n, splits_str, files, leak_mark, leak) in enumerate(rows):
            for c, txt in enumerate((kind, n, splits_str, files, leak_mark)):
                it = QTableWidgetItem(txt)
                it.setToolTip(
                    f"{kind}\n"
                    f"Изображений: {n}\n"
                    f"Сплиты: {splits_str}\n"
                    f"Файлы: {files}\n"
                    f"Утечка: {'ДА' if leak else 'нет'}\n"
                    f"Клик — выделить пары; двойной клик — открыть."
                )
                if leak:
                    it.setBackground(QColor(255, 220, 220))
                self.dup_table.setItem(r, c, it)

    @serialized_scan
    def export_filtered_dataset(self):
        if not self.filtered_pairs:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Папка для датасета")
        if not out_dir:
            return
        try:
            class_names = (
                audit_presentation.get_class_names_included
                if self.chk_exclude_classes_from_files.isChecked()
                else audit_presentation.get_class_names_by_orig_id
            )(self.class_table)

            included_class_ids = None
            if self.chk_exclude_classes_from_files.isChecked():
                included_class_ids = audit_presentation.get_included_class_ids(
                    self.class_table
                )
                self._audit_log(
                    f"Экспорт с фильтрацией разметки по классам: "
                    f"оставлены только {sorted(included_class_ids)}",
                    tag="EXPORT",
                )

            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(self.filtered_pairs))

            def on_progress(c, t):
                self.progress_bar.setValue(c)
                if c % 50 == 0:
                    QApplication.processEvents()

            DatasetAnalyzer.export_subset(
                self.filtered_pairs,
                out_dir,
                class_names,
                progress_cb=on_progress,
                log_cb=self.log,
                cancel_check=lambda: self._closing,
                included_class_ids=included_class_ids,
                split_by_image=self._analysis_splits,
                dataset_type=self.dataset_type,
                input_format=self.input_format,
            )

            self.progress_bar.setVisible(False)
            self._last_export_dir = out_dir
            self.btn_open_last_export.setEnabled(True)
            self._audit_log(
                f"Экспорт: {len(self.filtered_pairs)} снимков → {out_dir}",
                tag="EXPORT",
                level="OK",
            )
            QMessageBox.information(
                self,
                "Успех",
                f"Датасет создан: {out_dir}\nСнимков: {len(self.filtered_pairs)}",
            )
        except Exception as e:
            self.progress_bar.setVisible(False)
            self._audit_exception("Ошибка экспорта", e, tag="EXPORT")
            if not self._closing:
                QMessageBox.critical(self, "Ошибка экспорта", str(e))
