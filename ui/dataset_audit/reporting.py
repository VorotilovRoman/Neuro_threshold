"""Dataset analysis: reporting Qt controller."""

import os
import shutil

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from ui.dataset_audit import presentation as audit_presentation
from ui.dataset_audit.lifecycle import stale


class DatasetReportController:
    def _save_figures_for_report(self, assets_dir):
        rel = {}
        base_abs = os.path.abspath(assets_dir)
        parent_dir = os.path.dirname(base_abs)

        def _save_fig(fig, name):
            try:
                path = os.path.join(base_abs, name)
                fig.savefig(path, dpi=110, bbox_inches="tight")
                return os.path.relpath(path, parent_dir).replace("\\", "/")
            except Exception as e:
                self._audit_log(
                    f"Не удалось сохранить {name}: {e}", tag="REPORT", level="WARN"
                )
                return None

        def _copy_file(src, name, key):
            try:
                dst = os.path.join(base_abs, name)
                shutil.copy2(src, dst)
                rel[key] = os.path.relpath(dst, parent_dir).replace("\\", "/")
            except Exception as e:
                self._audit_log(
                    f"Не удалось скопировать {name}: {e}", tag="REPORT", level="WARN"
                )

        if self.class_stats_table.rowCount() > 0:
            p = _save_fig(self.figure, "areas.png")
            if p:
                rel["areas"] = p

        if (
            hasattr(self, "class_chart_figure")
            and self.class_stats_table.rowCount() > 0
        ):
            p = _save_fig(self.class_chart_figure, "classes.png")
            if p:
                rel["classes"] = p

        if self._last_quality:
            p = _save_fig(self.quality_figure, "quality.png")
            if p:
                rel["quality"] = p

        if self._last_heatmap:
            p = _save_fig(self.heatmap_figure, "heatmap.png")
            if p:
                rel["heatmap"] = p

        if self._last_cluster_data:
            p = _save_fig(self.cluster_figure, "cluster.png")
            if p:
                rel["cluster"] = p

        tr = self._last_training_results
        if tr and not tr.get("error"):
            for src_name, key in (
                ("results.png", "training_results_png"),
                ("labels.jpg", "training_labels"),
                ("confusion_matrix_normalized.png", "training_confusion"),
            ):
                src = tr["main"].get(src_name)
                if src and os.path.isfile(src):
                    _copy_file(src, src_name, key)

            for name, src in tr.get("curves", {}).items():
                if os.path.isfile(src):
                    _copy_file(src, name, f"train_curve_{name}")

            if tr.get("best_model") and os.path.isfile(tr["best_model"]):
                _copy_file(tr["best_model"], "best.pt", "best_model")

            if tr.get("best_model_meta") and os.path.isfile(tr["best_model_meta"]):
                _copy_file(tr["best_model_meta"], "best_meta.json", "best_model_meta")

        return rel

    def generate_full_report(self):
        if self._report_thread and self._report_thread.isRunning():
            QMessageBox.information(
                self, "Отчёт", "Дождитесь завершения предыдущего отчёта."
            )
            return
        if not self.pairs:
            QMessageBox.warning(self, "Нет данных", "Загрузите датасет.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчёт",
            "dataset_report.md",
            "Markdown (*.md);;Все файлы (*)",
        )
        if not save_path:
            return

        base, _ext = os.path.splitext(save_path)
        assets_dir = f"{base}_assets"
        try:
            os.makedirs(assets_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось создать папку для картинок:\n{e}"
            )
            return

        self._audit_log(f"Сохранение картинок в {assets_dir}", tag="REPORT")
        images = self._save_figures_for_report(assets_dir)
        self._audit_log(f"Файлы для отчёта: {sorted(images.keys())}", tag="REPORT")

        data = self._collect_report_data()
        data["images"] = images

        sections = {
            "quality": self.chk_report_include_quality.isChecked(),
            "dup": self.chk_report_include_dup.isChecked(),
            "heatmap": self.chk_report_include_heatmap.isChecked(),
            "audit": self.chk_report_include_audit.isChecked(),
            "validation": self.chk_report_include_validation.isChecked(),
            "cluster": self.chk_report_include_cluster.isChecked(),
            "training_results": self.chk_report_include_training_results.isChecked(),
        }

        self.btn_generate_report.setEnabled(False)
        self.btn_generate_report.setText("Генерация…")

        from ui.dataset_audit.workers import ReportWorker

        self._report_thread = ReportWorker(data, save_path, sections)
        self._report_thread.finished_signal.connect(self._on_report_finished)
        self._start_audit_worker(self._report_thread)

    def _on_report_finished(self, result):
        if stale(self, result):
            return
        self.btn_generate_report.setEnabled(True)
        self.btn_generate_report.setText("💾 Сгенерировать .md отчёт")
        err = result.get("error")
        if err:
            self._audit_log(f"Ошибка отчёта: {err}", tag="REPORT", level="ERROR")
            QMessageBox.critical(self, "Ошибка", err)
            return
        path = result["path"]
        self._last_report_dir = os.path.dirname(path)
        self.btn_open_report_folder.setEnabled(True)
        self._audit_log(f"Отчёт сохранён: {path}", tag="REPORT", level="OK")
        QMessageBox.information(self, "Готово", f"Отчёт сохранён:\n{path}")

    def _filter_settings(self):
        all_ids = sorted(
            audit_presentation.get_class_names_by_orig_id(self.class_table)
        )
        included_ids = sorted(
            audit_presentation.get_included_class_ids(self.class_table)
        )
        excluded_ids = sorted(set(all_ids) - set(included_ids))
        return {
            "min_w": self.filter_min_w.value(),
            "max_w": self.filter_max_w.value(),
            "min_h": self.filter_min_h.value(),
            "max_h": self.filter_max_h.value(),
            "min_brightness": self.filter_min_brightness.value(),
            "max_brightness": self.filter_max_brightness.value(),
            "min_objs": self.filter_min_objs.value(),
            "max_objs": self.filter_max_objs.value(),
            "min_area": self.filter_min_area.value(),
            "max_area": self.filter_max_area.value(),
            "min_area_pct": self.filter_min_area_pct.value(),
            "max_area_pct": self.filter_max_area_pct.value(),
            "min_aspect": self.filter_min_aspect.value(),
            "max_aspect": self.filter_max_aspect.value(),
            "include_empty": self.include_empty_annotations.isChecked(),
            "exclude_classes_from_files": self.chk_exclude_classes_from_files.isChecked(),
            "all_class_ids": all_ids,
            "included_class_ids": included_ids,
            "excluded_class_ids": excluded_ids,
        }

    def _collect_report_data(self):
        data = {}

        general = {
            "Pairs": len(self.pairs),
            "Filtered": len(self.filtered_pairs),
            "Dataset type": self.dataset_type,
            "Input format": self.input_format,
            "Classes in table": self.class_table.rowCount(),
            "Device": self._device or "не запрашивалось (базовый анализ)",
            "Область проверок": self.scope_combo.currentText(),
            "Снимков в области": len(self._get_analysis_pairs()),
        }
        if self.images_folder:
            general["Images"] = self.images_folder
        if self.labels_folder:
            general["Labels"] = self.labels_folder
        if self.masks_folder:
            general["Masks"] = self.masks_folder
        data["general"] = general

        current_settings = self._filter_settings()
        applied = getattr(self, "_applied_filter_settings", None)
        data["filter_params"] = dict(applied or current_settings)
        general["Фильтры изменены после применения"] = bool(
            applied and applied != current_settings
        )

        data["filter_stats"] = (
            dict(self._last_filter_stats) if self._last_filter_stats else None
        )

        cs = []
        for r in range(self.class_stats_table.rowCount()):
            row = []
            for c in range(3):
                it = self.class_stats_table.item(r, c)
                row.append(it.text() if it else "")
            cs.append(row)
        data["class_stats"] = cs

        data["imbalance"] = dict(self._last_imbalance) if self._last_imbalance else None

        if self._last_quality:
            res = self._last_quality.get("resolution")
            br = self._last_quality.get("brightness")
            data["quality"] = {
                "median_w": res["median_w"] if res else 0,
                "median_h": res["median_h"] if res else 0,
                "recommended_imgsz": res["recommended_imgsz"] if res else 0,
                "brightness": br,
                "thresholds": self._last_quality.get("thresholds", {}),
                "checked": self._last_quality.get("checked"),
                "unreadable": self._last_quality.get("unreadable", 0),
                "candidates": self._last_quality.get("candidates", 0),
                "review_rows": [
                    row for row in self._last_quality.get("rows", []) if row["issues"]
                ],
            }
        else:
            data["quality"] = None

        if self._last_dup:
            d = self._last_dup
            data["dup"] = {
                "exact_count": len(d["exact"]),
                "exact_images": sum(len(g) for g in d["exact"]),
                "near_count": len(d["near"]),
                "near_images": sum(len(g) for g in d["near"]),
                "leaked_count": d["leak"]["leaked_count"],
                "leaked_groups": [
                    f"{rg['splits']}: "
                    + ", ".join(os.path.basename(p) for p in rg["paths"][:5])
                    for rg in d["leak"]["groups"]
                    if rg["leak"]
                ],
            }
        else:
            data["dup"] = None

        if self._last_heatmap:
            hm = self._last_heatmap
            h = hm["heatmap"]
            g = hm["grid_size"]
            top = np.argsort(h.ravel())[::-1][:3]
            top_zones = []
            for idx in top:
                gy, gx = divmod(int(idx), g)
                top_zones.append(f"({gx},{gy}): {h[gy, gx]:.0f}")
            data["heatmap"] = {
                "total_objects": hm["total_objects"],
                "grid": g,
                "top_zones": top_zones,
            }
        else:
            data["heatmap"] = None

        data["audit_meta"] = (
            dict(self._last_audit_meta) if self._last_audit_meta else None
        )
        data["audit_rows"] = [
            (
                os.path.basename(it["img_path"]),
                it["gt_count"],
                it["pred_count"],
                it["mean_iou"],
            )
            for it in self._last_audit_results[:20]
        ]

        data["validation"] = (
            dict(self._last_validation) if self._last_validation else None
        )

        if self._last_cluster:
            c = self._last_cluster
            sizes = sorted(c["sizes"].items(), key=lambda x: -x[1])[:15]
            data["cluster"] = {
                "count": c["count"],
                "n_clusters": c["n_clusters"],
                "noise": c["noise"],
                "sizes": sizes,
            }
        else:
            data["cluster"] = None

        data["training_results"] = (
            dict(self._last_training_results) if self._last_training_results else None
        )

        return data
