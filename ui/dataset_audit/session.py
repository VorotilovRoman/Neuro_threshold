"""Dataset analysis: session Qt controller."""

import traceback

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from ui.dataset_audit import presentation as audit_presentation
from ui.settings_service import settings
from utils_new.dataset_audit.splits import split_memberships

get_class_color = settings.get_dataset_class_color


class DatasetSessionController:
    def _sync_all_states(self, *args):
        super()._sync_all_states(*args)
        self.btn_labels.setText(
            {0: "Папка разметки", 1: "Папка масок", 2: "Без разметки"}.get(
                self.dataset_type_combo.currentData(), "Папка разметки"
            )
        )

    def _invalidate_analysis_results(self):
        self._dataset_revision += 1
        if hasattr(self, "scans"):
            self.scans.invalidate()
        try:
            self._analysis_splits = split_memberships(
                getattr(self, "_source_yaml_path", None), self.pairs
            )
        except (OSError, ValueError) as exc:
            self._analysis_splits = {}
            self.log(f"Не удалось прочитать split YAML: {exc}")
        self._filters_applied = False
        self._applied_filter_settings = None
        self._stop_requested = False
        for name in (
            "_audit_thread",
            "_cluster_thread",
            "_report_thread",
            "_val_thread",
        ):
            worker = getattr(self, name, None)
            if worker is not None:
                worker.close()
        for name in (
            "_last_quality",
            "_last_dup",
            "_last_imbalance",
            "_last_heatmap",
            "_last_audit_meta",
            "_last_cluster",
            "_last_cluster_data",
            "_last_validation",
            "_last_filter_stats",
        ):
            setattr(self, name, None)
        for name in (
            "_last_audit_results",
            "_last_audit_all_results",
            "_last_audit_display",
            "_dup_groups_by_row",
        ):
            setattr(self, name, [])
        for name in ("dup_table", "audit_table", "val_table", "quality_table"):
            getattr(self, name).setRowCount(0)
        for name in (
            "quality_summary_label",
            "dup_summary_label",
            "heatmap_summary_label",
            "audit_summary_label",
            "cluster_summary_label",
            "val_summary_label",
        ):
            getattr(self, name).setText("Данные изменены. Выполните анализ заново.")
        for name in (
            "btn_run_quality",
            "btn_run_duplicates",
            "btn_run_heatmap",
            "btn_run_audit",
            "btn_run_cluster",
            "btn_run_validation",
            "btn_generate_report",
        ):
            getattr(self, name).setEnabled(True)
        for name in (
            "btn_stop_quality",
            "btn_stop_duplicates",
            "btn_stop_heatmap",
            "btn_stop_audit",
            "btn_stop_cluster",
            "btn_stop_validation",
        ):
            getattr(self, name).setEnabled(False)
        if self._heatmap_colorbar is not None:
            try:
                self._heatmap_colorbar.remove()
            except (ValueError, KeyError):
                pass
            self._heatmap_colorbar = None
        for prefix in ("quality", "heatmap", "cluster"):
            for ax in getattr(self, prefix + "_figure").axes:
                ax.clear()
            getattr(self, prefix + "_canvas").draw_idle()

        self.refresh_workspace_summary()

    def _audit_log(self, message, tag="AUDIT", level="INFO"):
        prefix = f"[{tag}]"
        if level == "WARN":
            prefix += "[WARN]"
        elif level == "ERROR":
            prefix += "[ERROR]"
        elif level == "OK":
            prefix += "[OK]"
        line = f"{prefix} {message}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            self.log(message)
        except Exception:
            pass

    def _audit_exception(self, context, exc=None, tag="AUDIT"):
        tb = traceback.format_exc()
        try:
            print(f"[{tag}][ERROR] {context}: {exc}")
            print(tb)
        except Exception:
            pass
        try:
            self.log(f"❌ {context}: {exc}")
            self.log(tb)
        except Exception:
            pass

    def _after_load_hook(self):
        self._invalidate_analysis_results()
        self._pair_class_cache = {}
        self._last_cluster_data = None
        self._audit_log(
            f"Датасет загружен: {len(self.pairs)} пар, "
            f"тип={self.dataset_type}, формат={self.input_format}",
            tag="ANALYSIS",
            level="OK",
        )
        self.apply_filters_and_update_metrics()

    def _after_type_change_hook(self, new_type):
        self._invalidate_analysis_results()
        self._pair_class_cache = {}
        self._last_cluster_data = None
        self._audit_log(f"Смена dataset_type → {new_type}", tag="ANALYSIS")
        self._sync_analyzer()
        if new_type == 2:
            self.filtered_pairs = []
            self._render_metrics({}, 0, [], {})
            self._update_class_legend()

    def _sync_analyzer(self):
        self.analyzer.set_dataset_type(self.dataset_type)
        self.analyzer.set_input_format(self.input_format)

    def toggle_log(self, checked):
        self.log_widget.setVisible(bool(checked))
        self.main_v_splitter.setSizes([500, 200] if checked else [700, 0])
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def get_class_names_from_table(self):
        if self.dataset_type == 2:
            return {}
        return audit_presentation.get_class_names_included(self.class_table)

    def _build_extra_info(self):
        if not self.annotation_types_stats:
            return ""
        return (
            "  ["
            + ", ".join(f"{k}:{v}" for k, v in self.annotation_types_stats.items())
            + "]"
        )

    def _get_analysis_pairs(self):
        if self.scope_combo.currentData() == "all":
            return list(self.pairs)
        return list(self.filtered_pairs if self._filters_applied else self.pairs)

    def _on_include_empty_toggled(self, *_):
        include_empty = bool(self.include_empty_annotations.isChecked())
        self._audit_log(
            f"Переключён include_empty_annotations: {include_empty}", tag="ANALYSIS"
        )
        super()._on_include_empty_toggled()
        if self.pairs:
            self.apply_filters_and_update_metrics()

    def update_included_classes(self):
        super().update_included_classes()
        try:
            self._update_class_legend()
        except Exception:
            pass

    def _update_class_legend(self, class_counts=None):
        if not hasattr(self, "class_legend_layout"):
            return

        while self.class_legend_layout.count() > 0:
            it = self.class_legend_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        if self.class_table.rowCount() == 0:
            self.class_legend_label.setVisible(False)
            self.class_legend_scroll.setVisible(False)
            return

        self.class_legend_label.setVisible(True)
        self.class_legend_scroll.setVisible(True)

        name_by_orig = audit_presentation.get_class_names_by_orig_id(self.class_table)
        for row in range(self.class_table.rowCount()):
            try:
                orig = int(self.class_table.item(row, 1).text())
            except (ValueError, TypeError, AttributeError):
                continue

            cb = self.class_table.cellWidget(row, 4)
            is_included = bool(cb and cb.isChecked())
            name = name_by_orig.get(orig, str(orig))

            count_str = ""
            if class_counts and orig in class_counts:
                count_str = f" ({class_counts[orig]})"

            rw = QWidget()
            rl = QHBoxLayout(rw)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)

            box = QLabel()
            box.setFixedSize(14, 14)
            try:
                bgr = get_class_color(orig)
                if isinstance(bgr, (tuple, list)) and len(bgr) >= 3:
                    r_, g_, b_ = int(bgr[2]), int(bgr[1]), int(bgr[0])
                    box.setStyleSheet(
                        f"background-color: rgb({r_},{g_},{b_}); "
                        f"border: 1px solid #333;"
                    )
                else:
                    box.setStyleSheet("background-color: #ccc; border: 1px solid #333;")
            except Exception:
                box.setStyleSheet("background-color: #ccc; border: 1px solid #333;")
            rl.addWidget(box)

            text = f"{orig}: {name}{count_str}"
            if not is_included:
                text += " (искл.)"
            lbl = QLabel(text)
            if not is_included:
                lbl.setStyleSheet("color: #888; text-decoration: line-through;")
            rl.addWidget(lbl)
            rl.addStretch()

            self.class_legend_layout.addWidget(rw)

        self.class_legend_layout.addStretch()
