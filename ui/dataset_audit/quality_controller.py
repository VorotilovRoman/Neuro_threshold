"""Quality results, guided overview and actions. Numerical work lives in dataset_audit."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem

from utils_new.dataset_audit.analysis import histogram_edges


class DatasetQualityController:
    def quality_parameters_changed(self, *_):
        if hasattr(self, "scans") and self.scans.kind in ("basic", "quality"):
            self.scans.invalidate()
        self._last_quality = None
        self._draw_quality_plots(None, None)
        self.quality_table.setRowCount(0)
        self.quality_summary_label.setText(
            "Пороги изменены. Запустите проверку заново."
        )
        self.refresh_workspace_summary()

    def duplicate_parameters_changed(self, *_):
        if hasattr(self, "scans") and self.scans.kind in ("basic", "duplicates"):
            self.scans.invalidate()
        self._last_dup = None
        self._dup_groups_by_row = []
        self.dup_table.setRowCount(0)
        self._clear_duplicate_highlights()
        self.dup_summary_label.setText(
            "Порог похожести изменён. Запустите проверку заново."
        )
        self.refresh_workspace_summary()

    def run_quality_analysis(self):
        self.scans.start("quality")

    def stop_quality_analysis(self):
        self.scans.cancel()

    def run_basic_analysis(self):
        self.scans.start("basic")

    def quality_options(self):
        return dict(
            dark=self.quality_dark.value(),
            bright=self.quality_bright.value(),
            min_contrast=self.quality_contrast.value(),
            min_sharpness=self.quality_sharpness.value(),
        )

    def present_quality(self, result):
        self._last_quality = result
        self.quality_summary_label.setText(
            f"Прочитано: {result['checked']}; не читается: {result['unreadable']}; кандидатов для проверки: {result['candidates']}. Пороги — эвристика, а не автоматический приговор изображению."
        )
        self._populate_quality_table()
        self._draw_quality_plots(result.get("resolution"), result.get("brightness"))

    def _populate_quality_table(self, *_):
        result = self._last_quality
        self.quality_table.setSortingEnabled(False)
        self.quality_table.setRowCount(0)
        if not result:
            return
        for row in result.get("rows", []):
            if self.quality_only_issues.isChecked() and not row["issues"]:
                continue
            index = self.quality_table.rowCount()
            self.quality_table.insertRow(index)
            values = [
                Path(row["path"]).name,
                f"{row['width']}×{row['height']}" if not row["error"] else "—",
                *[
                    f"{row[key]:.1f}" if key in row else "—"
                    for key in ("brightness", "contrast", "sharpness")
                ],
                "; ".join(row["issues"]) or "По заданным порогам замечаний нет",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row["path"])
                item.setToolTip(
                    "\n".join(
                        filter(
                            None, (row["path"], "; ".join(row["issues"]), row["error"])
                        )
                    )
                )
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.quality_table.setItem(index, col, item)
        self.quality_table.setSortingEnabled(True)

    def open_quality_row(self, row, column=0):
        item = self.quality_table.item(row, 0)
        if not item:
            return
        path = item.data(Qt.UserRole)
        for index in range(self.file_list.count()):
            entry = self.file_list.item(index)
            if entry.data(Qt.UserRole) == path:
                self.file_list.setCurrentRow(index)
                self.file_list.scrollToItem(entry)
                self._on_file_list_double_clicked(entry)
                break

    def _draw_quality_plots(self, res, bright):
        axes = (self.quality_ax1, self.quality_ax2, self.quality_ax3, self.quality_ax4)
        for ax in axes:
            ax.clear()
        if res:
            for ax, key, title in zip(
                axes[:3],
                ("widths", "heights", "aspects"),
                ("Ширина, px", "Высота, px", "W/H"),
            ):
                ax.hist(res[key], bins=histogram_edges(res[key], 30))
                ax.set_title(title)
        if bright:
            axes[3].hist(bright["means"], bins=histogram_edges(bright["means"], 30))
            axes[3].set_title("Яркость, шкала 0..255")
            for key in ("dark", "bright"):
                axes[3].axvline(self._last_quality["thresholds"][key], linestyle="--")
        self.quality_figure.tight_layout()
        self.quality_canvas.draw_idle()

    def _scope_changed(self, *_):
        applied = self._filters_applied
        stats = self._last_filter_stats
        filter_settings = getattr(self, "_applied_filter_settings", None)
        self._invalidate_analysis_results()
        self._filters_applied = applied
        self._last_filter_stats = stats
        self._applied_filter_settings = filter_settings
        self.refresh_workspace_summary()

    def refresh_workspace_summary(self):
        if not hasattr(self, "workspace_summary"):
            return
        pairs = getattr(self, "pairs", [])
        selected = (
            self._get_analysis_pairs() if hasattr(self, "_filters_applied") else []
        )
        stats = self._last_filter_stats or {}
        self.workspace_summary.setText(
            f"Источник: {len(pairs)} снимков · область проверок: {len(selected)} · объектов после фильтров: {stats.get('total_objects', '—')}"
        )
        advice = []
        if not pairs:
            advice.append(
                "1. Загрузите YOLO/U-Net YAML, COCO или папку снимков и разметки."
            )
        elif not selected:
            advice.append(
                "После фильтров нет снимков. Сбросьте ограничения или выберите «Весь набор»."
            )
        elif self._last_quality or self._last_dup:
            advice.append(
                "Откройте результаты выполненных проверок. Просмотрите кандидаты, затем сохраните отчёт или экспортируйте набор."
            )
        else:
            advice.append(
                "Запустите базовую проверку. Затем откройте найденные кандидаты в разделах качества и повторов."
            )
        quality = self._last_quality
        if quality:
            advice.append(
                f"Качество: {quality['candidates']} кандидатов, {quality['unreadable']} непрочитанных снимков."
            )
        duplicates = self._last_dup
        if duplicates:
            advice.append(
                f"Повторы: {len(duplicates.get('exact', []))} точных групп, {len(duplicates.get('near', []))} похожих. Утечки: {duplicates.get('leak', {}).get('leaked_count', 0)} групп."
            )
        if self.dataset_type == 2:
            advice.append(
                "Без разметки доступны проверки снимков; геометрию и качество аннотаций оценить нельзя."
            )
        advice.append(
            "Кандидат требует просмотра. Анализ не удаляет файлы. Экспорт создаёт отдельный набор."
        )
        self.workspace_guidance.setText("\n".join(advice))
