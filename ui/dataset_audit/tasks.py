"""Revision-safe background scans and guarded result delivery to the workspace."""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from ui.dataset_audit.workers import AuditWorker
from utils_new.dataset_audit.scans import ScanJob


class ScanWorker(AuditWorker):
    job_type = ScanJob


class ScanController(QObject):
    settled = pyqtSignal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.worker = None
        self.kind = None

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.owner.analysis_status.setText("Остановка проверки…")

    def invalidate(self):
        if self.worker:
            self.worker.close()
        self.worker = None
        self._busy(False)

    def _busy(self, busy):
        for name in (
            "btn_run_quality",
            "btn_run_duplicates",
            "btn_run_heatmap",
            "btn_run_basic",
        ):
            getattr(self.owner, name).setEnabled(not busy)
        for name in (
            "btn_stop_quality",
            "btn_stop_duplicates",
            "btn_stop_heatmap",
            "btn_stop_basic",
        ):
            getattr(self.owner, name).setEnabled(busy)
        self.owner.analysis_progress.setVisible(busy)

    def start(self, kind):
        owner = self.owner
        if owner._closing or self.worker:
            return
        pairs = owner._get_analysis_pairs()
        if not pairs:
            QMessageBox.warning(
                owner,
                "Нет данных",
                "В выбранном наборе нет снимков. Загрузите данные или измените фильтры.",
            )
            return
        try:
            options = dict(
                quality=owner.quality_options(),
                phash_threshold=owner.phash_threshold.value(),
                splits=dict(owner._analysis_splits),
                dataset_type=owner.dataset_type,
                input_format=owner.input_format,
                grid_size=owner.heatmap_grid_spin.value(),
            )
            self.worker = ScanWorker(kind, pairs, options)
            self.worker.revision = owner._dataset_revision
            self.kind = kind
            self.worker.progress.connect(self._progress)
            self.worker.finished_signal.connect(self._finished)
            self._busy(True)
            owner.analysis_status.setText(
                "Проверка выполняется. Можно продолжать просмотр; смена данных отменит результат."
            )
            self.worker.start()
        except Exception as exc:
            self.worker = None
            self._busy(False)
            owner._audit_exception("Не удалось запустить проверку", exc)
            owner.analysis_status.setText(f"Ошибка запуска: {exc}")

    def _progress(self, done, total):
        if self.sender() is not self.worker or self.owner._closing:
            return
        self.owner.analysis_progress.setRange(0, total)
        self.owner.analysis_progress.setValue(done)
        labels = {
            "quality": "quality_summary_label",
            "duplicates": "dup_summary_label",
            "heatmap": "heatmap_summary_label",
        }
        if self.kind in labels:
            getattr(self.owner, labels[self.kind]).setText(
                f"Проверка: {done}/{total}. Для отмены нажмите «Остановить»."
            )

    def _finished(self, result):
        if (
            self.sender() is not self.worker
            or self.owner._closing
            or result.get("_dataset_revision") != self.owner._dataset_revision
        ):
            return
        kind = self.kind
        self.worker = None
        self._busy(False)
        try:
            if result.get("error"):
                raise RuntimeError(result["error"])
            if result.get("cancelled"):
                self.owner.analysis_status.setText(
                    "Проверка отменена. Результат не обновлён."
                )
                labels = {
                    "quality": "quality_summary_label",
                    "duplicates": "dup_summary_label",
                    "heatmap": "heatmap_summary_label",
                }
                if kind in labels:
                    getattr(self.owner, labels[kind]).setText("Проверка отменена.")
                return
            results = result if kind == "basic" else {kind: result}
            for key, value in results.items():
                if key.startswith("_"):
                    continue
                getattr(self.owner, "present_" + key)(value)
            self.owner.analysis_status.setText(
                "Проверка завершена. Откройте результаты и проверьте кандидатов перед экспортом."
            )
            self.owner.refresh_workspace_summary()
        except Exception as exc:
            self.owner.analysis_status.setText(
                f"Ошибка отображения/проверки: {exc}. Подробности в журнале."
            )
            self.owner._audit_exception("Ошибка результата анализа", exc)
        finally:
            self.settled.emit()
