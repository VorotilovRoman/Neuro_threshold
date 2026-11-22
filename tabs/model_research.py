# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import sys
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from ui import dataset_source as source_dialogs
from ui.autoreseach_ui.evaluation_controller import EvaluationController
from ui.autoreseach_ui.model_research_ui import setup_model_research_ui
from ui.segmentation.preset_bindings import GradientPresetManager, ThresholdPresetManager
from utils_new.import_export.evaluation_export import export_evaluation
from utils_new.autoreseach_utils.workflow import input_guidance, selected_presets


class ModelResearchApp(QMainWindow):
    """One input set and one measured comparison; no destructive report filters."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Выбор методов по ручному эталону")
        self.resize(1250, 850)
        self._report_source = None
        self._report_paths = []
        self._collection = None
        self.ground_truth_path = None
        self._evaluation_revision = 0
        self._evaluation_result = None
        self._notice = ""
        self._error = False
        setup_model_research_ui(self)
        self.evaluation_controller = EvaluationController(self)
        self.btn_truth_yaml.clicked.connect(self.select_truth_yaml)
        self.btn_truth_folder.clicked.connect(self.select_truth_folder)
        self.btn_report_files.clicked.connect(self.select_report_files)
        self.btn_select.clicked.connect(self.select_folder)
        self.btn_clear_reports.clicked.connect(self.clear_report_inputs)
        self.btn_stop_evaluation.clicked.connect(self.stop_operation)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_export_evaluation.clicked.connect(self.save_evaluation)
        self.truth_format.currentIndexChanged.connect(self._invalidate_evaluation)
        self.evaluation_iou.valueChanged.connect(self._invalidate_evaluation)
        self.class_aware.toggled.connect(self._invalidate_evaluation)
        self._update_workflow()

    def log_msg(self, message):
        self.log_widget.log(message)
        print(message)

    def _update_workflow(self):
        if self.evaluation_controller.closed:
            return
        busy = self.evaluation_controller.busy
        ready, guidance = input_guidance(self.ground_truth_path, self._collection)
        for widget in (
            self.btn_truth_yaml,
            self.btn_truth_folder,
            self.btn_report_files,
            self.btn_select,
            self.truth_format,
            self.evaluation_iou,
            self.class_aware,
        ):
            widget.setEnabled(not busy)
        self.btn_analyze.setEnabled(ready and not busy)
        self.btn_analyze.setToolTip(
            guidance
            if not ready
            else "Сравнить каждый полный пресет со всем ручным эталоном. Оценка проверит файлы, соответствие снимков и качество геометрии объектов."
        )
        self.btn_stop_evaluation.setEnabled(busy)
        self.btn_stop_evaluation.setVisible(busy)
        self.btn_clear_reports.setEnabled(
            bool(self._report_paths)
            or (busy and self.evaluation_controller.task == "load")
        )
        self.btn_export_evaluation.setEnabled(
            self._evaluation_result is not None and not busy
        )
        self.results_widget.setEnabled(not busy)
        if busy:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.lbl_evaluation_status.setText(
                "Читаю и проверяю отчёты. Окно остаётся доступным; при необходимости нажмите «Остановить»."
                if self.evaluation_controller.task == "load"
                else "Проверяю ручной эталон, соответствие снимков и качество настроек…"
            )
        else:
            self.progress_bar.setVisible(False)
            if self._notice:
                self.lbl_evaluation_status.setText(
                    self._notice
                    + ("\n" + guidance if self._evaluation_result is None else "")
                )
            else:
                self.lbl_evaluation_status.setText(guidance)
        self.lbl_evaluation_status.setToolTip(
            "Подробности доступны в журнале выполнения." if self._error else ""
        )

    def _invalidate_evaluation(self, *args):
        self._evaluation_revision += 1
        self.evaluation_controller.stop()
        self._evaluation_result = None
        self.results_widget.clear()
        self._notice = ""
        self._error = False
        self._update_workflow()

    def set_ground_truth_source(self, path):
        self.ground_truth_path = str(Path(path).resolve())
        self.truth_path.setText(self.ground_truth_path)
        self.truth_path.setToolTip(self.ground_truth_path)
        self.lbl_truth.setText(
            "Эталон выбран. При сравнении будут проверены TXT и размеры снимков. Пустой TXT допустим; отсутствующий файл — ошибка."
        )
        self._invalidate_evaluation()

    def select_truth_yaml(self):
        path, _ = source_dialogs.get_open_file(self, "yolo_yaml")
        if path:
            self.set_ground_truth_source(path)

    def select_truth_folder(self):
        path = source_dialogs.get_directory(self, "reference_folder")
        if path:
            self.set_ground_truth_source(path)

    def select_report_files(self):
        paths, _ = source_dialogs.get_open_file(self, "reports", multiple=True)
        if paths:
            self._load_report_paths(paths)

    def _load_report_paths(self, paths):
        if self.evaluation_controller.busy:
            return False
        selected = sorted(
            {*self._report_paths, *(str(Path(path).resolve()) for path in paths)}
        )
        self._invalidate_evaluation()
        accepted = self.evaluation_controller.load(selected)
        self._update_workflow()
        return accepted

    def select_folder(self):
        folder = source_dialogs.get_directory(self, "report_folder")
        if folder and not self.evaluation_controller.busy:
            self._invalidate_evaluation()
            self.evaluation_controller.load_folder(folder, self._report_paths)
            self._update_workflow()

    def clear_report_inputs(self):
        self._invalidate_evaluation()
        self._report_paths = []
        self._report_source = None
        self._collection = None
        self.lbl_reports.setText(
            "Отчёты убраны из сравнения. Файлы на диске не изменены."
        )
        self._update_workflow()

    def _on_job_progress(self, task, done, total, name):
        if self.evaluation_controller.cancel.is_set():
            return
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(round(done * 1000 / total) if total else 0)
        self.lbl_evaluation_status.setText(
            f"{'Загрузка отчётов' if task == 'load' else 'Сравнение настроек'}: {done}/{total} — {name}"
        )

    def _task_finished(self, task, result, error):
        if isinstance(error, InterruptedError):
            self._notice = "Операция остановлена. Незавершённая оценка не используется. Можно изменить входы и запустить сравнение снова."
            self.log_msg(self._notice)
            return
        if error:
            self._error = True
            action = (
                "Выберите готовые отчёты без partial-файлов. Указанный файл можно исключить, выбрав нужные отчёты отдельно."
                if task == "load"
                else "Исправьте указанные файлы, имена или размеры. Проверьте формат разметки и повторите сравнение."
            )
            self._notice = f"Не удалось {'загрузить отчёты' if task == 'load' else 'выполнить сравнение'}: {error}\nЧто сделать: {action}"
            self.log_msg(self._notice)
            return
        self._error = False
        if task == "load":
            self._collection = result
            self._report_paths = list(result.paths)
            self._report_source = result.frame
            extra = (
                f" Ошибочных вычислений: {result.failed_results}; их причины будут показаны при сравнении."
                if result.failed_results
                else ""
            )
            self.lbl_reports.setText(
                f"Файлов: {len(result.paths)} · снимков в отчётах: {result.image_count}\nПороговых отчётов: {result.threshold_files} · градиентных: {result.gradient_files}."
                + extra
            )
            self._notice = "Готовые отчёты загружены."
            if not result.threshold_files or not result.gradient_files:
                self._notice += " Сейчас будет сравниваться только один вид методов; второй можно добавить позже."
        else:
            self._evaluation_finished(result, None)
        self.log_msg(self._notice)

    def run_analysis(self):
        if self.evaluation_controller.busy:
            return False
        ready, guidance = input_guidance(self.ground_truth_path, self._collection)
        if not ready:
            self._notice = guidance
            self._update_workflow()
            return False
        self._invalidate_evaluation()
        accepted = self.evaluation_controller.start(
            self._report_source,
            self.ground_truth_path,
            self.truth_format.currentData(),
            self.evaluation_iou.value(),
            self.class_aware.isChecked(),
        )
        self._update_workflow()
        return accepted

    def _evaluation_finished(self, result, error):
        if error:
            self._task_finished("compare", None, error)
            return
        self._evaluation_result = result
        self.results_widget.display_evaluation(result)
        self.lbl_truth.setText(
            f"Эталон проверен: {result.reference_images} снимков, {result.reference_objects} размеченных объектов."
        )
        valid = int(result.rankings.eligible.sum())
        self._notice = f"Сравнение завершено. Успешно на всём эталоне проверено настроек: {valid}/{len(result.rankings)}. Вывод и дальнейшие действия — в шаге 4."
        for message in result.diagnostics:
            self.log_msg(message)
        QTimer.singleShot(0, self._reveal_results)

    def _reveal_results(self):
        if (
            not self.evaluation_controller.closed
            and self._evaluation_result is not None
        ):
            self.centralWidget().ensureWidgetVisible(self.results_widget, 0, 12)

    def stop_operation(self):
        self.evaluation_controller.stop()
        self.lbl_evaluation_status.setText(
            "Останавливаю операцию… Дождитесь завершения текущего вычисления или чтения записи."
        )

    def save_evaluation(self):
        if self._evaluation_result is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите отдельную пустую папку для сводки, настроек и результатов по снимкам",
        )
        if not folder:
            return
        try:
            export_evaluation(self._evaluation_result, folder)
            self._notice = f"Сравнение сохранено: {folder}. Начните со сводки summary.md; подробности находятся в таблицах рядом."
            self._update_workflow()
            QMessageBox.information(self, "Сравнение сохранено", self._notice)
        except Exception as exc:
            self._notice = f"Не удалось сохранить сравнение: {exc}"
            self._error = True
            self._update_workflow()

    def add_to_presets(self, frame):
        try:
            presets = selected_presets(frame)
        except (ValueError, TypeError, KeyError) as exc:
            self._notice = str(exc)
            self._update_workflow()
            return False
        managers = {}
        added = []
        existing = []
        errors = []
        for state in presets:
            kind = state["kind"]
            if kind not in managers:
                managers[kind] = (
                    ThresholdPresetManager
                    if kind == "threshold"
                    else GradientPresetManager
                )(main_window=None)
            manager = managers[kind]
            name = next(
                (key for key, value in manager.presets.items() if value == state), None
            )
            if name is not None:
                existing.append(name)
                continue
            family = "Пороговые" if kind == "threshold" else "Градиентные"
            base = f"{family} — {state['method']} (автоподбор)"
            name = base
            counter = 1
            while name in manager.presets:
                counter += 1
                name = f"{base} {counter}"
            try:
                if manager.add_preset_dict(name, state):
                    added.append(name)
                else:
                    errors.append(name)
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {exc}")
        parts = []
        if added:
            parts.append("Сохранены настройки:\n" + "\n".join(added))
        if existing:
            parts.append("Уже есть в пресетах:\n" + "\n".join(existing))
        if errors:
            parts.append("Не удалось сохранить:\n" + "\n".join(errors))
        parts.append(
            "Откройте соответствующую вкладку методов и выберите сохранённое имя в списке пресетов. Затем проверьте результат на новых снимках."
        )
        self._notice = "\n\n".join(parts)
        self.log_msg(self._notice)
        self._update_workflow()
        QMessageBox.information(self, "Пресеты методов", self._notice)
        return bool(added or existing)

    def shutdown_analysis(self):
        self.evaluation_controller.close()

    def closeEvent(self, event):
        self.shutdown_analysis()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelResearchApp()
    window.show()
    from ui.application import exec_window

    sys.exit(exec_window(window, app))
