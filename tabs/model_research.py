# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import sys
from pathlib import Path
from ast import literal_eval

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QTableWidgetItem,
)

from path_setup import get_project_root
from ui.autoreseach_ui.model_research_ui import setup_model_research_ui
from ui.autoreseach_ui.evaluation_controller import EvaluationController
from utils_new.autoreseach_utils.evaluation_export import export_evaluation
from utils_new.autoreseach_utils.reports import parse_reports, preset_from_row
from utils_new.autoreseach_utils.metrics_core import DEFAULT_IOU_THRESHOLDS
from utils_new.autoreseach_utils.prepare import (
    add_cluster_labels,
    add_meanap_metrics,
    aggregate_by_params,
    cluster_open_close,
    filter_anomaly_count,
    filter_duplicates,
    filter_full_image_object,
    filter_invalid_params,
    filter_no_true_objects,
    filter_oversegmentation,
    filter_zero_true,
    optimize_invert,
    sample_rows,
    save_unique_params,
)
from ui.segmentation.preset_bindings import (
    ThresholdPresetManager,
    GradientPresetManager,
)


class ModelResearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Research")
        self.setGeometry(200, 200, 1200, 700)

        self._report_source = None
        self._report_paths = []
        self.ground_truth_path = None
        self._evaluation_revision = 0
        self._evaluation_result = None
        self.df_full = None
        self.current_folder = ""
        self.current_display_df = None
        self.is_aggregated = False
        self.is_top_methods = False  # флаг, что загружен датасет с топ-пресетами

        setup_model_research_ui(self)

        self.project_root = get_project_root()
        self.evaluation_controller = EvaluationController(self)
        self.btn_truth_yaml.clicked.connect(self.select_truth_yaml)
        self.btn_truth_folder.clicked.connect(self.select_truth_folder)
        self.btn_report_files.clicked.connect(self.select_report_files)
        self.btn_clear_reports.clicked.connect(self.clear_report_inputs)
        self.btn_stop_evaluation.clicked.connect(self.evaluation_controller.stop)
        self.btn_export_evaluation.clicked.connect(self.save_evaluation)
        self.truth_format.currentIndexChanged.connect(self._invalidate_evaluation)
        self.evaluation_iou.valueChanged.connect(self._invalidate_evaluation)
        self.class_aware.toggled.connect(self._invalidate_evaluation)

        self.btn_select.clicked.connect(self.select_folder)
        self.btn_load.clicked.connect(self.load_dataset)
        self.btn_save.clicked.connect(self.save_dataset)
        self.btn_first.clicked.connect(self.show_first)
        self.btn_last.clicked.connect(self.show_last)
        self.btn_all.clicked.connect(self.show_all)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_random.clicked.connect(self.show_random)
        self.btn_save_sample.clicked.connect(self.save_current_sample)

        self.btn_filter_zero_true.clicked.connect(self.filter_zero_true)
        self.btn_filter_overseg.clicked.connect(self.filter_oversegmentation)
        self.btn_filter_duplicates.clicked.connect(self.filter_duplicates)
        self.btn_filter_invalid_params.clicked.connect(self.filter_invalid_params)
        self.btn_filter_anomaly_count.clicked.connect(self.filter_anomaly_count)
        self.btn_filter_full_image.clicked.connect(self.filter_full_image_object)

        # Новая кнопка удаления строк без истинных объектов
        self.btn_remove_no_true.clicked.connect(self.filter_no_true_objects)

        # Кнопка mean F1 по IoU
        self.btn_compute_f1.clicked.connect(self.compute_meanap)
        self.btn_sample_n.clicked.connect(self.sample_n_rows)
        self.btn_filter_f1.clicked.connect(self.filter_by_meanf1_threshold)
        self.btn_aggregate.clicked.connect(self.aggregate_by_params)
        self.btn_cluster.clicked.connect(self.add_cluster_labels)
        self.btn_save_unique_params.clicked.connect(self.save_unique_params)
        self.btn_cluster_co.clicked.connect(self.cluster_open_close)
        self.btn_optimize_invert.clicked.connect(self.optimize_invert)
        self.btn_global_top5.clicked.connect(self.run_global_top5)
        self.btn_load_top_methods.clicked.connect(self.load_top_methods)

        # Кнопка удаления идеальных строк (только для топ-пресетов)
        self.btn_remove_perfect.clicked.connect(self.filter_perfect_rows)

    def _invalidate_evaluation(self, *args):
        self._evaluation_revision += 1
        self.evaluation_controller.stop()
        self._evaluation_result = None
        self.results_widget.display_results(pd.DataFrame())
        self.btn_export_evaluation.setEnabled(False)
        self.lbl_evaluation_status.setText(
            "Входы или условия изменены. Запустите оценку по эталону."
        )

    def set_ground_truth_source(self, path):
        self.ground_truth_path = str(Path(path).resolve())
        self._invalidate_evaluation()
        self.lbl_truth.setText("Эталон: " + self.ground_truth_path)

    def select_truth_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ручная разметка: YOLO YAML", "", "YOLO YAML (*.yaml *.yml)"
        )
        if path:
            self.set_ground_truth_source(path)

    def select_truth_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка эталонного датасета или только TXT"
        )
        if path:
            self.set_ground_truth_source(path)

    def select_report_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить пороговые и градиентные отчёты", "", "Отчёты (*.txt)"
        )
        if paths:
            self._load_report_paths(paths)

    def _load_report_paths(self, paths):
        paths = sorted(
            set([*self._report_paths, *(str(Path(path).resolve()) for path in paths)])
        )
        try:
            frame = parse_reports(
                paths,
                self.log_msg,
                strict=True,
                include_failures=True,
                load_adjacent_truth=False,
            )
            if frame is None or frame.empty:
                raise ValueError("В отчётах нет вычислений.")
        except Exception as exc:
            QMessageBox.critical(self, "Отчёты", str(exc))
            return False
        self._invalidate_evaluation()
        self._report_paths = paths
        self._report_source = frame.copy(deep=True)
        self.df_full = frame.copy(deep=True)
        self.is_aggregated = self.is_top_methods = False
        self.display_dataframe(self.df_full.head(100))
        self._enable_buttons()
        self.update_row_count_display()
        self.lbl_evaluation_status.setText(
            f"Загружено {len(paths)} отчётов, {len(frame)} вычислений. Оценка использует весь исходный набор независимо от фильтров таблицы."
        )
        self.log_msg(self.lbl_evaluation_status.text())
        return True

    def clear_report_inputs(self):
        self._invalidate_evaluation()
        self._report_paths = []
        self._report_source = None
        self.df_full = None
        self.is_aggregated = self.is_top_methods = False
        self.display_dataframe(None)
        self.update_row_count_display()
        self._update_buttons_state()

    def _evaluation_finished(self, result, error):
        if isinstance(error, InterruptedError):
            self.lbl_evaluation_status.setText(
                "Оценка отменена. Победитель не определён."
            )
            return
        if error:
            self.log_msg(f"Ошибка оценки: {error}")
            self.lbl_evaluation_status.setText(f"Ошибка оценки: {error}")
            QMessageBox.critical(self, "Оценка", str(error))
            return
        self._evaluation_result = result
        self.results_widget.display_evaluation(result)
        self.btn_export_evaluation.setEnabled(True)
        valid = int(result.rankings.eligible.sum())
        self.lbl_evaluation_status.setText(
            f"Эталон: {result.reference_images} снимков / {result.reference_objects} объектов. Полностью проверено пресетов: {valid}/{len(result.rankings)}. Рейтинг — фактический F1 по всему эталону."
        )
        self.log_msg(self.lbl_evaluation_status.text())
        for message in result.diagnostics:
            self.log_msg(message)

    def save_evaluation(self):
        if self._evaluation_result is None:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Отдельная пустая папка для оценки"
        )
        if not path:
            return
        try:
            export_evaluation(self._evaluation_result, path)
            QMessageBox.information(
                self,
                "Оценка сохранена",
                f"{path}\nsummary.md, rankings.csv, best_methods.csv, image_details.csv, evaluation.json",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт оценки", str(exc))

    def shutdown_analysis(self):
        self.evaluation_controller.close()

    def closeEvent(self, event):
        self.shutdown_analysis()
        event.accept()

    def log_msg(self, msg):
        self.log_widget.log(msg)
        print(msg)

    def update_row_count_display(self):
        count = len(self.df_full) if self.df_full is not None else 0
        self.lbl_row_count.setText(f"Строк в датасете: {count}")

    def _update_f1_filter_button_state(self):
        if self.df_full is not None and "mean_f1" in self.df_full.columns:
            self.btn_filter_f1.setEnabled(True)
        else:
            self.btn_filter_f1.setEnabled(False)

    def _update_aggregate_button_state(self):
        if (
            self.df_full is not None
            and "mean_f1" in self.df_full.columns
            and not self.is_aggregated
        ):
            self.btn_aggregate.setEnabled(True)
        else:
            self.btn_aggregate.setEnabled(False)

    def _update_buttons_by_aggregated_flag(self):
        if self.is_aggregated:
            self.btn_filter_full_image.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_cluster.setEnabled(False)
        else:
            if self.df_full is not None and "objects" in self.df_full.columns:
                self.btn_filter_full_image.setEnabled(True)
                self.btn_compute_f1.setEnabled(True)
            self._update_aggregate_button_state()
            if self.df_full is not None:
                self.btn_cluster.setEnabled(True)
        self._guard_exploration_buttons()

    def _guard_exploration_buttons(self):
        frame = self.df_full
        truth_available = (
            frame is not None
            and not frame.empty
            and "ground_truth_available" in frame
            and frame["ground_truth_available"].isin([True, "True"]).all()
        )
        has_counts = truth_available and "num_true_objects" in frame
        for button in (
            self.btn_filter_zero_true,
            self.btn_filter_overseg,
            self.btn_filter_anomaly_count,
            self.btn_remove_no_true,
        ):
            button.setEnabled(bool(has_counts and not self.is_aggregated))
        self.btn_compute_f1.setEnabled(bool(truth_available and not self.is_aggregated))

    def _apply_prepare_operation(self, operation_func, *args, **kwargs):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return False

        try:
            new_df, msg = operation_func(self.df_full, *args, **kwargs)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
                return True
            else:
                self.log_msg(msg)
                return False
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
            return False

    # ---------- Фильтры и операции ----------
    def filter_full_image_object(self):
        self._apply_prepare_operation(filter_full_image_object)

    def filter_invalid_params(self):
        self._apply_prepare_operation(filter_invalid_params)

    def filter_anomaly_count(self):
        self._apply_prepare_operation(filter_anomaly_count, max_objects=15)

    def filter_zero_true(self):
        self._apply_prepare_operation(filter_zero_true)

    def filter_oversegmentation(self):
        self._apply_prepare_operation(filter_oversegmentation, ratio=2.0)

    def filter_duplicates(self):
        self._apply_prepare_operation(filter_duplicates)

    def filter_no_true_objects(self):
        """Удаляет строки, в которых нет истинных объектов (num_true_objects == 0)"""
        self._apply_prepare_operation(filter_no_true_objects)

    # ---------- Расчёт mean F1 по IoU ----------
    def compute_meanap(self):
        if self.is_aggregated:
            QMessageBox.warning(
                self,
                "Операция недоступна",
                "Для агрегированного датасета пересчёт метрик невозможен.",
            )
            return
        self.btn_compute_f1.setEnabled(False)
        try:
            new_df, msg = add_meanap_metrics(
                self.df_full, iou_thresholds=DEFAULT_IOU_THRESHOLDS
            )
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg("Не удалось рассчитать mean F1 по IoU")
        except Exception as e:
            self.log_msg(f"Ошибка при расчёте mean F1 по IoU: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_compute_f1.setEnabled(True)

    # Фильтрация по mean_f1
    def filter_by_meanf1_threshold(self):
        threshold = self.f1_threshold_spin.value()
        if self.df_full is None:
            return
        if "mean_f1" not in self.df_full.columns:
            QMessageBox.warning(
                self, "Нет данных", "Сначала выполните 'Расчет mean F1 по IoU'."
            )
            return
        before = len(self.df_full)
        self.df_full = self.df_full[self.df_full["mean_f1"] >= threshold].reset_index(
            drop=True
        )
        after = len(self.df_full)
        self.log_msg(
            f"Фильтр по mean_f1 (порог {threshold:.2f}): удалено {before - after} строк. Осталось {after}."
        )
        self.update_row_count_display()
        self.display_dataframe(self.df_full.head(100))
        self._update_f1_filter_button_state()
        self._update_aggregate_button_state()
        self._update_buttons_by_aggregated_flag()

    def sample_n_rows(self):
        n = self.sample_spin.value()
        self._apply_prepare_operation(sample_rows, n=n)

    def aggregate_by_params(self):
        if self.is_aggregated:
            QMessageBox.warning(self, "Уже агрегирован", "Датасет уже агрегирован.")
            return
        top_k = self.top_k_spin.value()
        success = self._apply_prepare_operation(aggregate_by_params, top_k=top_k)
        if success:
            self.is_aggregated = True
            self._update_buttons_by_aggregated_flag()

    def add_cluster_labels(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return

        result_dir = os.path.join(self.project_root, "result")
        os.makedirs(result_dir, exist_ok=True)

        try:
            self.btn_cluster.setEnabled(False)
            new_df, msg = add_cluster_labels(self.df_full, result_dir)
            if new_df is not None and not new_df.empty:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                QMessageBox.information(
                    self,
                    "Готово",
                    f"Кластеризация завершена. Добавлена колонка 'cluster'.\n{msg}",
                )
            else:
                self.log_msg(msg)
                QMessageBox.warning(self, "Ошибка", msg)
        except Exception as e:
            self.log_msg(f"Ошибка кластеризации: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_cluster.setEnabled(True)

    def save_unique_params(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return

        result_dir = os.path.join(self.project_root, "result")
        os.makedirs(result_dir, exist_ok=True)

        try:
            self.btn_save_unique_params.setEnabled(False)
            _, msg = save_unique_params(self.df_full, result_dir)
            self.log_msg(msg)
            QMessageBox.information(
                self, "Готово", f"Уникальные параметры сохранены.\n{msg}"
            )
        except Exception as e:
            self.log_msg(f"Ошибка сохранения: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
        finally:
            self.btn_save_unique_params.setEnabled(True)

    def optimize_invert(self):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return
        if "mean_f1" not in self.df_full.columns:
            QMessageBox.warning(
                self, "Нет метрики", "Сначала выполните 'Расчет mean F1 по IoU'."
            )
            return
        try:
            new_df, msg = optimize_invert(self.df_full)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg(msg)
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def cluster_open_close(self):
        if self.df_full is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите датасет.")
            return
        if "cluster" not in self.df_full.columns:
            QMessageBox.warning(
                self,
                "Нет кластеров",
                "Сначала выполните 'Кластеризация параметров' (добавьте колонку 'cluster').",
            )
            return
        n_clusters = 3
        try:
            new_df, msg = cluster_open_close(self.df_full, n_clusters=n_clusters)
            if new_df is not None:
                self.df_full = new_df
                self.log_msg(msg)
                self.update_row_count_display()
                self.display_dataframe(self.df_full.head(100))
                self._update_f1_filter_button_state()
                self._update_aggregate_button_state()
                self._update_buttons_by_aggregated_flag()
            else:
                self.log_msg(msg)
        except Exception as e:
            self.log_msg(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- Работа с топ-пресетами ----------
    def load_top_methods(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить топ-пресеты по методам",
            "",
            "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".pkl"):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)
            required = [
                "preset_json",
                "erode_px",
                "dilate_px",
                "morphology_key",
                "method",
                "rank",
                "params",
                "invert",
                "close_px",
                "open_px",
                "actual_f1",
                "pred_f1",
                "cluster",
                "cluster_co",
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Файл не содержит необходимых колонок: {missing}")
            self._invalidate_evaluation()
            self._report_paths = []
            self._report_source = (
                df.copy(deep=True) if "annotations_json" in df else None
            )
            self.df_full = df
            self.is_top_methods = True
            self.is_aggregated = False
            self.log_msg(f"Загружен датасет топ-пресетов: {len(df)} строк.")
            self.display_dataframe(self.df_full.head(100))
            self._update_buttons_state()
            self.update_row_count_display()
        except Exception as e:
            self.log_msg(f"Ошибка загрузки: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    def filter_perfect_rows(self):
        """Удаляет строки с actual_f1 >= заданного порога (только для датасета топ-пресетов)."""
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "Нет данных", "Датасет пуст.")
            return
        if not self.is_top_methods:
            QMessageBox.warning(
                self,
                "Недоступно",
                "Эта операция доступна только для датасета топ-пресетов.",
            )
            return
        if "actual_f1" not in self.df_full.columns:
            QMessageBox.warning(
                self, "Нет колонки", "В датасете отсутствует колонка actual_f1."
            )
            return

        threshold = self.perfect_threshold_spin.value()
        before = len(self.df_full)
        self.df_full = self.df_full[self.df_full["actual_f1"] < threshold].reset_index(
            drop=True
        )
        after = len(self.df_full)
        self.log_msg(
            f"Удалено строк с actual_f1 >= {threshold}: {before - after}. Осталось {after}."
        )
        self.update_row_count_display()
        self.display_dataframe(self.df_full.head(100))

    # ---------- Загрузка данных ----------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Добавить папку отчётов")
        if folder:
            paths = sorted(
                str(path)
                for path in Path(folder).rglob("*.txt")
                if path.name.endswith(("_full_report.txt", "_gradient_report.txt"))
            )
            if not paths:
                QMessageBox.warning(
                    self,
                    "Отчёты",
                    "Готовые пороговые или градиентные отчёты не найдены.",
                )
                return
            self._load_report_paths(paths)

    def load_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Dataset",
            "",
            "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".pkl"):
                df = pd.read_pickle(file_path)
            else:
                df = pd.read_csv(file_path)

            required = [
                "preset_json",
                "erode_px",
                "dilate_px",
                "morphology_key",
                "ground_truth_available",
                "objects",
                "true_objects",
                "method",
                "params",
                "invert",
                "close_px",
                "open_px",
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                self.log_msg(f"Warning: missing columns: {missing}")

            self._invalidate_evaluation()
            self._report_paths = []
            self._report_source = (
                df.copy(deep=True) if "annotations_json" in df else None
            )
            self.df_full = df
            self.log_msg(f"Loaded {len(df)} rows from {file_path}")

            if (
                "num_true_objects" not in self.df_full.columns
                and "true_objects" in self.df_full
            ):
                self.df_full["num_true_objects"] = self.df_full["true_objects"].apply(
                    lambda s: (
                        len(literal_eval(s))
                        if isinstance(s, str) and s.startswith("[")
                        else None
                    )
                )
            if (
                "num_objects" not in self.df_full.columns
                and "objects" in self.df_full.columns
            ):
                self.df_full["num_objects"] = self.df_full["objects"].apply(
                    lambda s: (
                        len(literal_eval(s))
                        if isinstance(s, str) and s.startswith("[")
                        else 0
                    )
                )

            self.is_aggregated = False
            self.display_dataframe(self.df_full.head(100))
            self._enable_buttons()
            self._update_f1_filter_button_state()
            self._update_aggregate_button_state()
            self._update_buttons_by_aggregated_flag()
            self.update_row_count_display()
        except Exception as e:
            self.log_msg(f"Error loading dataset: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{e}")
        self.is_top_methods = False
        self._update_buttons_state()

    def _enable_buttons(self):
        self.btn_save.setEnabled(True)
        self.btn_first.setEnabled(True)
        self.btn_last.setEnabled(True)
        self.btn_all.setEnabled(True)
        self.btn_analyze.setEnabled(True)
        self.btn_random.setEnabled(True)
        self.btn_save_sample.setEnabled(True)
        self.btn_filter_zero_true.setEnabled(True)
        self.btn_filter_overseg.setEnabled(True)
        self.btn_filter_duplicates.setEnabled(True)
        self.btn_filter_invalid_params.setEnabled(True)
        self.btn_filter_anomaly_count.setEnabled(True)
        self.btn_filter_full_image.setEnabled(True)
        self.btn_remove_no_true.setEnabled(True)  # новая кнопка
        self.btn_sample_n.setEnabled(True)
        self.btn_cluster.setEnabled(True)
        self.btn_save_unique_params.setEnabled(True)
        self.btn_cluster_co.setEnabled(True)
        self.btn_optimize_invert.setEnabled(True)
        self.btn_global_top5.setEnabled(True)
        self._update_buttons_by_aggregated_flag()

    # ---------- Парсинг отчётов ----------
    def parse_reports(self, report_files):
        return parse_reports(report_files, self.log_msg)

    # ---------- Отображение и сохранение ----------
    def display_dataframe(self, df):
        if df is None or df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.current_display_df = None
            return

        sorting = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)
        for column in range(len(df.columns)):
            self.table.setColumnHidden(column, False)
        self.current_display_df = df

        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                val = row[col]
                if (
                    col in ("objects", "true_objects", "preset_json", "morphology_key")
                    and isinstance(val, str)
                    and len(val) > 100
                ):
                    val = val[:100] + "..."
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        self.table.setSortingEnabled(sorting)
        self.table.resizeColumnsToContents()
        for col in ("preset_json", "morphology_key"):
            if col in df.columns:
                self.table.setColumnHidden(df.columns.get_loc(col), True)

    def save_dataset(self):
        if self.df_full is None or self.df_full.empty:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dataset",
            "",
            "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                self.df_full.to_csv(file_path, index=False, encoding="utf-8")
            elif file_path.endswith(".pkl"):
                self.df_full.to_pickle(file_path)
            else:
                self.df_full.to_csv(file_path, index=False, encoding="utf-8")
            self.log_msg(f"Dataset saved to {file_path}")
            QMessageBox.information(self, "Success", f"Saved to {file_path}")
        except Exception as e:
            self.log_msg(f"Error saving: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def show_first(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full.head(100))

    def show_last(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full.tail(100))

    def show_all(self):
        if self.df_full is not None:
            self.display_dataframe(self.df_full)

    def show_random(self):
        if self.df_full is not None and len(self.df_full) > 0:
            sample = self.df_full.sample(n=min(100, len(self.df_full)))
            self.display_dataframe(sample)
        else:
            QMessageBox.warning(self, "No Data", "Нет данных для случайной выборки.")

    def save_current_sample(self):
        if self.current_display_df is None or self.current_display_df.empty:
            QMessageBox.warning(self, "No Data", "Нет данных для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Sample",
            "",
            "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                self.current_display_df.to_csv(file_path, index=False, encoding="utf-8")
            elif file_path.endswith(".pkl"):
                self.current_display_df.to_pickle(file_path)
            else:
                self.current_display_df.to_csv(file_path, index=False, encoding="utf-8")
            self.log_msg(f"Sample saved to {file_path}")
            QMessageBox.information(
                self, "Success", f"Saved {len(self.current_display_df)} rows."
            )
        except Exception as e:
            self.log_msg(f"Error saving sample: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    # ---------- Анализ ----------
    def run_analysis(self):
        source = (
            self._report_source if self._report_source is not None else self.df_full
        )
        if source is None or source.empty:
            QMessageBox.warning(
                self, "Оценка", "Сначала загрузите исходные отчёты автоподбора."
            )
            return
        if not self.ground_truth_path:
            QMessageBox.warning(
                self,
                "Эталон",
                "Выберите ручную разметку: YOLO YAML или папку датасета/TXT.",
            )
            return
        required = {
            "preset_json",
            "source_size",
            "working_size",
            "annotations_json",
            "image_path",
        }
        if not required.issubset(source.columns):
            QMessageBox.warning(
                self,
                "Оценка",
                "Нужны исходные отчёты с геометрией и размерами. Агрегированная таблица не заменяет результаты по снимкам.",
            )
            return
        if self.evaluation_controller.busy:
            self.log_msg("Оценка ещё выполняется. Дождитесь остановки.")
            return
        self._invalidate_evaluation()
        self.btn_analyze.setEnabled(False)
        self.btn_stop_evaluation.setEnabled(True)
        self.lbl_evaluation_status.setText("Чтение эталона и сопоставление снимков…")
        self.evaluation_controller.start(
            source,
            self.ground_truth_path,
            self.truth_format.currentData(),
            self.evaluation_iou.value(),
            self.class_aware.isChecked(),
        )

    def run_global_top5(self):
        if self._evaluation_result is None:
            self.run_analysis()
        else:
            self.results_widget.display_evaluation(self._evaluation_result)

    def save_dataframe(self, df, dialog_title="Сохранить датасет"):
        if df is None or df.empty:
            QMessageBox.warning(self, "Нет данных", "Нет данных для сохранения.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            dialog_title,
            "",
            "CSV Files (*.csv);;Pickle Files (*.pkl);;All Files (*)",
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".csv"):
                df.to_csv(file_path, index=False, encoding="utf-8")
            elif file_path.endswith(".pkl"):
                df.to_pickle(file_path)
            else:
                df.to_csv(file_path, index=False, encoding="utf-8")
            self.log_msg(f"Сохранено в {file_path}")
            QMessageBox.information(self, "Успех", f"Сохранено {len(df)} строк.")
        except Exception as e:
            self.log_msg(f"Ошибка сохранения: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))

    # ---------- Добавление в пресеты ----------
    def add_to_presets(self, top5):
        if top5 is None or top5.empty:
            QMessageBox.warning(
                self, "Нет данных", "Нет данных для добавления в пресеты."
            )
            return
        try:
            presets = [preset_from_row(row) for _, row in top5.iterrows()]
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Пресет отчёта", str(exc))
            return
        managers = {}
        added = []
        for preset in presets:
            kind = preset["kind"]
            if kind not in managers:
                managers[kind] = (
                    ThresholdPresetManager
                    if kind == "threshold"
                    else GradientPresetManager
                )(main_window=None)
            preset_mgr = managers[kind]
            method = preset["method"]
            base_name = f"{kind}: {method} auto"
            name = base_name
            counter = 1
            while name in preset_mgr.presets:
                name = f"{base_name} ({counter})"
                counter += 1
            if preset_mgr.add_preset_dict(name, preset):
                added.append(name)
        if added:
            self.log_msg(f"Добавлены пресеты: {', '.join(added)}")
            QMessageBox.information(self, "Успех", f"Добавлено {len(added)} пресетов.")

    def _update_buttons_state(self):
        """Обновляет доступность кнопок в зависимости от типа загруженных данных."""
        if self.df_full is None or self.df_full.empty:
            self.btn_save.setEnabled(False)
            self.btn_first.setEnabled(False)
            self.btn_last.setEnabled(False)
            self.btn_all.setEnabled(False)
            self.btn_random.setEnabled(False)
            self.btn_save_sample.setEnabled(False)
            self.btn_filter_zero_true.setEnabled(False)
            self.btn_filter_overseg.setEnabled(False)
            self.btn_filter_duplicates.setEnabled(False)
            self.btn_filter_invalid_params.setEnabled(False)
            self.btn_filter_anomaly_count.setEnabled(False)
            self.btn_filter_full_image.setEnabled(False)
            self.btn_remove_no_true.setEnabled(False)
            self.btn_sample_n.setEnabled(False)
            self.btn_cluster.setEnabled(False)
            self.btn_save_unique_params.setEnabled(False)
            self.btn_cluster_co.setEnabled(False)
            self.btn_optimize_invert.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_filter_f1.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_global_top5.setEnabled(False)
            self.btn_remove_perfect.setEnabled(False)  # для топ-пресетов
            return

        if self.is_top_methods:
            self.btn_save.setEnabled(True)
            self.btn_first.setEnabled(True)
            self.btn_last.setEnabled(True)
            self.btn_all.setEnabled(True)
            self.btn_random.setEnabled(True)
            self.btn_save_sample.setEnabled(True)
            self.btn_filter_zero_true.setEnabled(False)
            self.btn_filter_overseg.setEnabled(False)
            self.btn_filter_duplicates.setEnabled(False)
            self.btn_filter_invalid_params.setEnabled(False)
            self.btn_filter_anomaly_count.setEnabled(False)
            self.btn_filter_full_image.setEnabled(False)
            self.btn_remove_no_true.setEnabled(False)
            self.btn_sample_n.setEnabled(False)
            self.btn_cluster.setEnabled(False)
            self.btn_save_unique_params.setEnabled(False)
            self.btn_cluster_co.setEnabled(False)
            self.btn_optimize_invert.setEnabled(False)
            self.btn_aggregate.setEnabled(False)
            self.btn_compute_f1.setEnabled(False)
            self.btn_filter_f1.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_global_top5.setEnabled(True)
            self.btn_remove_perfect.setEnabled(True)  # активна для топ-пресетов
        else:
            self._enable_buttons()
            self.btn_remove_perfect.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelResearchApp()
    window.show()
    from ui.application import exec_window

    sys.exit(exec_window(window, app))
