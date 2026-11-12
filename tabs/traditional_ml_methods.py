
# Allow both `python -m tabs.<name>` and direct execution from any directory.


if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# traditional_ml_methods.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import sys

import joblib
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSpinBox,
)

from ui.segmentation import actions as segmentation_actions
from ui.segmentation import computation as segmentation_computation
from ui.segmentation import controller as segmentation_controller
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import presentation as segmentation_presentation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import TraditionalMLPresetManager
from ui_tabs.traditional_ml_methods_ui import setup_traditional_ml_ui
from utils_new.segmentation.postprocessing import extract_objects
from utils_ops.traditional_ml_ops import (
    create_superpixels,
    extract_pixel_features,
    extract_superpixel_features,
)


class TraditionalMLWindow(QMainWindow):
    segmentation_view_names = ('original_view', 'ml_view', 'morph_view', 'annotated_view')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traditional Machine Learning")
        self.annotations = []
        self.image_paths = []
        self.display_images = []
        self.gray_images = []
        self.current_index = 0

        # Данные для ML
        self.model = None
        self.model_metadata = None
        self.prediction_mask = None
        self.cluster_means = None
        self.current_cluster_labels = None

        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None

        self.ignore_ui_changes = False

        # Словарь путей к моделям (по типу модели) – хранится временно, не на диске
        self.model_paths = {}
        # Запоминаем выбранный кластер (label), чтобы сохранять его между снимками
        self._last_selected_cluster_label = None
        setup_traditional_ml_ui(self)

        # Пресеты
        self.preset_manager = TraditionalMLPresetManager(self, preset_file="presets_traditional_ml.json")
        self.update_preset_combo()


        # Навигация

        # Кнопки ML
        self.train_button.clicked.connect(self.train_model)
        self.load_model_button.clicked.connect(self.load_model)
        self.apply_button.clicked.connect(self.apply_model)
        self.find_best_k_btn.clicked.connect(self.find_best_k)
        self.save_model_btn.clicked.connect(self.save_model)
        self.reset_to_model_btn.clicked.connect(self.reset_to_model_settings)

        # Параметры модели
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.use_superpixels.stateChanged.connect(self.on_superpixel_toggled)
        self.on_superpixel_toggled(self.use_superpixels.isChecked())
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_changed)

        # Отслеживание изменений настроек
        self._connect_settings_widgets()

        # Морфология и отрисовка



        # Инициализация состояния
        self.update_navigation_state()
        self.on_model_changed(0)
        self.check_settings_match_model()
        self._update_model_path_display()
        self._update_apply_button_status()

    # ----------------------------------------------------------------------
    #  Управление путями к моделям (только в памяти, без файла)
    # ----------------------------------------------------------------------
    def _update_model_path_display(self):
        model_name = self.model_combo.currentText()
        path = self.model_paths.get(model_name, "")
        if path and os.path.exists(path):
            self.model_path_edit.setText(path)
        else:
            if path:
                # Файл удалён – очищаем путь в словаре
                self.model_paths[model_name] = ""
            self.model_path_edit.setText("")

    @staticmethod
    def _get_expected_type_name(model_name):
        mapping = {
            "K-Means Clustering": "KMeans",
            "MeanShift": "MeanShift",
            "SVM (pixel-wise)": "SVC",
            "Random Forest": "RandomForestClassifier",
            "XGBoost": "XGBClassifier",
            "Decision Tree": "DecisionTreeClassifier"
        }
        return mapping.get(model_name, "")

    @staticmethod
    def _get_model_type_from_class(model):
        class_name = type(model).__name__
        reverse_map = {
            "KMeans": "K-Means Clustering",
            "MeanShift": "MeanShift",
            "SVC": "SVM (pixel-wise)",
            "RandomForestClassifier": "Random Forest",
            "XGBClassifier": "XGBoost",
            "DecisionTreeClassifier": "Decision Tree"
        }
        return reverse_map.get(class_name, None)

    def _auto_load_model_for_current_type(self):
        if getattr(self, '_loading_model', False) or self.jobs.busy:
            return
        name = self.model_combo.currentText()
        path = self.model_paths.get(name, '')
        if path and os.path.isfile(path) and (self.model is None or getattr(self, '_loaded_model_path', None) != (name,path)):
            segmentation_computation.load_ml(self,path)

    def _apply_model_metadata(self, metadata):
        """Обновляет UI в соответствии с метаданными модели."""
        self.ignore_ui_changes = True
        model_type = metadata.get('model_type')
        if model_type and self.model_combo.currentText() != model_type:
            idx = self.model_combo.findText(model_type)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.color_feature.setChecked(metadata.get('use_intensity', True))
        self.texture_feature.setChecked(metadata.get('use_texture', False))
        self.spatial_feature.setChecked(metadata.get('use_spatial', False))
        self.use_superpixels.setChecked(metadata.get('use_superpixels', False))
        self.superpixel_size.setValue(metadata.get('superpixel_size', 50))
        self.normalize_features.setChecked(metadata.get('normalize_features', False))
        if 'n_clusters' in metadata and metadata['n_clusters'] is not None:
            self.kmeans_clusters.setValue(metadata['n_clusters'])
        if 'kernel' in metadata and metadata['kernel'] is not None:
            kidx = self.svm_kernel_combo.findText(metadata['kernel'])
            if kidx >= 0:
                self.svm_kernel_combo.setCurrentIndex(kidx)
        if 'C' in metadata and metadata['C'] is not None:
            self.svm_c.setValue(metadata['C'])
        if 'n_trees' in metadata and metadata['n_trees'] is not None:
            self.rf_trees.setValue(metadata['n_trees'])
            self.xgb_trees.setValue(metadata['n_trees'])
        self.ignore_ui_changes = False
        self.check_settings_match_model()

    def _update_apply_button_status(self):
        has_valid = (self.model is not None)
        if has_valid:
            model_name = self.model_combo.currentText()
            expected = self._get_expected_type_name(model_name)
            if expected and type(self.model).__name__ != expected:
                has_valid = False
        if has_valid:
            self.apply_button.setStyleSheet("background-color: green; color: white;")
            self.apply_button.setToolTip("Модель готова к применению")
        else:
            self.apply_button.setStyleSheet("background-color: #b22222; color: black;")
            self.apply_button.setToolTip("Модель не загружена или не соответствует выбранному типу")

    # ----------------------------------------------------------------------
    #  Загрузка / перезагрузка изображений
    # ----------------------------------------------------------------------
    def _load_images(self, source):
        segmentation_input.load_eager(self, source, max_side=640)

    def reload_current_images(self):
        segmentation_input.reload_images(self, self._load_images)

    def _reset_for_new_image(self):
        self._pending_apply = False
        self.prediction_mask = None
        self.current_cluster_labels = None
        if self.model is None:
            self.cluster_combo.setEnabled(False)
            return
        if self.jobs.busy:
            self._pending_apply = True
            return
        model_name = self.model_combo.currentText()
        if model_name in ("K-Means Clustering", "MeanShift"):
            self._prepare_cluster_selection_for_current_image()
        else:
            self.apply_model()

    # ----------------------------------------------------------------------
    #  Логирование и общие утилиты
    # ----------------------------------------------------------------------
    def log(self, message):
        segmentation_actions.log(self, message)

    def on_global_settings_changed(self, new_settings=None):
        segmentation_results.refresh_styles(self)

    def schedule_update(self):
        segmentation_controller.schedule(self)

    # ----------------------------------------------------------------------
    #  Пресеты
    # ----------------------------------------------------------------------
    def update_preset_combo(self):
        segmentation_actions.update_preset_combo(self)

    def on_preset_activated(self, index):
        segmentation_actions.apply_preset(self, index)

    def save_as_preset(self):
        segmentation_actions.save_as_preset(self)

    def delete_current_preset(self):
        segmentation_actions.delete_current_preset(self)

    # ----------------------------------------------------------------------
    #  Настройки модели
    # ----------------------------------------------------------------------
    def _connect_settings_widgets(self):
        widgets = [
            self.normalize_features, self.color_feature, self.texture_feature, self.spatial_feature,
            self.use_superpixels, self.superpixel_size,
            self.kmeans_clusters, self.svm_kernel_combo, self.svm_c,
            self.rf_trees, self.xgb_trees, self.model_combo
        ]
        for w in widgets:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(self.on_any_model_setting_changed)
            elif isinstance(w, QSpinBox):
                w.valueChanged.connect(self.on_any_model_setting_changed)
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.on_any_model_setting_changed)

    def on_any_model_setting_changed(self, *args):
        if not self.ignore_ui_changes:
            self.check_settings_match_model()
            segmentation_controller.invalidate_prediction(self)
            self.schedule_update()

    def on_model_changed(self, idx):
        if not self.ignore_ui_changes and not getattr(self, '_loading_model', False):
            segmentation_controller.invalidate_prediction(self)
        model_name = self.model_combo.currentText()
        self.kmeans_container.setVisible(model_name == "K-Means Clustering")
        self.svm_container.setVisible(model_name == "SVM (pixel-wise)")
        self.rf_container.setVisible(model_name == "Random Forest")
        self.xgb_container.setVisible(model_name == "XGBoost")
        self.update_features_availability()

        is_clustering = model_name in ("K-Means Clustering", "MeanShift")
        self.cluster_group.setVisible(is_clustering)
        self.find_best_k_btn.setVisible(model_name == "K-Means Clustering")

        if self.model is not None:
            current_type = type(self.model).__name__
            expected = self._get_expected_type_name(model_name)
            if expected and current_type != expected:
                self.model = None
                self.model_metadata = None
                self.prediction_mask = None
                self.current_cluster_labels = None
                self.cluster_means = None
                self.cluster_combo.clear()
                self.cluster_combo.setEnabled(False)
                self._last_selected_cluster_label = None
                self.log(f"Модель сброшена, так как выбран {model_name}.")
                self.schedule_update()

        if is_clustering:
            self.cluster_combo.setEnabled(self.model is not None)
        else:
            self.cluster_combo.setEnabled(False)

        self._update_model_path_display()
        self._auto_load_model_for_current_type()
        self.check_settings_match_model()

    def update_features_availability(self):
        self.texture_feature.setEnabled(True)
        self.spatial_feature.setEnabled(True)
        self.color_feature.setEnabled(True)

    def on_superpixel_toggled(self, state):
        self.superpixel_size.setEnabled(state == Qt.Checked)

    # ----------------------------------------------------------------------
    #  Гистограмма
    # ----------------------------------------------------------------------
    def update_histogram(self, gray_img):
        segmentation_histogram.update_histogram(self, gray_img)

    def update_current_histogram(self):
        segmentation_histogram.update_current(self)

    # ----------------------------------------------------------------------
    #  Признаки
    # ----------------------------------------------------------------------
    def get_features(self, image, use_superpixels=False, superpixel_size=50):
        use_intensity = self.color_feature.isChecked()
        use_texture = self.texture_feature.isChecked()
        use_spatial = self.spatial_feature.isChecked()
        if use_superpixels:
            segments = create_superpixels(image, superpixel_size)
            X = extract_superpixel_features(image, segments, use_intensity, use_texture, use_spatial)
            return X, segments
        else:
            X = extract_pixel_features(image, use_intensity, use_texture, use_spatial)
            return X, None

    # ----------------------------------------------------------------------
    #  Подготовка кластеров (K‑Means / MeanShift)
    # ----------------------------------------------------------------------
    def _prepare_cluster_selection_for_current_image(self):
        segmentation_computation.predict_ml(self, cluster=True)

    def _apply_kmeans_cluster(self, idx):
        if self.current_cluster_labels is None:
            return
        selected_label = self.cluster_combo.itemData(idx)
        self.prediction_mask = (self.current_cluster_labels == selected_label).astype(np.uint8) * 255
        self.schedule_update()

    def on_cluster_changed(self, idx):
        if idx >= 0 and self.model is not None:
            # Запоминаем выбранный label кластера для сохранения между снимками
            label = self.cluster_combo.itemData(idx)
            if label is not None:
                self._last_selected_cluster_label = int(label)
            self._apply_kmeans_cluster(idx)
    # ----------------------------------------------------------------------
    #  Обучение модели
    # ----------------------------------------------------------------------
    def train_model(self):
        segmentation_computation.train_ml(self)

    # ----------------------------------------------------------------------
    #  Поиск оптимального K
    # ----------------------------------------------------------------------
    def find_best_k(self):
        segmentation_computation.find_k(self)

    # ----------------------------------------------------------------------
    #  Сохранение / загрузка модели
    # ----------------------------------------------------------------------
    def save_model(self):
        if self.model is None:
            QMessageBox.warning(self, "Нет модели", "Сначала обучите модель.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить модель", "", "Model files (*.pkl *.joblib)")
        if not file_path:
            return

        data_to_save = {'model': self.model, 'metadata': self.model_metadata}
        try:
            joblib.dump(data_to_save, file_path)
            model_name = self.model_combo.currentText()
            self.model_paths[model_name] = file_path
            self._update_model_path_display()
            self._update_apply_button_status()
            self.log(f"Модель сохранена в {file_path}")
            QMessageBox.information(self, "Сохранение", "Модель успешно сохранена.")
        except Exception as e:
            self.log(f"Ошибка сохранения модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить модель:\n{e}")

    def load_model(self):
        segmentation_computation.load_ml(self)

    # ----------------------------------------------------------------------
    #  Проверка соответствия настроек и кнопка сброса
    # ----------------------------------------------------------------------
    def _get_current_settings_dict(self):
        model_name = self.model_combo.currentText()
        settings_dict = {
            'model_type': model_name,
            'use_intensity': self.color_feature.isChecked(),
            'use_texture': self.texture_feature.isChecked(),
            'use_spatial': self.spatial_feature.isChecked(),
            'use_superpixels': self.use_superpixels.isChecked(),
            'superpixel_size': self.superpixel_size.value(),
            'normalize_features': self.normalize_features.isChecked(),
        }
        if model_name == "K-Means Clustering":
            settings_dict['n_clusters'] = self.kmeans_clusters.value()
        elif model_name == "SVM (pixel-wise)":
            settings_dict['kernel'] = self.svm_kernel_combo.currentText()
            settings_dict['C'] = self.svm_c.value()
        elif model_name in ("Random Forest", "XGBoost"):
            settings_dict['n_trees'] = (self.rf_trees.value() if model_name == "Random Forest"
                                        else self.xgb_trees.value())
        return settings_dict

    def check_settings_match_model(self):
        if self.model_metadata is None:
            self.model_mismatch_label.setText("⚙️ Модель не обучена")
            self.model_mismatch_label.setStyleSheet("color: gray;")
            self.reset_to_model_btn.setEnabled(False)
            return False

        current = self._get_current_settings_dict()
        match = all(current.get(k) == v for k, v in self.model_metadata.items() if k in current)

        if match:
            self.model_mismatch_label.setText("✅ Настройки соответствуют модели")
            self.model_mismatch_label.setStyleSheet("color: green;")
            self.reset_to_model_btn.setEnabled(False)
        else:
            self.model_mismatch_label.setText("⚠️ Настройки отличаются от модели")
            self.model_mismatch_label.setStyleSheet("color: orange; font-weight: bold;")
            self.reset_to_model_btn.setEnabled(True)
        return match

    def reset_to_model_settings(self):
        if self.model_metadata is None:
            return
        self._apply_model_metadata(self.model_metadata)
        self._update_apply_button_status()
        self.log("Настройки сброшены к параметрам обученной модели.")

    # ----------------------------------------------------------------------
    #  Применение модели
    # ----------------------------------------------------------------------
    def apply_model(self):
        segmentation_computation.predict_ml(self)

    # ----------------------------------------------------------------------
    #  Отображение текущего изображения
    # ----------------------------------------------------------------------

    def _mask_signature(self, frame):
        return frame.key, id(self.prediction_mask)

    def _compute_mask(self, frame):
        if getattr(self, '_pending_apply', False) and self.model is not None and not self.jobs.busy:
            self._reset_for_new_image()
        return self.prediction_mask, {'ml_view': self.prediction_mask}

    def display_current_image(self):
        segmentation_presentation.display(self)

    # ----------------------------------------------------------------------
    #  Список объектов и аннотации
    # ----------------------------------------------------------------------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    def update_coordinates_display(self):
        segmentation_results.update_coordinates(self)

    # ---------- Сохранение аннотаций (включая пустой файл) ----------
    def save_current_annotations(self):
        segmentation_export.save_selected(self)

    # ----------------------------------------------------------------------
    #  Навигация и UI
    # ----------------------------------------------------------------------
    def update_navigation_state(self):
        segmentation_navigation.update_state(self)

    def prev_image(self):
        segmentation_navigation.step(self, -1, reset=self._reset_for_new_image)

    def next_image(self):
        segmentation_navigation.step(self, 1, reset=self._reset_for_new_image)

    def goto_image(self, page_num):
        segmentation_navigation.navigate(self, page_num-1, reset=self._reset_for_new_image)

    def load_images_from_dialog(self):
        segmentation_input.choose_images(self, self._load_images)

    def load_folder(self):
        segmentation_input.choose_folder(self, self._load_images)

    def _update_loaded_data(self, paths, imgs, grays, anns):
        self.image_paths = paths
        self.display_images = imgs
        self.gray_images = grays
        self.annotations = anns
        self.current_index = 0
        self._reset_for_new_image()
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))
        self.update_navigation_state()

    def on_resize_mode_changed(self, enabled):
        segmentation_controller.resize_changed(self, enabled)

    def clear_images(self):
        segmentation_input.clear_session(self, empty_fields=('display_images','gray_images','annotations'), none_fields=('prediction_mask','current_cluster_labels'))

    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    # ----------------------------------------------------------------------
    #  Отслеживание слайдеров и переключателей
    # ----------------------------------------------------------------------


    def on_draw_mode_changed(self, idx):
        segmentation_presentation.geometry_changed(self)

    def on_hull_changed(self, state):
        segmentation_presentation.geometry_changed(self)

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    # ----------------------------------------------------------------------
    #  Отрисовка объектов
    # ----------------------------------------------------------------------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentData(), self.hull_checkbox.isChecked())

    # ----------------------------------------------------------------------
    #  Лог и гистограмма
    # ----------------------------------------------------------------------
    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)


    def closeEvent(self, event):
        segmentation_controller.close(self)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TraditionalMLWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))
