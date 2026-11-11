
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

import cv2
import joblib
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSpinBox,
)
from sklearn.preprocessing import StandardScaler

from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import TraditionalMLPresetManager
from ui.settings_service import settings
from ui_tabs.traditional_ml_methods_ui import setup_traditional_ml_ui
from utils_new.segmentation.postprocessing import (
    apply_relative_morphology as apply_morphology,
)
from utils_new.segmentation.postprocessing import extract_objects
from utils_ops.traditional_ml_ops import (
    compute_elbow_kmeans,
    create_superpixels,
    extract_pixel_features,
    extract_superpixel_features,
    find_optimal_k_elbow,
    predict_superpixel,
    train_decision_tree,
    train_kmeans,
    train_meanshift,
    train_random_forest,
    train_svm,
    train_xgboost,
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
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Навигация
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

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
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.close_kernel_slider.valueChanged.connect(self.on_close_kernel_changed)
        self.open_kernel_slider.valueChanged.connect(self.on_open_kernel_changed)
        self.kernel_shape_combo.currentIndexChanged.connect(self.schedule_update)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        self.save_button.clicked.connect(self.save_current_annotations)
        settings.settings_changed.connect(self.on_global_settings_changed)

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
        model_name = self.model_combo.currentText()
        path = self.model_paths.get(model_name, "")
        if path and os.path.exists(path):
            self.log(f"Автозагрузка модели для {model_name} из {path}")
            try:
                data = joblib.load(path)
                if isinstance(data, dict) and 'model' in data:
                    model = data['model']
                    metadata = data.get('metadata', {})
                else:
                    model = data
                    metadata = None
                # Проверяем соответствие типа
                actual_type = self._get_model_type_from_class(model)
                if actual_type != model_name:
                    self.log(f"Предупреждение: модель типа {actual_type} загружена, но выбран {model_name}. Будет выполнено переключение.")
                    idx = self.model_combo.findText(actual_type)
                    if idx >= 0:
                        self.model_combo.blockSignals(True)
                        self.model_combo.setCurrentIndex(idx)
                        self.model_combo.blockSignals(False)
                    model_name = actual_type
                self.model = model
                self.model_metadata = metadata
                if metadata:
                    self._apply_model_metadata(metadata)
                self.log(f"Модель {model_name} успешно загружена")
                self._update_apply_button_status()
                if self.display_images:
                    if model_name in ("K-Means Clustering", "MeanShift"):
                        self._prepare_cluster_selection_for_current_image()
                    else:
                        self.apply_model()
                # Убеждаемся, что путь в словаре соответствует
                self.model_paths[model_name] = path
                self._update_model_path_display()
            except Exception as e:
                self.log(f"Ошибка автозагрузки модели {path}: {e}")
                self.model_paths[model_name] = ""
                self.model = None
                self.model_metadata = None
                self._update_model_path_display()
                self._update_apply_button_status()
        else:
            # Проверяем, не загружена ли модель другого типа (несовместимость)
            if self.model is not None:
                expected = self._get_expected_type_name(model_name)
                if expected and type(self.model).__name__ != expected:
                    self.model = None
                    self.model_metadata = None
                    self.cluster_combo.setEnabled(False)
                    self._update_apply_button_status()

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
        self.prediction_mask = None
        self.current_cluster_labels = None
        if self.model is None:
            self.cluster_combo.setEnabled(False)
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
        self.update_annotated_view()

    def schedule_update(self):
        self.update_timer.start(50)

    # ----------------------------------------------------------------------
    #  Пресеты
    # ----------------------------------------------------------------------
    def update_preset_combo(self):
        segmentation_actions.update_preset_combo(self)

    def on_preset_activated(self, index):
        name = self.preset_combo.itemText(index)
        if name:
            self.preset_manager.apply_preset(name)
            self.log(f"Применён пресет: {name}")
            self.schedule_update()

    def save_as_preset(self):
        segmentation_actions.save_as_preset(self)

    def delete_current_preset(self):
        segmentation_actions.delete_current_preset(self)

    # ----------------------------------------------------------------------
    #  Настройки модели
    # ----------------------------------------------------------------------
    def _connect_settings_widgets(self):
        widgets = [
            self.color_feature, self.texture_feature, self.spatial_feature,
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
        if self.ignore_ui_changes:
            return
        self.check_settings_match_model()

    def on_model_changed(self, idx):
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

        if self.model is None or not self.display_images:
            return
        model_name = self.model_combo.currentText()
        if model_name not in ("K-Means Clustering", "MeanShift"):
            return
        if not hasattr(self.model, "cluster_centers_"):
            return

        img = self.display_images[self.current_index]
        gray = self.gray_images[self.current_index]
        use_superpixels = self.use_superpixels.isChecked()
        superpixel_size = self.superpixel_size.value()

        X, segments = self.get_features(img, use_superpixels, superpixel_size)

        # Применяем нормализацию, если модель обучена с ней
        if self.model_metadata and 'scaler' in self.model_metadata:
            X = self.model_metadata['scaler'].transform(X)
        labels = self.model.predict(X)

        if use_superpixels:
            mask_labels = np.zeros_like(segments, dtype=np.int32)
            for seg_id, label in enumerate(labels):
                mask_labels[segments == seg_id] = label
            self.current_cluster_labels = mask_labels
        else:
            mask_labels = labels.reshape(gray.shape)
            self.current_cluster_labels = mask_labels

        unique_labels = np.unique(mask_labels)
        means = []
        for lbl in unique_labels:
            pixels = gray[mask_labels == lbl]
            means.append(np.mean(pixels) if len(pixels) > 0 else 0)

        self.cluster_means = means
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        for lbl, mean_val in zip(unique_labels, means):
            self.cluster_combo.addItem(f"Cluster {lbl} (mean intensity = {mean_val:.1f})", userData=int(lbl))
        self.cluster_combo.setEnabled(True)

        # Пытаемся восстановить ранее выбранный label кластера.
        # Если его нет на текущем снимке (например, кластер исчез после
        # переобучения или на этом снимке такой метки не встретилось) —
        # откатываемся к кластеру с минимальной средней интенсивностью.
        unique_labels_list = [int(x) for x in unique_labels]
        if (self._last_selected_cluster_label is not None
                and self._last_selected_cluster_label in unique_labels_list):
            chosen_idx = unique_labels_list.index(self._last_selected_cluster_label)
        else:
            chosen_idx = int(np.argmin(means))
            self._last_selected_cluster_label = unique_labels_list[chosen_idx]

        self.cluster_combo.setCurrentIndex(chosen_idx)
        self.cluster_combo.blockSignals(False)

        selected_label = unique_labels[chosen_idx]
        self.prediction_mask = (mask_labels == selected_label).astype(np.uint8) * 255

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
        if not self.display_images:
            QMessageBox.warning(self, "Нет изображений", "Загрузите хотя бы одно изображение для обучения.")
            return

        model_name = self.model_combo.currentText()
        use_superpixels = self.use_superpixels.isChecked()
        superpixel_size = self.superpixel_size.value()

        self.train_button.setEnabled(False)
        original_text = self.train_button.text()
        self.train_button.setText("Обучение...")
        self.train_button.setStyleSheet("background-color: green; color: white;")
        QApplication.processEvents()

        try:
            all_X, all_y = [], []
            for idx, img in enumerate(self.display_images):
                X, segments = self.get_features(img, use_superpixels, superpixel_size)
                gray = self.gray_images[idx]
                _, pseudo_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                if use_superpixels:
                    y = np.zeros(len(np.unique(segments)), dtype=np.int32)
                    for seg_id in np.unique(segments):
                        mask = (segments == seg_id)
                        y[seg_id] = 1 if np.mean(pseudo_mask[mask]) > 127 else 0
                else:
                    y = (pseudo_mask.ravel() > 127).astype(np.int32)
                all_X.append(X)
                all_y.append(y)

            X_train = np.vstack(all_X)
            y_train = np.hstack(all_y)
            n_samples = X_train.shape[0]
            if n_samples > 500000:
                reply = QMessageBox.question(
                    self, "Большой объём данных",
                    f"Обучение на {n_samples} пикселях может занять очень много времени и памяти.\n"
                    "Рекомендуется использовать суперпиксели или уменьшить количество изображений.\n"
                    "Продолжить?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            if self.normalize_features.isChecked():
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                if self.model_metadata is None:
                    self.model_metadata = {}
                self.model_metadata['scaler'] = scaler
            else:
                if self.model_metadata and 'scaler' in self.model_metadata:
                    del self.model_metadata['scaler']

            if model_name == "K-Means Clustering":
                n_clusters = self.kmeans_clusters.value()
                self.model = train_kmeans(X_train, n_clusters)
                self.log(f"K-Means (K={n_clusters}) обучена на {X_train.shape[0]} пикселях.")
                self._last_selected_cluster_label = None   # новый набор кластеров
                self._prepare_cluster_selection_for_current_image()
                if self.model_metadata is None:
                    self.model_metadata = {}
                self.model_metadata.update(self._get_current_settings_dict())
            elif model_name == "MeanShift":
                if not use_superpixels and X_train.shape[0] > 50000:
                    reply = QMessageBox.question(
                        self, "Большой объём данных",
                        "MeanShift на полном изображении может выполняться очень долго.\n"
                        "Рекомендуется включить суперпиксели.\nПродолжить?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                self.model = train_meanshift(X_train)
                self.log(f"MeanShift обучен на {X_train.shape[0]} пикселях.")
                self._last_selected_cluster_label = None   # новый набор кластеров
                self._prepare_cluster_selection_for_current_image()
                if self.model_metadata is None:
                    self.model_metadata = {}
                self.model_metadata.update(self._get_current_settings_dict())
            else:
                if model_name == "SVM (pixel-wise)":
                    kernel = self.svm_kernel_combo.currentText()
                    C = self.svm_c.value()
                    self.model = train_svm(X_train, y_train, kernel, C)
                elif model_name == "Random Forest":
                    n_trees = self.rf_trees.value()
                    self.model = train_random_forest(X_train, y_train, n_trees)
                elif model_name == "XGBoost":
                    n_trees = self.xgb_trees.value()
                    self.model = train_xgboost(X_train, y_train, n_trees)
                elif model_name == "Decision Tree":
                    self.model = train_decision_tree(X_train, y_train)
                else:
                    self.log(f"Модель {model_name} не поддерживается")
                    return
                self.log(f"{model_name} обучена на {X_train.shape[0]} пикселях.")
                self.cluster_combo.setEnabled(False)
                if self.model_metadata is None:
                    self.model_metadata = {}
                self.model_metadata.update(self._get_current_settings_dict())
                self.apply_model()

            # Обучение завершено – модель не сохранена, путь сбрасываем
            self.model_paths[model_name] = ""
            self._update_model_path_display()
            self._update_apply_button_status()
            self.check_settings_match_model()
            QMessageBox.information(self, "Обучение завершено", f"Модель {model_name} успешно обучена.")
            self.schedule_update()
        except Exception as e:
            self.log(f"Ошибка при обучении: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось обучить модель:\n{e}")
        finally:
            self.train_button.setEnabled(True)
            self.train_button.setText(original_text)
            self.train_button.setStyleSheet("")
            QApplication.processEvents()

    # ----------------------------------------------------------------------
    #  Поиск оптимального K
    # ----------------------------------------------------------------------
    def find_best_k(self):
        if not self.display_images:
            QMessageBox.warning(self, "Нет изображений", "Загрузите изображения для оценки.")
            return
        if self.model_combo.currentText() != "K-Means Clustering":
            QMessageBox.warning(self, "Не K-Means", "Метод применим только для K-Means.")
            return

        self.find_best_k_btn.setEnabled(False)
        original_text = self.find_best_k_btn.text()
        self.find_best_k_btn.setText("Вычисление локтя...")
        self.find_best_k_btn.setStyleSheet("background-color: orange; color: black;")
        QApplication.processEvents()

        try:
            use_superpixels = self.use_superpixels.isChecked()
            superpixel_size = self.superpixel_size.value()
            all_X = []
            for img in self.display_images:
                X, _ = self.get_features(img, use_superpixels, superpixel_size)
                all_X.append(X)
            X_total = np.vstack(all_X)
            inertias = compute_elbow_kmeans(X_total, max_k=10)
            optimal_k = find_optimal_k_elbow(inertias)
            self.log(f"Оптимальное K по методу локтя: {optimal_k}")
            self.kmeans_clusters.setValue(optimal_k)
            self.train_model()
        except Exception as e:
            self.log(f"Ошибка поиска K: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось найти оптимальное K:\n{e}")
        finally:
            self.find_best_k_btn.setEnabled(True)
            self.find_best_k_btn.setText(original_text)
            self.find_best_k_btn.setStyleSheet("")
            QApplication.processEvents()

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Загрузить модель", "", "Model files (*.pkl *.joblib)")
        if not file_path:
            return
        try:
            data = joblib.load(file_path)
            if isinstance(data, dict) and 'model' in data:
                model = data['model']
                metadata = data.get('metadata', {})
            else:
                model = data
                metadata = None

            actual_type = self._get_model_type_from_class(model)
            if actual_type is None:
                self.log("Не удалось определить тип загруженной модели.")
                QMessageBox.warning(self, "Неизвестная модель", "Не удалось определить тип модели.")
                return

            current_type = self.model_combo.currentText()
            if actual_type != current_type:
                self.log(f"Загружена модель типа {actual_type}, переключаем интерфейс.")
                idx = self.model_combo.findText(actual_type)
                if idx >= 0:
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(idx)
                    self.model_combo.blockSignals(False)
                else:
                    self.log(f"Тип {actual_type} отсутствует в списке.")
                    QMessageBox.warning(self, "Неизвестная модель", f"Тип модели {actual_type} не поддерживается.")
                    return

            self.model = model
            self.model_metadata = metadata
            if metadata:
                self._apply_model_metadata(metadata)

            self.model_paths[actual_type] = file_path
            self._update_model_path_display()
            self._update_apply_button_status()

            self.log(f"Модель загружена из {file_path}")
            QMessageBox.information(self, "Загрузка", "Модель успешно загружена.")

            if self.display_images:
                if actual_type in ("K-Means Clustering", "MeanShift"):
                    self._last_selected_cluster_label = None   # новая модель — метки другие
                    self._prepare_cluster_selection_for_current_image()
                else:
                    self.apply_model()
            else:
                self.cluster_combo.setEnabled(False)
            self.check_settings_match_model()
        except Exception as e:
            self.log(f"Ошибка загрузки модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")

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
        if not self.display_images:
            QMessageBox.warning(self, "Нет изображения", "Загрузите изображение.")
            return
        if self.model is None:
            QMessageBox.warning(self, "Нет модели", "Сначала обучите или загрузите модель.")
            return

        model_name = self.model_combo.currentText()
        use_superpixels = self.use_superpixels.isChecked()
        superpixel_size = self.superpixel_size.value()
        img = self.display_images[self.current_index]
        gray = self.gray_images[self.current_index]

        self.apply_button.setEnabled(False)
        original_text = self.apply_button.text()
        self.apply_button.setText("Применение...")
        self.apply_button.setStyleSheet("background-color: orange; color: black;")
        QApplication.processEvents()



        try:
            X, segments = self.get_features(img, use_superpixels, superpixel_size)
            if self.model_metadata and 'scaler' in self.model_metadata:
                X = self.model_metadata['scaler'].transform(X)
            if model_name in ("K-Means Clustering", "MeanShift"):
                self._prepare_cluster_selection_for_current_image()
            else:
                if use_superpixels:
                    self.prediction_mask = predict_superpixel(self.model, segments, X)
                else:
                    labels = self.model.predict(X)
                    self.prediction_mask = labels.reshape(gray.shape).astype(np.uint8) * 255
            self.schedule_update()
            self.log("Применение модели завершено.")
        except Exception as e:
            self.log(f"Ошибка при применении модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить модель:\n{e}")
        finally:
            self.apply_button.setEnabled(True)
            self.apply_button.setText(original_text)
            self.apply_button.setStyleSheet("")
            QApplication.processEvents()

    # ----------------------------------------------------------------------
    #  Отображение текущего изображения
    # ----------------------------------------------------------------------
    def display_current_image(self):
        if not self.display_images:
            self.update_current_histogram()
            self.original_view.set_pixmap(numpy_to_qpixmap(None))
            self.ml_view.set_pixmap(numpy_to_qpixmap(None))
            self.morph_view.set_pixmap(numpy_to_qpixmap(None))
            self.annotated_view.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            self.annotations = []
            return

        self.info_label.setText(f"Image {self.current_index+1} of {len(self.display_images)}")
        current_file = os.path.basename(self.image_paths[self.current_index])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"ml_original_{base_name}")
        self.ml_view.set_suggested_save_name(f"ml_mask_{base_name}")
        self.morph_view.set_suggested_save_name(f"ml_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"ml_annotated_{base_name}")

        original = self.display_images[self.current_index]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        gray = self.gray_images[self.current_index]
        segmentation_results.prepare_frame(self, gray)

        self.update_histogram(gray)
        self.original_view.set_pixmap(numpy_to_qpixmap(original))

        mask = self.prediction_mask if self.prediction_mask is not None else np.zeros(gray.shape, dtype=np.uint8)
        self.ml_view.set_pixmap(numpy_to_qpixmap(mask))

        close_factor = self.close_kernel_slider.value() / 100.0
        open_factor = self.open_kernel_slider.value() / 100.0
        kernel_shape = self.kernel_shape_combo.currentText()
        processed = apply_morphology(mask, close_factor, open_factor, kernel_shape, gray.shape)
        if self.invert_checkbox.isChecked():
            processed = cv2.bitwise_not(processed)
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        if len(original.shape) == 2:
            result = original.copy()
            result[processed == 0] = 255
        else:
            result = original.copy()
            result[processed == 0] = [255, 255, 255]

        display_img = result if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.current_base_image = display_img.copy()
        _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)
        self.update_object_list()

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
        if self.display_images:
            reply = QMessageBox.question(
                self, "Resize Mode Changed",
                "Resize mode changed. To apply, you need to reload images.\n"
                "Do you want to reload images now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.reload_current_images()
            else:
                self.clear_images()
                self.log("Resize mode changed, images cleared. Please load images again.")

    def clear_images(self):
        segmentation_input.clear_session(self, empty_fields=('display_images','gray_images','annotations'), none_fields=('prediction_mask','current_cluster_labels'))

    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    # ----------------------------------------------------------------------
    #  Отслеживание слайдеров и переключателей
    # ----------------------------------------------------------------------
    def on_close_kernel_changed(self, value):
        self.close_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_open_kernel_changed(self, value):
        self.open_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        self.hull_checkbox.setVisible(mode in ("Segmentation (Polygon)", "OBB (Oriented Box)"))
        if self.display_images:
            self.schedule_update()

    def on_hull_changed(self, state):
        if self.display_images:
            self.schedule_update()

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    # ----------------------------------------------------------------------
    #  Отрисовка объектов
    # ----------------------------------------------------------------------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentText(), self.hull_checkbox.isChecked())

    # ----------------------------------------------------------------------
    #  Лог и гистограмма
    # ----------------------------------------------------------------------
    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TraditionalMLWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))
