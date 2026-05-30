# traditional_ml_methods.py
from import_libs_internal import *
from import_libs_methods_ui import setup_traditional_ml_ui


class TraditionalMLWindow(QMainWindow):
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
    def reload_current_images(self):
        if not self.image_paths:
            self.log("Нет загруженных изображений для перезагрузки.")
            return
        self.log("Перезагрузка изображений с новыми настройками ресайза...")
        resize_enabled = self.nav_widget.is_resize_enabled()
        paths, imgs, grays, anns = load_images_universal(
            source=self.image_paths,
            require_annotations=False,
            resize_enabled=resize_enabled,
            max_side=640,
            parent=self
        )
        if not paths:
            self.log("Ошибка: ни одно изображение не загружено при перезагрузке.")
            return
        self.image_paths = paths
        self.display_images = imgs
        self.gray_images = grays
        self.annotations = anns
        self.current_index = 0
        self._reset_for_new_image()
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))
        self.update_navigation_state()
        self.log(f"Перезагружено {len(paths)} изображений.")

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
        if hasattr(self, 'log_text'):
            self.log_text.append(message)
        elif hasattr(self, 'log_widget'):
            self.log_widget.log(message)
        print(message)

    def on_global_settings_changed(self, new_settings=None):
        self.update_annotated_view()

    def schedule_update(self):
        self.update_timer.start(50)

    # ----------------------------------------------------------------------
    #  Пресеты
    # ----------------------------------------------------------------------
    def update_preset_combo(self):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_manager.load_presets()
        for name in self.preset_manager.get_preset_names():
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText("default")
        self.preset_combo.blockSignals(False)

    def on_preset_activated(self, index):
        name = self.preset_combo.itemText(index)
        if name:
            self.preset_manager.apply_preset(name)
            self.log(f"Применён пресет: {name}")
            self.schedule_update()

    def save_as_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите имя пресета:")
        if ok and name:
            if name in self.preset_manager.get_preset_names():
                reply = QMessageBox.question(
                    self, "Перезаписать пресет",
                    f"Пресет '{name}' уже существует. Перезаписать?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                self.preset_manager.presets[name] = self.preset_manager.get_current_settings()
                self.preset_manager.save_presets()
            else:
                self.preset_manager.add_preset(name)
            self.update_preset_combo()
            self.preset_combo.setCurrentText(name)
            self.log(f"Сохранён пресет: {name}")

    def delete_current_preset(self):
        current = self.preset_combo.currentText()
        reply = QMessageBox.question(
            self, "Удалить пресет",
            f"Удалить пресет '{current}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.preset_manager.delete_preset(current):
                self.update_preset_combo()
                self.log(f"Удалён пресет: {current}")

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
        self.hist_ax.clear()
        self.hist_ax.hist(gray_img.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
        self.hist_ax.set_title("Grayscale Histogram")
        self.hist_ax.set_xlabel("Pixel intensity")
        self.hist_ax.set_ylabel("Frequency")
        if self.cluster_means is not None and len(self.cluster_means) > 0:
            for mean_val in self.cluster_means:
                self.hist_ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1)
        self.hist_canvas.draw()

    def update_current_histogram(self):
        if self.display_images:
            gray = self.gray_images[self.current_index]
            self.update_histogram(gray)

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
        default_idx = np.argmin(means)
        self.cluster_combo.setCurrentIndex(default_idx)
        self.cluster_combo.blockSignals(False)

        selected_label = unique_labels[default_idx]
        self.prediction_mask = (mask_labels == selected_label).astype(np.uint8) * 255

    def _apply_kmeans_cluster(self, idx):
        if self.current_cluster_labels is None:
            return
        selected_label = self.cluster_combo.itemData(idx)
        self.prediction_mask = (self.current_cluster_labels == selected_label).astype(np.uint8) * 255
        self.schedule_update()

    def on_cluster_changed(self, idx):
        if idx >= 0 and self.model is not None:
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
        self.object_list.blockSignals(True)
        self.object_list.clear()
        self.current_selected_indices = []
        img = self.display_images[self.current_index]
        img_h, img_w = img.shape[:2]
        for i, obj in enumerate(self.current_objects_full):
            desc = format_object_for_list(i, obj, img_w, img_h)
            item = QListWidgetItem(desc)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.object_list.addItem(item)
            self.current_selected_indices.append(i)
        self.object_list.blockSignals(False)
        self.update_annotated_view()

    def update_annotated_view(self):
        if self.current_base_image is None:
            return
        img = self.current_base_image.copy()
        if not self.current_selected_indices:
            self.annotated_view.set_pixmap(numpy_to_qpixmap(img))
            self.update_coordinates_display()
            return
        thickness, font_scale, font_thickness, _ = get_display_params(img.shape)
        color_rect = settings.get_color('annotation')
        color_label = settings.get_color('label_text')
        draw_mode = self.draw_combo.currentText()
        use_hull = self.hull_checkbox.isChecked()
        img_annotated = draw_selected_objects(
            img, self.current_objects_full, self.current_selected_indices,
            draw_mode, use_hull, color_rect, color_label,
            thickness, font_scale, font_thickness
        )
        self.annotated_view.set_pixmap(numpy_to_qpixmap(img_annotated))
        self.update_coordinates_display()

    def update_coordinates_display(self):
        self.coord_text.clear()
        if not self.current_selected_indices:
            self.coord_text.append("No objects selected.")
            return
        img = self.display_images[self.current_index]
        img_h, img_w = img.shape[:2]
        for i, idx in enumerate(self.current_selected_indices, 1):
            obj = self.current_objects_full[idx]
            if isinstance(obj, tuple) and len(obj) > 0 and obj[0] in ('detect', 'obb', 'segment'):
                typ = obj[0]
                if typ == 'detect':
                    _, cls, cx, cy, w, h = obj
                    x = int((cx - w/2) * img_w)
                    y = int((cy - h/2) * img_h)
                    x2 = x + int(w * img_w)
                    y2 = y + int(h * img_h)
                    self.coord_text.append(f"{i}: detect class={cls}, rect=({x},{y},{x2},{y2})")
                elif typ == 'obb':
                    _, cls, points = obj
                    pts_str = ' '.join(f"{p:.3f}" for p in points)
                    self.coord_text.append(f"{i}: obb class={cls}, YOLO-OBB: {cls} {pts_str}")
                elif typ == 'segment':
                    _, cls, points = obj
                    preview = ' '.join(f"{p:.3f}" for p in points[:6]) + (' ...' if len(points) > 6 else '')
                    self.coord_text.append(f"{i}: segment class={cls}, YOLO-seg: {cls} {preview}")
            elif len(obj) == 4:
                self.coord_text.append(f"{i}: x={obj[0]}, y={obj[1]}, w={obj[2]}, h={obj[3]}")
            elif len(obj) == 5:
                self.coord_text.append(f"{i}: center=({obj[0]:.1f},{obj[1]:.1f}), size=({obj[2]:.1f}x{obj[3]:.1f}), angle={obj[4]:.1f}°")
            else:
                self.coord_text.append(f"{i}: {obj}")

    def save_current_annotations(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображения", "Нет загруженных изображений.")
            return
        img_path = self.image_paths[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        success = save_annotations(
            self.current_objects_full,
            txt_path,
            self.display_images[self.current_index].shape[1],
            self.display_images[self.current_index].shape[0]
        )
        if success:
            self.log(f"Сохранено {len(self.current_objects_full)} аннотаций в {txt_path}")
            QMessageBox.information(self, "Сохранение", f"Аннотации сохранены в {txt_path}")
        else:
            self.log(f"Ошибка сохранения {txt_path}")
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить аннотации.")

    # ----------------------------------------------------------------------
    #  Навигация и UI
    # ----------------------------------------------------------------------
    def update_navigation_state(self):
        has_images = len(self.display_images) > 0
        self.nav_widget.set_prev_enabled(has_images and self.current_index > 0)
        self.nav_widget.set_next_enabled(has_images and self.current_index < len(self.display_images) - 1)

    def prev_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index - 1) % len(self.display_images)
        self._reset_for_new_image()
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    def next_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index + 1) % len(self.display_images)
        self._reset_for_new_image()
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    def goto_image(self, page_num):
        if not self.display_images:
            return
        total = len(self.display_images)
        page_num = max(1, min(page_num, total))
        self.current_index = page_num - 1
        self._reset_for_new_image()
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, total)

    def load_images_from_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.log(f"Loading {len(file_paths)} images...")
        paths, imgs, grays, anns = load_images_universal(
            source=file_paths, require_annotations=False,
            resize_enabled=self.nav_widget.is_resize_enabled(),
            max_side=640, parent=self
        )
        if not paths:
            self.log("No images loaded.")
            return
        self._update_loaded_data(paths, imgs, grays, anns)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.log(f"Loading folder: {folder}")
        paths, imgs, grays, anns = load_images_universal(
            source=folder, require_annotations=False,
            resize_enabled=self.nav_widget.is_resize_enabled(),
            max_side=640, parent=self
        )
        if not paths:
            self.log("No images found.")
            return
        self._update_loaded_data(paths, imgs, grays, anns)

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
        self.display_images = []
        self.gray_images = []
        self.image_paths = []
        self.annotations = []
        self.current_index = 0
        self.prediction_mask = None
        self.current_cluster_labels = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        self.original_view.set_pixmap(numpy_to_qpixmap(None))
        self.ml_view.set_pixmap(numpy_to_qpixmap(None))
        self.morph_view.set_pixmap(numpy_to_qpixmap(None))
        self.annotated_view.set_pixmap(numpy_to_qpixmap(None))
        self.info_label.setText("No images")
        self.object_list.clear()
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    def reset_all_zooms(self):
        for view in (self.original_view, self.ml_view, self.morph_view, self.annotated_view):
            view.reset_view()

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
        idx = self.object_list.row(item)
        if item.checkState() == Qt.Checked:
            if idx not in self.current_selected_indices:
                self.current_selected_indices.append(idx)
        else:
            if idx in self.current_selected_indices:
                self.current_selected_indices.remove(idx)
        self.current_selected_indices.sort()
        self.update_annotated_view()

    # ----------------------------------------------------------------------
    #  Отрисовка объектов
    # ----------------------------------------------------------------------
    def draw_objects_on_image(self, img, binary, draw=True):
        mode = self.draw_combo.currentText()
        use_hull = self.hull_checkbox.isChecked()
        img_color = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if mode == "None":
            return img_color, []
        if mode == "Segmentation (Polygon)":
            return segment_contours(img_color if draw else img, binary, use_hull)
        if mode == "Bounding Box (Detect)":
            return segment_projections(img_color if draw else img, binary)
        if mode == "OBB (Oriented Box)":
            return segment_min_area_rect(img_color if draw else img, binary, use_hull)
        return img_color, []

    # ----------------------------------------------------------------------
    #  Лог и гистограмма
    # ----------------------------------------------------------------------
    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def toggle_histogram(self, checked):
        self.hist_container.setVisible(checked)
        self.toggle_hist_btn.setText("Скрыть гистограмму" if checked else "Показать гистограмму")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TraditionalMLWindow()
    window.show()
    sys.exit(app.exec_())