# deep_learning_methods.py
from import_libs_internal import *
from import_libs_methods_ui import setup_deep_learning_methods_ui


class DeepLearningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning Segmentation")
        self.annotations = []
        self.image_paths = []
        self.display_images = []
        self.gray_images = []
        self.current_index = 0

        self.segmentor = None
        self.model_metadata = None
        self.prediction_mask = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        self.ignore_ui_changes = False

        setup_deep_learning_methods_ui(self)

        # Устройство по умолчанию
        auto_device = "Auto (best available)"
        idx = self.device_combo.findText(auto_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.on_device_changed(idx)

        # Пресеты
        self.preset_manager = DeepLearningPresetManager(self, preset_file="presets_deep_learning.json")
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
        self.nav_widget.goto_page.connect(self.goto_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)

        # Кнопки
        self.segment_button.clicked.connect(self.run_segmentation)
        self.save_button.clicked.connect(self.save_current_annotations)

        # Подключение кнопок "Обзор" в виджетах
        self.unet_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("U-Net"))
        self.deeplab_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("DeepLabV3+"))
        self.segformer_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("SegFormer"))
        self.sam_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("SAM"))
        self.yolo_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("YOLO-seg"))
        self.custom_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("Custom ONNX"))

        # UI изменения
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        self._connect_settings_widgets()

        # Морфология
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.close_kernel_slider.valueChanged.connect(self.schedule_update)
        self.open_kernel_slider.valueChanged.connect(self.schedule_update)
        self.kernel_shape_combo.currentIndexChanged.connect(self.schedule_update)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.hull_checkbox.stateChanged.connect(self.schedule_update)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        settings.settings_changed.connect(self.on_global_settings_changed)

        self.update_navigation_state()
        self.on_model_changed(0)


    def _update_segment_button_color(self):
        """Обновляет цвет кнопки сегментации в зависимости от наличия модели."""
        if self.segmentor is not None:
            self.segment_button.setStyleSheet("background-color: green; color: white;")
        else:
            self.segment_button.setStyleSheet("background-color: #b22222; color: black;")

    # ---------- Загрузка модели из виджета ----------
    def load_model_from_widget(self, model_name):
        if model_name == "U-Net":
            file_path = self.unet_settings._model_path.text().strip()
        elif model_name == "DeepLabV3+":
            file_path = self.deeplab_settings._model_path.text().strip()
        elif model_name == "SegFormer":
            file_path = self.segformer_settings._model_path.text().strip()
        elif model_name == "SAM":
            file_path = self.sam_settings._model_path.text().strip()
        elif model_name == "YOLO-seg":
            file_path = self.yolo_settings._model_path.text().strip()
        elif model_name == "Custom ONNX":
            file_path = self.custom_settings._model_path.text().strip()
        else:
            return

        if not file_path:
            QMessageBox.warning(self, "Ошибка", "Не выбран файл модели.")
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", f"Файл не найден:\n{file_path}")
            return

        if model_name in ["SegFormer", "SAM"]:
            QMessageBox.information(self, "Информация",
                                    f"Модель {model_name} пока не реализована.\n"
                                    "Функциональность будет добавлена в следующих версиях.")
            self.segmentor = None
            self.model_metadata = None
            self.prediction_mask = None
            self.log(f"Попытка загрузить модель {model_name} – не реализована.")
            self._update_segment_button_color()
            self.display_current_image()
            return

        try:
            self.segmentor = create_segmentor(model_name, device=self.device)

            if model_name == "U-Net":
                try:
                    self.segmentor.load(weights_path=file_path)
                    if hasattr(self.segmentor, 'metadata') and self.segmentor.metadata:
                        self._update_unet_ui_from_metadata(self.segmentor.metadata)
                    else:
                        self.log("Предупреждение: метаданные модели не загружены, UI не обновлён.")
                except Exception as e:
                    self.log(f"Ошибка загрузки U-Net: {e}")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель U-Net:\n{e}")
                    self.segmentor = None
                    self._update_segment_button_color()
                    self.display_current_image()
                    return

            elif model_name == "DeepLabV3+":
                self.segmentor.load(
                    backbone=self.deeplab_settings._backbone.currentText(),
                    output_stride=int(self.deeplab_settings._output_stride.currentText())
                )
                if hasattr(self.segmentor, 'metadata'):
                    self._update_deeplab_ui_from_metadata(self.segmentor.metadata)

            elif model_name == "YOLO-seg":
                self.segmentor.load(model_path=file_path)

            elif model_name == "Custom ONNX":
                self.segmentor.load(model_path=file_path)

            else:
                self.segmentor.load()

            self.log(f"Модель {model_name} загружена из {file_path}")
            self.prediction_mask = None
            self._update_segment_button_color()  # зелёный
            self.display_current_image()
        except Exception as e:
            self.log(f"Ошибка загрузки модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            self.segmentor = None
            self._update_segment_button_color()  # красный
            self.display_current_image()




    def _update_unet_ui_from_metadata(self, metadata):
        """Заполняет поля U-Net из метаданных (поля уже заблокированы)."""
        if 'encoder' in metadata:
            encoder = metadata['encoder']
            idx = self.unet_settings._encoder.findText(encoder)
            if idx >= 0:
                self.unet_settings._encoder.setCurrentIndex(idx)
        if 'input_size' in metadata:
            input_size = str(metadata['input_size'])
            idx = self.unet_settings._input_size.findText(input_size)
            if idx >= 0:
                self.unet_settings._input_size.setCurrentIndex(idx)

    def _update_deeplab_ui_from_metadata(self, metadata):
        """Заполняет поля DeepLabV3+ из метаданных (поля уже заблокированы)."""
        if 'backbone' in metadata:
            backbone = metadata['backbone']
            idx = self.deeplab_settings._backbone.findText(backbone)
            if idx >= 0:
                self.deeplab_settings._backbone.setCurrentIndex(idx)
        if 'output_stride' in metadata:
            stride = str(metadata['output_stride'])
            idx = self.deeplab_settings._output_stride.findText(stride)
            if idx >= 0:
                self.deeplab_settings._output_stride.setCurrentIndex(idx)

    # ---------- Подключение виджетов для отслеживания изменений ----------
    def _connect_settings_widgets(self):
        widgets = [
            self.model_combo,
            self.device_combo,
            self.yolo_settings._conf,
            self.yolo_settings._iou,
            self.yolo_settings._imgsz,
            self.yolo_settings._save,
            self.custom_settings._model_path,
            self.unet_settings._threshold,  # добавлено
        ]
        for w in widgets:
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.on_any_setting_changed)

    def on_any_setting_changed(self, *args):
        if self.ignore_ui_changes:
            return
        if self.segmentor is not None:
            self.prediction_mask = None
        self.schedule_update()

    # ---------- Логирование ----------
    def log(self, message):
        if hasattr(self, 'log_text'):
            self.log_text.append(message)
        elif hasattr(self, 'log_widget'):
            self.log_widget.log(message)
        print(message)

    # ---------- Пресеты ----------
    def on_global_settings_changed(self, new_settings=None):
        self.update_annotated_view()

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
            self.prediction_mask = None
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

    # ---------- Выбор модели ----------
    def on_model_changed(self, idx):
        model_name = self.model_combo.currentText()
        # Скрываем все виджеты настроек
        for cont in [self.unet_settings, self.deeplab_settings, self.segformer_settings,
                     self.sam_settings, self.yolo_settings, self.custom_settings]:
            cont.setVisible(False)

        if model_name == "U-Net":
            self.unet_settings.setVisible(True)
        elif model_name == "DeepLabV3+":
            self.deeplab_settings.setVisible(True)
        elif model_name == "SegFormer":
            self.segformer_settings.setVisible(True)
        elif model_name == "SAM":
            self.sam_settings.setVisible(True)
        elif model_name == "YOLO-seg":
            self.yolo_settings.setVisible(True)
        elif model_name == "Custom ONNX":
            self.custom_settings.setVisible(True)

        # Сбрасываем текущую модель
        self.segmentor = None
        self.model_metadata = None
        self.prediction_mask = None
        self._update_segment_button_color()

        # Пытаемся загрузить модель, если путь уже указан
        if model_name == "U-Net":
            path = self.unet_settings._model_path.text().strip()
        elif model_name == "DeepLabV3+":
            path = self.deeplab_settings._model_path.text().strip()
        elif model_name == "SegFormer":
            path = self.segformer_settings._model_path.text().strip()
        elif model_name == "SAM":
            path = self.sam_settings._model_path.text().strip()
        elif model_name == "YOLO-seg":
            path = self.yolo_settings._model_path.text().strip()
        elif model_name == "Custom ONNX":
            path = self.custom_settings._model_path.text().strip()
        else:
            path = ""

        if path and os.path.exists(path):
            # Файл существует – пробуем загрузить
            self.load_model_from_widget(model_name)
        elif path and not os.path.exists(path):
            # Файл не существует – очищаем поле пути
            if model_name == "U-Net":
                self.unet_settings._model_path.setText("")
            elif model_name == "DeepLabV3+":
                self.deeplab_settings._model_path.setText("")
            elif model_name == "SegFormer":
                self.segformer_settings._model_path.setText("")
            elif model_name == "SAM":
                self.sam_settings._model_path.setText("")
            elif model_name == "YOLO-seg":
                self.yolo_settings._model_path.setText("")
            elif model_name == "Custom ONNX":
                self.custom_settings._model_path.setText("")
            self.log(f"Путь к модели {model_name} не существует, очищен.")

        self.display_current_image()

    def on_device_changed(self, idx):
        device_text = self.device_combo.currentText()
        if device_text.startswith("CPU"):
            new_device = "cpu"
        elif device_text.startswith("CUDA"):
            new_device = "cuda"
        elif "MPS" in device_text:
            new_device = "mps"
        elif device_text.startswith("Auto"):
            new_device = "auto"
        else:
            new_device = "cpu"

        # Если устройство не изменилось – ничего не делаем
        if hasattr(self, 'device') and self.device == new_device:
            return

        self.device = new_device
        self.log(f"Выбрано устройство: {self.device}")

        # Если модель уже загружена, перезагружаем её с новым устройством
        if self.segmentor is not None:
            model_name = self.model_combo.currentText()
            # Проверяем, есть ли путь к модели (он должен быть, так как модель загружена)
            if model_name == "U-Net":
                path = self.unet_settings._model_path.text().strip()
            elif model_name == "DeepLabV3+":
                path = self.deeplab_settings._model_path.text().strip()
            elif model_name == "SegFormer":
                path = self.segformer_settings._model_path.text().strip()
            elif model_name == "SAM":
                path = self.sam_settings._model_path.text().strip()
            elif model_name == "YOLO-seg":
                path = self.yolo_settings._model_path.text().strip()
            elif model_name == "Custom ONNX":
                path = self.custom_settings._model_path.text().strip()
            else:
                path = ""

            if path and os.path.exists(path):
                # Перезагружаем модель через универсальную функцию
                self.load_model_from_widget(model_name)
            else:
                # Путь пропал – сбрасываем модель
                self.segmentor = None
                self._update_segment_button_color()
                self.log(f"Модель {model_name} не найдена, сброшена.")

    # ---------- Гистограмма ----------
    def update_histogram(self, gray_img):
        self.hist_ax.clear()
        self.hist_ax.hist(gray_img.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
        self.hist_ax.set_title("Grayscale Histogram")
        self.hist_ax.set_xlabel("Pixel intensity")
        self.hist_ax.set_ylabel("Frequency")
        self.hist_canvas.draw()

    def update_current_histogram(self):
        if self.display_images:
            gray = self.gray_images[self.current_index]
            self.update_histogram(gray)

    # ---------- DL операции ----------
    def run_segmentation(self):
        if not self.display_images:
            QMessageBox.warning(self, "Нет изображения", "Загрузите изображение.")
            return
        if self.segmentor is None:
            QMessageBox.warning(self, "Нет модели", "Сначала загрузите модель.")
            return

        model_name = self.model_combo.currentText()
        if model_name in ["SegFormer", "SAM"]:
            QMessageBox.information(self, "Информация",
                                    f"Модель {model_name} не реализована, сегментация невозможна.")
            self.prediction_mask = np.zeros(self.gray_images[self.current_index].shape, dtype=np.uint8)
            return

        self.progress_bar.setValue(0)
        self.segment_button.setEnabled(False)
        QApplication.processEvents()

        try:
            img = self.display_images[self.current_index]
            kwargs = {}
            self.log(f"=== Запуск сегментации (модель: {model_name}) ===")
            self.log(f"Изображение: {img.shape}, dtype={img.dtype}, диапазон [{img.min()}, {img.max()}]")

            if model_name == "YOLO-seg":
                kwargs['conf'] = self.yolo_settings._conf.value()
                kwargs['iou'] = self.yolo_settings._iou.value()
                kwargs['imgsz'] = self.yolo_settings._imgsz.value()
                kwargs['save'] = self.yolo_settings._save.isChecked()
                self.log(f"Параметры YOLO: conf={kwargs['conf']}, iou={kwargs['iou']}, "
                         f"imgsz={kwargs['imgsz']}, save={kwargs['save']}")

            # ---------- Добавляем для U‑Net ----------
            if model_name == "U-Net":
                kwargs['threshold'] = self.unet_settings._threshold.value()
                self.log(f"Порог U‑Net: {kwargs['threshold']}")

            mask = self.segmentor.predict(img, **kwargs)

            self.log(f"Маска после модели: {mask.shape}, dtype={mask.dtype}, "
                     f"мин={mask.min()}, макс={mask.max()}, среднее={mask.mean():.3f}")

            if mask.dtype == np.uint8 and mask.max() <= 1:
                object_pixels = np.sum(mask > 0)
                self.log(f"Количество пикселей объекта: {object_pixels} ({100*object_pixels/mask.size:.2f}%)")
            else:
                object_pixels = np.sum(mask > 127)
                self.log(f"Количество пикселей объекта (порог 128): {object_pixels} ({100*object_pixels/mask.size:.2f}%)")

            self.prediction_mask = mask
            self.progress_bar.setValue(100)

            # Встроенное сохранение YOLO (если save=True) уже обрабатывается внутри predict
            # Кнопка "Save Labels" отвечает за сохранение аннотаций в .txt

        except Exception as e:
            self.log(f"Ошибка сегментации: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить сегментацию:\n{e}")
            self.prediction_mask = np.zeros(self.gray_images[self.current_index].shape, dtype=np.uint8)
        finally:
            self.segment_button.setEnabled(True)

    # ---------- Отображение ----------
    def display_current_image(self):
        if not self.display_images:
            self.update_current_histogram()
            for view in [self.original_view, self.dl_view, self.morph_view, self.annotated_view]:
                view.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            self.annotations = []
            return

        self.info_label.setText(f"Image {self.current_index+1} of {len(self.display_images)}")
        current_file = os.path.basename(self.image_paths[self.current_index])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"dl_original_{base_name}")
        self.dl_view.set_suggested_save_name(f"dl_mask_{base_name}")
        self.morph_view.set_suggested_save_name(f"dl_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"dl_annotated_{base_name}")

        original = self.display_images[self.current_index]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        gray = self.gray_images[self.current_index]

        self.update_histogram(gray)
        self.original_view.set_pixmap(numpy_to_qpixmap(original))

        if self.segmentor is not None and self.prediction_mask is None:
            self.run_segmentation()

        mask = self.prediction_mask if self.prediction_mask is not None else np.zeros(gray.shape, dtype=np.uint8)
        self._apply_morphology_and_draw(mask, gray, original)
        self.update_navigation_state()

    def _apply_morphology_and_draw(self, mask, gray, original):
        close_factor = self.close_kernel_slider.value() / 100.0
        open_factor = self.open_kernel_slider.value() / 100.0
        kernel_shape = self.kernel_shape_combo.currentText()

        self.log(f"=== Постобработка морфологией ===")
        self.log(f"Входная маска (DL): {mask.shape}, dtype={mask.dtype}, "
                 f"мин={mask.min()}, макс={mask.max()}, среднее={mask.mean():.3f}")
        self.log(f"Параметры морфологии: Closing={close_factor}, Opening={open_factor}, "
                 f"Kernel shape={kernel_shape}")

        processed = apply_morphology(mask, close_factor, open_factor, kernel_shape, gray.shape)

        self.log(f"Маска после морфологии: {processed.shape}, dtype={processed.dtype}, "
                 f"мин={processed.min()}, макс={processed.max()}, среднее={processed.mean():.3f}")

        if self.invert_checkbox.isChecked():
            self.log("Инверсия маски применена")
            processed = cv2.bitwise_not(processed)

        self.dl_view.set_pixmap(numpy_to_qpixmap(mask))
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        # Подготовка изображения для аннотаций (фон делаем белым)
        if len(original.shape) == 2:
            result = original.copy()
            result[processed == 0] = 255
        else:
            result = original.copy()
            mask_bool = processed == 0
            result[mask_bool] = [255, 255, 255]

        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()

        draw_mode = self.draw_combo.currentText()
        self.log(f"=== Отрисовка объектов (режим: {draw_mode}) ===")

        # Сохраняем чистое изображение (без аннотаций) и получаем список объектов
        self.current_base_image = display_img.copy()
        _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)

        if self.current_objects_full:
            self.log(f"Обнаружено объектов: {len(self.current_objects_full)}")
            for i, obj in enumerate(self.current_objects_full[:5]):
                if isinstance(obj, tuple) and len(obj) > 0 and obj[0] in ('detect', 'obb', 'segment'):
                    typ = obj[0]
                    if typ == 'detect':
                        _, cls, cx, cy, w, h = obj
                        self.log(f"  Объект {i+1}: detect (class {cls})")
                    elif typ == 'obb':
                        _, cls, pts = obj
                        self.log(f"  Объект {i+1}: OBB (class {cls})")
                    else:
                        _, cls, pts = obj
                        self.log(f"  Объект {i+1}: polygon (class {cls})")
                elif len(obj) == 4:
                    self.log(f"  Объект {i+1}: прямоугольник x={obj[0]}, y={obj[1]}, w={obj[2]}, h={obj[3]}")
                elif len(obj) == 5:
                    self.log(f"  Объект {i+1}: центр=({obj[0]:.1f},{obj[1]:.1f}), размер=({obj[2]:.1f}x{obj[3]:.1f}), angle={obj[4]:.1f}°")
            if len(self.current_objects_full) > 5:
                self.log(f"  ... и ещё {len(self.current_objects_full)-5} объектов")
        else:
            self.log("Объектов не обнаружено")

        self.update_object_list()

    # ---------- Работа со списком объектов ----------
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

    # ---------- Универсальное отображение координат ----------
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

    # ---------- Навигация ----------
    def update_navigation_state(self):
        has_images = len(self.display_images) > 0
        self.nav_widget.set_prev_enabled(has_images and self.current_index > 0)
        self.nav_widget.set_next_enabled(has_images and self.current_index < len(self.display_images) - 1)

    def prev_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index - 1) % len(self.display_images)
        self.prediction_mask = None
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    def next_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index + 1) % len(self.display_images)
        self.prediction_mask = None
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    def goto_image(self, page_num):
        if not self.display_images:
            return
        total = len(self.display_images)
        page_num = max(1, min(page_num, total))
        self.current_index = page_num - 1
        self.prediction_mask = None
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, total)

    def load_images_from_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self._load_images(file_paths)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self._load_images(folder)

    def _load_images(self, source):
        self.log(f"Loading images from {source}...")
        paths, imgs, grays, anns = load_images_universal(
            source=source,
            require_annotations=False,
            resize_enabled=self.nav_widget.is_resize_enabled(),
            max_side=640,
            parent=self
        )
        if not paths:
            self.log("No images loaded.")
            return
        self.image_paths = paths
        self.display_images = imgs
        self.gray_images = grays
        self.annotations = anns
        self.current_index = 0
        self.prediction_mask = None
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    # ---------- Сохранение аннотаций (универсальное) ----------
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

    # ---------- UI события ----------
    def reset_all_zooms(self):
        for view in [self.original_view, self.dl_view, self.morph_view, self.annotated_view]:
            view.reset_view()

    def on_resize_mode_changed(self, enabled):
        if self.display_images:
            reply = QMessageBox.question(
                self, "Resize Mode Changed",
                "Resize mode changed. Reload images?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                if self.image_paths:
                    self._load_images(self.image_paths)
                else:
                    self.log("Nothing to reload.")
            else:
                self.clear_images()

    def clear_images(self):
        self.display_images = []
        self.gray_images = []
        self.image_paths = []
        self.annotations = []
        self.current_index = 0
        self.prediction_mask = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        for view in [self.original_view, self.dl_view, self.morph_view, self.annotated_view]:
            view.set_pixmap(numpy_to_qpixmap(None))
        self.info_label.setText("No images")
        self.object_list.clear()
        self.update_navigation_state()
        self.nav_widget.set_current_index(0, 0)

    def schedule_update(self):
        self.update_timer.start(50)

    def on_close_kernel_changed(self, value):
        self.close_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_open_kernel_changed(self, value):
        self.open_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        if mode in ["Segmentation (Polygon)", "OBB (Oriented Box)"]:
            self.hull_checkbox.setVisible(True)
        else:
            self.hull_checkbox.setVisible(False)
        self.schedule_update()

    def on_hull_changed(self, state):
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

    def draw_objects_on_image(self, img, binary, draw=True):
        mode = self.draw_combo.currentText()
        use_hull = self.hull_checkbox.isChecked()

        if mode == "None":
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
            return img_color, []
        elif mode == "Segmentation (Polygon)":
            return segment_contours(img, binary, use_hull, draw=draw)
        elif mode == "Bounding Box (Detect)":
            return segment_projections(img, binary, draw=draw)
        elif mode == "OBB (Oriented Box)":
            return segment_min_area_rect(img, binary, use_hull, draw=draw)
        else:
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
            return img_color, []

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def toggle_histogram(self, checked):
        self.hist_container.setVisible(checked)
        self.toggle_hist_btn.setText("Скрыть гистограмму" if checked else "Показать гистограмму")


def center_window(window, width=1200, height=800):
    window.resize(width, height)
    frame = window.frameGeometry()
    screen = QApplication.primaryScreen().availableGeometry()
    frame.moveCenter(screen.center())
    window.move(frame.topLeft())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepLearningWindow()
    center_window(window)
    window.show()
    sys.exit(app.exec_())