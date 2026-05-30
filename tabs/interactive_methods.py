# interactive_methods.py
from import_libs_internal import *
from import_libs_methods_ui import setup_interactive_ui

class InteractiveMethodsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Segmentation")
        self.annotations = []
        self.image_paths = []
        self.display_images = []
        self.gray_images = []
        self.current_index = 0

        # Данные для интерактивной сегментации
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.current_tool = None
        self.current_img_w = 0
        self.current_img_h = 0

        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None

        setup_interactive_ui(self)

        # Ссылка на основной вид (SmartGraphicsView)
        self.image_view = self.original_view

        # Настройка коллбэков SmartGraphicsView
        self.image_view.set_callbacks(
            on_rect_drawn=self.on_rect_drawn,
            on_scribble_added=self.on_scribble_added,
            on_display_update=self.update_scribble_display,
            on_reset_tool=self.activate_pan_zoom
        )

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        # Пресеты
        self.preset_manager = InteractivePresetManager(self, preset_file="presets_interactive.json")
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

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

        # Кнопки инструментов
        self.rect_mode_btn.clicked.connect(self.activate_rect_tool)
        self.fg_scribble_btn.clicked.connect(self.activate_fg_tool)
        self.bg_scribble_btn.clicked.connect(self.activate_bg_tool)
        self.pan_zoom_btn.clicked.connect(self.activate_pan_zoom)
        self.run_seg_btn.clicked.connect(self.run_segmentation)
        self.reset_mask_btn.clicked.connect(self.reset_mask)

        # Параметры методов
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        # Морфология и отрисовка
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.close_kernel_slider.valueChanged.connect(self.on_close_kernel_changed)
        self.open_kernel_slider.valueChanged.connect(self.on_open_kernel_changed)
        self.kernel_shape_combo.currentIndexChanged.connect(self.schedule_update)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        # Универсальное сохранение аннотаций
        self.save_button.clicked.connect(self.save_current_annotations)

        settings.settings_changed.connect(self.on_global_settings_changed)

        self.image_view.setFocusPolicy(Qt.StrongFocus)

        self.update_navigation_state()
        self.activate_pan_zoom()
        self.on_method_changed(0)
        self.update_input_ui()
        self.update_params_visibility()
        self.params_widget.grabcut_mode.currentIndexChanged.connect(self.update_input_ui)
        self.setMinimumSize(0, 0)

        # ----------------------------------------------------------------------
    #  Вспомогательные методы
    # ----------------------------------------------------------------------
    def log(self, message):
        if hasattr(self, 'log_text'):
            self.log_text.append(message)
        elif hasattr(self, 'log_widget'):
            self.log_widget.log(message)
        print(message)

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
            self.schedule_update()

    def save_as_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите имя пресета:")
        if ok and name:
            if name in self.preset_manager.get_preset_names():
                reply = QMessageBox.question(self, "Перезаписать пресет",
                                             f"Пресет '{name}' уже существует. Перезаписать?",
                                             QMessageBox.Yes | QMessageBox.No)
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
        reply = QMessageBox.question(self, "Удалить пресет", f"Удалить пресет '{current}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.preset_manager.delete_preset(current):
                self.update_preset_combo()
                self.log(f"Удалён пресет: {current}")

    # ----------------------------------------------------------------------
    #  Управление инструментами (через SmartGraphicsView)
    # ----------------------------------------------------------------------
    def activate_rect_tool(self):
        self.current_tool = "rect"
        self.rect_mode_btn.setChecked(True)
        self.fg_scribble_btn.setChecked(False)
        self.bg_scribble_btn.setChecked(False)
        self.pan_zoom_btn.setChecked(False)
        self.image_view.set_drawing_tool("rect")
        self.log("Режим: рисование прямоугольника (один бокс, затем сброс)")

    def activate_fg_tool(self):
        if self.current_tool == "fg":
            self.activate_pan_zoom()
        else:
            self.current_tool = "fg"
            self.rect_mode_btn.setChecked(False)
            self.fg_scribble_btn.setChecked(True)
            self.bg_scribble_btn.setChecked(False)
            self.pan_zoom_btn.setChecked(False)
            self.image_view.set_drawing_tool("fg")
            self.log("Режим: штрихи переднего плана (повторное нажатие – выход)")

    def activate_bg_tool(self):
        if self.current_tool == "bg":
            self.activate_pan_zoom()
        else:
            self.current_tool = "bg"
            self.rect_mode_btn.setChecked(False)
            self.fg_scribble_btn.setChecked(False)
            self.bg_scribble_btn.setChecked(True)
            self.pan_zoom_btn.setChecked(False)
            self.image_view.set_drawing_tool("bg")
            self.log("Режим: штрихи фона (повторное нажатие – выход)")

    def activate_pan_zoom(self):
        self.current_tool = None
        self.rect_mode_btn.setChecked(False)
        self.fg_scribble_btn.setChecked(False)
        self.bg_scribble_btn.setChecked(False)
        self.pan_zoom_btn.setChecked(True)
        self.image_view.set_drawing_tool(None)
        self.log("Режим панорамирования/зума")

    # ----------------------------------------------------------------------
    #  Обработка рисования
    # ----------------------------------------------------------------------
    def on_rect_drawn(self, rect):
        self.rect = rect
        self.log(f"Прямоугольник задан: {self.rect}")
        self.update_input_ui()
        self.activate_pan_zoom()
        self.update_scribble_display()

    def on_scribble_added(self, x, y, tool):
        if tool == 'fg':
            if self.fg_scribbles is None:
                self.fg_scribbles = np.zeros((self.current_img_h, self.current_img_w), dtype=np.uint8)
            cv2.circle(self.fg_scribbles, (x, y), 5, 255, -1)
        else:
            if self.bg_scribbles is None:
                self.bg_scribbles = np.zeros((self.current_img_h, self.current_img_w), dtype=np.uint8)
            cv2.circle(self.bg_scribbles, (x, y), 5, 255, -1)
        self.update_input_ui()
        self.update_scribble_display()

    def update_input_ui(self):
        try:
            method = self.method_combo.currentText()
            rect_ok = bool(self.rect is not None)
            fg_ok = bool(self.fg_scribbles is not None and np.any(self.fg_scribbles == 255))
            bg_ok = bool(self.bg_scribbles is not None and np.any(self.bg_scribbles == 255))

            if method == "GrabCut":
                mode = self.params_widget.grabcut_mode.currentText()
                if mode == "GC_INIT_WITH_RECT":
                    rect_active, fg_active, bg_active = True, False, False
                    rect_required, fg_required, bg_required = True, False, False
                    can_run = rect_ok
                    requirements = "Требуется прямоугольник"
                elif mode == "GC_INIT_WITH_MASK":
                    rect_active, fg_active, bg_active = False, True, True
                    rect_required, fg_required, bg_required = False, True, True
                    can_run = (fg_ok and bg_ok)
                    requirements = "Требуются штрихи FG и BG"
                else:  # "Both"
                    rect_active, fg_active, bg_active = True, True, True
                    rect_required, fg_required, bg_required = False, False, False
                    can_run = rect_ok or (fg_ok and bg_ok)
                    requirements = "Требуется прямоугольник ИЛИ штрихи FG и BG"
            elif method == "Lazy Snapping":
                rect_active, fg_active, bg_active = False, True, True
                rect_required, fg_required, bg_required = False, True, True
                can_run = bool(fg_ok and bg_ok)
                requirements = "Требуются штрихи FG и BG"
            elif method in ("SuperCut", "Active Contours", "OneCut"):
                rect_active, fg_active, bg_active = True, False, False
                rect_required, fg_required, bg_required = True, False, False
                can_run = bool(rect_ok)
                requirements = "Требуется прямоугольник"
            elif method == "Random Walker":
                rect_active, fg_active, bg_active = False, True, True
                rect_required, fg_required, bg_required = False, True, True
                can_run = bool(fg_ok and bg_ok)
                requirements = "Требуются штрихи FG и BG"
            elif method == "Watershed (marker)":
                rect_active = fg_active = bg_active = False
                rect_required = fg_required = bg_required = False
                can_run = True
                requirements = "Не требует ввода (автоматическая сегментация)"
            else:
                rect_active = fg_active = bg_active = False
                rect_required = fg_required = bg_required = False
                can_run = False
                requirements = "Метод не реализован"

            self.rect_mode_btn.setText(f"Draw Bounding Box {'(✓)' if rect_ok else '(✗)'}")
            self.fg_scribble_btn.setText(f"Scribble FG {'(✓)' if fg_ok else '(✗)'}")
            self.bg_scribble_btn.setText(f"Scribble BG {'(✓)' if bg_ok else '(✗)'}")

            def set_style(btn, active, required, ok):
                btn.setEnabled(bool(active))
                if active:
                    if required and not ok:
                        btn.setStyleSheet("background-color: #ffcccc; color: black;")
                    elif ok:
                        btn.setStyleSheet("background-color: #ccffcc; color: black;")
                    else:
                        btn.setStyleSheet("")
                else:
                    btn.setStyleSheet("background-color: #cccccc; color: gray;")

            set_style(self.rect_mode_btn, rect_active, rect_required, rect_ok)
            set_style(self.fg_scribble_btn, fg_active, fg_required, fg_ok)
            set_style(self.bg_scribble_btn, bg_active, bg_required, bg_ok)

            self.requirements_text.setText(requirements)
            self.run_seg_btn.setEnabled(bool(can_run))

        except Exception as e:
            import traceback
            self.log(f"Ошибка в update_input_ui: {e}\n{traceback.format_exc()}")

    def update_scribble_display(self):
        if not self.display_images:
            return
        original = self.display_images[self.current_index]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        display = original.copy()
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        if self.fg_scribbles is not None:
            display[self.fg_scribbles == 255] = [0, 255, 0]
        if self.bg_scribbles is not None:
            display[self.bg_scribbles == 255] = [0, 0, 255]
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if self.image_view.temp_rect is not None:
            x1, y1, x2, y2 = self.image_view.temp_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        if self.image_view.temp_scribble is not None:
            x, y, r = self.image_view.temp_scribble
            color = (0, 255, 0) if self.current_tool == "fg" else (0, 0, 255)
            cv2.circle(display, (x, y), r, color, -1)

        pixmap = numpy_to_qpixmap(display)
        self.original_view.set_pixmap(pixmap)
        self.update_input_ui()

    # ----------------------------------------------------------------------
    #  Гистограмма
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    #  Параметры методов
    # ----------------------------------------------------------------------
    def on_method_changed(self, idx):
        method = self.method_combo.currentText()
        self.update_params_visibility()
        self.update_input_ui()

    def update_params_visibility(self):
        method = self.method_combo.currentText()
        self.params_widget.grabcut_group.setVisible(False)
        self.params_widget.lazy_group.setVisible(False)
        self.params_widget.supercut_group.setVisible(False)
        self.params_widget.onecut_group.setVisible(False)
        self.params_widget.watershed_group.setVisible(False)
        self.params_widget.rw_group.setVisible(False)
        self.params_widget.ac_group.setVisible(False)

        if method == "GrabCut":
            self.params_widget.grabcut_group.setVisible(True)
        elif method == "Lazy Snapping":
            self.params_widget.lazy_group.setVisible(True)
        elif method == "SuperCut":
            self.params_widget.lazy_group.setVisible(True)
            self.params_widget.supercut_group.setVisible(True)
        elif method == "OneCut":
            self.params_widget.lazy_group.setVisible(True)
            self.params_widget.onecut_group.setVisible(True)
        elif method == "Random Walker":
            self.params_widget.rw_group.setVisible(True)
        elif method == "Watershed (marker)":
            self.params_widget.watershed_group.setVisible(True)
        elif method == "Active Contours":
            self.params_widget.ac_group.setVisible(True)

    # ----------------------------------------------------------------------
    #  Запуск сегментации
    # ----------------------------------------------------------------------
    def run_segmentation(self):
        if not self.display_images:
            QMessageBox.warning(self, "Нет изображения", "Загрузите изображение.")
            return

        self.run_seg_btn.setEnabled(False)
        original_text = self.run_seg_btn.text()
        self.run_seg_btn.setText("Сегментация...")
        self.run_seg_btn.setStyleSheet("background-color: orange; color: black;")
        QApplication.processEvents()

        method = self.method_combo.currentText()
        img = self.display_images[self.current_index]
        gray = self.gray_images[self.current_index]

        # доп инеерсия
        #gray = cv2.bitwise_not(gray)

        if self.fg_scribbles is None:
            self.fg_scribbles = np.zeros(gray.shape, dtype=np.uint8)
        if self.bg_scribbles is None:
            self.bg_scribbles = np.zeros(gray.shape, dtype=np.uint8)

        self.log(f"Запуск сегментации методом {method}...")
        try:
            if method == "GrabCut":
                params = self.params_widget.get_grabcut_params()
                iters = params['iterations']
                mode = params['mode']
                self.mask = grabcut_segmentation(img, self.rect, None, self.fg_scribbles, self.bg_scribbles,
                                                 iterations=iters, mode=mode)
            elif method == "Watershed (marker)":
                params = self.params_widget.get_watershed_params()
                thresh = params['threshold']
                min_distance = params['min_distance']
                self.mask = watershed_segmentation(gray, distance_threshold=thresh, min_area=min_distance)
            elif method == "Lazy Snapping":
                params = self.params_widget.get_superpixel_params()
                self.mask = lazy_snapping(
                    img, self.fg_scribbles, self.bg_scribbles,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],
                    color_sigma=params['color_sigma']
                )
            elif method == "SuperCut":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для SuperCut необходимо нарисовать прямоугольник.")
                    return
                params = self.params_widget.get_supercut_params()
                self.mask = supercut_segmentation(
                    img, self.rect,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],  # сглаживание
                    lambda_val=params['lambda_val'],
                    sigma_color=params['sigma_color']  # добавленный параметр
                )
            elif method == "OneCut":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для OneCut необходимо нарисовать прямоугольник.")
                    return
                params = self.params_widget.get_onecut_params()
                self.mask = onecut_segmentation(
                    img, self.rect,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],
                    spatial_weight=params['spatial_weight'],
                    data_weight=params['data_weight'],
                    color_sigma=params['color_sigma']  # добавлено
                )
            elif method == "Random Walker":
                params = self.params_widget.get_random_walker_params()
                markers = np.zeros(gray.shape, dtype=np.int32)
                markers[self.fg_scribbles == 255] = 1
                markers[self.bg_scribbles == 255] = 2
                if np.sum(markers == 1) == 0 or np.sum(markers == 2) == 0:
                    QMessageBox.warning(self, "Недостаточно меток", "Нанесите штрихи переднего и фонового плана.")
                    return
                self.mask = random_walker_segmentation(img, markers, beta=params['beta'], mode=params['mode'])
            elif method == "Active Contours":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для Active Contours нарисуйте прямоугольник.")
                    return
                params = self.params_widget.get_active_contour_params()
                self.mask = active_contour_segmentation(
                    gray, self.rect,
                    alpha=params['alpha'],
                    beta=params['beta'],
                    gamma=params['gamma'],
                    max_iter=params['max_iter'],
                    convergence=params['convergence'],
                    smooth=params['smooth'],
                    init_type=params['init_type']
                )
            else:
                self.log(f"Метод {method} не реализован")
                return

            self.schedule_update()
            self.log("Сегментация завершена.")
        except Exception as e:
            self.log(f"Ошибка при сегментации: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить сегментацию:\n{e}")
        finally:
            self.run_seg_btn.setEnabled(True)
            self.run_seg_btn.setText(original_text)
            self.run_seg_btn.setStyleSheet("")
            QApplication.processEvents()

    def reset_mask(self):
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.image_view.set_image_data(self.current_img_w, self.current_img_h,
                                       self.fg_scribbles, self.bg_scribbles)
        self.log("Маска и все вводные данные сброшены.")
        self.update_input_ui()
        self.update_scribble_display()
        self.schedule_update()

    # ----------------------------------------------------------------------
    #  Отображение маски и результатов
    # ----------------------------------------------------------------------
    def display_current_image(self):
        if not self.display_images:
            self.update_current_histogram()
            for v in [self.original_view, self.segmentation_view, self.morph_view, self.annotated_view]:
                v.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            self.annotations = []
            return

        self.info_label.setText(f"Image {self.current_index + 1} of {len(self.display_images)}")
        current_file = os.path.basename(self.image_paths[self.current_index])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"inter_original_{base_name}")
        self.segmentation_view.set_suggested_save_name(f"inter_seg_{base_name}")
        self.morph_view.set_suggested_save_name(f"inter_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"inter_annotated_{base_name}")

        original = self.display_images[self.current_index]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        gray = self.gray_images[self.current_index]
        self.current_img_h, self.current_img_w = gray.shape

        self.image_view.set_image_data(self.current_img_w, self.current_img_h,
                                       self.fg_scribbles, self.bg_scribbles)

        self.update_histogram(gray)
        self.update_scribble_display()

        if self.mask is not None:
            self.segmentation_view.set_pixmap(numpy_to_qpixmap(self.mask))
        else:
            self.segmentation_view.set_pixmap(numpy_to_qpixmap(np.zeros_like(gray)))

        close_factor = self.close_kernel_slider.value() / 100.0
        open_factor = self.open_kernel_slider.value() / 100.0
        kernel_shape = self.kernel_shape_combo.currentText()
        if self.mask is not None:
            processed = apply_morphology(self.mask, close_factor, open_factor, kernel_shape, gray.shape)
            if self.invert_checkbox.isChecked():
                processed = cv2.bitwise_not(processed)
        else:
            processed = np.zeros_like(gray)
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        if self.mask is not None:
            if len(original.shape) == 2:
                result = original.copy()
                result[processed == 0] = 255
            else:
                result = original.copy()
                result[processed == 0] = [255, 255, 255]
        else:
            result = original.copy()
        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()

        self.current_base_image = display_img.copy()
        if self.mask is not None and self.draw_combo.currentText() != "None":
            _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)
        else:
            self.current_objects_full = []
        self.update_object_list()

        # Синхронизация спинбокса
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))

    # ----------------------------------------------------------------------
    #  Список объектов (универсальный)
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

    # Обновлённая версия с поддержкой detect/obb/segment
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

    # Универсальное сохранение аннотаций
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
    #  Навигация и загрузка
    # ----------------------------------------------------------------------
    def update_navigation_state(self):
        has_images = len(self.display_images) > 0
        if has_images:
            self.nav_widget.set_prev_enabled(self.current_index > 0)
            self.nav_widget.set_next_enabled(self.current_index < len(self.display_images) - 1)
        else:
            self.nav_widget.set_prev_enabled(False)
            self.nav_widget.set_next_enabled(False)

    def prev_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index - 1) % len(self.display_images)
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.display_current_image()
        self.update_navigation_state()

    def next_image(self):
        if not self.display_images:
            return
        self.current_index = (self.current_index + 1) % len(self.display_images)
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.display_current_image()
        self.update_navigation_state()

    def goto_image(self, page_num):
        if not self.display_images:
            return
        total = len(self.display_images)
        page_num = max(1, min(page_num, total))
        new_idx = page_num - 1
        if new_idx != self.current_index:
            self.current_index = new_idx
            self.mask = None
            self.rect = None
            self.fg_scribbles = None
            self.bg_scribbles = None
            self.display_current_image()
            self.update_navigation_state()

    def load_images_from_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.log(f"Loading {len(file_paths)} images...")
        paths, imgs, grays, anns = load_images_universal(
            source=file_paths,
            require_annotations=False,
            resize_enabled=self.nav_widget.is_resize_enabled(),
            parent=self
        )
        if not paths:
            self.log("No images loaded.")
            self.nav_widget.set_current_index(0, 0)
            return
        self.image_paths = paths
        self.display_images = imgs
        self.gray_images = grays
        self.annotations = anns
        self.current_index = 0
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))
        self.update_navigation_state()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.log(f"Loading folder: {folder}")
        paths, imgs, grays, anns = load_images_universal(
            source=folder,
            require_annotations=False,
            resize_enabled=self.nav_widget.is_resize_enabled(),
            parent=self
        )
        if not paths:
            self.log("No images found.")
            self.nav_widget.set_current_index(0, 0)
            return
        self.image_paths = paths
        self.display_images = imgs
        self.gray_images = grays
        self.annotations = anns
        self.current_index = 0
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))
        self.update_navigation_state()

    # Перезагрузка изображений (для смены режима ресайза)
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
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.display_images))
        self.update_navigation_state()
        self.log(f"Перезагружено {len(paths)} изображений.")

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
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        for v in [self.original_view, self.segmentation_view, self.morph_view, self.annotated_view]:
            v.set_pixmap(numpy_to_qpixmap(None))
        self.info_label.setText("No images")
        self.object_list.clear()
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    # ----------------------------------------------------------------------
    #  UI события
    # ----------------------------------------------------------------------
    def reset_all_zooms(self):
        for view in [self.original_view, self.segmentation_view, self.morph_view, self.annotated_view]:
            view.reset_view()

    def schedule_update(self):
        self.update_timer.start(50)

    def on_close_kernel_changed(self, value):
        self.close_kernel_label.setText(f"{value / 100:.2f}")
        self.schedule_update()

    def on_open_kernel_changed(self, value):
        self.open_kernel_label.setText(f"{value / 100:.2f}")
        self.schedule_update()

    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        if mode in ["Contours (simple)", "Min area rectangle"]:
            self.hull_checkbox.setVisible(True)
        else:
            self.hull_checkbox.setVisible(False)
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
        elif mode == "Contours (simple)":
            if draw:
                return segment_contours(img, binary, use_hull)
            else:
                dummy = img.copy()
                return segment_contours(dummy, binary, use_hull)
        elif mode == "Projections (connected components)":
            if draw:
                return segment_projections(img, binary)
            else:
                dummy = img.copy()
                return segment_projections(dummy, binary)
        elif mode == "Min area rectangle":
            if draw:
                return segment_min_area_rect(img, binary, use_hull)
            else:
                dummy = img.copy()
                return segment_min_area_rect(dummy, binary, use_hull)
        else:
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
    window = InteractiveMethodsWindow()
    window.show()
    sys.exit(app.exec_())