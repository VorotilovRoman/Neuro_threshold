# threshold_methods.py
from import_libs_internal import *
from import_libs_methods_ui import setup_threshold_ui
from path_setup import get_project_root


class ThresholdWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Threshold Methods")

        # UI создаётся функцией, которая сама устанавливает центральный виджет
        setup_threshold_ui(self)
        print(hasattr(self.nav_widget, 'load_folder'))
        print(self.nav_widget.load_folder)

        # Данные изображений.
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Текущее изображение и его grayscale-версия для всей логики вкладки.
        self.original_image = None
        self.current_gray = None

        self.current_threshold = None
        self.current_objects_full = []   # список объектов в унифицированном формате
        self.current_selected_indices = []
        self.current_base_image = None

        # Автонастройки
        _root = get_project_root()
        auto_settings_dir = os.path.join(_root, "settings", "auto_settings")
        os.makedirs(auto_settings_dir, exist_ok=True)
        self.settings_file = os.path.join(auto_settings_dir, "auto_settings.json")
        self.auto_settings = None
        self.load_auto_settings()

        self.auto_enable_checkbox.setChecked(False)
        self.configure_auto_button.clicked.connect(self.open_auto_settings)
        self.auto_button.clicked.connect(self.run_auto_optimization)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Пресеты
        self.preset_manager = PresetManager(self)
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

        # Подключение сигналов
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

        self.threshold_slider.valueChanged.connect(self.on_simple_threshold_changed)
        self.invert_checkbox.stateChanged.connect(self.on_invert_changed)
        self.close_kernel_slider.valueChanged.connect(self.on_close_kernel_changed)
        self.open_kernel_slider.valueChanged.connect(self.on_open_kernel_changed)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.adaptive_win.valueChanged.connect(self.on_param_changed)
        self.adaptive_c.valueChanged.connect(self.on_param_changed)
        self.niblack_win.valueChanged.connect(self.on_param_changed)
        self.niblack_k.valueChanged.connect(self.on_niblack_k_changed)
        self.sauvola_win.valueChanged.connect(self.on_param_changed)
        self.sauvola_k.valueChanged.connect(self.on_sauvola_k_changed)
        self.sauvola_r.valueChanged.connect(self.on_sauvola_r_changed)
        self.isodata_init.valueChanged.connect(self.on_param_changed)
        self.row_win.valueChanged.connect(self.on_param_changed)
        self.row_k.valueChanged.connect(self.on_row_k_changed)
        self.kernel_shape_combo.currentIndexChanged.connect(self.on_param_changed)
        self.bg_excess.valueChanged.connect(self.on_bg_excess_changed)

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)
        # Замена save_coordinates на универсальное сохранение
        self.save_button.clicked.connect(self.save_current_annotations)

        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.on_method_changed(self.method_combo.currentIndex())
        self.update_navigation_state()

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)
        settings.settings_changed.connect(self.on_global_settings_changed)

        # Применить пресет по умолчанию (если есть)
        preset_names = self.preset_manager.get_preset_names()
        if preset_names:
            default_preset = "default" if "default" in preset_names else preset_names[0]
            # Блокируем сигналы комбобокса, чтобы не вызывать повторное применение
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(default_preset)
            self.preset_combo.blockSignals(False)
            self.preset_manager.apply_preset(default_preset)
            self.log(f"Применён пресет по умолчанию: {default_preset}")

    def log(self, message):
        self.log_text.append(message)
        print(message)

    # ---------- Загрузка/сохранение настроек автоподбора ----------
    def load_auto_settings(self):
        if not os.path.exists(self.settings_file):
            self.log("Файл настроек автоподбора не найден, будут использованы настройки по умолчанию.")
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                self.auto_settings = json.load(f)
            self.log(f"Настройки автоподбора загружены из {self.settings_file}")
        except Exception as e:
            self.log(f"Ошибка загрузки настроек автоподбора: {e}")

    def save_auto_settings(self):
        if self.auto_settings is None:
            return
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.auto_settings, f, indent=4, ensure_ascii=False)
            self.log(f"Настройки автоподбора сохранены в {self.settings_file}")
        except Exception as e:
            self.log(f"Ошибка сохранения настроек автоподбора: {e}")

    def closeEvent(self, event):
        # Аккуратно останавливаем фоновый пул, чтобы процесс не висел
        # на незавершённых задачах загрузки.
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        self.save_auto_settings()
        event.accept()

    def open_auto_settings(self):
        dlg = AutoSettingsDialog(self, self.auto_settings)
        if dlg.exec_():
            self.auto_settings = dlg.get_settings()
            self.save_auto_settings()
            self.log("Настройки автоподбора обновлены и сохранены")
        else:
            self.log("Настройки автоподбора не изменены")

    def run_auto_optimization(self):
        if not self.auto_enable_checkbox.isChecked():
            self.log("Автоподбор не включён. Установите галочку Enable auto research.")
            QMessageBox.warning(self, "Автоподбор", "Включите автоподбор галочкой Enable auto research.")
            return
        if not self.auto_settings:
            self.log("Нет настроек автоподбора. Нажмите Configure auto research для настройки.")
            QMessageBox.warning(self, "Автоподбор", "Сначала настройте параметры автоподбора (Configure auto research).")
            return
        self.log("Запуск автоподбора параметров...")
        search_all = self.search_all_checkbox.isChecked()
        research = AutoResearch(self, search_all)
        research.run()

    # ---------- Гистограмма ----------
    def update_histogram(self, gray_img, threshold=None):
        ax = self.hist_ax
        ax.clear()
        ax.hist(gray_img.ravel(), bins=256, range=(0, 256), color='black', alpha=0.7)
        ax.set_title("Grayscale Histogram")
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Frequency")
        if threshold is not None:
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold}')
            ax.legend()
        self.hist_canvas.draw()

    def update_current_histogram(self):
        if self.current_gray is not None:
            self.update_histogram(self.current_gray, self.current_threshold)

    # ---------- Сбор параметров для бинаризации ----------
    def get_current_threshold_params(self):
        method = self.method_combo.currentText()
        params = {}
        if method == "Simple Threshold":
            params["threshold"] = self.threshold_slider.value()
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            params["window"] = self.adaptive_win.value()
            params["c"] = self.adaptive_c.value()
        elif method == "Niblack":
            params["window"] = self.niblack_win.value()
            params["k"] = self.niblack_k.value() / 100.0
        elif method == "Sauvola":
            params["window"] = self.sauvola_win.value()
            params["k"] = self.sauvola_k.value() / 100.0
            params["r"] = self.sauvola_r.value()
        elif method == "ISODATA":
            params["init"] = self.isodata_init.value()
        elif method == "Background Symmetry":
            params["excess"] = self.bg_excess.value() / 100.0
        elif method == "Row Adaptive":
            params["window"] = self.row_win.value()
            params["k"] = self.row_k.value() / 100.0
        return params

    # ---------- Отображение текущего изображения ----------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.update_current_histogram()
            self.original_view.set_pixmap(numpy_to_qpixmap(None))
            self.binary_view.set_pixmap(numpy_to_qpixmap(None))
            self.morph_view.set_pixmap(numpy_to_qpixmap(None))
            self.annotated_view.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        self.info_label.setText(f"Image {idx + 1} of {total}")

        # --- Синхронно получаем элемент (LRU-кэш вернёт готовый или загрузит) ---
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Не удалось загрузить снимок: {self.image_paths[idx]}")
            return

        original = item["image"]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        gray = item["gray"]

        self.original_image = original
        self.current_gray = gray

        method = self.method_combo.currentText()
        params = self.get_current_threshold_params()
        binary, thresh = apply_threshold_method(gray, method, params)
        self.current_threshold = thresh
        binary = cv2.bitwise_not(binary)

        current_file = os.path.basename(self.image_paths[idx])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"thresh_original_{base_name}")
        self.binary_view.set_suggested_save_name(f"thresh_binary_{base_name}")
        self.morph_view.set_suggested_save_name(f"thresh_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"thresh_annotated_{base_name}")

        self.update_histogram(gray, self.current_threshold)
        self.original_view.set_pixmap(numpy_to_qpixmap(original))
        self.binary_view.set_pixmap(numpy_to_qpixmap(binary))

        close_factor = self.close_kernel_slider.value() / 100.0
        open_factor = self.open_kernel_slider.value() / 100.0
        kernel_shape = self.kernel_shape_combo.currentText()
        processed = apply_morphology(binary, close_factor, open_factor, kernel_shape, gray.shape)
        if self.invert_checkbox.isChecked():
            processed = cv2.bitwise_not(processed)

        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        if len(original.shape) == 2 or (len(original.shape) == 3 and original.shape[2] == 1):
            result = original.copy()
            result[processed == 0] = 255
        else:
            result = original.copy()
            mask = processed == 0
            result[mask] = [255, 255, 255]

        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()
        self.current_base_image = display_img.copy()
        # Отрисовка объектов: получаем изображение и список унифицированных объектов
        _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)
        self.update_object_list()

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ---------- Работа со списком объектов (универсальная) ----------
    def update_object_list(self):
        self.object_list.blockSignals(True)
        self.object_list.clear()
        self.current_selected_indices = []

        img = self.original_image
        if img is None:
            self.object_list.blockSignals(False)
            return
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
        img = self.original_image
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        for i, idx in enumerate(self.current_selected_indices, 1):
            obj = self.current_objects_full[idx]
            # Определим тип по длине или по первому элементу (если кортеж унифицирован)
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
                    _, cls, points = obj  # 8 нормал. координат
                    pts_str = ' '.join(f"{p:.3f}" for p in points)
                    self.coord_text.append(f"{i}: obb class={cls}, YOLO-OBB: {cls} {pts_str}")
                elif typ == 'segment':
                    _, cls, points = obj
                    # Показываем первые 3 точки (6 чисел) для краткости
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
        total = len(self.image_paths) if self.image_paths else 0
        if total > 0:
            self.nav_widget.set_prev_enabled(self.current_index > 0)
            self.nav_widget.set_next_enabled(self.current_index < total - 1)
        else:
            self.nav_widget.set_prev_enabled(False)
            self.nav_widget.set_next_enabled(False)

    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.image_paths))
        self.update_navigation_state()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, len(self.image_paths))
        self.update_navigation_state()

    def goto_image(self, page_num):
        total = len(self.image_paths) if self.image_paths else 0
        if total == 0:
            return
        page_num = max(1, min(page_num, total))
        self.current_index = page_num - 1
        self.display_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_index, total)

    # ---------- Загрузка изображений ----------
    def load_images_from_dialog(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.log(f"Loading {len(file_paths)} images...")
        self._load_images(file_paths)

    def load_folder(self):
        print("DEBUG: load_folder вызван")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print("DEBUG: выбрана папка", folder)
        if not folder:
            return
        self.log(f"Loading folder: {folder}")
        self._load_images(folder)

    def reload_current_images(self):
        """Перезагружает текущий набор изображений с учётом текущего режима ресайза."""
        if not self.image_paths:
            self.log("Нет загруженных изображений для перезагрузки.")
            return
        self.log("Перезагрузка изображений с новыми настройками ресайза...")
        self._load_images(list(self.image_paths))

    # ------- Ленивая загрузка (единая точка входа) -------
    def _load_images(self, source):
        # --- Собираем список файлов БЕЗ загрузки картинок ---
        if isinstance(source, str) and os.path.isdir(source):
            exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
            all_files = sorted(
                os.path.join(source, f)
                for f in os.listdir(source)
                if os.path.splitext(f)[1].lower() in exts
            )
        elif isinstance(source, list):
            all_files = list(source)
        else:
            self.log("No images loaded.")
            return

        if not all_files:
            self.log("No images loaded.")
            return

        resize_enabled = self.nav_widget.is_resize_enabled()
        max_side = 1024

        # --- Останавливаем предыдущий кэш ---
        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None

        # --- Создаём кэш ---
        loader = lambda p: load_one_image_item(
            p,
            resize_enabled=resize_enabled,
            max_side=max_side,
            safe_max_side=None,
        )
        self._lazy_cache = LazyImageCache(
            paths=all_files,
            loader=loader,
            window=8,
            prefetch_ahead=4,
            max_resident=16,
            max_workers=2,
        )

        # --- Сброс состояния ---
        self.image_paths = all_files
        self.current_index = 0
        self.original_image = None
        self.current_gray = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        for view in (self.original_view, self.binary_view,
                     self.morph_view, self.annotated_view):
            view.set_pixmap(numpy_to_qpixmap(None))
        self.object_list.clear()

        # --- Синхронно грузим первый снимок ---
        first = self._lazy_cache.get(0)
        if first is None:
            self.log("Не удалось загрузить первое изображение.")
            return

        # --- Остальное окно — в фон ---
        total = len(self._lazy_cache)
        self._lazy_cache.prefetch(
            range(1, min(1 + self._lazy_cache.prefetch_ahead + 1, total))
        )

        self.log(f"Загружено {total} изображений (первое — сразу, остальное в фоне).")
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, total)
        self.update_navigation_state()

    # ---------- Сохранение аннотаций (универсальное) ----------
    # ---------- Сохранение аннотаций (только выбранные объекты) ----------
    # ---------- Сохранение аннотаций (включая пустой файл) ----------
    def save_current_annotations(self):
        if not self.image_paths or self.original_image is None:
            QMessageBox.warning(self, "Нет изображения", "Нет загруженных изображений.")
            return

        total_objects = len(self.current_objects_full)

        # Отбираем только те объекты, у которых стоит галочка.
        selected = [
            self.current_objects_full[i]
            for i in self.current_selected_indices
            if 0 <= i < total_objects
        ]

        img_path = self.image_paths[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        h, w = self.original_image.shape[:2]

        success = save_annotations(selected, txt_path, w, h)
        if not success:
            self.log(f"Ошибка сохранения {txt_path}")
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить аннотации.")
            return

        if len(selected) == 0:
            # Пустой файл — это валидный негативный пример для YOLO.
            if total_objects == 0:
                self.log(f"Сохранён пустой файл аннотаций (на снимке не найдено объектов) в {txt_path}")
                QMessageBox.information(
                    self, "Сохранение",
                    f"На снимке не найдено объектов.\n"
                    f"Сохранён пустой файл аннотаций:\n{txt_path}"
                )
            else:
                self.log(f"Сохранён пустой файл аннотаций "
                         f"(сняты галочки со всех {total_objects} объектов) в {txt_path}")
                QMessageBox.information(
                    self, "Сохранение",
                    f"Сняты галочки со всех объектов.\n"
                    f"Сохранён пустой файл аннотаций:\n{txt_path}"
                )
        else:
            self.log(f"Сохранено {len(selected)} из {total_objects} "
                     f"аннотаций в {txt_path}")
            QMessageBox.information(
                self, "Сохранение",
                f"Сохранено {len(selected)} из {total_objects} объектов в {txt_path}"
            )
    # ---------- События UI ----------
    def reset_all_zooms(self):
        for view in (self.original_view, self.binary_view,
                     self.morph_view, self.annotated_view):
            view.reset_view()

    def on_resize_mode_changed(self, enabled):
        if self.image_paths:
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
        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None
        self.image_paths = []
        self.current_index = 0
        self.original_image = None
        self.current_gray = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        for view in (self.original_view, self.binary_view,
                     self.morph_view, self.annotated_view):
            view.set_pixmap(numpy_to_qpixmap(None))
        self.info_label.setText("No images")
        self.object_list.clear()
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    def reset_params_for_method(self, method):
        if method == "Simple Threshold":
            self.threshold_slider.setValue(127)
            self.threshold_value_label.setText("127")
        elif method in ("Adaptive Mean", "Adaptive Gauss"):
            self.adaptive_win.setValue(25)
            self.adaptive_c.setValue(3)
        elif method == "Niblack":
            self.niblack_win.setValue(25)
            self.niblack_k.setValue(20)
            self.niblack_k_label.setText("0.20")
        elif method == "Sauvola":
            self.sauvola_win.setValue(25)
            self.sauvola_k.setValue(20)
            self.sauvola_r.setValue(128)
            self.sauvola_k_label.setText("0.20")
            self.sauvola_r_label.setText("128")
        elif method == "ISODATA":
            self.isodata_init.setValue(128)
        elif method == "Background Symmetry":
            self.bg_excess.setValue(20)
            self.bg_excess_label.setText("0.20")
        elif method == "Row Adaptive":
            self.row_win.setValue(50)
            self.row_k.setValue(50)
            self.row_k_label.setText("0.50")

    def on_method_changed(self, idx):
        method = self.method_combo.currentText()
        self.simple_threshold_container.setVisible(False)
        self.adaptive_container.setVisible(False)
        self.niblack_container.setVisible(False)
        self.sauvola_container.setVisible(False)
        self.isodata_container.setVisible(False)
        self.background_container.setVisible(False)
        self.row_adaptive_container.setVisible(False)

        if method == "Simple Threshold":
            self.simple_threshold_container.setVisible(True)
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            self.adaptive_container.setVisible(True)
        elif method == "Niblack":
            self.niblack_container.setVisible(True)
        elif method == "Sauvola":
            self.sauvola_container.setVisible(True)
        elif method == "ISODATA":
            self.isodata_container.setVisible(True)
        elif method == "Background Symmetry":
            self.background_container.setVisible(True)
        elif method == "Row Adaptive":
            self.row_adaptive_container.setVisible(True)

        self.reset_params_for_method(method)
        if self.original_image is not None:
            self.schedule_update()

    def schedule_update(self):
        self.update_timer.start(50)

    def on_param_changed(self):
        if self.original_image is not None:
            self.schedule_update()

    def on_simple_threshold_changed(self, value):
        self.threshold_value_label.setText(str(value))
        if self.original_image is not None and self.method_combo.currentText() == "Simple Threshold":
            self.schedule_update()

    def on_niblack_k_changed(self, value):
        self.niblack_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_sauvola_k_changed(self, value):
        self.sauvola_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_sauvola_r_changed(self, value):
        self.sauvola_r_label.setText(str(value))
        self.schedule_update()

    def on_row_k_changed(self, value):
        self.row_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_bg_excess_changed(self, value):
        self.bg_excess_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_close_kernel_changed(self, value):
        self.close_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_open_kernel_changed(self, value):
        self.open_kernel_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_invert_changed(self, state):
        if self.original_image is not None:
            self.schedule_update()

    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        if mode in ["Segmentation (Polygon)", "OBB (Oriented Box)"]:
            self.hull_checkbox.setVisible(True)
        else:
            self.hull_checkbox.setVisible(False)
        if self.original_image is not None:
            self.schedule_update()

    def on_hull_changed(self, state):
        if self.original_image is not None:
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

    # ---------- Отрисовка объектов ----------
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
            if draw:
                return segment_contours(img, binary, use_hull)
            else:
                dummy = img.copy()
                return segment_contours(dummy, binary, use_hull)
        elif mode == "Bounding Box (Detect)":
            if draw:
                return segment_projections(img, binary)
            else:
                dummy = img.copy()
                return segment_projections(dummy, binary)
        elif mode == "OBB (Oriented Box)":
            if draw:
                return segment_min_area_rect(img, binary, use_hull)
            else:
                dummy = img.copy()
                return segment_min_area_rect(dummy, binary, use_hull)
        else:
            # fallback
            if len(img.shape) == 2:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_color = img.copy()
            return img_color, []

    # ---------- Пресеты ----------
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

    def save_as_preset(self):
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите имя пресета:")
        if ok and name:
            if name in self.preset_manager.get_preset_names():
                reply = QMessageBox.question(
                    self, "Перезаписать пресет",
                    f"Пресет '{name}' уже существует. Перезаписать?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
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

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def toggle_histogram(self, checked):
        self.hist_container.setVisible(checked)
        self.toggle_hist_btn.setText("Скрыть гистограмму" if checked else "Показать гистограмму")

    def on_global_settings_changed(self, new_settings=None):
        if self.original_image is not None:
            self.update_annotated_view()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdWindow()
    window.show()
    sys.exit(app.exec_())