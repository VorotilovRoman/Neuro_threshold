from import_libs_internal import *
from import_libs_methods_ui import setup_yolo_find_img_ui

TEMP_THUMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_thumbs")

# ------------------------------------------------------------
# ClickableLabel – безопасная обработка двойного клика
# ------------------------------------------------------------
class ClickableLabel(QLabel):
    """QLabel, который обрабатывает двойной клик и вызывает callback с путём файла."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path = ""
        self._open_callback = None

    def set_file_path(self, file_path: str):
        self._file_path = file_path

    def set_open_callback(self, callback):
        self._open_callback = callback

    def mouseDoubleClickEvent(self, event):
        if self._open_callback and self._file_path:
            self._open_callback(self._file_path)
        super().mouseDoubleClickEvent(event)

# ------------------------------------------------------------
# Поток для сканирования
# ------------------------------------------------------------
class ScanThread(QThread):
    progress = pyqtSignal(int, int)
    log_msg = pyqtSignal(str)
    file_done = pyqtSignal(str, float, bool)
    finished = pyqtSignal(list)

    def __init__(self, root_dir, model_path, target_class, conf, iou, imgsz, device):
        super().__init__()
        self.root_dir = root_dir
        self.model_path = model_path
        self.target_class = target_class
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True
        self.log_msg.emit("Отмена сканирования...")

    def run(self):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        all_images = []
        self.log_msg.emit(f"Сбор изображений в папке: {self.root_dir}")
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    all_images.append(os.path.join(root, file))
        total = len(all_images)
        self.log_msg.emit(f"Найдено изображений: {total}")
        if total == 0:
            self.finished.emit([])
            return

        results = []
        self.log_msg.emit(f"Загрузка модели YOLO: {self.model_path}")
        model = YOLO(self.model_path)
        self.log_msg.emit("Модель загружена, начало инференса")

        for idx, img_path in enumerate(all_images):
            if self._is_canceled:
                self.log_msg.emit("Сканирование прервано пользователем")
                break
            self.progress.emit(idx + 1, total)

            try:
                img = read_image_with_fallback_find(img_path)
                if img is None:
                    self.log_msg.emit(f"Не удалось загрузить: {os.path.basename(img_path)}")
                    continue

                img = normalize_to_uint8(img)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.shape[2] != 3:
                    img = img[:, :, :3]

                results_yolo = model(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device)
                max_conf = 0.0
                contains_target = False
                if results_yolo[0].boxes is not None:
                    for box in results_yolo[0].boxes:
                        cls = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        if cls == self.target_class:
                            contains_target = True
                            max_conf = max(max_conf, conf_val)
                if contains_target:
                    results.append((img_path, max_conf))
                self.file_done.emit(img_path, max_conf, contains_target)

            except Exception as e:
                self.log_msg.emit(f"Критическая ошибка при обработке {os.path.basename(img_path)}: {e}")
                continue

        self.log_msg.emit(f"Сканирование завершено. Найдено файлов с целевой меткой: {len(results)}")
        self.finished.emit(results)

# ------------------------------------------------------------
# Главное окно
# ------------------------------------------------------------
class FindImagesWindow(QMainWindow, setup_yolo_find_img_ui):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.scan_thread = None
        self.current_root_dir = ""
        self.result_file_paths = []          # список (file_path, max_conf)
        self.thumbnail_widgets = {}
        self.thumbnail_pixmaps = {}

        self.select_all_checkbox.setTristate(False)

        # Подключение сигналов
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.scan_btn.clicked.connect(self.start_scan)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.browse_target_btn.clicked.connect(self.browse_target_folder)
        self.copy_btn.clicked.connect(self.copy_selected_files)
        self.move_btn.clicked.connect(self.move_selected_files)
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        self.toggle_view_btn.clicked.connect(self.toggle_view_mode)
        self.sort_btn.clicked.connect(self.sort_by_confidence)
        self.results_list.itemDoubleClicked.connect(self.open_file_from_list)

        # Настройка дерева файлов
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.file_model.setNameFilters(["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"])
        self.file_model.setNameFilterDisables(False)
        self.file_model.setRootPath("")
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(QDir.rootPath()))
        self.file_tree.doubleClicked.connect(self.open_file_from_tree)

        # Подключение виджета настроек YOLO
        self.yolo_settings.model_path_changed.connect(self.on_model_path_changed)
        self.log = self.log_widget.log

        # Добавляем комбобокс устройства (если его нет в UI)
        if hasattr(self, 'device_combo'):
            self.device_combo = getattr(self, 'device_combo')
        else:
            self.device_combo = QComboBox()
            self.device_combo.addItems(["auto", "0", "cpu", "mps"])
            parent_layout = self.yolo_settings.layout()
            if parent_layout:
                parent_layout.addLayout(self.yolo_settings._row("Устройство", self.device_combo))
            else:
                device_widget = QWidget()
                device_layout = QHBoxLayout(device_widget)
                device_layout.addWidget(QLabel("Устройство:"))
                device_layout.addWidget(self.device_combo)
                self.yolo_settings.layout().addWidget(device_widget)

        # Режим отображения
        self.is_thumbnail_mode = False
        self.stacked_view.setCurrentIndex(0)
        self.toggle_view_btn.setText("Режим: миниатюры")

        self._ensure_temp_dir()
        self._clear_temp_thumbs()

        # Загрузка классов текущей модели
        current_model = self.yolo_settings._model_path.text().strip()
        if current_model:
            self.update_model_classes(current_model)

        self.thumbnail_container.resizeEvent = self.on_thumbnail_container_resize

    # --------------------------------------------------------
    # Вспомогательные
    # --------------------------------------------------------
    def _ensure_temp_dir(self):
        if not os.path.exists(TEMP_THUMB_DIR):
            os.makedirs(TEMP_THUMB_DIR)
            self.log(f"Создана временная папка для миниатюр: {TEMP_THUMB_DIR}")

    def _clear_temp_thumbs(self):
        if os.path.exists(TEMP_THUMB_DIR):
            count = len(os.listdir(TEMP_THUMB_DIR))
            for f in os.listdir(TEMP_THUMB_DIR):
                try:
                    os.remove(os.path.join(TEMP_THUMB_DIR, f))
                except:
                    pass
            self.log(f"Очищена временная папка (удалено {count} файлов)")

    # --------------------------------------------------------
    # Классы модели
    # --------------------------------------------------------
    def update_model_classes(self, model_path):
        if not model_path or not os.path.exists(model_path):
            self.class_list_widget.clear()
            self.class_list_widget.addItem("Не удалось загрузить классы")
            return
        try:
            self.log(f"Загрузка классов модели: {model_path}")
            model = YOLO(model_path)
            names = model.names
            self.class_list_widget.clear()
            for idx in sorted(names.keys()):
                self.class_list_widget.addItem(f"{idx}: {names[idx]}")
            self.log(f"Загружено {len(names)} классов из модели")
        except Exception as e:
            self.class_list_widget.clear()
            self.class_list_widget.addItem("Ошибка загрузки классов")
            self.log(f"Не удалось загрузить классы модели: {e}")

    def on_model_path_changed(self, model_path):
        self.update_model_classes(model_path)

    # --------------------------------------------------------
    # Выбор папки и дерево
    # --------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите корневую папку для поиска")
        if not folder:
            return
        self.current_root_dir = folder
        self.file_tree.setRootIndex(self.file_model.index(folder))
        self.log(f"Выбрана папка: {folder}")

    def open_file_from_tree(self, index):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path):
            self.open_file(file_path)

    def open_file_from_list(self, item):
        if item is None:
            return
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            self.open_file(file_path)
        else:
            QMessageBox.warning(self, "Файл не найден", f"Файл {file_path} не существует.")

    def open_file(self, file_path):
        self.log(f"Открытие файла: {file_path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    # --------------------------------------------------------
    # Сканирование
    # --------------------------------------------------------
    def start_scan(self):
        if not self.current_root_dir:
            QMessageBox.warning(self, "Нет папки", "Сначала выберите папку для поиска.")
            return

        model_path = self.yolo_settings._model_path.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Нет модели", "Укажите существующий файл модели YOLO.")
            return

        target_class = self.target_class_spin.value()
        conf = self.yolo_settings._conf.value()
        iou = self.yolo_settings._iou.value()
        imgsz = self.yolo_settings._imgsz.value()
        device = self.device_combo.currentText()
        device = self.resolve_device(device)

        self.log("=" * 50)
        self.log("ЗАПУСК СКАНИРОВАНИЯ")
        self.log(f"Модель: {model_path}")
        self.log(f"Целевая метка: {target_class}")
        self.log(f"Параметры: conf={conf}, iou={iou}, imgsz={imgsz}")
        self.log(f"Используется устройство: {device}")
        self.log(f"Корневая папка: {self.current_root_dir}")

        self.results_list.clear()
        self._clear_thumbnail_grid()
        self.result_file_paths = []
        self.select_all_checkbox.setChecked(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Сканирование: 0%")
        self.scan_btn.setEnabled(False)
        self.select_folder_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.toggle_view_btn.setEnabled(False)
        self.sort_btn.setEnabled(False)

        self.scan_thread = ScanThread(
            root_dir=self.current_root_dir,
            model_path=model_path,
            target_class=target_class,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device
        )
        self.scan_thread.progress.connect(self.update_progress)
        self.scan_thread.log_msg.connect(self.log)
        self.scan_thread.file_done.connect(self.on_file_done)
        self.scan_thread.finished.connect(self.on_scan_finished)
        self.scan_thread.start()

    def stop_scan(self):
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.cancel()
            self.stop_btn.setEnabled(False)
            self.log("Запрошена остановка сканирования...")

    def resolve_device(self, device_str):
        if device_str == 'auto':
            return '0' if torch.cuda.is_available() else 'cpu'
        elif device_str == 'mps':
            return 'cpu' if not torch.backends.mps.is_available() else 'mps'
        return device_str

    def update_progress(self, current, total):
        percent = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"Сканирование: {percent}% ({current}/{total})")
        QApplication.processEvents()

    def on_file_done(self, file_path, max_conf, contains_target):
        if contains_target:
            self.result_file_paths.append((file_path, max_conf))
            item = QListWidgetItem(f"{os.path.basename(file_path)} (conf: {max_conf:.3f})")
            item.setData(Qt.UserRole, file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.results_list.addItem(item)

    def on_scan_finished(self, result_files):
        self.scan_btn.setEnabled(True)
        self.select_folder_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.toggle_view_btn.setEnabled(True)
        self.sort_btn.setEnabled(True)
        self.progress_bar.setFormat("Готово")
        self.log(f"Сканирование завершено. Найдено файлов с меткой {self.target_class_spin.value()}: {len(result_files)}")
        QMessageBox.information(self, "Результат", f"Найдено {len(result_files)} файлов.\nСписок отображён на правой панели.")

        if len(result_files) > 0 and not self.is_thumbnail_mode:
            self.log("Автоматическое переключение в режим миниатюр...")
            self.toggle_view_btn.setChecked(True)
            self.toggle_view_mode()
        elif len(result_files) > 0 and self.is_thumbnail_mode:
            self.generate_thumbnails_sync()
        else:
            self.log("Нет файлов для отображения миниатюр.")

    # --------------------------------------------------------
    # Сортировка
    # --------------------------------------------------------
    def sort_by_confidence(self):
        if not self.result_file_paths:
            return
        self.result_file_paths.sort(key=lambda x: x[1], reverse=True)
        self.results_list.blockSignals(True)
        self.results_list.clear()
        for file_path, conf in self.result_file_paths:
            item = QListWidgetItem(f"{os.path.basename(file_path)} (conf: {conf:.3f})")
            item.setData(Qt.UserRole, file_path)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.results_list.addItem(item)
        self.results_list.blockSignals(False)
        if self.is_thumbnail_mode and self.thumbnail_widgets:
            self._clear_thumbnail_grid()
            self.thumbnail_widgets.clear()
            self.generate_thumbnails_sync()
        self.log("Список отсортирован по убыванию уверенности.")

    # --------------------------------------------------------
    # Миниатюры (исправленная версия)
    # --------------------------------------------------------
    def generate_thumbnails_sync(self):
        if not self.result_file_paths:
            self.log("Нет файлов для генерации миниатюр.")
            return

        self.log(f"Начало синхронной генерации миниатюр для {len(self.result_file_paths)} файлов")
        self._clear_thumbnail_grid()
        self.thumbnail_widgets.clear()
        self.thumbnail_pixmaps.clear()

        total = len(self.result_file_paths)
        success_count = 0
        cache_hit_count = 0
        error_count = 0

        for idx, (file_path, conf) in enumerate(self.result_file_paths):
            percent = int((idx + 1) / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"Генерация миниатюр: {percent}% ({idx+1}/{total})")
            QApplication.processEvents()

            hash_name = hashlib.md5(file_path.encode('utf-8')).hexdigest() + ".png"
            thumb_path = os.path.join(TEMP_THUMB_DIR, hash_name)

            if os.path.exists(thumb_path):
                self.add_thumbnail_widget(file_path, thumb_path, conf)
                cache_hit_count += 1
                success_count += 1
                continue

            try:
                img = read_image_with_fallback_find(file_path)
                if img is None:
                    self.log(f"Не удалось загрузить: {os.path.basename(file_path)}")
                    error_count += 1
                    continue

                h, w = img.shape[:2]
                scale = 150 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(thumb_path, img_resized)
                self.add_thumbnail_widget(file_path, thumb_path, conf)
                success_count += 1

                if (idx + 1) % 50 == 0:
                    self.log(f"Сгенерировано миниатюр: {success_count}/{total} (кеш: {cache_hit_count})")
            except Exception as e:
                self.log(f"Ошибка для {os.path.basename(file_path)}: {e}")
                error_count += 1

        self.progress_bar.setFormat("Готово")
        self.log(f"Генерация миниатюр завершена. Успешно: {success_count}, кэш: {cache_hit_count}, ошибок: {error_count}")
        self.relayout_thumbnails()
        self.thumbnail_container.adjustSize()
        self.scroll_area.update()

    def add_thumbnail_widget(self, file_path, thumb_path, confidence):
        pixmap = QPixmap(thumb_path)
        if pixmap.isNull():
            self.log(f"Ошибка загрузки миниатюры для {os.path.basename(file_path)}")
            return

        self.thumbnail_pixmaps[file_path] = pixmap

        container = QFrame()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        cb = QCheckBox()
        cb.setChecked(False)
        cb.setFocusPolicy(Qt.NoFocus)
        cb.stateChanged.connect(lambda state, fp=file_path: self.on_thumbnail_checkbox_changed(fp, state))
        layout.addWidget(cb, alignment=Qt.AlignTop | Qt.AlignHCenter)

        # Используем ClickableLabel вместо обычного QLabel
        label = ClickableLabel()
        label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        label.setToolTip(f"{file_path}\nУверенность: {confidence:.3f}")
        label.set_file_path(file_path)
        label.set_open_callback(self.open_file)
        layout.addWidget(label)

        name_label = QLabel(os.path.basename(file_path))
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setWordWrap(True)
        layout.addWidget(name_label)

        conf_label = QLabel(f"conf: {confidence:.3f}")
        conf_label.setAlignment(Qt.AlignCenter)
        conf_label.setStyleSheet("QLabel { font-size: 10px; color: #888; }")
        layout.addWidget(conf_label)

        container.setFrameShape(QFrame.Box)
        container.setMaximumSize(180, 240)

        self.thumbnail_widgets[file_path] = (cb, label, container)

        row = len(self.thumbnail_widgets) // 4
        col = len(self.thumbnail_widgets) % 4
        self.thumbnail_grid.addWidget(container, row, col)
        self.thumbnail_grid.update()
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _clear_thumbnail_grid(self):
        for i in reversed(range(self.thumbnail_grid.count())):
            widget = self.thumbnail_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.thumbnail_widgets.clear()
        self.thumbnail_pixmaps.clear()

    def on_thumbnail_container_resize(self, event):
        if self.thumbnail_widgets:
            self.relayout_thumbnails()
        event.accept()

    def relayout_thumbnails(self):
        if not self.thumbnail_widgets:
            return
        width = self.thumbnail_container.width()
        item_width = 190
        cols = max(1, width // item_width)
        rows = (len(self.thumbnail_widgets) + cols - 1) // cols
        item_height = 240
        total_height = rows * item_height + 20
        self.thumbnail_container.setMinimumHeight(total_height)
        for i in reversed(range(self.thumbnail_grid.count())):
            widget = self.thumbnail_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        for idx, (file_path, (cb, label, container)) in enumerate(self.thumbnail_widgets.items()):
            row = idx // cols
            col = idx % cols
            self.thumbnail_grid.addWidget(container, row, col)
        self.thumbnail_grid.update()

    # --------------------------------------------------------
    # Переключение режима
    # --------------------------------------------------------
    def toggle_view_mode(self):
        if self.scan_thread and self.scan_thread.isRunning():
            self.log("Нельзя переключить режим во время сканирования. Подождите окончания.")
            self.toggle_view_btn.setChecked(not self.toggle_view_btn.isChecked())
            return

        self.is_thumbnail_mode = self.toggle_view_btn.isChecked()
        if self.is_thumbnail_mode:
            self.stacked_view.setCurrentIndex(1)
            self.toggle_view_btn.setText("Режим: список")
            self.log("Переключение в режим миниатюр")
            self.sync_selection_to_thumbnails()
            if self.result_file_paths and not self.thumbnail_widgets:
                self.generate_thumbnails_sync()
            else:
                self.relayout_thumbnails()
        else:
            self.stacked_view.setCurrentIndex(0)
            self.toggle_view_btn.setText("Режим: миниатюры")
            self.log("Переключение в режим списка")
            self.sync_selection_to_list()

    def sync_selection_to_thumbnails(self):
        selected_set = set()
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.checkState() == Qt.Checked:
                file_path = item.data(Qt.UserRole)
                selected_set.add(file_path)
        for file_path, (cb, _, _) in self.thumbnail_widgets.items():
            cb.blockSignals(True)
            cb.setChecked(file_path in selected_set)
            cb.blockSignals(False)

    def sync_selection_to_list(self):
        if not self.thumbnail_widgets:
            return
        thumb_checked = {fp: cb.isChecked() for fp, (cb, _, _) in self.thumbnail_widgets.items()}
        self.results_list.blockSignals(True)
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            fp = item.data(Qt.UserRole)
            if fp in thumb_checked:
                item.setCheckState(Qt.Checked if thumb_checked[fp] else Qt.Unchecked)
        self.results_list.blockSignals(False)
        self.update_select_all_state()

    def on_thumbnail_checkbox_changed(self, file_path, state):
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.data(Qt.UserRole) == file_path:
                item.setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
                break
        self.update_select_all_state()

    def update_select_all_state(self):
        total = self.results_list.count()
        if total == 0:
            self.select_all_checkbox.blockSignals(True)
            self.select_all_checkbox.setChecked(False)
            self.select_all_checkbox.blockSignals(False)
            return

        checked_count = 0
        for i in range(total):
            if self.results_list.item(i).checkState() == Qt.Checked:
                checked_count += 1

        new_checked = (checked_count == total)
        self.select_all_checkbox.blockSignals(True)
        self.select_all_checkbox.setChecked(new_checked)
        self.select_all_checkbox.blockSignals(False)

    def toggle_select_all(self, state):
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        self.results_list.blockSignals(True)
        for i in range(self.results_list.count()):
            self.results_list.item(i).setCheckState(check_state)
        self.results_list.blockSignals(False)
        for _, (cb, _, _) in self.thumbnail_widgets.items():
            cb.blockSignals(True)
            cb.setChecked(state == Qt.Checked)
            cb.blockSignals(False)

    def get_selected_files(self):
        selected = []
        for i in range(self.results_list.count()):
            item = self.results_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.data(Qt.UserRole))
        return selected

    # --------------------------------------------------------
    # Копирование/перемещение
    # --------------------------------------------------------
    def browse_target_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите целевую папку")
        if folder:
            self.target_folder_edit.setText(folder)
            self.log(f"Выбрана целевая папка: {folder}")

    def copy_selected_files(self):
        self._copy_move_files(copy=True)

    def move_selected_files(self):
        self._copy_move_files(copy=False)

    def _copy_move_files(self, copy=True):
        target_dir = self.target_folder_edit.text().strip()
        if not target_dir:
            QMessageBox.warning(self, "Нет папки", "Укажите целевую папку.")
            return
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
                self.log(f"Создана целевая папка: {target_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось создать папку: {e}")
                return

        selected = self.get_selected_files()
        if not selected:
            QMessageBox.information(self, "Нет файлов", "Не выбрано ни одного файла.")
            return

        operation = "Копирование" if copy else "Перемещение"
        self.log(f"{operation} {len(selected)} файлов в {target_dir}")

        success = 0
        errors = 0
        for src in selected:
            dst = os.path.join(target_dir, os.path.basename(src))
            try:
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
                success += 1
                self.log(f"{'Скопирован' if copy else 'Перемещён'}: {os.path.basename(src)}")
            except Exception as e:
                self.log(f"Ошибка {os.path.basename(src)}: {e}")
                errors += 1

        QMessageBox.information(self, "Результат", f"{operation} завершено.\nУспешно: {success}\nОшибок: {errors}")
        self.log(f"{operation} завершено. Успешно: {success}, ошибок: {errors}")

        if not copy:
            new_result_paths = []
            for fp, conf in self.result_file_paths:
                if fp not in selected:
                    new_result_paths.append((fp, conf))
            self.result_file_paths = new_result_paths

            for i in range(self.results_list.count() - 1, -1, -1):
                item = self.results_list.item(i)
                if item.data(Qt.UserRole) in selected:
                    self.results_list.takeItem(i)

            for fp in selected:
                if fp in self.thumbnail_widgets:
                    _, _, container = self.thumbnail_widgets[fp]
                    container.deleteLater()
                    del self.thumbnail_widgets[fp]
                    if fp in self.thumbnail_pixmaps:
                        del self.thumbnail_pixmaps[fp]

            if self.is_thumbnail_mode and self.thumbnail_widgets:
                self.relayout_thumbnails()
            self.update_select_all_state()

    def closeEvent(self, event):
        self.log("Закрытие приложения, остановка потоков...")
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.cancel()
            self.scan_thread.wait(2000)
        self._clear_temp_thumbs()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FindImagesWindow()
    window.show()
    sys.exit(app.exec_())