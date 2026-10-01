# layout_dataset.py
from import_libs_internal import *
from import_libs_methods_ui import setup_layout_dataset_ui


class Labeler(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Labeler (YOLO format)")

        # Данные (ленивая загрузка)
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Карта {путь_к_изображению: путь_к_файлу_меток}.
        # Заполняется при загрузке из YAML (там .txt лежат НЕ рядом с картинкой).
        self._yaml_label_map = {}

        self.img_w = 0
        self.img_h = 0
        self.all_annotations = []          # список аннотаций на каждое изображение (лениво)
        self.current_annotations = []      # аннотации текущего изображения

        # Для работы с YAML (маски не нужны)
        self.mask_folder = None
        self.loaded_masks = []
        self.class_colors = {}
        self.mask_opacity = 0.5

        self._updating_selection = False

        # UI
        setup_layout_dataset_ui(self)

        # Настройка SmartGraphicsView – используем унифицированные аннотации
        self.image_view.set_callbacks(
            on_rect_drawn=self.on_rect_drawn,
            on_display_update=self.update_image_display,
            on_reset_tool=self.reset_drawing_tool,
            on_annotation_modified=self.on_annotation_modified,
            on_selection_changed=self.on_selection_changed,
            on_log=self.log
        )

        # Сигналы
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.load_images.connect(self.load_images)
        self.save_button.clicked.connect(self.save_labels)
        self.add_rect_button.clicked.connect(self.toggle_drawing_mode)
        self.delete_button.clicked.connect(self.delete_selected_object)

        self.btn_load_labels.clicked.connect(self.load_labels_folder)
        self.btn_load_yaml.clicked.connect(self.load_yaml)

        self.object_list.itemClicked.connect(self.on_object_selected_from_list)
        self.object_list.itemDoubleClicked.connect(self.on_object_double_clicked)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        self.object_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(self.show_object_context_menu)

        # Начальное состояние
        self.add_rect_button.setChecked(True)
        # Включаем режим рисования только если уже есть картинки.
        # Если картинок нет — не сбрасываем галочку, а просто ждём загрузки.
        if self.image_paths:
            self.toggle_drawing_mode(True)
        self.update_navigation_state()
        self.log_widget.setVisible(False)

        settings.settings_changed.connect(self.on_global_settings_changed)

    # ----------------------------------------------------------------------
    #  Логирование
    # ----------------------------------------------------------------------
    def log(self, message):
        self.log_widget.log(message)
        print(message)

    # ----------------------------------------------------------------------
    #  Ленивая загрузка изображений (единая точка входа)
    # ----------------------------------------------------------------------
    def _load_images(self, source, label_map=None):
        """
        source    — папка (str) или список путей.
        label_map — опциональная карта {img_path: label_path} (YAML-режим).
        """
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
            self.nav_widget.set_current_index(0, 0)
            return

        if not all_files:
            self.log("No images loaded.")
            self.nav_widget.set_current_index(0, 0)
            return

        resize_enabled = self.nav_widget._resize_cb.isChecked()
        max_side = 1024

        # --- Останавливаем предыдущий кэш ---
        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None

        # --- Создаём новый кэш ---
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
        self.current_annotations = []
        self.all_annotations = [None] * len(all_files)
        self._yaml_label_map = dict(label_map) if label_map else {}

        self.image_view.set_annotations([], 0, 0)
        self.image_view.set_selected_index(-1)
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

        # Кнопка рисования была снята при пустом старте — включаем её снова,
        # чтобы режим рисования работал как раньше.
        if not self.add_rect_button.isChecked():
            self.add_rect_button.setChecked(True)
            self.toggle_drawing_mode(True)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.log(f"Loading folder: {folder}")
        self._load_images(folder)

    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.log(f"Loading {len(file_paths)} images...")
        self._load_images(file_paths)

    def reload_current_images(self):
        if not self.image_paths:
            self.log("Нет загруженных изображений для перезагрузки.")
            return
        self.log("Перезагрузка изображений с новыми настройками ресайза...")
        # Сохраняем YAML-карту, если работали в YAML-режиме.
        label_map = self._yaml_label_map if self._yaml_label_map else None
        self._load_images(list(self.image_paths), label_map=label_map)
        self.log(f"Перезагружено {len(self.image_paths)} изображений.")

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
        self.all_annotations = []
        self.current_annotations = []
        self.current_index = 0
        self._yaml_label_map = {}
        self.image_view.set_annotations([], 0, 0)
        self.image_view.set_selected_index(-1)
        self.update_image_display()
        self.info_label.setText("No image")
        self.object_list.clear()
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    def closeEvent(self, event):
        # Останавливаем фоновый пул, чтобы процесс не висел
        # на незавершённых задачах загрузки.
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        super().closeEvent(event)

    # ----------------------------------------------------------------------
    #  Загрузка меток / YAML
    # ----------------------------------------------------------------------
    def load_labels_folder(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений", "Сначала загрузите изображения.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
        if not folder:
            return
        self.log(f"Загрузка меток из {folder}")

        # Переключаемся из YAML-режима в «метки из папки»
        self._yaml_label_map = {}

        self.all_annotations = [[] for _ in self.image_paths]
        count = 0
        for i, img_path in enumerate(self.image_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(folder, base + '.txt')
            if os.path.exists(label_path):
                ann = load_annotations(label_path, 1, 1)
                self.all_annotations[i] = ann
                if ann:
                    count += 1
        self.display_current_image()
        self.log(f"Загружены метки для {count} изображений.")

    def _collect_yaml_items(self, yaml_path):
        """
        Собирает изображения из YAML, НЕ загружая их в память.
        Возвращает (image_paths, {img_path: label_path}).
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        base_path = config.get('path', os.path.dirname(yaml_path))
        base_path = os.path.abspath(base_path)

        items = []
        for split in ['train', 'val', 'test']:
            split_val = config.get(split)
            if not split_val:
                continue
            if isinstance(split_val, str):
                split_dirs = [split_val]
            elif isinstance(split_val, list):
                split_dirs = split_val
            else:
                continue

            for rel_dir in split_dirs:
                img_dir = os.path.join(base_path, rel_dir)
                if not os.path.isdir(img_dir):
                    continue
                for fname in os.listdir(img_dir):
                    if not fname.lower().endswith(
                            ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')):
                        continue
                    img_path = os.path.join(img_dir, fname)
                    stem = os.path.splitext(fname)[0]

                    label_path = None
                    for lbl_dir in [
                        os.path.join(base_path, 'labels', split),
                        os.path.join(base_path, 'labels', os.path.basename(rel_dir)),
                        os.path.join(base_path, 'labels'),
                        os.path.join(os.path.dirname(img_dir), 'labels'),
                    ]:
                        candidate = os.path.join(lbl_dir, stem + '.txt')
                        if os.path.exists(candidate):
                            label_path = candidate
                            break

                    items.append((img_path, label_path))

        image_paths = [p for p, _ in items]
        label_map = {p: lp for p, lp in items if lp is not None}
        return image_paths, label_map

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML file", "", "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return
        try:
            image_paths, label_map = self._collect_yaml_items(file_path)
            if not image_paths:
                QMessageBox.warning(self, "Ошибка", "Не найдено изображений в YAML.")
                return

            self._load_images(image_paths, label_map=label_map)
            self.log(f"Загружено {len(image_paths)} изображений из YAML.")
            QMessageBox.information(self, "Успех",
                                    f"Загружено {len(image_paths)} изображений.")
        except Exception as e:
            self.log(f"Ошибка загрузки YAML: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить YAML:\n{e}")

    # ----------------------------------------------------------------------
    #  Отображение текущего изображения
    # ----------------------------------------------------------------------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        # --- Синхронно получаем элемент (LRU-кэш вернёт готовый или загрузит) ---
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Не удалось загрузить снимок: {self.image_paths[idx]}")
            return

        img = item["image"]
        gray = item["gray"]
        self.img_h, self.img_w = img.shape[:2]

        self.update_histogram(gray)

        # --- Ленивая инициализация аннотаций для текущего индекса ---
        if (idx >= len(self.all_annotations)) or (self.all_annotations[idx] is None):
            label_path = self._yaml_label_map.get(self.image_paths[idx]) \
                if self._yaml_label_map else None
            if label_path and os.path.exists(label_path):
                anns = load_annotations(label_path, self.img_w, self.img_h)
            else:
                # .txt рядом с изображением уже загружен лоадером
                anns = list(item.get("annotations", []) or [])
            while len(self.all_annotations) <= idx:
                self.all_annotations.append(None)
            self.all_annotations[idx] = anns

        self.current_annotations = list(self.all_annotations[idx])

        # Передаём в smart_view все унифицированные аннотации
        self.image_view.set_annotations(self.current_annotations, self.img_w, self.img_h)
        self.image_view.set_selected_index(-1)

        update_annotation_list(self.object_list, self.current_annotations,
                               self.img_w, self.img_h)

        self.update_image_display()
        self.info_label.setText(f"Image {idx + 1} of {total}")
        self.nav_widget.set_current_index(idx, total)

        if self.add_rect_button.isChecked():
            self.image_view.set_drawing_tool("rect")
        else:
            self.image_view.set_drawing_tool(None)

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    def update_image_display(self):
        if self._lazy_cache is None or not self.image_paths:
            return
        item = self._lazy_cache.get(self.current_index)
        if item is None:
            return

        img = item["image"].copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        thickness, font_scale, font_thickness, _ = get_display_params(img.shape)
        color_normal = settings.get_color('annotation')
        color_selected = settings.get_color('selected')
        selected_idx = self.object_list.currentRow()

        for i, ann in enumerate(self.current_annotations):
            typ = ann[0]
            color = color_selected if i == selected_idx else color_normal
            if typ == 'detect':
                _, cls, cx, cy, bw, bh = ann
                x = int((cx - bw / 2) * self.img_w)
                y = int((cy - bh / 2) * self.img_h)
                x2 = x + int(bw * self.img_w)
                y2 = y + int(bh * self.img_h)
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
            elif typ in ('obb', 'segment'):
                _, cls, points = ann
                pts = []
                for j in range(0, len(points), 2):
                    px = int(points[j] * self.img_w)
                    py = int(points[j + 1] * self.img_h)
                    pts.append([px, py])
                if len(pts) >= 2:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(img, [pts], isClosed=True, color=color,
                                  thickness=thickness)

        pixmap = numpy_to_qpixmap(img)
        self.image_view.set_pixmap(pixmap)

    def update_histogram(self, gray_img):
        self.hist_ax.clear()
        self.hist_ax.hist(gray_img.ravel(), bins=256, range=(0, 256),
                          color='black', alpha=0.7)
        self.hist_ax.set_title("Grayscale Histogram")
        self.hist_ax.set_xlabel("Pixel intensity")
        self.hist_ax.set_ylabel("Frequency")
        self.hist_canvas.draw()

    # ----------------------------------------------------------------------
    #  Синхронизация текущих аннотаций в хранилище (ленивое)
    # ----------------------------------------------------------------------
    def _sync_current_annotations_to_storage(self):
        idx = self.current_index
        while len(self.all_annotations) <= idx:
            self.all_annotations.append(None)
        self.all_annotations[idx] = list(self.current_annotations)

    # ----------------------------------------------------------------------
    #  Навигация
    # ----------------------------------------------------------------------
    def prev_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.display_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.display_current_image()

    def goto_image(self, page_num):
        if not self.image_paths:
            return
        total = len(self.image_paths)
        page_num = max(1, min(page_num, total))
        new_idx = page_num - 1
        if new_idx != self.current_index:
            self.current_index = new_idx
            self.display_current_image()
            self.update_navigation_state()

    def update_navigation_state(self):
        has_images = len(self.image_paths) > 0
        self.save_button.setEnabled(has_images)
        self.add_rect_button.setEnabled(has_images)
        self.delete_button.setEnabled(has_images and len(self.current_annotations) > 0)
        self.nav_widget.set_navigation_enabled(has_images)

    # ----------------------------------------------------------------------
    #  Рисование новых прямоугольников (detect)
    # ----------------------------------------------------------------------
    def toggle_drawing_mode(self, checked):
        if not self.image_paths:
            self.add_rect_button.setChecked(False)
            self.log("No image loaded. Cannot enable drawing mode.")
            return
        if checked:
            if self.image_view.edit_mode:
                self.image_view.set_edit_mode(False)
            self.image_view.set_selected_index(-1)
            self.image_view.set_drawing_tool("rect")
            self.log("Drawing mode ON. Click and drag to add rectangle.")
        else:
            self.image_view.set_drawing_tool(None)
            self.log("Drawing mode OFF (panning mode).")

    def on_rect_drawn(self, rect):
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return
        cx = (x1 + x2) / 2.0 / self.img_w
        cy = (y1 + y2) / 2.0 / self.img_h
        bw = w / self.img_w
        bh = h / self.img_h
        new_ann = ('detect', 0, cx, cy, bw, bh)
        self.current_annotations.append(new_ann)
        self._sync_current_annotations_to_storage()
        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        update_annotation_list(self.object_list, self.current_annotations,
                               self.img_w, self.img_h)
        self.update_image_display()
        self.log(f"Added rectangle: class=0, center=({cx:.3f},{cy:.3f}), "
                 f"size=({bw:.3f},{bh:.3f})")

    def reset_drawing_tool(self):
        if self.add_rect_button.isChecked():
            self.add_rect_button.setChecked(False)
            self.toggle_drawing_mode(False)
        self.image_view.set_selected_index(-1)
        self.log("Drawing mode OFF (right click).")

    # ----------------------------------------------------------------------
    #  Редактирование аннотаций (общее для detect, obb, segment)
    # ----------------------------------------------------------------------
    def on_annotation_modified(self, idx, new_ann):
        """Новая аннотация уже в унифицированном формате."""
        if idx < 0 or idx >= len(self.current_annotations):
            return
        self.current_annotations[idx] = new_ann
        self._sync_current_annotations_to_storage()
        update_annotation_list(self.object_list, self.current_annotations,
                               self.img_w, self.img_h)
        self.update_image_display()
        self.log(f"Modified object {idx + 1}")

    def on_selection_changed(self, idx):
        """Вызывается при выделении через smart_view."""
        if self._updating_selection:
            return
        self._updating_selection = True
        try:
            if idx != -1 and self.add_rect_button.isChecked():
                self.add_rect_button.setChecked(False)
                self.toggle_drawing_mode(False)
            if idx != -1:
                self.object_list.blockSignals(True)
                self.object_list.setCurrentRow(idx)
                self.object_list.blockSignals(False)
            self.update_image_display()
        finally:
            self._updating_selection = False

    # ----------------------------------------------------------------------
    #  Управление объектами из списка
    # ----------------------------------------------------------------------
    def on_object_selected_from_list(self, item):
        if self._updating_selection:
            return
        self._updating_selection = True
        try:
            idx = item.data(Qt.UserRole)
            if idx is not None and 0 <= idx < len(self.current_annotations):
                self.image_view.set_selected_index(idx)
                self.log(f"Selected object {idx + 1}")
                self.update_image_display()
                if not self.image_view.edit_mode:
                    self.image_view.set_edit_mode(True)
        finally:
            self._updating_selection = False

    def delete_selected_object(self):
        selected_row = self.object_list.currentRow()
        if 0 <= selected_row < len(self.current_annotations):
            self.delete_object_by_index(selected_row)
        else:
            self.log("No object selected for deletion.")

    def delete_object_by_index(self, idx):
        if idx < 0 or idx >= len(self.current_annotations):
            return
        del self.current_annotations[idx]
        self._sync_current_annotations_to_storage()
        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        if self.image_view.selected_index == idx:
            self.image_view.set_selected_index(-1)
            self.image_view.set_edit_mode(False)
        elif self.image_view.selected_index > idx:
            self.image_view.set_selected_index(self.image_view.selected_index - 1)
        update_annotation_list(self.object_list, self.current_annotations,
                               self.img_w, self.img_h)
        self.update_image_display()
        self.log(f"Deleted object {idx + 1}")

    def on_object_double_clicked(self, item):
        # При двойном клике выключаем режим рисования, если он активен
        if self.add_rect_button.isChecked():
            self.add_rect_button.setChecked(False)
            self.toggle_drawing_mode(False)

        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.current_annotations):
            return
        ann = self.current_annotations[idx]
        typ = ann[0]

        if typ != 'detect':
            # Для OBB и segment меняем только class ID
            old_cls = ann[1]
            new_cls_str, ok = QInputDialog.getText(
                self, "Change Class ID",
                f"Enter new class ID (integer >= 0):\nCurrent ID = {old_cls}",
                text=str(old_cls)
            )
            if ok and new_cls_str:
                try:
                    new_cls = int(new_cls_str.strip())
                    if new_cls < 0:
                        raise ValueError
                    new_ann = (typ, new_cls, ann[2])
                    self.current_annotations[idx] = new_ann
                    self._sync_current_annotations_to_storage()
                    self.image_view.set_annotations(self.current_annotations,
                                                    self.img_w, self.img_h)
                    update_annotation_list(self.object_list, self.current_annotations,
                                           self.img_w, self.img_h)
                    self.update_image_display()
                    self.log(f"Changed object {idx + 1} class to {new_cls}")
                except ValueError:
                    QMessageBox.warning(self, "Error",
                                        "Invalid class ID. Enter a non-negative integer.")
            return

        # Для detect – меняем класс
        _, cls, cx, cy, w, h = ann
        new_cls_str, ok = QInputDialog.getText(
            self, "Change Class ID",
            f"Enter new class ID (integer >= 0):\nCurrent ID = {cls}",
            text=str(cls)
        )
        if ok and new_cls_str:
            new_cls_str = new_cls_str.strip()
            if not new_cls_str:
                QMessageBox.warning(self, "Error", "Class ID cannot be empty.")
                return
            try:
                new_cls = int(new_cls_str)
                if new_cls < 0:
                    raise ValueError
            except ValueError:
                QMessageBox.warning(self, "Error",
                                    "Invalid class ID. Enter a non-negative integer.")
                return
            new_ann = ('detect', new_cls, cx, cy, w, h)
            self.current_annotations[idx] = new_ann
            self._sync_current_annotations_to_storage()
            self.image_view.set_annotations(self.current_annotations,
                                            self.img_w, self.img_h)
            update_annotation_list(self.object_list, self.current_annotations,
                                   self.img_w, self.img_h)
            self.update_image_display()
            self.log(f"Changed object {idx + 1} class to {new_cls}")

    def show_object_context_menu(self, pos):
        item = self.object_list.itemAt(pos)
        if item is not None:
            idx = item.data(Qt.UserRole)
            menu = QMenu()
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self.delete_object_by_index(idx))
            menu.addAction(delete_action)
            menu.exec_(self.object_list.mapToGlobal(pos))

    # ----------------------------------------------------------------------
    #  Сохранение аннотаций
    # ----------------------------------------------------------------------
    def save_labels(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        img_path = self.image_paths[self.current_index]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        success = save_annotations(self.current_annotations, txt_path,
                                   self.img_w, self.img_h)
        if success:
            self.log(f"Saved {len(self.current_annotations)} labels to {txt_path}")
            QMessageBox.information(self, "Save", f"Labels saved to {txt_path}")
        else:
            self.log(f"Error saving labels to {txt_path}")
            QMessageBox.critical(self, "Error", f"Failed to save labels to {txt_path}")

    # ----------------------------------------------------------------------
    #  Обработка клавиш
    # ----------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.image_view.edit_mode:
                self.image_view.set_edit_mode(False)
            elif self.add_rect_button.isChecked():
                self.reset_drawing_tool()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # Не выключаем edit_mode при отпускании Ctrl
        super().keyReleaseEvent(event)

    # ----------------------------------------------------------------------
    #  Вспомогательные методы
    # ----------------------------------------------------------------------
    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    def toggle_histogram(self, checked):
        self.hist_container.setVisible(checked)
        self.toggle_hist_btn.setText("Скрыть гистограмму" if checked else "Показать гистограмму")

    def on_global_settings_changed(self, new_settings=None):
        if self.image_paths:
            self.update_image_display()
            update_annotation_list(self.object_list, self.current_annotations,
                                   self.img_w, self.img_h)

    def showEvent(self, event):
        self.image_view.setFocus()
        super().showEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Labeler()
    window.show()
    sys.exit(app.exec_())