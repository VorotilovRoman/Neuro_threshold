# viewing_dataset.py
from collections import OrderedDict

from import_libs_internal import *
from import_libs_methods_ui import setup_viewing_dataset_ui


class ViewingDataset(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("View Dataset: YOLO annotations (detect, OBB, segment)")

        # --- Данные (ленивая загрузка) ---
        self.image_paths = []
        self.current_idx = 0
        self._lazy_cache = None

        # Аннотации — ленивый список: [None] * N
        self.annotations = []
        self.current_annotations = []
        self.current_img_w = 0
        self.current_img_h = 0
        self.current_folder = None
        self.selected_index = -1

        # --- Маски: ленивый LRU ---
        self.mask_folder = None             # папка масок (режим «load masks folder»)
        self._label_folder = None           # папка меток (режим «load labels folder»)
        self._yaml_label_map = {}           # {img_path: label_path}
        self._yaml_mask_map = {}            # {img_path: mask_path}
        self._mask_cache = OrderedDict()    # idx -> numpy mask | None
        self._mask_cache_max = 16

        self.class_colors = {}
        self.mask_opacity = 0.5

        setup_viewing_dataset_ui(self)

        # --- Сигналы ---
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.load_images.connect(self.load_images)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.btn_save.clicked.connect(self.save_current_annotations)
        self.btn_delete.clicked.connect(self.delete_selected_object)
        self.btn_load_labels.clicked.connect(self.load_labels_folder)
        self.btn_load_masks.clicked.connect(self.load_masks_folder)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.coord_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.coord_list.customContextMenuRequested.connect(self.show_object_context_menu)
        self.coord_list.itemClicked.connect(self.on_object_selected)
        self.coord_list.itemDoubleClicked.connect(self.on_object_double_clicked)
        self.btn_load_yaml.clicked.connect(self.load_yaml)
        settings.settings_changed.connect(self.on_settings_changed)
        self.toggle_log_btn.clicked.connect(self.toggle_log)

        self.update_navigation_state()

    # ------------------------------------------------------------------
    # Логирование / сигналы
    # ------------------------------------------------------------------
    def log(self, message):
        self.log_widget.log(message)
        print(message)

    def on_settings_changed(self, new_settings):
        if self.image_paths:
            self.show_current_image()

    def update_navigation_state(self):
        has_images = len(self.image_paths) > 0
        self.nav_widget.set_navigation_enabled(has_images)
        self.btn_save.setEnabled(has_images)
        self.btn_delete.setEnabled(has_images and len(self.current_annotations) > 0)

    def closeEvent(self, event):
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Ленивая загрузка изображений (единая точка входа)
    # ------------------------------------------------------------------
    def _load_images(self, source, label_map=None, mask_map=None,
                     label_folder=None, mask_folder=None):
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
            self.log("Не загружено ни одного изображения.")
            self._clear_view()
            return

        if not all_files:
            self.log("Не загружено ни одного изображения.")
            self._clear_view()
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
        self.current_idx = 0
        self.selected_index = -1
        self.current_annotations = []
        self.annotations = [None] * len(all_files)

        self._yaml_label_map = dict(label_map) if label_map else {}
        self._yaml_mask_map = dict(mask_map) if mask_map else {}
        self._label_folder = label_folder
        self.mask_folder = mask_folder
        self._mask_cache = OrderedDict()
        self.class_colors = {}

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
        self.show_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_idx, total)

    def _clear_view(self):
        self.viewer.set_pixmap(numpy_to_qpixmap(None))
        self.coord_list.clear()
        self.info_label.setText("No images")
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.current_folder = folder
        self.log(f"Selected folder: {folder}")
        self._load_images(folder, mask_folder=self.mask_folder)

    def load_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.current_folder = None
        self._load_images(file_paths, mask_folder=self.mask_folder)

    def load_images_from_source(self, source):
        """Совместимость со старым API — проксирует в _load_images."""
        self._load_images(source, mask_folder=self.mask_folder)

    # ------------------------------------------------------------------
    # Ленивая инициализация аннотаций и масок
    # ------------------------------------------------------------------
    def _get_annotations(self, idx):
        if idx < len(self.annotations) and self.annotations[idx] is not None:
            return self.annotations[idx]

        ann = []
        img_path = self.image_paths[idx]

        if self._yaml_label_map:
            lp = self._yaml_label_map.get(img_path)
            if lp and os.path.exists(lp):
                ann = load_annotations(lp, 1, 1)
        elif self._label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lp = os.path.join(self._label_folder, base + '.txt')
            if os.path.exists(lp):
                ann = load_annotations(lp, 1, 1)
        else:
            # .txt рядом с картинкой уже подгружен лоадером
            item = self._lazy_cache.get(idx) if self._lazy_cache else None
            if item:
                ann = list(item.get("annotations", []) or [])

        while len(self.annotations) <= idx:
            self.annotations.append(None)
        self.annotations[idx] = ann
        return ann

    def _sync_current_annotations(self):
        idx = self.current_idx
        while len(self.annotations) <= idx:
            self.annotations.append(None)
        self.annotations[idx] = list(self.current_annotations)

    def _get_mask(self, idx):
        """Ленивая загрузка маски для индекса idx (LRU)."""
        if idx in self._mask_cache:
            self._mask_cache.move_to_end(idx)
            return self._mask_cache[idx]

        mask_path = None
        img_path = self.image_paths[idx]

        if self._yaml_mask_map:
            mask_path = self._yaml_mask_map.get(img_path)
        if mask_path is None and self.mask_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                cand = os.path.join(self.mask_folder, base + ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break

        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is not None and len(mask.shape) == 3:
                    mask = mask[:, :, 0]
            except Exception as e:
                self.log(f"Warning: could not load mask {mask_path}: {e}")

        # Догенерируем цвета для новых классов
        if mask is not None:
            for cls in np.unique(mask):
                c = int(cls)
                if c == 0 or c in self.class_colors:
                    continue
                hue = (c * 37) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                                     cv2.COLOR_HSV2BGR)[0][0]
                self.class_colors[c] = (int(color[0]), int(color[1]), int(color[2]))

        self._mask_cache[idx] = mask
        while len(self._mask_cache) > self._mask_cache_max:
            self._mask_cache.popitem(last=False)
        return mask

    # ------------------------------------------------------------------
    # Загрузка меток / масок (ленивые режимы)
    # ------------------------------------------------------------------
    def load_labels_folder(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений", "Сначала загрузите изображения.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
        if not folder:
            return
        self.log(f"Loading labels from: {folder}")

        # Переключаем источник аннотаций на папку
        self._label_folder = folder
        self._yaml_label_map = {}
        self.annotations = [None] * len(self.image_paths)
        self.current_annotations = []
        self.selected_index = -1

        # Быстрый подсчёт найденных .txt (без парсинга содержимого)
        found = 0
        for img_path in self.image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            if os.path.exists(os.path.join(folder, base + '.txt')):
                found += 1
        self.log(f"Found label files for {found} / {len(self.image_paths)} images.")
        self.show_current_image()

    def load_masks_folder(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений", "Сначала загрузите изображения.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with masks")
        if not folder:
            return
        self.mask_folder = folder
        self._yaml_mask_map = {}
        self._mask_cache = OrderedDict()
        self.class_colors = {}
        self.log(f"Masks folder set: {folder} (маски грузятся по мере просмотра).")
        self.show_current_image()

    def on_opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.show_current_image()

    def apply_mask_overlay(self, img, mask):
        overlay = img.copy()
        for cls, color in self.class_colors.items():
            overlay[mask == cls] = color
        return cv2.addWeighted(img, 1 - self.mask_opacity,
                               overlay, self.mask_opacity, 0)

    def update_info_label(self):
        total = len(self.image_paths)
        if total == 0:
            self.info_label.setText("No images")
        else:
            self.info_label.setText(f"Image {self.current_idx + 1} of {total}")

    # ------------------------------------------------------------------
    # Список аннотаций / отрисовка
    # ------------------------------------------------------------------
    def update_annotation_list(self, list_widget, annotations, img_w, img_h):
        list_widget.clear()
        for idx, ann in enumerate(annotations):
            if ann[0] == 'detect':
                _, cls, cx, cy, w, h = ann
                x = int((cx - w / 2) * img_w)
                y = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                text = f"{idx}: cls={cls}, rect=({x},{y},{x2},{y2})"
            elif ann[0] == 'obb':
                _, cls, points = ann
                text = f"{idx}: cls={cls}, OBB (4 points)"
            elif ann[0] == 'segment':
                _, cls, points = ann
                num_pts = len(points) // 2
                text = f"{idx}: cls={cls}, polygon ({num_pts} points)"
            else:
                text = f"{idx}: unknown type"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            list_widget.addItem(item)

    def draw_annotations_with_selection(self, img, annotations, selected_idx):
        if not annotations:
            return img
        img_copy = img.copy()
        thickness, font_scale, font_thickness, _ = get_display_params(img.shape)
        h, w = self.current_img_h, self.current_img_w

        color_normal = settings.get_color('annotation')
        color_selected = settings.get_color('selected')

        for i, ann in enumerate(annotations):
            color = color_selected if i == selected_idx else color_normal
            if ann[0] == 'detect':
                _, cls, cx, cy, bw, bh = ann
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                x2 = x + int(bw * w)
                y2 = y + int(bh * h)
                cv2.rectangle(img_copy, (x, y), (x2, y2), color, thickness)
            elif ann[0] in ('obb', 'segment'):
                _, cls, points = ann
                pts = []
                for j in range(0, len(points), 2):
                    px = int(points[j] * w)
                    py = int(points[j + 1] * h)
                    pts.append([px, py])
                if len(pts) >= 2:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(img_copy, [pts], isClosed=True,
                                  color=color, thickness=thickness)
        return img_copy

    # ------------------------------------------------------------------
    # Отображение текущего изображения
    # ------------------------------------------------------------------
    def show_current_image(self):
        if self._lazy_cache is None or not self.image_paths:
            return

        idx = self.current_idx
        total = len(self._lazy_cache)

        # --- Синхронно получаем элемент ---
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Failed to load image {self.image_paths[idx]}")
            return

        img = item["image"].copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self.current_img_w = img.shape[1]
        self.current_img_h = img.shape[0]

        # --- Ленивая инициализация аннотаций ---
        self.current_annotations = self._get_annotations(idx)

        # --- Ленивая маска ---
        mask = self._get_mask(idx)
        if mask is not None:
            if mask.shape[:2] != (self.current_img_h, self.current_img_w):
                mask = cv2.resize(mask, (self.current_img_w, self.current_img_h),
                                  interpolation=cv2.INTER_NEAREST)
            img = self.apply_mask_overlay(img, mask)

        annotated = self.draw_annotations_with_selection(
            img, self.current_annotations, self.selected_index
        )
        pixmap = numpy_to_qpixmap(annotated)
        if pixmap.isNull():
            self.log("Failed to convert image to QPixmap")
        else:
            self.viewer.set_pixmap(pixmap)

        self.update_annotation_list(self.coord_list, self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        self.update_info_label()
        self.update_navigation_state()
        self.nav_widget.set_current_index(idx, total)

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def save_current_annotations(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        img_path = self.image_paths[self.current_idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        success = save_annotations(self.current_annotations, txt_path,
                                   self.current_img_w, self.current_img_h)
        if success:
            self._sync_current_annotations()
            self.log(f"Saved {len(self.current_annotations)} annotations to {txt_path}")
            QMessageBox.information(self, "Save", f"Annotations saved to {txt_path}")
        else:
            self.log(f"Error saving annotations to {txt_path}")
            QMessageBox.critical(self, "Error", f"Failed to save to {txt_path}")

    # ------------------------------------------------------------------
    # Навигация
    # ------------------------------------------------------------------
    def prev_image(self):
        if not self.image_paths:
            return
        self.current_idx = (self.current_idx - 1) % len(self.image_paths)
        self.selected_index = -1
        self.show_current_image()

    def next_image(self):
        if not self.image_paths:
            return
        self.current_idx = (self.current_idx + 1) % len(self.image_paths)
        self.selected_index = -1
        self.show_current_image()

    def goto_image(self, page_num):
        if not self.image_paths:
            return
        total = len(self.image_paths)
        page_num = max(1, min(page_num, total))
        new_idx = page_num - 1
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self.selected_index = -1
            self.show_current_image()
            self.nav_widget.set_current_index(self.current_idx, total)
            self.update_navigation_state()

    def on_resize_mode_changed(self, state):
        if not self.image_paths:
            return
        reply = QMessageBox.question(
            self, "Resize Mode Changed",
            "Resize mode changed. To apply, you need to reload images.\n"
            "Do you want to reload images now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self._load_images(
                list(self.image_paths),
                label_map=self._yaml_label_map if self._yaml_label_map else None,
                mask_map=self._yaml_mask_map if self._yaml_mask_map else None,
                label_folder=self._label_folder,
                mask_folder=self.mask_folder,
            )
        else:
            if self._lazy_cache is not None:
                self._lazy_cache.shutdown()
                self._lazy_cache = None
            self.image_paths = []
            self.annotations = []
            self.current_annotations = []
            self.current_idx = 0
            self.selected_index = -1
            self._mask_cache = OrderedDict()
            self.class_colors = {}
            self.viewer.set_pixmap(numpy_to_qpixmap(None))
            self.coord_list.clear()
            self.nav_widget.set_current_index(0, 0)
            self.update_navigation_state()
            self.info_label.setText("No images")
            self.log("Resize mode changed, images cleared. Please load images again.")

    # ------------------------------------------------------------------
    # Работа со списком объектов
    # ------------------------------------------------------------------
    def on_object_selected(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None and idx != self.selected_index:
            self.selected_index = idx
            self.show_current_image()
            if 0 <= idx < len(self.current_annotations):
                self.log(f"Selected object {idx + 1}: {self.current_annotations[idx]}")

    def show_object_context_menu(self, pos):
        item = self.coord_list.itemAt(pos)
        if item is not None:
            idx = item.data(Qt.UserRole)
            menu = QMenu()
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(
                lambda: self.delete_annotation_by_index(idx))
            menu.addAction(delete_action)
            menu.exec_(self.coord_list.mapToGlobal(pos))

    def delete_annotation_by_index(self, idx):
        if 0 <= idx < len(self.current_annotations):
            new_ann = self.current_annotations[:idx] + self.current_annotations[idx + 1:]
            self.current_annotations = new_ann
            self._sync_current_annotations()
            new_sel = -1 if self.selected_index == idx else (
                self.selected_index - 1 if self.selected_index > idx
                else self.selected_index
            )
            self.selected_index = new_sel
            self.update_annotation_list(self.coord_list, self.current_annotations,
                                        self.current_img_w, self.current_img_h)
            self.show_current_image()
            self.log(f"Deleted object {idx + 1}")
            self.update_navigation_state()

    def delete_selected_object(self):
        current_row = self.coord_list.currentRow()
        if 0 <= current_row < len(self.current_annotations):
            self.delete_annotation_by_index(current_row)
        else:
            self.log("No object selected for deletion.")

    def on_object_double_clicked(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.current_annotations):
            return

        ann = self.current_annotations[idx]
        typ = ann[0]
        if typ == 'detect':
            _, cls, cx, cy, w, h = ann
        elif typ == 'obb':
            _, cls, points = ann
        elif typ == 'segment':
            _, cls, points = ann
        else:
            return

        new_cls_str, ok = QInputDialog.getText(
            self, "Изменение ID класса",
            f"Введите новый ID класса (целое число ≥ 0):\nТекущий ID = {cls}",
            text=str(cls)
        )
        if ok and new_cls_str:
            new_cls_str = new_cls_str.strip()
            if not new_cls_str:
                QMessageBox.warning(self, "Ошибка", "ID не может быть пустым.")
                return
            try:
                new_cls = int(new_cls_str)
                if new_cls < 0:
                    raise ValueError("ID не может быть отрицательным")
            except ValueError as e:
                QMessageBox.warning(self, "Ошибка",
                                    f"Некорректный ID: {e}\n"
                                    f"Введите целое неотрицательное число.")
                return

            if typ == 'detect':
                self.current_annotations[idx] = ('detect', new_cls, cx, cy, w, h)
            else:
                self.current_annotations[idx] = (typ, new_cls, points)

            self._sync_current_annotations()
            self.update_annotation_list(self.coord_list, self.current_annotations,
                                        self.current_img_w, self.current_img_h)
            self.show_current_image()
            self.log(f"Объекту {idx + 1} присвоен новый ID класса: {new_cls}")

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    # ------------------------------------------------------------------
    # YAML датасет: собираем пути/карты, но НЕ грузим картинки
    # ------------------------------------------------------------------
    def _collect_yaml_items_with_masks(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        base_path = config.get('path', '')
        if not base_path:
            base_path = os.path.dirname(yaml_path)
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

                    # --- label ---
                    label_path = None
                    for lbl_sub in [split, '']:
                        lbl_dir = (os.path.join(base_path, 'labels', lbl_sub)
                                   if lbl_sub else os.path.join(base_path, 'labels'))
                        cand = os.path.join(lbl_dir, stem + '.txt')
                        if os.path.exists(cand):
                            label_path = cand
                            break

                    # --- mask ---
                    mask_path = None
                    for mask_sub in [split, '']:
                        mask_dir = (os.path.join(base_path, 'masks', mask_sub)
                                    if mask_sub else os.path.join(base_path, 'masks'))
                        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                            cand = os.path.join(mask_dir, stem + ext)
                            if os.path.exists(cand):
                                mask_path = cand
                                break
                        if mask_path:
                            break

                    items.append((img_path, label_path, mask_path))

        image_paths = [p for p, _, _ in items]
        label_map = {p: lp for p, lp, _ in items if lp}
        mask_map = {p: mp for p, _, mp in items if mp}
        return image_paths, label_map, mask_map

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите YAML файл датасета", "", "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return

        try:
            image_paths, label_map, mask_map = self._collect_yaml_items_with_masks(file_path)
            if not image_paths:
                QMessageBox.warning(self, "Ошибка", "Не найдено изображений в YAML.")
                return

            self._load_images(
                image_paths,
                label_map=label_map,
                mask_map=mask_map,
                label_folder=None,
                mask_folder=None,
            )
            self.log(f"Загружено {len(image_paths)} изображений из YAML "
                     f"(метки: {len(label_map)}, маски: {len(mask_map)}).")
            QMessageBox.information(self, "Успех",
                                    f"Загружено {len(image_paths)} изображений.")
        except Exception as e:
            self.log(f"Ошибка загрузки YAML: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить YAML:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewingDataset()
    window.show()
    sys.exit(app.exec_())