from import_libs_internal import *
from import_libs_methods_ui import setup_viewing_dataset_ui

class ViewingDataset(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        print(f"[INIT] ViewingDataset.__init__ called, self={self}, parent={parent}")
        self.setWindowTitle("View Dataset: YOLO annotations (detect, OBB, segment)")

        self.image_paths = []
        self.annotations = []          # список списков аннотаций для каждого изображения
        self.current_idx = 0
        self.current_annotations = []  # аннотации текущего изображения в формате load_annotations
        self.current_img_w = 0
        self.current_img_h = 0
        self.current_folder = None
        self.selected_index = -1
        self.images = []
        self.gray_images = []
        self.mask_folder = None
        self.loaded_masks = []
        self.class_colors = {}
        self.mask_opacity = 0.5

        setup_viewing_dataset_ui(self)

        # Подключение сигналов
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

    # ---------- Загрузка изображений ----------
    def load_folder(self):
        print("\n[SLOT] select_folder called")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.current_folder = folder
        self.log(f"Selected folder: {folder}")
        self.load_images_from_source(folder)

    def load_images(self):
        print("\n[SLOT] load_images called")
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        )
        if not file_paths:
            return
        self.current_folder = None
        self.load_images_from_source(file_paths)

    def load_images_from_source(self, source):
        image_paths, images, gray_images, annotations = load_images_universal(
            source=source,
            require_annotations=False,
            resize_enabled=self.nav_widget._resize_cb.isChecked(),
            parent=self
        )
        if not image_paths:
            self.log("Не загружено ни одного изображения.")
            self.viewer.set_pixmap(numpy_to_qpixmap(None))
            self.coord_list.clear()
            self.info_label.setText("No images")
            self.update_navigation_state()
            self.nav_widget.set_current_index(0, 0)
            return

        self.image_paths = image_paths
        self.images = images
        self.gray_images = gray_images
        self.annotations = annotations if annotations is not None else [[] for _ in image_paths]
        self.current_idx = 0
        self.loaded_masks = [None] * len(self.image_paths)

        self.show_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_idx, len(self.image_paths))

    # ---------- Загрузка аннотаций и масок ----------
    def load_labels_folder(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений", "Сначала загрузите изображения.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
        if not folder:
            return
        self.log(f"Loading labels from: {folder}")
        self.annotations = [[] for _ in self.image_paths]
        for i, img_path in enumerate(self.image_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(folder, base + '.txt')
            if os.path.exists(label_path):
                # Размеры изображения для load_annotations не важны, но передаём текущие
                w = self.images[i].shape[1] if self.images else 1
                h = self.images[i].shape[0] if self.images else 1
                ann = load_annotations(label_path, w, h)
                self.annotations[i] = ann
        self.log(f"Loaded annotations for {sum(1 for ann in self.annotations if ann)} images.")
        self.show_current_image()

    def load_masks_folder(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений", "Сначала загрузите изображения.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select folder with masks")
        if not folder:
            return
        self.mask_folder = folder
        self.log(f"Masks folder: {folder}")

        self.loaded_masks = [None] * len(self.image_paths)
        all_classes = set()
        for i, img_path in enumerate(self.image_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                candidate = os.path.join(folder, base + ext)
                if os.path.exists(candidate):
                    mask = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                    if mask is not None:
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]
                        self.loaded_masks[i] = mask
                        all_classes.update(np.unique(mask))
                    else:
                        self.log(f"Warning: Could not read mask {candidate}")
                    break

        self.class_colors = {}
        for cls in sorted(all_classes):
            if cls == 0:
                continue
            hue = (cls * 37) % 180
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            self.class_colors[cls] = (int(color[0]), int(color[1]), int(color[2]))

        self.log(f"Loaded masks for {sum(1 for m in self.loaded_masks if m is not None)} images. "
                 f"Classes found: {sorted(self.class_colors.keys())}")
        self.show_current_image()

    def on_opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.show_current_image()

    def apply_mask_overlay(self, img, mask):
        overlay = img.copy()
        for cls, color in self.class_colors.items():
            overlay[mask == cls] = color
        result = cv2.addWeighted(img, 1 - self.mask_opacity, overlay, self.mask_opacity, 0)
        return result

    def update_info_label(self):
        total = len(self.image_paths)
        if total == 0:
            self.info_label.setText("No images")
        else:
            self.info_label.setText(f"Image {self.current_idx + 1} of {total}")

    # ---------- Обновление списка аннотаций ----------
    def update_annotation_list(self, list_widget, annotations, img_w, img_h):
        """Отображает список объектов с учётом их типа."""
        list_widget.clear()
        for idx, ann in enumerate(annotations):
            if ann[0] == 'detect':
                _, cls, cx, cy, w, h = ann
                x = int((cx - w/2) * img_w)
                y = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                text = f"{idx}: cls={cls}, rect=({x},{y},{x2},{y2})"
            elif ann[0] == 'obb':
                _, cls, points = ann
                # points: [x1,y1,x2,y2,x3,y3,x4,y4]
                text = f"{idx}: cls={cls}, OBB (4 points)"
            elif ann[0] == 'segment':
                _, cls, points = ann
                num_pts = len(points)//2
                text = f"{idx}: cls={cls}, polygon ({num_pts} points)"
            else:
                text = f"{idx}: unknown type"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            list_widget.addItem(item)

    # ---------- Отрисовка аннотаций ----------
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
                x = int((cx - bw/2) * w)
                y = int((cy - bh/2) * h)
                x2 = x + int(bw * w)
                y2 = y + int(bh * h)
                cv2.rectangle(img_copy, (x, y), (x2, y2), color, thickness)
            elif ann[0] == 'obb':
                _, cls, points = ann
                # points: [x1,y1, x2,y2, x3,y3, x4,y4]
                pts = []
                for j in range(0, len(points), 2):
                    px = int(points[j] * w)
                    py = int(points[j+1] * h)
                    pts.append([px, py])
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
            elif ann[0] == 'segment':
                _, cls, points = ann
                pts = []
                for j in range(0, len(points), 2):
                    px = int(points[j] * w)
                    py = int(points[j+1] * h)
                    pts.append([px, py])
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
        return img_copy

    # ---------- Отображение текущего изображения ----------
    def show_current_image(self):
        if not self.image_paths:
            return

        if hasattr(self, 'images') and self.current_idx < len(self.images):
            img = self.images[self.current_idx].copy()
            ann = self.annotations[self.current_idx] if self.current_idx < len(self.annotations) else []
        else:
            img_path = self.image_paths[self.current_idx]
            self.log(f"Loading {os.path.basename(img_path)} (compression={'ON' if self.nav_widget._resize_cb.isChecked() else 'OFF'})")
            try:
                img_original = read_image_with_fallback(img_path)
            except Exception as e:
                self.log(f"Error loading image {img_path}: {e}")
                return

            if img_original is None:
                self.log(f"Failed to load image {img_path} - image is None")
                return

            img_original = normalize_to_uint8(img_original)

            if self.nav_widget._resize_cb.isChecked():
                img = resize_to_max_side(img_original, max_side=1024)
            else:
                img = img_original

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] != 3:
                img = img[:, :, :3]

            txt_path = os.path.splitext(self.image_paths[self.current_idx])[0] + ".txt"
            ann = load_annotations(txt_path, img.shape[1], img.shape[0])

        self.current_img_w = img.shape[1]
        self.current_img_h = img.shape[0]
        self.current_annotations = ann

        # Наложение маски
        if self.current_idx < len(self.loaded_masks) and self.loaded_masks[self.current_idx] is not None:
            mask = self.loaded_masks[self.current_idx]
            if mask.shape[:2] != (self.current_img_h, self.current_img_w):
                mask = cv2.resize(mask, (self.current_img_w, self.current_img_h), interpolation=cv2.INTER_NEAREST)
            img = self.apply_mask_overlay(img, mask)

        annotated = self.draw_annotations_with_selection(img, self.current_annotations, self.selected_index)
        pixmap = numpy_to_qpixmap(annotated)
        if pixmap.isNull():
            self.log("Failed to convert image to QPixmap")
        else:
            self.viewer.set_pixmap(pixmap)

        self.update_annotation_list(self.coord_list, self.current_annotations, self.current_img_w, self.current_img_h)
        self.update_info_label()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_idx, len(self.image_paths))

    # ---------- Сохранение ----------
    def save_current_annotations(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        img_path = self.image_paths[self.current_idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        success = save_annotations(self.current_annotations, txt_path, self.current_img_w, self.current_img_h)
        if success:
            self.log(f"Saved {len(self.current_annotations)} annotations to {txt_path}")
            QMessageBox.information(self, "Save", f"Annotations saved to {txt_path}")
        else:
            self.log(f"Error saving annotations to {txt_path}")
            QMessageBox.critical(self, "Error", f"Failed to save to {txt_path}")

    # ---------- Навигация ----------
    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.selected_index = -1
            self.show_current_image()
            self.nav_widget.set_current_index(self.current_idx, len(self.image_paths))
            self.update_navigation_state()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.selected_index = -1
            self.show_current_image()
            self.nav_widget.set_current_index(self.current_idx, len(self.image_paths))
            self.update_navigation_state()

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
            if self.image_paths:
                self.load_images_from_source(self.image_paths)
            else:
                self.show_current_image()
        else:
            self.image_paths = []
            self.images = []
            self.gray_images = []
            self.annotations = []
            self.current_idx = 0
            self.current_annotations = []
            self.selected_index = -1
            self.loaded_masks = []
            self.viewer.set_pixmap(numpy_to_qpixmap(None))
            self.coord_list.clear()
            self.nav_widget.set_current_index(0, 0)
            self.update_navigation_state()
            self.info_label.setText("No images")
            self.log("Resize mode changed, images cleared. Please load images again.")

    # ---------- Обработка выбора объектов ----------
    def on_object_selected(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None and idx != self.selected_index:
            self.selected_index = idx
            self.show_current_image()
            self.log(f"Selected object {idx + 1}: {self.current_annotations[idx]}")

    def show_object_context_menu(self, pos):
        item = self.coord_list.itemAt(pos)
        if item is not None:
            idx = item.data(Qt.UserRole)
            menu = QMenu()
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self.delete_annotation_by_index(idx))
            menu.addAction(delete_action)
            menu.exec_(self.coord_list.mapToGlobal(pos))

    def delete_annotation_by_index(self, idx):
        if 0 <= idx < len(self.current_annotations):
            new_ann = self.current_annotations[:idx] + self.current_annotations[idx+1:]
            self.current_annotations = new_ann
            if self.current_idx < len(self.annotations):
                self.annotations[self.current_idx] = self.current_annotations.copy()
            new_sel = -1 if self.selected_index == idx else (
                self.selected_index - 1 if self.selected_index > idx else self.selected_index
            )
            self.selected_index = new_sel
            self.update_annotation_list(self.coord_list, self.current_annotations, self.current_img_w, self.current_img_h)
            self.show_current_image()
            self.log(f"Deleted object {idx + 1}")
            self.update_navigation_state()

    def delete_selected_object(self):
        current_row = self.coord_list.currentRow()
        if current_row >= 0 and current_row < len(self.current_annotations):
            self.delete_annotation_by_index(current_row)
        else:
            self.log("No object selected for deletion.")

    # ---------- Изменение класса объекта ----------
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
                QMessageBox.warning(self, "Ошибка", f"Некорректный ID: {e}\nВведите целое неотрицательное число.")
                return

            if typ == 'detect':
                self.current_annotations[idx] = ('detect', new_cls, cx, cy, w, h)
            elif typ == 'obb':
                self.current_annotations[idx] = ('obb', new_cls, points)
            else:  # segment
                self.current_annotations[idx] = ('segment', new_cls, points)

            if self.current_idx < len(self.annotations):
                self.annotations[self.current_idx] = self.current_annotations.copy()

            self.update_annotation_list(self.coord_list, self.current_annotations, self.current_img_w, self.current_img_h)
            self.show_current_image()
            self.log(f"Объекту {idx + 1} присвоен новый ID класса: {new_cls}")

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    # ---------- Загрузка YAML датасета ----------
    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите YAML файл датасета", "", "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return

        try:
            (paths, imgs, grays, anns,
             mask_paths, mask_imgs) = load_dataset_from_yaml_with_masks(
                yaml_path=file_path,
                resize_enabled=self.nav_widget._resize_cb.isChecked(),
                max_side=1024,
                parent=self
            )

            if not paths:
                QMessageBox.warning(self, "Ошибка", "Не найдено изображений в YAML.")
                return

            self.image_paths = paths
            self.images = imgs
            self.gray_images = grays
            self.annotations = anns
            self.loaded_masks = mask_imgs

            # Генерация цветов для классов масок
            all_classes = set()
            for mask in mask_imgs:
                if mask is not None:
                    all_classes.update(np.unique(mask))
            self.class_colors = {}
            for cls in sorted(all_classes):
                if cls == 0:
                    continue
                hue = (cls * 37) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                self.class_colors[cls] = (int(color[0]), int(color[1]), int(color[2]))

            self.current_idx = 0
            self.show_current_image()
            self.nav_widget.set_current_index(0, len(self.image_paths))
            self.update_navigation_state()
            self.log(f"Загружено {len(self.image_paths)} изображений из YAML.")
            QMessageBox.information(self, "Успех", f"Загружено {len(self.image_paths)} изображений.")

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