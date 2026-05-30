from import_libs_internal import *
from import_libs_methods_ui import setup_preparing_dataset_yaml_ui

ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    # Не выводим в консоль, только сохраняем статус
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class DatasetPreparationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Preparation for YOLO")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)

        setup_preparing_dataset_yaml_ui(self)

        if ULTRALYTICS_AVAILABLE:
            self.log("Ultralytics YOLO установлен")
        else:
            self.log("Ultralytics YOLO не установлен. Установите: pip install ultralytics")

        self.images_folder = None
        self.labels_folder = None
        self.masks_folder = None
        self.output_folder = None
        self.pairs = []
        self.generator_thread = None
        self.unique_classes = []
        self.original_ids = []
        self.mask_class_remap = {}
        self.included_orig_ids = set()
        self.class_remap = {}
        self.dataset_type = 0
        self.num_classes = 0
        self.annotation_types_stats = {}

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Готово: %p%")

        self.update_split(self.train_slider.value(), 'train')

        self.flip_horizontal.stateChanged.connect(self.update_multiplier_slider_state)
        self.flip_vertical.stateChanged.connect(self.update_multiplier_slider_state)
        self.rotate_90.stateChanged.connect(self.update_multiplier_slider_state)
        self.random_rotate.stateChanged.connect(self.update_multiplier_slider_state)
        self.random_crop.stateChanged.connect(self.update_multiplier_slider_state)
        self.shear.stateChanged.connect(self.update_multiplier_slider_state)
        self.btn_open_output.clicked.connect(self.open_output_folder)
        self.aug_multiplier_slider.valueChanged.connect(self.update_multiplier_label)
        self.aug_multiplier_slider.valueChanged.connect(self.update_split_counts)
        self.train_slider.valueChanged.connect(self.update_split_counts)
        self.val_slider.valueChanged.connect(self.update_split_counts)
        self.test_slider.valueChanged.connect(self.update_split_counts)

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.dataset_type_combo.currentIndexChanged.connect(self._on_dataset_type_changed)
        self.class_table.itemChanged.connect(self.on_class_table_item_changed)

        self.update_multiplier_slider_state()

    def _on_dataset_type_changed(self, idx):
        self.dataset_type = idx
        if idx == 0:
            self.btn_labels.setText("Load Labels")
            self.btn_labels.setEnabled(True)
        elif idx == 1:
            self.btn_labels.setText("Load Masks")
            self.btn_labels.setEnabled(True)
        else:
            self.btn_labels.setText("No labels (images only)")
            self.btn_labels.setEnabled(False)
            self.pairs = []
            self.file_list.clear()
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.update_generate_button_state()
            self.update_split_counts()
        if self.images_folder:
            self.scan_pairs()

    def update_multiplier_slider_state(self):
        any_aug = (self.flip_horizontal.isChecked() or
                   self.flip_vertical.isChecked() or
                   self.rotate_90.isChecked() or
                   self.random_rotate.isChecked() or
                   self.random_crop.isChecked() or
                   self.shear.isChecked())
        self.aug_multiplier_slider.setEnabled(any_aug)
        if not any_aug:
            self.aug_multiplier_slider.setValue(2)
            self.aug_multiplier_label.setText("2x")
        else:
            discrete_count = 0
            if self.flip_horizontal.isChecked():
                discrete_count += 1
            if self.flip_vertical.isChecked():
                discrete_count += 1
            if self.rotate_90.isChecked():
                discrete_count += 2
            if self.random_rotate.isChecked() or self.random_crop.isChecked() or self.shear.isChecked():
                max_unique = "неограничено"
            else:
                max_unique = 2 ** discrete_count if discrete_count > 0 else 1
                if self.aug_multiplier_slider.value() > max_unique:
                    self.log(f"Предупреждение: выбранный множитель {self.aug_multiplier_slider.value()} может привести к дублированию, "
                             f"так как максимальное количество уникальных вариантов при выбранных аугментациях — {max_unique}.")
        self.update_split_counts()

    def update_multiplier_label(self, value):
        self.aug_multiplier_label.setText(f"{value}x")
        self.update_split_counts()

    def update_split(self, value, source):
        self.train_slider.blockSignals(True)
        self.val_slider.blockSignals(True)
        self.test_slider.blockSignals(True)
        try:
            train = self.train_slider.value()
            val = self.val_slider.value()
            test = self.test_slider.value()
            if source == 'train':
                if val + test == 0:
                    new_val = 50
                    new_test = 50
                else:
                    total_remaining = val + test
                    new_val = int(val * (100 - train) / total_remaining)
                    new_test = (100 - train) - new_val
                new_val = max(0, min(100, new_val))
                new_test = max(0, min(100, new_test))
                if new_val + new_test != 100 - train:
                    new_val = (100 - train) - new_test
                self.val_slider.setValue(new_val)
                self.test_slider.setValue(new_test)
            elif source == 'val':
                new_test = 100 - train - value
                if new_test < 0:
                    new_test = 0
                    value = 100 - train
                self.test_slider.setValue(new_test)
                self.val_slider.setValue(value)
            elif source == 'test':
                new_val = 100 - train - value
                if new_val < 0:
                    new_val = 0
                    value = 100 - train
                self.val_slider.setValue(new_val)
                self.test_slider.setValue(value)
        finally:
            self.train_slider.blockSignals(False)
            self.val_slider.blockSignals(False)
            self.test_slider.blockSignals(False)
        self.train_label.setText(f"{self.train_slider.value()}%")
        self.val_label.setText(f"{self.val_slider.value()}%")
        self.test_label.setText(f"{self.test_slider.value()}%")
        self.update_split_counts()

    def select_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self.images_folder = folder
            self.log(f"Images folder: {folder}")
            self.scan_pairs()

    def select_labels_folder(self):
        if self.dataset_type == 2:
            return
        if self.dataset_type_combo.currentIndex() == 0:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with label files")
            if folder:
                self.labels_folder = folder
                self.masks_folder = None
                self.log(f"Labels folder: {folder}")
                self.scan_pairs()
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with mask images")
            if folder:
                self.masks_folder = folder
                self.labels_folder = folder
                self.log(f"Masks folder: {folder}")
                self.scan_pairs()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save YAML and dataset")
        if folder:
            self.output_folder = folder
            self.output_path_display.setText(folder)
            self.btn_open_output.setEnabled(True)
            self.log(f"Output folder: {folder}")
            self.update_generate_button_state()

    def open_output_folder(self):
        if self.output_folder and os.path.exists(self.output_folder):
            if sys.platform == 'win32':
                os.startfile(self.output_folder)
            elif sys.platform == 'darwin':
                os.system(f'open "{self.output_folder}"')
            else:
                os.system(f'xdg-open "{self.output_folder}"')
        else:
            QMessageBox.warning(self, "Ошибка", "Папка не выбрана или не существует.")

    def _detect_annotation_type(self, label_path):
        types = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    l = len(parts)
                    if l == 5:
                        types.add('detect')
                    elif l == 9:
                        types.add('obb')
                    elif l >= 7 and (l - 1) % 2 == 0:
                        types.add('segment')
                    else:
                        types.add('unknown')
        except Exception:
            return 'error'
        if len(types) == 0:
            return 'empty'
        if len(types) == 1:
            return next(iter(types))
        return 'mixed'

    def _check_pair_type_consistency(self):
        return None

    def scan_pairs(self):
        self.pairs.clear()
        self.file_list.clear()
        self.annotation_types_stats = {}
        if not self.images_folder:
            self.log("Папка с изображениями не выбрана")
            return

        dataset_type_idx = self.dataset_type_combo.currentIndex()
        self.log(f"Начинаем сканирование пар для типа {dataset_type_idx}")

        if dataset_type_idx == 0:  # Метки
            if not self.labels_folder:
                self.log("Папка с метками не выбрана")
                self._update_pair_count_label("", error=True)
                return
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            img_files = {}
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_files[name] = os.path.join(self.images_folder, f)
            except Exception as e:
                self.log(f"Ошибка при чтении папки изображений: {e}")
                return
            for name, img_path in img_files.items():
                label_path = os.path.join(self.labels_folder, name + '.txt')
                if os.path.exists(label_path):
                    try:
                        ann_type = self._detect_annotation_type(label_path)
                        self.annotation_types_stats[ann_type] = self.annotation_types_stats.get(ann_type, 0) + 1
                        self.pairs.append((img_path, label_path))
                        self.file_list.addItem(f"{name} (image + label) [{ann_type}]")
                    except Exception as e:
                        self.log(f"Ошибка обработки пары {name}: {e}")

        elif dataset_type_idx == 1:  # Маски
            if not self.masks_folder:
                self.log("Папка с масками не выбрана")
                self._update_pair_count_label("", error=True)
                return
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            img_files = {}
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_files[name] = os.path.join(self.images_folder, f)
            except Exception as e:
                self.log(f"Ошибка при чтении папки изображений: {e}")
                return
            for name, img_path in img_files.items():
                found = False
                for ext in img_extensions:
                    mask_path = os.path.join(self.masks_folder, name + ext)
                    try:
                        if os.path.exists(mask_path):
                            if os.path.abspath(mask_path) == os.path.abspath(img_path):
                                self.log(f"Предупреждение: файл маски совпадает с файлом изображения {img_path}, пара пропущена")
                                break
                            self.pairs.append((img_path, mask_path))
                            self.file_list.addItem(f"{name} (image + mask)")
                            found = True
                            break
                    except Exception as e:
                        self.log(f"Ошибка при проверке маски {mask_path}: {e}")
                        continue
                if not found:
                    try:
                        for f in os.listdir(self.masks_folder):
                            mask_name, mask_ext = os.path.splitext(f)
                            if mask_name == name and mask_ext.lower() in img_extensions:
                                mask_path = os.path.join(self.masks_folder, f)
                                if os.path.abspath(mask_path) == os.path.abspath(img_path):
                                    self.log(f"Предупреждение: файл маски совпадает с файлом изображения {img_path}, пара пропущена")
                                    continue
                                self.pairs.append((img_path, mask_path))
                                self.file_list.addItem(f"{name} (image + mask)")
                                found = True
                                break
                    except Exception as e:
                        self.log(f"Ошибка при поиске маски для {name}: {e}")

            self.log(f"Проверка валидности {len(self.pairs)} масок...")
            valid_pairs = []
            for img_path, mask_path in self.pairs:
                try:
                    test_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if test_mask is None:
                        self.log(f"Предупреждение: не удалось прочитать маску {mask_path} – файл повреждён или не является изображением")
                        continue
                    valid_pairs.append((img_path, mask_path))
                except Exception as e:
                    self.log(f"Критическая ошибка при проверке маски {mask_path}: {e}")
                    continue
            self.pairs = valid_pairs
            self.log(f"После проверки осталось {len(self.pairs)} валидных пар")

        else:  # Тип 2 – только изображения
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
            try:
                for f in os.listdir(self.images_folder):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        img_path = os.path.join(self.images_folder, f)
                        self.pairs.append((img_path, None))
                        self.file_list.addItem(f"{name} (image only)")
                self.annotation_types_stats = {}
            except Exception as e:
                self.log(f"Ошибка при сканировании изображений: {e}")

        error_msg = self._check_pair_type_consistency()
        extra_info = ""
        if self.annotation_types_stats:
            type_str = ", ".join([f"{k}:{v}" for k, v in self.annotation_types_stats.items()])
            extra_info = f"  [{type_str}]"
        self._update_pair_count_label(extra_info, error=bool(error_msg))
        self.log(f"Найдено пар: {len(self.pairs)}")
        if error_msg:
            self.log(error_msg)

        try:
            self.collect_classes()
            self.update_generate_button_state()
            self.update_split_counts()
        except Exception as e:
            self.log(f"Ошибка при обновлении интерфейса после сканирования: {e}")
            import traceback
            self.log(traceback.format_exc())

    def _update_pair_count_label(self, extra_text, error=False):
        text = f"Всего пар: {len(self.pairs)}"
        if extra_text:
            text += f"  {extra_text}"
        self.pair_count_label.setText(text)
        if error:
            self.pair_count_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pair_count_label.setStyleSheet("")

    def rebuild_class_mapping(self):
        if self.dataset_type != 0:
            return
        self.class_remap = {}
        for row in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(row, 3)
            if cb and cb.isChecked():
                try:
                    original = int(self.class_table.item(row, 1).text())
                    new = int(self.class_table.item(row, 0).text())
                    self.class_remap[original] = new
                except (ValueError, TypeError):
                    continue

    def update_included_classes(self):
        if self.dataset_type == 0:
            self.included_orig_ids = set()
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 3)
                if cb and cb.isChecked():
                    original = int(self.class_table.item(row, 1).text())
                    self.included_orig_ids.add(original)
            self.rebuild_class_mapping()
            self.unique_classes = sorted(set(self.class_remap.values()))
            self.num_classes = len(self.unique_classes)
            self.mask_class_remap = {}
        elif self.dataset_type == 1:
            if self.class_table.rowCount() == 0:
                return
            self.included_orig_ids = set()
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 3)
                if cb and cb.isChecked():
                    try:
                        orig_id = int(self.class_table.item(row, 1).text())
                        self.included_orig_ids.add(orig_id)
                    except (ValueError, TypeError, AttributeError):
                        continue
            sorted_included = sorted(self.included_orig_ids)
            self.mask_class_remap = {old: new for new, old in enumerate(sorted_included)}
            self.unique_classes = list(range(len(sorted_included)))
            self.num_classes = len(self.unique_classes)
            for row in range(self.class_table.rowCount()):
                try:
                    orig_id = int(self.class_table.item(row, 1).text())
                    if orig_id in self.mask_class_remap:
                        self.class_table.item(row, 0).setText(str(self.mask_class_remap[orig_id]))
                    else:
                        self.class_table.item(row, 0).setText('-')
                except Exception:
                    pass
        else:
            self.included_orig_ids = set()
            self.class_remap = {}
            self.mask_class_remap = {}
            self.unique_classes = []
            self.num_classes = 0

    def collect_classes(self):
        if not self.pairs:
            self.class_table.setRowCount(0)
            return
        if self.dataset_type == 0:
            unique_ids = set()
            for _, label_path in self.pairs:
                anns = load_annotations(label_path, 1, 1)
                for ann in anns:
                    cls = ann[1]
                    unique_ids.add(cls)
            self.original_ids = sorted(unique_ids)
            self.class_table.setRowCount(len(self.original_ids))
            for i, cid in enumerate(self.original_ids):
                id_item = QTableWidgetItem(str(cid))
                id_item.setFlags(id_item.flags() | Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, id_item)
                orig_item = QTableWidgetItem(str(cid))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)
                name_item = QTableWidgetItem(str(cid))
                self.class_table.setItem(i, 2, name_item)
                cb = QCheckBox()
                cb.setChecked(True)
                cb.stateChanged.connect(lambda state, row=i: self.update_included_classes())
                self.class_table.setCellWidget(i, 3, cb)
            self.update_included_classes()
        elif self.dataset_type == 1:
            unique_vals = set()
            valid_pairs_for_classes = []
            for img_path, mask_path in self.pairs:
                try:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        self.log(f"Предупреждение: не удалось прочитать маску {mask_path} при сборе классов – пропускаем")
                        continue
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    if mask.size == 0:
                        continue
                    if mask.dtype != np.uint8:
                        mask = mask.astype(np.uint8)
                    uniq = np.unique(mask)
                    unique_vals.update(uniq)
                    valid_pairs_for_classes.append((img_path, mask_path))
                except Exception as e:
                    self.log(f"Ошибка при обработке маски {mask_path}: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            if not valid_pairs_for_classes:
                self.log("Нет валидных масок для определения классов")
                self.class_table.setRowCount(0)
                self.original_ids = []
                self.update_included_classes()
                return
            self.pairs = valid_pairs_for_classes
            self.log(f"После фильтрации для классов осталось {len(self.pairs)} пар")
            all_vals = sorted(unique_vals)
            self.original_ids = all_vals
            temp_remap = {old: new for new, old in enumerate(all_vals)}
            self.class_table.setRowCount(0)
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(len(all_vals))
                for i, old_id in enumerate(all_vals):
                    new_id = temp_remap[old_id]
                    new_item = QTableWidgetItem(str(new_id))
                    new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 0, new_item)
                    orig_item = QTableWidgetItem(str(old_id))
                    orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(i, 1, orig_item)
                    name_item = QTableWidgetItem(str(old_id))
                    self.class_table.setItem(i, 2, name_item)
                    cb = QCheckBox()
                    cb.setChecked(True)
                    cb.stateChanged.connect(lambda state, row=i: self.update_included_classes())
                    self.class_table.setCellWidget(i, 3, cb)
            finally:
                self.class_table.blockSignals(False)
            self.update_included_classes()
        else:
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.update_included_classes()

    def get_class_names_from_table(self):
        names = {}
        if self.dataset_type == 2:
            return names
        for i in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(i, 3)
            if cb and cb.isChecked():
                new_id = int(self.class_table.item(i, 0).text())
                name = self.class_table.item(i, 2).text().strip()
                if not name:
                    name = str(new_id)
                names[new_id] = name
        return names

    def on_class_table_item_changed(self, item):
        if item.column() == 0:
            self.update_included_classes()

    def update_generate_button_state(self):
        has_pairs = len(self.pairs) > 0
        has_output = bool(self.output_folder)
        self.generate_btn.setEnabled(has_pairs and has_output)

    def on_generate(self):
        if not self.images_folder or not self.output_folder:
            QMessageBox.warning(self, "Error", "Please select images and output folders!")
            return
        if self.dataset_type_combo.currentIndex() == 0 and not self.labels_folder:
            QMessageBox.warning(self, "Error", "Please select labels folder for detection!")
            return
        if self.dataset_type_combo.currentIndex() == 1 and not self.masks_folder:
            QMessageBox.warning(self, "Error", "Please select masks folder for segmentation!")
            return
        if len(self.pairs) == 0:
            QMessageBox.warning(self, "Error", "No pairs found!")
            return

        error_msg = self._check_pair_type_consistency()
        if error_msg:
            reply = QMessageBox.question(self, "Несоответствие типа",
                                         f"{error_msg}\nВсё равно продолжить?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        train_pct = self.train_slider.value() / 100.0
        val_pct = self.val_slider.value() / 100.0
        test_pct = self.test_slider.value() / 100.0

        if self.resize_keep.isChecked():
            resize_mode = "keep"
            target_size = None
        elif self.resize_stretch.isChecked():
            resize_mode = "stretch"
            target_size = self.resize_size.value()
        else:
            resize_mode = "fixed"
            target_size = self.resize_size.value()

        augmentations = {
            'horizontal_flip': self.flip_horizontal.isChecked(),
            'vertical_flip': self.flip_vertical.isChecked(),
            'rotate_90': self.rotate_90.isChecked(),
            'random_rotate': self.random_rotate.isChecked(),
            'random_rotate_angle': self.random_rotate_angle.value(),
            'random_crop': self.random_crop.isChecked(),
            'shear': self.shear.isChecked(),
            'shear_angle': self.shear_angle.value()
        }
        has_augmentations = any(v for k, v in augmentations.items() if k not in ['random_rotate_angle', 'shear_angle'])
        aug_multiplier = self.aug_multiplier_slider.value() if has_augmentations else 1

        if target_size is None and has_augmentations:
            reply = QMessageBox.question(
                self, "Предупреждение",
                "Аугментации включены, но размер изображений не задан. "
                "Некоторые аугментации могут не работать. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        class_names = self.get_class_names_from_table()
        num_classes = len(class_names)

        included_orig_ids = None
        if self.dataset_type == 1:
            included_orig_ids = self.included_orig_ids.copy()

        self.log("=== Dataset generation started ===")
        self.log(f"Train: {train_pct*100:.0f}%, Val: {val_pct*100:.0f}%, Test: {test_pct*100:.0f}%")
        self.log(f"Image size: {'keep original' if target_size is None else target_size}")
        self.log(f"Augmentation: {'enabled' if has_augmentations else 'disabled'}")
        if has_augmentations:
            self.log(f"Augmentation multiplier: {aug_multiplier}x")
        self.log(f"Classes: {num_classes}")

        self._set_generate_button_active(True)
        self.cancel_btn.setEnabled(True)
        bg_color = (0, 0, 0) if self.bg_color_combo.currentText() == "Black" else (255, 255, 255)

        if self.dataset_type == 0:
            class_mapping = self.class_remap
        elif self.dataset_type == 1:
            class_mapping = self.mask_class_remap
        else:
            class_mapping = {}

        if self.dataset_type_combo.currentIndex() == 0:
            labels_path = self.labels_folder
        elif self.dataset_type_combo.currentIndex() == 1:
            labels_path = self.masks_folder
        else:
            labels_path = None

        self.generator_thread = DatasetGeneratorThread(
            self.images_folder,
            labels_path,
            self.output_folder, self.pairs, train_pct, val_pct, test_pct, target_size,
            augmentations, has_augmentations, aug_multiplier,
            class_names, class_mapping, bg_color,
            self.dataset_type_combo.currentIndex(),
            self.masks_folder,
            num_classes,
            self.mask_class_remap,
            included_orig_ids=included_orig_ids,
            included_class_ids=None,
            total_orig_classes=len(self.original_ids),
            resize_mode=resize_mode
        )

        self.generator_thread.log_signal.connect(self.log)
        self.generator_thread.progress_signal.connect(self.update_progress)
        self.generator_thread.finished_signal.connect(self.on_generation_finished)
        self.generator_thread.start()

    def _set_generate_button_active(self, active):
        if active:
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("Генерация...")
            self.generate_btn.setStyleSheet("background-color: orange; color: black; font-weight: bold;")
        else:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate Dataset")
            self.generate_btn.setStyleSheet("")

    def on_cancel_generation(self):
        if self.generator_thread and self.generator_thread.isRunning():
            self.log("Отмена генерации датасета...")
            self.generator_thread.cancel()
            self.cancel_btn.setEnabled(False)

    def update_progress(self, current, total):
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_bar.setFormat(f"Готово: {percent}% ({current}/{total})")
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Готово: 0% (0/0)")

    def on_generation_finished(self, success, message):
        self._set_generate_button_active(False)
        self.cancel_btn.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Успех", message)
        else:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Ошибка", f"Ошибка при генерации датасета:\n{message}")
        self.log(message)

    def log(self, message):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        QApplication.processEvents()

    def update_split_counts(self):
        total_pairs = len(self.pairs)
        if total_pairs == 0:
            self.train_count_label.setText("0")
            self.val_count_label.setText("0")
            self.test_count_label.setText("0")
            self.total_count_with_multiplier.setText("Итоговое кол-во снимков: 0")
            return
        train_pct = self.train_slider.value() / 100.0
        val_pct = self.val_slider.value() / 100.0
        test_pct = self.test_slider.value() / 100.0
        multiplier = self.aug_multiplier_slider.value() if self.aug_multiplier_slider.isEnabled() else 1
        total_generated = total_pairs * multiplier
        train_count = int(total_generated * train_pct)
        val_count = int(total_generated * val_pct)
        test_count = total_generated - train_count - val_count
        self.train_count_label.setText(str(train_count))
        self.val_count_label.setText(str(val_count))
        self.test_count_label.setText(str(test_count))
        self.total_count_with_multiplier.setText(f"Итоговое кол-во снимков: {total_generated}")

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetPreparationWindow()
    window.show()
    sys.exit(app.exec_())