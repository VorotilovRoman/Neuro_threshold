# preparing_dataset_yaml.py
from import_libs_internal import *
from import_libs_methods_ui import setup_preparing_dataset_yaml_ui


ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class DatasetPreparationWindow(DatasetLoaderMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Preparation for YOLO / U-Net / COCO")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)

        setup_preparing_dataset_yaml_ui(self)

        if ULTRALYTICS_AVAILABLE:
            self.log("Ultralytics YOLO установлен")
        else:
            self.log("Ultralytics YOLO не установлен. Установите: pip install ultralytics")

        # --- Пути и пары ---
        self.images_folder = None
        self.labels_folder = None
        self.masks_folder = None
        self.output_folder = None
        self.pairs = []
        self.generator_thread = None

        # --- Классы ---
        self.unique_classes = []
        self.original_ids = []
        self.mask_class_remap = {}
        self.included_orig_ids = set()
        self.class_remap = {}
        self.num_classes = 0
        self.annotation_types_stats = {}

        # --- Режим ---
        self.dataset_type = 0
        self.input_format = 'auto'
        self.mask_pixel_offset_in = 1

        # --- Валидатор ---
        self.validator = PairValidator(dataset_type=0, input_format='auto')

        # --- Статистика ---
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        # --- YAML-режим / COCO / служебные (нужны mixin) ---
        self._last_patched_yaml = None
        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None
        self._coco_tmp_labels_dir = None
        self._coco_json_counter = 0
        self._updating_class_table = False

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Готово: %p%")

        # --- Инициализация UI ---
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
        self.input_format_combo.currentIndexChanged.connect(self._on_input_format_changed)
        self.class_table.itemChanged.connect(self.on_class_table_item_changed)
        self.out_masks_offset.valueChanged.connect(self._on_out_masks_offset_changed)

        self.include_empty_annotations.stateChanged.connect(self._on_include_empty_toggled)

        self.file_list.itemDoubleClicked.connect(self._on_file_list_double_clicked)

        self._copy_shortcut = QShortcut(QKeySequence.Copy, self.file_list)
        self._copy_shortcut.setContext(Qt.WidgetShortcut)
        self._copy_shortcut.activated.connect(self._copy_selected_rows_to_clipboard)

        if hasattr(self, 'btn_convert_gray'):
            self.btn_convert_gray.clicked.connect(lambda: self.convert_all_images('gray'))
        if hasattr(self, 'btn_convert_rgb'):
            self.btn_convert_rgb.clicked.connect(lambda: self.convert_all_images('color'))
        if hasattr(self, 'btn_convert_8bit_png'):
            self.btn_convert_8bit_png.clicked.connect(self.convert_to_8bit_png)

        self.update_multiplier_slider_state()

        # Первичная синхронизация UI — после того, как все атрибуты
        # (dataset_type, input_format, validator, ...) установлены.
        try:
            if hasattr(self, '_sync_all_states'):
                self._sync_all_states()
        except Exception as e:
            self._log_exception("Ошибка первичной синхронизации UI", e)

        try:
            self.log_widget.setVisible(True)
            self.toggle_log_btn.setChecked(True)
            self.toggle_log_btn.setText("Скрыть лог")
        except Exception:
            pass

    # ==================================================================
    # Хуки DatasetLoaderMixin
    # ==================================================================
    def _after_load_hook(self):
        """После успешной загрузки/сканирования — пересчитать состояние кнопок."""
        self.update_generate_button_state()
        self.update_split_counts()

    def _after_type_change_hook(self, new_type):
        """После смены dataset_type — обновить состояние кнопок."""
        self.update_generate_button_state()
        self.update_split_counts()

    # ==================================================================
    # Overrides: поведение отличается от mixin
    # ==================================================================
    def _update_color_stats_label(self):
        """Override: дополнительно включает/выключает кнопки конвертации."""
        if not hasattr(self, 'color_stats_label'):
            return
        text, enabled = ImageStats.format_stats_text(
            self.color_type_stats, self.bit_type_stats, bool(self.pairs))
        self.color_stats_label.setText(text)
        for name in ('btn_convert_gray', 'btn_convert_rgb',
                     'btn_convert_8bit_png'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(enabled)

    def _make_list_item(self, text, tag=None, error_msg=""):
        """Override: другой tooltip для invalid-пар."""
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()
        if tag == 'invalid':
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Некорректная пара: {error_msg}\n\n"
                            f"Двойной клик — всё равно открыть.")
        elif tag == 'empty':
            if include_empty:
                item.setForeground(QColor(0, 140, 0))
                item.setToolTip("Пустая аннотация — будет включена как негатив.")
            else:
                item.setForeground(QColor(200, 120, 0))
                item.setToolTip("Пустая аннотация — будет пропущена.")
        elif tag == 'error':
            item.setForeground(QColor(200, 0, 0))
            item.setToolTip("Ошибка чтения аннотации.")
        elif tag == 'mixed':
            item.setForeground(QColor(160, 0, 200))
            item.setToolTip("Смешанные типы аннотаций в файле.")
        elif tag == 'unknown':
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _refresh_mask_pixel_column(self):
        """
        Обновляет колонку 3 ("Mask pixel") — какое значение пикселя
        будет у класса в выходной маске.

        • input=masks        → pixel = New ID (dense, из _remap_mask_pixels);
        • input=labels       → pixel = New ID + out_masks_offset
                               (из export_masks_from_annotations).

        Колонка показывается только при input=masks или out_masks=True.
        """
        input_type = getattr(self, 'dataset_type', 0)
        try:
            out_masks = self.out_masks.isChecked()
        except Exception:
            out_masks = False
        try:
            offset = int(self.out_masks_offset.value())
        except Exception:
            offset = 1

        show = (input_type == 1) or out_masks
        try:
            self.class_table.setColumnHidden(3, not show)
        except Exception:
            pass

        self.class_table.blockSignals(True)
        try:
            for row in range(self.class_table.rowCount()):
                new_id_item = self.class_table.item(row, 0)
                if new_id_item is None:
                    continue

                try:
                    new_id = int(new_id_item.text())
                except (ValueError, TypeError):
                    item = QTableWidgetItem("-")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.class_table.setItem(row, 3, item)
                    continue

                # Выходной пиксель в маске всегда = New ID + offset
                # (offset ≥ 1, чтобы первый класс не слился с фоном).
                # Это справедливо и для input=masks, и для input=labels:
                # _remap_mask_pixels применяет ту же формулу, что и
                # export_masks_from_annotations.
                eff_offset = max(offset, 1)
                pixel = new_id + eff_offset

                item = QTableWidgetItem(str(pixel))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(row, 3, item)
        finally:
            self.class_table.blockSignals(False)

    def get_class_names_from_table(self):
        """Override: возвращает {new_id: name}, а не {orig_id: name}."""
        names = {}
        if self.dataset_type == 2:
            return names
        for i in range(self.class_table.rowCount()):
            cb = self.class_table.cellWidget(i, 4)
            if cb and cb.isChecked():
                try:
                    new_id = int(self.class_table.item(i, 0).text())
                except (ValueError, TypeError, AttributeError):
                    continue
                name_item = self.class_table.item(i, 2)
                name = name_item.text().strip() if name_item else str(new_id)
                if not name:
                    name = str(new_id)
                names[new_id] = name
        return names

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    # ==================================================================
    # YAML-патч для channels
    # ==================================================================
    def _get_selected_channels(self):
        if hasattr(self, 'channels_combo'):
            try:
                return int(self.channels_combo.currentText().split()[0])
            except (ValueError, IndexError, AttributeError):
                pass
        return 3

    def _find_dataset_yaml(self, folder, max_depth=4):
        if not folder or not os.path.isdir(folder):
            return None
        candidates = ('data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml')
        for name in candidates:
            p = os.path.join(folder, name)
            if os.path.exists(p):
                return p
        for root, dirs, files in os.walk(folder):
            depth = root[len(folder):].count(os.sep)
            if depth > max_depth:
                dirs[:] = []
                continue
            for f in files:
                if f.lower() in candidates:
                    return os.path.join(root, f)
        return None

    def _inject_channels_into_yaml(self, yaml_path, channels):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            old = cfg.get('channels', '—')
            cfg['channels'] = int(channels)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            self.log(f"✅ {os.path.basename(yaml_path)}: channels = {channels} (было: {old})")
            self._last_patched_yaml = yaml_path
            return True
        except Exception as e:
            self.log(f"⚠️ Не удалось обновить {yaml_path}: {e}")
            return False

    # ==================================================================
    # Массовая конвертация
    # ==================================================================
    def convert_all_images(self, target='gray'):
        if not self.pairs:
            return
        to_convert = select_pairs_for_conversion(self.pairs, self.validator, target)
        if not to_convert:
            QMessageBox.information(self, "Конвертация не требуется",
                                     "Все изображения уже соответствуют формату.")
            return

        target_label = "ЧБ (1 канал)" if target == 'gray' else "RGB (3 канала)"
        reply = QMessageBox.question(
            self, "Конвертация изображений",
            f"Будет перезаписано {len(to_convert)} из {len(self.pairs)} "
            f"изображений (в {target_label}).\n"
            f"Действие необратимо. Продолжить?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        total = len(to_convert)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Конвертация: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Конвертация: {current}/{tot} ({percent}%)")
            QApplication.processEvents()

        converted, errors = convert_pairs_color(
            to_convert, target, progress_cb=on_progress, log_cb=self.log)

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        self.log(f"Конвертация завершена: перезаписано {converted}, "
                 f"ошибок {errors}")

        self.validator.clear_cache()
        self._recompute_color_stats()
        self._rebuild_file_list_display(pairs=self.pairs)

    def convert_to_8bit_png(self):
        if not self.pairs:
            return
        all_convert, need_convert = select_pairs_for_8bit_png(
            self.pairs, self.validator)
        if not need_convert:
            QMessageBox.information(self, "Нечего конвертировать",
                                     "Все изображения уже 8-битные PNG.")
            return

        box = QMessageBox(self)
        box.setWindowTitle("Конвертация в 8 бит PNG")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"Найдено <b>{len(need_convert)}</b> изображений, не являющихся "
            f"8-битным PNG, из <b>{len(all_convert)}</b> всего.<br><br>"
            f"Маски не конвертируются (остаются в своём dtype)."
        )
        btn_all = box.addButton(f"Все ({len(all_convert)})", QMessageBox.AcceptRole)
        btn_only = box.addButton(f"Только не-8bit-PNG ({len(need_convert)})",
                                  QMessageBox.AcceptRole)
        btn_cancel = box.addButton("Отмена", QMessageBox.RejectRole)
        box.setDefaultButton(btn_only)
        box.exec_()

        clicked = box.clickedButton()
        if clicked == btn_cancel or clicked is None:
            return
        to_convert = all_convert if clicked == btn_all else need_convert

        total = len(to_convert)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Конвертация: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Конвертация: {current}/{tot} ({percent}%)")
            QApplication.processEvents()

        skip_masks = (self.dataset_type == 1)
        converted, errors, path_map = convert_pairs_to_8bit_png(
            to_convert, progress_cb=on_progress, log_cb=self.log,
            skip_masks=skip_masks)

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        self.log(f"8-битный PNG: перезаписано {converted}, ошибок {errors}")

        if path_map:
            new_pairs = []
            for pair in self.pairs:
                img_path = pair[0]
                if img_path in path_map:
                    new_pairs.append((path_map[img_path],
                                       pair[1] if len(pair) > 1 else None))
                else:
                    new_pairs.append(pair)
            self.pairs = new_pairs

        self.validator.clear_cache()
        self._recompute_color_stats()
        self._rebuild_file_list_display(pairs=self.pairs)

    # ==================================================================
    # Множители / Split
    # ==================================================================
    def _on_out_masks_offset_changed(self, *_):
        self._refresh_mask_pixel_column()

    def update_multiplier_slider_state(self):
        is_masks = (getattr(self, 'dataset_type', 0) == 1)
        any_aug = (self.flip_horizontal.isChecked() or
                   self.flip_vertical.isChecked() or
                   self.rotate_90.isChecked() or
                   (not is_masks and (self.random_rotate.isChecked() or
                                       self.random_crop.isChecked() or
                                       self.shear.isChecked())))
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
            if is_masks:
                max_unique = 2 ** discrete_count if discrete_count > 0 else 1
                max_unique = min(max_unique, 8)
                if self.aug_multiplier_slider.value() > max_unique:
                    self.log(f"⚠️ При input=masks максимум {max_unique} "
                             f"уникальных вариантов (только flip и rotate 90).")
            else:
                if self.random_rotate.isChecked() or self.random_crop.isChecked() \
                        or self.shear.isChecked():
                    pass
                else:
                    max_unique = 2 ** discrete_count if discrete_count > 0 else 1
                    if self.aug_multiplier_slider.value() > max_unique:
                        self.log(f"⚠️ Множитель {self.aug_multiplier_slider.value()} "
                                 f"может дать дубликаты: максимум уникальных "
                                 f"вариантов — {max_unique}.")
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
        multiplier = (self.aug_multiplier_slider.value()
                       if self.aug_multiplier_slider.isEnabled() else 1)
        total_generated = total_pairs * multiplier
        train_count = int(total_generated * train_pct)
        val_count = int(total_generated * val_pct)
        test_count = total_generated - train_count - val_count
        self.train_count_label.setText(str(train_count))
        self.val_count_label.setText(str(val_count))
        self.test_count_label.setText(str(test_count))
        self.total_count_with_multiplier.setText(
            f"Итоговое кол-во снимков: {total_generated}")

    # ==================================================================
    # Выходная папка
    # ==================================================================
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder to save YAML and dataset")
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

    def update_generate_button_state(self):
        has_pairs = len(self.pairs) > 0
        has_output = bool(self.output_folder)
        any_out = (self.out_yolo_txt.isChecked()
                   or self.out_masks.isChecked()
                   or self.out_coco.isChecked()
                   or self.dataset_type == 2)
        self.generate_btn.setEnabled(has_pairs and has_output and any_out)

    # ==================================================================
    # ГЕНЕРАЦИЯ
    # ==================================================================
    def on_generate(self):
        if not self.output_folder:
            QMessageBox.warning(self, "Error", "Please select output folder!")
            return
        if len(self.pairs) == 0:
            QMessageBox.warning(self, "Error", "No pairs found!")
            return

        input_type = self.dataset_type
        input_format = self.input_format_combo.currentData() or 'auto'

        out_yolo_txt = self.out_yolo_txt.isChecked()
        out_yolo_subtype = self.out_yolo_subtype.currentData() or 'keep'
        out_yolo_obb_format = self.out_yolo_obb_format.currentData() or 'corners'
        out_masks = self.out_masks.isChecked()
        out_masks_offset = int(self.out_masks_offset.value())
        out_unet_yaml = self.out_unet_yaml.isChecked()
        out_coco = self.out_coco.isChecked()

        if input_type != 2 and not any([out_yolo_txt, out_masks, out_coco]):
            QMessageBox.warning(
                self, "Форматы вывода",
                "Не выбран ни один формат вывода:\n"
                "включите YOLO labels, U-Net masks или COCO.")
            return
        if out_unet_yaml and not out_masks:
            reply = QMessageBox.question(
                self, "U-Net YAML без масок",
                "masks_data.yaml без масок бессмысленен.\n"
                "Включить U-Net masks автоматически?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                out_masks = True
                self.out_masks.setChecked(True)
            else:
                out_unet_yaml = False

        if input_type == 1 and (out_yolo_txt or out_coco):
            reply = QMessageBox.question(
                self, "⚠️ Lossy-конвертация из масок",
                "Маски будут преобразованы в полигоны через findContours.\n"
                "Это НЕ воспроизведёт маску пиксель в пиксель:\n"
                "  • дырки внутри объектов теряются;\n"
                "  • мелкие детали и тонкие структуры теряются;\n"
                "  • границы смещаются на 1–2 пикселя.\n\n"
                "Исходные маски в masks/{split} останутся нетронутыми.\n\n"
                "Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        _, invalid_count = self._validate_all_pairs()
        if invalid_count > 0:
            reply = QMessageBox.question(
                self, "Некорректные пары",
                f"Обнаружено {invalid_count} некорректных пар.\n"
                f"Они будут исключены из датасета. Продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return

        error_msg = PairValidator.check_pair_type_consistency()
        if error_msg:
            reply = QMessageBox.question(
                self, "Несоответствие типа",
                f"{error_msg}\nВсё равно продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        non_8bit = (self.bit_type_stats.get('uint16', 0) +
                    self.bit_type_stats.get('int16', 0) +
                    self.bit_type_stats.get('float', 0) +
                    self.bit_type_stats.get('other', 0))
        if non_8bit > 0 and input_type != 1:
            reply = QMessageBox.question(
                self, "Не-8bit изображения",
                f"В датасете {non_8bit} изображений не являются 8-битными.\n"
                f"YOLO может упасть с ошибкой CV_16U.\n"
                f"Рекомендуется «Перевести в 8 бит PNG».\n\n"
                f"Всё равно продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
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
            'random_rotate': self.random_rotate.isChecked() and input_type != 1,
            'random_rotate_angle': self.random_rotate_angle.value(),
            'random_crop': self.random_crop.isChecked() and input_type != 1,
            'shear': self.shear.isChecked() and input_type != 1,
            'shear_angle': self.shear_angle.value(),
        }
        has_augmentations = any(v for k, v in augmentations.items()
                                 if k not in ['random_rotate_angle', 'shear_angle'])
        aug_multiplier = (self.aug_multiplier_slider.value()
                           if has_augmentations else 1)

        selected_channels = self._get_selected_channels()
        class_names = self.get_class_names_from_table()
        num_classes = len(class_names)

        included_orig_ids = None
        if input_type == 1:
            included_orig_ids = self.included_orig_ids.copy()

        include_empty_annotations = self.include_empty_annotations.isChecked()
        delete_source_after_success = self.delete_source_after_success.isChecked()
        min_ann_area_px = int(self.min_ann_area_spin.value())

        pairs_for_generation = []
        skipped_invalid = 0
        for pair in self.pairs:
            ok, _ = self.validator.validate(pair)
            if ok:
                pairs_for_generation.append(pair)
            else:
                skipped_invalid += 1

        if not pairs_for_generation:
            QMessageBox.critical(self, "Ошибка",
                                  "Все пары некорректны. Генерация невозможна.")
            return

        norm_pairs = []
        for p in pairs_for_generation:
            img = p[0]
            aux = p[1] if len(p) > 1 else None
            if input_type == 1:
                norm_pairs.append((img, None, aux))
            else:
                norm_pairs.append((img, aux, None))

        self.log("=== Dataset generation started ===")
        self.log(f"Input: type={input_type}, format={input_format}")
        self.log(f"Outputs: yolo_txt={out_yolo_txt} (subtype={out_yolo_subtype}, "
                 f"obb={out_yolo_obb_format}), masks={out_masks} "
                 f"(offset={out_masks_offset}), unet_yaml={out_unet_yaml}, "
                 f"coco={out_coco}")
        self.log(f"Split: train={train_pct*100:.0f}%, val={val_pct*100:.0f}%, "
                 f"test={test_pct*100:.0f}%")
        self.log(f"Image size: {'keep' if target_size is None else target_size} "
                 f"(mode={resize_mode})")
        self.log(f"Augmentation: {'enabled' if has_augmentations else 'disabled'}")
        if has_augmentations:
            self.log(f"Augmentation multiplier: {aug_multiplier}x")
        self.log(f"Classes: {num_classes}")
        if skipped_invalid:
            self.log(f"Пропущено некорректных пар: {skipped_invalid}")
        self.log(f"Min annotation area: {min_ann_area_px} px² "
                 f"({'off' if min_ann_area_px <= 0 else 'on'})")
        self._set_generate_button_active(True)
        self.cancel_btn.setEnabled(True)

        bg_color = ((0, 0, 0) if self.bg_color_combo.currentText() == "Black"
                    else (255, 255, 255))

        if input_type == 0:
            class_mapping = self.class_remap
        elif input_type == 1:
            class_mapping = self.mask_class_remap
        else:
            class_mapping = {}

        labels_path = None
        if input_type == 0:
            labels_path = self.labels_folder
        elif input_type == 1:
            labels_path = self.masks_folder

        self.generator_thread = DatasetGeneratorThread(
            self.images_folder,
            labels_path,
            self.output_folder,
            norm_pairs,
            train_pct, val_pct, test_pct,
            target_size,
            augmentations, has_augmentations, aug_multiplier,
            class_names, class_mapping, bg_color,
            input_type,
            self.masks_folder,
            num_classes,
            self.mask_class_remap,
            included_orig_ids=included_orig_ids,
            included_class_ids=None,
            total_orig_classes=len(self.original_ids),
            resize_mode=resize_mode,
            include_empty_annotations=include_empty_annotations,
            delete_source_after_success=delete_source_after_success,
            input_format=input_format,
            out_yolo_txt=out_yolo_txt,
            out_yolo_subtype=out_yolo_subtype,
            out_yolo_obb_format=out_yolo_obb_format,
            out_masks=out_masks,
            out_masks_offset=out_masks_offset,
            out_unet_yaml=out_unet_yaml,
            out_coco=out_coco,
            mask_pixel_offset_in=self.mask_pixel_offset_in,
            min_ann_area_px=min_ann_area_px,
        )

        self.generator_thread.log_signal.connect(self.log)
        self.generator_thread.progress_signal.connect(self.update_progress)
        self.generator_thread.finished_signal.connect(self.on_generation_finished)
        self.generator_thread.start()

    def _set_generate_button_active(self, active):
        if active:
            self.generate_btn.setEnabled(False)
            self.generate_btn.setText("Генерация...")
            self.generate_btn.setStyleSheet(
                "background-color: orange; color: black; font-weight: bold;")
        else:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate Dataset")
            self.generate_btn.setStyleSheet("")

    def on_cancel_generation(self):
        if self.generator_thread and self.generator_thread.isRunning():
            self.log("Отмена генерации...")
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

            if self.out_yolo_txt.isChecked():
                yaml_path = self._find_dataset_yaml(self.output_folder)
                if yaml_path:
                    self._inject_channels_into_yaml(yaml_path,
                                                     self._get_selected_channels())
                else:
                    self.log("⚠️ data.yaml не найден — channels не прописан")

            if self.delete_source_after_success.isChecked():
                self.pairs = []
                self.file_list.clear()
                self.class_table.setRowCount(0)
                self.original_ids = []
                self.unique_classes = []
                self.num_classes = 0
                self.annotation_types_stats = {}
                self.class_remap = {}
                self.mask_class_remap = {}
                self.included_orig_ids = set()
                self.validator.clear_cache()
                self.color_type_stats = ImageStats.empty_color_stats()
                self.bit_type_stats = ImageStats.empty_bit_stats()
                self._yaml_mode = False
                self.validation_warning_label.setVisible(False)
                self._update_pair_count_label("")
                self._update_color_stats_label()
                self.update_split_counts()
                self.update_generate_button_state()
                self.log("Исходные файлы удалены, список пар очищен.")
        else:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Ошибка",
                                  f"Ошибка генерации датасета:\n{message}")
        self.log(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetPreparationWindow()
    window.show()
    sys.exit(app.exec_())