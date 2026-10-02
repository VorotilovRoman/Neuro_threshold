# preparing_dataset_yaml.py
from import_libs_internal import *
from import_libs_methods_ui import setup_preparing_dataset_yaml_ui

from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence


ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class DatasetPreparationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Preparation for YOLO")
        self.setMinimumWidth(1100)
        self.setMinimumHeight(750)

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

        # Единый валидатор пар (валидация + кэш + определение характеристик).
        self.validator = PairValidator(dataset_type=0)

        # Статистика по цветности и битности — словари, которые
        # пересчитываются через self.validator.recompute_stats(self.pairs).
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        self._last_patched_yaml = None

        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None

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

    # ==================================================================
    # Утилиты списка пар
    # ==================================================================
    def _copy_selected_rows_to_clipboard(self):
        items = self.file_list.selectedItems()
        if not items:
            return
        text = "\n".join(it.text() for it in items)
        QApplication.clipboard().setText(text)
        self.log(f"Скопировано строк: {len(items)}")

    def _on_file_list_double_clicked(self, item):
        if item is None:
            return
        pair = item.data(Qt.UserRole + 1)
        if not pair:
            return
        self._open_pair_validation(pair)

    # ==================================================================
    # Валидация пары / статистика — тонкие обёртки над dataset_validation
    # ==================================================================
    def _validate_all_pairs(self):
        if not self.pairs:
            return 0, 0

        total = len(self.pairs)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Проверка пар: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Проверка пар: {current}/{tot} ({percent}%)"
            )
            QApplication.processEvents()

        valid, invalid = self.validator.validate_all(
            self.pairs, progress_cb=on_progress
        )

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        QApplication.processEvents()

        return valid, invalid

    def _recompute_color_stats(self):
        self.color_type_stats, self.bit_type_stats = \
            self.validator.recompute_stats(self.pairs)

    def _update_color_stats_label(self):
        if not hasattr(self, 'color_stats_label'):
            return
        text, enabled = ImageStats.format_stats_text(
            self.color_type_stats, self.bit_type_stats, bool(self.pairs)
        )
        self.color_stats_label.setText(text)
        for name in ('btn_convert_gray', 'btn_convert_rgb',
                     'btn_convert_8bit_png'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(enabled)

    def _open_pair_validation(self, pair):
        PairValidationDialog(self, pair, self.dataset_type,
                             self.validator).exec_()

    # ==================================================================
    # Channels в data.yaml
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
            import yaml
        except ImportError:
            self.log("⚠️ PyYAML не установлен — channels не прописан в data.yaml")
            return False
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
    # Load YAML
    # ==================================================================
    def load_from_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select data.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if not file_path:
            return

        try:
            import yaml
        except ImportError:
            QMessageBox.critical(self, "Ошибка",
                                 "PyYAML не установлен. Установите: pip install pyyaml")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать YAML:\n{e}")
            return

        if not isinstance(cfg, dict):
            QMessageBox.critical(self, "Ошибка", "YAML не содержит словаря.")
            return

        root = cfg.get('path', '')
        if root:
            if not os.path.isabs(root):
                root = os.path.normpath(os.path.join(os.path.dirname(file_path), root))
        else:
            root = os.path.dirname(file_path)
        root = os.path.abspath(root)

        splits_dirs = []
        for key in ('train', 'val', 'test', 'valid', 'validation'):
            val = cfg.get(key)
            if not val:
                continue
            items = val if isinstance(val, (list, tuple)) else [val]
            for it in items:
                if not isinstance(it, str):
                    continue
                p = it if os.path.isabs(it) else os.path.join(root, it)
                p = os.path.normpath(p)
                if os.path.isdir(p):
                    splits_dirs.append(p)
                elif os.path.isfile(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                q = line if os.path.isabs(line) else os.path.join(root, line)
                                q = os.path.normpath(q)
                                if os.path.isdir(q):
                                    splits_dirs.append(q)
                    except Exception:
                        pass

        if not splits_dirs:
            QMessageBox.warning(self, "Ошибка",
                                "В YAML не найдено ни одной существующей папки с изображениями.\n"
                                f"Проверено path: {root}")
            return

        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        all_images = []
        for sp in splits_dirs:
            try:
                for f in os.listdir(sp):
                    name, ext = os.path.splitext(f)
                    if ext.lower() in img_extensions:
                        all_images.append(os.path.join(sp, f))
            except Exception as e:
                self.log(f"Ошибка чтения {sp}: {e}")

        if not all_images:
            QMessageBox.warning(self, "Ошибка", "В папках не найдено изображений.")
            return

        def find_label(img_path):
            img_path_abs = os.path.abspath(img_path)
            d = os.path.dirname(img_path_abs)
            base = os.path.splitext(os.path.basename(img_path_abs))[0]

            same = os.path.join(d, base + '.txt')
            if os.path.exists(same):
                return same

            norm = img_path_abs.replace('\\', '/')
            parts = norm.split('/')
            for i, p in enumerate(parts):
                if p == 'images':
                    new_parts = parts[:i] + ['labels'] + parts[i + 1:]
                    new_parts[-1] = base + '.txt'
                    cand = os.sep.join(new_parts)
                    if os.path.exists(cand):
                        return cand
            return None

        new_pairs = []
        missing = 0
        for img in all_images:
            lbl = find_label(img)
            if lbl:
                new_pairs.append((img, lbl))
            else:
                missing += 1

        if not new_pairs:
            QMessageBox.warning(
                self, "Ошибка",
                f"Найдено {len(all_images)} изображений, но ни одного .txt-файла метки.\n"
                f"Ожидается структура images/... и labels/..."
            )
            return

        self.dataset_type_combo.blockSignals(True)
        self.dataset_type_combo.setCurrentIndex(0)
        self.dataset_type_combo.blockSignals(False)
        self.dataset_type = 0
        self.btn_labels.setText("Load Labels")
        self.btn_labels.setEnabled(True)

        self.pairs = new_pairs
        self._yaml_mode = True
        self._yaml_root = root
        self._yaml_path = file_path
        self._yaml_names = cfg.get('names')

        self.images_folder = root
        self.labels_folder = root
        self.masks_folder = None

        self.validator.set_dataset_type(0)
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
        self.annotation_types_stats = {}

        self.log(f"=== Загрузка из YAML: {file_path} ===")
        self.log(f"Root: {root}")
        self.log(f"Сплитов найдено: {len(splits_dirs)}")
        for sp in splits_dirs:
            self.log(f"  • {sp}")
        self.log(f"Изображений: {len(all_images)}")
        self.log(f"Пар с метками: {len(new_pairs)}")
        if missing:
            self.log(f"⚠️ Изображений без .txt: {missing}")

        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

        extra = f"  [YAML: {len(splits_dirs)} splits]"
        if missing:
            extra += f"  ⚠️ без меток: {missing}"
        self._update_pair_count_label(extra, error=invalid_count > 0)
        self.log(f"Найдено пар: {len(self.pairs)} (корректных: {valid_count}, некорректных: {invalid_count})")

        try:
            self.collect_classes()
            self.update_generate_button_state()
            self.update_split_counts()
        except Exception as e:
            import traceback
            self.log(f"Ошибка после загрузки YAML: {e}")
            self.log(traceback.format_exc())

    # ==================================================================
    # Массовая конвертация цветности
    # ==================================================================
    def convert_all_images(self, target='gray'):
        """target='gray' — в ЧБ (1 канал); target='color' — в RGB (3 канала)."""
        if not self.pairs:
            return

        to_convert = select_pairs_for_conversion(self.pairs, self.validator, target)

        if not to_convert:
            QMessageBox.information(
                self, "Конвертация не требуется",
                "Все изображения уже соответствуют выбранному формату."
            )
            return

        target_label = "ЧБ (1 канал)" if target == 'gray' else "RGB (3 канала)"
        reply = QMessageBox.question(
            self, "Конвертация изображений",
            f"Будет перезаписано {len(to_convert)} из {len(self.pairs)} изображений "
            f"(в формат {target_label}).\n"
            f"Остальные уже соответствуют целевому формату и останутся без изменений.\n\n"
            f"Действие необратимо. Продолжить?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        total = len(to_convert)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Конвертация: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Конвертация: {current}/{tot} ({percent}%)"
            )
            QApplication.processEvents()

        converted, errors = convert_pairs_color(
            to_convert, target,
            progress_cb=on_progress,
            log_cb=self.log,
        )

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        self.log(f"Конвертация завершена: перезаписано {converted}, ошибок {errors} "
                 f"(пропущено без изменений: {len(self.pairs) - total})")

        self.validator.clear_cache()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

    # ==================================================================
    # Конвертация в 8 бит PNG
    # ==================================================================
    def convert_to_8bit_png(self):
        """
        Конвертирует изображения в 8-битный PNG.
        Спрашивает пользователя: конвертировать всё или только то,
        что не является 8-битным PNG.
        """
        if not self.pairs:
            return

        all_convert, need_convert = select_pairs_for_8bit_png(
            self.pairs, self.validator
        )

        if not need_convert:
            QMessageBox.information(
                self, "Нечего конвертировать",
                "Все изображения уже являются 8-битными PNG."
            )
            return

        # --- Диалог выбора объёма конвертации ---
        box = QMessageBox(self)
        box.setWindowTitle("Конвертация в 8 бит PNG")
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"Найдено <b>{len(need_convert)}</b> изображений, не являющихся 8-битным PNG,\n"
            f"из <b>{len(all_convert)}</b> всего.\n\n"
            f"Какие снимки конвертировать?"
        )
        btn_all = box.addButton(f"Все ({len(all_convert)})", QMessageBox.AcceptRole)
        btn_only = box.addButton(f"Только не-8bit-PNG ({len(need_convert)})", QMessageBox.AcceptRole)
        btn_cancel = box.addButton("Отмена", QMessageBox.RejectRole)
        box.setDefaultButton(btn_only)
        box.exec_()

        clicked = box.clickedButton()
        if clicked == btn_cancel or clicked is None:
            return
        elif clicked == btn_all:
            to_convert = all_convert
        else:
            to_convert = need_convert

        total = len(to_convert)
        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Конвертация в 8 бит PNG: 0/{total}...")
        QApplication.processEvents()

        def on_progress(current, tot):
            percent = int(current / tot * 100)
            self.validation_progress_label.setText(
                f"⏳ Конвертация в 8 бит PNG: {current}/{tot} ({percent}%)"
            )
            QApplication.processEvents()

        converted, errors, path_map = convert_pairs_to_8bit_png(
            to_convert,
            progress_cb=on_progress,
            log_cb=self.log,
        )

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        self.log(f"Конвертация в 8 бит PNG завершена: перезаписано {converted}, ошибок {errors} "
                 f"(пропущено без изменений: {len(self.pairs) - total})")

        # Обновить пути в self.pairs
        if path_map:
            new_pairs = []
            for pair in self.pairs:
                img_path = pair[0]
                if img_path in path_map:
                    new_pairs.append((path_map[img_path], pair[1] if len(pair) > 1 else None))
                else:
                    new_pairs.append(pair)
            self.pairs = new_pairs

        # Сброс кэша и пересчёт
        self.validator.clear_cache()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

    # ==================================================================
    # Список пар
    # ==================================================================
    def _make_list_item(self, text, tag=None, error_msg=""):
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()

        if tag == 'invalid':
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Не корректная пара: {error_msg}\n\n"
                            f"Двойной клик — всё равно открыть снимок.")
        elif tag == 'empty':
            if include_empty:
                item.setForeground(QColor(0, 140, 0))
                item.setToolTip("Пустая аннотация — будет включена как негативный пример.")
            else:
                item.setForeground(QColor(200, 120, 0))
                item.setToolTip("Пустая аннотация — будет пропущена.")
        elif tag == 'error':
            item.setForeground(QColor(200, 0, 0))
            item.setToolTip("Ошибка чтения файла аннотации.")
        elif tag == 'mixed':
            item.setForeground(QColor(160, 0, 200))
            item.setToolTip("Смешанные типы аннотаций в одном файле.")
        elif tag == 'unknown':
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _rebuild_file_list_display(self):
        if not self.pairs:
            self.validation_warning_label.setVisible(False)
            self.validation_progress_label.setVisible(False)
            self._update_color_stats_label()
            return

        self.file_list.blockSignals(True)
        self.file_list.clear()

        valid_count = 0
        invalid_count = 0
        invalid_names = []

        total = len(self.pairs)
        step = max(1, total // 100)
        include_empty = self._is_include_empty()

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Построение списка: 0/{total}...")
        QApplication.processEvents()

        for idx, pair in enumerate(self.pairs):
            info = build_pair_display(
                pair, self.validator, self.dataset_type, include_empty
            )

            if info['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_names.append(info['name'])

            item = self._make_list_item(
                info['text'], tag=info['tag'], error_msg=info['err']
            )
            item.setData(Qt.UserRole, pair[0])
            item.setData(Qt.UserRole + 1, tuple(pair))
            item.setData(Qt.UserRole + 2,
                         'valid' if info['is_valid'] else 'invalid')
            item.setData(Qt.UserRole + 3, info['color_type'])
            item.setData(Qt.UserRole + 4, info['bit_type'])
            self.file_list.addItem(item)

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Построение списка: {idx + 1}/{total} ({percent}%)"
                )
                QApplication.processEvents()

        self.file_list.blockSignals(False)
        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")

        if invalid_count > 0:
            preview = ", ".join(invalid_names[:5])
            more = "" if invalid_count <= 5 else f" и ещё {invalid_count - 5}"
            self.validation_warning_label.setText(
                f"⚠️ Обнаружено {invalid_count} некорректных пар: {preview}{more}. "
                f"Некорректные пары НЕ будут включены в датасет."
            )
            self.validation_warning_label.setVisible(True)
            self.log(f"Валидация: {valid_count} корректных, {invalid_count} некорректных")
        else:
            self.validation_warning_label.setVisible(False)
            self.log(f"Валидация: все {valid_count} пар корректны")

        self._update_color_stats_label()

    # ==================================================================
    # Флаги
    # ==================================================================
    def _on_include_empty_toggled(self, *_):
        if not self.pairs:
            return
        self._rebuild_file_list_display()

    def _is_include_empty(self):
        return bool(self.include_empty_annotations.isChecked())

    # ==================================================================
    # Смена типа датасета
    # ==================================================================
    def _on_dataset_type_changed(self, idx):
        self.dataset_type = idx
        self.validator.set_dataset_type(idx)
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
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
            self.validation_warning_label.setVisible(False)
            self._yaml_mode = False
            self._update_color_stats_label()
            self.update_generate_button_state()
            self.update_split_counts()
        if self.images_folder and not self._yaml_mode:
            self.scan_pairs()

    # ==================================================================
    # Множители и split
    # ==================================================================
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
                    self.log(f"Предупреждение: выбранный множитель {self.aug_multiplier_slider.value()} "
                             f"может привести к дублированию, так как максимальное количество "
                             f"уникальных вариантов — {max_unique}.")
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

    # ==================================================================
    # Выбор папок
    # ==================================================================
    def select_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self._yaml_mode = False
            self.images_folder = folder
            self.log(f"Images folder: {folder}")
            self.validator.clear_cache()
            self.scan_pairs()

    def select_labels_folder(self):
        if self.dataset_type == 2:
            return
        self._yaml_mode = False
        self.validator.clear_cache()
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

    # ==================================================================
    # Сканирование пар
    # ==================================================================
    def scan_pairs(self):
        if self._yaml_mode:
            self.validator.clear_cache()
            self._recompute_color_stats()
            self._rebuild_file_list_display()
            return

        self.pairs = []
        self.file_list.clear()
        self.annotation_types_stats = {}
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
        self.validator.clear_cache()

        if not self.images_folder:
            self.log("Папка с изображениями не выбрана")
            self._update_color_stats_label()
            return

        dataset_type_idx = self.dataset_type_combo.currentIndex()
        self.log(f"Начинаем сканирование пар для типа {dataset_type_idx}")

        if dataset_type_idx == 0:
            if not self.labels_folder:
                self.log("Папка с метками не выбрана")
                self._update_pair_count_label("", error=True)
                self._update_color_stats_label()
                return
            self.pairs, self.annotation_types_stats = scan_detection_pairs(
                self.images_folder, self.labels_folder, log_cb=self.log
            )

        elif dataset_type_idx == 1:
            if not self.masks_folder:
                self.log("Папка с масками не выбрана")
                self._update_pair_count_label("", error=True)
                self._update_color_stats_label()
                return
            self.pairs = scan_segmentation_pairs(
                self.images_folder, self.masks_folder, log_cb=self.log
            )
            self.log(f"Найдено {len(self.pairs)} валидных пар изображение↔маска")

        else:
            self.pairs = scan_image_only_pairs(self.images_folder)
            self.annotation_types_stats = {}

        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

        error_msg = PairValidator.check_pair_type_consistency()
        extra_info = ""
        if self.annotation_types_stats:
            type_str = ", ".join([f"{k}:{v}" for k, v in self.annotation_types_stats.items()])
            extra_info = f"  [{type_str}]"
        if invalid_count > 0:
            extra_info += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra_info, error=bool(error_msg) or invalid_count > 0)
        self.log(f"Найдено пар: {len(self.pairs)} (корректных: {valid_count}, некорректных: {invalid_count})")
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

    # ==================================================================
    # Классы
    # ==================================================================
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

    def _get_yaml_name_for_id(self, cid):
        names = self._yaml_names
        if not names:
            return None
        try:
            if isinstance(names, dict):
                if cid in names:
                    return str(names[cid])
                if str(cid) in names:
                    return str(names[str(cid)])
            elif isinstance(names, (list, tuple)):
                if 0 <= cid < len(names):
                    return str(names[cid])
        except Exception:
            pass
        return None

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

                yaml_name = self._get_yaml_name_for_id(cid)
                name_item = QTableWidgetItem(yaml_name if yaml_name else str(cid))
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
                        self.log(f"Предупреждение: не удалось прочитать маску {mask_path}")
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

    # ==================================================================
    # Генерация
    # ==================================================================
    def on_generate(self):
        if not self.output_folder:
            QMessageBox.warning(self, "Error", "Please select output folder!")
            return
        if len(self.pairs) == 0:
            QMessageBox.warning(self, "Error", "No pairs found!")
            return

        has_labels_in_pairs = bool(self.pairs[0][1]) if self.pairs else False
        if self.dataset_type_combo.currentIndex() == 0 and not has_labels_in_pairs and not self.labels_folder:
            QMessageBox.warning(self, "Error", "Please select labels folder for detection!")
            return
        if self.dataset_type_combo.currentIndex() == 1 and not has_labels_in_pairs and not self.masks_folder:
            QMessageBox.warning(self, "Error", "Please select masks folder for segmentation!")
            return

        _, invalid_count = self._validate_all_pairs()
        if invalid_count > 0:
            reply = QMessageBox.question(
                self, "Некорректные пары",
                f"Обнаружено {invalid_count} некорректных пар.\n"
                f"Они будут автоматически исключены из датасета.\n\n"
                f"Продолжить генерацию?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

        error_msg = PairValidator.check_pair_type_consistency()
        if error_msg:
            reply = QMessageBox.question(self, "Несоответствие типа",
                                         f"{error_msg}\nВсё равно продолжить?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # --- Проверка битности ---
        non_8bit = (self.bit_type_stats.get('uint16', 0) +
                    self.bit_type_stats.get('int16', 0) +
                    self.bit_type_stats.get('float', 0) +
                    self.bit_type_stats.get('other', 0))
        if non_8bit > 0:
            reply = QMessageBox.question(
                self, "Обнаружены не-8bit изображения",
                f"В датасете {non_8bit} изображений не являются 8-битными.\n\n"
                f"При обучении YOLO может упасть с ошибкой\n"
                f"«Unsupported depth of input image: CV_16U».\n\n"
                f"Рекомендуется сначала нажать «Перевести в 8 бит PNG».\n\n"
                f"Всё равно продолжить?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
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
            'random_rotate': self.random_rotate.isChecked(),
            'random_rotate_angle': self.random_rotate_angle.value(),
            'random_crop': self.random_crop.isChecked(),
            'shear': self.shear.isChecked(),
            'shear_angle': self.shear_angle.value()
        }
        has_augmentations = any(v for k, v in augmentations.items()
                                if k not in ['random_rotate_angle', 'shear_angle'])
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

        selected_channels = self._get_selected_channels()
        if selected_channels == 1:
            color_count = self.color_type_stats.get('color', 0)
            if color_count > 0:
                reply = QMessageBox.question(
                    self, "Channels: 1 (Grayscale)",
                    f"Выбран режим channels: 1 (Grayscale).\n"
                    f"В датасете {color_count} реально цветных изображений — "
                    f"при обучении они будут прочитаны как Ч/Б (цвет потеряется).\n\n"
                    f"Рекомендуется сначала нажать «Конвертировать всё в ЧБ».\n\n"
                    f"Продолжить генерацию?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        class_names = self.get_class_names_from_table()
        num_classes = len(class_names)

        included_orig_ids = None
        if self.dataset_type == 1:
            included_orig_ids = self.included_orig_ids.copy()

        include_empty_annotations = self.include_empty_annotations.isChecked()
        delete_source_after_success = self.delete_source_after_success.isChecked()

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
                                 "Все пары оказались некорректными. Генерация невозможна.")
            return

        self.log("=== Dataset generation started ===")
        self.log(f"Train: {train_pct*100:.0f}%, Val: {val_pct*100:.0f}%, Test: {test_pct*100:.0f}%")
        self.log(f"Image size: {'keep original' if target_size is None else target_size}")
        self.log(f"Augmentation: {'enabled' if has_augmentations else 'disabled'}")
        if has_augmentations:
            self.log(f"Augmentation multiplier: {aug_multiplier}x")
        self.log(f"Classes: {num_classes}")
        self.log(f"Channels (для data.yaml): {selected_channels}")
        self.log(f"Include empty annotations: {include_empty_annotations}")
        self.log(f"Delete source files after success: {delete_source_after_success}")
        if skipped_invalid:
            self.log(f"Пропущено некорректных пар: {skipped_invalid}")

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
            self.output_folder, pairs_for_generation,
            train_pct, val_pct, test_pct, target_size,
            augmentations, has_augmentations, aug_multiplier,
            class_names, class_mapping, bg_color,
            self.dataset_type_combo.currentIndex(),
            self.masks_folder,
            num_classes,
            self.mask_class_remap,
            included_orig_ids=included_orig_ids,
            included_class_ids=None,
            total_orig_classes=len(self.original_ids),
            resize_mode=resize_mode,
            include_empty_annotations=include_empty_annotations,
            delete_source_after_success=delete_source_after_success,
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

            yaml_path = self._find_dataset_yaml(self.output_folder)
            if yaml_path:
                self._inject_channels_into_yaml(yaml_path, self._get_selected_channels())
            else:
                self.log("⚠️ data.yaml не найден в папке вывода — channels не прописан")

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