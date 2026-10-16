# preparing_dataset_yaml.py
from import_libs_internal import *
from import_libs_methods_ui import setup_preparing_dataset_yaml_ui


ULTRALYTICS_AVAILABLE = False
try:
    version = ultralytics.__version__
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class DatasetPreparationWindow(QMainWindow):
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

        # --- YAML-режим ---
        self._last_patched_yaml = None
        self._yaml_mode = False
        self._yaml_root = None
        self._yaml_path = None
        self._yaml_names = None

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
    # Логирование
    # ==================================================================
    def _log_debug(self, msg):
        try:
            print(f"[DEBUG] {msg}")
        except Exception:
            pass
        try:
            self.log(f"[DEBUG] {msg}")
        except Exception:
            pass

    def _log_exception(self, context, exc=None):
        tb = traceback.format_exc()
        head = f"❌ {context}: {exc if exc is not None else ''}"
        try:
            print(head)
            print(tb)
        except Exception:
            pass
        try:
            self.log(head)
            self.log(tb)
        except Exception:
            pass

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
        try:
            if item is None:
                return
            raw_data = item.data(Qt.UserRole + 1)
            if not raw_data:
                return
            self._open_pair_validation(raw_data)
        except Exception as e:
            self._log_exception("Ошибка в _on_file_list_double_clicked", e)

    # ==================================================================
    # Валидация
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
                f"⏳ Проверка пар: {current}/{tot} ({percent}%)")
            QApplication.processEvents()

        valid, invalid = self.validator.validate_all(
            self.pairs, progress_cb=on_progress)

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
            self.color_type_stats, self.bit_type_stats, bool(self.pairs))
        self.color_stats_label.setText(text)
        for name in ('btn_convert_gray', 'btn_convert_rgb',
                     'btn_convert_8bit_png'):
            if hasattr(self, name):
                getattr(self, name).setEnabled(enabled)

    def _open_pair_validation(self, pair):
        try:
            img_path = None
            aux_path = None
            if isinstance(pair, str):
                img_path = pair
            elif isinstance(pair, (tuple, list)) and len(pair) >= 1:
                img_path = pair[0]
                if len(pair) >= 2:
                    aux_path = pair[1]

            if not img_path or not os.path.exists(str(img_path)):
                QMessageBox.warning(self, "Файл не найден",
                                     f"Изображение не существует:\n{img_path}")
                return

            dlg = PairValidationDialog(
                self, (str(img_path), str(aux_path) if aux_path else None),
                self.dataset_type, self.validator)
            dlg.exec_()
        except Exception as e:
            self._log_exception("Ошибка валидации пары", e)
            QMessageBox.critical(self, "Ошибка валидации",
                                  f"{type(e).__name__}: {e}\n\nСм. лог.")

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
    # Load YOLO YAML
    # ==================================================================
    def load_from_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO data.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not file_path:
            return
        if is_unet_yaml(file_path):
            reply = QMessageBox.question(
                self, "Похоже на U-Net YAML",
                f"Файл {os.path.basename(file_path)} похож на U-Net-датасет "
                f"(есть masks-поля). Загрузить его как U-Net?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self._load_unet_from_path(file_path)
                return
        self._load_yolo_yaml_from_path(file_path)

    def _load_yolo_yaml_from_path(self, file_path):
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
                                "В YAML не найдено ни одной существующей папки "
                                "с изображениями.\nПроверено path: " + root)
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
                f"Найдено {len(all_images)} изображений, но ни одного .txt.\n"
                f"Ожидается структура images/... и labels/...")
            return

        self._yaml_mode = True
        self._yaml_root = root
        self._yaml_path = file_path
        self._yaml_names = cfg.get('names')

        self.images_folder = root
        self.labels_folder = root
        self.masks_folder = None

        self._apply_loaded_pairs(
            pairs=new_pairs,
            input_type=0,
            input_format='auto',
            class_names_from_yaml=self._yaml_names,
            source_label=f"YAML: {file_path}",
        )

        self.log(f"=== Загрузка из YAML: {file_path} ===")
        self.log(f"Root: {root}")
        self.log(f"Сплитов найдено: {len(splits_dirs)}")
        self.log(f"Изображений: {len(all_images)}, пар с метками: {len(new_pairs)}")
        if missing:
            self.log(f"⚠️ Изображений без .txt: {missing}")

    # ==================================================================
    # Load COCO
    # ==================================================================
    # ==================================================================
    # Load / Add COCO
    # ==================================================================
    def load_from_coco(self):
        """Загрузка COCO вместо текущего датасета."""
        self._handle_load_coco(replace=True)

    def add_from_coco(self):
        """Добавление COCO к уже загруженному датасету."""
        self._handle_load_coco(replace=False)

    def _handle_load_coco(self, replace):
        choice = self._ask_coco_source()
        if choice is None:
            return

        if choice == 'files':
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select COCO JSON(s) — можно несколько (train/val/test)",
                "",
                "JSON files (*.json);;All files (*)")
            if not paths:
                return
        else:  # 'folder'
            folder = QFileDialog.getExistingDirectory(
                self, "Select folder with COCO JSONs (recursive search)")
            if not folder:
                return
            paths = self._find_coco_json_in_folder(folder)
            if not paths:
                QMessageBox.warning(
                    self, "COCO",
                    f"В папке не найдено COCO JSON.\n{folder}")
                return
            self.log(f"[COCO] Найдено {len(paths)} JSON в {folder}")

        self._load_coco_from_paths(paths, replace=replace)

    def _ask_coco_source(self):
        """Спрашивает у пользователя, как выбирать JSON. Возвращает 'files' | 'folder' | None."""
        box = QMessageBox(self)
        box.setWindowTitle("COCO")
        box.setText("Как выбрать COCO JSON?")
        btn_files = box.addButton("Select files...", QMessageBox.AcceptRole)
        btn_folder = box.addButton("Select folder...", QMessageBox.AcceptRole)
        box.addButton("Отмена", QMessageBox.RejectRole)
        box.setDefaultButton(btn_files)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_files:
            return 'files'
        if clicked is btn_folder:
            return 'folder'
        return None

    def _find_coco_json_in_folder(self, folder):
        """
        Рекурсивно ищет .json-файлы, которые похожи на COCO:
        словарь с ключами 'images' и 'annotations'.
        """
        candidates = []
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if not f.lower().endswith('.json'):
                    continue
                p = os.path.join(root, f)
                try:
                    with open(p, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    if (isinstance(data, dict)
                            and 'images' in data
                            and 'annotations' in data):
                        candidates.append(p)
                except Exception:
                    continue
        return sorted(candidates)

    def _get_coco_tmp_labels_dir(self, first_json_path=None):
        """
        Возвращает общую временную папку для .txt из COCO.
        Создаётся рядом с первым JSON, переиспользуется при Add.
        """
        existing = getattr(self, '_coco_tmp_labels_dir', None)
        if existing:
            return existing
        if first_json_path:
            base = os.path.dirname(first_json_path)
        else:
            base = os.path.expanduser('~')
        tmp = os.path.join(base, '_coco_labels_tmp')
        os.makedirs(tmp, exist_ok=True)
        self._coco_tmp_labels_dir = tmp
        return tmp

    def _load_coco_from_paths(self, json_paths, replace=True):
        """
        Загружает / добавляет COCO JSON(ы).

        replace=True  — пары и классы текущего датасета заменяются.
        replace=False — пары и классы текущего датасета объединяются с новыми.
                        Категории объединяются по имени: если 'cat' уже
                        есть, новый 'cat' получит тот же global id.
        """
        if not json_paths:
            return

        if replace:
            # Сбрасываем состояние COCO-загрузки.
            self._coco_tmp_labels_dir = None
            self._coco_json_counter = 0

        # --- 1. Читаем все JSON ---
        per_json_data = []
        for jp in json_paths:
            try:
                items, categories, root = load_coco(jp, log_cb=self.log)
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка COCO",
                    f"Не удалось загрузить:\n{jp}\n\n{e}")
                return
            if not items:
                self.log(f"⚠️ {os.path.basename(jp)}: нет изображений — пропуск.")
                continue
            per_json_data.append((jp, items, categories, root))

        if not per_json_data:
            QMessageBox.warning(self, "COCO", "Не найдено ни одного изображения.")
            return

        # --- 2. Существующие классы (при Add) ---
        if replace:
            name_to_global = {}
            next_gid = 0
            base_pairs = []
        else:
            name_to_global = {}
            max_gid = -1
            for row in range(self.class_table.rowCount()):
                try:
                    gid = int(self.class_table.item(row, 0).text())
                    name_item = self.class_table.item(row, 2)
                    if name_item is None:
                        continue
                    name = name_item.text().strip() or str(gid)
                    name_to_global[name] = gid
                    if gid > max_gid:
                        max_gid = gid
                except (ValueError, TypeError, AttributeError):
                    continue
            next_gid = max_gid + 1
            base_pairs = list(self.pairs)

        # --- 3. Обработка ---
        tmp_dir = self._get_coco_tmp_labels_dir(per_json_data[0][0])
        new_pairs = []

        for idx, (jp, items, categories, _root) in enumerate(per_json_data):
            local_map = {}
            for coco_cat_id, name in categories.items():
                name_str = str(name) if name is not None else str(coco_cat_id)
                if name_str not in name_to_global:
                    name_to_global[name_str] = next_gid
                    next_gid += 1
                local_map[int(coco_cat_id)] = name_to_global[name_str]

            for it in items:
                new_anns = []
                for a in it['annotations']:
                    coco_cat_id = int(a[1]) + 1
                    gid = local_map.get(coco_cat_id)
                    if gid is None:
                        name_str = str(coco_cat_id)
                        if name_str not in name_to_global:
                            name_to_global[name_str] = next_gid
                            next_gid += 1
                        gid = name_to_global[name_str]
                        local_map[coco_cat_id] = gid
                    new_anns.append((a[0], gid, *a[2:]))
                it['annotations'] = new_anns

            # Уникальный префикс — счётчик глобальный по сессии.
            if not hasattr(self, '_coco_json_counter'):
                self._coco_json_counter = 0
            prefix = f"j{self._coco_json_counter:03d}_"
            self._coco_json_counter += 1

            try:
                pairs_i = write_yolo_txt_for_coco_items(
                    items, tmp_dir,
                    class_remap=None,
                    log_cb=self.log,
                    name_prefix=prefix,
                )
                new_pairs.extend(pairs_i)
            except Exception as e:
                self._log_exception(f"Ошибка записи .txt из {jp}", e)
                return

        # --- 4. Классы ---
        class_names = {gid: name for name, gid in name_to_global.items()}

        # --- 5. Пары ---
        self.pairs = (new_pairs if replace else base_pairs + new_pairs)

        # --- 6. Окно ---
        self.images_folder = os.path.dirname(json_paths[0])
        self.labels_folder = tmp_dir
        self.masks_folder = None

        self.dataset_type = 0
        self.input_format = 'auto'
        self.validator.set_dataset_type(0)
        self.validator.set_input_format('auto')

        self.dataset_type_combo.blockSignals(True)
        self.input_format_combo.blockSignals(True)
        try:
            self.dataset_type_combo.setCurrentIndex(0)
            idx = self.input_format_combo.findData('auto')
            if idx >= 0:
                self.input_format_combo.setCurrentIndex(idx)
        finally:
            self.dataset_type_combo.blockSignals(False)
            self.input_format_combo.blockSignals(False)

        if hasattr(self, '_sync_all_states'):
            self._sync_all_states()

        # --- 7. Валидация + список ---
        self.validator.clear_cache()
        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

        # --- 8. Классы в таблицу (сохраняя Include по именам) ---
        self._fill_class_table_from_dict(class_names, preserve_include=True)

        # --- 9. Лог ---
        action = "Загрузка" if replace else "Добавление"
        self.log(f"=== {action} COCO ({len(per_json_data)} файлов) ===")
        for jp, items, _, _ in per_json_data:
            self.log(f"  • {os.path.basename(jp)}: images={len(items)}")
        self.log(f"Итого: пар={len(self.pairs)}, классов={len(class_names)}")
        self.log(f"Временные labels: {tmp_dir}")

        extra = f"  [COCO: {len(per_json_data)} json, {action.lower()}]"
        if invalid_count > 0:
            extra += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra, error=invalid_count > 0)

        self.update_generate_button_state()
        self.update_split_counts()
    # ==================================================================
    # Load U-Net YAML
    # ==================================================================
    def load_from_unet_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net YAML", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not yaml_path:
            return
        self._load_unet_from_path(yaml_path)

    def _load_unet_from_path(self, yaml_path):
        try:
            data = load_unet_yaml(yaml_path, log_cb=self.log)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка U-Net YAML",
                                  f"Не удалось загрузить:\n{e}")
            return

        pairs = data['pairs']
        yolo_class_names = data['class_names']       # {0:'cat', 1:'dog', ...}
        mask_offset = data['mask_class_offset']
        root = data['root']

        if not pairs:
            QMessageBox.warning(self, "U-Net", "Не найдено ни одной пары.")
            return

        # ВАЖНО: приводим ключи к РАW PIXEL в маске.
        # yolo_cls i соответствует pixel (i + offset).
        pixel_class_names = {int(yc) + int(mask_offset): name
                             for yc, name in yolo_class_names.items()}

        self.images_folder = root
        self.labels_folder = None
        self.masks_folder = root
        self.mask_pixel_offset_in = 0  # ключи уже задают сырой пиксель

        ui_pairs = [(img, mask) for (img, mask) in pairs]

        self._apply_loaded_pairs(
            pairs=ui_pairs,
            input_type=1,
            input_format='auto',
            class_names_override=pixel_class_names,
            source_label=f"U-Net YAML: {yaml_path}",
        )
        self.log(f"=== Загрузка из U-Net YAML: {yaml_path} ===")
        self.log(f"Root: {root}")
        self.log(f"Пар: {len(pairs)}, классов: {len(yolo_class_names)}, "
                 f"mask_class_offset: {mask_offset}")
        self.log(f"Пиксели в маске: {sorted(pixel_class_names.keys())}")

    # ==================================================================
    # Общая инициализация после загрузки
    # ==================================================================
    def _apply_loaded_pairs(self, pairs, input_type, input_format,
                              class_names_from_yaml=None,
                              class_names_override=None,
                              source_label=""):
        # Если грузим НЕ COCO — сбрасываем COCO-временное состояние.
        if input_type != 0 or class_names_override is None:
            self._coco_tmp_labels_dir = None
            self._coco_json_counter = 0
        self.validator.clear_cache()
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
        self.annotation_types_stats = {}

        self.dataset_type = input_type
        self.input_format = input_format
        self.validator.set_dataset_type(input_type)
        self.validator.set_input_format(input_format)

        self.dataset_type_combo.blockSignals(True)
        self.input_format_combo.blockSignals(True)
        try:
            self.dataset_type_combo.setCurrentIndex(input_type)
            idx = self.input_format_combo.findData(input_format)
            if idx >= 0:
                self.input_format_combo.setCurrentIndex(idx)
        finally:
            self.dataset_type_combo.blockSignals(False)
            self.input_format_combo.blockSignals(False)

        if hasattr(self, '_sync_all_states'):
            self._sync_all_states()

        self.pairs = list(pairs)

        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._rebuild_file_list_display()

        self._yaml_names = class_names_from_yaml
        if class_names_override:
            self._fill_class_table_from_dict(class_names_override)
        else:
            self.collect_classes()

        extra = ""
        if source_label:
            extra = f"  [{source_label}]"
        if invalid_count > 0:
            extra += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra, error=invalid_count > 0)
        self.log(f"Пар: {len(self.pairs)} "
                 f"(корректных: {valid_count}, некорректных: {invalid_count})")

        self.update_generate_button_state()
        self.update_split_counts()

    def _fill_class_table_from_dict(self, class_names_dict,
                                     preserve_include=True):
        """
        class_names_dict: {gid: name}.
        preserve_include=True — сохраняет текущие чекбоксы Include
        по именам классов (для Add COCO).
        """
        include_by_name = {}
        if preserve_include and self.class_table.rowCount() > 0:
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 4)
                name_item = self.class_table.item(row, 2)
                if cb is None or name_item is None:
                    continue
                include_by_name[name_item.text().strip()] = cb.isChecked()

        self.class_table.blockSignals(True)
        try:
            self.class_table.setRowCount(0)
            if not class_names_dict:
                self.original_ids = []
                self.update_included_classes()
                return

            sorted_ids = sorted(class_names_dict.keys())
            self.original_ids = list(sorted_ids)
            self.class_table.setRowCount(len(sorted_ids))
            for i, cid in enumerate(sorted_ids):
                new_item = QTableWidgetItem(str(i))
                new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, new_item)

                orig_item = QTableWidgetItem(str(cid))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)

                name_str = str(class_names_dict[cid])
                name_item = QTableWidgetItem(name_str)
                self.class_table.setItem(i, 2, name_item)

                cb = QCheckBox()
                cb.setChecked(include_by_name.get(name_str, True))
                cb.stateChanged.connect(
                    lambda state, row=i: self.update_included_classes())
                self.class_table.setCellWidget(i, 4, cb)
        finally:
            self.class_table.blockSignals(False)

        self.update_included_classes()
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
        self._rebuild_file_list_display()

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
        self._rebuild_file_list_display()

    # ==================================================================
    # Список пар
    # ==================================================================
    def _make_list_item(self, text, tag=None, error_msg=""):
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
            try:
                info = build_pair_display(
                    pair, self.validator, self.dataset_type, include_empty)
            except Exception as e:
                self._log_exception(
                    f"build_pair_display упал на паре #{idx} ({pair!r})", e)
                info = {
                    'text': f"<ошибка построения: {pair!r}>",
                    'tag': 'error', 'err': f"{type(e).__name__}: {e}",
                    'is_valid': False, 'name': f"pair#{idx}",
                    'color_type': None, 'bit_type': None,
                }

            if info['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_names.append(info['name'])

            item = self._make_list_item(info['text'], tag=info['tag'],
                                          error_msg=info['err'])

            img_path = pair[0] if isinstance(pair, (tuple, list)) and pair else None
            aux_path = (pair[1] if isinstance(pair, (tuple, list)) and len(pair) > 1
                        else None)
            pair_norm = (img_path, aux_path)
            item.setData(Qt.UserRole, img_path)
            item.setData(Qt.UserRole + 1, pair_norm)
            item.setData(Qt.UserRole + 2,
                          'valid' if info['is_valid'] else 'invalid')
            item.setData(Qt.UserRole + 3, info['color_type'])
            item.setData(Qt.UserRole + 4, info['bit_type'])

            self.file_list.addItem(item)

            if (idx + 1) % step == 0 or idx == total - 1:
                percent = int((idx + 1) / total * 100)
                self.validation_progress_label.setText(
                    f"⏳ Построение списка: {idx + 1}/{total} ({percent}%)")
                QApplication.processEvents()

        self.file_list.blockSignals(False)
        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")

        if invalid_count > 0:
            preview = ", ".join(invalid_names[:5])
            more = "" if invalid_count <= 5 else f" и ещё {invalid_count - 5}"
            self.validation_warning_label.setText(
                f"⚠️ Обнаружено {invalid_count} некорректных пар: "
                f"{preview}{more}. Они НЕ будут включены в датасет.")
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
    # Смена типа датасета / input_format
    # ==================================================================
    def _on_dataset_type_changed(self, idx):
        new_type = self.dataset_type_combo.currentData()
        if new_type is None:
            new_type = idx
        self.dataset_type = new_type
        self.validator.set_dataset_type(new_type)

        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()

        if new_type == 2:
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

    def _on_out_masks_offset_changed(self, *_):
        self._refresh_mask_pixel_column()

    def _on_input_format_changed(self, idx):
        fmt = self.input_format_combo.currentData()
        if fmt is None:
            fmt = 'auto'
        self.input_format = fmt
        self.validator.set_input_format(fmt)
        if self.pairs:
            self._validate_all_pairs()
            self._rebuild_file_list_display()

    # ==================================================================
    # Множители / Split
    # ==================================================================
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
        if self.dataset_type == 0:
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

    # ==================================================================
    # Сканирование
    # ==================================================================
    def scan_pairs(self):
        try:
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

            dataset_type_idx = self.dataset_type

            if dataset_type_idx == 0:
                if not self.labels_folder:
                    self.log("Папка с метками не выбрана")
                    self._update_pair_count_label("", error=True)
                    return
                self.pairs, self.annotation_types_stats = scan_detection_pairs(
                    self.images_folder, self.labels_folder,
                    log_cb=self.log, format_hint=self.input_format)

            elif dataset_type_idx == 1:
                if not self.masks_folder:
                    self.log("Папка с масками не выбрана")
                    self._update_pair_count_label("", error=True)
                    return
                self.pairs = scan_segmentation_pairs(
                    self.images_folder, self.masks_folder, log_cb=self.log)

            else:
                self.pairs = scan_image_only_pairs(self.images_folder)
                self.annotation_types_stats = {}

            valid_count, invalid_count = self._validate_all_pairs()
            self._recompute_color_stats()
            self._rebuild_file_list_display()

            extra_info = ""
            if self.annotation_types_stats:
                type_str = ", ".join(
                    [f"{k}:{v}" for k, v in self.annotation_types_stats.items()])
                extra_info = f"  [{type_str}]"
            if invalid_count > 0:
                extra_info += f"  ⚠️ некорректных: {invalid_count}"
            self._update_pair_count_label(
                extra_info, error=invalid_count > 0)
            self.log(f"Найдено пар: {len(self.pairs)} "
                     f"(корректных: {valid_count}, некорректных: {invalid_count})")

            self.collect_classes()
            self.update_generate_button_state()
            self.update_split_counts()
        except Exception as e:
            self._log_exception("Ошибка в scan_pairs", e)

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
            cb = self.class_table.cellWidget(row, 4)
            if cb and cb.isChecked():
                try:
                    original = int(self.class_table.item(row, 1).text())
                    new = int(self.class_table.item(row, 0).text())
                    self.class_remap[original] = new
                except (ValueError, TypeError):
                    continue

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

                if input_type == 1:
                    # input=masks: выходной пиксель = dense New ID.
                    pixel = new_id
                else:
                    # input=labels + out_masks: pixel = New ID + offset.
                    pixel = new_id + offset

                item = QTableWidgetItem(str(pixel))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(row, 3, item)
        finally:
            self.class_table.blockSignals(False)

    def update_included_classes(self):
        if self.dataset_type == 0:
            self.included_orig_ids = set()
            included_rows = []
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 4)
                if cb and cb.isChecked():
                    try:
                        original = int(self.class_table.item(row, 1).text())
                        self.included_orig_ids.add(original)
                        included_rows.append((row, original))
                    except (ValueError, TypeError):
                        continue

            included_rows.sort(key=lambda t: t[1])
            self.class_remap = {}
            for new_id, (row, orig) in enumerate(included_rows):
                self.class_remap[orig] = new_id
                try:
                    item = self.class_table.item(row, 0)
                    if item is not None:
                        self.class_table.blockSignals(True)
                        item.setText(str(new_id))
                        self.class_table.blockSignals(False)
                except Exception:
                    pass

            self.unique_classes = list(range(len(self.class_remap)))
            self.num_classes = len(self.class_remap)
            self.mask_class_remap = {}

        elif self.dataset_type == 1:
            if self.class_table.rowCount() == 0:
                return
            self.included_orig_ids = set()
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 4)
                if cb and cb.isChecked():
                    try:
                        orig_id = int(self.class_table.item(row, 1).text())
                        self.included_orig_ids.add(orig_id)
                    except (ValueError, TypeError, AttributeError):
                        continue
            sorted_included = sorted(self.included_orig_ids)
            self.mask_class_remap = {old: new
                                       for new, old in enumerate(sorted_included)}
            self.unique_classes = list(range(len(sorted_included)))
            self.num_classes = len(self.unique_classes)
            for row in range(self.class_table.rowCount()):
                try:
                    orig_id = int(self.class_table.item(row, 1).text())
                    self.class_table.blockSignals(True)
                    if orig_id in self.mask_class_remap:
                        self.class_table.item(row, 0).setText(
                            str(self.mask_class_remap[orig_id]))
                    else:
                        self.class_table.item(row, 0).setText('-')
                    self.class_table.blockSignals(False)
                except Exception:
                    self.class_table.blockSignals(False)
        else:
            self.included_orig_ids = set()
            self.class_remap = {}
            self.mask_class_remap = {}
            self.unique_classes = []
            self.num_classes = 0
        self._refresh_mask_pixel_column()

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
            for pair in self.pairs:
                label_path = pair[1] if len(pair) > 1 else None
                if not label_path:
                    continue
                anns = load_annotations(label_path, 1, 1,
                                         format_hint=self.input_format)
                for ann in anns:
                    unique_ids.add(int(ann[1]))
            self.original_ids = sorted(unique_ids)
            self.class_table.blockSignals(True)
            self.class_table.setRowCount(len(self.original_ids))
            for i, cid in enumerate(self.original_ids):
                new_item = QTableWidgetItem(str(i))
                new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, new_item)

                orig_item = QTableWidgetItem(str(cid))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)

                yaml_name = self._get_yaml_name_for_id(cid)
                name_item = QTableWidgetItem(yaml_name if yaml_name else str(cid))
                self.class_table.setItem(i, 2, name_item)

                cb = QCheckBox()
                cb.setChecked(True)
                cb.stateChanged.connect(
                    lambda state, row=i: self.update_included_classes())
                self.class_table.setCellWidget(i, 4, cb)
            self.class_table.blockSignals(False)
            self.update_included_classes()
            self._refresh_mask_pixel_column()


        elif self.dataset_type == 1:
            unique_vals = set()
            valid_pairs_for_classes = []
            for pair in self.pairs:
                mask_path = pair[1] if len(pair) > 1 else None
                if not mask_path:
                    continue
                try:
                    mask = read_mask_with_fallback(str(mask_path))
                    if mask is None:
                        continue
                    if mask.size == 0:
                        continue
                    if mask.dtype not in (np.uint8, np.uint16):
                        mask = mask.astype(np.uint8)
                    uniq = np.unique(mask)
                    unique_vals.update(int(v) for v in uniq)
                    valid_pairs_for_classes.append((pair[0], mask_path))
                except Exception as e:
                    self.log(f"Ошибка маски {mask_path}: {e}")
            if not valid_pairs_for_classes:
                self.log("Нет валидных масок для определения классов")
                self.class_table.setRowCount(0)
                self.original_ids = []
                self.update_included_classes()
                return
            self.pairs = valid_pairs_for_classes
            all_vals = sorted(unique_vals)
            self.original_ids = all_vals
            self.class_table.blockSignals(True)
            self.class_table.setRowCount(len(all_vals))
            for i, old_id in enumerate(all_vals):
                new_item = QTableWidgetItem(str(i))
                new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 0, new_item)
                orig_item = QTableWidgetItem(str(old_id))
                orig_item.setFlags(orig_item.flags() & ~Qt.ItemIsEditable)
                self.class_table.setItem(i, 1, orig_item)
                name_item = QTableWidgetItem(str(old_id))
                self.class_table.setItem(i, 2, name_item)
                cb = QCheckBox()
                cb.setChecked(True)
                cb.stateChanged.connect(
                    lambda state, row=i: self.update_included_classes())
                self.class_table.setCellWidget(i, 4, cb)
            self.class_table.blockSignals(False)
            self.update_included_classes()

        else:
            self.class_table.setRowCount(0)
            self.original_ids = []
            self.update_included_classes()
            self._refresh_mask_pixel_column()

    def get_class_names_from_table(self):
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

    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetPreparationWindow()
    window.show()
    sys.exit(app.exec_())