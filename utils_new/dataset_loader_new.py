# -*- coding: utf-8 -*-
"""Qt DatasetLoaderMixin for Preparation/Analysis using V17 format/validation core.

UI attribute/callback contract stays compatible with the old mixin. Does not
modify other tabs. All label decoding/conversion is via utils_new.
"""
import os
import tempfile
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QTableWidgetItem,
)

from ui.pair_validation_dialog_new import PairValidationDialog
from utils_new.dataset_catalog_new import (
    CatalogSourceError,
    collect_label_classes,
    collect_mask_classes,
    dense_class_remap,
    discover_coco_json,
    import_coco_catalog,
    load_unet_catalog,
    load_yolo_catalog,
    resolve_class_name,
)
from utils_new.dataset_coco_new import write_yolo_txt_for_coco_items
from utils_new.dataset_image_stats_new import ImageStats
from utils_new.dataset_pair_tools_new import (
    build_pair_display,
    scan_detection_pairs,
    scan_image_only_pairs,
    scan_segmentation_pairs,
)
from utils_new.dataset_pair_validator_new import PairValidator
from utils_new.dataset_unet_new import is_unet_yaml


class DatasetLoaderMixin:
    """
    Общие методы загрузки и валидации для вкладок «Анализ» и «Подготовка».

    Подкласс должен:
      1. В __init__ вызвать setup_*_ui(self) (создаст нужные виджеты).
      2. Инициализировать атрибуты:
         images_folder, labels_folder, masks_folder, pairs, filtered_pairs,
         dataset_type, input_format, mask_pixel_offset_in, validator,
         color_type_stats, bit_type_stats, annotation_types_stats,
         unique_classes, original_ids, mask_class_remap, included_orig_ids,
         class_remap, num_classes,
         _yaml_mode, _yaml_root, _yaml_path, _yaml_names,
         _coco_tmp_labels_dir, _coco_json_counter, _updating_class_table.
      3. Переопределить хуки:
         _after_load_hook()             — после успешной загрузки/сканирования;
         _after_type_change_hook(new)   — после смены dataset_type.
      4. При необходимости переопределить _refresh_mask_pixel_column()
         и _sync_all_states().
    """

    # ---------------- хуки ----------------
    def _after_load_hook(self):
        """Вызывается в конце scan_pairs / _apply_loaded_pairs /
        _load_coco_from_paths / _on_input_format_changed.
        Переопределить в подклассе (Анализ: apply_filters_and_update_metrics;
        Подготовка: update_generate_button_state + update_split_counts)."""
        pass

    def _after_type_change_hook(self, new_type):
        """Вызывается в конце _on_dataset_type_changed. Переопределить."""
        pass

    # ---------------- синхронизация UI ----------------
    def _sync_all_states(self, *_):
        """Дефолт: синхронизирует input_format_combo и кнопку Load Labels.
        Подготовка переопределяет (там ещё output-group и аугментации)."""
        input_type = self.dataset_type_combo.currentData()
        self.input_format_combo.setEnabled(input_type == 0)
        if input_type == 0:
            self.btn_labels.setText("Load Labels")
            self.btn_labels.setEnabled(True)
        elif input_type == 1:
            self.btn_labels.setText("Load Masks")
            self.btn_labels.setEnabled(True)
        else:
            self.btn_labels.setText("No labels (images only)")
            self.btn_labels.setEnabled(False)
        self._refresh_mask_pixel_column()

    def _refresh_mask_pixel_column(self):
        """Дефолт: колонка Mask pixel видна только при input=masks.
        Подготовка переопределяет (там ещё out_masks и заполнение значений)."""
        input_type = getattr(self, 'dataset_type', 0)
        try:
            self.class_table.setColumnHidden(3, input_type != 1)
        except Exception:
            pass

    # ---------------- логирование ----------------
    def log(self, message):
        try:
            self.log_text.append(message)
            sb = self.log_text.verticalScrollBar()
            sb.setValue(sb.maximum())
            QApplication.processEvents()
        except Exception:
            print(f"[log] {message}")

    def _log_debug(self, msg):
        try:
            print(f"[DEBUG] {msg}")
            self.log(f"[DEBUG] {msg}")
        except Exception:
            pass

    def _log_exception(self, context, exc=None):
        tb = traceback.format_exc()
        head = f"❌ {context}: {exc if exc is not None else ''}"
        try:
            print(head)
            print(tb)
            self.log(head)
            self.log(tb)
        except Exception:
            pass

    # ---------------- клипборд / двойной клик ----------------
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

    # ---------------- выбор папок ----------------
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

    # ---------------- YOLO YAML ----------------
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
            catalog = load_yolo_catalog(file_path, log_cb=self.log)
            pairs = catalog.pairs
            names = catalog.class_names
            root = catalog.root
            splits_info = catalog.metadata['split_counts']
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать YAML:\n{e}")
            return

        if not pairs:
            QMessageBox.warning(self, "Ошибка",
                                "В YAML не найдено ни одной существующей папки с изображениями.")
            return

        missing = sum(1 for _, lbl in pairs if lbl is None)
        with_labels = [(img, lbl) for img, lbl in pairs if lbl is not None]
        if not with_labels:
            QMessageBox.warning(
                self, "Ошибка",
                f"Найдено {len(pairs)} изображений, но ни одного .txt.")
            return

        self._yaml_mode = True
        self._yaml_root = root
        self._yaml_path = file_path
        self._yaml_names = names

        self.images_folder = root
        self.labels_folder = root
        self.masks_folder = None

        self._apply_loaded_pairs(
            pairs=with_labels,
            input_type=0,
            input_format='auto',
            class_names_from_yaml=names,
            source_label=f"YAML: {file_path}",
        )

        self.log(f"=== Загрузка из YAML: {file_path} ===")
        self.log(f"Root: {root}")
        for split, n in splits_info.items():
            if n:
                self.log(f"  • {split}: {n} изображений")
        self.log(f"Пар с метками: {len(with_labels)}, без .txt: {missing}")

    # ---------------- COCO ----------------
    def load_from_coco(self):
        self._handle_load_coco(replace=True)

    def add_from_coco(self):
        self._handle_load_coco(replace=False)

    def _handle_load_coco(self, replace):
        choice = self._ask_coco_source()
        if choice is None:
            return
        if choice == 'files':
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select COCO JSON(s)", "", "JSON files (*.json);;All files (*)")
            if not paths:
                return
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select folder with COCO JSONs")
            if not folder:
                return
            paths = self._find_coco_json_in_folder(folder)
            if not paths:
                QMessageBox.warning(self, "COCO", f"В папке не найдено COCO JSON.\n{folder}")
                return
            self.log(f"[COCO] Найдено {len(paths)} JSON в {folder}")
        self._load_coco_from_paths(paths, replace=replace)

    def _ask_coco_source(self):
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
        return discover_coco_json(folder)


    def _get_coco_tmp_labels_dir(self, first_json_path=None):
        existing = getattr(self, '_coco_tmp_labels_dir', None)
        if existing:
            return existing
        tmp = tempfile.mkdtemp(prefix='threshold_coco_labels_')
        self._coco_tmp_labels_dir = tmp
        return tmp

    def _load_coco_from_paths(self, json_paths, replace=True):
        if not json_paths:
            return
        if replace:
            self._coco_tmp_labels_dir = None
            self._coco_json_counter = 0
            existing_names = {}
            base_pairs = []
        else:
            existing_names = {}
            for row in range(self.class_table.rowCount()):
                try:
                    gid = int(self.class_table.item(row, 0).text())
                    name_item = self.class_table.item(row, 2)
                    if name_item is None:
                        continue
                    existing_names[name_item.text().strip() or str(gid)] = gid
                except (ValueError, TypeError, AttributeError):
                    continue
            base_pairs = list(self.pairs)
        try:
            catalog = import_coco_catalog(json_paths, existing_names, log_cb=self.log)
        except CatalogSourceError as exc:
            QMessageBox.critical(self, 'Ошибка COCO', f'Не удалось загрузить:\n{exc}')
            return
        if not catalog.sources:
            QMessageBox.warning(self, 'COCO', 'Не найдено ни одного изображения.')
            return
        tmp_dir = self._get_coco_tmp_labels_dir(catalog.sources[0].path)
        new_pairs = []
        for source in catalog.sources:
            prefix = f'j{self._coco_json_counter:03d}_'
            self._coco_json_counter += 1
            try:
                new_pairs.extend(write_yolo_txt_for_coco_items(
                    source.items, tmp_dir, class_remap=None,
                    log_cb=self.log, name_prefix=prefix))
            except Exception as exc:
                self._log_exception(f'Ошибка записи .txt из {source.path}', exc)
                return
        class_names = catalog.class_names
        self.pairs = new_pairs if replace else base_pairs + new_pairs
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
        self._sync_all_states()
        self.validator.clear_cache()
        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()
        self._fill_class_table_from_dict(class_names, preserve_include=True)
        self._rebuild_file_list_display(pairs=self.pairs)
        action = 'Загрузка' if replace else 'Добавление'
        self.log(f'=== {action} COCO ({len(catalog.sources)} файлов) ===')
        for source in catalog.sources:
            self.log(f'  • {os.path.basename(source.path)}: images={len(source.items)}')
        self.log(f'Итого: пар={len(self.pairs)}, классов={len(class_names)}')
        extra = f'  [COCO: {len(catalog.sources)} json, {action.lower()}]'
        if invalid_count > 0:
            extra += f'  ⚠️ некорректных: {invalid_count}'
        self._update_pair_count_label(extra, error=invalid_count > 0)
        self._after_load_hook()


    # ---------------- U-Net YAML ----------------
    def load_from_unet_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net YAML", "", "YAML files (*.yaml *.yml);;All files (*)")
        if not yaml_path:
            return
        self._load_unet_from_path(yaml_path)

    def _load_unet_from_path(self, yaml_path):
        try:
            catalog = load_unet_catalog(yaml_path, log_cb=self.log)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка U-Net YAML",
                                 f"Не удалось загрузить:\n{e}")
            return

        pairs = catalog.pairs
        mask_offset = catalog.metadata['mask_class_offset']
        root = catalog.root

        if not pairs:
            QMessageBox.warning(self, "U-Net", "Не найдено ни одной пары.")
            return

        pixel_class_names = catalog.class_names
        self.images_folder = root
        self.labels_folder = None
        self.masks_folder = root
        self.mask_pixel_offset_in = 0

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
        self.log(f"Пар: {len(pairs)}, классов: {len(pixel_class_names)}, "
                 f"mask_class_offset: {mask_offset}")

    # ---------------- общая инициализация после загрузки ----------------
    def _apply_loaded_pairs(self, pairs, input_type, input_format,
                            class_names_from_yaml=None,
                            class_names_override=None,
                            source_label=""):
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

        self._sync_all_states()
        self.pairs = list(pairs)

        valid_count, invalid_count = self._validate_all_pairs()
        self._recompute_color_stats()

        self._yaml_names = class_names_from_yaml
        if class_names_override:
            self._fill_class_table_from_dict(class_names_override)
        else:
            self.collect_classes()

        self._rebuild_file_list_display(pairs=self.pairs)

        extra = ""
        if source_label:
            extra = f"  [{source_label}]"
        if invalid_count > 0:
            extra += f"  ⚠️ некорректных: {invalid_count}"
        self._update_pair_count_label(extra, error=invalid_count > 0)
        self.log(f"Пар: {len(self.pairs)} (корректных: {valid_count}, "
                 f"некорректных: {invalid_count})")

        self._after_load_hook()

    # ---------------- сканирование папок ----------------
    def scan_pairs(self):
        try:
            if self._yaml_mode:
                self.validator.clear_cache()
                self._recompute_color_stats()
                self._rebuild_file_list_display(pairs=self.pairs)
                self._after_load_hook()
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
                    self.log("Папка с метками не выбрана — пары пока не строим.")
                    self._update_pair_count_label("", error=False)
                    return
                self.log("=== Сканирование пар (детекция) ===")
                self.pairs, self.annotation_types_stats = scan_detection_pairs(
                    self.images_folder, self.labels_folder,
                    log_cb=self.log, format_hint=self.input_format)
            elif dataset_type_idx == 1:
                if not self.masks_folder:
                    self.log("Папка с масками не выбрана.")
                    self._update_pair_count_label("", error=False)
                    return
                self.log("=== Сканирование пар (сегментация) ===")
                self.pairs = scan_segmentation_pairs(
                    self.images_folder, self.masks_folder, log_cb=self.log)
            else:
                self.log("=== Сканирование изображений (без меток) ===")
                self.pairs = scan_image_only_pairs(self.images_folder)
                self.annotation_types_stats = {}

            self.log(f"Найдено пар до валидации: {len(self.pairs)}")
            valid_count, invalid_count = self._validate_all_pairs()
            self._recompute_color_stats()

            extra_info = ""
            if self.annotation_types_stats:
                type_str = ", ".join(f"{k}:{v}" for k, v in self.annotation_types_stats.items())
                extra_info = f"  [{type_str}]"
            if invalid_count > 0:
                extra_info += f"  ⚠️ некорректных: {invalid_count}"
            self._update_pair_count_label(extra_info, error=invalid_count > 0)
            self.log(f"Валидация: корректных={valid_count}, некорректных={invalid_count}")

            self.collect_classes()
            self._rebuild_file_list_display(pairs=self.pairs)
            self._after_load_hook()
        except Exception as e:
            self._log_exception("Ошибка в scan_pairs", e)

    # ---------------- валидация ----------------
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

        valid, invalid = self.validator.validate_all(self.pairs, progress_cb=on_progress)

        self.validation_progress_label.setVisible(False)
        self.validation_progress_label.setText("")
        QApplication.processEvents()
        return valid, invalid

    def _recompute_color_stats(self):
        try:
            self.color_type_stats, self.bit_type_stats = self.validator.recompute_stats(self.pairs)
        except Exception:
            pass

    def _update_color_stats_label(self):
        if not hasattr(self, 'color_stats_label'):
            return
        try:
            text, _ = ImageStats.format_stats_text(
                self.color_type_stats, self.bit_type_stats, bool(self.pairs))
            self.color_stats_label.setText(text)
        except Exception:
            pass

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

    # ---------------- отрисовка списка пар ----------------
    def _make_list_item(self, text, tag=None, error_msg=""):
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()
        if tag == 'invalid':
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Некорректная пара: {error_msg}")
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
            item.setToolTip("Смешанные типы аннотаций.")
        elif tag == 'unknown':
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _rebuild_file_list_display(self, pairs=None):
        if pairs is None:
            pairs = self.pairs

        if not pairs:
            self.file_list.clear()
            self.validation_warning_label.setVisible(False)
            self.validation_progress_label.setVisible(False)
            self._update_color_stats_label()
            return

        self.file_list.blockSignals(True)
        self.file_list.clear()
        valid_count = 0
        invalid_count = 0
        invalid_names = []
        total = len(pairs)
        step = max(1, total // 100)
        include_empty = self._is_include_empty()

        self.validation_progress_label.setVisible(True)
        self.validation_progress_label.setText(f"⏳ Построение списка: 0/{total}...")
        QApplication.processEvents()

        for idx, pair in enumerate(pairs):
            try:
                info = build_pair_display(pair, self.validator, self.dataset_type, include_empty)
            except Exception as e:
                self._log_exception(f"build_pair_display упал на паре #{idx}", e)
                info = {
                    'text': f"<ошибка: {pair!r}>", 'tag': 'error',
                    'err': f"{type(e).__name__}: {e}",
                    'is_valid': False, 'name': f"pair#{idx}",
                    'color_type': None, 'bit_type': None,
                }

            if info['is_valid']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_names.append(info['name'])

            item = self._make_list_item(info['text'], tag=info['tag'], error_msg=info['err'])

            img_path = pair[0] if isinstance(pair, (tuple, list)) and pair else None
            aux_path = (pair[1] if isinstance(pair, (tuple, list)) and len(pair) > 1 else None)
            item.setData(Qt.UserRole, img_path)
            item.setData(Qt.UserRole + 1, (img_path, aux_path))
            item.setData(Qt.UserRole + 2, 'valid' if info['is_valid'] else 'invalid')
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
                f"⚠️ Обнаружено {invalid_count} некорректных пар: {preview}{more}. "
                f"Они НЕ будут включены в датасет.")
            self.validation_warning_label.setVisible(True)
        else:
            self.validation_warning_label.setVisible(False)

        self._update_color_stats_label()

    def _is_include_empty(self):
        return bool(self.include_empty_annotations.isChecked())

    def _on_include_empty_toggled(self, *_):
        if not self.pairs:
            return
        self._rebuild_file_list_display(pairs=self.pairs)

    # ---------------- смена режима ----------------
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

        self._after_type_change_hook(new_type)

        if self.images_folder and not self._yaml_mode:
            self.scan_pairs()

    def _on_input_format_changed(self, idx):
        fmt = self.input_format_combo.currentData()
        if fmt is None:
            fmt = 'auto'
        self.input_format = fmt
        self.validator.set_input_format(fmt)
        if self.pairs:
            self._validate_all_pairs()
            self._rebuild_file_list_display(pairs=self.pairs)
            self._after_load_hook()

    # ---------------- работа с класс-таблицей ----------------
    def _fill_class_table_from_dict(self, class_names_dict, preserve_include=True):
        include_by_name = {}
        if preserve_include and self.class_table.rowCount() > 0:
            for row in range(self.class_table.rowCount()):
                cb = self.class_table.cellWidget(row, 4)
                name_item = self.class_table.item(row, 2)
                if cb is None or name_item is None:
                    continue
                include_by_name[name_item.text().strip()] = cb.isChecked()

        self._updating_class_table = True
        self.class_table.blockSignals(True)
        try:
            self.class_table.setRowCount(0)
            if not class_names_dict:
                self.original_ids = []
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
                cb.stateChanged.connect(self.update_included_classes)
                self.class_table.setCellWidget(i, 4, cb)
        finally:
            self.class_table.blockSignals(False)
            self._updating_class_table = False

        self.update_included_classes()

    def _get_yaml_name_for_id(self, cid):
        return resolve_class_name(self._yaml_names, cid)


    def collect_classes(self):
        if not self.pairs:
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(0)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            return

        if self.dataset_type == 0:
            unique_ids = collect_label_classes(self.pairs, self.input_format)
            self.original_ids = sorted(unique_ids)
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
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
                    cb.stateChanged.connect(self.update_included_classes)
                    self.class_table.setCellWidget(i, 4, cb)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False

            self.update_included_classes()

        elif self.dataset_type == 1:
            all_vals, valid_pairs_for_classes = collect_mask_classes(
                self.pairs, log_cb=self.log)
            if not valid_pairs_for_classes:
                self.original_ids = []
                self._updating_class_table = True
                self.class_table.blockSignals(True)
                try:
                    self.class_table.setRowCount(0)
                finally:
                    self.class_table.blockSignals(False)
                    self._updating_class_table = False
                return
            self.pairs = valid_pairs_for_classes
            self.original_ids = all_vals
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
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
                    cb.stateChanged.connect(self.update_included_classes)
                    self.class_table.setCellWidget(i, 4, cb)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            self.update_included_classes()

        else:
            self._updating_class_table = True
            self.class_table.blockSignals(True)
            try:
                self.class_table.setRowCount(0)
            finally:
                self.class_table.blockSignals(False)
                self._updating_class_table = False
            self.original_ids = []
            self.update_included_classes()

    def update_included_classes(self):
        if getattr(self, '_updating_class_table', False):
            return

        self._updating_class_table = True
        self.class_table.blockSignals(True)
        try:
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
                self.class_remap = dense_class_remap(orig for _row, orig in included_rows)
                included_set = set()
                for row, orig in included_rows:
                    new_id = self.class_remap[orig]
                    included_set.add(row)
                    item = self.class_table.item(row, 0)
                    if item is not None and item.text() != str(new_id):
                        item.setText(str(new_id))

                for row in range(self.class_table.rowCount()):
                    if row not in included_set:
                        item = self.class_table.item(row, 0)
                        if item is not None and item.text() != "-":
                            item.setText("-")

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
                self.mask_class_remap = dense_class_remap(sorted_included)
                self.unique_classes = list(range(len(sorted_included)))
                self.num_classes = len(self.unique_classes)
                for row in range(self.class_table.rowCount()):
                    try:
                        orig_id = int(self.class_table.item(row, 1).text())
                        item = self.class_table.item(row, 0)
                        if item is None:
                            continue
                        new_text = str(self.mask_class_remap[orig_id]) \
                            if orig_id in self.mask_class_remap else "-"
                        if item.text() != new_text:
                            item.setText(new_text)
                    except (ValueError, TypeError, AttributeError):
                        continue
            else:
                self.included_orig_ids = set()
                self.class_remap = {}
                self.mask_class_remap = {}
                self.unique_classes = []
                self.num_classes = 0
        finally:
            self.class_table.blockSignals(False)
            self._updating_class_table = False

        self._refresh_mask_pixel_column()

    def on_class_table_item_changed(self, item):
        if getattr(self, '_updating_class_table', False):
            return
        if item.column() == 0:
            self.update_included_classes()

    def _update_pair_count_label(self, extra_text, error=False):
        text = f"Всего пар: {len(self.pairs)}"
        if extra_text:
            text += f"  {extra_text}"
        self.pair_count_label.setText(text)
        if error:
            self.pair_count_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.pair_count_label.setStyleSheet("")

