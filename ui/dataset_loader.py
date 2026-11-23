# -*- coding: utf-8 -*-
"""Shared asynchronous dataset loading and cached presentation for dataset tabs."""

import os
import traceback

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QListWidgetItem,
    QMessageBox,
    QTableWidgetItem,
)

from ui import dataset_source as source_dialogs
from ui.pair_validation_dialog_new import PairValidationDialog
from utils_new.import_export.catalog import (
    dense_class_remap,
    resolve_class_name,
)
from utils_new.images.processing import ImageStats
from utils_new.import_export.mask_yaml import is_unet_yaml


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
         _updating_class_table.
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
        if hasattr(self, "dataset_source"):
            self.dataset_source.set_dataset_type(input_type)
        self._refresh_mask_pixel_column()

    def _refresh_mask_pixel_column(self):
        """Дефолт: колонка Mask pixel видна только при input=masks.
        Подготовка переопределяет (там ещё out_masks и заполнение значений)."""
        input_type = getattr(self, "dataset_type", 0)
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
        folder = source_dialogs.get_directory(self, "folder")
        if folder:
            self._yaml_mode = False
            self._source_yaml_path = None
            self.images_folder = folder
            self.log(f"Images folder: {folder}")
            self.validator.clear_cache()
            self.scan_pairs()

    def select_labels_folder(self):
        if self.dataset_type == 2:
            return
        kind = "labels" if self.dataset_type == 0 else "masks"
        folder = source_dialogs.get_directory(self, kind)
        if not folder:
            return
        self._yaml_mode = False
        self._source_yaml_path = None
        self.validator.clear_cache()
        if self.dataset_type == 0:
            self.labels_folder = folder
            self.masks_folder = None
        else:
            self.masks_folder = folder
            self.labels_folder = None
        self.log(f"{source_dialogs.ACTIONS[kind][0]} {folder}")
        self.scan_pairs()

    # ---------------- YOLO YAML ----------------
    def load_from_yaml(self):
        file_path, _ = source_dialogs.get_open_file(self, "yolo_yaml")
        if not file_path:
            return
        if is_unet_yaml(file_path):
            if self.dataset_type_combo.findData(1) < 0:
                QMessageBox.warning(
                    self,
                    "Неподдерживаемый источник",
                    "Эта вкладка не принимает индексные маски. Выберите YOLO YAML с TXT или COCO.",
                )
                return
            self._load_unet_from_path(file_path)
            return
        self._load_yolo_yaml_from_path(file_path)

    def _load_yolo_yaml_from_path(self, file_path):
        self._start_dataset_import('yolo', [file_path])

    # ---------------- COCO ----------------
    def load_from_coco(self):
        self._handle_load_coco(replace=True)

    def add_from_coco(self):
        self._handle_load_coco(replace=False)

    def _handle_load_coco(self, replace):
        paths = source_dialogs.choose_coco_sources(self)
        if paths:
            self._load_coco_from_paths(paths, replace=replace)

    def _load_coco_from_paths(self, json_paths, replace=True):
        self._start_dataset_import('coco', json_paths, replace=replace)

    # ---------------- U-Net YAML ----------------
    def load_from_unet_yaml(self):
        yaml_path, _ = source_dialogs.get_open_file(self, "unet_yaml")
        if not yaml_path:
            return
        self._load_unet_from_path(yaml_path)

    def _load_unet_from_path(self, yaml_path):
        if self.dataset_type_combo.findData(1) < 0:
            QMessageBox.warning(self, 'Источник', 'Эта вкладка не принимает индексные маски.')
            return
        self._start_dataset_import('unet', [yaml_path])

    def _dataset_task_manager(self):
        from ui.dataset_loading import dataset_tasks
        tasks = dataset_tasks(self)
        if not getattr(self, '_dataset_tasks_connected', False):
            tasks.settled.connect(self._on_dataset_task_settled)
            self._dataset_tasks_connected = True
        return tasks

    def _on_dataset_task_settled(self, key):
        if key == 'filters':
            self.progress_bar.setVisible(False)
            self.btn_export_filtered.setEnabled(self._filters_applied and bool(self.filtered_pairs))
        if key in ('load', 'inspect'):
            if key == 'load':
                self._loading_source = False
            self._after_loading_state_changed()

    def _after_loading_state_changed(self):
        if hasattr(self, 'update_generate_button_state'):
            self.update_generate_button_state()

    def _dataset_ready(self):
        return (not getattr(self, '_loading_source', False)
                and getattr(self, '_inspection_result', None) is not None)

    def _start_dataset_import(self, kind, paths, replace=True, options=None):
        from utils_new.import_export.catalog import DatasetImportJob
        tasks = self._dataset_task_manager()
        tasks.invalidate()
        self._loading_source = True
        self._after_loading_state_changed()
        options = dict(options or {})
        base_pairs = [] if replace else list(self.pairs)
        if kind == 'coco' and not replace:
            options['existing_names'] = {self.class_table.item(row, 2).text(): int(self.class_table.item(row, 0).text())
                                        for row in range(self.class_table.rowCount())}
        def loaded(result):
            catalog = result['catalog']
            if not catalog.pairs:
                self.log('В выбранном источнике не найдено снимков; текущий набор сохранён.')
                return
            self._loading_source = False
            self._yaml_mode = kind not in ('folder', 'images')
            self._yaml_root = catalog.root
            self._yaml_path = paths[0] if self._yaml_mode else None
            self.images_folder = catalog.root
            if kind == 'folder':
                pass
            elif kind == 'unet':
                self.labels_folder, self.masks_folder = None, catalog.root
                self.mask_pixel_offset_in = 0
            else:
                self.labels_folder, self.masks_folder = catalog.root, None
            self._apply_loaded_pairs(base_pairs + catalog.pairs, catalog.dataset_type, catalog.input_format,
                class_names_from_yaml=catalog.class_names,
                source_label=source_dialogs.display_dataset_source(paths[0]) if self._yaml_mode else catalog.root,
                source_yaml=paths[0] if self._yaml_mode else None)
        tasks.start('load', DatasetImportJob, kind, paths, options,
                    label='Загрузка списка файлов / разметки…', finished=loaded)

    # ---------------- общая инициализация после загрузки ----------------
    def _apply_loaded_pairs(
        self,
        pairs,
        input_type,
        input_format,
        class_names_from_yaml=None,
        class_names_override=None,
        source_label="",
        source_yaml=None,
    ):
        self._source_yaml_path = source_yaml
        self.dataset_type, self.input_format = input_type, input_format
        self.validator.set_dataset_type(input_type)
        self.validator.set_input_format(input_format)
        self.pairs = list(pairs)
        self._yaml_names = class_names_override or class_names_from_yaml or {}
        self.dataset_type_combo.blockSignals(True)
        self.input_format_combo.blockSignals(True)
        try:
            self.dataset_type_combo.setCurrentIndex(input_type)
            self.input_format_combo.setCurrentIndex(max(0, self.input_format_combo.findData(input_format)))
        finally:
            self.dataset_type_combo.blockSignals(False)
            self.input_format_combo.blockSignals(False)
        self._sync_all_states()
        if hasattr(self, '_invalidate_analysis_results'):
            self._invalidate_analysis_results()
        self._inspection_result = None
        self.color_type_stats = ImageStats.empty_color_stats()
        self.bit_type_stats = ImageStats.empty_bit_stats()
        self._pair_display_cache = {}
        self._fill_class_table_from_dict(self._yaml_names, preserve_include=False)
        self._update_pair_count_label('Проверка и статистика в фоне…')
        self.log(f'Список загружен: {len(self.pairs)} снимков. {source_label}. Проверки выполняются в фоне.')
        self._start_pair_inspection()

    def _start_pair_inspection(self):
        self._inspection_result = None
        if hasattr(self, '_overlap_review_panel'):
            self._overlap_review_panel.set_dataset([])
        self._review_dataset_key = None
        self._pair_display_cache = {}
        self._rebuild_file_list_display()
        self._inspection_revision = getattr(self, "_inspection_revision", 0) + 1
        from utils_new.dataset.pairs import PairInspectionJob
        self._after_loading_state_changed()
        tasks = self._dataset_task_manager()
        tasks.invalidate('filters')
        def inspected(result):
            self._inspection_result = result
            self._pair_display_cache = {tuple(pair): info for pair, info in zip(result['pairs'], result['rows'])}
            self.validator._cache = result['validation_cache']
            self.color_type_stats = result['colors']
            self.bit_type_stats = result['bits']
            self.annotation_types_stats = result['types']
            ids = set(result['classes']) | set(self._yaml_names or {})
            current_names = {int(self.class_table.item(row, 1).text()): self.class_table.item(row, 2).text()
                             for row in range(self.class_table.rowCount())}
            self._fill_class_table_from_dict({cid: current_names.get(cid, resolve_class_name(self._yaml_names, cid) or str(cid))
                                              for cid in sorted(ids)})
            self._update_pair_count_label(f"Проверено: {result['valid']}/{len(self.pairs)}")
            self._rebuild_file_list_display(finished=self._after_load_hook)
            self._after_loading_state_changed()
        tasks.start('inspect', PairInspectionJob, list(self.pairs), self.dataset_type, self.input_format,
                    self._is_include_empty(), label='Проверка пар, классы и свойства снимков…', finished=inspected)

    # ---------------- сканирование папок ----------------
    def scan_pairs(self):
        if self._yaml_mode:
            self._start_pair_inspection()
        elif self.images_folder:
            self._start_dataset_import('folder', [self.images_folder], options=dict(
                dataset_type=self.dataset_type,
                annotations=self.labels_folder if self.dataset_type == 0 else self.masks_folder))

    def _validate_all_pairs(self):
        result = getattr(self, '_inspection_result', None)
        if result is None:
            return 0, 0
        return result['valid'], len(result['rows']) - result['valid']

    def _recompute_color_stats(self):
        # Populated by PairInspectionJob, never decode a whole dataset in the GUI.
        return

    def _update_color_stats_label(self):
        if not hasattr(self, "color_stats_label"):
            return
        try:
            text, _ = ImageStats.format_stats_text(
                self.color_type_stats, self.bit_type_stats, bool(self.pairs)
            )
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
                QMessageBox.warning(
                    self, "Файл не найден", f"Изображение не существует:\n{img_path}"
                )
                return

            dlg = PairValidationDialog(
                self,
                (str(img_path), str(aux_path) if aux_path else None),
                self.dataset_type,
                self.validator,
            )
            dlg.exec_()
        except Exception as e:
            self._log_exception("Ошибка валидации пары", e)
            QMessageBox.critical(
                self, "Ошибка валидации", f"{type(e).__name__}: {e}\n\nСм. лог."
            )

    # ---------------- отрисовка списка пар ----------------
    def _make_list_item(self, text, tag=None, error_msg=""):
        item = QListWidgetItem(text)
        include_empty = self._is_include_empty()
        if tag == "invalid":
            item.setForeground(QColor(220, 0, 0))
            item.setToolTip(f"❌ Некорректная пара: {error_msg}")
        elif tag == "empty":
            if include_empty:
                item.setForeground(QColor(0, 140, 0))
                item.setToolTip("Пустая аннотация — будет включена как негатив.")
            else:
                item.setForeground(QColor(200, 120, 0))
                item.setToolTip("Пустая аннотация — будет пропущена.")
        elif tag == "error":
            item.setForeground(QColor(200, 0, 0))
            item.setToolTip("Ошибка чтения аннотации.")
        elif tag == "mixed":
            item.setForeground(QColor(160, 0, 200))
            item.setToolTip("Смешанные типы аннотаций.")
        elif tag == "unknown":
            item.setForeground(QColor(120, 120, 120))
            item.setToolTip("Неизвестный формат аннотации.")
        return item

    def _rebuild_file_list_display(self, pairs=None, finished=None):
        pairs = list(self.pairs if pairs is None else pairs)
        self._file_list_revision = getattr(self, '_file_list_revision', 0) + 1
        revision = self._file_list_revision
        self.file_list.clear()
        cache = getattr(self, '_pair_display_cache', {})
        position = 0
        invalid_names = []
        def batch():
            nonlocal position
            if revision != self._file_list_revision:
                return
            self.file_list.blockSignals(True)
            try:
                for pair in pairs[position:position + 300]:
                    info = cache.get(tuple(pair))
                    if info is None:
                        name = os.path.basename(pair[0])
                        info = dict(text=name + ' [ожидает проверки]', tag=None, err='', is_valid=None,
                                    name=name, color_type=None, bit_type=None)
                    elif info.get('tag') == 'empty':
                        info = dict(info, text=info['text'].replace('empty → skip', 'empty → include') if self._is_include_empty()
                                    else info['text'].replace('empty → include', 'empty → skip'))
                    item = self._make_list_item(info['text'], tag=info['tag'], error_msg=info['err'])
                    item.setData(Qt.UserRole, pair[0])
                    item.setData(Qt.UserRole + 1, tuple(pair))
                    item.setData(Qt.UserRole + 2, 'pending' if info['is_valid'] is None else 'valid' if info['is_valid'] else 'invalid')
                    item.setData(Qt.UserRole + 3, info['color_type'])
                    item.setData(Qt.UserRole + 4, info['bit_type'])
                    self.file_list.addItem(item)
                    if info['is_valid'] is False:
                        invalid_names.append(info['name'])
                position += 300
            finally:
                self.file_list.blockSignals(False)
            if position < len(pairs):
                self.validation_progress_label.setText(f'Построение списка: {position}/{len(pairs)}')
                self.validation_progress_label.setVisible(True)
                QTimer.singleShot(0, batch)
                return
            self.validation_progress_label.setVisible(False)
            self.validation_warning_label.setVisible(bool(invalid_names))
            if invalid_names:
                self.validation_warning_label.setText(f'Некорректных пар: {len(invalid_names)}. Они не будут экспортированы. ' + ', '.join(invalid_names[:5]))
            self._update_color_stats_label()
            if getattr(self, '_inspection_result', None) is not None:
                if not hasattr(self, '_overlap_review_panel'):
                    from ui.annotation_review import AnnotationReviewWidget
                    self._overlap_review_panel = AnnotationReviewWidget(external_images=self.file_list, parent=self)
                    layout = self.file_list.parentWidget().layout()
                    if layout is not None:
                        layout.insertWidget(0, self._overlap_review_panel)
                dataset_key = (getattr(self, "_inspection_revision", 0), tuple(map(tuple, self.pairs)), self.dataset_type, self.input_format)
                if getattr(self, '_review_dataset_key', None) != dataset_key:
                    self._review_dataset_key = dataset_key
                    self._overlap_review_panel.set_dataset(self.pairs, self.dataset_type, self.input_format)
            if finished:
                finished()
        batch()

    def closeEvent(self, event):
        self.shutdown_annotation_review()
        super().closeEvent(event)

    def shutdown_annotation_review(self):
        self._file_list_revision = getattr(self, "_file_list_revision", 0) + 1
        if hasattr(self, "_dataset_tasks"):
            self._dataset_tasks.close()
        if hasattr(self, "_overlap_review_panel"):
            self._overlap_review_panel.close_review()

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
            self._source_yaml_path = None
            self._update_color_stats_label()

        self._after_type_change_hook(new_type)

        if self.images_folder and not self._yaml_mode:
            self.scan_pairs()

    def _on_input_format_changed(self, idx):
        fmt = self.input_format_combo.currentData()
        if fmt is None:
            fmt = "auto"
        self.input_format = fmt
        self.validator.set_input_format(fmt)
        if self.pairs:
            self._start_pair_inspection()

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
        self._start_pair_inspection()

    def update_included_classes(self):
        if getattr(self, "_updating_class_table", False):
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
                self.class_remap = dense_class_remap(
                    orig for _row, orig in included_rows
                )
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
                        new_text = (
                            str(self.mask_class_remap[orig_id])
                            if orig_id in self.mask_class_remap
                            else "-"
                        )
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
        if getattr(self, "_updating_class_table", False):
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
