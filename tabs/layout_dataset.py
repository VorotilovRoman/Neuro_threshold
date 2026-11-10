
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# layout_dataset.py
import os
import sys
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSpinBox,
    QVBoxLayout,
)

from ui_tabs.layout_dataset_ui import setup_layout_dataset_ui
from utils.settings import settings
from utils_new.dataset_catalog_new import import_coco_catalog
from utils_new.dataset_coco_new import write_yolo_txt_for_coco_items

# V20: shared dataset services; legacy utility imports are not required.
# The UI import above is intentionally left unchanged.
from utils_new.dataset_files_new import (
    LazyImageCache,
    convert_to_grayscale,
    image_size_from_file,
    load_one_image_item,
    normalize_to_uint8,
    read_image_with_fallback,
    read_mask_with_fallback,
    resize_to_max_side,
)
from utils_new.dataset_unet_new import (
    load_unet_yaml_pairs,
    write_yolo_txt_for_unet_pairs,
)
from utils_new.dataset_yaml_new import find_mask_for_image, load_yolo_yaml
from utils_new.dataset_yolo_labels_new import (
    canonicalize_obb_corners,
    load_annotations,
    load_annotations_for_image,
    save_annotations,
    save_annotations_for_image,
)
from utils_new.smart_view_new import numpy_to_qpixmap, set_fill_color_button_preview
from utils_ops.segmentation_ops import update_annotation_list


# ============================================================
# Валидация одной унифицированной аннотации YOLO
# ============================================================
def _validate_unified_annotation(ann):
    """Возвращает (is_valid, err_msg)."""
    try:
        typ = ann[0]
        if typ == 'detect':
            _, cls, cx, cy, w, h = ann
            if not isinstance(cls, int) or cls < 0:
                return False, f"некорректный class id {cls!r}"
            if not (0.0 <= cx <= 1.0) or not (0.0 <= cy <= 1.0):
                return False, f"центр ({cx:.3f},{cy:.3f}) вне [0,1]"
            if w <= 0 or h <= 0:
                return False, f"неположительный размер ({w:.3f}x{h:.3f})"
            if w > 1 or h > 1:
                return False, f"размер ({w:.3f}x{h:.3f}) > 1"
        elif typ == 'obb':
            _, cls, points = ann
            if not isinstance(cls, int) or cls < 0:
                return False, f"некорректный class id {cls!r}"
            if len(points) != 8:
                return False, f"OBB: нужно 8 чисел, есть {len(points)}"
            for i in range(0, 8, 2):
                if not (0.0 <= points[i] <= 1.0) or not (0.0 <= points[i + 1] <= 1.0):
                    return False, f"OBB: точка {i // 2 + 1} вне [0,1]"
        elif typ == 'segment':
            _, cls, points = ann
            if not isinstance(cls, int) or cls < 0:
                return False, f"некорректный class id {cls!r}"
            if len(points) % 2 != 0:
                return False, "segment: нечётное число координат"
            n_pts = len(points) // 2
            if n_pts < 3:
                return False, f"segment: нужно ≥3 точек, есть {n_pts}"
            for i in range(0, len(points), 2):
                if not (0.0 <= points[i] <= 1.0) or not (0.0 <= points[i + 1] <= 1.0):
                    return False, f"segment: точка {i // 2 + 1} вне [0,1]"
        else:
            return False, f"неизвестный тип '{typ}'"
        return True, ""
    except Exception as e:
        return False, f"ошибка проверки: {e}"


# ============================================================
# Конвертация detect → OBB
# ============================================================
def _detect_annotation_to_obb(ann):
    """
    ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
    Углы нормализованы в [0,1]. Порядок углов соответствует тому,
    что возвращает SmartGraphicsView._obb_to_points при angle=0:
    BR → BL → TL → TR (по часовой от правого-нижнего).
    """
    if not ann or ann[0] != 'detect':
        return ann
    _, cls, cx, cy, w, h = ann
    x1 = float(cx) - float(w) / 2.0
    y1 = float(cy) - float(h) / 2.0
    x2 = float(cx) + float(w) / 2.0
    y2 = float(cy) + float(h) / 2.0
    # BR, BL, TL, TR
    corners = [x2, y2, x1, y2, x1, y1, x2, y1]
    corners = [max(0.0, min(1.0, float(p))) for p in corners]
    try:
        corners = canonicalize_obb_corners(corners)
    except Exception:
        pass
    return ('obb', int(cls), corners)


class LabelValidationThread(QThread):
    """Фоновая валидация всех аннотаций загруженного набора."""
    progress = pyqtSignal(int, int)
    finished_signal = pyqtSignal(dict)   # {idx: [(valid, err), ...]}

    def __init__(self, image_paths, yaml_label_map, label_folder):
        super().__init__()
        self.image_paths = list(image_paths)
        self.yaml_label_map = dict(yaml_label_map) if yaml_label_map else {}
        self.label_folder = label_folder
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _resolve_label_path(self, img_path):
        if self.yaml_label_map:
            lp = self.yaml_label_map.get(img_path)
            if lp and os.path.exists(lp):
                return lp
        if self.label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            cand = os.path.join(self.label_folder, base + '.txt')
            if os.path.exists(cand):
                return cand
        cand = os.path.splitext(img_path)[0] + '.txt'
        if os.path.exists(cand):
            return cand
        return None

    def run(self):
        results = {}
        total = len(self.image_paths)
        for idx, img_path in enumerate(self.image_paths):
            if self._cancel:
                break
            label_path = self._resolve_label_path(img_path)
            anns = []
            if label_path:
                try:
                    anns = load_annotations_for_image(label_path, img_path)
                except Exception:
                    anns = []
            results[idx] = [_validate_unified_annotation(a) for a in anns]
            if (idx + 1) % 25 == 0 or idx == total - 1:
                self.progress.emit(idx + 1, total)
        if not self._cancel:
            self.finished_signal.emit(results)


class Labeler(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Labeler (YOLO format)")

        # Данные (ленивая загрузка)
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Карта {путь_к_изображению: путь_к_файлу_меток}.
        self._yaml_label_map = {}
        self._label_folder = None

        self.img_w = 0
        self.img_h = 0
        self.all_annotations = []          # список аннотаций на каждое изображение (лениво)
        self.current_annotations = []      # аннотации текущего изображения

        # Для работы с YAML (маски не нужны)
        self.mask_folder = None
        self.loaded_masks = []
        self.class_colors = {}
        self.mask_opacity = 0.5

        # Валидация аннотаций
        self._validation_per_image = {}     # idx -> [(valid, err), ...]
        self._validation_thread = None

        # Модель авторазметки
        self.auto_model = None

        # UI
        setup_layout_dataset_ui(self)
        self._fill_color = QColor("#FFFFFF")
        set_fill_color_button_preview(self.btn_fill_color, self._fill_color)

        # Настройка SmartGraphicsView — используем унифицированные аннотации
        self.image_view.set_callbacks(
            on_rect_drawn=self.on_rect_drawn,
            on_obb_drawn=self.on_obb_drawn,
            on_segment_drawn=self.on_segment_drawn,
            on_display_update=self.update_image_display,
            on_reset_tool=self.reset_drawing_tool,
            on_annotation_modified=self.on_annotation_modified,
            on_selection_changed=self.on_selection_changed,
            on_annotation_right_clicked=self.on_annotation_right_clicked,  # NEW
            on_log=self.log,
        )

        # --- Сигналы навигации и загрузки ---
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.load_images.connect(self.load_images)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

        self.btn_load_labels.clicked.connect(self.load_labels_folder)
        self.btn_load_yaml.clicked.connect(self.load_yaml)
        self.btn_load_coco.clicked.connect(self.load_coco_json)
        self.btn_load_unet.clicked.connect(self.load_unet_yaml)
        self.save_button.clicked.connect(self.save_labels)
        self.btn_save_all.clicked.connect(self.save_all_annotations)
        self.btn_convert_all_to_obb.clicked.connect(self.convert_all_detect_to_obb)

        # --- Инструменты рисования ---
        self.tool_group.buttonClicked.connect(self.on_tool_selected)

        # --- Формат сохранения OBB ---
        self.chk_obb_xywhr.toggled.connect(self.on_obb_format_toggled)

        # --- Auto-label (YOLO) ---
        self.btn_load_auto_model.clicked.connect(self.load_auto_model)
        self.btn_auto_current.clicked.connect(
            lambda: self.run_auto_label(all_images=False))
        self.btn_auto_all.clicked.connect(
            lambda: self.run_auto_label(all_images=True))

        # --- Список объектов ---
        self.delete_button.clicked.connect(self.delete_selected_object)
        self.delete_all_button.clicked.connect(self.delete_all_annotations)
        self.object_list.currentItemChanged.connect(self.on_object_selection_changed)
        self.object_list.itemDoubleClicked.connect(self.on_object_double_clicked)
        self.object_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(self.show_object_context_menu)

        # --- Отображение аннотаций (только векторный слой Qt) ---
        self.btn_hide_frames.toggled.connect(self._update_annotation_display_options)
        self.btn_hide_labels.toggled.connect(self._update_annotation_display_options)
        self.btn_fill_objects.toggled.connect(self._update_annotation_display_options)
        self.btn_fill_color.clicked.connect(self.choose_fill_color)
        self._update_annotation_display_options()

        # --- Лог / гистограмма ---
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        # Начальное состояние
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
    #  Пересборка QListWidget: единая точка, защищена от реентерабельности
    # ----------------------------------------------------------------------
    def _rebuild_object_list(self, keep_row=None):
        """Пересобирает QListWidget с аннотациями текущего снимка.
        Сигналы блокируются, чтобы clear() не запускал каскад
        currentItemChanged → on_object_selection_changed → сброс
        выделения во view. Если keep_row задан — восстанавливает
        current row на этой строке.
        keep_row == -1 (или None) — current row снимается."""
        self.object_list.blockSignals(True)
        try:
            update_annotation_list(self.object_list, self.current_annotations,
                                   self.img_w, self.img_h)
            if keep_row is not None and 0 <= keep_row < self.object_list.count():
                self.object_list.setCurrentRow(keep_row)
            else:
                self.object_list.setCurrentRow(-1)
        finally:
            self.object_list.blockSignals(False)

    # ----------------------------------------------------------------------
    #  Инструменты рисования
    # ----------------------------------------------------------------------
    def _active_tool_name(self):
        if self.btn_tool_rect.isChecked():
            return "rect"
        if self.btn_tool_segment.isChecked():
            return "segment"
        if self.btn_tool_obb.isChecked():
            return "obb"
        if self.btn_tool_segment_line.isChecked():   # NEW
            return "segment_line"
        return None

    def _uncheck_all_tools(self):
        self.tool_group.setExclusive(False)
        for b in (self.btn_tool_rect, self.btn_tool_segment,
                  self.btn_tool_obb, self.btn_tool_segment_line):   # NEW
            b.setChecked(False)
        self.tool_group.setExclusive(True)

    def on_tool_selected(self, btn):
        if btn is self.btn_tool_rect:
            tool = "rect"
        elif btn is self.btn_tool_segment:
            tool = "segment"
        elif btn is self.btn_tool_obb:
            tool = "obb"
        elif btn is self.btn_tool_segment_line:      # NEW
            tool = "segment_line"
        else:
            return

        if not self.image_paths:
            self.log(f"Инструмент: {tool} (изображения не загружены).")
            return

        if self.image_view.edit_mode:
            self.image_view.set_edit_mode(False)
        self.image_view.set_selected_index(-1)
        self.image_view.set_drawing_tool(tool)
        self.log(f"Инструмент: {tool}")

    def toggle_drawing_mode(self, checked):
        if not self.image_paths:
            self._uncheck_all_tools()
            self.log("No image loaded. Cannot enable drawing mode.")
            return
        if checked:
            tool = self._active_tool_name()
            if tool is None:
                self.btn_tool_rect.setChecked(True)
                tool = "rect"
            if self.image_view.edit_mode:
                self.image_view.set_edit_mode(False)
            self.image_view.set_selected_index(-1)
            self.image_view.set_drawing_tool(tool)
            self.log(f"Drawing mode ON ({tool}).")
        else:
            self._uncheck_all_tools()
            self.image_view.set_drawing_tool(None)
            self.log("Drawing mode OFF (panning mode).")

    def reset_drawing_tool(self):
        self._uncheck_all_tools()
        self.image_view.set_drawing_tool(None)
        self.image_view.set_selected_index(-1)
        self.log("Drawing mode OFF.")

    # ----------------------------------------------------------------------
    #  Формат сохранения OBB
    # ----------------------------------------------------------------------
    def on_obb_format_toggled(self, checked):
        fmt = "xywhr (cx cy w h angle_rad)" if checked else \
              "corners (x1 y1 x2 y2 x3 y3 x4 y4)"
        self.log(f"Формат сохранения OBB: {fmt}")

    # ----------------------------------------------------------------------
    #  Ленивая загрузка изображений
    # ----------------------------------------------------------------------
    def _load_images(self, source, label_map=None):
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

        if self._lazy_cache is not None:
            self._lazy_cache.shutdown()
            self._lazy_cache = None

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

        self.image_paths = all_files
        self.current_index = 0
        self.current_annotations = []
        self.all_annotations = [None] * len(all_files)
        self._yaml_label_map = dict(label_map) if label_map else {}
        self._label_folder = None

        self.image_view.set_annotations([], 0, 0)
        self.image_view.set_selected_index(-1)
        self.object_list.clear()

        first = self._lazy_cache.get(0)
        if first is None:
            self.log("Не удалось загрузить первое изображение.")
            return

        total = len(self._lazy_cache)
        self._lazy_cache.prefetch(
            range(1, min(1 + self._lazy_cache.prefetch_ahead + 1, total))
        )

        self.log(f"Загружено {total} изображений (первое — сразу, остальное в фоне).")
        self.display_current_image()
        self.nav_widget.set_current_index(self.current_index, total)
        self.update_navigation_state()

        if self._active_tool_name() is None:
            self.btn_tool_rect.setChecked(True)
        self.image_view.set_drawing_tool(self._active_tool_name() or "rect")

        self._start_annotation_validation()

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
        self._label_folder = None
        self._validation_per_image = {}
        self.validation_status_label.setVisible(False)
        self.validation_status_label.setText("")
        self.image_view.set_annotations([], 0, 0)
        self.image_view.set_selected_index(-1)
        self.update_image_display()
        self.info_label.setText("No image")
        self.object_list.clear()
        self.nav_widget.set_current_index(0, 0)
        self.update_navigation_state()

    def closeEvent(self, event):
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        if self._validation_thread is not None:
            try:
                self._validation_thread.cancel()
                self._validation_thread.wait(2000)
            except Exception:
                pass
            self._validation_thread = None
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

        self._yaml_label_map = {}
        self._label_folder = folder

        self.all_annotations = [[] for _ in self.image_paths]
        count = 0
        for i, img_path in enumerate(self.image_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(folder, base + '.txt')
            if os.path.exists(label_path):
                ann = load_annotations_for_image(label_path, img_path)
                self.all_annotations[i] = ann
                if ann:
                    count += 1
        self.display_current_image()
        self.log(f"Загружены метки для {count} изображений.")
        self._start_annotation_validation()

    def load_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YAML file", "", "YAML files (*.yaml *.yml)"
        )
        if not file_path:
            return
        try:
            pairs, names, root, splits_info = load_yolo_yaml(
                file_path, log_cb=self.log)
            if not pairs:
                QMessageBox.warning(self, "Ошибка", "Не найдено изображений в YAML.")
                return

            image_paths = [p for p, _ in pairs]
            label_map = {p: lp for p, lp in pairs if lp is not None}

            self._load_images(image_paths, label_map=label_map)
            self.log(f"=== Загрузка YAML: {file_path} ===")
            self.log(f"Root: {root}")
            for split, n in splits_info.items():
                if n:
                    self.log(f"  • {split}: {n} изображений")
            self.log(f"Загружено {len(image_paths)} изображений.")
            QMessageBox.information(self, "Успех",
                                    f"Загружено {len(image_paths)} изображений.")
        except Exception as e:
            self.log(f"Ошибка загрузки YAML: {e}")
            self._log_exception("Ошибка загрузки YAML", e)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить YAML:\n{e}")

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


    def load_coco_json(self):
        json_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select COCO JSON(s) — можно несколько",
            "", "JSON files (*.json);;All files (*)")
        if not json_paths:
            return
        try:
            self._load_coco_into_labeler(json_paths)
        except Exception as e:
            self._log_exception("Ошибка загрузки COCO", e)
            QMessageBox.critical(self, "Ошибка COCO", str(e))

    def _load_coco_into_labeler(self, json_paths):
        """Import COCO via the shared catalog and shared YOLO writer."""
        if not json_paths:
            return
        catalog = import_coco_catalog(json_paths, log_cb=self.log)
        if not catalog.sources:
            QMessageBox.warning(self, "COCO", "Не найдено ни одного изображения.")
            return
        tmp_dir = os.path.join(os.path.dirname(str(json_paths[0])), '_coco_labels_tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        pairs = []
        for idx, source in enumerate(catalog.sources):
            pairs.extend(write_yolo_txt_for_coco_items(
                source.items, tmp_dir, class_remap=None,
                log_cb=self.log, name_prefix=f'j{idx:03d}_'))
        image_paths = [img for img, _ in pairs]
        label_map = {img: label for img, label in pairs}
        self._load_images(image_paths, label_map=label_map)
        self.log(f"=== Загрузка COCO ({len(catalog.sources)} JSON) ===")
        self.log(f"Изображений: {len(image_paths)}, классов: {len(catalog.class_names)}")
        self.log(f"Временные labels: {tmp_dir}")
        QMessageBox.information(
            self, "COCO", f"Загружено изображений: {len(image_paths)}\n"
            f"Классов: {len(catalog.class_names)}\nВременные labels: {tmp_dir}")


    def load_unet_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net YAML", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not yaml_path:
            return
        try:
            self._load_unet_into_labeler(yaml_path)
        except Exception as e:
            self._log_exception("Ошибка загрузки U-Net YAML", e)
            QMessageBox.critical(self, "Ошибка U-Net YAML", str(e))

    def _load_unet_into_labeler(self, yaml_path):
        """Import U-Net via the shared mask-to-annotation service."""
        pairs_raw, class_names, mask_offset, root, _ = load_unet_yaml_pairs(
            yaml_path, log_cb=self.log)
        if not pairs_raw:
            QMessageBox.warning(self, "U-Net", "Не найдено ни одной пары.")
            return
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(yaml_path)),
                               '_unet_labels_tmp')
        pairs = write_yolo_txt_for_unet_pairs(
            pairs_raw, tmp_dir, class_names=class_names,
            mask_offset=mask_offset, include_empty=True, log_cb=self.log)
        if not pairs:
            QMessageBox.warning(self, "U-Net", "Не удалось сконвертировать маски.")
            return
        image_paths = [img for img, _ in pairs]
        label_map = {img: label for img, label in pairs}
        self._load_images(image_paths, label_map=label_map)
        self.log(f"=== Загрузка U-Net YAML: {yaml_path} ===")
        self.log(f"Root: {root}, пар: {len(image_paths)}, "
                 f"classes: {len(class_names)}, offset={mask_offset}")
        self.log(f"Временные labels (из масок): {tmp_dir}")
        QMessageBox.information(
            self, "U-Net YAML", f"Загружено изображений: {len(image_paths)}\n"
            f"Классов: {len(class_names)}\n"
            "Маски сконвертированы в polygon-аннотации.\n"
            f"Временные labels: {tmp_dir}")

    # ----------------------------------------------------------------------
    #  Отображение текущего изображения
    # ----------------------------------------------------------------------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Не удалось загрузить снимок: {self.image_paths[idx]}")
            return

        img = item["image"]
        gray = item["gray"]
        self.img_h, self.img_w = img.shape[:2]

        self.update_histogram(gray)

        if (idx >= len(self.all_annotations)) or (self.all_annotations[idx] is None):
            label_path = self._yaml_label_map.get(self.image_paths[idx]) \
                if self._yaml_label_map else None
            if label_path and os.path.exists(label_path):
                anns = load_annotations_for_image(label_path, self.image_paths[idx])
            else:
                anns = list(item.get("annotations", []) or [])
            while len(self.all_annotations) <= idx:
                self.all_annotations.append(None)
            self.all_annotations[idx] = anns

        self.current_annotations = list(self.all_annotations[idx])

        self.image_view.set_annotations(self.current_annotations, self.img_w, self.img_h)
        self.image_view.set_selected_index(-1)

        # Новый снимок — выделения в списке быть не должно.
        self._rebuild_object_list(keep_row=None)
        self._apply_validation_colors_to_list()

        self.update_image_display()
        self.info_label.setText(f"Image {idx + 1} of {total}")
        self.nav_widget.set_current_index(idx, total)

        tool = self._active_tool_name()
        self.image_view.set_drawing_tool(tool)

        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    def update_image_display(self):
        """Отображает только исходное изображение и отдельно Qt-векторные метки.

        Ключ изображения предотвращает повторное формирование QPixmap при выборе
        или редактировании рамок, а слой Qt перерисовывается самостоятельно.
        """
        if self._lazy_cache is None or not self.image_paths:
            return
        item = self._lazy_cache.get(self.current_index)
        if item is None:
            return
        img = item['image']
        if img is None:
            return

        # Геометрия рисуется в координатах рабочей матрицы, даже когда
        # numpy_to_qpixmap уменьшит свой pixmap до безопасного размера.
        img_h, img_w = img.shape[:2]
        self.image_view.set_annotations(self.current_annotations, img_w, img_h)

        # Новую QPixmap создаём только при смене изображения или его буфера.
        image_key = (id(self._lazy_cache), self.current_index,
                     self.image_paths[self.current_index], id(img), img_w, img_h)
        if image_key != getattr(self, '_last_display_image_key', None) \
                or self.image_view._current_pixmap is None:
            self.image_view.set_pixmap(numpy_to_qpixmap(img))
            self._last_display_image_key = image_key
        else:
            self.image_view.refresh_annotations()

    def choose_fill_color(self):
        """Цвет применится к Qt-слою, а не к исходному изображению."""
        color = QColorDialog.getColor(self._fill_color, self, 'Цвет заливки объектов')
        if not color.isValid():
            return
        self._fill_color = QColor(color)
        set_fill_color_button_preview(self.btn_fill_color, self._fill_color)
        self._update_annotation_display_options()

    def _update_annotation_display_options(self, _checked=None):
        """Применить UI-переключатели, не изменяя данные или QPixmap.

        Скрытие рамок и меток независимо; цветная заливка может быть включена
        одновременно с любым из переключателей. Исходное изображение и
        файлы разметки не изменяются.
        """
        self.image_view.set_annotation_visibility(
            show_frames=not self.btn_hide_frames.isChecked(),
            show_labels=not self.btn_hide_labels.isChecked(),
            fill_white=self.btn_fill_objects.isChecked(),
            fill_color=self._fill_color.name(),
        )

    def update_histogram(self, gray_img):
        self.hist_ax.clear()
        self.hist_ax.hist(gray_img.ravel(), bins=256, range=(0, 256),
                          color='black', alpha=0.7)
        self.hist_ax.set_title("Grayscale Histogram")
        self.hist_ax.set_xlabel("Pixel intensity")
        self.hist_ax.set_ylabel("Frequency")
        self.hist_canvas.draw()

    def _sync_current_annotations_to_storage(self):
        idx = self.current_index
        while len(self.all_annotations) <= idx:
            self.all_annotations.append(None)
        self.all_annotations[idx] = list(self.current_annotations)

    # ----------------------------------------------------------------------
    #  Валидация
    # ----------------------------------------------------------------------
    def _start_annotation_validation(self):
        if self._validation_thread is not None and self._validation_thread.isRunning():
            try:
                self._validation_thread.cancel()
                self._validation_thread.wait(2000)
            except Exception:
                pass

        self._validation_per_image = {}
        self.validation_status_label.setVisible(False)
        self.validation_status_label.setText("")

        self._validation_thread = LabelValidationThread(
            image_paths=self.image_paths,
            yaml_label_map=self._yaml_label_map,
            label_folder=self._label_folder,
        )
        self._validation_thread.progress.connect(self._on_validation_progress)
        self._validation_thread.finished_signal.connect(self._on_validation_finished)
        self._validation_thread.start()

    def _on_validation_progress(self, done, total):
        if total > 0:
            self.validation_status_label.setText(f"⏳ Проверка разметки: {done}/{total}")
            self.validation_status_label.setStyleSheet(
                "color: #0055aa; font-weight: bold; padding: 4px;"
            )
            self.validation_status_label.setVisible(True)

    def _on_validation_finished(self, results):
        self._validation_per_image = results
        self._update_validation_status()
        self._apply_validation_colors_to_list()

    def _update_validation_status(self):
        bad_idx = []
        for idx, per_file in self._validation_per_image.items():
            if any(not ok for ok, _ in per_file):
                bad_idx.append(idx + 1)

        if not bad_idx:
            self.validation_status_label.setVisible(False)
            self.validation_status_label.setText("")
            return

        bad_idx.sort()
        preview = ", ".join(f"#{i}" for i in bad_idx[:10])
        more = "" if len(bad_idx) <= 10 else f" и ещё {len(bad_idx) - 10}"
        self.validation_status_label.setText(
            f"⚠️ Некорректная разметка у {len(bad_idx)} снимков: {preview}{more}. "
            f"Некорректные объекты выделены красным в списке."
        )
        self.validation_status_label.setStyleSheet(
            "color: #b00000; font-weight: bold; padding: 4px;"
        )
        self.validation_status_label.setVisible(True)

    def _apply_validation_colors_to_list(self):
        if not hasattr(self, 'object_list'):
            return
        per_file = self._validation_per_image.get(self.current_index, [])
        for row in range(self.object_list.count()):
            item = self.object_list.item(row)
            if item is None:
                continue
            item.setForeground(QColor(0, 0, 0))
            item.setToolTip("")
            base_text = item.text()
            if "[INVALID:" in base_text:
                base_text = base_text.split("  [INVALID:")[0]
                item.setText(base_text)
            if row < len(per_file):
                ok, err = per_file[row]
                if not ok:
                    item.setForeground(QColor(220, 0, 0))
                    item.setText(base_text + f"  [INVALID: {err}]")
                    item.setToolTip(f"❌ {err}")

    def _revalidate_current(self):
        self._validation_per_image[self.current_index] = [
            _validate_unified_annotation(a) for a in self.current_annotations
        ]
        self._update_validation_status()
        self._apply_validation_colors_to_list()

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
        self.btn_save_all.setEnabled(has_images)
        self.btn_convert_all_to_obb.setEnabled(has_images)
        self.btn_tool_rect.setEnabled(has_images)
        self.btn_tool_segment.setEnabled(has_images)
        self.btn_tool_obb.setEnabled(has_images)
        self.btn_tool_segment_line.setEnabled(has_images)
        self.delete_button.setEnabled(has_images and len(self.current_annotations) > 0)
        self.delete_all_button.setEnabled(has_images and len(self.current_annotations) > 0)
        self.nav_widget.set_navigation_enabled(has_images)

    # ----------------------------------------------------------------------
    #  Рисование новых аннотаций
    # ----------------------------------------------------------------------
    def _append_annotation(self, new_ann, log_text):
        self.current_annotations.append(new_ann)
        self._sync_current_annotations_to_storage()
        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        new_row = len(self.current_annotations) - 1
        self._rebuild_object_list(keep_row=new_row)
        # NEW: синхронизируем выделение во view, иначе объект окажется
        # выделенным в списке, но не в представлении — и сразу после
        # рисования его нельзя ни выделить кликом, ни редактировать.
        self.image_view.set_selected_index(new_row)
        self.update_image_display()
        self.log(log_text)
        self._revalidate_current()
        self.update_navigation_state()

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
        self._append_annotation(
            new_ann,
            f"Added rectangle: class=0, center=({cx:.3f},{cy:.3f}), "
            f"size=({bw:.3f},{bh:.3f})",
        )

    def on_obb_drawn(self, norm_pts):
        if not norm_pts or len(norm_pts) != 8:
            return
        norm_pts = [max(0.0, min(1.0, float(p))) for p in norm_pts]
        norm_pts = canonicalize_obb_corners(norm_pts)
        new_ann = ('obb', 0, norm_pts)
        self._append_annotation(
            new_ann,
            f"Added OBB: class=0, "
            f"corners=[{', '.join(f'{p:.3f}' for p in norm_pts)}]",
        )

    def on_segment_drawn(self, norm_pts):
        if not norm_pts or len(norm_pts) < 6 or len(norm_pts) % 2 != 0:
            return
        norm_pts = [max(0.0, min(1.0, float(p))) for p in norm_pts]
        new_ann = ('segment', 0, norm_pts)
        n_pts = len(norm_pts) // 2
        self._append_annotation(
            new_ann,
            f"Added segment: class=0, points={n_pts}",
        )

    # ======================================================================
    #  AUTO-LABEL (YOLO)
    # ======================================================================
    def _resolve_auto_device(self):
        import torch
        d = self.auto_device_combo.currentText()
        if d == "auto":
            try:
                return "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if d == "mps":
            try:
                mps = getattr(torch.backends, "mps", None)
                if mps is not None and mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        return d

    def load_auto_model(self):
        from ultralytics import YOLO
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель YOLO", "",
            "YOLO weights (*.pt *.onnx *.engine);;Все файлы (*)"
        )
        if not path:
            return
        try:
            self.auto_model = YOLO(path)
            self.auto_model_edit.setText(path)

            task = getattr(self.auto_model, "task", "unknown")
            self.auto_kind_label.setText(f"Тип вывода: {task}")

            names = getattr(self.auto_model, "names", {})
            self.log(f"[Auto-label] Модель загружена: {path}")
            self.log(f"[Auto-label] Задача: {task}")
            if isinstance(names, dict) and names:
                self.log(f"[Auto-label] Классы: {dict(names)}")

            if not self.image_paths:
                self.log("[Auto-label] Изображения ещё не загружены.")
        except Exception as e:
            self.auto_model = None
            self.auto_model_edit.clear()
            self.auto_kind_label.setText("Тип вывода: —")
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось загрузить модель:\n{e}")
            self.log(f"[Auto-label] Ошибка загрузки модели: {e}")

    def _auto_infer_one(self, img_bgr):
        """Прогон YOLO на одном изображении.
        Возвращает (unified_annotations, kind)."""
        if self.auto_model is None:
            return [], "unknown"

        conf = float(self.auto_conf_spin.value())
        iou = float(self.auto_iou_spin.value())
        imgsz = int(self.auto_imgsz_spin.value())
        device = self._resolve_auto_device()

        results = self.auto_model(
            img_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if not results:
            return [], "unknown"

        r = results[0]
        H, W = img_bgr.shape[:2]
        if W <= 0 or H <= 0:
            return [], "unknown"

        # --- Опции постобработки ---
        try:
            to_bbox = bool(self.chk_auto_to_bbox.isChecked())
            convex_hull = bool(self.chk_auto_convex_hull.isChecked())
        except AttributeError:
            # На случай, если UI-файл почему-то не обновили.
            to_bbox = False
            convex_hull = False

        # --- 0. Принудительное преобразование в обычный bbox ---
        if to_bbox:
            unified = self._auto_extract_boxes_as_detect(r, W, H)
            return unified, "detect"

        # --- 1. OBB ---
        obb = getattr(r, "obb", None)
        if (obb is not None
                and getattr(obb, "xyxyxyxy", None) is not None
                and len(obb) > 0):
            corners_all = obb.xyxyxyxy.cpu().numpy()
            cls_all = obb.cls.cpu().numpy().astype(int)
            unified = []
            for i in range(len(cls_all)):
                corners = corners_all[i].tolist()
                norm = []
                for (x, y) in corners:
                    nx = max(0.0, min(1.0, float(x) / W))
                    ny = max(0.0, min(1.0, float(y) / H))
                    norm.append(nx)
                    norm.append(ny)
                norm = canonicalize_obb_corners(norm)
                unified.append(("obb", int(cls_all[i]), norm))
            return unified, "obb"

        # --- 2. Segmentation ---
        masks = getattr(r, "masks", None)
        if (masks is not None
                and getattr(masks, "xy", None) is not None
                and len(masks.xy) > 0):
            boxes = getattr(r, "boxes", None)
            cls_all = []
            if boxes is not None and len(boxes) > 0:
                cls_all = boxes.cls.cpu().numpy().astype(int)
            unified = []
            for i, poly in enumerate(masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                cls_id = int(cls_all[i]) if i < len(cls_all) else 0

                # NEW: сглаживание контура выпуклой оболочкой
                if convex_hull:
                    poly = self._segment_convex_hull(poly)
                    if poly is None or len(poly) < 3:
                        continue

                flat = []
                for p in poly:
                    nx = max(0.0, min(1.0, float(p[0]) / W))
                    ny = max(0.0, min(1.0, float(p[1]) / H))
                    flat.append(nx)
                    flat.append(ny)
                if len(flat) >= 6 and len(flat) % 2 == 0:
                    unified.append(("segment", cls_id, flat))
            return unified, "segment"

        # --- 3. Detection ---
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            unified = self._auto_extract_boxes_as_detect(r, W, H)
            return unified, "detect"

        return [], "unknown"

    # ------------------------------------------------------------------
    #  Извлечение обычных bounding box из результатов YOLO
    # ------------------------------------------------------------------
    def _auto_extract_boxes_as_detect(self, r, W, H):
        """Возвращает список ('detect', cls, cx, cy, bw, bh) из любой
        задачи YOLO (detect / segment / OBB).

        Порядок источников:
          1) r.boxes.xyxy — «родные» axis-aligned bbox модели;
          2) r.obb.xyxy — axis-aligned bbox из OBB (ultralytics свойство);
          3) r.obb.xyxyxyxy — вычисляем габарит по 4 углам;
          4) r.masks.xy — вычисляем габарит по контуру маски."""
        xyxy_all = None
        cls_all = None

        # 1) boxes.xyxy
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            try:
                xyxy_all = boxes.xyxy.cpu().numpy()
                cls_all = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy_all = None
                cls_all = None

        # 2) obb.xyxy / obb.xyxyxyxy
        if xyxy_all is None:
            obb = getattr(r, "obb", None)
            if obb is not None and len(obb) > 0:
                try:
                    obb_xyxy = getattr(obb, "xyxy", None)
                    if obb_xyxy is not None:
                        xyxy_all = obb_xyxy.cpu().numpy()
                    else:
                        corners_all = obb.xyxyxyxy.cpu().numpy()
                        xyxy_all = []
                        for corners in corners_all:
                            xs = corners[:, 0]
                            ys = corners[:, 1]
                            xyxy_all.append(
                                [xs.min(), ys.min(), xs.max(), ys.max()]
                            )
                        xyxy_all = np.array(xyxy_all, dtype=np.float32)
                    cls_all = obb.cls.cpu().numpy().astype(int)
                except Exception:
                    xyxy_all = None
                    cls_all = None

        # 3) masks.xy
        if xyxy_all is None:
            masks = getattr(r, "masks", None)
            if (masks is not None
                    and getattr(masks, "xy", None) is not None
                    and len(masks.xy) > 0):
                xyxy_all = []
                for poly in masks.xy:
                    if poly is None or len(poly) < 3:
                        xyxy_all.append([0, 0, 0, 0])
                        continue
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    xyxy_all.append(
                        [xs.min(), ys.min(), xs.max(), ys.max()]
                    )
                xyxy_all = np.array(xyxy_all, dtype=np.float32)
                cls_all = np.zeros(len(xyxy_all), dtype=int)
                boxes = getattr(r, "boxes", None)
                if (boxes is not None
                        and getattr(boxes, "cls", None) is not None
                        and len(boxes) == len(xyxy_all)):
                    try:
                        cls_all = boxes.cls.cpu().numpy().astype(int)
                    except Exception:
                        pass

        if xyxy_all is None or cls_all is None:
            return []

        unified = []
        n = min(len(xyxy_all), len(cls_all))
        for i in range(n):
            try:
                x1, y1, x2, y2 = [float(v) for v in xyxy_all[i]]
            except Exception:
                continue
            x1 = max(0.0, min(x1, W))
            y1 = max(0.0, min(y1, H))
            x2 = max(0.0, min(x2, W))
            y2 = max(0.0, min(y2, H))
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            unified.append(("detect", int(cls_all[i]), cx, cy, bw, bh))
        return unified

    # ------------------------------------------------------------------
    #  Сглаживание контура сегментации
    # ------------------------------------------------------------------
    @staticmethod
    def _segment_convex_hull(poly):
        """Возвращает выпуклую оболочку контура poly.
        poly — numpy (N, 2) или список [x, y]. При неудаче возвращает
        исходный poly, чтобы вызов не «съел» объект."""
        try:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
            if len(pts) < 3:
                return poly
            hull = cv2.convexHull(pts)
            if hull is None or len(hull) < 3:
                return poly
            return hull.reshape(-1, 2)
        except Exception:
            return poly


    def _apply_auto_annotations(self, unified, replace):
        if replace:
            self.current_annotations = list(unified)
        else:
            self.current_annotations = list(self.current_annotations) + list(unified)
        self._sync_current_annotations_to_storage()
        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        self._rebuild_object_list(keep_row=None)
        self.update_image_display()
        self._revalidate_current()
        self.update_navigation_state()

    def _resolve_label_path_for_img(self, img_path):
        if self._yaml_label_map:
            lp = self._yaml_label_map.get(img_path)
            if lp and os.path.exists(lp):
                return lp
        if self._label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            cand = os.path.join(self._label_folder, base + '.txt')
            if os.path.exists(cand):
                return cand
        cand = os.path.splitext(img_path)[0] + '.txt'
        if os.path.exists(cand):
            return cand
        return None

    def run_auto_label(self, all_images=False):
        if self.auto_model is None:
            QMessageBox.warning(
                self, "Auto-label",
                "Сначала загрузите модель YOLO (кнопка «Загрузить»)."
            )
            return
        if not self.image_paths:
            QMessageBox.warning(
                self, "Auto-label",
                "Сначала загрузите изображения."
            )
            return

        if not all_images:
            self._run_auto_label_current()
        else:
            self._run_auto_label_all()

    def _run_auto_label_current(self):
        if self._lazy_cache is None:
            return
        item = self._lazy_cache.get(self.current_index)
        if item is None:
            self.log("[Auto-label] Не удалось загрузить текущий снимок.")
            return

        img = item["image"]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            unified, kind = self._auto_infer_one(img)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Ошибка",
                f"Не удалось выполнить инференс:\n{e}"
            )
            self.log(f"[Auto-label] Ошибка инференса: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.auto_kind_label.setText(f"Тип вывода: {kind}")
        replace = self.chk_auto_replace.isChecked()
        opts = []
        if self.chk_auto_to_bbox.isChecked():
            opts.append("→ bbox")
        if self.chk_auto_convex_hull.isChecked():
            opts.append("convex hull")
        if self.chk_auto_replace.isChecked():
            opts.append("replace")
        opt_str = ", ".join(opts) if opts else "—"
        self.log(f"[Auto-label] Опции: {opt_str}")

        self._apply_auto_annotations(unified, replace)

        action = "заменено" if replace else "добавлено"
        self.log(f"[Auto-label] Текущий снимок: {action} {len(unified)} "
                 f"объектов (тип={kind})")

    def _run_auto_label_all(self):
        total = len(self.image_paths)
        if total == 0:
            return

        reply = QMessageBox.question(
            self, "Auto-label",
            f"Разметить все изображения ({total})?\n\n"
            f"Текущее поведение: "
            f"{'заменить существующие аннотации' if self.chk_auto_replace.isChecked() else 'дополнить существующие'}.\n"
            f"Операция может занять продолжительное время.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        progress = QProgressDialog(
            "Авторазметка...", "Отмена", 0, total, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        replace = self.chk_auto_replace.isChecked()

        # NEW: единый набор опций для логов и итогового сообщения
        opts = []
        if self.chk_auto_to_bbox.isChecked():
            opts.append("→ bbox")
        if self.chk_auto_convex_hull.isChecked():
            opts.append("convex hull")
        if self.chk_auto_replace.isChecked():
            opts.append("replace")
        opt_str = ", ".join(opts) if opts else "—"
        kind_seen = None
        processed = 0
        errors = 0
        added_objects = 0

        for idx in range(total):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            progress.setLabelText(
                f"Обработка {idx + 1}/{total}: "
                f"{os.path.basename(self.image_paths[idx])}"
            )
            QApplication.processEvents()

            # --- Загрузка изображения ---
            try:
                img = read_image_with_fallback(self.image_paths[idx])
                if img is None:
                    errors += 1
                    self.log(f"[Auto-label] Не удалось прочитать "
                             f"{os.path.basename(self.image_paths[idx])}")
                    continue
                img = normalize_to_uint8(img)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.shape[2] != 3:
                    img = img[:, :, :3]
            except Exception as e:
                errors += 1
                self.log(f"[Auto-label] Ошибка чтения "
                         f"{os.path.basename(self.image_paths[idx])}: {e}")
                continue

            # --- Инференс ---
            try:
                unified, kind = self._auto_infer_one(img)
                kind_seen = kind
            except Exception as e:
                errors += 1
                self.log(f"[Auto-label] Ошибка инференса "
                         f"{os.path.basename(self.image_paths[idx])}: {e}")
                continue

            # --- Применение ---
            try:
                while len(self.all_annotations) <= idx:
                    self.all_annotations.append(None)

                if replace:
                    self.all_annotations[idx] = list(unified)
                else:
                    existing = self.all_annotations[idx]
                    if existing is None:
                        label_path = self._resolve_label_path_for_img(
                            self.image_paths[idx])
                        if label_path:
                            try:
                                existing = load_annotations_for_image(label_path, self.image_paths[idx])
                            except Exception:
                                existing = []
                        else:
                            existing = []
                    self.all_annotations[idx] = list(existing) + list(unified)

                self._validation_per_image[idx] = [
                    _validate_unified_annotation(a)
                    for a in self.all_annotations[idx]
                ]
                processed += 1
                added_objects += len(unified)
            except Exception as e:
                errors += 1
                self.log(f"[Auto-label] Ошибка применения "
                         f"{os.path.basename(self.image_paths[idx])}: {e}")
                continue

        progress.setValue(total)
        progress.close()

        # --- Обновляем текущий снимок, если его разметка изменилась ---
        if (self.current_index < total
                and self.all_annotations[self.current_index] is not None):
            self.current_annotations = list(
                self.all_annotations[self.current_index])
            self.image_view.set_annotations(self.current_annotations,
                                            self.img_w, self.img_h)
            self._rebuild_object_list(keep_row=None)
            self.update_image_display()
            self._revalidate_current()
            self.update_navigation_state()

        self._update_validation_status()
        self._apply_validation_colors_to_list()

        if kind_seen:
            self.auto_kind_label.setText(f"Тип вывода: {kind_seen}")

        self.log(f"[Auto-label] Обработано {processed}/{total} "
                 f"(объектов: {added_objects}, ошибок: {errors}); "
                 f"тип={kind_seen or '—'}")

        QMessageBox.information(
            self, "Auto-label",
            f"Обработано изображений: {processed}/{total}\n"
            f"Опции: {opt_str}\n"
            f"Добавлено/заменено объектов: {added_objects}\n"
            f"Ошибок чтения/инференса: {errors}\n"
            f"Тип вывода модели: {kind_seen or '—'}\n\n"
            f"⚠️ Изменения пока только в памяти.\n"
            f"Не забудьте сохранить разметку (Save Labels / Save All)."
        )

    # ----------------------------------------------------------------------
    #  Редактирование аннотаций
    # ----------------------------------------------------------------------
    def on_annotation_modified(self, idx, new_ann):
        if idx < 0 or idx >= len(self.current_annotations):
            return
        self.current_annotations[idx] = new_ann
        self._sync_current_annotations_to_storage()
        self._rebuild_object_list(keep_row=idx)
        # NEW: если после пересборки списка выделение во view «отвалилось»
        # (или в него прилетел другой индекс) — восстанавливаем.
        if self.image_view.selected_index != idx:
            self.image_view.set_selected_index(idx)
        if not self.image_view.edit_mode:
            self.image_view.set_edit_mode(True)
        self.update_image_display()
        self.log(f"Modified object {idx + 1}")
        self._revalidate_current()

    # ----------------------------------------------------------------------
    #  Единая точка синхронизации выделения: список ↔ image_view
    # ----------------------------------------------------------------------
    def on_object_selection_changed(self, current, previous):
        """Срабатывает при смене текущего элемента списка: клик пользователя,
        стрелки клавиатуры, программный setCurrentRow (если сигналы не
        заблокированы). previous игнорируется — мы всегда смотрим на current."""
        idx = current.data(Qt.UserRole) if current is not None else -1

        # Если выделение уже совпадает с состоянием view — просто обновим
        # картинку (мог поменяться, например, цвет выделения).
        if idx == self.image_view.selected_index:
            self.update_image_display()
            return

        # Перешли из режима рисования к выделению объекта —
        # гасим активный инструмент.
        if idx != -1 and self._active_tool_name() is not None:
            self._uncheck_all_tools()
            self.image_view.set_drawing_tool(None)

        # Обновляем view. set_selected_index сам дёрнет on_selection_changed,
        # но тот сразу выйдет (idx уже == currentRow в списке).
        self.image_view.set_selected_index(idx)

        if idx != -1 and not self.image_view.edit_mode:
            self.image_view.set_edit_mode(True)

        self.update_image_display()
        self.update_navigation_state()

    def on_selection_changed(self, idx):
        """Вызывается из SmartGraphicsView при выделении/снятии выделения
        объекта мышью. Синхронизирует список, не создавая рекурсии."""
        if idx == self.object_list.currentRow():
            # Уже синхронно — просто перерисуем.
            self.update_image_display()
            return

        # Синхронизируем список. Блокируем сигналы только здесь и на одну
        # операцию — это безопаснее, чем глобальный флаг-«заглушка».
        self.object_list.blockSignals(True)
        try:
            self.object_list.setCurrentRow(idx)
        finally:
            self.object_list.blockSignals(False)

        # Раз выделили объект — выключаем инструмент рисования.
        if idx != -1 and self._active_tool_name() is not None:
            self._uncheck_all_tools()
            self.image_view.set_drawing_tool(None)

        self.update_image_display()
        self.update_navigation_state()

    # ----------------------------------------------------------------------
    #  Управление объектами из списка
    # ----------------------------------------------------------------------
    def delete_selected_object(self):
        """Удаляет выделенный объект. Приоритет — выделение в object_list,
        но если там ничего не выбрано, используем выделение во view."""
        selected_row = self.object_list.currentRow()
        if selected_row < 0:
            selected_row = self.image_view.selected_index
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

        # Соседний объект вместо пустоты.
        new_idx = -1
        if self.current_annotations:
            new_idx = min(idx, len(self.current_annotations) - 1)

        self._rebuild_object_list(keep_row=new_idx if new_idx >= 0 else None)
        self.image_view.set_selected_index(new_idx)
        if new_idx != -1 and not self.image_view.edit_mode:
            self.image_view.set_edit_mode(True)
        elif new_idx == -1:
            self.image_view.set_edit_mode(False)

        self.update_image_display()
        self.log(f"Deleted object {idx + 1}")
        self._revalidate_current()
        self.update_navigation_state()

    # ----------------------------------------------------------------------
    #  Удалить все аннотации текущего снимка
    # ----------------------------------------------------------------------
    def delete_all_annotations(self):
        if not self.image_paths:
            self.log("Нет загруженных изображений.")
            return
        if not self.current_annotations:
            self.log("На текущем снимке нет аннотаций.")
            return

        n = len(self.current_annotations)
        reply = QMessageBox.question(
            self, "Delete All",
            f"Удалить все аннотации ({n}) на текущем снимке?\n"
            f"Другие изображения не затрагиваются.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.current_annotations = []
        self._sync_current_annotations_to_storage()

        self.image_view.set_selected_index(-1)
        self.image_view.set_edit_mode(False)
        self.image_view.set_annotations([], self.img_w, self.img_h)
        self._rebuild_object_list(keep_row=None)
        self.update_image_display()
        self._revalidate_current()
        self.update_navigation_state()
        self.log(f"Удалены все аннотации текущего снимка ({n}).")

    # ----------------------------------------------------------------------
    #  Смена класса выделенного объекта (Ctrl+0..9)
    # ----------------------------------------------------------------------
    def _set_class_of_selected(self, new_class):
        """Устанавливает класс new_class выбранному объекту.
        Приоритет — выделение в object_list, затем — во view."""
        if not self.current_annotations:
            self.log(f"Нет аннотаций — Ctrl+{new_class} пропущен.")
            return

        idx = self.object_list.currentRow()
        if idx < 0 or idx >= len(self.current_annotations):
            idx = self.image_view.selected_index
        if idx < 0 or idx >= len(self.current_annotations):
            self.log(f"Нет выделенного объекта — Ctrl+{new_class} пропущен.")
            return

        ann = self.current_annotations[idx]
        old_cls = int(ann[1])
        new_class = int(new_class)
        if old_cls == new_class:
            self.log(f"Класс объекта #{idx + 1} уже {new_class}.")
            return

        typ = ann[0]
        if typ == 'detect':
            _, _, cx, cy, w, h = ann
            new_ann = ('detect', new_class, cx, cy, w, h)
        elif typ in ('obb', 'segment'):
            _, _, points = ann
            new_ann = (typ, new_class, points)
        else:
            self.log(f"Неподдерживаемый тип аннотации '{typ}'.")
            return

        self.current_annotations[idx] = new_ann
        self._sync_current_annotations_to_storage()

        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        self._rebuild_object_list(keep_row=idx)
        self.image_view.set_selected_index(idx)
        self.update_image_display()
        self._revalidate_current()

        self.log(f"Класс объекта #{idx + 1} ({typ}): {old_cls} → {new_class}.")

    # ----------------------------------------------------------------------
    #  Конвертация detect → OBB (один объект / все снимки)
    # ----------------------------------------------------------------------
    def _convert_object_to_obb(self, idx):
        """Перевести конкретную detect-аннотацию в OBB (по индексу)."""
        if not (0 <= idx < len(self.current_annotations)):
            return
        ann = self.current_annotations[idx]
        if not ann or ann[0] != 'detect':
            self.log(f"Объект #{idx + 1} не detect — конвертация не требуется.")
            return

        new_ann = _detect_annotation_to_obb(ann)
        self.current_annotations[idx] = new_ann
        self._sync_current_annotations_to_storage()

        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        self._rebuild_object_list(keep_row=idx)
        self.image_view.set_selected_index(idx)
        self.update_image_display()
        self._revalidate_current()
        self.log(f"Объект #{idx + 1}: detect → OBB (class={new_ann[1]}).")

    def convert_all_detect_to_obb(self):
        """Переводит все detect-аннотации во всех уже загруженных
        снимках (self.all_annotations[i] is not None) в OBB."""
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений",
                                "Сначала загрузите изображения.")
            return

        total_detect = 0
        loaded_imgs = 0
        skipped_imgs = 0
        for i in range(len(self.image_paths)):
            if i >= len(self.all_annotations) or self.all_annotations[i] is None:
                skipped_imgs += 1
                continue
            loaded_imgs += 1
            for a in self.all_annotations[i]:
                if a and a[0] == 'detect':
                    total_detect += 1

        if total_detect == 0:
            QMessageBox.information(
                self, "Box → OBB",
                "Обычных bounding box (detect) не найдено "
                "среди загруженных в память снимков."
            )
            return

        extra = ""
        if skipped_imgs:
            extra = (f"\n\n⚠️ Пропущено снимков (аннотации не загружены): "
                     f"{skipped_imgs}.\n"
                     f"Откройте их, чтобы включить в конвертацию.")

        reply = QMessageBox.question(
            self, "Box → OBB",
            f"Перевести {total_detect} detect-аннотаций "
            f"в {loaded_imgs} снимках в OBB?\n\n"
            f"Действие необратимо (только в памяти — файлы .txt\n"
            f"перезапишутся при следующем Save Labels / Save All)."
            f"{extra}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        converted = 0
        for i in range(len(self.image_paths)):
            if i >= len(self.all_annotations) or self.all_annotations[i] is None:
                continue
            old = self.all_annotations[i]
            new = []
            changed = False
            for a in old:
                if a and a[0] == 'detect':
                    new.append(_detect_annotation_to_obb(a))
                    converted += 1
                    changed = True
                else:
                    new.append(a)
            if changed:
                self.all_annotations[i] = new

        # Обновляем текущий снимок
        if (0 <= self.current_index < len(self.all_annotations)
                and self.all_annotations[self.current_index] is not None):
            self.current_annotations = list(self.all_annotations[self.current_index])
            self.image_view.set_annotations(self.current_annotations,
                                            self.img_w, self.img_h)
            self._rebuild_object_list(keep_row=None)
            self.update_image_display()
            self._revalidate_current()
            self.update_navigation_state()

        self.log(f"[Box → OBB] Переведено detect-аннотаций: {converted} "
                 f"в {loaded_imgs} снимках.")
        QMessageBox.information(
            self, "Box → OBB",
            f"Переведено detect-аннотаций: {converted}.\n"
            f"Не забудьте сохранить (Save All)."
        )

    # ----------------------------------------------------------------------
    #  Диалог редактирования аннотации
    # ----------------------------------------------------------------------
    def _open_annotation_edit_dialog(self, idx):
        if idx < 0 or idx >= len(self.current_annotations):
            return
        ann = self.current_annotations[idx]
        typ = ann[0]

        if typ == 'detect':
            type_desc = "Bounding Box (detect) — 4 числа: cx cy w h"
        elif typ == 'obb':
            type_desc = "OBB (4 точки) — 8 чисел: x1 y1 x2 y2 x3 y3 x4 y4"
        elif typ == 'segment':
            type_desc = "Segment (полигон) — чётное число ≥ 6: x1 y1 x2 y2 ..."
        else:
            QMessageBox.warning(self, "Ошибка",
                                f"Тип '{typ}' не поддерживает редактирование.")
            return

        if typ == 'detect':
            _, cls_orig, cx, cy, w, h = ann
            coords = [cx, cy, w, h]
        else:
            _, cls_orig, points = ann
            coords = list(points)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Редактирование аннотации #{idx + 1}")
        dlg.setMinimumWidth(520)
        vbox = QVBoxLayout(dlg)

        form = QFormLayout()
        form.addRow("Тип:", QLabel(type_desc))

        cls_spin = QSpinBox()
        cls_spin.setRange(0, 999999)
        cls_spin.setValue(int(cls_orig))
        form.addRow("Class ID:", cls_spin)

        coords_edit = QLineEdit()
        coords_edit.setText(" ".join(f"{c:.6f}" for c in coords))
        coords_edit.setPlaceholderText("Числа через пробел")
        form.addRow("Координаты:", coords_edit)

        vbox.addLayout(form)

        hint = QLabel(
            "Координаты нормализованы в [0, 1] и разделены пробелами. "
            "Можно вводить любые значения — валидация подсветит их красным, "
            "если они вне диапазона, но сохранить всё равно можно."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        vbox.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return

        new_cls = int(cls_spin.value())

        raw = coords_edit.text().replace(",", " ").replace(";", " ").split()
        try:
            new_coords = [float(t) for t in raw]
        except ValueError as e:
            QMessageBox.warning(self, "Ошибка",
                                f"Не удалось разобрать координаты: {e}")
            return

        if typ == 'detect' and len(new_coords) != 4:
            QMessageBox.warning(self, "Ошибка", "Для detect нужно ровно 4 числа.")
            return
        if typ == 'obb' and len(new_coords) != 8:
            QMessageBox.warning(self, "Ошибка", "Для OBB нужно ровно 8 чисел.")
            return
        if typ == 'segment' and (len(new_coords) % 2 != 0 or len(new_coords) < 6):
            QMessageBox.warning(self, "Ошибка",
                                "Для segment нужно чётное число ≥ 6.")
            return

        if typ == 'detect':
            new_ann = ('detect', new_cls, *new_coords)
        elif typ == 'obb':
            norm_canon = canonicalize_obb_corners(new_coords)
            new_ann = ('obb', new_cls, norm_canon)
        else:
            new_ann = (typ, new_cls, new_coords)

        self.current_annotations[idx] = new_ann
        self._sync_current_annotations_to_storage()
        self.image_view.set_annotations(self.current_annotations,
                                        self.img_w, self.img_h)
        self._rebuild_object_list(keep_row=idx)
        self.image_view.set_selected_index(idx)
        self.update_image_display()
        self.log(f"Изменена аннотация #{idx + 1}: type={typ}, class={new_cls}, "
                 f"координат={len(new_coords)}")
        self._revalidate_current()

    def on_object_double_clicked(self, item):
        if self._active_tool_name() is not None:
            self._uncheck_all_tools()
            self.image_view.set_drawing_tool(None)

        idx = item.data(Qt.UserRole)
        if idx is None:
            return
        self._open_annotation_edit_dialog(idx)

    # ----------------------------------------------------------------------
    #  Контекстное меню объекта (общая точка для списка и для изображения)
    # ----------------------------------------------------------------------
    def show_object_context_menu(self, pos):
        """ПКМ по элементу в QListWidget."""
        item = self.object_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        if idx is None:
            return
        self._show_context_menu_for_index(idx, self.object_list.mapToGlobal(pos))

    def on_annotation_right_clicked(self, idx, global_pos):
        """ПКМ по аннотации прямо на изображении.
        Синхронизирует выделение (список ↔ view), затем показывает то же
        контекстное меню, что и для списка объектов."""
        if idx is None or not (0 <= idx < len(self.current_annotations)):
            return

        # Синхронизируем выделение с состоянием view.
        if self.object_list.currentRow() != idx:
            self.object_list.blockSignals(True)
            try:
                self.object_list.setCurrentRow(idx)
            finally:
                self.object_list.blockSignals(False)
            self.image_view.set_selected_index(idx)
            if not self.image_view.edit_mode:
                self.image_view.set_edit_mode(True)
            self.update_image_display()
            self.update_navigation_state()

        self._show_context_menu_for_index(idx, global_pos)

    def _show_context_menu_for_index(self, idx, global_pos):
        """Единая точка построения контекстного меню объекта по его индексу."""
        if not (0 <= idx < len(self.current_annotations)):
            return

        ann = self.current_annotations[idx]
        menu = QMenu(self)

        edit_action = QAction("Edit…", self)
        edit_action.triggered.connect(
            lambda: self._open_annotation_edit_dialog(idx))
        menu.addAction(edit_action)

        # Конвертация detect → OBB (только для обычных боксов)
        if ann and ann[0] == 'detect':
            to_obb_action = QAction("Convert to OBB", self)
            to_obb_action.setToolTip(
                "Заменить этот прямоугольник (detect) на OBB с теми же углами."
            )
            to_obb_action.triggered.connect(
                lambda: self._convert_object_to_obb(idx))
            menu.addAction(to_obb_action)

        menu.addSeparator()
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self.delete_object_by_index(idx))
        menu.addAction(delete_action)

        menu.exec_(global_pos)

    # ----------------------------------------------------------------------
    #  Сохранение
    # ----------------------------------------------------------------------
    def save_labels(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        img_path = self.image_paths[self.current_index]
        txt_path = self._resolve_save_path_for_img(img_path)

        obb_as_xywhr = self.chk_obb_xywhr.isChecked()
        success = save_annotations_for_image(
            self.current_annotations, txt_path, img_path,
            obb_as_xywhr=obb_as_xywhr,
        )
        if success:
            fmt = "xywhr" if obb_as_xywhr else "corners"
            self.log(f"Saved {len(self.current_annotations)} labels to {txt_path} "
                     f"(OBB format: {fmt})")
            QMessageBox.information(self, "Save",
                                    f"Labels saved to {txt_path}")
        else:
            self.log(f"Error saving labels to {txt_path}")
            QMessageBox.critical(self, "Error",
                                 f"Failed to save labels to {txt_path}")

    # ----------------------------------------------------------------------
    #  Сохранить все аннотации
    # ----------------------------------------------------------------------
    def _resolve_save_path_for_img(self, img_path):
        """Куда сохранять .txt для данного изображения (без требования
        существования файла)."""
        if self._yaml_label_map:
            lp = self._yaml_label_map.get(img_path)
            if lp:
                return lp
        if self._label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            return os.path.join(self._label_folder, base + '.txt')
        return os.path.splitext(img_path)[0] + '.txt'

    def save_all_annotations(self):
        """Сохраняет аннотации всех уже загруженных/изменённых изображений."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No images loaded.")
            return

        loaded = []
        for i in range(len(self.image_paths)):
            if i >= len(self.all_annotations):
                continue
            anns = self.all_annotations[i]
            if anns is None:
                continue
            loaded.append((i, anns))

        if not loaded:
            QMessageBox.information(
                self, "Save All",
                "Нет загруженных аннотаций для сохранения.\n"
                "Откройте хотя бы один снимок или запустите авторазметку."
            )
            return

        obb_as_xywhr = self.chk_obb_xywhr.isChecked()
        fmt = "xywhr" if obb_as_xywhr else "corners"

        reply = QMessageBox.question(
            self, "Save All",
            f"Сохранить аннотации для {len(loaded)} изображений?\n\n"
            f"Формат OBB: {fmt}.\n"
            f"Существующие .txt будут перезаписаны.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        ok = 0
        err = 0
        for idx, anns in loaded:
            img_path = self.image_paths[idx]
            txt_path = self._resolve_save_path_for_img(img_path)
            try:
                success = save_annotations_for_image(
                    anns, txt_path, img_path,
                    obb_as_xywhr=obb_as_xywhr,
                )
                if success:
                    ok += 1
                else:
                    err += 1
                    self.log(f"[Save All] ошибка записи: {txt_path}")
            except Exception as e:
                err += 1
                self.log(f"[Save All] исключение для {txt_path}: {e}")

        self.log(f"[Save All] сохранено: {ok}, ошибок: {err} (формат OBB: {fmt})")
        QMessageBox.information(
            self, "Save All",
            f"Сохранено изображений: {ok}\n"
            f"Ошибок: {err}"
        )

    # ----------------------------------------------------------------------
    #  Обработка клавиш
    # ----------------------------------------------------------------------
    def keyPressEvent(self, event):
        # Delete — удалить выделенный объект (в списке или во view).
        if event.key() == Qt.Key_Delete and event.modifiers() == Qt.NoModifier:
            idx = self.object_list.currentRow()
            if idx < 0:
                idx = self.image_view.selected_index
            if 0 <= idx < len(self.current_annotations):
                self.delete_selected_object()
                event.accept()
                return

        # Ctrl+0..9 — установить класс выделенному объекту.
        if event.modifiers() == Qt.ControlModifier:
            key = event.key()
            if Qt.Key_0 <= key <= Qt.Key_9:
                new_class = int(key - Qt.Key_0)
                self._set_class_of_selected(new_class)
                event.accept()
                return

        if event.key() == Qt.Key_Escape:
            if self.image_view.edit_mode:
                self.image_view.set_edit_mode(False)
            elif self._active_tool_name() is not None:
                self.reset_drawing_tool()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)

    # ----------------------------------------------------------------------
    #  Вспомогательные
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
            # Сохраняем текущее выделение при пересборке списка.
            current_row = self.object_list.currentRow()
            self._rebuild_object_list(keep_row=current_row
                                      if current_row >= 0 else None)
            self._apply_validation_colors_to_list()

    def showEvent(self, event):
        self.image_view.setFocus()
        super().showEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Labeler()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))