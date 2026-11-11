
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# viewing_dataset.py
import os
import sys
import traceback
from collections import OrderedDict

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPalette
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
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QShortcut,
    QSpinBox,
    QVBoxLayout,
)

from ui_tabs.viewing_dataset_ui import setup_viewing_dataset_ui
from utils.settings import settings
from utils_new.annotation_processing import (
    annotation_in_crop,
    class_id_for_annotation,
    crop_for_annotation,
    find_strong_overlaps,
    mask_roi_at_image_scale,
    normalized_annotation,
    prepare_annotation_preview,
)
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
    read_index_mask,
    read_mask_with_fallback,
    resize_to_max_side,
)
from utils_new.dataset_mask_preview_new import blend_index_mask
from utils_new.dataset_unet_new import (
    load_unet_yaml_pairs,
    write_yolo_txt_for_unet_pairs,
)
from utils_new.dataset_yaml_new import (
    build_yolo_save_target_map,
    find_mask_for_image,
    load_yolo_yaml,
)
from utils_new.dataset_yolo_labels_new import (
    build_label_save_plan,
    canonicalize_obb_corners,
    format_label_save_confirmation,
    load_annotations,
    load_annotations_for_image,
    save_annotations,
    save_annotations_for_image,
)
from utils_new.smart_view_new import numpy_to_qpixmap, set_fill_color_button_preview


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
        self.mask_folder = None
        self._label_folder = None
        self._yaml_label_map = {}
        self._save_label_map = {}
        self._save_source_kind = 'images'
        self._yaml_mask_map = {}
        self._mask_cache = OrderedDict()
        self._mask_cache_max = 16

        # Цвета масок берутся из единой редактируемой dataset_class_palette.
        # Для обычных индексных масок: 0=фон, 1=класс 0, 2=класс 1.
        self.class_colors = {}
        self.mask_opacity = 0.5
        self._mask_pixel_offset = 1
        self._mask_background_value = 0
        self._overlap_warnings = {}
        self._source_image_index = None
        self._source_image_for_crop = None
        self._syncing_selection = False

        # --- Валидация аннотаций ---
        self._validation_per_image = {}
        self._validation_thread = None

        setup_viewing_dataset_ui(self)
        self._fill_color = QColor("#FFFFFF")
        set_fill_color_button_preview(self.btn_fill_color, self._fill_color)
        # Оба окна только просматривают, без инструментов редактирования.
        self.viewer.set_callbacks(
            on_selection_changed=self.on_view_selection_changed,
            on_annotation_right_clicked=self.on_annotation_right_clicked,
        )
        self.viewer.set_display_style(label_mode="all")
        self.crop_view.set_display_style(label_mode="all")

        # --- Сигналы ---
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.load_images.connect(self.load_images)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.btn_save.clicked.connect(self.save_current_annotations)
        self.btn_save_all.clicked.connect(self.save_all_annotations)
        self.btn_delete.clicked.connect(self.delete_selected_object)
        self.btn_load_labels.clicked.connect(self.load_labels_folder)
        self.btn_load_masks.clicked.connect(self.load_masks_folder)
        self.btn_convert_all_to_obb.clicked.connect(self.convert_all_detect_to_obb)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        self.crop_padding_spin.valueChanged.connect(self._update_crop_view)
        self.btn_hide_frames.toggled.connect(self._apply_visibility_controls)
        self.btn_hide_labels.toggled.connect(self._apply_visibility_controls)
        self.btn_fill_objects.toggled.connect(self._apply_visibility_controls)
        self.btn_fill_color.clicked.connect(self.choose_fill_color)
        # Явная привязка шорткатов действует также при фокусе в QListWidget.
        self._class_shortcuts = []
        for cls_id in range(10):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{cls_id}"), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda cls=cls_id: self._set_class_of_selected(cls))
            self._class_shortcuts.append(shortcut)
        self.coord_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.coord_list.customContextMenuRequested.connect(self.show_object_context_menu)
        self.coord_list.currentItemChanged.connect(self.on_object_selection_changed)
        self.coord_list.itemDoubleClicked.connect(self.on_object_double_clicked)
        self.btn_load_yaml.clicked.connect(self.load_yaml)
        self.btn_load_coco.clicked.connect(self.load_coco_json)
        self.btn_load_unet.clicked.connect(self.load_unet_yaml)
        settings.settings_changed.connect(self.on_settings_changed)
        self.toggle_log_btn.clicked.connect(self.toggle_log)

        self.update_navigation_state()
        self._apply_visibility_controls()
        self.on_view_mode_changed()
        self.log_widget.setVisible(False)
        self.toggle_log_btn.setChecked(False)
        self.toggle_log_btn.setText("Показать лог")

    # ------------------------------------------------------------------
    # Логирование / сигналы
    # ------------------------------------------------------------------
    def log(self, message):
        self.log_widget.log(message)
        print(message)

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

    def on_settings_changed(self, new_settings):
        """Все окна Qt обновляют палитру сами, маску перекрашиваем отдельно."""
        if self.image_paths:
            self._refresh_main_pixmap()
            self._update_crop_view()
            self._apply_validation_colors_to_list()

    def choose_fill_color(self):
        """Выбрать цвет Qt-слоя заливки, не редактируя пиксели изображения."""
        color = QColorDialog.getColor(self._fill_color, self, 'Цвет заливки объектов')
        if not color.isValid():
            return
        self._fill_color = QColor(color)
        set_fill_color_button_preview(self.btn_fill_color, self._fill_color)
        self._apply_visibility_controls()

    def _apply_visibility_controls(self, *_):
        """Переключатели и выбранный цвет — сразу в обоих Qt-окнах."""
        visible = dict(
            show_frames=not self.btn_hide_frames.isChecked(),
            show_labels=not self.btn_hide_labels.isChecked(),
            fill_white=self.btn_fill_objects.isChecked(),
            fill_color=self._fill_color.name(),
        )
        self.viewer.set_annotation_visibility(**visible)
        self.crop_view.set_annotation_visibility(**visible)

    def on_view_mode_changed(self, *_):
        mode = int(self.view_mode_combo.currentData() or 1)
        self.full_image_panel.setVisible(mode in (1, 2))
        self.crop_panel.setVisible(mode in (2, 3))
        self.crop_padding_spin.setEnabled(mode in (2, 3))
        if mode in (2, 3) and self.current_annotations:
            if not (0 <= self.selected_index < len(self.current_annotations)):
                self._select_annotation(0)
            self._update_crop_view()
        else:
            self.crop_detail_label.setText("Выберите объект в списке")

    def _select_annotation(self, idx):
        """Синхронно выделяем индекс в списке и на снимке (без рекурсии)."""
        if idx < 0 or idx >= len(self.current_annotations):
            idx = -1
        self.selected_index = idx
        self._syncing_selection = True
        try:
            self.coord_list.blockSignals(True)
            self.coord_list.setCurrentRow(idx)
            self.coord_list.blockSignals(False)
            self.viewer.set_selected_index(idx)
        finally:
            self._syncing_selection = False
        self._update_crop_view()

    def on_view_selection_changed(self, idx):
        """Клик по аннотации на снимке выделяет строку и обновляет кроп."""
        if not self._syncing_selection:
            self._select_annotation(idx)

    def on_annotation_right_clicked(self, idx, global_pos):
        self._select_annotation(idx)
        self._show_object_context_menu_for_index(idx, global_pos)

    def _refresh_main_pixmap(self):
        """Обновляет пиксели только при новом снимке/маске/смене палитры."""
        if self._lazy_cache is None or not self.image_paths:
            return
        item = self._lazy_cache.get(self.current_idx)
        if item is None:
            return
        image = item['image']
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = self._get_mask(self.current_idx)
        if mask is not None:
            image = self.apply_mask_overlay(image, mask)
        # Контуры НЕ внедряются в пиксели — их рисует Qt.
        self.viewer.set_pixmap(numpy_to_qpixmap(image))
        self.viewer.refresh_annotations()

    def _get_source_image_for_crop(self):
        """При включённом resize извлекает кроп из полноразмерного файла.

        В режиме без resize используется уже загруженный numpy-массив.
        Кэшируем ровно один снимок для перемещения по списку аннотаций.
        """
        if not self.image_paths or self._lazy_cache is None:
            return None
        if (self._source_image_index == self.current_idx
                and self._source_image_for_crop is not None):
            return self._source_image_for_crop
        item = self._lazy_cache.get(self.current_idx)
        if item is None:
            return None
        image = item['image']
        if self.nav_widget._resize_cb.isChecked():
            raw = read_image_with_fallback(self.image_paths[self.current_idx])
            if raw is not None:
                image = normalize_to_uint8(raw)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.ndim == 3 and image.shape[2] != 3:
                    image = image[:, :, :3]
            else:
                self.log("[Просмотр объекта] Не удалось прочитать исходное разрешение; "
                         "используется изображение из текущего кэша.")
        self._source_image_for_crop = image
        self._source_image_index = self.current_idx
        return image

    def _update_crop_view(self, *_):
        mode = int(self.view_mode_combo.currentData() or 1)
        if mode == 1:
            return
        idx = self.selected_index
        if idx < 0 or idx >= len(self.current_annotations):
            self.crop_view.set_annotations([], 0, 0)
            self.crop_view.set_pixmap(numpy_to_qpixmap(None))
            self.crop_detail_label.setText("Выберите объект в списке")
            return
        image = self._get_source_image_for_crop()
        if image is None:
            self.crop_detail_label.setText("Не удалось загрузить исходный снимок")
            return
        ann = self.current_annotations[idx]
        h, w = image.shape[:2]
        mask = self._get_mask(self.current_idx)
        crop, crop_ann, bounds = prepare_annotation_preview(
            image, ann, padding=self.crop_padding_spin.value(),
        )
        if crop is None:
            self.crop_view.set_annotations([], 0, 0)
            self.crop_view.set_pixmap(numpy_to_qpixmap(None))
            self.crop_detail_label.setText("Некорректная геометрия аннотации")
            return
        if mask is not None:
            mask_crop = mask_roi_at_image_scale(mask, w, h, bounds)
            crop, _info = blend_index_mask(
                crop, mask_crop, settings.get_dataset_class_color,
                opacity=self.mask_opacity,
                pixel_offset=self._mask_pixel_offset,
                background_value=self._mask_background_value,
                annotations=[crop_ann] if crop_ann else None,
                prefer_annotation_classes=True,
            )
        ch, cw = crop.shape[:2]
        self.crop_view.set_annotations([crop_ann] if crop_ann else [], cw, ch)
        # QPixmap не перерисовывается из исходного изображения и не проходит
        # через cv2.resize. Qt делает только экранный zoom/fitInView.
        self.crop_view.set_pixmap(
            numpy_to_qpixmap(crop, max_side=max(cw, ch)), preserve_view=False)
        cls = class_id_for_annotation(ann)
        self.crop_detail_label.setText(
            f"Объект #{idx + 1} · класс {cls} · {cw}×{ch} px")

    def update_navigation_state(self):
        has_images = len(self.image_paths) > 0
        self.nav_widget.set_navigation_enabled(has_images)
        self.btn_save.setEnabled(has_images)
        self.btn_save_all.setEnabled(has_images)
        self.btn_delete.setEnabled(has_images and len(self.current_annotations) > 0)
        self.btn_convert_all_to_obb.setEnabled(has_images)

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
        self._source_image_for_crop = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Ленивая загрузка изображений (единая точка входа)
    # ------------------------------------------------------------------
    def _load_images(self, source, label_map=None, mask_map=None,
                     label_folder=None, mask_folder=None,
                     mask_pixel_offset=1):
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
        self.current_idx = 0
        self.selected_index = -1
        self.current_annotations = []
        self.annotations = [None] * len(all_files)

        self._yaml_label_map = dict(label_map) if label_map else {}
        self._save_label_map = {}
        self._save_source_kind = 'images'
        self._yaml_mask_map = dict(mask_map) if mask_map else {}
        self._label_folder = label_folder
        self.mask_folder = mask_folder
        self._mask_cache = OrderedDict()
        self._mask_pixel_offset = int(mask_pixel_offset)
        # В U-Net offset=0 означает pixel 0 == class 0 (не фон).
        self._mask_background_value = (None if self._mask_pixel_offset == 0 else 0)
        self.class_colors = {}
        self._source_image_index = None
        self._source_image_for_crop = None
        self._overlap_warnings = {}

        first = self._lazy_cache.get(0)
        if first is None:
            self.log("Не удалось загрузить первое изображение.")
            return

        total = len(self._lazy_cache)
        self._lazy_cache.prefetch(
            range(1, min(1 + self._lazy_cache.prefetch_ahead + 1, total))
        )

        self.log(f"Загружено {total} изображений (первое — сразу, остальное в фоне).")
        self.show_current_image()
        self.update_navigation_state()
        self.nav_widget.set_current_index(self.current_idx, total)

        # Запускаем фоновую валидацию всех пар
        self._start_annotation_validation()

    def _clear_view(self):
        self._source_image_index = None
        self._source_image_for_crop = None
        self._overlap_warnings = {}
        self.viewer.set_annotations([], 0, 0)
        self.viewer.set_pixmap(numpy_to_qpixmap(None))
        self.crop_view.set_annotations([], 0, 0)
        self.crop_view.set_pixmap(numpy_to_qpixmap(None))
        self.coord_list.clear()
        self.crop_detail_label.setText("Выберите объект в списке")
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
                ann = load_annotations_for_image(lp, img_path)
        elif self._label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lp = os.path.join(self._label_folder, base + '.txt')
            if os.path.exists(lp):
                ann = load_annotations_for_image(lp, img_path)
        else:
            item = self._lazy_cache.get(idx) if self._lazy_cache else None
            if item:
                ann = list(item.get("annotations", []) or [])

        while len(self.annotations) <= idx:
            self.annotations.append(None)
        ann = [normalized_annotation(a) for a in ann]
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
                if os.path.exists(cand) and os.path.normcase(os.path.abspath(cand)) != os.path.normcase(os.path.abspath(img_path)):
                    mask_path = cand
                    break

        mask = None
        if mask_path and os.path.exists(mask_path):
            try:
                mask = read_index_mask(mask_path)
                if mask.size and not np.any(mask != 0) and self._mask_background_value == 0:
                    self.log(f"⚠️ Маска состоит только из фона (pixel=0): {mask_path}")
                elif idx < 3:
                    values, counts = np.unique(mask, return_counts=True)
                    summary = ', '.join(f'{int(v)}:{int(n)}'
                                        for v, n in zip(values[:12], counts[:12]))
                    self.log(f"[Mask preview] {os.path.basename(mask_path)}: "
                             f"dtype={mask.dtype}, values={{ {summary} }}")
            except Exception as e:
                self.log(f"Warning: could not load mask {mask_path}: {e}")

        if mask is None and (self._yaml_mask_map or self.mask_folder):
            self.log(f"⚠️ Маска не найдена/не прочитана для {os.path.basename(img_path)}"
                     + (f" (путь: {mask_path})" if mask_path else ""))
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

        self._label_folder = folder
        self._save_label_map = {}
        self._save_source_kind = 'labels'
        self._yaml_label_map = {}
        self.annotations = [None] * len(self.image_paths)
        self.current_annotations = []
        self.selected_index = -1

        found = 0
        for img_path in self.image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            if os.path.exists(os.path.join(folder, base + '.txt')):
                found += 1
        self.log(f"Found label files for {found} / {len(self.image_paths)} images.")
        self._start_annotation_validation()
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
        self._mask_pixel_offset = 1  # обычная ручная маска: 0=фон, 1=класс 0
        self._mask_background_value = 0  # сброс после U-Net offset=0
        self._mask_cache = OrderedDict()
        self.class_colors = {}
        self.log(f"Masks folder set: {folder} (маски грузятся по мере просмотра).")
        self.show_current_image()

    def on_opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self._refresh_main_pixmap()
        self._update_crop_view()

    def apply_mask_overlay(self, img, mask):
        """Цвет пикселя маски соответствует цвету КЛАССА в общей палитре."""
        blended, _info = blend_index_mask(
            img, mask, settings.get_dataset_class_color,
            opacity=self.mask_opacity,
            pixel_offset=self._mask_pixel_offset,
            background_value=self._mask_background_value,
            annotations=self.current_annotations,
            prefer_annotation_classes=True,
        )
        return blended

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
        list_widget.blockSignals(True)
        try:
            list_widget.clear()
            for idx, original_ann in enumerate(annotations):
                ann = normalized_annotation(original_ann)
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
        finally:
            list_widget.blockSignals(False)

    # ------------------------------------------------------------------
    # Отображение текущего изображения
    # ------------------------------------------------------------------
    def show_current_image(self):
        """Новый снимок/изменение данных; Qt-слои разметки отдельно от QPixmap."""
        if self._lazy_cache is None or not self.image_paths:
            return
        idx = self.current_idx
        total = len(self._lazy_cache)
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Failed to load image {self.image_paths[idx]}")
            return
        image = item['image']
        self.current_img_h, self.current_img_w = image.shape[:2]
        self.current_annotations = list(self._get_annotations(idx))
        self._overlap_warnings = find_strong_overlaps(
            self.current_annotations, self.current_img_w, self.current_img_h,
            threshold=0.70)
        self.viewer.set_annotations(self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        self._refresh_main_pixmap()
        self.update_annotation_list(self.coord_list, self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        self._apply_validation_colors_to_list()
        if not (0 <= self.selected_index < len(self.current_annotations)):
            mode = int(self.view_mode_combo.currentData() or 1)
            self.selected_index = 0 if mode in (2, 3) and self.current_annotations else -1
        self._select_annotation(self.selected_index)
        self.update_info_label()
        self.update_navigation_state()
        self.nav_widget.set_current_index(idx, total)
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ------------------------------------------------------------------
    # Валидация аннотаций (фоновый поток)
    # ------------------------------------------------------------------
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
                "color: #0055aa; font-weight: bold; padding: 2px;"
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
            f"Некорректные объекты выделены красным в списке справа."
        )
        self.validation_status_label.setStyleSheet(
            "color: #b00000; font-weight: bold; padding: 2px;"
        )
        self.validation_status_label.setVisible(True)

    def _apply_validation_colors_to_list(self):
        """Красный: ошибка валидации или перекрытие >= 70% меньшего объекта."""
        per_file = self._validation_per_image.get(self.current_idx, [])
        normal = self.coord_list.palette().color(QPalette.Text)
        red = QColor(220, 40, 40)
        for row in range(self.coord_list.count()):
            item = self.coord_list.item(row)
            if item is None:
                continue
            text = item.text().split("  [INVALID:")[0].split("  [OVERLAP:")[0]
            if text.startswith("⚠ "):
                text = text[2:]
            notes = []
            if row < len(per_file):
                valid, error = per_file[row]
                if not valid:
                    notes.append(f"Ошибка разметки: {error}")
                    text += f"  [INVALID: {error}]"
            overlaps = self._overlap_warnings.get(row, [])
            if overlaps:
                neighbors = ", ".join(
                    f"#{other + 1} ({score * 100:.1f}%)"
                    for other, score in overlaps)
                notes.append("Сильное пересечение с " + neighbors)
                text += "  [OVERLAP: ≥70%]"
            item.setText("⚠ " + text if notes else text)
            item.setForeground(red if notes else normal)
            item.setToolTip("\n".join(notes))

    def _revalidate_current(self):
        self._validation_per_image[self.current_idx] = [
            _validate_unified_annotation(a) for a in self.current_annotations
        ]
        self._update_validation_status()
        self._apply_validation_colors_to_list()

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    def save_current_annotations(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        img_path = self.image_paths[self.current_idx]
        as_xywhr = self.chk_obb_xywhr.isChecked()
        try:
            plan = self._confirm_label_save([img_path], 'Save', as_xywhr)
            if plan is None:
                return
            txt_path = plan.entries[0][1]
            success = save_annotations_for_image(
                self.current_annotations, txt_path, img_path,
                obb_as_xywhr=as_xywhr)
            if success:
                self._sync_current_annotations()
                self._after_label_saved(img_path, txt_path)
                self.log(f"Saved {len(self.current_annotations)} annotations to {txt_path}")
                QMessageBox.information(self, "Save", f"Annotations saved to {txt_path}")
            else:
                self.log(f"Error saving annotations to {txt_path}")
                QMessageBox.critical(self, "Error", f"Failed to save to {txt_path}")
        except Exception as exc:
            self.log(f"[Save] ошибка: {exc}")
            QMessageBox.critical(self, "Save", str(exc))

    def _resolve_save_path_for_img(self, img_path):
        """Destination is separate from the label path used during loading."""
        if img_path in self._save_label_map:
            return self._save_label_map[img_path]
        if self._label_folder:
            base = os.path.splitext(os.path.basename(img_path))[0]
            return os.path.join(self._label_folder, base + '.txt')
        if self._save_source_kind in ('yolo', 'unet', 'coco'):
            raise ValueError(f'Missing save destination for {img_path}')
        return os.path.splitext(img_path)[0] + '.txt'

    def _confirm_label_save(self, items, operation, obb_as_xywhr):
        """One shared plan/summary for Save and Save All, both tabs."""
        plan = build_label_save_plan(
            (img, self._resolve_save_path_for_img(img)) for img in items)
        summary = format_label_save_confirmation(
            plan, operation=operation,
            obb_format='xywhr' if obb_as_xywhr else 'corners',
            source_kind=self._save_source_kind,
            unloaded_count=(max(0, len(self.image_paths)-len(items))
                            if operation == 'Save All' else 0))
        answer = QMessageBox.question(
            self, operation, summary, QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)
        return plan if answer == QMessageBox.Yes else None

    def _after_label_saved(self, image_path, target_path):
        """Reopen future lazy reads from the newly saved canonical YOLO path."""
        if self._save_source_kind == 'yolo':
            self._yaml_label_map[image_path] = target_path

    def save_all_annotations(self):
        """Save All по образцу вкладки разметки; ленивые None не трогаем."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Image", "No images loaded.")
            return
        loaded = [(idx, anns) for idx, anns in enumerate(self.annotations)
                  if idx < len(self.image_paths) and anns is not None]
        if not loaded:
            QMessageBox.information(
                self, "Save All", "Нет загруженных аннотаций для сохранения.\n"
                "Просмотрите хотя бы один снимок.")
            return
        as_xywhr = self.chk_obb_xywhr.isChecked()
        fmt = "xywhr" if as_xywhr else "corners"
        try:
            plan = self._confirm_label_save(
                [self.image_paths[idx] for idx, _ in loaded], 'Save All', as_xywhr)
        except Exception as exc:
            self.log(f"[Save All] ошибка подготовки: {exc}")
            QMessageBox.critical(self, "Save All", str(exc))
            return
        if plan is None:
            return

        success_count = error_count = 0
        for idx, anns in loaded:
            path = self._resolve_save_path_for_img(self.image_paths[idx])
            try:
                success = save_annotations_for_image(
                    anns, path, self.image_paths[idx], obb_as_xywhr=as_xywhr)
                # Реальные размеры берутся из соответствующего снимка.
                if success:
                    success_count += 1
                    self._after_label_saved(self.image_paths[idx], path)
                else:
                    error_count += 1
                    self.log(f"[Save All] ошибка записи: {path}")
            except Exception as exc:
                error_count += 1
                self.log(f"[Save All] исключение для {path}: {exc}")
        self.log(f"[Save All] сохранено: {success_count}, ошибок: {error_count} "
                 f"(формат OBB: {fmt})")
        QMessageBox.information(
            self, "Save All", f"Сохранено изображений: {success_count}\n"
            f"Ошибок: {error_count}")

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
            saved_targets = dict(self._save_label_map)
            saved_kind = self._save_source_kind
            self._load_images(
                list(self.image_paths),
                label_map=self._yaml_label_map if self._yaml_label_map else None,
                mask_map=self._yaml_mask_map if self._yaml_mask_map else None,
                label_folder=self._label_folder,
                mask_folder=self.mask_folder,
                mask_pixel_offset=self._mask_pixel_offset,
            )
            self._save_label_map = saved_targets
            self._save_source_kind = saved_kind
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
            self._save_label_map = {}
            self._save_source_kind = 'images'
            self.class_colors = {}
            self._validation_per_image = {}
            self.validation_status_label.setVisible(False)
            self.validation_status_label.setText("")
            self._source_image_index = None
            self._source_image_for_crop = None
            self.viewer.set_annotations([], 0, 0)
            self.viewer.set_pixmap(numpy_to_qpixmap(None))
            self.crop_view.set_annotations([], 0, 0)
            self.crop_view.set_pixmap(numpy_to_qpixmap(None))
            self.coord_list.clear()
            self.nav_widget.set_current_index(0, 0)
            self.update_navigation_state()
            self.info_label.setText("No images")
            self.log("Resize mode changed, images cleared. Please load images again.")

    # ------------------------------------------------------------------
    # Работа со списком объектов
    # ------------------------------------------------------------------
    def on_object_selection_changed(self, current, previous):
        if self._syncing_selection:
            return
        idx = current.data(Qt.UserRole) if current is not None else -1
        if idx != self.selected_index:
            self._select_annotation(idx)
            if idx >= 0:
                self.log(f"Selected object {idx + 1}")

    def show_object_context_menu(self, pos):
        item = self.coord_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        if idx is None or not (0 <= idx < len(self.current_annotations)):
            return

        self._select_annotation(idx)
        self._show_object_context_menu_for_index(
            idx, self.coord_list.mapToGlobal(pos))

    def _show_object_context_menu_for_index(self, idx, global_pos):
        if idx < 0 or idx >= len(self.current_annotations):
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
        delete_action.triggered.connect(
            lambda: self.delete_annotation_by_index(idx))
        menu.addAction(delete_action)

        menu.exec_(global_pos)

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
        self._sync_current_annotations()

        # Обновляем только список и вид — выделение сохраняем
        self.update_annotation_list(self.coord_list, self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        self.coord_list.blockSignals(True)
        self.coord_list.setCurrentRow(idx)
        self.coord_list.blockSignals(False)

        self.selected_index = idx
        self.show_current_image()
        self._revalidate_current()
        self.log(f"Объект #{idx + 1}: detect → OBB (class={new_ann[1]}).")

    def convert_all_detect_to_obb(self):
        """Переводит все detect-аннотации во всех уже загруженных
        снимках (self.annotations[i] is not None) в OBB."""
        if not self.image_paths:
            QMessageBox.warning(self, "Нет изображений",
                                "Сначала загрузите изображения.")
            return

        total_detect = 0
        loaded_imgs = 0
        skipped_imgs = 0
        for i in range(len(self.image_paths)):
            if i >= len(self.annotations) or self.annotations[i] is None:
                skipped_imgs += 1
                continue
            loaded_imgs += 1
            for a in self.annotations[i]:
                if a and a[0] == 'detect':
                    total_detect += 1

        if total_detect == 0:
            QMessageBox.information(
                self, "Box → OBB",
                "Обычных bounding box (detect) не найдено "
                "среди уже загруженных в память снимков."
            )
            return

        extra = ""
        if skipped_imgs:
            extra = (f"\n\n⚠️ Пропущено снимков (аннотации ещё не загружены "
                     f"в память): {skipped_imgs}.\n"
                     f"Просмотрите их, чтобы включить в конвертацию.")

        reply = QMessageBox.question(
            self, "Box → OBB",
            f"Перевести {total_detect} detect-аннотаций "
            f"в {loaded_imgs} снимках в OBB?\n\n"
            f"Действие необратимо (только в памяти — файлы .txt\n"
            f"перезапишутся при следующем Save Labels)."
            f"{extra}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        converted = 0
        for i in range(len(self.image_paths)):
            if i >= len(self.annotations) or self.annotations[i] is None:
                continue
            old = self.annotations[i]
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
                self.annotations[i] = new

        # Обновляем текущий снимок
        if (0 <= self.current_idx < len(self.annotations)
                and self.annotations[self.current_idx] is not None):
            self.current_annotations = list(self.annotations[self.current_idx])
            self.update_annotation_list(self.coord_list, self.current_annotations,
                                        self.current_img_w, self.current_img_h)
            self.show_current_image()
            self._revalidate_current()

        self.log(f"[Box → OBB] Переведено detect-аннотаций: {converted} "
                 f"в {loaded_imgs} снимках.")
        QMessageBox.information(
            self, "Box → OBB",
            f"Переведено detect-аннотаций: {converted}.\n"
            f"Не забудьте сохранить (Save Labels)."
        )

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
            self._revalidate_current()

    def delete_selected_object(self):
        current_row = self.coord_list.currentRow()
        if 0 <= current_row < len(self.current_annotations):
            self.delete_annotation_by_index(current_row)
        else:
            self.log("No object selected for deletion.")

    # ------------------------------------------------------------------
    #  Смена класса выделенного объекта (Ctrl+0..9)
    # ------------------------------------------------------------------
    def _set_class_of_selected(self, new_class):
        """Устанавливает класс new_class выбранному объекту.
        Приоритет — выделение в coord_list, затем — self.selected_index."""
        if not self.current_annotations:
            self.log(f"Нет аннотаций — Ctrl+{new_class} пропущен.")
            return

        idx = self.coord_list.currentRow()
        if idx < 0 or idx >= len(self.current_annotations):
            idx = self.selected_index
        if idx < 0 or idx >= len(self.current_annotations):
            self.log(f"Нет выделенного объекта — Ctrl+{new_class} пропущен.")
            return

        ann = self.current_annotations[idx]
        try:
            old_cls = int(ann[1])
        except (TypeError, ValueError):
            old_cls = 0

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
        self._sync_current_annotations()

        # Обновляем список и восстанавливаем выделение
        self.update_annotation_list(self.coord_list, self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        self.coord_list.blockSignals(True)
        self.coord_list.setCurrentRow(idx)
        self.coord_list.blockSignals(False)

        self.selected_index = idx
        self.show_current_image()
        self._revalidate_current()

        self.log(f"Класс объекта #{idx + 1} ({typ}): {old_cls} → {new_class}.")
    # ------------------------------------------------------------------
    # Диалог редактирования аннотации: класс + координаты
    # ------------------------------------------------------------------
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
        self._sync_current_annotations()
        self.update_annotation_list(self.coord_list, self.current_annotations,
                                    self.current_img_w, self.current_img_h)
        if 0 <= idx < self.coord_list.count():
            self.coord_list.blockSignals(True)
            self.coord_list.setCurrentRow(idx)
            self.coord_list.blockSignals(False)
        self.show_current_image()
        self.log(f"Изменена аннотация #{idx + 1}: type={typ}, class={new_cls}, "
                 f"координат={len(new_coords)}")
        self._revalidate_current()

    def on_object_double_clicked(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None:
            return
        self._open_annotation_edit_dialog(idx)


    def toggle_log(self, checked):
        self.log_widget.setVisible(checked)
        self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

    # ------------------------------------------------------------------
    # YAML датасет
    # ------------------------------------------------------------------
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
                QMessageBox.warning(self, "Ошибка",
                                     "Не найдено изображений в YAML.")
                return

            image_paths = [p for p, _ in pairs]
            label_map = {p: lp for p, lp in pairs if lp is not None}
            save_target_map = build_yolo_save_target_map(
                file_path, image_paths, root=root)

            # Ищем маски (необязательно)
            mask_map = {}
            for img in image_paths:
                m = find_mask_for_image(img, base_path=root)
                if m:
                    mask_map[img] = m

            self._load_images(
                image_paths,
                label_map=label_map,
                mask_map=mask_map,
                label_folder=None,
                mask_folder=None,
            )
            self._save_label_map = save_target_map
            self._save_source_kind = 'yolo'

            self.log(f"=== Загрузка YAML: {file_path} ===")
            self.log(f"Root: {root}")
            for split, n in splits_info.items():
                if n:
                    self.log(f"  • {split}: {n} изображений")
            self.log(f"Загружено {len(image_paths)} изображений "
                     f"(метки: {len(label_map)}, маски: {len(mask_map)}).")
            QMessageBox.information(
                self, "Успех",
                f"Загружено {len(image_paths)} изображений.\n"
                f"Метки: {len(label_map)}\n"
                f"Маски: {len(mask_map)}"
            )
        except Exception as e:
            self._log_exception("Ошибка загрузки YAML", e)
            QMessageBox.critical(self, "Ошибка",
                                  f"Не удалось загрузить YAML:\n{e}")


    # ------------------------------------------------------------------
    # COCO
    # ------------------------------------------------------------------
    def load_coco_json(self):
        json_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select COCO JSON(s) — можно несколько",
            "", "JSON files (*.json);;All files (*)")
        if not json_paths:
            return
        try:
            self._load_coco_into_viewer(json_paths)
        except Exception as e:
            self._log_exception("Ошибка загрузки COCO", e)
            QMessageBox.critical(self, "Ошибка COCO", str(e))

    def _load_coco_into_viewer(self, json_paths):
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
        self._save_label_map = dict(label_map)
        self._save_source_kind = 'coco'
        self.log(f"=== Загрузка COCO ({len(catalog.sources)} JSON) ===")
        self.log(f"Изображений: {len(image_paths)}, классов: {len(catalog.class_names)}")
        self.log(f"Временные labels: {tmp_dir}")
        QMessageBox.information(
            self, "COCO", f"Загружено изображений: {len(image_paths)}\n"
            f"Классов: {len(catalog.class_names)}\nВременные labels: {tmp_dir}")

    # ------------------------------------------------------------------
    # U-Net YAML
    # ------------------------------------------------------------------
    def load_unet_yaml(self):
        yaml_path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net YAML", "",
            "YAML files (*.yaml *.yml);;All files (*)")
        if not yaml_path:
            return
        try:
            self._load_unet_into_viewer(yaml_path)
        except Exception as e:
            self._log_exception("Ошибка загрузки U-Net YAML", e)
            QMessageBox.critical(self, "Ошибка U-Net YAML", str(e))

    def _load_unet_into_viewer(self, yaml_path):
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
        original_masks = dict(pairs_raw)
        mask_map = {img: original_masks[img] for img in image_paths}
        self._load_images(image_paths, label_map=label_map, mask_map=mask_map,
                          mask_pixel_offset=mask_offset)
        self._save_label_map = dict(label_map)
        self._save_source_kind = 'unet'
        self.log(f"=== Загрузка U-Net YAML: {yaml_path} ===")
        self.log(f"Root: {root}, пар: {len(image_paths)}, "
                 f"classes: {len(class_names)}, offset={mask_offset}")
        self.log("Оригинальные маски сохранены для отображения.")
        QMessageBox.information(
            self, "U-Net YAML", f"Загружено изображений: {len(image_paths)}\n"
            f"Классов: {len(class_names)}\nМаски сконвертированы в полигоны.\n"
            "Оригинальные маски отображаются как overlay.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ViewingDataset()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))