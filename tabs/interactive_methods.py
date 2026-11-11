
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# interactive_methods.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
)

from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import InteractivePresetManager
from ui.settings_service import settings
from ui_tabs.interactive_methods_ui import setup_interactive_ui
from utils_new.morphology import apply_morphology
from utils_new.segmentation.postprocessing import extract_objects
from utils_ops.interactive_ops import (
    active_contour_segmentation,
    grabcut_segmentation,
    lazy_snapping,
    onecut_segmentation,
    random_walker_segmentation,
    supercut_segmentation,
    watershed_segmentation,
)


class InteractiveMethodsWindow(QMainWindow):
    segmentation_view_names = ('original_view', 'segmentation_view', 'morph_view', 'annotated_view')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Segmentation")

        # Данные изображений.
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Текущее изображение и его grayscale-версия для всей логики вкладки.
        self.original_image = None
        self.current_gray = None

        # Данные для интерактивной сегментации
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.current_tool = None
        self.current_img_w = 0
        self.current_img_h = 0

        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None

        setup_interactive_ui(self)

        # Ссылка на основной вид (SmartGraphicsView)
        self.image_view = self.original_view

        # Настройка коллбэков SmartGraphicsView
        self.image_view.set_callbacks(
            on_rect_drawn=self.on_rect_drawn,
            on_scribble_added=self.on_scribble_added,
            on_display_update=self.update_scribble_display,
            on_reset_tool=self.activate_pan_zoom
        )

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        # Пресеты
        self.preset_manager = InteractivePresetManager(self, preset_file="presets_interactive.json")
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Навигация
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

        # Кнопки инструментов
        self.rect_mode_btn.clicked.connect(self.activate_rect_tool)
        self.fg_scribble_btn.clicked.connect(self.activate_fg_tool)
        self.bg_scribble_btn.clicked.connect(self.activate_bg_tool)
        self.pan_zoom_btn.clicked.connect(self.activate_pan_zoom)
        self.run_seg_btn.clicked.connect(self.run_segmentation)
        self.reset_mask_btn.clicked.connect(self.reset_mask)

        # Параметры методов
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        # Морфология и отрисовка
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.morphology_widget.parameters_changed.connect(self.schedule_update)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        # Универсальное сохранение аннотаций
        self.save_button.clicked.connect(self.save_current_annotations)

        settings.settings_changed.connect(self.on_global_settings_changed)

        self.image_view.setFocusPolicy(Qt.StrongFocus)

        self.update_navigation_state()
        self.activate_pan_zoom()
        self.on_method_changed(0)
        self.update_input_ui()
        self.update_params_visibility()
        self.params_widget.grabcut_mode.currentIndexChanged.connect(self.update_input_ui)
        self.setMinimumSize(0, 0)

    # ----------------------------------------------------------------------
    #  Вспомогательные методы
    # ----------------------------------------------------------------------
    def log(self, message):
        segmentation_actions.log(self, message)

    def on_global_settings_changed(self, new_settings=None):
        self.update_annotated_view()

    def update_preset_combo(self):
        segmentation_actions.update_preset_combo(self)

    def on_preset_activated(self, index):
        segmentation_actions.apply_preset(self, index)

    def save_as_preset(self):
        segmentation_actions.save_as_preset(self)

    def delete_current_preset(self):
        segmentation_actions.delete_current_preset(self)

    # ----------------------------------------------------------------------
    #  Управление инструментами (через SmartGraphicsView)
    # ----------------------------------------------------------------------
    def activate_rect_tool(self):
        self.current_tool = "rect"
        self.rect_mode_btn.setChecked(True)
        self.fg_scribble_btn.setChecked(False)
        self.bg_scribble_btn.setChecked(False)
        self.pan_zoom_btn.setChecked(False)
        self.image_view.set_drawing_tool("rect")
        self.log("Режим: рисование прямоугольника (один бокс, затем сброс)")

    def activate_fg_tool(self):
        if self.current_tool == "fg":
            self.activate_pan_zoom()
        else:
            self.current_tool = "fg"
            self.rect_mode_btn.setChecked(False)
            self.fg_scribble_btn.setChecked(True)
            self.bg_scribble_btn.setChecked(False)
            self.pan_zoom_btn.setChecked(False)
            self.image_view.set_drawing_tool("fg")
            self.log("Режим: штрихи переднего плана (повторное нажатие – выход)")

    def activate_bg_tool(self):
        if self.current_tool == "bg":
            self.activate_pan_zoom()
        else:
            self.current_tool = "bg"
            self.rect_mode_btn.setChecked(False)
            self.fg_scribble_btn.setChecked(False)
            self.bg_scribble_btn.setChecked(True)
            self.pan_zoom_btn.setChecked(False)
            self.image_view.set_drawing_tool("bg")
            self.log("Режим: штрихи фона (повторное нажатие – выход)")

    def activate_pan_zoom(self):
        self.current_tool = None
        self.rect_mode_btn.setChecked(False)
        self.fg_scribble_btn.setChecked(False)
        self.bg_scribble_btn.setChecked(False)
        self.pan_zoom_btn.setChecked(True)
        self.image_view.set_drawing_tool(None)
        self.log("Режим панорамирования/зума")

    # ----------------------------------------------------------------------
    #  Обработка рисования
    # ----------------------------------------------------------------------
    def on_rect_drawn(self, rect):
        self.rect = rect
        self.log(f"Прямоугольник задан: {self.rect}")
        self.update_input_ui()
        self.activate_pan_zoom()
        self.update_scribble_display()

    def on_scribble_added(self, x, y, tool):
        if tool == 'fg':
            if self.fg_scribbles is None:
                self.fg_scribbles = np.zeros((self.current_img_h, self.current_img_w), dtype=np.uint8)
            cv2.circle(self.fg_scribbles, (x, y), 5, 255, -1)
        else:
            if self.bg_scribbles is None:
                self.bg_scribbles = np.zeros((self.current_img_h, self.current_img_w), dtype=np.uint8)
            cv2.circle(self.bg_scribbles, (x, y), 5, 255, -1)
        self.update_input_ui()
        self.update_scribble_display()

    def update_input_ui(self):
        try:
            method = self.method_combo.currentText()
            rect_ok = bool(self.rect is not None)
            fg_ok = bool(self.fg_scribbles is not None and np.any(self.fg_scribbles == 255))
            bg_ok = bool(self.bg_scribbles is not None and np.any(self.bg_scribbles == 255))

            if method == "GrabCut":
                mode = self.params_widget.grabcut_mode.currentText()
                if mode == "GC_INIT_WITH_RECT":
                    rect_active, fg_active, bg_active = True, False, False
                    rect_required, fg_required, bg_required = True, False, False
                    can_run = rect_ok
                    requirements = "Требуется прямоугольник"
                elif mode == "GC_INIT_WITH_MASK":
                    rect_active, fg_active, bg_active = False, True, True
                    rect_required, fg_required, bg_required = False, True, True
                    can_run = (fg_ok and bg_ok)
                    requirements = "Требуются штрихи FG и BG"
                else:  # "Both"
                    rect_active, fg_active, bg_active = True, True, True
                    rect_required, fg_required, bg_required = False, False, False
                    can_run = rect_ok or (fg_ok and bg_ok)
                    requirements = "Требуется прямоугольник ИЛИ штрихи FG и BG"
            elif method == "Lazy Snapping":
                rect_active, fg_active, bg_active = False, True, True
                rect_required, fg_required, bg_required = False, True, True
                can_run = bool(fg_ok and bg_ok)
                requirements = "Требуются штрихи FG и BG"
            elif method in ("SuperCut", "Active Contours", "OneCut"):
                rect_active, fg_active, bg_active = True, False, False
                rect_required, fg_required, bg_required = True, False, False
                can_run = bool(rect_ok)
                requirements = "Требуется прямоугольник"
            elif method == "Random Walker":
                rect_active, fg_active, bg_active = False, True, True
                rect_required, fg_required, bg_required = False, True, True
                can_run = bool(fg_ok and bg_ok)
                requirements = "Требуются штрихи FG и BG"
            elif method == "Watershed (marker)":
                rect_active = fg_active = bg_active = False
                rect_required = fg_required = bg_required = False
                can_run = True
                requirements = "Не требует ввода (автоматическая сегментация)"
            else:
                rect_active = fg_active = bg_active = False
                rect_required = fg_required = bg_required = False
                can_run = False
                requirements = "Метод не реализован"

            self.rect_mode_btn.setText(f"Draw Bounding Box {'(✓)' if rect_ok else '(✗)'}")
            self.fg_scribble_btn.setText(f"Scribble FG {'(✓)' if fg_ok else '(✗)'}")
            self.bg_scribble_btn.setText(f"Scribble BG {'(✓)' if bg_ok else '(✗)'}")

            def set_style(btn, active, required, ok):
                btn.setEnabled(bool(active))
                if active:
                    if required and not ok:
                        btn.setStyleSheet("background-color: #ffcccc; color: black;")
                    elif ok:
                        btn.setStyleSheet("background-color: #ccffcc; color: black;")
                    else:
                        btn.setStyleSheet("")
                else:
                    btn.setStyleSheet("background-color: #cccccc; color: gray;")

            set_style(self.rect_mode_btn, rect_active, rect_required, rect_ok)
            set_style(self.fg_scribble_btn, fg_active, fg_required, fg_ok)
            set_style(self.bg_scribble_btn, bg_active, bg_required, bg_ok)

            self.requirements_text.setText(requirements)
            self.run_seg_btn.setEnabled(bool(can_run))

        except Exception as e:
            import traceback
            self.log(f"Ошибка в update_input_ui: {e}\n{traceback.format_exc()}")

    def update_scribble_display(self):
        if self.original_image is None:
            return
        original = self.original_image
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        display = original.copy()
        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        if self.fg_scribbles is not None:
            display[self.fg_scribbles == 255] = [0, 255, 0]
        if self.bg_scribbles is not None:
            display[self.bg_scribbles == 255] = [0, 0, 255]
        if self.rect is not None:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if self.image_view.temp_rect is not None:
            x1, y1, x2, y2 = self.image_view.temp_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        if self.image_view.temp_scribble is not None:
            x, y, r = self.image_view.temp_scribble
            color = (0, 255, 0) if self.current_tool == "fg" else (0, 0, 255)
            cv2.circle(display, (x, y), r, color, -1)

        pixmap = numpy_to_qpixmap(display)
        self.original_view.set_pixmap(pixmap)
        self.update_input_ui()

    # ----------------------------------------------------------------------
    #  Гистограмма
    # ----------------------------------------------------------------------
    def update_histogram(self, gray_img):
        segmentation_histogram.update_histogram(self, gray_img)

    def update_current_histogram(self):
        segmentation_histogram.update_current(self)

    # ----------------------------------------------------------------------
    #  Параметры методов
    # ----------------------------------------------------------------------
    def on_method_changed(self, idx):
        method = self.method_combo.currentText()
        self.update_params_visibility()
        self.update_input_ui()

    def update_params_visibility(self):
        method = self.method_combo.currentText()
        self.params_widget.grabcut_group.setVisible(False)
        self.params_widget.lazy_group.setVisible(False)
        self.params_widget.supercut_group.setVisible(False)
        self.params_widget.onecut_group.setVisible(False)
        self.params_widget.watershed_group.setVisible(False)
        self.params_widget.rw_group.setVisible(False)
        self.params_widget.ac_group.setVisible(False)

        if method == "GrabCut":
            self.params_widget.grabcut_group.setVisible(True)
        elif method == "Lazy Snapping":
            self.params_widget.lazy_group.setVisible(True)
        elif method == "SuperCut":
            self.params_widget.lazy_group.setVisible(True)
            self.params_widget.supercut_group.setVisible(True)
        elif method == "OneCut":
            self.params_widget.lazy_group.setVisible(True)
            self.params_widget.onecut_group.setVisible(True)
        elif method == "Random Walker":
            self.params_widget.rw_group.setVisible(True)
        elif method == "Watershed (marker)":
            self.params_widget.watershed_group.setVisible(True)
        elif method == "Active Contours":
            self.params_widget.ac_group.setVisible(True)

    # ----------------------------------------------------------------------
    #  Запуск сегментации
    # ----------------------------------------------------------------------
    def run_segmentation(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Нет изображения", "Загрузите изображение.")
            return

        self.run_seg_btn.setEnabled(False)
        original_text = self.run_seg_btn.text()
        self.run_seg_btn.setText("Сегментация...")
        self.run_seg_btn.setStyleSheet("background-color: orange; color: black;")
        QApplication.processEvents()

        method = self.method_combo.currentText()
        img = self.original_image
        gray = self.current_gray

        if self.fg_scribbles is None:
            self.fg_scribbles = np.zeros(gray.shape, dtype=np.uint8)
        if self.bg_scribbles is None:
            self.bg_scribbles = np.zeros(gray.shape, dtype=np.uint8)

        self.log(f"Запуск сегментации методом {method}...")
        try:
            if method == "GrabCut":
                params = self.params_widget.get_grabcut_params()
                iters = params['iterations']
                mode = params['mode']
                self.mask = grabcut_segmentation(img, self.rect, None, self.fg_scribbles, self.bg_scribbles,
                                                 iterations=iters, mode=mode)
            elif method == "Watershed (marker)":
                params = self.params_widget.get_watershed_params()
                thresh = params['threshold']
                min_distance = params['min_distance']
                self.mask = watershed_segmentation(gray, distance_threshold=thresh, min_area=min_distance)
            elif method == "Lazy Snapping":
                params = self.params_widget.get_superpixel_params()
                self.mask = lazy_snapping(
                    img, self.fg_scribbles, self.bg_scribbles,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],
                    color_sigma=params['color_sigma']
                )
            elif method == "SuperCut":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для SuperCut необходимо нарисовать прямоугольник.")
                    return
                params = self.params_widget.get_supercut_params()
                self.mask = supercut_segmentation(
                    img, self.rect,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],
                    lambda_val=params['lambda_val'],
                    sigma_color=params['sigma_color']
                )
            elif method == "OneCut":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для OneCut необходимо нарисовать прямоугольник.")
                    return
                params = self.params_widget.get_onecut_params()
                self.mask = onecut_segmentation(
                    img, self.rect,
                    superpixel_size=params['superpixel_size'],
                    compactness=params['compactness'],
                    sigma=params['sigma'],
                    spatial_weight=params['spatial_weight'],
                    data_weight=params['data_weight'],
                    color_sigma=params['color_sigma']
                )
            elif method == "Random Walker":
                params = self.params_widget.get_random_walker_params()
                markers = np.zeros(gray.shape, dtype=np.int32)
                markers[self.fg_scribbles == 255] = 1
                markers[self.bg_scribbles == 255] = 2
                if np.sum(markers == 1) == 0 or np.sum(markers == 2) == 0:
                    QMessageBox.warning(self, "Недостаточно меток", "Нанесите штрихи переднего и фонового плана.")
                    return
                self.mask = random_walker_segmentation(img, markers, beta=params['beta'], mode=params['mode'])
            elif method == "Active Contours":
                if self.rect is None:
                    QMessageBox.warning(self, "Нет рамки", "Для Active Contours нарисуйте прямоугольник.")
                    return
                params = self.params_widget.get_active_contour_params()
                self.mask = active_contour_segmentation(
                    gray, self.rect,
                    alpha=params['alpha'],
                    beta=params['beta'],
                    gamma=params['gamma'],
                    max_iter=params['max_iter'],
                    convergence=params['convergence'],
                    smooth=params['smooth'],
                    init_type=params['init_type']
                )
            else:
                self.log(f"Метод {method} не реализован")
                return

            self.schedule_update()
            self.log("Сегментация завершена.")
        except Exception as e:
            self.log(f"Ошибка при сегментации: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить сегментацию:\n{e}")
        finally:
            self.run_seg_btn.setEnabled(True)
            self.run_seg_btn.setText(original_text)
            self.run_seg_btn.setStyleSheet("")
            QApplication.processEvents()

    def reset_mask(self):
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None
        self.image_view.set_image_data(self.current_img_w, self.current_img_h,
                                       self.fg_scribbles, self.bg_scribbles)
        self.log("Маска и все вводные данные сброшены.")
        self.update_input_ui()
        self.update_scribble_display()
        self.schedule_update()

    # ----------------------------------------------------------------------
    #  Отображение маски и результатов
    # ----------------------------------------------------------------------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.update_current_histogram()
            for v in (self.original_view, self.segmentation_view,
                      self.morph_view, self.annotated_view):
                v.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        self.info_label.setText(f"Image {idx + 1} of {total}")
        current_file = os.path.basename(self.image_paths[idx])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"inter_original_{base_name}")
        self.segmentation_view.set_suggested_save_name(f"inter_seg_{base_name}")
        self.morph_view.set_suggested_save_name(f"inter_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"inter_annotated_{base_name}")

        # --- Синхронно получаем элемент ---
        item = self._lazy_cache.get(idx)
        if item is None:
            self.log(f"Не удалось загрузить снимок: {self.image_paths[idx]}")
            return

        original = item["image"]
        if len(original.shape) == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        gray = item["gray"]
        segmentation_results.prepare_frame(self, gray)

        self.original_image = original
        self.current_gray = gray
        self.current_img_h, self.current_img_w = gray.shape

        self.image_view.set_image_data(self.current_img_w, self.current_img_h,
                                       self.fg_scribbles, self.bg_scribbles)

        self.update_histogram(gray)
        self.update_scribble_display()

        if self.mask is not None:
            self.segmentation_view.set_pixmap(numpy_to_qpixmap(self.mask))
        else:
            self.segmentation_view.set_pixmap(numpy_to_qpixmap(np.zeros_like(gray)))

        if self.mask is not None:
            processed = apply_morphology(self.mask, self.morphology_widget.get_steps(),
                                         shape=self.morphology_widget.kernel_shape())
            if self.invert_checkbox.isChecked():
                processed = cv2.bitwise_not(processed)
        else:
            processed = np.zeros_like(gray)
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        if self.mask is not None:
            if len(original.shape) == 2:
                result = original.copy()
                result[processed == 0] = 255
            else:
                result = original.copy()
                result[processed == 0] = [255, 255, 255]
        else:
            result = original.copy()
        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()

        self.current_base_image = display_img.copy()
        if self.mask is not None and self.draw_combo.currentText() != "None":
            _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)
        else:
            self.current_objects_full = []
        self.update_object_list()

        # Синхронизация спинбокса
        self.nav_widget.set_current_index(idx, total)

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ----------------------------------------------------------------------
    #  Список объектов (универсальный)
    # ----------------------------------------------------------------------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    def update_coordinates_display(self):
        segmentation_results.update_coordinates(self)

    def save_current_annotations(self):
        segmentation_export.save_selected(self)
    # ----------------------------------------------------------------------
    #  Навигация и загрузка
    # ----------------------------------------------------------------------
    def update_navigation_state(self):
        segmentation_navigation.update_state(self)

    def _reset_interactive_state(self):
        self.mask = None
        self.rect = None
        self.fg_scribbles = None
        self.bg_scribbles = None

    def prev_image(self):
        segmentation_navigation.step(self, -1, reset=self._reset_interactive_state)

    def next_image(self):
        segmentation_navigation.step(self, 1, reset=self._reset_interactive_state)

    def goto_image(self, page_num):
        segmentation_navigation.navigate(self, page_num-1, reset=self._reset_interactive_state, only_if_changed=True)

    def load_images_from_dialog(self):
        segmentation_input.choose_images(self, self._load_images)

    def load_folder(self):
        segmentation_input.choose_folder(self, self._load_images)

    def reload_current_images(self):
        segmentation_input.reload_images(self, self._load_images)

    # ------- Ленивая загрузка (единая точка входа) -------
    def _load_images(self, source):
        segmentation_input.load_cached(self, source, self.segmentation_view_names, max_side=1024, reset=self._reset_interactive_state)

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
        segmentation_input.clear_session(self, reset=self._reset_interactive_state)

    # ----------------------------------------------------------------------
    #  UI события
    # ----------------------------------------------------------------------
    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    def schedule_update(self):
        self.update_timer.start(50)



    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        if mode in ["Contours (simple)", "Min area rectangle"]:
            self.hull_checkbox.setVisible(True)
        else:
            self.hull_checkbox.setVisible(False)
        if self.original_image is not None:
            self.schedule_update()

    def on_hull_changed(self, state):
        if self.original_image is not None:
            self.schedule_update()

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    # ----------------------------------------------------------------------
    #  Отрисовка объектов
    # ----------------------------------------------------------------------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentText(), self.hull_checkbox.isChecked())

    # ----------------------------------------------------------------------
    #  Лог и гистограмма
    # ----------------------------------------------------------------------
    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)

    # ----------------------------------------------------------------------
    #  Завершение работы
    # ----------------------------------------------------------------------
    def closeEvent(self, event):
        # Аккуратно останавливаем фоновый пул, чтобы процесс не висел
        # на незавершённых задачах загрузки.
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveMethodsWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))