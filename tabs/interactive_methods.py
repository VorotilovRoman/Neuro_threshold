
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# interactive_methods.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
)

from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import computation as segmentation_computation
from ui.segmentation import controller as segmentation_controller
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import presentation as segmentation_presentation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import InteractivePresetManager
from ui_tabs.interactive_methods_ui import setup_interactive_ui
from utils_new.segmentation.postprocessing import extract_objects


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


        # Пресеты
        self.preset_manager = InteractivePresetManager(self, preset_file="presets_interactive.json")
        self.update_preset_combo()


        # Навигация

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


        # Универсальное сохранение аннотаций


        self.image_view.setFocusPolicy(Qt.StrongFocus)

        self.update_navigation_state()
        self.activate_pan_zoom()
        self.on_method_changed(0)
        self.update_input_ui()
        self.update_params_visibility()
        self.params_widget.grabcut_mode.currentIndexChanged.connect(self.update_input_ui)
        for widget in self.preset_manager._parameter_controls().values():
            signal = widget.currentIndexChanged if hasattr(widget, 'currentIndexChanged') else widget.toggled if hasattr(widget, 'toggled') else widget.valueChanged
            signal.connect(self.on_segmentation_parameters_changed)
        self.setMinimumSize(0, 0)

    # ----------------------------------------------------------------------
    #  Вспомогательные методы
    # ----------------------------------------------------------------------
    def log(self, message):
        segmentation_actions.log(self, message)

    def on_global_settings_changed(self, new_settings=None):
        segmentation_results.refresh_styles(self)

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
        self.on_segmentation_parameters_changed()
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
        self.on_segmentation_parameters_changed()
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
    def on_segmentation_parameters_changed(self, *_):
        segmentation_controller.invalidate_prediction(self)
        self.schedule_update()

    def on_method_changed(self, idx):
        self.on_segmentation_parameters_changed()
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
        segmentation_computation.run_interactive(self)

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

    def _mask_signature(self, frame):
        return frame.key, id(self.mask)

    def _compute_mask(self, frame):
        return self.mask, {'segmentation_view': self.mask}

    def _present_input_frame(self, frame):
        self.current_img_h, self.current_img_w = frame.gray.shape
        self.image_view.set_image_data(self.current_img_w, self.current_img_h, self.fg_scribbles, self.bg_scribbles)
        self.update_scribble_display()

    def display_current_image(self):
        segmentation_presentation.display(self)

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
        segmentation_controller.resize_changed(self, enabled)

    def clear_images(self):
        segmentation_input.clear_session(self, reset=self._reset_interactive_state)

    # ----------------------------------------------------------------------
    #  UI события
    # ----------------------------------------------------------------------
    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    def schedule_update(self):
        segmentation_controller.schedule(self)



    def on_draw_mode_changed(self, idx):
        segmentation_presentation.geometry_changed(self)

    def on_hull_changed(self, state):
        segmentation_presentation.geometry_changed(self)

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    # ----------------------------------------------------------------------
    #  Отрисовка объектов
    # ----------------------------------------------------------------------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentData(), self.hull_checkbox.isChecked())

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
        segmentation_controller.close(self)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveMethodsWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))