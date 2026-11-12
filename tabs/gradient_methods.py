
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# gradient_methods.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import json
import os
import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
)

from path_setup import get_project_root
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import controller as segmentation_controller
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import presentation as segmentation_presentation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import GradientPresetManager
from ui_tabs.gradient_methods_ui import setup_gradient_ui
from utils_auto_research.gradient_auto_settings_dialog import GradientAutoSettingsDialog
from utils_new.segmentation.postprocessing import extract_objects
from utils_ops.gradient_ops import apply_gradient_method


class GradientMethodsWindow(QMainWindow):
    segmentation_view_names = ('original_view', 'gradient_view', 'binary_view', 'morph_view', 'filled_view', 'annotated_view')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gradient Methods")

        # Данные изображений.
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Текущее изображение и его grayscale-версия для всей логики вкладки.
        self.original_image = None
        self.current_gray = None

        self.current_gradient_mask = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None

        setup_gradient_ui(self)

        _root = get_project_root()
        auto_settings_dir = os.path.join(_root, "settings", "auto_settings")
        os.makedirs(auto_settings_dir, exist_ok=True)
        self.settings_file = os.path.join(auto_settings_dir, "auto_settings_gradient.json")

        self.auto_settings = None
        self.load_auto_settings()
        # Пресеты
        self.preset_manager = GradientPresetManager(self, preset_file="presets_gradient.json")
        self.update_preset_combo()


        # Сигналы

        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        self.sobel_kernel.valueChanged.connect(self.schedule_update)
        self.sobel_scale.valueChanged.connect(self.schedule_update)
        self.lap_kernel.valueChanged.connect(self.schedule_update)
        self.lap_scale.valueChanged.connect(self.schedule_update)
        self.canny_thresh1.valueChanged.connect(self.on_canny_thresh_changed)
        self.canny_thresh2.valueChanged.connect(self.on_canny_thresh_changed)
        self.canny_aperture.valueChanged.connect(self.schedule_update)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.fill_contours_checkbox.stateChanged.connect(self.schedule_update)
        self.min_area_spinbox.valueChanged.connect(self.on_min_area_changed)

        # Замена сохранения на универсальное

        self.configure_auto_button.clicked.connect(self.open_auto_settings)
        self.auto_button.clicked.connect(self.run_auto_optimization)


        self.on_method_changed(self.method_combo.currentIndex())
        self.update_navigation_state()

    # ---------- Логирование, пресеты, настройки ----------
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

    def load_auto_settings(self):
        if not os.path.exists(self.settings_file):
            self.auto_settings = None
            self.log("Файл настроек автоподбора для градиентов не найден, будут использованы настройки по умолчанию.")
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8-sig') as f:
                content = f.read().strip()
                if not content:
                    raise ValueError("Empty file")
                self.auto_settings = json.loads(content)
            self.log(f"Настройки автоподбора загружены из {self.settings_file}")
        except Exception as e:
            self.log(f"Ошибка загрузки настроек автоподбора: {e}. Будет создан файл с настройками по умолчанию.")
            self.auto_settings = None
            self.save_auto_settings()

    def save_auto_settings(self):
        if self.auto_settings is None:
            from utils_auto_research.gradient_auto_settings_dialog import (
                GradientAutoSettingsDialog,
            )
            self.auto_settings = GradientAutoSettingsDialog._default_settings(None)
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.auto_settings, f, indent=4, ensure_ascii=False)
            self.log(f"Настройки автоподбора сохранены в {self.settings_file}")
        except Exception as e:
            self.log(f"Ошибка сохранения настроек автоподбора: {e}")

    def closeEvent(self, event):
        segmentation_controller.close(self)
        self.save_auto_settings()
        event.accept()

    def open_auto_settings(self):
        dlg = GradientAutoSettingsDialog(self, self.auto_settings)
        if dlg.exec_():
            self.auto_settings = dlg.get_settings()
            self.save_auto_settings()
            self.log("Настройки автоподбора для градиентов обновлены и сохранены")
        else:
            self.log("Настройки автоподбора не изменены")

    def run_auto_optimization(self):
        from ui.autoreseach_ui.controls import GRADIENT_UNAVAILABLE
        QMessageBox.information(self, "Автоподбор", GRADIENT_UNAVAILABLE)

    # ---------- Гистограмма ----------
    def update_histogram(self, gray_img, threshold=None):
        segmentation_histogram.update_histogram(self, gray_img)

    def update_current_histogram(self):
        segmentation_histogram.update_current(self)

    # ---------- Параметры градиентного метода ----------
    def get_current_gradient_params(self):
        method = self.method_combo.currentText()
        params = {}
        if method in ("Sobel", "Scharr"):
            params["scale"] = self.sobel_scale.value()
            if method == "Sobel":
                params["ksize"] = self.sobel_kernel.value()
        elif method == "Laplacian":
            params["ksize"] = self.lap_kernel.value()
            params["scale"] = self.lap_scale.value()
        elif method == "Canny":
            params["threshold1"] = self.canny_thresh1.value()
            params["threshold2"] = self.canny_thresh2.value()
            params["aperture"] = self.canny_aperture.value()
        return params

    # ---------- События слайдеров ----------
    def on_threshold_changed(self, value):
        self.threshold_label.setText(str(value))
        self.schedule_update()

    def on_min_area_changed(self, value):
        self.schedule_update()



    # ---------- Основная логика ----------

    def _mask_signature(self, frame):
        return frame.key, self.method_combo.currentText(), repr(self.get_current_gradient_params()), self.threshold_slider.value()

    def _compute_mask(self, frame):
        gradient = apply_gradient_method(cv2.bitwise_not(frame.gray), self.method_combo.currentText(), self.get_current_gradient_params())
        self.current_gradient_mask = gradient
        _, binary = cv2.threshold(gradient, self.threshold_slider.value(), 255, cv2.THRESH_BINARY)
        return binary, {'gradient_view': gradient, 'binary_view': binary}

    def _finalize_mask(self, processed, frame):
        min_area = self.min_area_spinbox.value() / 100.0 * frame.gray.size
        if min_area > 0:
            contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    cv2.drawContours(processed, [contour], -1, 0, thickness=cv2.FILLED)
        if not self.fill_contours_checkbox.isChecked():
            return processed
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(processed)
        for contour in contours:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
        return filled

    def display_current_image(self):
        segmentation_presentation.display(self)

    # ---------- Работа со списком объектов ----------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    # ---------- Навигация, загрузка, UI события ----------
    def update_navigation_state(self):
        segmentation_navigation.update_state(self)

    def prev_image(self):
        segmentation_navigation.step(self, -1)

    def next_image(self):
        segmentation_navigation.step(self, 1)

    def goto_image(self, page_num):
        segmentation_navigation.navigate(self, page_num-1, only_if_changed=True)

    def load_images_from_dialog(self):
        segmentation_input.choose_images(self, self._load_images)

    def load_folder(self):
        segmentation_input.choose_folder(self, self._load_images)

    def reload_current_images(self):
        segmentation_input.reload_images(self, self._load_images)

    # ------- Ленивая загрузка (единая точка входа) -------
    def _load_images(self, source):
        segmentation_input.load_cached(self, source, self.segmentation_view_names, max_side=1024, reset=None)

    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    def on_resize_mode_changed(self, enabled):
        segmentation_controller.resize_changed(self, enabled)

    def clear_images(self):
        segmentation_input.clear_session(self)

    def schedule_update(self):
        segmentation_controller.schedule(self)

    def on_method_changed(self, idx):
        method = self.method_combo.currentText()
        self.sobel_container.setVisible(False)
        self.laplacian_container.setVisible(False)
        self.canny_container.setVisible(False)
        self.prewitt_container.setVisible(False)
        self.roberts_container.setVisible(False)
        self.kirsch_container.setVisible(False)

        self.sobel_kernel.setVisible(False)

        if method == "Sobel" or method == "Scharr":
            self.sobel_container.setVisible(True)
            if method == "Sobel":
                self.sobel_kernel.setVisible(True)
        elif method == "Laplacian":
            self.laplacian_container.setVisible(True)
        elif method == "Canny":
            self.canny_container.setVisible(True)
        elif method == "Prewitt":
            self.prewitt_container.setVisible(True)
        elif method == "Roberts":
            self.roberts_container.setVisible(True)
        elif method == "Kirsch":
            self.kirsch_container.setVisible(True)

        if self.original_image is not None:
            self.schedule_update()

    def on_canny_thresh_changed(self, value):
        sender = self.sender()
        if sender == self.canny_thresh1:
            self.canny_label1.setText(str(value))
        else:
            self.canny_label2.setText(str(value))
        self.schedule_update()

    def on_draw_mode_changed(self, idx):
        segmentation_presentation.geometry_changed(self)

    def on_hull_changed(self, state):
        segmentation_presentation.geometry_changed(self)

    # ---------- Отрисовка объектов (унифицированная) ----------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentData(), self.hull_checkbox.isChecked())

    # ---------- Сохранение аннотаций ----------
    def save_current_annotations(self):
        segmentation_export.save_selected(self)
    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GradientMethodsWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))