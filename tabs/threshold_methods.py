# -*- coding: utf-8 -*-
"""Пороговые методы: вычисления OpenCV, просмотр Qt через utils_new.

Поддерживается legacy API ThresholdWindow для автоподбора и пресетов.
Графические окна read-only; векторная разметка рисуется поверх базового QPixmap.
"""

# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import sys

import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
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
from ui.segmentation.preset_bindings import ThresholdPresetManager
from ui_tabs.threshold_methods_ui import setup_threshold_ui
from ui.autoreseach_ui.controls import SearchController
from utils_new.autoreseach_utils.search_space import default_settings, combination_count, search_kind
from utils_new.json_store import read_json_object, write_json_atomic
from ui.autoreseach_ui.settings_dialog import AutoSettingsDialog
from utils_new.segmentation.postprocessing import extract_objects
from utils_new.threshold_processing import apply_threshold_method


class ThresholdWindow(QMainWindow):
    segmentation_view_names = ('original_view', 'binary_view', 'morph_view', 'annotated_view')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Threshold Methods")

        # UI создаётся функцией, которая сама устанавливает центральный виджет
        setup_threshold_ui(self)

        # Данные изображений.
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Текущее изображение и его grayscale-версия для всей логики вкладки.
        self.original_image = None
        self.current_gray = None

        self.current_threshold = None
        self.current_objects_full = []   # список объектов в унифицированном формате
        self.current_selected_indices = []
        self.current_base_image = None

        # Автонастройки
        _root = get_project_root()
        auto_settings_dir = os.path.join(_root, "settings", "auto_settings")
        os.makedirs(auto_settings_dir, exist_ok=True)
        self.settings_file = os.path.join(auto_settings_dir, "auto_settings.json")
        self.auto_settings = default_settings()
        self.load_auto_settings()

        self.auto_search = SearchController(self)
        self.configure_auto_button.clicked.connect(self.open_auto_settings)
        self.auto_button.clicked.connect(self.run_auto_optimization)


        # Пресеты
        self.preset_manager = ThresholdPresetManager(self)
        self.update_preset_combo()

        # Подключение сигналов

        self.threshold_slider.valueChanged.connect(self.on_simple_threshold_changed)

        self.adaptive_win.valueChanged.connect(self.on_param_changed)
        self.adaptive_c.valueChanged.connect(self.on_param_changed)
        self.niblack_win.valueChanged.connect(self.on_param_changed)
        self.niblack_k.valueChanged.connect(self.on_niblack_k_changed)
        self.sauvola_win.valueChanged.connect(self.on_param_changed)
        self.sauvola_k.valueChanged.connect(self.on_sauvola_k_changed)
        self.sauvola_r.valueChanged.connect(self.on_sauvola_r_changed)
        self.isodata_init.valueChanged.connect(self.on_param_changed)
        self.row_win.valueChanged.connect(self.on_param_changed)
        self.row_k.valueChanged.connect(self.on_row_k_changed)
        self.bg_excess.valueChanged.connect(self.on_bg_excess_changed)

        # Замена save_coordinates на универсальное сохранение

        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.on_method_changed(self.method_combo.currentIndex())
        self.update_navigation_state()


        # Применить пресет по умолчанию (если есть)
        preset_names = self.preset_manager.get_preset_names()
        if preset_names:
            default_preset = "default" if "default" in preset_names else preset_names[0]
            # Блокируем сигналы комбобокса, чтобы не вызывать повторное применение
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText(default_preset)
            self.preset_combo.blockSignals(False)
            self.preset_manager.apply_preset(default_preset)
            self.log(f"Применён пресет по умолчанию: {default_preset}")

    def log(self, message):
        segmentation_actions.log(self, message)

    # ---------- Загрузка/сохранение настроек автоподбора ----------
    def load_auto_settings(self):
        try:
            value = read_json_object(self.settings_file)
            if search_kind(value)!="threshold":raise ValueError("Нужны настройки порогового поиска.")
            combination_count(value)
            self.auto_settings = value
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.log(f"Настройки автоподбора отклонены: {exc}")

    def save_auto_settings(self):
        try:
            combination_count(self.auto_settings)
            write_json_atomic(self.settings_file, self.auto_settings)
        except Exception as exc:
            self.log(f"Не удалось сохранить настройки автоподбора: {exc}")

    def closeEvent(self, event):
        self.auto_search.close()
        segmentation_controller.close(self)
        self.save_auto_settings()
        event.accept()

    def open_auto_settings(self):
        dlg = AutoSettingsDialog(self, self.auto_settings)
        if dlg.exec_():
            self.auto_settings = dlg.get_settings()
            self.save_auto_settings()
            self.log("Настройки автоподбора обновлены и сохранены")
        else:
            self.log("Настройки автоподбора не изменены")

    def run_auto_optimization(self):
        self.auto_search.start()

    # ---------- Гистограмма ----------
    def update_histogram(self, gray_img, threshold=None):
        segmentation_histogram.update_histogram(self, gray_img, threshold)

    def update_current_histogram(self):
        segmentation_histogram.update_current(self)

    # ---------- Сбор параметров для бинаризации ----------
    def get_current_threshold_params(self):
        method = self.method_combo.currentText()
        params = {}
        if method == "Simple Threshold":
            params["threshold"] = self.threshold_slider.value()
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            params["window"] = self.adaptive_win.value()
            params["c"] = self.adaptive_c.value()
        elif method == "Niblack":
            params["window"] = self.niblack_win.value()
            params["k"] = self.niblack_k.value() / 100.0
        elif method == "Sauvola":
            params["window"] = self.sauvola_win.value()
            params["k"] = self.sauvola_k.value() / 100.0
            params["r"] = self.sauvola_r.value()
        elif method == "ISODATA":
            params["init"] = self.isodata_init.value()
        elif method == "Background Symmetry":
            params["excess"] = self.bg_excess.value() / 100.0
        elif method == "Row Adaptive":
            params["window"] = self.row_win.value()
            params["k"] = self.row_k.value() / 100.0
        return params

    # ---------- Отображение через Qt (utils_new) ----------

    # ---------- Отображение текущего изображения ----------

    def _mask_signature(self, frame):
        return frame.key, self.method_combo.currentText(), repr(self.get_current_threshold_params())

    def _compute_mask(self, frame):
        binary, self.current_threshold = apply_threshold_method(frame.gray, self.method_combo.currentText(), self.get_current_threshold_params())
        binary = cv2.bitwise_not(binary)
        return binary, {'binary_view': binary}

    def display_current_image(self):
        segmentation_presentation.display(self)

    # ---------- Работа со списком объектов (универсальная) ----------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    # ---------- Навигация ----------
    def update_navigation_state(self):
        segmentation_navigation.update_state(self)

    def prev_image(self):
        segmentation_navigation.step(self, -1)

    def next_image(self):
        segmentation_navigation.step(self, 1)

    def goto_image(self, page_num):
        segmentation_navigation.navigate(self, page_num-1)

    # ---------- Загрузка изображений ----------
    def load_images_from_dialog(self):
        segmentation_input.choose_images(self, self._load_images)

    def load_folder(self):
        segmentation_input.choose_folder(self, self._load_images)

    def reload_current_images(self):
        segmentation_input.reload_images(self, self._load_images)

    # ------- Ленивая загрузка (единая точка входа) -------
    def _load_images(self, source):
        segmentation_input.load_cached(self, source, self.segmentation_view_names, max_side=1024, reset=lambda: setattr(self, 'current_threshold', None))

    # ---------- Сохранение аннотаций (универсальное) ----------
    # ---------- Сохранение аннотаций (только выбранные объекты) ----------
    # ---------- Сохранение аннотаций (включая пустой файл) ----------
    def save_current_annotations(self):
        segmentation_export.save_selected(self)
    # ---------- События UI ----------
    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    def on_resize_mode_changed(self, enabled):
        segmentation_controller.resize_changed(self, enabled)

    def clear_images(self):
        segmentation_input.clear_session(self, none_fields=('current_threshold',))

    def reset_params_for_method(self, method):
        if method == "Simple Threshold":
            self.threshold_slider.setValue(127)
            self.threshold_value_label.setText("127")
        elif method in ("Adaptive Mean", "Adaptive Gauss"):
            self.adaptive_win.setValue(25)
            self.adaptive_c.setValue(3)
        elif method == "Niblack":
            self.niblack_win.setValue(25)
            self.niblack_k.setValue(20)
            self.niblack_k_label.setText("0.20")
        elif method == "Sauvola":
            self.sauvola_win.setValue(25)
            self.sauvola_k.setValue(20)
            self.sauvola_r.setValue(128)
            self.sauvola_k_label.setText("0.20")
            self.sauvola_r_label.setText("128")
        elif method == "ISODATA":
            self.isodata_init.setValue(128)
        elif method == "Background Symmetry":
            self.bg_excess.setValue(20)
            self.bg_excess_label.setText("0.20")
        elif method == "Row Adaptive":
            self.row_win.setValue(50)
            self.row_k.setValue(50)
            self.row_k_label.setText("0.50")

    def on_method_changed(self, idx):
        method = self.method_combo.currentText()
        self.simple_threshold_container.setVisible(False)
        self.adaptive_container.setVisible(False)
        self.niblack_container.setVisible(False)
        self.sauvola_container.setVisible(False)
        self.isodata_container.setVisible(False)
        self.background_container.setVisible(False)
        self.row_adaptive_container.setVisible(False)

        if method == "Simple Threshold":
            self.simple_threshold_container.setVisible(True)
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            self.adaptive_container.setVisible(True)
        elif method == "Niblack":
            self.niblack_container.setVisible(True)
        elif method == "Sauvola":
            self.sauvola_container.setVisible(True)
        elif method == "ISODATA":
            self.isodata_container.setVisible(True)
        elif method == "Background Symmetry":
            self.background_container.setVisible(True)
        elif method == "Row Adaptive":
            self.row_adaptive_container.setVisible(True)

        self.reset_params_for_method(method)
        if self.original_image is not None:
            self.schedule_update()

    def schedule_update(self):
        segmentation_controller.schedule(self)



    def on_morphology_changed(self):
        # При ручной настройке отменяем отложенную конвертацию старого пресета.
        self.on_param_changed()

    def on_param_changed(self):
        if self.original_image is not None:
            self.schedule_update()

    def on_simple_threshold_changed(self, value):
        self.threshold_value_label.setText(str(value))
        if self.original_image is not None and self.method_combo.currentText() == "Simple Threshold":
            self.schedule_update()

    def on_niblack_k_changed(self, value):
        self.niblack_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_sauvola_k_changed(self, value):
        self.sauvola_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_sauvola_r_changed(self, value):
        self.sauvola_r_label.setText(str(value))
        self.schedule_update()

    def on_row_k_changed(self, value):
        self.row_k_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_bg_excess_changed(self, value):
        self.bg_excess_label.setText(f"{value/100:.2f}")
        self.schedule_update()

    def on_invert_changed(self, state):
        if self.original_image is not None:
            self.schedule_update()

    def on_draw_mode_changed(self, idx):
        segmentation_presentation.geometry_changed(self)

    def on_hull_changed(self, state):
        segmentation_presentation.geometry_changed(self)

    # ---------- Извлечение аннотаций из бинарной маски ----------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentData(), self.hull_checkbox.isChecked())

    # ---------- Пресеты ----------
    def update_preset_combo(self):
        segmentation_actions.update_preset_combo(self)

    def on_preset_activated(self, index):
        segmentation_actions.apply_preset(self, index)

    def save_as_preset(self):
        segmentation_actions.save_as_preset(self)

    def delete_current_preset(self):
        segmentation_actions.delete_current_preset(self)

    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)

    def on_global_settings_changed(self, new_settings=None):
        segmentation_results.refresh_styles(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))