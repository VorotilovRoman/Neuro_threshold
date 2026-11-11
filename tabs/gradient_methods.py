
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
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
)

from path_setup import get_project_root
from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import GradientPresetManager
from ui.settings_service import settings
from ui_tabs.gradient_methods_ui import setup_gradient_ui
from utils_auto_research.gradient_auto_research import GradientAutoResearch
from utils_auto_research.gradient_auto_settings_dialog import GradientAutoSettingsDialog
from utils_new.segmentation.postprocessing import (
    apply_relative_morphology as apply_morphology,
)
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
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)
        # Пресеты
        self.preset_manager = GradientPresetManager(self, preset_file="presets_gradient.json")
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Сигналы
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

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
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.close_kernel_slider.valueChanged.connect(self.on_close_kernel_changed)
        self.open_kernel_slider.valueChanged.connect(self.on_open_kernel_changed)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)
        self.kernel_shape_combo.currentIndexChanged.connect(self.schedule_update)

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)
        # Замена сохранения на универсальное
        self.save_button.clicked.connect(self.save_current_annotations)

        self.auto_enable_checkbox.setChecked(False)
        self.configure_auto_button.clicked.connect(self.open_auto_settings)
        self.auto_button.clicked.connect(self.run_auto_optimization)

        settings.settings_changed.connect(self.on_global_settings_changed)

        self.on_method_changed(self.method_combo.currentIndex())
        self.update_navigation_state()

    # ---------- Логирование, пресеты, настройки ----------
    def log(self, message):
        segmentation_actions.log(self, message)

    def on_global_settings_changed(self, new_settings=None):
        self.update_annotated_view()

    def update_preset_combo(self):
        segmentation_actions.update_preset_combo(self)

    def on_preset_activated(self, index):
        name = self.preset_combo.itemText(index)
        if name:
            self.preset_manager.apply_preset(name)
            self.log(f"Применён пресет: {name}")

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
        # Аккуратно останавливаем фоновый пул, чтобы процесс не висел
        # на незавершённых задачах загрузки.
        if self._lazy_cache is not None:
            try:
                self._lazy_cache.shutdown()
            except Exception:
                pass
            self._lazy_cache = None
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
        if not self.auto_enable_checkbox.isChecked():
            self.log("Автоподбор не включён. Установите галочку Enable auto research.")
            QMessageBox.warning(self, "Автоподбор", "Включите автоподбор галочкой Enable auto research.")
            return
        if not self.auto_settings:
            self.log("Нет настроек автоподбора. Нажмите Configure auto research для настройки.")
            QMessageBox.warning(self, "Автоподбор", "Сначала настройте параметры автоподбора (Configure auto research).")
            return
        self.log("Запуск автоподбора параметров для градиентных методов...")
        search_all = self.search_all_checkbox.isChecked()
        research = GradientAutoResearch(self, search_all)
        research.run()

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

    def on_close_kernel_changed(self, value):
        self.close_kernel_label.setText(f"{value/200.0:.3f}")
        self.schedule_update()

    def on_open_kernel_changed(self, value):
        self.open_kernel_label.setText(f"{value/200.0:.3f}")
        self.schedule_update()

    # ---------- Основная логика ----------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.update_current_histogram()
            for v in (self.original_view, self.gradient_view, self.binary_view,
                      self.morph_view, self.filled_view, self.annotated_view):
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

        self.original_view.set_suggested_save_name(f"grad_original_{base_name}")
        self.gradient_view.set_suggested_save_name(f"grad_edges_{base_name}")
        self.binary_view.set_suggested_save_name(f"grad_binary_{base_name}")
        self.morph_view.set_suggested_save_name(f"grad_morph_{base_name}")
        self.filled_view.set_suggested_save_name(f"grad_filled_{base_name}")
        self.annotated_view.set_suggested_save_name(f"grad_annotated_{base_name}")

        # --- Синхронно получаем элемент (LRU-кэш вернёт готовый или загрузит) ---
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

        self.update_histogram(gray)
        # 1. Градиентная маска
        method = self.method_combo.currentText()
        params = self.get_current_gradient_params()

        gray = cv2.bitwise_not(gray)
        gradient = apply_gradient_method(gray, method, params)
        self.current_gradient_mask = gradient
        self.original_view.set_pixmap(numpy_to_qpixmap(original))
        self.gradient_view.set_pixmap(numpy_to_qpixmap(gradient))

        # 2. Бинаризация
        thresh_val = self.threshold_slider.value()
        _, binary = cv2.threshold(gradient, thresh_val, 255, cv2.THRESH_BINARY)
        self.binary_view.set_pixmap(numpy_to_qpixmap(binary))

        # 3. Морфология (closing + opening)
        close_factor = self.close_kernel_slider.value() / 200.0
        open_factor = self.open_kernel_slider.value() / 200.0
        kernel_shape = self.kernel_shape_combo.currentText()
        processed = apply_morphology(binary, close_factor, open_factor, kernel_shape, gray.shape)
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        # 4. Удаление мелких контуров (по проценту от площади изображения)
        min_area_percent = self.min_area_spinbox.value()
        fill_enabled = self.fill_contours_checkbox.isChecked()

        if min_area_percent > 0:
            total_area = gray.shape[0] * gray.shape[1]
            min_area_abs = (min_area_percent / 100.0) * total_area
            contours, hierarchy = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area_abs:
                    cv2.drawContours(processed, [cnt], -1, 0, thickness=cv2.FILLED)

        # 5. Заливка контуров
        if fill_enabled:
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_for_objects = np.zeros_like(processed)
            for cnt in contours:
                cv2.drawContours(mask_for_objects, [cnt], -1, 255, thickness=cv2.FILLED)
            close_factor = self.close_kernel_slider.value() / 200.0
            if close_factor > 0:
                kernel_size = max(3, int(min(gray.shape) * close_factor))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                mask_for_objects = cv2.morphologyEx(mask_for_objects, cv2.MORPH_CLOSE, kernel)
        else:
            mask_for_objects = processed

        # 6. Инверсия маски
        if self.invert_checkbox.isChecked():
            mask_for_objects = cv2.bitwise_not(mask_for_objects)

        self.filled_view.set_pixmap(numpy_to_qpixmap(mask_for_objects))

        # 7. Создание изображения для аннотаций (фон белый)
        if len(original.shape) == 2:
            result = original.copy()
            result[mask_for_objects == 0] = 255
        else:
            result = original.copy()
            result[mask_for_objects == 0] = [255, 255, 255]

        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()

        self.current_base_image = display_img.copy()

        # 8. Выделение объектов в унифицированном формате через draw_objects_on_image
        _, self.current_objects_full = self.draw_objects_on_image(display_img, mask_for_objects, draw=False)

        self.update_object_list()
        self.nav_widget.set_current_index(idx, total)

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ---------- Работа со списком объектов ----------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    def update_coordinates_display(self):
        segmentation_results.update_coordinates(self)

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
        segmentation_input.clear_session(self)

    def schedule_update(self):
        self.update_timer.start(50)

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
        mode = self.draw_combo.currentText()
        if mode in ["Segmentation (Polygon)", "OBB (Oriented Box)"]:
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

    # ---------- Отрисовка объектов (унифицированная) ----------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentText(), self.hull_checkbox.isChecked())

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