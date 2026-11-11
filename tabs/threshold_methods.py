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

import json
import os
import sys

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
)

from path_setup import get_project_root
from ui.image_conversion import numpy_to_qpixmap
from ui.segmentation import actions as segmentation_actions
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import PresetManager
from ui.settings_service import settings
from ui_tabs.threshold_methods_ui import setup_threshold_ui
from utils_auto_research.threshold_auto_research import AutoResearch
from utils_auto_research.threshold_auto_settings_dialog import AutoSettingsDialog
from utils_new.morphology import apply_morphology as apply_shared_morphology
from utils_new.segmentation.postprocessing import extract_objects
from utils_new.threshold_processing import apply_threshold_method, normalize_method_name


class _LegacyMorphologyRatioSlider(QSlider):
    """Невидимый Qt-совместимый адаптер для старого AutoResearch.

    Перенаправляет setValue в новый виджет и при блокировке сигналов,
    поскольку некоторые исследовательские сценарии выставляют сразу весь
    набор параметров с blockSignals(True).
    """
    def __init__(self, owner, operation):
        super().__init__(Qt.Horizontal, owner)
        self.owner = owner
        self.operation = operation
        self._sync_only = False
        self.setRange(0, 10)
        self.hide()

    def setValue(self, value):
        super().setValue(value)
        if not self._sync_only:
            self.owner._on_legacy_ratio_change(self.operation, self.value())


class ThresholdMorphologyPresetManager(PresetManager):
    """Сохраняет новую пиксельную морфологию и читает старые Threshold-пресеты.

    Старые close_factor/open_factor зависят от размера снимка: их перевод
    откладывается до появления изображения, чтобы не терять размер ядра.
    Общий utils.presets.PresetManager намеренно не изменяется — он нужен
    другим вкладкам с историческими полями.
    """

    def get_current_settings(self):
        m = self.main
        state = {
            "method": normalize_method_name(m.method_combo.currentText()),
            "params": m.get_current_threshold_params(),
            "invert": m.invert_checkbox.isChecked(),
            "draw_mode": m.draw_combo.currentText(),
            "use_hull": m.hull_checkbox.isChecked(),
        }
        if m._pending_legacy_morphology is not None:
            # Ещё не открыт снимок для перевода относительного ядра в px;
            # сохраняем параметры без потери (старый формат временно допустим).
            legacy = m._pending_legacy_morphology
            state.update(close_factor=legacy["close_factor"],
                         open_factor=legacy["open_factor"],
                         kernel_shape=legacy["shape"])
        else:
            state["morphology"] = m.morphology_widget.get_settings()
        return state

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        m = self.main
        preset = self.presets[name]
        method = normalize_method_name(str(preset.get("method", "Simple Threshold")))
        # Без автоматического пересчёта по промежуточным значениям ползунков.
        widgets = [m.method_combo, m.threshold_slider, m.adaptive_win, m.adaptive_c,
                   m.niblack_win, m.niblack_k, m.sauvola_win, m.sauvola_k,
                   m.sauvola_r, m.isodata_init, m.bg_excess, m.row_win,
                   m.row_k, m.draw_combo, m.hull_checkbox, m.invert_checkbox,
                   m.morphology_widget]
        previous = [w.blockSignals(True) for w in widgets]
        try:
            idx = m.method_combo.findText(method)
            if idx >= 0:
                m.method_combo.setCurrentIndex(idx)
                m.on_method_changed(idx)
            else:
                m.log(f"Предупреждение: метод '{method}' отсутствует.")

            params = preset.get("params", {})
            if method == "Simple Threshold":
                m.threshold_slider.setValue(int(params.get("threshold", 127)))
                m.threshold_value_label.setText(str(m.threshold_slider.value()))
            elif method in ("Adaptive Mean", "Adaptive Gauss"):
                window = int(params.get("window", 25))
                m.adaptive_win.setValue(window + (window % 2 == 0))
                m.adaptive_c.setValue(int(params.get("c", 3)))
            elif method == "Niblack":
                m.niblack_win.setValue(int(params.get("window", 25)))
                m.niblack_k.setValue(round(float(params.get("k", 0.2)) * 100))
                m.niblack_k_label.setText(f"{m.niblack_k.value() / 100:.2f}")
            elif method == "Sauvola":
                m.sauvola_win.setValue(int(params.get("window", 25)))
                m.sauvola_k.setValue(round(float(params.get("k", 0.2)) * 100))
                m.sauvola_r.setValue(int(params.get("r", 128)))
                m.sauvola_k_label.setText(f"{m.sauvola_k.value() / 100:.2f}")
                m.sauvola_r_label.setText(str(m.sauvola_r.value()))
            elif method == "ISODATA":
                m.isodata_init.setValue(int(params.get("init", 128)))
            elif method == "Background Symmetry":
                m.bg_excess.setValue(round(float(params.get("excess", 0.2)) * 100))
                m.bg_excess_label.setText(f"{m.bg_excess.value() / 100:.2f}")
            elif method == "Row Adaptive":
                m.row_win.setValue(int(params.get("window", 50)))
                m.row_k.setValue(round(float(params.get("k", 0.5)) * 100))
                m.row_k_label.setText(f"{m.row_k.value() / 100:.2f}")

            m.invert_checkbox.setChecked(bool(preset.get("invert", True)))
            m.draw_combo.setCurrentText(str(preset.get("draw_mode", m.draw_combo.currentText())))
            m.hull_checkbox.setChecked(bool(preset.get("use_hull", False)))
            if "morphology" in preset:
                m._pending_legacy_morphology = None
                m.morphology_widget.set_settings(preset["morphology"])
            elif "close_factor" in preset or "open_factor" in preset:
                legacy = {
                    "close_factor": float(preset.get("close_factor", 0)),
                    "open_factor": float(preset.get("open_factor", 0)),
                    "shape": preset.get("kernel_shape", "Rectangle"),
                }
                if m.current_gray is None:
                    m._pending_legacy_morphology = legacy
                else:
                    m._pending_legacy_morphology = None
                    m.morphology_widget.set_legacy_threshold(
                        legacy["close_factor"], legacy["open_factor"],
                        m.current_gray.shape, legacy["shape"])
        finally:
            for widget, prev in zip(widgets, previous):
                widget.blockSignals(prev)
        if m._lazy_cache is not None:
            m.display_current_image()
        return True


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
        self.auto_settings = None
        self.load_auto_settings()

        self.auto_enable_checkbox.setChecked(False)
        self.configure_auto_button.clicked.connect(self.open_auto_settings)
        self.auto_button.clicked.connect(self.run_auto_optimization)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Пресеты
        self._pending_legacy_morphology = None
        # Служебные (невидимые) адаптеры старого API для внешнего AutoResearch.
        # Новый интерфейс содержит только MorphologyWidget. Старые алгоритмы
        # автоподбора иногда обращаются к close/open_kernel_slider напрямую.
        self.close_kernel_slider = _LegacyMorphologyRatioSlider(self, "close")
        self.open_kernel_slider = _LegacyMorphologyRatioSlider(self, "open")
        self.kernel_shape_combo = self.morphology_widget.shape_combo
        self.close_kernel_label = QLabel(self)
        self.open_kernel_label = QLabel(self)
        self.close_kernel_label.hide()
        self.open_kernel_label.hide()
        self.preset_manager = ThresholdMorphologyPresetManager(self)
        self.update_preset_combo()
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)

        # Подключение сигналов
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)
        self.nav_widget.goto_page.connect(self.goto_image)

        self.threshold_slider.valueChanged.connect(self.on_simple_threshold_changed)
        self.invert_checkbox.stateChanged.connect(self.on_invert_changed)
        self.morphology_widget.parameters_changed.connect(self.on_morphology_changed)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

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

        self.hull_checkbox.stateChanged.connect(self.on_hull_changed)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)
        # Замена save_coordinates на универсальное сохранение
        self.save_button.clicked.connect(self.save_current_annotations)

        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        self.on_method_changed(self.method_combo.currentIndex())
        self.update_navigation_state()

        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)
        settings.settings_changed.connect(self.on_global_settings_changed)

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
        if not os.path.exists(self.settings_file):
            self.log("Файл настроек автоподбора не найден, будут использованы настройки по умолчанию.")
            return
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                self.auto_settings = json.load(f)
            self.log(f"Настройки автоподбора загружены из {self.settings_file}")
        except Exception as e:
            self.log(f"Ошибка загрузки настроек автоподбора: {e}")

    def save_auto_settings(self):
        if self.auto_settings is None:
            return
        try:
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
        dlg = AutoSettingsDialog(self, self.auto_settings)
        if dlg.exec_():
            self.auto_settings = dlg.get_settings()
            self.save_auto_settings()
            self.log("Настройки автоподбора обновлены и сохранены")
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
        self.log("Запуск автоподбора параметров...")
        search_all = self.search_all_checkbox.isChecked()
        research = AutoResearch(self, search_all)
        research.run()

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
    @staticmethod
    def _show_image_in_view(view, image):
        """Чистый снимок + корректная система координат пикселей изображения.

        Если numpy_to_qpixmap уменьшит изображение для ограничения памяти,
        аннотации останутся привязанными к исходным координатам пикселей.
        """
        if image is None:
            view.set_annotations([], 0, 0)
            view.set_pixmap(numpy_to_qpixmap(None))
            return
        h, w = image.shape[:2]
        view.set_annotations([], w, h)
        view.set_pixmap(numpy_to_qpixmap(image))

    # ---------- Отображение текущего изображения ----------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.update_current_histogram()
            for view in (self.original_view, self.binary_view,
                         self.morph_view, self.annotated_view):
                self._show_image_in_view(view, None)
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        self.info_label.setText(f"Image {idx + 1} of {total}")

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
        # До первой загрузки изображения невозможно перевести относительные
        # коэффициенты старого пресета в пиксели. Делаем это один раз здесь.
        if self._pending_legacy_morphology is not None:
            legacy = self._pending_legacy_morphology
            self._pending_legacy_morphology = None
            blocked = self.morphology_widget.blockSignals(True)
            try:
                self.morphology_widget.set_legacy_threshold(
                    legacy["close_factor"], legacy["open_factor"],
                    gray.shape, legacy["shape"])
            finally:
                self.morphology_widget.blockSignals(blocked)
        self._sync_legacy_morph_sliders()
        method = self.method_combo.currentText()
        params = self.get_current_threshold_params()
        binary, thresh = apply_threshold_method(gray, method, params)
        self.current_threshold = thresh
        binary = cv2.bitwise_not(binary)

        current_file = os.path.basename(self.image_paths[idx])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"thresh_original_{base_name}")
        self.binary_view.set_suggested_save_name(f"thresh_binary_{base_name}")
        self.morph_view.set_suggested_save_name(f"thresh_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"thresh_annotated_{base_name}")

        self.update_histogram(gray, self.current_threshold)
        self._show_image_in_view(self.original_view, original)
        self._show_image_in_view(self.binary_view, binary)

        processed = apply_shared_morphology(
            binary, self.morphology_widget.get_steps(),
            shape=self.morphology_widget.kernel_shape(),
        )
        if self.invert_checkbox.isChecked():
            processed = cv2.bitwise_not(processed)

        self._show_image_in_view(self.morph_view, processed)

        if len(original.shape) == 2 or (len(original.shape) == 3 and original.shape[2] == 1):
            result = original.copy()
            result[processed == 0] = 255
        else:
            result = original.copy()
            mask = processed == 0
            result[mask] = [255, 255, 255]

        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()
        self.current_base_image = display_img.copy()
        # На 4-м виде базовый снимок устанавливаем один раз.
        # Далее галочки объектов обновляют только векторный слой, без
        # повторного преобразования изображения в QPixmap.
        self._show_image_in_view(self.annotated_view, self.current_base_image)
        # Геометрию объектов теперь вычисляет общий utils_new.annotation_processing.
        _, self.current_objects_full = self.draw_objects_on_image(
            display_img, processed, draw=False)
        self.update_object_list()

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    # ---------- Работа со списком объектов (универсальная) ----------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    def update_coordinates_display(self):
        segmentation_results.update_coordinates(self)

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
        self.update_timer.start(50)

    def _sync_legacy_morph_sliders(self):
        """Отражаем px в старых служебных регуляторах без обратной перезаписи."""
        if self.current_gray is None:
            return
        smallest = max(1, min(self.current_gray.shape[:2]))
        for op, slider, label in (
            ("close", self.close_kernel_slider, self.close_kernel_label),
            ("open", self.open_kernel_slider, self.open_kernel_label),
        ):
            px = self.morphology_widget.operation_size(op)
            ratio = min(10, max(0, round(px * 100 / smallest))) if px > 1 else 0
            prev = slider.blockSignals(True)
            slider._sync_only = True
            try:
                slider.setValue(ratio)
            finally:
                slider._sync_only = False
                slider.blockSignals(prev)
            label.setText(f"{ratio / 100:.2f}")

    def _on_legacy_ratio_change(self, operation, value):
        """Сохраняет внешние вызовы AutoResearch в старом масштабе 0..10%."""
        if self.current_gray is None:
            legacy = self._pending_legacy_morphology or {
                "close_factor": 0.0, "open_factor": 0.0,
                "shape": self.morphology_widget.kernel_shape(),
            }
            legacy[f"{operation}_factor"] = value / 100.0
            self._pending_legacy_morphology = legacy
            return
        size = max(1, min(self.current_gray.shape[:2]))
        pixels = max(1, int((value / 100.0) * size)) if value > 0 else 0
        self.morphology_widget.set_operation_size(operation, pixels)

    def on_morphology_changed(self):
        # При ручной настройке отменяем отложенную конвертацию старого пресета.
        self._pending_legacy_morphology = None
        self._sync_legacy_morph_sliders()
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

    # ---------- Извлечение аннотаций из бинарной маски ----------
    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentText(), self.hull_checkbox.isChecked())

    # ---------- Пресеты ----------
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

    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)

    def on_global_settings_changed(self, new_settings=None):
        # SmartGraphicsViewNew сам подписан на settings_changed. Пиксели
        # повторно создавать не требуется, только обновить Qt overlay.
        if self.original_image is not None:
            self.annotated_view.refresh_annotations()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdWindow()
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))