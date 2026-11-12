
# Allow both `python -m tabs.<name>` and direct execution from any directory.
if __name__ == "__main__" and not __package__:
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath

    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))

# deep_learning_methods.py
from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import os
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
)

from ui.segmentation import actions as segmentation_actions
from ui.segmentation import computation as segmentation_computation
from ui.segmentation import controller as segmentation_controller
from ui.segmentation import export as segmentation_export
from ui.segmentation import histogram as segmentation_histogram
from ui.segmentation import image_input as segmentation_input
from ui.segmentation import navigation as segmentation_navigation
from ui.segmentation import presentation as segmentation_presentation
from ui.segmentation import results as segmentation_results
from ui.segmentation.preset_bindings import DeepLearningPresetManager
from ui_tabs.deep_learning_methods_ui import setup_deep_learning_methods_ui
from utils_new.segmentation.postprocessing import extract_objects

# Дополнительные Qt-виджеты (на случай, если не подтянуты из import_libs_internal)
try:
    from PyQt5.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QSpinBox,
    )
except ImportError:
    pass


class DeepLearningWindow(QMainWindow):
    segmentation_view_names = ('original_view', 'dl_view', 'morph_view', 'annotated_view')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Learning Segmentation")

        # Данные изображений.
        # Полный список путей хранится всегда; сами картинки — в _lazy_cache.
        self.image_paths = []
        self.current_index = 0
        self._lazy_cache = None

        # Текущее изображение и его grayscale-версия для всей логики вкладки.
        self.original_image = None
        self.current_gray = None

        self.segmentor = None
        self.model_metadata = None
        self.prediction_mask = None
        self.current_objects_full = []
        self.current_selected_indices = []
        self.current_base_image = None
        self.ignore_ui_changes = False

        setup_deep_learning_methods_ui(self)



        # Устройство по умолчанию
        auto_device = "Auto (best available)"
        idx = self.device_combo.findText(auto_device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)
        self.on_device_changed(idx)

        # Пресеты
        self.preset_manager = DeepLearningPresetManager(self, preset_file="presets_deep_learning.json")
        self.update_preset_combo()


        # Навигация

        # Кнопки
        self.segment_button.clicked.connect(self.run_segmentation)

        # Подключение кнопок "Обзор" в виджетах
        self.unet_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("U-Net"))
        self.deeplab_settings._browse_btn.clicked.connect(self._browse_deeplab_model)
        self.segformer_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("SegFormer"))
        self.sam_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("SAM"))
        self.yolo_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("YOLO-seg"))
        self.custom_settings._browse_btn.clicked.connect(lambda: self.load_model_from_widget("Custom ONNX"))

        # UI изменения
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        self._connect_settings_widgets()

        # Морфология



        self.update_navigation_state()
        self.on_model_changed(0)


    def _update_segment_button_color(self):
        """Обновляет цвет кнопки сегментации в зависимости от наличия модели."""
        if self.segmentor is not None:
            self.segment_button.setStyleSheet("background-color: green; color: white;")
        else:
            self.segment_button.setStyleSheet("background-color: #b22222; color: black;")

    def _browse_deeplab_model(self):
        """
        Загружает DeepLabV3/V3+ из пути, указанного в deeplab_settings.
        Реальная архитектура читается из *_meta.json сегментатором;
        имя в model_combo нужно только для выбора ветки в load_model_from_widget.
        """
        model_name = self.model_combo.currentText()
        if model_name not in ("DeepLabV3+", "DeepLabV3"):
            model_name = "DeepLabV3+"
        self.load_model_from_widget(model_name)

    # ---------- Загрузка модели из виджета ----------
    def load_model_from_widget(self, model_name):
        segmentation_computation.load_dl(self, model_name)

    def _update_unet_ui_from_metadata(self, metadata):
        """
        Заполняет поля U-Net из метаданных и логирует диагностику.
        Вся работа с override-полями — внутри UNetInferenceSettings.
        """
        # Архитектурные поля — как было
        if 'encoder' in metadata:
            encoder = metadata['encoder']
            idx = self.unet_settings._encoder.findText(encoder)
            if idx >= 0:
                self.unet_settings._encoder.setCurrentIndex(idx)
        if 'input_size' in metadata:
            input_size = str(metadata['input_size'])
            idx = self.unet_settings._input_size.findText(input_size)
            if idx >= 0:
                self.unet_settings._input_size.setCurrentIndex(idx)

        # Override-поля — через публичное API виджета
        diag = self.unet_settings.apply_metadata(metadata)

        # Лог

        arch = metadata.get('architecture', 'unet').upper()
        nc = metadata.get('num_classes', '?')
        norm = metadata.get('normalize', '?')
        tr_rm = diag['train_resize_mode'] or '?'
        inf_rm = diag['inference_resize_mode']

        if diag['train_resize_mode'] == 'random_crop':
            inf_rm_str = inf_rm if inf_rm in ('hard', 'keep_aspect_pad') \
                else 'keep_aspect_pad (default)'
        else:
            inf_rm_str = '— (не нужен)'

        self.log("=== U-Net метаданные ===")
        self.log(f"  Архитектура: {arch}")
        self.log(f"  num_classes: {nc}")
        self.log(f"  normalize: {norm}")
        self.log(f"  resize_mode (train): {tr_rm}")
        self.log(f"  inference_resize_mode: {inf_rm_str}")
        if nc == 2:
            self.log("  Режим: бинарная сегментация (фон + объект). "
                     "Threshold применяется.")
        elif isinstance(nc, int) and nc > 2:
            self.log(f"  Режим: {nc}-классовая сегментация. "
                     "Threshold не применяется (argmax).")
        if diag['train_resize_mode'] == 'random_crop':
            self.log("  ℹ️ Обучение было с random_crop. Выберите режим ресайза "
                     "для инференса (hard / keep_aspect_pad).")
        if diag['missing']:
            self.log(f"  ⚠️ Отсутствуют: {diag['missing']} → "
                     "override-поля активированы")
        self.log("=========================")



    def _update_deeplab_ui_from_metadata(self, metadata):
        """
        Заполняет поля DeepLabV3/V3+ из метаданных через публичное API
        DeepLabV3InferenceSettings.apply_metadata и логирует диагностику.
        """
        diag = self.deeplab_settings.apply_metadata(metadata)

        arch = diag.get('architecture') or '?'
        nc = metadata.get('num_classes', '?')
        norm = metadata.get('normalize', '?')
        os_val = metadata.get('output_stride', '?')
        act = metadata.get('output_activation', '?')
        tr_rm = diag['train_resize_mode'] or '?'
        inf_rm = diag['inference_resize_mode']

        if diag['train_resize_mode'] == 'random_crop':
            inf_rm_str = inf_rm if inf_rm in ('hard', 'keep_aspect_pad') \
                else 'hard (default)'
        else:
            inf_rm_str = '— (не нужен)'

        self.log("=== DeepLab метаданные ===")
        self.log(f"  Архитектура: {arch}")
        self.log(f"  Backbone: {metadata.get('backbone', '?')}")
        self.log(f"  Output stride: {os_val}")
        self.log(f"  Atrous rates: {metadata.get('atrous_rates', '?')}")
        self.log(f"  input_size: {metadata.get('input_size', '?')}")
        self.log(f"  num_classes: {nc}")
        self.log(f"  output_activation: {act}")
        self.log(f"  normalize: {norm}")
        self.log(f"  resize_mode (train): {tr_rm}")
        self.log(f"  inference_resize_mode: {inf_rm_str}")

        try:
            nc_int = int(nc)
        except (TypeError, ValueError):
            nc_int = None
        if nc_int == 1:
            self.log("  Режим: бинарная (sigmoid + threshold).")
        elif nc_int == 2:
            self.log("  Режим: «фон + объект» (softmax[:,1] или sigmoid + threshold).")
        elif nc_int is not None and nc_int > 2:
            self.log(f"  Режим: {nc_int}-классовая сегментация (argmax).")

        if diag['train_resize_mode'] == 'random_crop':
            self.log("  ℹ️ Обучение было с random_crop. Выберите режим ресайза "
                     "для инференса (hard / keep_aspect_pad).")
        if diag['missing']:
            self.log(f"  ⚠️ Отсутствуют: {diag['missing']} → "
                     "override-поля активированы")
        self.log("=========================")

    # ---------- Подключение виджетов для отслеживания изменений ----------
    def _connect_settings_widgets(self):
        widgets = [
            self.model_combo,
            self.device_combo,
            self.yolo_settings._conf,
            self.yolo_settings._iou,
            self.yolo_settings._imgsz,
            self.yolo_settings._save,
            self.custom_settings._model_path,
            self.unet_settings._threshold,
            self.deeplab_settings._threshold,
        ]
        # Override-поля U-Net теперь внутри виджета — подключаемся к сигналу
        try:
            self.unet_settings.override_changed.connect(self.on_any_setting_changed)
        except Exception:
            pass
        try:
            self.deeplab_settings.override_changed.connect(self.on_any_setting_changed)
        except Exception:
            pass
        for w in widgets:
            if w is None:
                continue
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self.on_any_setting_changed)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self.on_any_setting_changed)

    def on_any_setting_changed(self, *args):
        if not self.ignore_ui_changes:
            segmentation_controller.invalidate_prediction(self)
            self.schedule_update()

    # ---------- Логирование ----------
    def log(self, message):
        segmentation_actions.log(self, message)

    # ---------- Пресеты ----------
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

    # ---------- Выбор модели ----------
    def on_model_changed(self, idx):
        if not self.ignore_ui_changes and not getattr(self, '_loading_model', False):
            segmentation_controller.invalidate_prediction(self)
        model_name = self.model_combo.currentText()
        for cont in [self.unet_settings, self.deeplab_settings, self.segformer_settings,
                     self.sam_settings, self.yolo_settings, self.custom_settings]:
            cont.setVisible(False)

        if model_name == "U-Net":
            self.unet_settings.setVisible(True)
        elif model_name in ("DeepLabV3+", "DeepLabV3"):
            self.deeplab_settings.setVisible(True)
        elif model_name == "SegFormer":
            self.segformer_settings.setVisible(True)
        elif model_name == "SAM":
            self.sam_settings.setVisible(True)
        elif model_name == "YOLO-seg":
            self.yolo_settings.setVisible(True)
        elif model_name == "Custom ONNX":
            self.custom_settings.setVisible(True)

        self.segmentor = None
        self.model_metadata = None
        self.prediction_mask = None
        self._update_segment_button_color()

        if model_name == "U-Net":
            path = self.unet_settings._model_path.text().strip()
        elif model_name in ("DeepLabV3+", "DeepLabV3"):
            path = self.deeplab_settings._model_path.text().strip()
        elif model_name == "SegFormer":
            path = self.segformer_settings._model_path.text().strip()
        elif model_name == "SAM":
            path = self.sam_settings._model_path.text().strip()
        elif model_name == "YOLO-seg":
            path = self.yolo_settings._model_path.text().strip()
        elif model_name == "Custom ONNX":
            path = self.custom_settings._model_path.text().strip()
        else:
            path = ""

        if path and os.path.exists(path):
            self.load_model_from_widget(model_name)
        elif path and not os.path.exists(path):
            if model_name == "U-Net":
                self.unet_settings._model_path.setText("")
            elif model_name in ("DeepLabV3+", "DeepLabV3"):
                self.deeplab_settings._model_path.setText("")
            elif model_name == "SegFormer":
                self.segformer_settings._model_path.setText("")
            elif model_name == "SAM":
                self.sam_settings._model_path.setText("")
            elif model_name == "YOLO-seg":
                self.yolo_settings._model_path.setText("")
            elif model_name == "Custom ONNX":
                self.custom_settings._model_path.setText("")
            self.log(f"Путь к модели {model_name} не существует, очищен.")

        self.display_current_image()

    def on_device_changed(self, idx):
        device_text = self.device_combo.currentText()
        if device_text.startswith("CPU"):
            new_device = "cpu"
        elif device_text.startswith("CUDA"):
            new_device = "cuda"
        elif "MPS" in device_text:
            new_device = "mps"
        elif device_text.startswith("Auto"):
            new_device = "auto"
        else:
            new_device = "cpu"

        if hasattr(self, 'device') and self.device == new_device:
            return

        self.device = new_device
        self.log(f"Выбрано устройство: {self.device}")

        if self.segmentor is not None:
            model_name = self.model_combo.currentText()
            if model_name == "U-Net":
                path = self.unet_settings._model_path.text().strip()
            elif model_name in ("DeepLabV3+", "DeepLabV3"):
                path = self.deeplab_settings._model_path.text().strip()
            elif model_name == "SegFormer":
                path = self.segformer_settings._model_path.text().strip()
            elif model_name == "SAM":
                path = self.sam_settings._model_path.text().strip()
            elif model_name == "YOLO-seg":
                path = self.yolo_settings._model_path.text().strip()
            elif model_name == "Custom ONNX":
                path = self.custom_settings._model_path.text().strip()
            else:
                path = ""

            if path and os.path.exists(path):
                self.load_model_from_widget(model_name)
            else:
                self.segmentor = None
                self._update_segment_button_color()
                self.log(f"Модель {model_name} не найдена, сброшена.")

    # ---------- Гистограмма ----------
    def update_histogram(self, gray_img):
        segmentation_histogram.update_histogram(self, gray_img)

    def update_current_histogram(self):
        segmentation_histogram.update_current(self)

    # ---------- DL операции ----------
    def run_segmentation(self):
        segmentation_computation.run_dl(self)

    # ---------- Отображение ----------

    def _mask_signature(self, frame):
        return frame.key, id(self.segmentor), id(self.prediction_mask)

    def _compute_mask(self, frame):
        from utils_new.segmentation.result import ResultStatus
        if self.segmentor is not None and self.prediction_mask is None and self.result_state.status not in (ResultStatus.ERROR, ResultStatus.RUNNING):
            self.run_segmentation()
        return self.prediction_mask, {'dl_view': self.prediction_mask}

    def display_current_image(self):
        segmentation_presentation.display(self)


    # ---------- Работа со списком объектов ----------
    def update_object_list(self):
        segmentation_results.update_object_list(self)

    def update_annotated_view(self):
        segmentation_results.update_overlay(self)

    # ---------- Универсальное отображение координат ----------
    def update_coordinates_display(self):
        segmentation_results.update_coordinates(self)

    # ---------- Навигация ----------
    def update_navigation_state(self):
        segmentation_navigation.update_state(self)

    def prev_image(self):
        segmentation_navigation.step(self, -1, reset=lambda: setattr(self, 'prediction_mask', None))

    def next_image(self):
        segmentation_navigation.step(self, 1, reset=lambda: setattr(self, 'prediction_mask', None))

    def goto_image(self, page_num):
        segmentation_navigation.navigate(self, page_num-1, reset=lambda: setattr(self, 'prediction_mask', None))

    def load_images_from_dialog(self):
        segmentation_input.choose_images(self, self._load_images)

    def load_folder(self):
        segmentation_input.choose_folder(self, self._load_images)

    def reload_current_images(self):
        segmentation_input.reload_images(self, self._load_images)

    # ------- Ленивая загрузка (единая точка входа) -------
    def _load_images(self, source):
        segmentation_input.load_cached(self, source, self.segmentation_view_names, max_side=640, reset=lambda: setattr(self, 'prediction_mask', None))

    # ---------- Сохранение аннотаций (только выбранные) ----------
    def save_current_annotations(self):
        segmentation_export.save_selected(self)

    # ---------- UI события ----------
    def reset_all_zooms(self):
        segmentation_navigation.reset_zooms(self)

    def on_resize_mode_changed(self, enabled):
        segmentation_controller.resize_changed(self, enabled)

    def clear_images(self):
        segmentation_input.clear_session(self, none_fields=('prediction_mask',))

    def schedule_update(self):
        segmentation_controller.schedule(self)



    def on_draw_mode_changed(self, idx):
        segmentation_presentation.geometry_changed(self)

    def on_hull_changed(self, state):
        segmentation_presentation.geometry_changed(self)

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentData(), self.hull_checkbox.isChecked())

    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)

    # ---------- Завершение работы ----------
    def closeEvent(self, event):
        segmentation_controller.close(self)
        super().closeEvent(event)


def center_window(window, width=1200, height=800):
    window.resize(width, height)
    frame = window.frameGeometry()
    screen = QApplication.primaryScreen().availableGeometry()
    frame.moveCenter(screen.center())
    window.move(frame.topLeft())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepLearningWindow()
    center_window(window)
    window.show()
    from ui.application import exec_window
    sys.exit(exec_window(window, app))