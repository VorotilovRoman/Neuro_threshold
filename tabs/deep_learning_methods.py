
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

import cv2
import numpy as np
from PyQt5.QtCore import QTimer
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
from ui.segmentation.preset_bindings import DeepLearningPresetManager
from ui.settings_service import settings
from ui_tabs.deep_learning_methods_ui import setup_deep_learning_methods_ui
from utils_new.morphology import apply_morphology
from utils_new.segmentation.postprocessing import extract_objects
from utils_ops.deep_learning_ops import create_segmentor

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
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.save_preset_btn.clicked.connect(self.save_as_preset)
        self.delete_preset_btn.clicked.connect(self.delete_current_preset)
        self.toggle_log_btn.clicked.connect(self.toggle_log)
        self.toggle_hist_btn.clicked.connect(self.toggle_histogram)

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.display_current_image)

        # Навигация
        self.nav_widget.load_images.connect(self.load_images_from_dialog)
        self.nav_widget.load_folder.connect(self.load_folder)
        self.nav_widget.prev.connect(self.prev_image)
        self.nav_widget.next.connect(self.next_image)
        self.nav_widget.goto_page.connect(self.goto_image)
        self.nav_widget.resize_toggled.connect(self.on_resize_mode_changed)

        # Кнопки
        self.segment_button.clicked.connect(self.run_segmentation)
        self.save_button.clicked.connect(self.save_current_annotations)

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
        self.invert_checkbox.stateChanged.connect(self.schedule_update)
        self.morphology_widget.parameters_changed.connect(self.schedule_update)
        self.reset_zoom_button.clicked.connect(self.reset_all_zooms)

        self.hull_checkbox.stateChanged.connect(self.schedule_update)
        self.object_list.itemChanged.connect(self.on_object_selection_changed)
        self.draw_combo.currentIndexChanged.connect(self.on_draw_mode_changed)

        settings.settings_changed.connect(self.on_global_settings_changed)

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
        if model_name == "U-Net":
            file_path = self.unet_settings._model_path.text().strip()
        elif model_name in ("DeepLabV3+", "DeepLabV3"):
            file_path = self.deeplab_settings._model_path.text().strip()
        elif model_name == "SegFormer":
            file_path = self.segformer_settings._model_path.text().strip()
        elif model_name == "SAM":
            file_path = self.sam_settings._model_path.text().strip()
        elif model_name == "YOLO-seg":
            file_path = self.yolo_settings._model_path.text().strip()
        elif model_name == "Custom ONNX":
            file_path = self.custom_settings._model_path.text().strip()
        else:
            return

        if not file_path:
            QMessageBox.warning(self, "Ошибка", "Не выбран файл модели.")
            return
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Ошибка", f"Файл не найден:\n{file_path}")
            return

        if model_name in ["SegFormer", "SAM"]:
            QMessageBox.information(self, "Информация",
                                    f"Модель {model_name} пока не реализована.\n"
                                    "Функциональность будет добавлена в следующих версиях.")
            self.segmentor = None
            self.model_metadata = None
            self.prediction_mask = None
            self.log(f"Попытка загрузить модель {model_name} – не реализована.")
            self._update_segment_button_color()
            self.display_current_image()
            return

        try:
            self.segmentor = create_segmentor(model_name, device=self.device)

            if model_name == "U-Net":
                try:
                    self.segmentor.load(weights_path=file_path)
                    if hasattr(self.segmentor, 'metadata') and self.segmentor.metadata:
                        self._update_unet_ui_from_metadata(self.segmentor.metadata)
                    else:
                        self.log("Предупреждение: метаданные модели не загружены, UI не обновлён.")
                except Exception as e:
                    self.log(f"Ошибка загрузки U-Net: {e}")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель U-Net:\n{e}")
                    self.segmentor = None
                    self._update_segment_button_color()
                    self.display_current_image()
                    return


            elif model_name in ("DeepLabV3+", "DeepLabV3"):
                try:
                    self.segmentor.load(weights_path=file_path)
                    if hasattr(self.segmentor, 'metadata') and self.segmentor.metadata:
                        self._update_deeplab_ui_from_metadata(self.segmentor.metadata)
                    else:
                        self.log("Предупреждение: метаданные DeepLab не загружены.")
                except Exception as e:
                    self.log(f"Ошибка загрузки DeepLab: {e}")
                    QMessageBox.critical(self, "Ошибка",
                                         f"Не удалось загрузить модель DeepLab:\n{e}")
                    self.segmentor = None
                    self._update_segment_button_color()
                    self.display_current_image()
                    return

            elif model_name == "YOLO-seg":
                self.segmentor.load(model_path=file_path)

            elif model_name == "Custom ONNX":
                self.segmentor.load(model_path=file_path)

            else:
                self.segmentor.load()

            self.log(f"Модель {model_name} загружена из {file_path}")
            self.prediction_mask = None
            self._update_segment_button_color()  # зелёный
            self.display_current_image()
        except Exception as e:
            self.log(f"Ошибка загрузки модели: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            self.segmentor = None
            self._update_segment_button_color()  # красный
            self.display_current_image()

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
        if self.ignore_ui_changes:
            return
        if self.segmentor is not None:
            self.prediction_mask = None
        self.schedule_update()

    # ---------- Логирование ----------
    def log(self, message):
        segmentation_actions.log(self, message)

    # ---------- Пресеты ----------
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

    # ---------- Выбор модели ----------
    def on_model_changed(self, idx):
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
        if self.original_image is None:
            QMessageBox.warning(self, "Нет изображения", "Загрузите изображение.")
            return
        if self.segmentor is None:
            QMessageBox.warning(self, "Нет модели", "Сначала загрузите модель.")
            return

        model_name = self.model_combo.currentText()
        if model_name in ["SegFormer", "SAM"]:
            QMessageBox.information(self, "Информация",
                                    f"Модель {model_name} не реализована, сегментация невозможна.")
            self.prediction_mask = np.zeros(self.current_gray.shape, dtype=np.uint8)
            return

        self.progress_bar.setValue(0)
        self.segment_button.setEnabled(False)
        QApplication.processEvents()

        try:
            img = self.original_image
            kwargs = {}
            self.log(f"=== Запуск сегментации (модель: {model_name}) ===")
            self.log(f"Изображение: {img.shape}, dtype={img.dtype}, диапазон [{img.min()}, {img.max()}]")

            if model_name == "YOLO-seg":
                kwargs['conf'] = self.yolo_settings._conf.value()
                kwargs['iou'] = self.yolo_settings._iou.value()
                kwargs['imgsz'] = self.yolo_settings._imgsz.value()
                kwargs['save'] = self.yolo_settings._save.isChecked()
                self.log(f"Параметры YOLO: conf={kwargs['conf']}, iou={kwargs['iou']}, "
                         f"imgsz={kwargs['imgsz']}, save={kwargs['save']}")

            if model_name == "U-Net":
                kwargs['threshold'] = self.unet_settings._threshold.value()
                self.log(f"Порог U‑Net: {kwargs['threshold']}")

                # Валидация перед запуском
                ok, err = self.unet_settings.validate_for_prediction()
                if not ok:
                    QMessageBox.warning(self, "Ошибка", err)
                    self.log(f"❌ {err}")
                    self.segment_button.setEnabled(True)
                    return

                # Применяем override к metadata (если модель загружена)
                if self.segmentor is not None and self.segmentor.metadata is not None:
                    md = self.segmentor.metadata
                    overrides = self.unet_settings.get_override_params()
                    for k, v in overrides.items():
                        md[k] = v
                        self.log(f"  Override {k} = {v}")


            if model_name in ("DeepLabV3+", "DeepLabV3"):
                kwargs['threshold'] = self.deeplab_settings._threshold.value()
                self.log(f"Порог DeepLab: {kwargs['threshold']}")

                ok, err = self.deeplab_settings.validate_for_prediction()
                if not ok:
                    QMessageBox.warning(self, "Ошибка", err)
                    self.log(f"❌ {err}")
                    self.segment_button.setEnabled(True)
                    return

                if self.segmentor is not None and self.segmentor.metadata is not None:
                    md = self.segmentor.metadata
                    overrides = self.deeplab_settings.get_override_params()
                    for k, v in overrides.items():
                        md[k] = v
                        self.log(f"  Override {k} = {v}")

            mask = self.segmentor.predict(img, **kwargs)

            self.log(f"Маска после модели: {mask.shape}, dtype={mask.dtype}, "
                     f"мин={mask.min()}, макс={mask.max()}, среднее={mask.mean():.3f}")

            if mask.dtype == np.uint8 and mask.max() <= 1:
                object_pixels = np.sum(mask > 0)
                self.log(f"Количество пикселей объекта: {object_pixels} ({100*object_pixels/mask.size:.2f}%)")
            else:
                object_pixels = np.sum(mask > 127)
                self.log(f"Количество пикселей объекта (порог 128): {object_pixels} ({100*object_pixels/mask.size:.2f}%)")

            self.prediction_mask = mask
            self.progress_bar.setValue(100)

        except Exception as e:
            self.log(f"Ошибка сегментации: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить сегментацию:\n{e}")
            self.prediction_mask = np.zeros(self.current_gray.shape, dtype=np.uint8)
        finally:
            self.segment_button.setEnabled(True)

    # ---------- Отображение ----------
    def display_current_image(self):
        if self._lazy_cache is None or len(self._lazy_cache) == 0:
            self.update_current_histogram()
            for view in (self.original_view, self.dl_view,
                         self.morph_view, self.annotated_view):
                view.set_pixmap(numpy_to_qpixmap(None))
            self.info_label.setText("No images loaded")
            self.object_list.clear()
            return

        idx = self.current_index
        total = len(self._lazy_cache)

        self.info_label.setText(f"Image {idx + 1} of {total}")
        current_file = os.path.basename(self.image_paths[idx])
        base_name = os.path.splitext(current_file)[0]
        self.log(f"Отображён снимок: {current_file}")

        self.original_view.set_suggested_save_name(f"dl_original_{base_name}")
        self.dl_view.set_suggested_save_name(f"dl_mask_{base_name}")
        self.morph_view.set_suggested_save_name(f"dl_morph_{base_name}")
        self.annotated_view.set_suggested_save_name(f"dl_annotated_{base_name}")

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
        self.original_view.set_pixmap(numpy_to_qpixmap(original))

        if self.segmentor is not None and self.prediction_mask is None:
            self.run_segmentation()

        mask = self.prediction_mask if self.prediction_mask is not None else np.zeros(gray.shape, dtype=np.uint8)
        self._apply_morphology_and_draw(mask, gray, original)
        self.update_navigation_state()

        # --- Префетч следующего окна + вытеснение дальних ---
        ahead = self._lazy_cache.prefetch_ahead
        self._lazy_cache.prefetch(range(idx + 1, min(idx + 1 + ahead + 1, total)))
        self._lazy_cache.trim_around(idx, keep_behind=3, keep_ahead=ahead + 4)

    def _apply_morphology_and_draw(self, mask, gray, original):

        self.log(f"=== Постобработка морфологией ===")
        self.log(f"Входная маска (DL): {mask.shape}, dtype={mask.dtype}, "
                 f"мин={mask.min()}, макс={mask.max()}, среднее={mask.mean():.3f}")
        self.log(f"Морфология, px: {self.morphology_widget.get_settings()}")

        processed = apply_morphology(mask, self.morphology_widget.get_steps(),
                                         shape=self.morphology_widget.kernel_shape())

        self.log(f"Маска после морфологии: {processed.shape}, dtype={processed.dtype}, "
                 f"мин={processed.min()}, макс={processed.max()}, среднее={processed.mean():.3f}")

        if self.invert_checkbox.isChecked():
            self.log("Инверсия маски применена")
            processed = cv2.bitwise_not(processed)

        self.dl_view.set_pixmap(numpy_to_qpixmap(mask))
        self.morph_view.set_pixmap(numpy_to_qpixmap(processed))

        # Подготовка изображения для аннотаций (фон делаем белым)
        if len(original.shape) == 2:
            result = original.copy()
            result[processed == 0] = 255
        else:
            result = original.copy()
            mask_bool = processed == 0
            result[mask_bool] = [255, 255, 255]

        if len(result.shape) == 2:
            display_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            display_img = result.copy()

        draw_mode = self.draw_combo.currentText()
        self.log(f"=== Отрисовка объектов (режим: {draw_mode}) ===")

        self.current_base_image = display_img.copy()
        _, self.current_objects_full = self.draw_objects_on_image(display_img, processed, draw=False)

        if self.current_objects_full:
            self.log(f"Обнаружено объектов: {len(self.current_objects_full)}")
            for i, obj in enumerate(self.current_objects_full[:5]):
                if isinstance(obj, tuple) and len(obj) > 0 and obj[0] in ('detect', 'obb', 'segment'):
                    typ = obj[0]
                    if typ == 'detect':
                        _, cls, cx, cy, w, h = obj
                        self.log(f"  Объект {i+1}: detect (class {cls})")
                    elif typ == 'obb':
                        _, cls, pts = obj
                        self.log(f"  Объект {i+1}: OBB (class {cls})")
                    else:
                        _, cls, pts = obj
                        self.log(f"  Объект {i+1}: polygon (class {cls})")
                elif len(obj) == 4:
                    self.log(f"  Объект {i+1}: прямоугольник x={obj[0]}, y={obj[1]}, w={obj[2]}, h={obj[3]}")
                elif len(obj) == 5:
                    self.log(f"  Объект {i+1}: центр=({obj[0]:.1f},{obj[1]:.1f}), размер=({obj[2]:.1f}x{obj[3]:.1f}), angle={obj[4]:.1f}°")
            if len(self.current_objects_full) > 5:
                self.log(f"  ... и ещё {len(self.current_objects_full)-5} объектов")
        else:
            self.log("Объектов не обнаружено")

        self.update_object_list()

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
        if self.image_paths:
            reply = QMessageBox.question(
                self, "Resize Mode Changed",
                "Resize mode changed. Reload images?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.reload_current_images()
            else:
                self.clear_images()

    def clear_images(self):
        segmentation_input.clear_session(self, none_fields=('prediction_mask',))

    def schedule_update(self):
        self.update_timer.start(50)



    def on_draw_mode_changed(self, idx):
        mode = self.draw_combo.currentText()
        if mode in ["Segmentation (Polygon)", "OBB (Oriented Box)"]:
            self.hull_checkbox.setVisible(True)
        else:
            self.hull_checkbox.setVisible(False)
        self.schedule_update()

    def on_hull_changed(self, state):
        self.schedule_update()

    def on_object_selection_changed(self, item):
        segmentation_results.on_selection_changed(self, item)

    def draw_objects_on_image(self, img, binary, draw=True):
        return extract_objects(img, binary, self.draw_combo.currentText(), self.hull_checkbox.isChecked())

    def toggle_log(self, checked):
        segmentation_actions.toggle_log(self, checked)

    def toggle_histogram(self, checked):
        segmentation_actions.toggle_histogram(self, checked)

    # ---------- Завершение работы ----------
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