from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)

from ui.DL_predict_settings import (
    CustomONNXInferenceSettings,
    DeepLabV3InferenceSettings,
    SAMInferenceSettings,
    SegFormerInferenceSettings,
    UNetInferenceSettings,
    YOLOInferenceSettings,
)
from ui.morphology_ui import MorphologyWidget

# deep_learning_methods_ui.py
from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout


def setup_deep_learning_methods_ui(parent):
    controls_layout = begin_layout(parent, 'deep_learning')

    # Выбор модели
    model_layout = QHBoxLayout()
    model_layout.addWidget(QLabel("Model:"))
    parent.model_combo = QComboBox()
    models = ["YOLO-seg", "U-Net", "DeepLabV3+", "SegFormer", "SAM", "Custom ONNX"]
    parent.model_combo.addItems(models)
    model_layout.addWidget(parent.model_combo)
    controls_layout.addLayout(model_layout)

    # Виджеты настроек
    parent.unet_settings = UNetInferenceSettings()
    parent.unet_settings.setVisible(False)
    parent.unet_settings._encoder.setEnabled(False)
    parent.unet_settings._input_size.setEnabled(False)
    controls_layout.addWidget(parent.unet_settings)

    parent.deeplab_settings = DeepLabV3InferenceSettings()
    parent.deeplab_settings.setVisible(False)
    parent.deeplab_settings._backbone.setEnabled(False)
    parent.deeplab_settings._output_stride.setEnabled(False)
    controls_layout.addWidget(parent.deeplab_settings)

    parent.segformer_settings = SegFormerInferenceSettings()
    parent.segformer_settings.setVisible(False)
    parent.segformer_settings._variant.setEnabled(False)
    controls_layout.addWidget(parent.segformer_settings)

    parent.sam_settings = SAMInferenceSettings()
    parent.sam_settings.setVisible(False)
    parent.sam_settings._model_type.setEnabled(False)
    parent.sam_settings._prompt.setEnabled(False)
    controls_layout.addWidget(parent.sam_settings)

    parent.yolo_settings = YOLOInferenceSettings()
    parent.yolo_settings.setVisible(False)
    parent.yolo_settings._conf.setEnabled(True)
    parent.yolo_settings._iou.setEnabled(True)
    controls_layout.addWidget(parent.yolo_settings)

    parent.custom_settings = CustomONNXInferenceSettings()
    parent.custom_settings.setVisible(False)
    controls_layout.addWidget(parent.custom_settings)

    # Общий выбор устройства
    device_container = QWidget()
    device_layout = QHBoxLayout(device_container)
    device_layout.addWidget(QLabel("Device:"))
    parent.device_combo = QComboBox()
    parent.device_combo.addItems(["Auto (best available)", "CPU", "CUDA (GPU)", "MPS (Apple Silicon)"])
    device_layout.addWidget(parent.device_combo)
    device_layout.addStretch()
    controls_layout.addWidget(device_container)

    # Кнопка сегментации и прогресс
    parent.segment_button = QPushButton("Segment Current Image")
    controls_layout.addWidget(parent.segment_button)
    parent.progress_bar = QProgressBar()
    controls_layout.addWidget(parent.progress_bar)

    # Invert mask
    parent.invert_checkbox = QCheckBox("Invert mask")
    controls_layout.addWidget(parent.invert_checkbox)

    # Морфология
    morph_label = QLabel("Post-processing morphology:")
    controls_layout.addWidget(morph_label)

    parent.morphology_widget = MorphologyWidget(contour_available=False)
    controls_layout.addWidget(parent.morphology_widget)

    # Отрисовка объектов
    controls_layout.addWidget(ResultPanel(parent))

    finish_layout(parent)
