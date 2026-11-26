# deep_learning_training_ui.py
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget

from ui.DL_train_settings_base import UNetTrainWidget, YOLOTrainWidget
from ui.DL_train_settings_new_model import (
    DeepLabV3TrainWidget,
    SegFormerTrainWidget,
)
from ui.sam.training import SAMTrainWidget
from ui.log_widget import LogWidget
from ui.model_export import ModelExportWidget
from ui.autoencoder_training import AutoencoderTrainWidget


def setup_deep_learning_training_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    splitter = QSplitter(Qt.Horizontal)
    main_layout.addWidget(splitter, 1)

    # ----- Левая панель: лог -----
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(5, 5, 5, 5)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    left_layout.addWidget(parent.log_widget)
    splitter.addWidget(left_widget)

    # ----- Правая панель: вкладки с параметрами (с прокруткой внутри каждой вкладки) -----
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 5, 5, 5)

    parent.tab_widget = QTabWidget()
    right_layout.addWidget(parent.tab_widget)

    # Функция для обёртки виджета в QScrollArea
    def wrap_in_scroll(widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    # 1. YOLO
    yolo_widget = YOLOTrainWidget(parent)
    yolo_scroll = wrap_in_scroll(yolo_widget)
    parent.tab_widget.addTab(yolo_scroll, "YOLO")
    yolo_widget.run_training.connect(parent._run_yolo_training_from_widget)
    yolo_widget.stop_training.connect(parent.on_stop_training)
    parent.yolo_train_widget = yolo_widget

    # 2. U-Net
    unet_widget = UNetTrainWidget(parent)
    unet_scroll = wrap_in_scroll(unet_widget)
    parent.tab_widget.addTab(unet_scroll, "U-Net")
    unet_widget.run_training.connect(parent._run_unet_training_from_widget)
    unet_widget.stop_training.connect(parent.on_stop_unet)
    parent.unet_train_widget = unet_widget

    # 3. DeepLabV3+
    deeplab_widget = DeepLabV3TrainWidget(parent)
    deeplab_scroll = wrap_in_scroll(deeplab_widget)
    parent.tab_widget.addTab(deeplab_scroll, "DeepLabV3/V3+")
    deeplab_widget.run_training.connect(parent._run_deeplab_training)
    deeplab_widget.stop_training.connect(parent.on_stop_deeplab)
    parent.deeplab_train_widget = deeplab_widget

    # 4. SegFormer
    segformer_widget = SegFormerTrainWidget(parent)
    segformer_scroll = wrap_in_scroll(segformer_widget)
    parent.tab_widget.addTab(segformer_scroll, "SegFormer")
    segformer_widget.run_training.connect(parent._run_segformer_training)
    segformer_widget.stop_training.connect(parent.on_stop_segformer)
    parent.segformer_train_widget = segformer_widget

    # 5. SAM
    sam_widget = SAMTrainWidget(parent)
    sam_scroll = wrap_in_scroll(sam_widget)
    parent.tab_widget.addTab(sam_scroll, "SAM")
    sam_widget.run_training.connect(parent._run_sam_training)
    sam_widget.stop_training.connect(parent.on_stop_sam)
    parent.sam_train_widget = sam_widget

    parent.model_export_widget = ModelExportWidget(parent)
    parent.model_export_widget.log_message.connect(parent.log_text.append)
    parent.tab_widget.addTab(wrap_in_scroll(parent.model_export_widget), "Экспорт модели")
    parent.autoencoder_train_widget = AutoencoderTrainWidget(parent)
    parent.autoencoder_train_widget.run_training.connect(parent._run_autoencoder_training)
    parent.autoencoder_train_widget.stop_training.connect(parent.on_stop_autoencoder)
    parent.tab_widget.addTab(wrap_in_scroll(parent.autoencoder_train_widget), 'Обучение VAE/AE')

    splitter.addWidget(right_widget)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([400, 500])   # начальные пропорции левой и правой части
