from import_libs_external import *

class BaseInferenceSettings(QGroupBox):
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self._setup_ui()

    def _setup_ui(self):
        raise NotImplementedError

    def _row(self, label, widget_or_layout):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        if isinstance(widget_or_layout, QHBoxLayout):
            container = QWidget()
            container.setLayout(widget_or_layout)
            row.addWidget(container)
        else:
            row.addWidget(widget_or_layout)
        row.addStretch()
        return row

    def get_params(self):
        raise NotImplementedError


class YOLOInferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)
    conf_changed = pyqtSignal(float)
    iou_changed = pyqtSignal(float)
    imgsz_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Модель YOLO", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Выбор файла
        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pt")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        # Строка: порог уверенности и порог IoU в одной строке
        conf_iou_row = QHBoxLayout()
        conf_iou_row.addWidget(QLabel("Порог уверенности:"))
        self._conf = QDoubleSpinBox()
        self._conf.setRange(0.01, 1.0)
        self._conf.setValue(0.25)
        self._conf.setSingleStep(0.01)
        self._conf.setDecimals(3)
        conf_iou_row.addWidget(self._conf)

        conf_iou_row.addSpacing(20)
        conf_iou_row.addWidget(QLabel("Порог IoU (NMS):"))
        self._iou = QDoubleSpinBox()
        self._iou.setRange(0.01, 1.0)
        self._iou.setValue(0.45)
        self._iou.setSingleStep(0.01)
        self._iou.setDecimals(3)
        conf_iou_row.addWidget(self._iou)
        conf_iou_row.addStretch()
        layout.addLayout(conf_iou_row)

        # Строка: размер входа и чекбокс сохранения
        imgsz_save_row = QHBoxLayout()
        imgsz_save_row.addWidget(QLabel("Размер входа:"))
        self._imgsz = QSpinBox()
        self._imgsz.setRange(32, 1280)
        self._imgsz.setValue(640)
        self._imgsz.setSingleStep(80)
        imgsz_save_row.addWidget(self._imgsz)

        imgsz_save_row.addSpacing(20)
        self._save = QCheckBox("Сохранять результаты")
        imgsz_save_row.addWidget(self._save)
        imgsz_save_row.addStretch()
        layout.addLayout(imgsz_save_row)

        # Сигналы
        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._imgsz.valueChanged.connect(self.imgsz_changed.emit)
        self._conf.valueChanged.connect(self.conf_changed.emit)
        self._iou.valueChanged.connect(self.iou_changed.emit)



    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель YOLO", "", "PyTorch models (*.pt)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {
            'model_path': self._model_path.text(),
            'imgsz': self._imgsz.value(),
            'conf': self._conf.value(),
            'iou': self._iou.value(),
            'save': self._save.isChecked()
        }


class UNetInferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)
    encoder_changed = pyqtSignal(str)
    input_size_changed = pyqtSignal(int)
    threshold_changed = pyqtSignal(float)   # новый сигнал

    def __init__(self, parent=None):
        super().__init__("Параметры U‑Net", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Выбор файла
        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        self._encoder = QComboBox()
        self._encoder.addItems(["resnet18", "resnet34", "resnet50", "efficientnet-b0", "vgg16"])
        layout.addLayout(self._row("Энкодер", self._encoder))

        self._input_size = QComboBox()
        self._input_size.addItems(["256", "512", "1024"])
        self._input_size.setCurrentText("512")
        layout.addLayout(self._row("Размер входа", self._input_size))

        # ---------- Новый элемент: порог ----------
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.01, 0.99)
        self._threshold.setSingleStep(0.01)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.5)
        self._threshold.setToolTip("Порог бинаризации для получения маски (0.5 – стандарт)")
        layout.addLayout(self._row("Threshold", self._threshold))

        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._encoder.currentTextChanged.connect(self.encoder_changed.emit)
        self._input_size.currentTextChanged.connect(lambda v: self.input_size_changed.emit(int(v)))
        self._threshold.valueChanged.connect(self.threshold_changed.emit)   # новый сигнал

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель U‑Net", "", "PyTorch models (*.pth)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {
            'model_path': self._model_path.text(),
            'threshold': self._threshold.value()   # добавили
        }


class DeepLabV3InferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)
    backbone_changed = pyqtSignal(str)
    output_stride_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Параметры DeepLabV3+", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        self._backbone = QComboBox()
        self._backbone.addItems(["resnet50", "resnet101", "mobilenet_v3", "xception"])
        layout.addLayout(self._row("Backbone", self._backbone))

        self._output_stride = QComboBox()
        self._output_stride.addItems(["8", "16"])
        layout.addLayout(self._row("Output stride", self._output_stride))

        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._backbone.currentTextChanged.connect(self.backbone_changed.emit)
        self._output_stride.currentTextChanged.connect(lambda v: self.output_stride_changed.emit(int(v)))

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель DeepLabV3+", "", "PyTorch models (*.pth)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {'model_path': self._model_path.text()}


class SegFormerInferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)
    variant_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Параметры SegFormer", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        self._variant = QComboBox()
        self._variant.addItems(["mit-b0", "mit-b1", "mit-b2", "mit-b3", "mit-b4", "mit-b5"])
        layout.addLayout(self._row("Вариант", self._variant))

        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._variant.currentTextChanged.connect(self.variant_changed.emit)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель SegFormer", "", "PyTorch models (*.pth)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {'model_path': self._model_path.text()}


class SAMInferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)
    model_type_changed = pyqtSignal(str)
    prompt_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Параметры SAM", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        self._model_type = QComboBox()
        self._model_type.addItems(["vit_h", "vit_l", "vit_b"])
        layout.addLayout(self._row("Тип модели", self._model_type))

        self._prompt = QComboBox()
        self._prompt.addItems(["box", "points", "everything"])
        layout.addLayout(self._row("Тип промпта", self._prompt))

        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._model_type.currentTextChanged.connect(self.model_type_changed.emit)
        self._prompt.currentTextChanged.connect(self.prompt_changed.emit)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель SAM", "", "PyTorch models (*.pth)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {'model_path': self._model_path.text()}


class CustomONNXInferenceSettings(BaseInferenceSettings):
    model_path_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Custom ONNX", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите ONNX модель")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите ONNX модель", "", "ONNX files (*.onnx)")
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {'model_path': self._model_path.text()}