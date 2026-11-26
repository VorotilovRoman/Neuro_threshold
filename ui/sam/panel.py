"""One set of local SAM/SAM2/SAM3/Grounding DINO controls."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)
from utils_new.import_export.sam_weights import MODELS
from utils_new.segmentation.backends.sam import MODELS_DIR


class AutoLabelStack(QStackedWidget):
    """Size to the active model's controls, including when SAM is hidden."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _: self.updateGeometry())

    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() else super().minimumSizeHint()


class SamPanel(QWidget):
    settings_restored = pyqtSignal()

    preset_fields = ('model', 'weights', 'tokenizer', 'device', 'mode', 'dino_directory',
                     'dino_device', 'dino_box_score', 'dino_text_score', 'dino_nms',
                     'dino_max_objects', 'text', 'class_id', 'kind', 'score', 'area')

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        self.model = QComboBox()
        for key, item in MODELS.items():
            self.model.addItem(item.name, key)
        self.model.setCurrentIndex(self.model.findData("sam2.1-tiny"))
        form.addRow("Архитектура:", self.model)
        self.weights, self.browse_weights = self._path_row(form, "Веса:", "Локальный .pt / .pth")
        self.tokenizer, self.browse_tokenizer = self._path_row(form, "Токенизатор:", "SAM3: BPE GZIP рядом с весами")
        self.tokenizer_row = form.itemAt(form.rowCount() - 1, QFormLayout.FieldRole).widget()
        self.tokenizer_label = form.labelForField(self.tokenizer_row)
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda:0"])
        form.addRow("Устройство:", self.device)
        self.load = QPushButton("Загрузить SAM")
        self.download = QPushButton("Скачать файлы…")
        self.download.setToolTip("Найти локальные веса выбранной архитектуры и предложить скачать отсутствующие файлы.")
        model_actions = QHBoxLayout()
        model_actions.addWidget(self.load)
        model_actions.addWidget(self.download)
        form.addRow(model_actions)
        self.mode = QComboBox()
        self.mode.addItem("Точки и рамка", "points")
        self.mode.addItem("Текст и рамка (SAM3)", "text")
        self.mode.addItem("Текст + DINO (SAM2)", "grounded")
        self.mode.setToolTip("При смене режима загрузите модель заново: в памяти хранится один выбранный backend.")
        form.addRow("Подсказки:", self.mode)
        self.mode_label = form.labelForField(self.mode)
        self.grounding_group = QGroupBox("Grounding DINO (локально)")
        grounding_layout = QVBoxLayout(self.grounding_group)
        grounding_form = QFormLayout()
        grounding_layout.addLayout(grounding_form)
        self.dino_directory, self.browse_dino = self._path_row(grounding_form, "Папка:", "config.json + model.safetensors + токенизатор")
        defaults = (MODELS_DIR / "grounding_dino" / "tiny",
                    MODELS_DIR / "segment_anything" / "grounding_dino" / "tiny")
        self.dino_directory.setText(str(next((path for path in defaults if (path / "config.json").is_file()
                                              and (path / "model.safetensors").is_file()), defaults[1])))
        self.dino_options_button = QPushButton("Параметры Grounding DINO…")
        self.dino_options_button.setCheckable(True)
        grounding_layout.addWidget(self.dino_options_button)
        self.dino_options = QWidget()
        grounding_form = QFormLayout(self.dino_options)
        grounding_layout.addWidget(self.dino_options)
        self.dino_options.setVisible(False)
        self.dino_options_button.toggled.connect(self.dino_options.setVisible)
        self.dino_device = QComboBox()
        self.dino_device.addItem("CPU — меньше VRAM", "cpu")
        self.dino_device.addItem("Как SAM2", "same")
        grounding_form.addRow("Устройство:", self.dino_device)
        for name, caption, value, tip in (
            ("dino_box_score", "Порог боксов:", 0.35, "Минимальная уверенность детектора Grounding DINO."),
            ("dino_text_score", "Порог текста:", 0.25, "Порог токенов, формирующих текстовую метку детектора."),
            ("dino_nms", "IoU дублей:", 0.5, "NMS: подавление перекрывающихся рамок до сегментации."),
        ):
            spin = QDoubleSpinBox()
            spin.setRange(0, 1)
            spin.setSingleStep(0.05)
            spin.setValue(value)
            spin.setToolTip(tip)
            setattr(self, name, spin)
            grounding_form.addRow(caption, spin)
        self.dino_max_objects = QSpinBox()
        self.dino_max_objects.setRange(1, 500)
        self.dino_max_objects.setValue(50)
        grounding_form.addRow("Макс. объектов:", self.dino_max_objects)
        form.addRow(self.grounding_group)
        self.text = QLineEdit()
        self.text.setPlaceholderText("Например: metal screw")
        self.text.setToolTip("SAM3 или Grounding DINO + SAM2. Короткое английское описание одного класса; ID датасета задаётся отдельно.")
        form.addRow("Текст:", self.text)
        self.text_label = form.labelForField(self.text)
        buttons = QHBoxLayout()
        self.positive, self.negative, self.box = (QPushButton(text) for text in ("+ Объект", "− Фон", "Рамка"))
        for button in (self.positive, self.negative, self.box):
            buttons.addWidget(button)
        form.addRow(buttons)
        buttons = QHBoxLayout()
        self.selected_box = QPushButton("Бокс из объекта")
        self.selected_box.setToolTip("Взять рамку и ID класса из выделенной аннотации, не изменяя её.")
        self.clear = QPushButton("Очистить")
        self.clear.setToolTip("Очистить все точки, рамку и предпросмотр SAM.")
        buttons.addWidget(self.selected_box)
        buttons.addWidget(self.clear)
        form.addRow(buttons)
        self.class_id = QSpinBox()
        self.class_id.setRange(0, 1000000)
        self.class_id.setToolTip("ID класса вашего датасета. SAM не назначает классы YOLO автоматически.")
        form.addRow("ID класса:", self.class_id)
        self.kind = QComboBox()
        for label, key in (("Segment", "segment"), ("BBox", "detect"), ("OBB", "obb")):
            self.kind.addItem(label, key)
        self.kind.setToolTip("Маски преобразуются в редактируемые объекты. Полигон YOLO не сохраняет отверстия маски.")
        form.addRow("Результат:", self.kind)
        self.kind_label = form.labelForField(self.kind)
        self.score = QDoubleSpinBox()
        self.score.setRange(0, 1)
        self.score.setSingleStep(0.05)
        self.score.setValue(0.5)
        self.score.setToolTip("SAM/SAM2: предсказанный IoU маски; SAM3 с текстом: уверенность. Порог боксов DINO задаётся отдельно.")
        form.addRow("Мин. оценка:", self.score)
        self.area = QSpinBox()
        self.area.setRange(1, 1000000000)
        self.area.setValue(10)
        form.addRow("Мин. площадь, px²:", self.area)
        self.replace = QCheckBox("Заменить все аннотации на снимке")
        form.addRow(self.replace)
        self.run = QPushButton("Предпросмотр")
        self.run.setToolTip("Вычислить маски по текущим подсказкам, не изменяя аннотации.")
        self.accept = QPushButton("Принять результат")
        self.accept.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run)
        buttons.addWidget(self.accept)
        form.addRow(buttons)
        layout.addWidget(self.controls)
        self.status = QLabel("Выберите локальные веса и загрузите модель.")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.status)
        self.hint = QLabel("Точки: ЛКМ — объект, ПКМ — фон. Рамка: протяните ЛКМ.")
        self.hint.setToolTip("Навигация, Escape и ручные инструменты выключают ввод подсказок. "
                             "Голубой контур — предпросмотр; сохранение доступно после принятия.")
        self.hint.setWordWrap(True)
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.hint)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.cancel = QPushButton("Отменить вычисление")
        self.cancel.setEnabled(False)
        self.cancel.setVisible(False)
        layout.addWidget(self.cancel)
        self.capabilities()

    @staticmethod
    def _path_row(form, label, placeholder):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        edit, button = QLineEdit(), QPushButton("…")
        edit.setPlaceholderText(placeholder)
        button.setMaximumWidth(32)
        row.addWidget(edit)
        row.addWidget(button)
        form.addRow(label, widget)
        return edit, button

    def capabilities(self):
        key = self.model.currentData()
        sam3, sam2 = key == "sam3", key.startswith(("sam2-", "sam2.1-"))
        self.mode.model().item(1).setEnabled(sam3)
        self.mode.model().item(2).setEnabled(sam2)
        self.mode.setEnabled(sam3 or sam2)
        self.mode.setVisible(sam3 or sam2)
        self.mode_label.setVisible(sam3 or sam2)
        if not ((self.mode.currentData() == "text" and sam3) or (self.mode.currentData() == "grounded" and sam2)):
            self.mode.setCurrentIndex(0)
        native_text = sam3 and self.mode.currentData() == "text"
        grounded = sam2 and self.mode.currentData() == "grounded"
        text_mode = native_text or grounded
        self.grounding_group.setVisible(grounded)
        self.tokenizer_row.setVisible(native_text)
        self.tokenizer_label.setVisible(native_text)
        self.text.setEnabled(text_mode)
        self.text.setVisible(text_mode)
        self.text_label.setVisible(text_mode)
        self.positive.setEnabled(not text_mode)
        self.negative.setEnabled(not text_mode)
        self.box.setToolTip("Ограничить область текстового поиска DINO (ROI)." if grounded else "Нарисовать рамку-подсказку.")
        self.load.setText("Загрузить SAM2 + DINO" if grounded else "Загрузить SAM")
        self.hint.setText("Grounding DINO находит рамки по тексту, SAM2 строит маски. Рамка ограничивает область поиска."
                          if grounded else "SAM3: текст описывает объекты, рамка задаёт визуальную подсказку."
                          if native_text else "Точки: ЛКМ — объект, ПКМ — фон. Рамка: протяните ЛКМ.")

    def get_settings(self):
        from ui.segmentation.preset_bindings import control_value
        return {name: control_value(getattr(self, name)) for name in self.preset_fields}

    def validate_settings(self, values):
        from ui.segmentation.preset_bindings import validate_control
        if not isinstance(values, dict) or set(values) != set(self.preset_fields):
            raise ValueError('Неполные настройки общего блока SAM')
        for name, value in values.items():
            validate_control(getattr(self, name), value)
        key = self.model.itemData(self.model.findText(values['model']))
        mode = self.mode.itemData(self.mode.findText(values['mode']))
        if ((mode == 'text' and key != 'sam3')
                or (mode == 'grounded' and not key.startswith(('sam2-', 'sam2.1-')))):
            raise ValueError('Режим подсказок не поддерживается выбранной моделью SAM')

    def set_settings(self, values):
        from ui.segmentation.preset_bindings import blocked_signals, set_control_value
        self.validate_settings(values)
        with blocked_signals([getattr(self, name) for name in values]):
            for name, value in values.items():
                set_control_value(getattr(self, name), value)
        self.capabilities()
        self.settings_restored.emit()
