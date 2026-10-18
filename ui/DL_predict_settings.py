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

        # Строка: порог уверенности и порог IoU
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
    threshold_changed = pyqtSignal(float)
    override_changed = pyqtSignal()          # эмитится при ручном изменении override-полей

    def __init__(self, parent=None):
        # Флаг выставляем ДО вызова super().__init__, потому что он вызывает
        # _setup_ui, где могут быть обращения к атрибуту.
        # _show_inference_resize — логический признак «нужно ли поле
        # inference_resize_mode», не зависящий от фактической Qt-видимости
        # (QWidget.isVisible() может быть False, пока не показан родитель).
        self._show_inference_resize = False
        super().__init__("Параметры U‑Net", parent)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---------- Файл модели ----------
        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        # ---------- Архитектура ----------
        self._encoder = QComboBox()
        self._encoder.addItems(["resnet18", "resnet34", "resnet50",
                                "efficientnet-b0", "vgg16"])
        layout.addLayout(self._row("Энкодер", self._encoder))

        self._input_size = QComboBox()
        self._input_size.addItems(["256", "512", "1024"])
        self._input_size.setCurrentText("512")
        layout.addLayout(self._row("Размер входа", self._input_size))


        # --- ДОБАВЛЯЕМ ОТОБРАЖЕНИЕ АРХИТЕКТУРЫ ---
        self._architecture = QLineEdit()
        self._architecture.setReadOnly(True)
        self._architecture.setPlaceholderText("Читается из метаданных модели")
        layout.addLayout(self._row("Архитектура", self._architecture))

        # ---------- Порог ----------
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.01, 0.99)
        self._threshold.setSingleStep(0.01)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.5)
        self._threshold.setToolTip(
            "Порог бинаризации для получения маски (0.5 – стандарт).\n"
            "Для num_classes=1 или 2 применяется; для num_classes>2 "
            "используется argmax."
        )
        layout.addLayout(self._row("Threshold", self._threshold))

        # ---------- Override-параметры ----------
        # Поля активны только если параметра нет в *_meta.json.
        # Если параметр есть — поле заполняется и блокируется.
        ov_group = QGroupBox("Override параметров инференса (если нет в meta)")
        ov_form = QFormLayout()
        ov_form.setSpacing(8)
        ov_group.setLayout(ov_form)

        self._num_classes_override = QSpinBox()
        self._num_classes_override.setRange(1, 100)
        self._num_classes_override.setValue(2)
        self._num_classes_override.setEnabled(False)
        self._num_classes_override.setToolTip(
            "Общее число классов, включая фон (фон = 0).\n\n"
            "  • 1 → бинарная (sigmoid + threshold);\n"
            "  • 2 → «фон + один объект»: softmax[:, 1] + threshold;\n"
            "  • >2 → multiclass (argmax, threshold не используется).\n\n"
            "Активно только если параметра нет в *_meta.json."
        )
        ov_form.addRow("num_classes:", self._num_classes_override)

        self._normalize_override = QCheckBox("Применять ImageNet-нормализацию")
        self._normalize_override.setChecked(False)
        self._normalize_override.setEnabled(False)
        self._normalize_override.setToolTip(
            "Только если модель обучалась с нормализацией. Иначе предсказания "
            "будут некорректны."
        )
        ov_form.addRow("Нормализация:", self._normalize_override)

        self._resize_mode_override = QComboBox()
        self._resize_mode_override.addItems(
            ["hard", "keep_aspect_pad", "random_crop"]
        )
        self._resize_mode_override.setEnabled(False)
        self._resize_mode_override.setToolTip(
            "Режим, использованный при ОБУЧЕНИИ:\n"
            "  • hard — жёсткий ресайз;\n"
            "  • keep_aspect_pad — AR + padding;\n"
            "  • random_crop — случайный кроп (только обучение).\n\n"
            "Если выбран random_crop — требуется отдельный выбор режима\n"
            "ресайза при инференсе (поле ниже)."
        )
        ov_form.addRow("resize_mode (train):", self._resize_mode_override)

        self._inference_resize_label = QLabel("Режим ресайза при ИНФЕРЕНСЕ:")
        self._inference_resize_override = QComboBox()
        self._inference_resize_override.addItems(["keep_aspect_pad", "hard"])
        self._inference_resize_override.setCurrentText("keep_aspect_pad")
        self._inference_resize_override.setToolTip(
            "Как приводить кадр к input_size при инференсе.\n"
            "Используется только если обучение шло с random_crop\n"
            "(оба варианта НЕ теряют информацию о кадре):\n"
            "  • keep_aspect_pad — масштаб с сохранением AR + чёрные поля;\n"
            "  • hard — жёсткий ресайз до input_size × input_size."
        )
        self._inference_resize_label.setVisible(False)
        self._inference_resize_override.setVisible(False)
        ov_form.addRow(self._inference_resize_label,
                       self._inference_resize_override)

        self._override_hint = QLabel("")
        self._override_hint.setWordWrap(True)
        self._override_hint.setStyleSheet(
            "color: #b8860b; font-size: 11px; padding: 4px;"
        )
        ov_form.addRow(self._override_hint)

        layout.addWidget(ov_group)

        # ---------- Сигналы ----------
        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._encoder.currentTextChanged.connect(self.encoder_changed.emit)
        self._input_size.currentTextChanged.connect(
            lambda v: self.input_size_changed.emit(int(v))
        )
        self._threshold.valueChanged.connect(self.threshold_changed.emit)

        # Override-поля: любое изменение эмитит override_changed
        self._num_classes_override.valueChanged.connect(
            lambda _v: self.override_changed.emit()
        )
        self._normalize_override.stateChanged.connect(
            lambda _s: self.override_changed.emit()
        )
        self._resize_mode_override.currentTextChanged.connect(
            self._on_train_resize_mode_changed
        )
        self._inference_resize_override.currentIndexChanged.connect(
            lambda _i: self.override_changed.emit()
        )

    # ------------------------------------------------------------------
    # Внутренние обработчики
    # ------------------------------------------------------------------
    def _on_train_resize_mode_changed(self, text):
        """
        Обновляет логический флаг _show_inference_resize и видимость поля
        inference_resize_mode. Флаг НЕ зависит от Qt-видимости, поэтому
        корректно работает даже когда виджет ещё не отрисован (например,
        при загрузке модели до первого показа окна).
        """
        show = (text == 'random_crop')
        self._show_inference_resize = show
        self._inference_resize_label.setVisible(show)
        self._inference_resize_override.setVisible(show)
        self.override_changed.emit()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель U‑Net", "", "PyTorch models (*.pth)"
        )
        if path:
            self._model_path.setText(path)

    # ------------------------------------------------------------------
    # Публичное API для окна инференса
    # ------------------------------------------------------------------
    def apply_metadata(self, metadata):
        """
        Применяет метаданные модели к UI.
        Возвращает dict с диагностикой:
            {
                'missing': [список отсутствующих параметров],
                'train_resize_mode': str или None,
                'inference_resize_mode': str или None,
            }
        """
        result = {
            'missing': [],
            'train_resize_mode': None,
            'inference_resize_mode': None,
        }

        # --- num_classes ---
        if 'num_classes' in metadata:
            try:
                self._num_classes_override.setValue(int(metadata['num_classes']))
            except (TypeError, ValueError):
                pass
            self._num_classes_override.setEnabled(False)
        else:
            self._num_classes_override.setEnabled(True)
            result['missing'].append('num_classes')

        # --- normalize ---
        if 'normalize' in metadata:
            self._normalize_override.setChecked(bool(metadata['normalize']))
            self._normalize_override.setEnabled(False)
        else:
            self._normalize_override.setChecked(False)
            self._normalize_override.setEnabled(True)
            result['missing'].append('normalize')

        # --- resize_mode (с backward-compat fallback) ---
        train_rm = None
        if 'resize_mode' in metadata and metadata['resize_mode'] in (
            'hard', 'keep_aspect_pad', 'random_crop'
        ):
            train_rm = metadata['resize_mode']
            idx = self._resize_mode_override.findText(train_rm)
            if idx >= 0:
                self._resize_mode_override.setCurrentIndex(idx)
            self._resize_mode_override.setEnabled(False)
        elif 'keep_aspect_ratio' in metadata:
            train_rm = 'keep_aspect_pad' if metadata['keep_aspect_ratio'] else 'hard'
            idx = self._resize_mode_override.findText(train_rm)
            if idx >= 0:
                self._resize_mode_override.setCurrentIndex(idx)
            self._resize_mode_override.setEnabled(False)
        else:
            self._resize_mode_override.setCurrentIndex(0)  # hard
            self._resize_mode_override.setEnabled(True)
            result['missing'].append('resize_mode')


        result['train_resize_mode'] = train_rm


        # --- Архитектура ---
        if 'architecture' in metadata:
            arch = metadata['architecture'].upper()
            self._architecture.setText(arch)
        else:
            self._architecture.setText("U-Net (по умолчанию)")


        # --- inference_resize_mode ---
        # 1) Если в meta сохранён явный выбор — используем его.
        # 2) Иначе — сбрасываем в дефолт 'keep_aspect_pad',
        #    чтобы не тащить значение от предыдущей модели.
        saved = metadata.get('inference_resize_mode')
        if saved in ('hard', 'keep_aspect_pad'):
            idx = self._inference_resize_override.findText(saved)
            if idx >= 0:
                self._inference_resize_override.setCurrentIndex(idx)
        else:
            idx = self._inference_resize_override.findText('keep_aspect_pad')
            if idx >= 0:
                self._inference_resize_override.setCurrentIndex(idx)

        # Синхронизируем флаг _show_inference_resize и видимость поля.
        # Вызываем явно, потому что setCurrentIndex мог не сгенерировать
        # сигнал currentTextChanged (если индекс совпал с текущим).
        current_train_rm = self._resize_mode_override.currentText()
        # Используем _on_train_resize_mode_changed, но без эмита override_changed
        # (метаданные только что применены — перерисовка не нужна).
        self._show_inference_resize = (current_train_rm == 'random_crop')
        self._inference_resize_label.setVisible(self._show_inference_resize)
        self._inference_resize_override.setVisible(self._show_inference_resize)

        if current_train_rm == 'random_crop':
            result['inference_resize_mode'] = \
                self._inference_resize_override.currentText()
        else:
            result['inference_resize_mode'] = None

        # --- Подсказка ---
        if result['missing']:
            self._override_hint.setText(
                "⚠️ В *_meta.json отсутствуют параметры: "
                + ", ".join(result['missing']) + ".\n"
                "Задайте их вручную — они будут применены при инференсе."
            )
        else:
            self._override_hint.setText("")

        return result

    def get_override_params(self):
        """
        Возвращает dict с АКТИВНЫМИ override-значениями.
        Ключи включаются только если соответствующее поле enabled
        (или, для inference_resize_mode — если установлен логический
        флаг _show_inference_resize, не зависящий от Qt-видимости).
        """
        out = {}
        if self._num_classes_override.isEnabled():
            out['num_classes'] = self._num_classes_override.value()
        if self._normalize_override.isEnabled():
            out['normalize'] = self._normalize_override.isChecked()
        if self._resize_mode_override.isEnabled():
            out['resize_mode'] = self._resize_mode_override.currentText()
        # inference_resize_mode применяется, если train == random_crop
        # (проверяем через логический флаг, а не isVisible())
        if (self._resize_mode_override.currentText() == 'random_crop'
                and self._show_inference_resize):
            out['inference_resize_mode'] = \
                self._inference_resize_override.currentText()
        return out

    def validate_for_prediction(self):
        """
        Проверка перед запуском predict.
        Возвращает (ok: bool, message: str).
        """
        if self._resize_mode_override.currentText() == 'random_crop':
            if not self._show_inference_resize:
                return False, ("Для модели, обученной с random_crop, поле "
                               "«Режим ресайза при ИНФЕРЕНСЕ» должно быть "
                               "доступно.")
            val = self._inference_resize_override.currentText()
            if val not in ('hard', 'keep_aspect_pad'):
                return False, ("Выберите режим ресайза при инференсе:\n"
                               "  • keep_aspect_pad\n  • hard")
        return True, ""

    def get_params(self):
        return {
            'model_path': self._model_path.text(),
            'threshold': self._threshold.value(),
        }



# =============================================================================
# DeepLabV3 / DeepLabV3+
# =============================================================================
class DeepLabV3InferenceSettings(BaseInferenceSettings):
    """
    Настройки инференса для DeepLabV3 и DeepLabV3+ (segmentation_models_pytorch).

    Отличие от U-Net-виджета: backbone и output_stride задаются воркером
    и читаются из *_meta.json, поэтому здесь они read-only. Но добавлены
    override-поля на случай, если в метаданных их нет.
    """
    model_path_changed = pyqtSignal(str)
    architecture_changed = pyqtSignal(str)
    backbone_changed = pyqtSignal(str)
    output_stride_changed = pyqtSignal(int)
    threshold_changed = pyqtSignal(float)
    override_changed = pyqtSignal()

    def __init__(self, parent=None):
        self._show_inference_resize = False
        super().__init__("Параметры DeepLabV3/V3+", parent)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ---------- Файл модели ----------
        path_layout = QHBoxLayout()
        self._model_path = QLineEdit()
        self._model_path.setPlaceholderText("Выберите файл .pth")
        self._browse_btn = QPushButton("Обзор")
        path_layout.addWidget(self._model_path)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

        # ---------- Архитектура (read-only) ----------
        self._architecture = QLineEdit()
        self._architecture.setReadOnly(True)
        self._architecture.setPlaceholderText("Читается из метаданных модели")
        layout.addLayout(self._row("Архитектура", self._architecture))

        # ---------- Backbone ----------
        self._backbone = QComboBox()
        self._backbone.addItems(["resnet50", "resnet101", "mobilenet_v3", "xception"])
        layout.addLayout(self._row("Backbone", self._backbone))

        # ---------- Output stride ----------
        self._output_stride = QComboBox()
        self._output_stride.addItems(["8", "16", "32"])
        self._output_stride.setCurrentText("16")
        layout.addLayout(self._row("Output stride", self._output_stride))

        # ---------- Input size (read-only — читается из meta) ----------
        self._input_size = QLineEdit()
        self._input_size.setReadOnly(True)
        self._input_size.setPlaceholderText("Читается из метаданных модели")
        layout.addLayout(self._row("Размер входа", self._input_size))

        # ---------- Output activation (read-only) ----------
        self._output_activation = QLineEdit()
        self._output_activation.setReadOnly(True)
        self._output_activation.setPlaceholderText("Читается из метаданных модели")
        layout.addLayout(self._row("Активация", self._output_activation))

        # ---------- Threshold ----------
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.01, 0.99)
        self._threshold.setSingleStep(0.01)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.5)
        self._threshold.setToolTip(
            "Порог бинаризации для получения маски (0.5 – стандарт).\n"
            "Применяется для num_classes=1 (sigmoid) и num_classes=2 "
            "(softmax[:, 1]). Для num_classes>2 используется argmax."
        )
        layout.addLayout(self._row("Threshold", self._threshold))

        # ---------- Override-параметры ----------
        ov_group = QGroupBox("Override параметров инференса (если нет в meta)")
        ov_form = QFormLayout()
        ov_form.setSpacing(8)
        ov_group.setLayout(ov_form)

        self._num_classes_override = QSpinBox()
        self._num_classes_override.setRange(1, 100)
        self._num_classes_override.setValue(21)
        self._num_classes_override.setEnabled(False)
        self._num_classes_override.setToolTip(
            "Общее число классов, включая фон (фон = 0).\n\n"
            "  • 1 → бинарная (sigmoid + threshold);\n"
            "  • 2 → «фон + один объект»: softmax[:, 1] + threshold;\n"
            "  • >2 → multiclass (argmax, threshold не используется).\n\n"
            "Активно только если параметра нет в *_meta.json."
        )
        ov_form.addRow("num_classes:", self._num_classes_override)

        self._normalize_override = QCheckBox("Применять ImageNet-нормализацию")
        self._normalize_override.setChecked(False)
        self._normalize_override.setEnabled(False)
        self._normalize_override.setToolTip(
            "Только если модель обучалась с нормализацией. Иначе предсказания "
            "будут некорректны."
        )
        ov_form.addRow("Нормализация:", self._normalize_override)

        self._resize_mode_override = QComboBox()
        self._resize_mode_override.addItems(
            ["hard", "keep_aspect_pad", "random_crop"]
        )
        self._resize_mode_override.setEnabled(False)
        self._resize_mode_override.setToolTip(
            "Режим, использованный при ОБУЧЕНИИ:\n"
            "  • hard — жёсткий ресайз;\n"
            "  • keep_aspect_pad — AR + padding;\n"
            "  • random_crop — случайный кроп (только обучение).\n\n"
            "Для DeepLab обычно используется hard."
        )
        ov_form.addRow("resize_mode (train):", self._resize_mode_override)

        self._inference_resize_label = QLabel("Режим ресайза при ИНФЕРЕНСЕ:")
        self._inference_resize_override = QComboBox()
        self._inference_resize_override.addItems(["keep_aspect_pad", "hard"])
        self._inference_resize_override.setCurrentText("hard")
        self._inference_resize_override.setToolTip(
            "Как приводить кадр к input_size при инференсе.\n"
            "Используется только если обучение шло с random_crop."
        )
        self._inference_resize_label.setVisible(False)
        self._inference_resize_override.setVisible(False)
        ov_form.addRow(self._inference_resize_label,
                       self._inference_resize_override)

        self._override_hint = QLabel("")
        self._override_hint.setWordWrap(True)
        self._override_hint.setStyleSheet(
            "color: #b8860b; font-size: 11px; padding: 4px;"
        )
        ov_form.addRow(self._override_hint)

        layout.addWidget(ov_group)

        # ---------- Сигналы ----------
        self._model_path.textChanged.connect(self.model_path_changed.emit)
        self._browse_btn.clicked.connect(self._browse_file)
        self._backbone.currentTextChanged.connect(self.backbone_changed.emit)
        self._output_stride.currentTextChanged.connect(
            lambda v: self.output_stride_changed.emit(int(v))
        )
        self._threshold.valueChanged.connect(self.threshold_changed.emit)

        self._num_classes_override.valueChanged.connect(
            lambda _v: self.override_changed.emit()
        )
        self._normalize_override.stateChanged.connect(
            lambda _s: self.override_changed.emit()
        )
        self._resize_mode_override.currentTextChanged.connect(
            self._on_train_resize_mode_changed
        )
        self._inference_resize_override.currentIndexChanged.connect(
            lambda _i: self.override_changed.emit()
        )

    def _on_train_resize_mode_changed(self, text):
        show = (text == 'random_crop')
        self._show_inference_resize = show
        self._inference_resize_label.setVisible(show)
        self._inference_resize_override.setVisible(show)
        self.override_changed.emit()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель DeepLabV3/V3+", "", "PyTorch models (*.pth)"
        )
        if path:
            self._model_path.setText(path)

    # ------------------------------------------------------------------
    # Публичное API
    # ------------------------------------------------------------------
    def apply_metadata(self, metadata):
        """
        Заполняет поля из *_meta.json.
        Возвращает dict диагностики:
            {
                'missing': [...],
                'train_resize_mode': str | None,
                'inference_resize_mode': str | None,
                'architecture': str | None,
            }
        """
        result = {
            'missing': [],
            'train_resize_mode': None,
            'inference_resize_mode': None,
            'architecture': None,
        }

        # --- Архитектура ---
        if 'architecture' in metadata:
            arch = str(metadata['architecture']).lower()
            if arch in ('deeplabv3plus', 'deeplabv3+'):
                self._architecture.setText("DeepLabV3+")
            elif arch == 'deeplabv3':
                self._architecture.setText("DeepLabV3")
            else:
                self._architecture.setText(arch.upper())
            result['architecture'] = arch
        else:
            self._architecture.setText("—")

        # --- Backbone ---
        if 'backbone' in metadata:
            idx = self._backbone.findText(str(metadata['backbone']))
            if idx >= 0:
                self._backbone.setCurrentIndex(idx)
            self._backbone.setEnabled(False)
        else:
            self._backbone.setEnabled(True)
            result['missing'].append('backbone')

        # --- Output stride ---
        if 'output_stride' in metadata:
            os_text = str(metadata['output_stride'])
            idx = self._output_stride.findText(os_text)
            if idx >= 0:
                self._output_stride.setCurrentIndex(idx)
            self._output_stride.setEnabled(False)
        else:
            self._output_stride.setEnabled(True)
            result['missing'].append('output_stride')

        # --- Input size ---
        if 'input_size' in metadata:
            self._input_size.setText(str(metadata['input_size']))
        else:
            self._input_size.setText("—")
            result['missing'].append('input_size')

        # --- Output activation ---
        if 'output_activation' in metadata:
            self._output_activation.setText(str(metadata['output_activation']))
        else:
            # Дефолт по числу классов
            nc = metadata.get('num_classes', 21)
            try:
                nc_int = int(nc)
            except (TypeError, ValueError):
                nc_int = 21
            if nc_int <= 2:
                self._output_activation.setText("sigmoid (по num_classes)")
            else:
                self._output_activation.setText("argmax (по num_classes)")

        # --- num_classes ---
        if 'num_classes' in metadata:
            try:
                self._num_classes_override.setValue(int(metadata['num_classes']))
            except (TypeError, ValueError):
                pass
            self._num_classes_override.setEnabled(False)
        else:
            self._num_classes_override.setEnabled(True)
            result['missing'].append('num_classes')

        # --- normalize ---
        if 'normalize' in metadata:
            self._normalize_override.setChecked(bool(metadata['normalize']))
            self._normalize_override.setEnabled(False)
        else:
            self._normalize_override.setChecked(False)
            self._normalize_override.setEnabled(True)
            result['missing'].append('normalize')

        # --- resize_mode (с backward-compat) ---
        train_rm = None
        if 'resize_mode' in metadata and metadata['resize_mode'] in (
            'hard', 'keep_aspect_pad', 'random_crop'
        ):
            train_rm = metadata['resize_mode']
            idx = self._resize_mode_override.findText(train_rm)
            if idx >= 0:
                self._resize_mode_override.setCurrentIndex(idx)
            self._resize_mode_override.setEnabled(False)
        elif 'keep_aspect_ratio' in metadata:
            train_rm = 'keep_aspect_pad' if metadata['keep_aspect_ratio'] else 'hard'
            idx = self._resize_mode_override.findText(train_rm)
            if idx >= 0:
                self._resize_mode_override.setCurrentIndex(idx)
            self._resize_mode_override.setEnabled(False)
        else:
            self._resize_mode_override.setCurrentIndex(0)  # hard
            self._resize_mode_override.setEnabled(True)
            result['missing'].append('resize_mode')

        result['train_resize_mode'] = train_rm

        # --- inference_resize_mode ---
        saved = metadata.get('inference_resize_mode')
        if saved in ('hard', 'keep_aspect_pad'):
            idx = self._inference_resize_override.findText(saved)
            if idx >= 0:
                self._inference_resize_override.setCurrentIndex(idx)
        else:
            # для DeepLab безопасный дефолт — hard
            idx = self._inference_resize_override.findText('hard')
            if idx >= 0:
                self._inference_resize_override.setCurrentIndex(idx)

        current_train_rm = self._resize_mode_override.currentText()
        self._show_inference_resize = (current_train_rm == 'random_crop')
        self._inference_resize_label.setVisible(self._show_inference_resize)
        self._inference_resize_override.setVisible(self._show_inference_resize)

        if current_train_rm == 'random_crop':
            result['inference_resize_mode'] = \
                self._inference_resize_override.currentText()
        else:
            result['inference_resize_mode'] = None

        # --- Подсказка ---
        if result['missing']:
            self._override_hint.setText(
                "⚠️ В *_meta.json отсутствуют параметры: "
                + ", ".join(result['missing']) + ".\n"
                "Задайте их вручную — они будут применены при инференсе."
            )
        else:
            self._override_hint.setText("")

        return result

    def get_override_params(self):
        """
        Возвращает dict с АКТИВНЫМИ override-значениями.
        Ключи включаются только если соответствующее поле enabled
        (или, для inference_resize_mode — если установлен логический
        флаг _show_inference_resize).
        """
        out = {}
        if self._num_classes_override.isEnabled():
            out['num_classes'] = self._num_classes_override.value()
        if self._normalize_override.isEnabled():
            out['normalize'] = self._normalize_override.isChecked()
        if self._resize_mode_override.isEnabled():
            out['resize_mode'] = self._resize_mode_override.currentText()
        if (self._resize_mode_override.currentText() == 'random_crop'
                and self._show_inference_resize):
            out['inference_resize_mode'] = \
                self._inference_resize_override.currentText()
        if self._backbone.isEnabled():
            out['backbone'] = self._backbone.currentText()
        if self._output_stride.isEnabled():
            try:
                out['output_stride'] = int(self._output_stride.currentText())
            except (TypeError, ValueError):
                out['output_stride'] = 16
        return out

    def validate_for_prediction(self):
        """
        Проверка перед запуском predict.
        Возвращает (ok: bool, message: str).
        """
        if self._resize_mode_override.currentText() == 'random_crop':
            if not self._show_inference_resize:
                return False, ("Для модели, обученной с random_crop, поле "
                               "«Режим ресайза при ИНФЕРЕНСЕ» должно быть "
                               "доступно.")
            val = self._inference_resize_override.currentText()
            if val not in ('hard', 'keep_aspect_pad'):
                return False, ("Выберите режим ресайза при инференсе:\n"
                               "  • hard\n  • keep_aspect_pad")
        if not self._num_classes_override.isEnabled():
            # значение известное, ок
            pass
        elif self._num_classes_override.value() < 1:
            return False, "num_classes должно быть ≥ 1."
        return True, ""

    def get_params(self):
        return {
            'model_path': self._model_path.text(),
            'threshold': self._threshold.value(),
        }




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
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель SegFormer", "", "PyTorch models (*.pth)"
        )
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель SAM", "", "PyTorch models (*.pth)"
        )
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите ONNX модель", "", "ONNX files (*.onnx)"
        )
        if path:
            self._model_path.setText(path)

    def get_params(self):
        return {'model_path': self._model_path.text()}