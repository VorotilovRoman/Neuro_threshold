# widgets/DL_train_settings_base.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
import os
import sys


class BaseTrainWidget(QWidget):
    run_training = pyqtSignal(dict)
    stop_training = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent_window = parent
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.form_layout.setSpacing(12)
        self.layout.addWidget(self.form_widget)
        self.layout.addStretch()

    def add_dataset_structure_hint(self, hint_text):
        hint_widget = QLabel(hint_text)
        hint_widget.setWordWrap(True)
        #hint_widget.setStyleSheet("background-color: #f0f0f0; padding: 8px; border-radius: 4px; font-family: monospace;")
        self.form_layout.addRow(hint_widget)

    def add_experiment_name_field(self):
        self.experiment_name = QLineEdit()
        self.experiment_name.setPlaceholderText("Название подпапки (опционально)")
        self.form_layout.addRow("Имя эксперимента:", self.experiment_name)

    def add_project_field(self):
        self.project_path = QLineEdit()
        self.project_path.setPlaceholderText("Куда сохранить результаты (модели, логи)")
        self.browse_project_btn = QPushButton("Обзор")
        self.browse_project_btn.clicked.connect(self._browse_project)
        row = QHBoxLayout()
        row.addWidget(self.project_path)
        row.addWidget(self.browse_project_btn)
        self.form_layout.addRow("📁 Папка проекта:", row)

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения результатов")
        if path:
            self.project_path.setText(path)

    def add_device_field(self):
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "0", "cpu", "mps"])
        self.form_layout.addRow("Устройство (device):", self.device_combo)

    def add_common_training_params(self):
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        self.form_layout.addRow("epochs:", self.epochs)

        batch_layout = QHBoxLayout()
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(16)
        self.auto_batch = QCheckBox("Auto batch")
        self.auto_batch.setChecked(True)
        batch_layout.addWidget(self.batch)
        batch_layout.addWidget(self.auto_batch)
        self.form_layout.addRow("batch:", batch_layout)
        self.auto_batch.toggled.connect(self.batch.setDisabled)
        self.batch.setDisabled(self.auto_batch.isChecked())

    def add_buttons(self):
        self.buttons_widget = QWidget()
        btn_layout = QHBoxLayout(self.buttons_widget)
        self.run_btn = QPushButton("Запустить обучение")
        self.stop_btn = QPushButton("Остановить обучение")
        self.stop_btn.setEnabled(False)
        self.open_folder_btn = QPushButton("Открыть папку проекта")
        # Кнопка активна, если указан путь к папке проекта
        self.open_folder_btn.setEnabled(bool(self.project_path.text().strip()))
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addStretch()
        self.form_layout.addRow(self.buttons_widget)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.form_layout.addRow(self.status_label)

        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_folder_btn.clicked.connect(self._open_project_folder)

        # Подключаем обновление состояния кнопки при изменении пути
        self.project_path.textChanged.connect(self._update_open_folder_button_state)

    def _update_open_folder_button_state(self):
        """Включает/отключает кнопку открытия папки в зависимости от наличия пути."""
        if hasattr(self, 'open_folder_btn'):
            self.open_folder_btn.setEnabled(bool(self.project_path.text().strip()))

    def add_progress_bar(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Готов")
        self.progress_bar.setValue(0)
        self.form_layout.addRow("Прогресс:", self.progress_bar)

    def _on_run(self):
        if hasattr(self, 'dataset_path') and self.dataset_path.text().strip():
            if not os.path.exists(self.dataset_path.text().strip()):
                QMessageBox.warning(self, "Ошибка", "Папка датасета не существует")
                return
        params = self.get_params()
        if params is not None:
            self.run_training.emit(params)

    def _on_stop(self):
        self.stop_training.emit()

    def _open_project_folder(self):
        folder = self.project_path.text().strip()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "Ошибка", "Папка проекта не выбрана или не существует")
            return
        if sys.platform == 'win32':
            os.startfile(folder)
        elif sys.platform == 'darwin':
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    def set_running(self, running):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if not running:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Готов")

    def get_params(self):
        raise NotImplementedError


# =============================================================================
# YOLO Training Widget
# =============================================================================
class YOLOTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        hint = (
            "<b>📁 Структура датасета YOLO:</b><br>"
            "dataset/<br>"
            "├── images/<br>"
            "│   ├── train/<br>"
            "│   └── val/<br>"
            "├── labels/<br>"
            "│   ├── train/<br>"
            "│   └── val/<br>"
            "└── data.yaml"
        )
        self.add_dataset_structure_hint(hint)

        # Выбор датасета: папка или YAML
        self.dataset_folder = QLineEdit()
        self.dataset_folder.setPlaceholderText("Корневая папка датасета")
        self.browse_folder_btn = QPushButton("Обзор папки")
        self.browse_folder_btn.clicked.connect(lambda: self._browse_dataset_folder())
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.dataset_folder)
        folder_row.addWidget(self.browse_folder_btn)
        self.form_layout.addRow("Папка датасета:", folder_row)

        self.dataset_yaml = QLineEdit()
        self.dataset_yaml.setPlaceholderText("Или YAML-файл датасета")
        self.browse_yaml_btn = QPushButton("Обзор YAML")
        self.browse_yaml_btn.clicked.connect(lambda: self._browse_dataset_yaml())
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.dataset_yaml)
        yaml_row.addWidget(self.browse_yaml_btn)
        self.form_layout.addRow("YAML файл:", yaml_row)

        self.dataset_folder.textChanged.connect(lambda: self.dataset_yaml.clear() if self.dataset_folder.text() else None)
        self.dataset_yaml.textChanged.connect(lambda: self.dataset_folder.clear() if self.dataset_yaml.text() else None)

        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()
        self._add_model_selection()
        self._add_yolo_specific()
        self.add_buttons()
        self.add_progress_bar()
        self._update_preview()

    def _browse_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите корневую папку датасета")
        if path:
            self.dataset_folder.setText(path)
            self.dataset_yaml.clear()

    def _browse_dataset_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите YAML-файл датасета", "", "YAML files (*.yaml *.yml)")
        if path:
            self.dataset_yaml.setText(path)
            self.dataset_folder.clear()

    def _add_model_selection(self):
        group = QGroupBox("Источник модели")
        layout = QVBoxLayout()
        self.model_from_preset = QRadioButton("Использовать предустановленную YOLO")
        self.model_from_file = QRadioButton("Загрузить свою модель (.pt)")
        self.model_from_preset.setChecked(True)
        layout.addWidget(self.model_from_preset)
        layout.addWidget(self.model_from_file)
        group.setLayout(layout)
        self.form_layout.addRow(group)

        self.preset_widget = QWidget()
        self.preset_layout = QFormLayout(self.preset_widget)
        self.form_layout.addRow(self.preset_widget)

        self.file_widget = QWidget()
        file_row = QHBoxLayout(self.file_widget)
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setPlaceholderText("Путь к .pt файлу")
        self.browse_custom_btn = QPushButton("Обзор")
        file_row.addWidget(self.custom_model_path)
        file_row.addWidget(self.browse_custom_btn)
        self.form_layout.addRow(self.file_widget)

        self.preset_widget.setVisible(True)
        self.file_widget.setVisible(False)

        self.model_from_preset.toggled.connect(self._toggle_model_source)
        self.model_from_file.toggled.connect(self._toggle_model_source)
        self.browse_custom_btn.clicked.connect(self._browse_custom_model)

        self._add_preset_yolo_fields()

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель YOLO", "", "PyTorch models (*.pt)")
        if path:
            self.custom_model_path.setText(path)

    def _add_preset_yolo_fields(self):
        self.yolo_version = QComboBox()
        self.yolo_version.addItems(["8", "11", "26"])
        self.preset_layout.addRow("Версия YOLO:", self.yolo_version)

        self.yolo_task = QComboBox()
        self.yolo_task.addItems(["Detection", "Segmentation", "OBB"])
        self.preset_layout.addRow("Тип модели:", self.yolo_task)

        self.yolo_size = QComboBox()
        self.yolo_size.addItems(["n", "s", "m", "l", "x"])
        self.preset_layout.addRow("Размер:", self.yolo_size)

        self.preview_model_name = QLineEdit()
        self.preview_model_name.setReadOnly(True)
        self.preset_layout.addRow("Модель:", self.preview_model_name)

        self.yolo_version.currentTextChanged.connect(self._update_preview)
        self.yolo_task.currentTextChanged.connect(self._update_preview)
        self.yolo_size.currentTextChanged.connect(self._update_preview)

    def _update_preview(self):
        ver = self.yolo_version.currentText()
        task = self.yolo_task.currentText()
        size = self.yolo_size.currentText()
        prefix = "yolov8" if ver == "8" else f"yolo{ver}"
        if task == "Detection":
            suffix = ".pt"
        elif task == "Segmentation":
            suffix = "-seg.pt"
        else:  # OBB
            suffix = "-obb.pt"
        model = f"{prefix}{size}{suffix}"
        self.preview_model_name.setText(model)

    def _add_yolo_specific(self):
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 1280)
        self.imgsz.setValue(640)
        self.imgsz.setSingleStep(32)
        self.form_layout.addRow("Размер входа (imgsz):", self.imgsz)

        self.enable_mosaic = QCheckBox("Enable Mosaic")
        self.enable_mosaic.setChecked(True)
        self.form_layout.addRow(self.enable_mosaic)

        self.close_mosaic = QSpinBox()
        self.close_mosaic.setRange(0, 100)
        self.close_mosaic.setValue(10)
        self.mosaic = QDoubleSpinBox()
        self.mosaic.setRange(0.0, 1.0)
        self.mosaic.setValue(1.0)
        self.enable_mosaic.toggled.connect(self.close_mosaic.setEnabled)
        self.enable_mosaic.toggled.connect(self.mosaic.setEnabled)
        self.form_layout.addRow("close_mosaic (epochs):", self.close_mosaic)
        self.form_layout.addRow("mosaic (probability):", self.mosaic)

        self.disable_aug = QCheckBox("Disable default augmentations")
        self.disable_aug.setChecked(True)
        self.form_layout.addRow(self.disable_aug)

        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.0001, 0.1)
        self.lr0.setSingleStep(0.001)
        self.lr0.setDecimals(4)
        self.lr0.setValue(0.01)
        self.form_layout.addRow("lr0:", self.lr0)

        self.lrf = QDoubleSpinBox()
        self.lrf.setRange(0.0001, 0.1)
        self.lrf.setSingleStep(0.001)
        self.lrf.setDecimals(4)
        self.lrf.setValue(0.01)
        self.form_layout.addRow("lrf:", self.lrf)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["auto", "SGD", "AdamW", "MuSGD", "Adam", "NAdam", "RMSProp"])
        self.form_layout.addRow("optimizer:", self.optimizer)

    def get_params(self):
        if self.model_from_preset.isChecked():
            ver = self.yolo_version.currentText()
            task = self.yolo_task.currentText()
            size = self.yolo_size.currentText()
            prefix = "yolov8" if ver == "8" else f"yolo{ver}"
            if task == "Detection":
                suffix = ".pt"
            elif task == "Segmentation":
                suffix = "-seg.pt"
            else:  # OBB
                suffix = "-obb.pt"
            model_path = f"{prefix}{size}{suffix}"
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан путь к модели YOLO")
                return None

        data_path = self.dataset_yaml.text().strip()
        if not data_path:
            data_path = self.dataset_folder.text().strip()
            if data_path:
                data_path = os.path.join(data_path, "data.yaml")

        if not data_path:
            QMessageBox.warning(self, "Ошибка", "Не указан датасет (папка или YAML-файл)")
            return None

        if not os.path.exists(data_path):
            QMessageBox.warning(self, "Ошибка", f"YAML-файл датасета не найден:\n{data_path}")
            return None

        return {
            "model": model_path,
            "data": data_path,
            "project": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": self.epochs.value(),
            "imgsz": self.imgsz.value(),
            "batch": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),
            "optimizer": self.optimizer.currentText(),
            "lr0": self.lr0.value(),
            "lrf": self.lrf.value(),
            "disable_aug": self.disable_aug.isChecked(),
            "close_mosaic": self.close_mosaic.value() if self.enable_mosaic.isChecked() else 0,
            "mosaic": self.mosaic.value() if self.enable_mosaic.isChecked() else 0.0,
        }


# =============================================================================
# U-Net Training Widget
# =============================================================================
# =============================================================================
# U-Net Training Widget
# =============================================================================
class UNetTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        hint = (
            "<b>📁 Структура датасета U‑Net:</b><br>"
            "dataset/<br>"
            "├── images/<br>"
            "│   ├── train/<br>"
            "│   └── val/<br>"
            "└── masks/<br>"
            "    ├── train/<br>"
            "    └── val/<br>"
            "Имена файлов изображений и масок совпадают.<br><br>"
            "<b>Значения пикселей в маске:</b><br>"
            "• 0 — фон (всегда);<br>"
            "• 1, 2, 3, … — индексы классов объектов.<br><br>"
            "<b>«Количество классов» в форме — ВКЛЮЧАЯ фон.</b><br>"
            "Пример: фон + 1 объект → 2; фон + 3 объекта → 4."
        )
        self.add_dataset_structure_hint(hint)

        # --- Папка датасета ---
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Корневая папка датасета (содержит images/ и masks/)")
        self.browse_dataset_btn = QPushButton("Обзор")
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_path)
        row.addWidget(self.browse_dataset_btn)
        self.form_layout.addRow("📁 Папка датасета:", row)

        # --- NEW: U-Net YAML (masks_data.yaml) ---
        # Если указан — воркер читает path/num_classes/class_names из него.
        # Поля «Папка датасета» и «Количество классов» заполняются автоматически.
        self.unet_yaml = QLineEdit()
        self.unet_yaml.setPlaceholderText(
            "Или masks_data.yaml — переопределит path и num_classes"
        )
        self.browse_yaml_btn = QPushButton("Обзор YAML")
        self.browse_yaml_btn.clicked.connect(self._browse_unet_yaml)
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.unet_yaml)
        yaml_row.addWidget(self.browse_yaml_btn)
        self.form_layout.addRow("📄 U-Net YAML:", yaml_row)

        yaml_hint = QLabel(
            "Если указан YAML — его поля <code>path</code> и <code>num_classes</code> "
            "переопределяют значения из этой формы. Удобно, когда датасет "
            "сгенерирован утилитой и уже имеет готовый <code>masks_data.yaml</code>."
        )
        yaml_hint.setWordWrap(True)
        yaml_hint.setStyleSheet("color: #666; font-size: 11px;")
        self.form_layout.addRow(yaml_hint)

        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()
        self._add_model_selection()
        self._add_unet_specific()
        self.add_buttons()
        self.add_progress_bar()

        # Реагируем на ручное изменение поля YAML
        self.unet_yaml.textChanged.connect(self._on_unet_yaml_changed)

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите корневую папку датасета")
        if path:
            self.dataset_path.setText(path)
            self.unet_yaml.clear()

    def _browse_unet_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите masks_data.yaml", "",
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            self.unet_yaml.setText(path)
            # Авто-заполнение полей ниже — через сигнал textChanged

    def _on_unet_yaml_changed(self, text):
        """Автозаполнение dataset_path и num_classes из YAML."""
        path = (text or "").strip()
        if not path or not os.path.isfile(path):
            return
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            # Тихо игнорируем — детальная валидация будет в _validate_unet_params
            print(f"[U-Net YAML] не удалось прочитать {path}: {e}")
            return

        # 1. path → dataset_path
        root = cfg.get('path', '')
        if root:
            if not os.path.isabs(root):
                root = os.path.normpath(os.path.join(os.path.dirname(path), root))
            # Не перезаписываем, если поле уже совпадает (чтобы не мигало)
            if self.dataset_path.text().strip() != root:
                self.dataset_path.setText(root)

        # 2. num_classes → спинбокс (только если валидно и >= 2)
        nc = cfg.get('num_classes')
        try:
            nc_int = int(nc) if nc is not None else None
        except (ValueError, TypeError):
            nc_int = None
        if nc_int is not None and nc_int >= 2 and self.num_classes.value() != nc_int:
            self.num_classes.setValue(nc_int)

    def _add_model_selection(self):
        group = QGroupBox("Источник модели")
        layout = QVBoxLayout()
        self.model_from_preset = QRadioButton("Использовать предустановленную U-Net (backbone)")
        self.model_from_file = QRadioButton("Загрузить свою модель (.pth)")
        self.model_from_preset.setChecked(True)
        layout.addWidget(self.model_from_preset)
        layout.addWidget(self.model_from_file)
        group.setLayout(layout)
        self.form_layout.addRow(group)

        self.preset_widget = QWidget()
        self.preset_layout = QFormLayout(self.preset_widget)
        self.form_layout.addRow(self.preset_widget)

        self.file_widget = QWidget()
        file_row = QHBoxLayout(self.file_widget)
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setPlaceholderText("Путь к .pth файлу")
        self.browse_custom_btn = QPushButton("Обзор")
        file_row.addWidget(self.custom_model_path)
        file_row.addWidget(self.browse_custom_btn)
        self.form_layout.addRow(self.file_widget)

        self.preset_widget.setVisible(True)
        self.file_widget.setVisible(False)

        self.model_from_preset.toggled.connect(self._toggle_model_source)
        self.model_from_file.toggled.connect(self._toggle_model_source)
        self.browse_custom_btn.clicked.connect(self._browse_custom_model)

        self._add_preset_unet_fields()

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель U-Net", "", "PyTorch models (*.pth)")
        if path:
            self.custom_model_path.setText(path)

    def _add_preset_unet_fields(self):
        self.encoder = QComboBox()
        self.encoder.addItems(["resnet18", "resnet34", "resnet50", "efficientnet-b0", "vgg16"])
        self.preset_layout.addRow("Энкодер (backbone):", self.encoder)

        self.input_size = QComboBox()
        self.input_size.addItems(["256", "512", "1024"])
        self.input_size.setCurrentText("512")
        self.preset_layout.addRow("Размер входа:", self.input_size)

    def _add_unet_specific(self):
        # --- Количество классов (включая фон) ---
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 100)
        self.num_classes.setValue(2)
        self.num_classes.setToolTip(
            "Общее число классов, ВКЛЮЧАЯ фон.\n"
            "• 0 — всегда фон;\n"
            "• 1, 2, … — индексы классов объектов в маске.\n\n"
            "Примеры:\n"
            "  фон + 1 объект   → 2\n"
            "  фон + 3 объекта → 4\n\n"
            "Если указан U-Net YAML — это поле будет перезаписано "
            "значением num_classes из YAML."
        )
        self.form_layout.addRow("Количество классов (вкл. фон):", self.num_classes)

        # --- Функция потерь (BCE+Dice по умолчанию) ---
        self.loss = QComboBox()
        self.loss.addItems(["BCE+Dice", "BCE", "Dice", "Focal", "Tversky"])
        self.loss.setToolTip(
            "Функция потерь.\n"
            "«BCE+Dice» — рекомендуется для сегментации по умолчанию:\n"
            "  BCE даёт стабильные градиенты, Dice оптимизирует IoU."
        )
        self.form_layout.addRow("Функция потерь:", self.loss)

        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.5)
        self.dropout.setSingleStep(0.05)
        self.dropout.setValue(0.0)
        self.form_layout.addRow("Dropout:", self.dropout)

        self.scheduler = QComboBox()
        self.scheduler.addItems(["None", "ReduceLROnPlateau", "CosineAnnealing", "StepLR"])
        self.scheduler.setToolTip(
            "Планировщик learning rate.\n"
            "«ReduceLROnPlateau» — самый безопасный вариант:\n"
            "снижает LR, когда val_loss перестаёт падать."
        )
        self.form_layout.addRow("LR Scheduler:", self.scheduler)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 0.1)
        self.lr.setSingleStep(0.0001)
        self.lr.setDecimals(5)
        self.lr.setValue(0.0001)
        self.form_layout.addRow("Learning rate:", self.lr)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.form_layout.addRow("optimizer:", self.optimizer)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        self.threshold.setToolTip("Порог для бинаризации предсказаний при вычислении метрик (0.5 – стандарт)")
        self.form_layout.addRow("Порог (threshold):", self.threshold)

        # --- Аугментации (по умолчанию выключены) ---
        aug_group = QGroupBox("Аугментации")
        aug_layout = QVBoxLayout(aug_group)
        self.aug_hflip = QCheckBox("Горизонтальное отражение")
        self.aug_hflip.setChecked(False)
        self.aug_vflip = QCheckBox("Вертикальное отражение")
        self.aug_vflip.setChecked(False)
        self.aug_rotate = QCheckBox("Поворот (±10°)")
        self.aug_rotate.setChecked(False)
        self.aug_scale = QCheckBox("Масштабирование")
        self.aug_scale.setChecked(False)
        self.aug_noise = QCheckBox("Гауссов шум")
        self.aug_noise.setChecked(False)
        aug_layout.addWidget(self.aug_hflip)
        aug_layout.addWidget(self.aug_vflip)
        aug_layout.addWidget(self.aug_rotate)
        aug_layout.addWidget(self.aug_scale)
        aug_layout.addWidget(self.aug_noise)

        hint_aug = QLabel(
            "Аугментации применяются синхронно к изображению и маске.\n"
            "Для масок используется интерполяция NEAREST — без размытия границ."
        )
        hint_aug.setWordWrap(True)
        hint_aug.setStyleSheet("color: #666; font-size: 11px;")
        aug_layout.addWidget(hint_aug)

        self.form_layout.addRow(aug_group)

    def get_params(self):
        if self.model_from_preset.isChecked():
            model_path = None
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан файл модели U-Net")
                return None

        dataset_path = self.dataset_path.text().strip()
        unet_yaml = self.unet_yaml.text().strip()

        # Должен быть указан хотя бы один источник: папка или YAML.
        if not dataset_path and not unet_yaml:
            QMessageBox.warning(
                self, "Ошибка",
                "Укажите либо папку датасета, либо U-Net YAML (masks_data.yaml)."
            )
            return None

        if unet_yaml and not os.path.isfile(unet_yaml):
            QMessageBox.warning(
                self, "Ошибка",
                f"U-Net YAML не найден:\n{unet_yaml}"
            )
            return None

        if dataset_path and not os.path.isdir(dataset_path):
            QMessageBox.warning(
                self, "Ошибка",
                f"Папка датасета не существует:\n{dataset_path}"
            )
            return None

        return {
            "dataset_path": dataset_path or None,
            "unet_yaml": unet_yaml or None,          # NEW
            "model_path": model_path,
            "project_path": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": self.epochs.value(),
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "encoder": self.encoder.currentText(),
            "input_size": int(self.input_size.currentText()),
            "num_classes": self.num_classes.value(),
            "loss": self.loss.currentText(),
            "dropout": self.dropout.value(),
            "scheduler": self.scheduler.currentText(),
            "augmentations": {
                "hflip": self.aug_hflip.isChecked(),
                "vflip": self.aug_vflip.isChecked(),
                "rotate": self.aug_rotate.isChecked(),
                "scale": self.aug_scale.isChecked(),
                "noise": self.aug_noise.isChecked(),
            },
            "threshold": self.threshold.value()
        }