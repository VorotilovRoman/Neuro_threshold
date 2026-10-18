# widgets/DL_train_settings_new_model.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
import os
import sys
import ast

from .DL_train_settings_base import BaseTrainWidget


# =============================================================================
# DeepLabV3 / DeepLabV3+ Training Widget
# =============================================================================
class DeepLabV3TrainWidget(BaseTrainWidget):
    HIGH_RES_PRESET = {
        "architecture": "deeplabv3plus",
        "backbone": "resnet101",
        "output_stride": 8,
        "input_size": 1025,
        "lr": 0.001,
        "optimizer": "AdamW",
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "scheduler": "Poly",
        "epochs": 40,
        "batch_size": 4,
        "auto_batch": False,
        "loss": "CrossEntropy",
        "loss_weights": {"ce": 1.0, "dice": 0.0, "focal": 0.0},
        "use_ema": True,
        "ema_decay": 0.999,
        "two_phase_training": False,
        "freeze_epochs": 0,
        "unfreeze_epochs": 0,
        "use_layerwise_lr": True,
        "layerwise_decay": 0.1,
        "poly_power": 0.9,
        "multi_scale_infer": False,
        "infer_scales": [0.75, 1.0, 1.25],
        "freeze_bn_second_phase": True,
        "unfreeze_last_n_stages": -1,
        "normalize": True,
        "resize_mode": "hard",
        "use_amp": True,
        "accum_steps": 1,
        "threshold": 0.5,
        "num_classes": 21,
        "ignore_index": 255,
        "class_weights": None,
        "output_activation": "argmax",
        "augmentations": {
            "hflip": False, "vflip": False,
            "rotate": False, "scale": False, "noise": False,
        },
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # ========== 1. ПРЕСЕТ (В САМОМ ВЕРХУ) ==========
        self._add_preset_controls()

        # ========== 2. Подсказка по структуре датасета ==========
        hint = (
            "<b>📁 Датасет DeepLabV3/V3+ поддерживает 2 структуры:</b><br><br>"
            "<b>A) PASCAL VOC:</b><br>"
            "VOCdevkit/VOC2007/<br>"
            "├── JPEGImages/          # *.jpg<br>"
            "├── SegmentationClass/   # *.png (палитра, 0..N, 255=ignore)<br>"
            "└── ImageSets/Segmentation/train.txt, val.txt<br><br>"
            "<b>B) U-Net-style:</b><br>"
            "dataset/<br>"
            "├── images/train, images/val<br>"
            "└── masks/train,  masks/val<br><br>"
            "Можно указать папку (автоопределение структуры)<br>"
            "или YAML с точными путями (приоритет над папкой)."
        )
        self.add_dataset_structure_hint(hint)

        # ========== 3. Датасет: папка + YAML ==========
        # Папка датасета
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Корневая папка датасета (VOCdevkit/VOC2007 или с images/, masks/)")
        self.browse_dataset_btn = QPushButton("Обзор")
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_path)
        row.addWidget(self.browse_dataset_btn)
        self.form_layout.addRow("📁 Папка датасета:", row)

        # YAML с точными путями
        self.deeplab_yaml = QLineEdit()
        self.deeplab_yaml.setPlaceholderText(
            "Или YAML с путями (переопределит папку и num_classes)"
        )
        self.browse_yaml_btn = QPushButton("Обзор YAML")
        self.browse_yaml_btn.clicked.connect(self._browse_deeplab_yaml)
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(self.deeplab_yaml)
        yaml_row.addWidget(self.browse_yaml_btn)
        self.form_layout.addRow("📄 DeepLab YAML:", yaml_row)

        yaml_hint = QLabel(
            "Если указан YAML — поля <code>path</code>, <code>num_classes</code> "
            "и пути сплитов (<code>train_txt</code>/<code>train_images</code> и т.п.) "
            "переопределяют значения из формы. Удобно, когда датасет сгенерирован "
            "утилитой и уже имеет готовый YAML с точными путями."
        )
        yaml_hint.setWordWrap(True)
        yaml_hint.setStyleSheet("color: #666; font-size: 11px;")
        self.form_layout.addRow(yaml_hint)

        # Взаимная очистка
        self.dataset_path.textChanged.connect(
            lambda: self.deeplab_yaml.clear() if self.dataset_path.text() else None
        )
        self.deeplab_yaml.textChanged.connect(
            lambda: self.dataset_path.clear() if self.deeplab_yaml.text() else None
        )

        # ========== 4. Проект, эксперимент, устройство, epochs/batch ==========
        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()

        # ========== 5. Остальные параметры ==========
        self._add_model_selection()
        self._add_architecture_fields()
        self._add_class_weights_fields()
        self._add_loss_fields()

        self.ignore_index = QSpinBox()
        self.ignore_index.setRange(-1, 255)
        self.ignore_index.setValue(255)
        self.ignore_index.setToolTip("Игнорируемый класс (255 для VOC, -1 = не игнорировать)")
        self.form_layout.addRow("Игнор. класс (ignore_index):", self.ignore_index)

        self._add_two_phase_fields()
        self._add_optimizer_fields()
        self._add_scheduler_fields()
        self._add_preprocessing_fields()
        self._add_advanced_training_fields()
        self._add_augmentation_fields()

        # ========== 6. Кнопки и прогресс ==========
        self.add_buttons()
        self.add_progress_bar()

        # Применяем HIGH_RES_PRESET
        self._apply_preset_dict(self.HIGH_RES_PRESET)
        self.preset_path.setText("preset.txt")

    # ------------------------------------------------------------------
    # Пресет (вверху окна)
    # ------------------------------------------------------------------
    def _add_preset_controls(self):
        preset_group = QGroupBox("Пресет обучения")
        preset_layout = QHBoxLayout(preset_group)

        self.preset_path = QLineEdit()
        self.preset_path.setPlaceholderText("Путь к preset.txt")
        self.preset_path.setMinimumWidth(220)
        preset_layout.addWidget(QLabel("Файл:"))
        preset_layout.addWidget(self.preset_path)

        # Кнопка выбора файла через стандартный диалог
        self.browse_preset_btn = QPushButton("Выбрать файл…")
        self.browse_preset_btn.setToolTip("Открыть диалог выбора файла пресета")
        self.browse_preset_btn.clicked.connect(self._browse_preset_file)
        preset_layout.addWidget(self.browse_preset_btn)

        self.load_preset_btn = QPushButton("Загрузить")
        self.load_preset_btn.setToolTip("Загрузить параметры из указанного файла")
        self.load_preset_btn.clicked.connect(self._load_preset)
        preset_layout.addWidget(self.load_preset_btn)

        self.save_preset_btn = QPushButton("Сохранить")
        self.save_preset_btn.setToolTip("Сохранить текущие параметры в указанный файл")
        self.save_preset_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(self.save_preset_btn)

        preset_layout.addStretch()
        self.form_layout.addRow(preset_group)

    def _browse_preset_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл пресета", "",
            "Preset files (*.txt *.yaml *.yml);;All files (*)"
        )
        if path:
            self.preset_path.setText(path)

    # ------------------------------------------------------------------
    # Датасет
    # ------------------------------------------------------------------
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self, "Выберите корневую папку датасета (VOCdevkit или с images/, masks/)"
        )
        if path:
            self.dataset_path.setText(path)
            self.deeplab_yaml.clear()

    def _browse_deeplab_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите YAML с путями датасета", "",
            "YAML files (*.yaml *.yml);;All files (*)"
        )
        if path:
            self.deeplab_yaml.setText(path)
            self.dataset_path.clear()

    # ------------------------------------------------------------------
    # Источник модели
    # ------------------------------------------------------------------
    def _add_model_selection(self):
        group = QGroupBox("Источник модели")
        layout = QVBoxLayout()
        self.model_from_preset = QRadioButton("Предустановленный backbone")
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

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите модель DeepLabV3/V3+", "", "PyTorch models (*.pth)"
        )
        if path:
            self.custom_model_path.setText(path)

    # ------------------------------------------------------------------
    # Архитектура
    # ------------------------------------------------------------------
    def _add_architecture_fields(self):
        arch_layout = QHBoxLayout()
        self.architecture = QComboBox()
        self.architecture.addItems(["DeepLabV3+", "DeepLabV3"])
        arch_layout.addWidget(QLabel("Архитектура:"))
        arch_layout.addWidget(self.architecture)

        self.backbone = QComboBox()
        self.backbone.addItems(["resnet50", "resnet101", "mobilenet_v3", "xception"])
        arch_layout.addWidget(QLabel("  Backbone:"))
        arch_layout.addWidget(self.backbone)

        self.output_stride = QComboBox()
        self.output_stride.addItems(["32", "16", "8"])
        arch_layout.addWidget(QLabel("  OS:"))
        arch_layout.addWidget(self.output_stride)

        self.output_activation = QComboBox()
        self.output_activation.addItems(["argmax", "softmax", "sigmoid"])
        arch_layout.addWidget(QLabel("  Активация:"))
        arch_layout.addWidget(self.output_activation)

        arch_layout.addStretch()
        self.form_layout.addRow("Архитектура:", arch_layout)

        self.input_size = QSpinBox()
        self.input_size.setRange(32, 2048)
        self.input_size.setSingleStep(16)
        self.input_size.setValue(1025)
        self.input_size.setToolTip(
            "Размер входа. Формула DeepLab: crop_size = output_stride * k + 1.\n"
            "Рекомендуется: 513 (OS=16), 1025 (OS=8)."
        )
        self.form_layout.addRow("Размер входа (input_size):", self.input_size)

        self.atrous_rates = QLineEdit()
        self.atrous_rates.setPlaceholderText("Авто (по OS). Пример: 6,12,18")
        self.atrous_rates.setToolTip(
            "Atrous rates для ASPP. Если пусто:\n"
            "OS=32 → 3,6,9; OS=16 → 6,12,18; OS=8 → 12,24,36."
        )
        self.form_layout.addRow("Atrous rates:", self.atrous_rates)

    # ------------------------------------------------------------------
    # Веса классов / Loss / Two-phase / Optimizer / Scheduler
    # (без изменений — оставляем как было)
    # ------------------------------------------------------------------
    def _add_class_weights_fields(self):
        cw_layout = QHBoxLayout()
        self.class_weights_mode = QComboBox()
        self.class_weights_mode.addItems(["None", "Auto (Dynamic)", "From file"])
        cw_layout.addWidget(QLabel("Веса классов:"))
        cw_layout.addWidget(self.class_weights_mode)

        self.class_weights_file = QLineEdit()
        self.class_weights_file.setPlaceholderText("Файл с весами")
        self.class_weights_file.setEnabled(False)
        cw_layout.addWidget(self.class_weights_file)

        self.browse_weights_btn = QPushButton("Обзор")
        self.browse_weights_btn.clicked.connect(self._browse_weights_file)
        self.browse_weights_btn.setEnabled(False)
        cw_layout.addWidget(self.browse_weights_btn)
        cw_layout.addStretch()
        self.form_layout.addRow(" ", cw_layout)

        self.class_weights_mode.currentTextChanged.connect(self._on_class_weights_mode_changed)

    def _on_class_weights_mode_changed(self, mode):
        is_file = (mode == "From file")
        self.class_weights_file.setEnabled(is_file)
        self.browse_weights_btn.setEnabled(is_file)

    def _browse_weights_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл весов классов", "",
            "JSON (*.json);;TXT (*.txt);;Все файлы (*.*)"
        )
        if path:
            self.class_weights_file.setText(path)

    def _add_loss_fields(self):
        loss_layout = QHBoxLayout()
        self.loss = QComboBox()
        self.loss.addItems([
            "CrossEntropy", "Dice", "Focal",
            "CrossEntropy+Dice", "CrossEntropy+Focal", "Dice+Focal"
        ])
        loss_layout.addWidget(QLabel("Loss:"))
        loss_layout.addWidget(self.loss)

        loss_layout.addWidget(QLabel("  Веса:"))
        self.loss_weight_ce = QDoubleSpinBox()
        self.loss_weight_ce.setRange(0.0, 10.0)
        self.loss_weight_ce.setValue(1.0)
        loss_layout.addWidget(self.loss_weight_ce)
        loss_layout.addWidget(QLabel("CE"))

        self.loss_weight_dice = QDoubleSpinBox()
        self.loss_weight_dice.setRange(0.0, 10.0)
        self.loss_weight_dice.setValue(0.0)
        loss_layout.addWidget(self.loss_weight_dice)
        loss_layout.addWidget(QLabel("Dice"))

        self.loss_weight_focal = QDoubleSpinBox()
        self.loss_weight_focal.setRange(0.0, 10.0)
        self.loss_weight_focal.setValue(0.0)
        loss_layout.addWidget(self.loss_weight_focal)
        loss_layout.addWidget(QLabel("Focal"))
        loss_layout.addStretch()
        self.form_layout.addRow("Потери:", loss_layout)

    def _add_two_phase_fields(self):
        self.two_phase_check = QCheckBox("Двухфазное обучение (заморозка backbone)")
        self.form_layout.addRow(self.two_phase_check)

        phase_layout = QHBoxLayout()
        self.freeze_epochs = QSpinBox()
        self.freeze_epochs.setRange(0, 500)
        self.freeze_epochs.setValue(10)
        self.freeze_epochs.setEnabled(False)
        phase_layout.addWidget(QLabel("Заморозка:"))
        phase_layout.addWidget(self.freeze_epochs)

        self.unfreeze_epochs = QSpinBox()
        self.unfreeze_epochs.setRange(0, 500)
        self.unfreeze_epochs.setValue(30)
        self.unfreeze_epochs.setEnabled(False)
        phase_layout.addWidget(QLabel("  Разморозка:"))
        phase_layout.addWidget(self.unfreeze_epochs)

        self.unfreeze_last_n_stages = QSpinBox()
        self.unfreeze_last_n_stages.setRange(-1, 10)
        self.unfreeze_last_n_stages.setValue(-1)
        self.unfreeze_last_n_stages.setToolTip(
            "Сколько последних стадий энкодера размораживать. -1 = весь энкодер."
        )
        phase_layout.addWidget(QLabel("  Разморозить N стадий:"))
        phase_layout.addWidget(self.unfreeze_last_n_stages)

        phase_layout.addStretch()
        self.form_layout.addRow(" ", phase_layout)

        self.freeze_bn_second_phase = QCheckBox("Заморозить BatchNorm на 2-й фазе")
        self.freeze_bn_second_phase.setChecked(True)
        self.form_layout.addRow(self.freeze_bn_second_phase)

        self.two_phase_check.toggled.connect(self._on_two_phase_toggled)

    def _on_two_phase_toggled(self, checked):
        self.freeze_epochs.setEnabled(checked)
        self.unfreeze_epochs.setEnabled(checked)
        self.unfreeze_last_n_stages.setEnabled(checked)
        self.freeze_bn_second_phase.setEnabled(checked)
        if checked:
            total = self.freeze_epochs.value() + self.unfreeze_epochs.value()
            self.epochs.setValue(total)
            self.epochs.setEnabled(False)
            self.freeze_epochs.valueChanged.connect(self._update_total_epochs)
            self.unfreeze_epochs.valueChanged.connect(self._update_total_epochs)
        else:
            self.epochs.setEnabled(True)
            try:
                self.freeze_epochs.valueChanged.disconnect(self._update_total_epochs)
                self.unfreeze_epochs.valueChanged.disconnect(self._update_total_epochs)
            except TypeError:
                pass

    def _update_total_epochs(self):
        if self.two_phase_check.isChecked():
            total = self.freeze_epochs.value() + self.unfreeze_epochs.value()
            self.epochs.setValue(total)

    def _add_optimizer_fields(self):
        optim_layout = QHBoxLayout()
        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "RMSprop"])
        optim_layout.addWidget(QLabel("Opt:"))
        optim_layout.addWidget(self.optimizer)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 0.1)
        self.lr.setValue(0.001)
        self.lr.setSingleStep(0.0001)
        self.lr.setDecimals(5)
        optim_layout.addWidget(QLabel("  LR:"))
        optim_layout.addWidget(self.lr)

        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 0.99)
        self.momentum.setValue(0.9)
        optim_layout.addWidget(QLabel("  Momentum:"))
        optim_layout.addWidget(self.momentum)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.001)
        self.weight_decay.setValue(0.0001)
        self.weight_decay.setDecimals(5)
        optim_layout.addWidget(QLabel("  WD:"))
        optim_layout.addWidget(self.weight_decay)
        optim_layout.addStretch()
        self.form_layout.addRow("Оптимизация:", optim_layout)

    def _add_scheduler_fields(self):
        sched_layout = QHBoxLayout()
        self.scheduler = QComboBox()
        self.scheduler.addItems(["Poly", "Step", "Cosine", "ReduceLROnPlateau", "None"])
        sched_layout.addWidget(QLabel("Scheduler:"))
        sched_layout.addWidget(self.scheduler)

        self.poly_power = QDoubleSpinBox()
        self.poly_power.setRange(0.1, 2.0)
        self.poly_power.setSingleStep(0.1)
        self.poly_power.setDecimals(2)
        self.poly_power.setValue(0.9)
        sched_layout.addWidget(QLabel("  Poly power:"))
        sched_layout.addWidget(self.poly_power)

        self.num_classes = QSpinBox()
        self.num_classes.setRange(1, 100)
        self.num_classes.setValue(21)
        sched_layout.addWidget(QLabel("  Классов:"))
        sched_layout.addWidget(self.num_classes)
        sched_layout.addStretch()
        self.form_layout.addRow("Прочее:", sched_layout)

    def _add_preprocessing_fields(self):
        prep_layout = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("ImageNet-нормализация")
        self.normalize_checkbox.setChecked(True)
        prep_layout.addWidget(self.normalize_checkbox)

        prep_layout.addWidget(QLabel("  Режим ресайза:"))
        self.resize_mode_combo = QComboBox()
        self.resize_mode_combo.addItems(["hard", "keep_aspect_pad"])
        self.resize_mode_combo.setCurrentText("hard")
        prep_layout.addWidget(self.resize_mode_combo)
        prep_layout.addStretch()
        self.form_layout.addRow("Препроцессинг:", prep_layout)

    def _add_advanced_training_fields(self):
        adv_group = QGroupBox("Дополнительные улучшения обучения")
        adv_layout = QFormLayout(adv_group)

        self.use_amp = QCheckBox("Mixed Precision (AMP)")
        self.use_amp.setChecked(True)
        adv_layout.addRow(self.use_amp)

        self.accum_steps = QSpinBox()
        self.accum_steps.setRange(1, 32)
        self.accum_steps.setValue(1)
        adv_layout.addRow("Gradient accumulation:", self.accum_steps)

        self.use_ema = QCheckBox("EMA (усреднение весов)")
        self.use_ema.setChecked(True)
        adv_layout.addRow(self.use_ema)

        self.ema_decay = QDoubleSpinBox()
        self.ema_decay.setRange(0.9, 0.9999)
        self.ema_decay.setSingleStep(0.001)
        self.ema_decay.setDecimals(4)
        self.ema_decay.setValue(0.999)
        adv_layout.addRow("EMA decay:", self.ema_decay)

        self.use_layerwise_lr = QCheckBox("Layer-wise LR для энкодера")
        self.use_layerwise_lr.setChecked(True)
        adv_layout.addRow(self.use_layerwise_lr)

        self.layerwise_decay = QDoubleSpinBox()
        self.layerwise_decay.setRange(0.01, 1.0)
        self.layerwise_decay.setSingleStep(0.05)
        self.layerwise_decay.setDecimals(2)
        self.layerwise_decay.setValue(0.1)
        adv_layout.addRow("Layer-wise decay:", self.layerwise_decay)

        self.multi_scale_infer = QCheckBox("Multi-scale inference на валидации")
        self.multi_scale_infer.setChecked(False)
        adv_layout.addRow(self.multi_scale_infer)

        self.infer_scales = QLineEdit()
        self.infer_scales.setText("0.75,1.0,1.25")
        adv_layout.addRow("Scales:", self.infer_scales)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 0.99)
        self.threshold.setSingleStep(0.01)
        self.threshold.setDecimals(2)
        self.threshold.setValue(0.5)
        adv_layout.addRow("Threshold:", self.threshold)

        self.form_layout.addRow(adv_group)

    def _add_augmentation_fields(self):
        aug_group = QGroupBox("Аугментации")
        aug_layout = QHBoxLayout(aug_group)
        self.aug_hflip = QCheckBox("HFlip")
        self.aug_vflip = QCheckBox("VFlip")
        self.aug_rotate = QCheckBox("Rotate")
        self.aug_scale = QCheckBox("Scale")
        self.aug_noise = QCheckBox("Noise")
        for cb in (self.aug_hflip, self.aug_vflip, self.aug_rotate,
                   self.aug_scale, self.aug_noise):
            cb.setChecked(False)
            aug_layout.addWidget(cb)
        aug_layout.addStretch()
        self.form_layout.addRow(aug_group)

    # ------------------------------------------------------------------
    # Пресеты: парсинг / сохранение / загрузка
    # ------------------------------------------------------------------
    def _parse_preset_file(self, path):
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    continue
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.split('#', 1)[0].strip()
                try:
                    data[key] = ast.literal_eval(val)
                except Exception:
                    data[key] = val
        return data

    def _load_preset(self):
        path = self.preset_path.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Ошибка", f"Файл пресета не найден:\n{path}")
            return
        try:
            data = self._parse_preset_file(path)
            self._apply_preset_dict(data)
            QMessageBox.information(self, "Пресет", f"Пресет загружен из {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить пресет:\n{e}")

    def _save_preset(self):
        path = self.preset_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Ошибка", "Укажите путь к файлу пресета.")
            return
        try:
            data = self._collect_preset_dict()
            with open(path, 'w', encoding='utf-8') as f:
                for k, v in data.items():
                    f.write(f"{k}: {v}\n")
            QMessageBox.information(self, "Пресет", f"Пресет сохранён в {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить пресет:\n{e}")

    def _collect_preset_dict(self):
        arch = "deeplabv3plus" if self.architecture.currentText() == "DeepLabV3+" else "deeplabv3"
        two_phase = self.two_phase_check.isChecked()
        if two_phase:
            total_epochs = self.freeze_epochs.value() + self.unfreeze_epochs.value()
        else:
            total_epochs = self.epochs.value()
        return {
            # Датасет
            "dataset_path": self.dataset_path.text().strip(),
            "deeplab_yaml": self.deeplab_yaml.text().strip(),
            # Архитектура
            "architecture": arch,
            "backbone": self.backbone.currentText(),
            "output_stride": int(self.output_stride.currentText()),
            "input_size": self.input_size.value(),
            "atrous_rates": self.atrous_rates.text().strip(),
            "output_activation": self.output_activation.currentText(),
            "num_classes": self.num_classes.value(),
            "ignore_index": self.ignore_index.value() if self.ignore_index.value() != -1 else None,
            "class_weights": (
                "auto" if self.class_weights_mode.currentText() == "Auto (Dynamic)"
                else (self.class_weights_file.text().strip() or None)
            ),
            # Loss
            "loss": self.loss.currentText(),
            "loss_weights": {
                "ce": self.loss_weight_ce.value(),
                "dice": self.loss_weight_dice.value(),
                "focal": self.loss_weight_focal.value(),
            },
            # Оптимизация
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "momentum": self.momentum.value(),
            "weight_decay": self.weight_decay.value(),
            "scheduler": self.scheduler.currentText(),
            "poly_power": self.poly_power.value(),
            "epochs": total_epochs,
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            # Two-phase
            "two_phase_training": two_phase,
            "freeze_epochs": self.freeze_epochs.value() if two_phase else 0,
            "unfreeze_epochs": self.unfreeze_epochs.value() if two_phase else 0,
            "unfreeze_last_n_stages": self.unfreeze_last_n_stages.value(),
            "freeze_bn_second_phase": self.freeze_bn_second_phase.isChecked(),
            # Препроцессинг
            "normalize": self.normalize_checkbox.isChecked(),
            "resize_mode": self.resize_mode_combo.currentText(),
            # Улучшения
            "use_amp": self.use_amp.isChecked(),
            "accum_steps": self.accum_steps.value(),
            "use_ema": self.use_ema.isChecked(),
            "ema_decay": self.ema_decay.value(),
            "use_layerwise_lr": self.use_layerwise_lr.isChecked(),
            "layerwise_decay": self.layerwise_decay.value(),
            "multi_scale_infer": self.multi_scale_infer.isChecked(),
            "infer_scales": [float(s.strip()) for s in self.infer_scales.text().split(',') if s.strip()],
            "threshold": self.threshold.value(),
            # Аугментации
            "augmentations": {
                "hflip": self.aug_hflip.isChecked(),
                "vflip": self.aug_vflip.isChecked(),
                "rotate": self.aug_rotate.isChecked(),
                "scale": self.aug_scale.isChecked(),
                "noise": self.aug_noise.isChecked(),
            },
        }

    def _apply_preset_dict(self, d):
        # Датасет
        if "dataset_path" in d:
            self.dataset_path.setText(str(d["dataset_path"] or ""))
        if "deeplab_yaml" in d:
            self.deeplab_yaml.setText(str(d["deeplab_yaml"] or ""))

        # Архитектура
        if "architecture" in d:
            arch_text = "DeepLabV3+" if str(d["architecture"]).lower() in ("deeplabv3plus", "deeplabv3+") else "DeepLabV3"
            self.architecture.setCurrentText(arch_text)
        if "backbone" in d:
            idx = self.backbone.findText(str(d["backbone"]))
            if idx >= 0:
                self.backbone.setCurrentIndex(idx)
        if "output_stride" in d:
            idx = self.output_stride.findText(str(d["output_stride"]))
            if idx >= 0:
                self.output_stride.setCurrentIndex(idx)
        if "input_size" in d:
            self.input_size.setValue(int(d["input_size"]))

        # atrous_rates — поддерживаем и строку "6,12,18", и список [6, 12, 18]
        if "atrous_rates" in d:
            val = d["atrous_rates"]
            if isinstance(val, (list, tuple)):
                self.atrous_rates.setText(",".join(str(int(x)) for x in val))
            else:
                self.atrous_rates.setText(str(val) if val not in (None, "") else "")

        if "output_activation" in d:
            idx = self.output_activation.findText(str(d["output_activation"]))
            if idx >= 0:
                self.output_activation.setCurrentIndex(idx)
        if "num_classes" in d:
            self.num_classes.setValue(int(d["num_classes"]))
        if "ignore_index" in d:
            val = d["ignore_index"]
            self.ignore_index.setValue(int(val) if val is not None else -1)
        if "class_weights" in d:
            cw = d["class_weights"]
            if cw == "auto":
                self.class_weights_mode.setCurrentText("Auto (Dynamic)")
            elif cw:
                self.class_weights_mode.setCurrentText("From file")
                self.class_weights_file.setText(str(cw))
            else:
                self.class_weights_mode.setCurrentText("None")

        # Loss
        if "loss" in d:
            idx = self.loss.findText(str(d["loss"]))
            if idx >= 0:
                self.loss.setCurrentIndex(idx)
        if "loss_weights" in d and isinstance(d["loss_weights"], dict):
            lw = d["loss_weights"]
            self.loss_weight_ce.setValue(float(lw.get("ce", 1.0)))
            self.loss_weight_dice.setValue(float(lw.get("dice", 0.0)))
            self.loss_weight_focal.setValue(float(lw.get("focal", 0.0)))

        # Оптимизация
        if "optimizer" in d:
            idx = self.optimizer.findText(str(d["optimizer"]))
            if idx >= 0:
                self.optimizer.setCurrentIndex(idx)
        if "lr" in d:
            self.lr.setValue(float(d["lr"]))
        if "momentum" in d:
            self.momentum.setValue(float(d["momentum"]))
        if "weight_decay" in d:
            self.weight_decay.setValue(float(d["weight_decay"]))
        if "scheduler" in d:
            idx = self.scheduler.findText(str(d["scheduler"]))
            if idx >= 0:
                self.scheduler.setCurrentIndex(idx)
        if "poly_power" in d:
            self.poly_power.setValue(float(d["poly_power"]))

        # epochs и batch
        if "batch_size" in d:
            bs = int(d["batch_size"])
            if bs == -1:
                self.auto_batch.setChecked(True)
            else:
                self.auto_batch.setChecked(False)
                self.batch.setValue(bs)

        # two-phase
        two_phase = bool(d.get("two_phase_training", False))
        self.two_phase_check.setChecked(two_phase)
        if two_phase:
            self.freeze_epochs.setValue(int(d.get("freeze_epochs", 10)))
            self.unfreeze_epochs.setValue(int(d.get("unfreeze_epochs", 30)))
            self._update_total_epochs()
        else:
            if "epochs" in d:
                self.epochs.setValue(int(d["epochs"]))
        if "unfreeze_last_n_stages" in d:
            val = d["unfreeze_last_n_stages"]
            self.unfreeze_last_n_stages.setValue(int(val) if val is not None else -1)
        if "freeze_bn_second_phase" in d:
            self.freeze_bn_second_phase.setChecked(bool(d["freeze_bn_second_phase"]))

        # Препроцессинг
        if "normalize" in d:
            self.normalize_checkbox.setChecked(bool(d["normalize"]))
        if "resize_mode" in d:
            idx = self.resize_mode_combo.findText(str(d["resize_mode"]))
            if idx >= 0:
                self.resize_mode_combo.setCurrentIndex(idx)

        # Улучшения
        if "use_amp" in d:
            self.use_amp.setChecked(bool(d["use_amp"]))
        if "accum_steps" in d:
            self.accum_steps.setValue(int(d["accum_steps"]))
        if "use_ema" in d:
            self.use_ema.setChecked(bool(d["use_ema"]))
        if "ema_decay" in d:
            self.ema_decay.setValue(float(d["ema_decay"]))
        if "use_layerwise_lr" in d:
            self.use_layerwise_lr.setChecked(bool(d["use_layerwise_lr"]))
        if "layerwise_decay" in d:
            self.layerwise_decay.setValue(float(d["layerwise_decay"]))
        if "multi_scale_infer" in d:
            self.multi_scale_infer.setChecked(bool(d["multi_scale_infer"]))
        if "infer_scales" in d:
            scales = d["infer_scales"]
            if isinstance(scales, (list, tuple)):
                self.infer_scales.setText(",".join(str(x) for x in scales))
            else:
                self.infer_scales.setText(str(scales))
        if "threshold" in d:
            self.threshold.setValue(float(d["threshold"]))

        # Аугментации
        if "augmentations" in d and isinstance(d["augmentations"], dict):
            aug = d["augmentations"]
            self.aug_hflip.setChecked(bool(aug.get("hflip", False)))
            self.aug_vflip.setChecked(bool(aug.get("vflip", False)))
            self.aug_rotate.setChecked(bool(aug.get("rotate", False)))
            self.aug_scale.setChecked(bool(aug.get("scale", False)))
            self.aug_noise.setChecked(bool(aug.get("noise", False)))

    # ------------------------------------------------------------------
    # get_params — передача в воркер
    # ------------------------------------------------------------------
    def get_params(self):
        if self.model_from_preset.isChecked():
            model_path = None
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан файл модели DeepLabV3/V3+")
                return None

        dataset_path = self.dataset_path.text().strip()
        deeplab_yaml = self.deeplab_yaml.text().strip()

        if not dataset_path and not deeplab_yaml:
            QMessageBox.warning(
                self, "Ошибка",
                "Укажите либо папку датасета, либо DeepLab YAML с путями."
            )
            return None
        if deeplab_yaml and not os.path.isfile(deeplab_yaml):
            QMessageBox.warning(self, "Ошибка", f"DeepLab YAML не найден:\n{deeplab_yaml}")
            return None
        if dataset_path and not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "Ошибка", f"Папка датасета не существует:\n{dataset_path}")
            return None

        # class_weights
        class_weights = None
        cw_mode = self.class_weights_mode.currentText()
        if cw_mode == "Auto (Dynamic)":
            class_weights = "auto"
        elif cw_mode == "From file":
            fpath = self.class_weights_file.text().strip()
            if not fpath:
                QMessageBox.warning(self, "Ошибка", "Выбран 'From file', но файл не указан")
                return None
            class_weights = fpath

        # two-phase
        two_phase = self.two_phase_check.isChecked()
        if two_phase:
            freeze_epochs = self.freeze_epochs.value()
            unfreeze_epochs = self.unfreeze_epochs.value()
            total_epochs = freeze_epochs + unfreeze_epochs
        else:
            freeze_epochs = 0
            unfreeze_epochs = 0
            total_epochs = self.epochs.value()

        # atrous_rates
        atrous_str = self.atrous_rates.text().strip()
        atrous_rates = None
        if atrous_str:
            try:
                atrous_rates = [int(x.strip()) for x in atrous_str.split(",") if x.strip()]
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "atrous_rates должны быть целыми числами через запятую")
                return None

        arch = "deeplabv3plus" if self.architecture.currentText() == "DeepLabV3+" else "deeplabv3"

        try:
            infer_scales = [float(s.strip()) for s in self.infer_scales.text().split(",") if s.strip()]
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "infer_scales должны быть числами через запятую")
            return None

        return {
            "model_path": model_path,
            "dataset_path": dataset_path or None,
            "deeplab_yaml": deeplab_yaml or None,
            "project_path": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": total_epochs,
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),

            "architecture": arch,
            "backbone": self.backbone.currentText(),
            "output_stride": int(self.output_stride.currentText()),
            "atrous_rates": atrous_rates,
            "output_activation": self.output_activation.currentText(),
            "input_size": self.input_size.value(),

            "class_weights": class_weights,

            "loss": self.loss.currentText(),
            "loss_weights": {
                "ce": self.loss_weight_ce.value(),
                "dice": self.loss_weight_dice.value(),
                "focal": self.loss_weight_focal.value(),
            },

            "ignore_index": self.ignore_index.value() if self.ignore_index.value() != -1 else None,

            "two_phase_training": two_phase,
            "freeze_epochs": freeze_epochs,
            "unfreeze_epochs": unfreeze_epochs,
            "unfreeze_last_n_stages": self.unfreeze_last_n_stages.value() if self.unfreeze_last_n_stages.value() != -1 else None,
            "freeze_bn_second_phase": self.freeze_bn_second_phase.isChecked(),

            "num_classes": self.num_classes.value(),
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "momentum": self.momentum.value(),
            "weight_decay": self.weight_decay.value(),
            "scheduler": self.scheduler.currentText(),
            "poly_power": self.poly_power.value(),

            "normalize": self.normalize_checkbox.isChecked(),
            "resize_mode": self.resize_mode_combo.currentText(),

            "use_amp": self.use_amp.isChecked(),
            "accum_steps": self.accum_steps.value(),
            "use_ema": self.use_ema.isChecked(),
            "ema_decay": self.ema_decay.value(),
            "use_layerwise_lr": self.use_layerwise_lr.isChecked(),
            "layerwise_decay": self.layerwise_decay.value(),
            "multi_scale_infer": self.multi_scale_infer.isChecked(),
            "infer_scales": infer_scales,
            "threshold": self.threshold.value(),

            "augmentations": {
                "hflip": self.aug_hflip.isChecked(),
                "vflip": self.aug_vflip.isChecked(),
                "rotate": self.aug_rotate.isChecked(),
                "scale": self.aug_scale.isChecked(),
                "noise": self.aug_noise.isChecked(),
            },
        }

# =============================================================================
# SegFormer Training Widget
# =============================================================================
class SegFormerTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        hint = (
            "<b>📁 Структура датасета SegFormer:</b><br>"
            "dataset/<br>"
            "├── img_dir/<br>"
            "│   ├── train/<br>"
            "│   └── val/<br>"
            "└── ann_dir/<br>"
            "    ├── train/   # маски PNG, значения = класс<br>"
            "    └── val/"
        )
        self.add_dataset_structure_hint(hint)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Корневая папка датасета (содержит img_dir и ann_dir)")
        self.browse_dataset_btn = QPushButton("Обзор")
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_path)
        row.addWidget(self.browse_dataset_btn)
        self.form_layout.addRow("📁 Папка датасета:", row)

        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()
        self._add_model_selection()
        self._add_segformer_specific()
        self.add_buttons()
        self.add_progress_bar()

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите корневую папку датасета")
        if path:
            self.dataset_path.setText(path)

    def _add_model_selection(self):
        group = QGroupBox("Источник модели")
        layout = QVBoxLayout()
        self.model_from_preset = QRadioButton("Использовать предустановленную SegFormer")
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

        self._add_preset_segformer_fields()

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель SegFormer", "", "PyTorch models (*.pth)")
        if path:
            self.custom_model_path.setText(path)

    def _add_preset_segformer_fields(self):
        self.variant = QComboBox()
        self.variant.addItems(["mit-b0", "mit-b1", "mit-b2", "mit-b3", "mit-b4", "mit-b5"])
        self.preset_layout.addRow("Variant:", self.variant)

    def _add_segformer_specific(self):
        self.num_classes = QSpinBox()
        self.num_classes.setRange(1, 100)
        self.num_classes.setValue(1)
        self.form_layout.addRow("Number of classes:", self.num_classes)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 0.1)
        self.lr.setValue(0.0001)
        self.lr.setSingleStep(0.00001)
        self.lr.setDecimals(6)
        self.form_layout.addRow("Learning rate:", self.lr)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["AdamW", "Adam", "SGD"])
        self.form_layout.addRow("optimizer:", self.optimizer)

    def get_params(self):
        if self.model_from_preset.isChecked():
            model_path = None
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан файл модели SegFormer")
                return None
        dataset_path = self.dataset_path.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Не указана папка датасета")
            return None

        return {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_path": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": self.epochs.value(),
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "variant": self.variant.currentText(),
            "num_classes": self.num_classes.value(),
        }


# =============================================================================
# SAM Training Widget
# =============================================================================
class SAMTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        hint = (
            "<b>📁 Структура датасета SAM:</b><br>"
            "dataset/<br>"
            "├── train/<br>"
            "│   ├── images/<br>"
            "│   └── labels/   # JSON с RLE‑масками<br>"
            "└── val/<br>"
            "    ├── images/<br>"
            "    └── labels/"
        )
        self.add_dataset_structure_hint(hint)

        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Корневая папка датасета (содержит train/, val/ с images/ и labels/)")
        self.browse_dataset_btn = QPushButton("Обзор")
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_path)
        row.addWidget(self.browse_dataset_btn)
        self.form_layout.addRow("📁 Папка датасета:", row)

        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()
        self._add_model_selection()
        self._add_sam_specific()
        self.add_buttons()
        self.add_progress_bar()

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите корневую папку датасета SAM")
        if path:
            self.dataset_path.setText(path)

    def _add_model_selection(self):
        group = QGroupBox("Источник модели")
        layout = QVBoxLayout()
        self.model_from_preset = QRadioButton("Использовать предустановленную SAM")
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

        self._add_preset_sam_fields()

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель SAM", "", "PyTorch models (*.pth)")
        if path:
            self.custom_model_path.setText(path)

    def _add_preset_sam_fields(self):
        self.model_type = QComboBox()
        self.model_type.addItems(["vit_b", "vit_l", "vit_h"])
        self.preset_layout.addRow("Model type:", self.model_type)

        self.prompt_type = QComboBox()
        self.prompt_type.addItems(["box", "points", "everything"])
        self.preset_layout.addRow("Prompt type:", self.prompt_type)

    def _add_sam_specific(self):
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 0.1)
        self.lr.setValue(0.0001)
        self.lr.setDecimals(5)
        self.form_layout.addRow("Learning rate:", self.lr)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["AdamW", "Adam", "SGD"])
        self.form_layout.addRow("optimizer:", self.optimizer)

    def get_params(self):
        if self.model_from_preset.isChecked():
            model_path = None
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан файл модели SAM")
                return None
        dataset_path = self.dataset_path.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Не указана папка датасета")
            return None

        return {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_path": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": self.epochs.value(),
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "model_type": self.model_type.currentText(),
            "prompt_type": self.prompt_type.currentText(),
        }