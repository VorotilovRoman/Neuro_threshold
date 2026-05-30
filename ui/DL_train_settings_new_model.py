# widgets/DL_train_settings_new_model.py
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
import os
import sys

# Импортируем базовый класс из первого файла
from .DL_train_settings_base import BaseTrainWidget


# =============================================================================
# DeepLabV3+ Training Widget
# =============================================================================
class DeepLabV3TrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # В файле DL_train_settings_new_model.py, класс DeepLabV3TrainWidget
        hint = (
            "<b>📁 Структура датасета DeepLabV3+ (PASCAL VOC):</b><br>"
            "VOCdevkit/<br>"
            "└── VOC2007 (или VOC2012)/<br>"
            "    ├── JPEGImages/          # исходные изображения (*.jpg)<br>"
            "    ├── SegmentationClass/   # маски сегментации (PNG, палитра, индексы классов)<br>"
            "    └── ImageSets/<br>"
            "        └── Segmentation/    # train.txt, val.txt – списки имён файлов<br>"
            "Маски должны быть в формате 8‑битных PNG с палитрой (0 – фон, 1..N – объекты)."
        )
        self.add_dataset_structure_hint(hint)

        # ----- Датасет -----
        self.dataset_path = QLineEdit()
        self.dataset_path.setPlaceholderText("Корневая папка датасета (VOCdevkit)")
        self.browse_dataset_btn = QPushButton("Обзор")
        self.browse_dataset_btn.clicked.connect(self._browse_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_path)
        row.addWidget(self.browse_dataset_btn)
        self.form_layout.addRow("📁 Папка датасета:", row)

        # ----- Проект и эксперимент -----
        self.add_project_field()
        self.add_experiment_name_field()
        self.add_device_field()
        self.add_common_training_params()

        # ----- Источник модели -----
        self._add_model_selection()

        # ----- Параметры архитектуры (backbone, output_stride, output_activation) -----
        arch_layout = QHBoxLayout()
        self.backbone = QComboBox()
        self.backbone.addItems(["resnet50", "resnet101", "mobilenet_v3", "xception"])
        self.backbone.setToolTip("Backbone (кодировщик) для извлечения признаков.")
        arch_layout.addWidget(QLabel("Backbone:"))
        arch_layout.addWidget(self.backbone)

        self.output_stride = QComboBox()
        self.output_stride.addItems(["16", "8"])
        self.output_stride.setToolTip("Output stride: 16 (экономия памяти) или 8 (лучшее качество).")
        arch_layout.addWidget(QLabel("  Out stride:"))
        arch_layout.addWidget(self.output_stride)

        self.output_activation = QComboBox()
        self.output_activation.addItems(["argmax", "sigmoid", "softmax"])
        self.output_activation.setToolTip("Финальная активация: argmax (мультикласс), sigmoid (бинарная), softmax.")
        arch_layout.addWidget(QLabel("  Активация:"))
        arch_layout.addWidget(self.output_activation)
        arch_layout.addStretch()
        self.form_layout.addRow("Архитектура:", arch_layout)

        # ----- Веса классов (class_weights) -----
        cw_layout = QHBoxLayout()
        self.class_weights_mode = QComboBox()
        self.class_weights_mode.addItems(["None", "Auto (Dynamic)", "From file"])
        self.class_weights_mode.setToolTip("Компенсация дисбаланса классов.")
        cw_layout.addWidget(QLabel("Веса классов:"))
        cw_layout.addWidget(self.class_weights_mode)

        self.class_weights_file = QLineEdit()
        self.class_weights_file.setPlaceholderText("Файл с весами")
        self.class_weights_file.setEnabled(False)
        self.class_weights_file.setToolTip("JSON или TXT, например: [0.5,1.2,1.8]")
        cw_layout.addWidget(self.class_weights_file)

        self.browse_weights_btn = QPushButton("Обзор")
        self.browse_weights_btn.clicked.connect(self._browse_weights_file)
        self.browse_weights_btn.setEnabled(False)
        cw_layout.addWidget(self.browse_weights_btn)
        cw_layout.addStretch()
        self.form_layout.addRow(" ", cw_layout)  # пустая метка для отступа

        self.class_weights_mode.currentTextChanged.connect(self._on_class_weights_mode_changed)

        # ----- Функция потерь и веса (в одной строке) -----
        loss_layout = QHBoxLayout()
        self.loss = QComboBox()
        self.loss.addItems(["CrossEntropy", "Dice", "Focal", "CrossEntropy+Dice", "CrossEntropy+Focal", "Dice+Focal"])
        self.loss.setToolTip("Основная функция потерь.")
        loss_layout.addWidget(QLabel("Loss:"))
        loss_layout.addWidget(self.loss)

        loss_layout.addWidget(QLabel("  Веса:"))
        self.loss_weight_ce = QDoubleSpinBox()
        self.loss_weight_ce.setRange(0.0, 10.0)
        self.loss_weight_ce.setValue(1.0)
        self.loss_weight_ce.setToolTip("Вклад CrossEntropy")
        loss_layout.addWidget(self.loss_weight_ce)
        loss_layout.addWidget(QLabel("CE"))

        self.loss_weight_dice = QDoubleSpinBox()
        self.loss_weight_dice.setRange(0.0, 10.0)
        self.loss_weight_dice.setValue(1.0)
        self.loss_weight_dice.setToolTip("Вклад Dice")
        loss_layout.addWidget(self.loss_weight_dice)
        loss_layout.addWidget(QLabel("Dice"))

        self.loss_weight_focal = QDoubleSpinBox()
        self.loss_weight_focal.setRange(0.0, 10.0)
        self.loss_weight_focal.setValue(0.0)
        self.loss_weight_focal.setToolTip("Вклад Focal")
        loss_layout.addWidget(self.loss_weight_focal)
        loss_layout.addWidget(QLabel("Focal"))
        loss_layout.addStretch()
        self.form_layout.addRow("Потери:", loss_layout)

        # ----- ignore_index -----
        self.ignore_index = QSpinBox()
        self.ignore_index.setRange(-1, 255)
        self.ignore_index.setValue(-1)
        self.ignore_index.setToolTip("Игнорируемый класс (255 для VOC, -1 = не игнорировать)")
        self.form_layout.addRow("Игнор. класс (ignore_index):", self.ignore_index)

        # ----- Двухфазное обучение -----
        self.two_phase_check = QCheckBox("Двухфазное обучение (заморозка backbone)")
        self.two_phase_check.setToolTip("Сначала frozen backbone, затем разморозка всей сети")
        self.form_layout.addRow(self.two_phase_check)

        phase_layout = QHBoxLayout()
        self.freeze_epochs = QSpinBox()
        self.freeze_epochs.setRange(0, 500)
        self.freeze_epochs.setValue(20)
        self.freeze_epochs.setEnabled(False)
        self.freeze_epochs.setToolTip("Эпох с frozen backbone")
        phase_layout.addWidget(QLabel("Заморозка:"))
        phase_layout.addWidget(self.freeze_epochs)

        self.unfreeze_epochs = QSpinBox()
        self.unfreeze_epochs.setRange(0, 500)
        self.unfreeze_epochs.setValue(30)
        self.unfreeze_epochs.setEnabled(False)
        self.unfreeze_epochs.setToolTip("Эпох с размороженным backbone")
        phase_layout.addWidget(QLabel("  Разморозка:"))
        phase_layout.addWidget(self.unfreeze_epochs)
        phase_layout.addStretch()
        self.form_layout.addRow(" ", phase_layout)

        self.two_phase_check.toggled.connect(self._on_two_phase_toggled)

        # ----- Оптимизатор, LR, momentum, weight_decay (в одной строке) -----
        optim_layout = QHBoxLayout()
        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW", "RMSprop"])
        self.optimizer.setToolTip("Оптимизатор")
        optim_layout.addWidget(QLabel("Opt:"))
        optim_layout.addWidget(self.optimizer)

        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 0.1)
        self.lr.setValue(0.007)
        self.lr.setSingleStep(0.0001)
        self.lr.setDecimals(5)
        self.lr.setToolTip("Начальная скорость обучения")
        optim_layout.addWidget(QLabel("  LR:"))
        optim_layout.addWidget(self.lr)

        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 0.99)
        self.momentum.setValue(0.9)
        self.momentum.setToolTip("Momentum (для SGD)")
        optim_layout.addWidget(QLabel("  Momentum:"))
        optim_layout.addWidget(self.momentum)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.001)
        self.weight_decay.setValue(0.0001)
        self.weight_decay.setDecimals(5)
        self.weight_decay.setToolTip("L2-регуляризация")
        optim_layout.addWidget(QLabel("  WD:"))
        optim_layout.addWidget(self.weight_decay)
        optim_layout.addStretch()
        self.form_layout.addRow("Оптимизация:", optim_layout)

        # ----- Планировщик и количество классов (в одной строке) -----
        sched_layout = QHBoxLayout()
        self.scheduler = QComboBox()
        self.scheduler.addItems(["Poly", "Step", "Cosine", "None"])
        self.scheduler.setToolTip("Планировщик LR")
        sched_layout.addWidget(QLabel("Scheduler:"))
        sched_layout.addWidget(self.scheduler)

        self.num_classes = QSpinBox()
        self.num_classes.setRange(1, 100)
        self.num_classes.setValue(21)
        self.num_classes.setToolTip("Количество классов (включая фон)")
        sched_layout.addWidget(QLabel("  Классов:"))
        sched_layout.addWidget(self.num_classes)
        sched_layout.addStretch()
        self.form_layout.addRow("Прочее:", sched_layout)

        # ----- Аугментации (компактно в строки) -----
        aug_group = QGroupBox("Аугментации")
        aug_layout = QHBoxLayout(aug_group)  # горизонтальная компоновка
        self.aug_hflip = QCheckBox("HFlip")
        self.aug_hflip.setChecked(True)
        self.aug_vflip = QCheckBox("VFlip")
        self.aug_vflip.setChecked(True)
        self.aug_rotate = QCheckBox("Rotate")
        self.aug_rotate.setChecked(True)
        self.aug_scale = QCheckBox("Scale")
        self.aug_scale.setChecked(True)
        self.aug_noise = QCheckBox("Noise")
        self.aug_noise.setChecked(True)
        aug_layout.addWidget(self.aug_hflip)
        aug_layout.addWidget(self.aug_vflip)
        aug_layout.addWidget(self.aug_rotate)
        aug_layout.addWidget(self.aug_scale)
        aug_layout.addWidget(self.aug_noise)
        aug_layout.addStretch()
        self.form_layout.addRow(aug_group)

        # ----- Кнопки и прогресс-бар -----
        self.add_buttons()
        self.add_progress_bar()

    # --- Все вспомогательные методы остаются без изменений, за исключением добавления сигналов ---
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Выберите корневую папку датасета (VOCdevkit)")
        if path:
            self.dataset_path.setText(path)

    # ... (методы _add_model_selection, _toggle_model_source, _browse_custom_model,
    # ... _add_preset_deeplab_fields, _on_class_weights_mode_changed, _browse_weights_file,
    # ... _on_two_phase_toggled, _update_total_epochs) такие же, как в предыдущей версии

    # Ниже приведены тела этих методов (скопируйте их из предыдущего большого класса,
    # они не меняются). Я приведу их здесь для полноты, но в реальном коде они уже есть.

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

        self._add_preset_deeplab_fields()

    def _toggle_model_source(self, checked):
        is_preset = self.model_from_preset.isChecked()
        self.preset_widget.setVisible(is_preset)
        self.file_widget.setVisible(not is_preset)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите модель DeepLabV3+", "", "PyTorch models (*.pth)")
        if path:
            self.custom_model_path.setText(path)

    def _add_preset_deeplab_fields(self):
        # Обратите внимание: поля backbone уже добавлены в общую строку архитектуры.
        # Поэтому здесь оставим только пустую заглушку, чтобы не дублировать.
        # Можно просто pass.
        pass

    def _on_class_weights_mode_changed(self, mode):
        is_file = (mode == "From file")
        self.class_weights_file.setEnabled(is_file)
        self.browse_weights_btn.setEnabled(is_file)

    def _browse_weights_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл весов классов", "",
                                              "JSON (*.json);;TXT (*.txt);;Все файлы (*.*)")
        if path:
            self.class_weights_file.setText(path)

    def _on_two_phase_toggled(self, checked):
        self.freeze_epochs.setEnabled(checked)
        self.unfreeze_epochs.setEnabled(checked)
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

    def get_params(self):
        if self.model_from_preset.isChecked():
            model_path = None
        else:
            model_path = self.custom_model_path.text().strip()
            if not model_path:
                QMessageBox.warning(self, "Ошибка", "Не указан файл модели DeepLabV3+")
                return None

        dataset_path = self.dataset_path.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, "Ошибка", "Не указана папка датасета")
            return None

        # Обработка class_weights
        class_weights = None
        cw_mode = self.class_weights_mode.currentText()
        if cw_mode == "Auto (Dynamic)":
            class_weights = "auto"
        elif cw_mode == "From file":
            fpath = self.class_weights_file.text().strip()
            if fpath:
                class_weights = fpath
            else:
                QMessageBox.warning(self, "Ошибка", "Выбран 'From file', но файл не указан")
                return None

        # Двухфазное обучение
        two_phase = self.two_phase_check.isChecked()
        if two_phase:
            freeze_epochs = self.freeze_epochs.value()
            unfreeze_epochs = self.unfreeze_epochs.value()
            total_epochs = freeze_epochs + unfreeze_epochs
        else:
            freeze_epochs = 0
            unfreeze_epochs = 0
            total_epochs = self.epochs.value()

        params = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "project_path": self.project_path.text().strip() or None,
            "name": self.experiment_name.text().strip() or None,
            "epochs": total_epochs,
            "batch_size": -1 if self.auto_batch.isChecked() else self.batch.value(),
            "device": self.device_combo.currentText(),

            # Архитектура
            "backbone": self.backbone.currentText(),
            "output_stride": int(self.output_stride.currentText()),
            "output_activation": self.output_activation.currentText(),

            # Веса классов
            "class_weights": class_weights,

            # Потери и веса
            "loss": self.loss.currentText(),
            "loss_weights": {
                "ce": self.loss_weight_ce.value(),
                "dice": self.loss_weight_dice.value(),
                "focal": self.loss_weight_focal.value(),
            },

            # Игнорируемый класс
            "ignore_index": self.ignore_index.value() if self.ignore_index.value() != -1 else None,

            # Двухфазное обучение
            "two_phase_training": two_phase,
            "freeze_epochs": freeze_epochs,
            "unfreeze_epochs": unfreeze_epochs,

            # Общие параметры
            "num_classes": self.num_classes.value(),
            "optimizer": self.optimizer.currentText(),
            "lr": self.lr.value(),
            "momentum": self.momentum.value(),
            "weight_decay": self.weight_decay.value(),
            "scheduler": self.scheduler.currentText(),

            # Аугментации
            "augmentations": {
                "hflip": self.aug_hflip.isChecked(),
                "vflip": self.aug_vflip.isChecked(),
                "rotate": self.aug_rotate.isChecked(),
                "scale": self.aug_scale.isChecked(),
                "noise": self.aug_noise.isChecked(),
            }
        }
        return params

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