"""Prompt-model fine-tuning controls using the same model catalog and source dialogs."""

from pathlib import Path

from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QWidget

from ui import dataset_source as sources
from ui.DL_train_settings_base import BaseTrainWidget
from utils_new.configuration.sam_training import sam_training_config
from utils_new.import_export.sam_weights import MODELS, plan_model_download
from utils_new.import_export.training import read_model_meta


class SAMTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.form_layout.setSpacing(8)
        self.add_dataset_structure_hint(
            "<b>Дообучение SAM/SAM2 по маскам отдельных объектов</b><br>"
            "Кодировщик снимков заморожен. Обучается декодер масок, дополнительно — кодировщик подсказок. "
            "SAM3 пока доступен только для применения.<br>"
            "YOLO Segment YAML, COCO с полигонами/RLE или YAML индексных масок. "
            "BBox/OBB не заменяют маски. В индексных масках объекты выделяются по связным компонентам класса.")
        self.dataset_source = sources.DatasetSourceWidget(("yolo_yaml", "coco", "unet_yaml"), profile="yolo_train")
        self.source = QLineEdit()
        self.dataset_source.add_path_field("Датасет:", self.source)
        for key in ("yolo_yaml", "coco", "unet_yaml"):
            self.dataset_source.button(key).clicked.connect(lambda _, action=key: self._choose_source(action))
        self.form_layout.addRow(self.dataset_source)
        self.val_source = self._path("Отдельная валидация:", lambda: self._choose_validation(), "Необязательно: YAML или COCO JSON")
        note = QLabel("Используется split val. Если его нет — выделяется 20% снимков по seed. "
                      "Test не участвует. Одинаковые снимки в train/val запрещены.")
        note.setWordWrap(True)
        self.form_layout.addRow(note)
        self.model = QComboBox()
        for key, value in MODELS.items():
            if key != "sam3":
                self.model.addItem(value.name, key)
        self.model.setCurrentIndex(self.model.findData("sam2.1-tiny"))
        self.form_layout.addRow("Архитектура:", self.model)
        self.weights = self._path("Начальные веса:", self._choose_weights, "Локальный .pt / .pth, по умолчанию models/")
        self.model.activated.connect(self._model_changed)
        self.resume = self._path("Продолжить обучение:", self._choose_resume, "last.resume.pt; очистите для нового запуска")
        self.add_project_field()
        self.project_path.setText("runs/sam")
        self.add_experiment_name_field()
        self.add_device_field()
        self.device_combo.clear()
        self.device_combo.addItems(["auto", "cuda:0", "cpu"])
        self.add_common_training_params()
        self.epochs.setValue(20)
        self.batch.setValue(1)
        self.batch.setMaximum(1)
        self.batch.setEnabled(False)
        self.auto_batch.setChecked(False)
        self.auto_batch.setEnabled(False)
        self.auto_batch.setToolTip("Один снимок на шаг. Увеличивайте накопление градиентов; AutoBatch YOLO здесь не применяется.")
        self.accum_steps = self._integer("Накопление градиентов:", 1, 128, 4)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(7)
        self.lr.setRange(0.0000001, 0.1)
        self.lr.setValue(0.0001)
        self.form_layout.addRow("Learning rate (AdamW):", self.lr)
        self.train_prompt_encoder = QCheckBox("Также обучать кодировщик подсказок")
        self.form_layout.addRow(self.train_prompt_encoder)
        self.use_amp = QCheckBox("AMP на CUDA (на CPU отключается)")
        self.use_amp.setChecked(True)
        self.form_layout.addRow(self.use_amp)
        self.prompt_type = QComboBox()
        for label, key in (("Смешанные: рамки и точки", "mixed"), ("Рамки", "box"), ("Точки", "points")):
            self.prompt_type.addItem(label, key)
        self.form_layout.addRow("Обучающие подсказки:", self.prompt_type)
        self.advanced_button = QPushButton("Параметры подсказок, масок и воспроизводимости…")
        self.advanced_button.setCheckable(True)
        self.form_layout.addRow(self.advanced_button)
        self.advanced = QWidget()
        self.advanced_form = QFormLayout(self.advanced)
        self.advanced_form.setContentsMargins(0, 0, 0, 0)
        self.form_layout.addRow(self.advanced)
        self.advanced.hide()
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        self.positive_points = self._integer("Точек объекта:", 1, 64, 1)
        self.negative_points = self._integer("Точек фона:", 0, 64, 1)
        self.max_objects = self._integer("Объектов/снимок на эпоху (train):", 1, 10000, 8, form=self.form_layout)
        self.max_objects.setToolTip("Случайная выборка в train; валидация проверяет все объекты.")
        self.max_side = self._integer("Макс. сторона рабочих снимков:", 32, 16384, 1024, form=self.form_layout)
        self.max_side.setToolTip("Размер для масок и метрик. Внутренний размер выбранной SAM остаётся штатным.")
        self.seed = self._integer("Seed:", 0, 2147483647, 42, form=self.form_layout)
        self.background_id = self._integer("Pixel ID фона (индексные маски):", 0, 65535, 0, form=self.form_layout)
        self.ignore_index = self._integer("Игнорируемый Pixel ID (-1 = нет):", -1, 65535, 255, form=self.form_layout)
        self.add_buttons()
        self.add_progress_bar()
        self._find_weights()
        self.advanced_button.setChecked(True)
        self.advanced_button.hide()
        self.organize_training_stages()

    def _integer(self, label, minimum, maximum, value, *, form=None):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        (form if form is not None else getattr(self, "advanced_form", self.form_layout)).addRow(label, widget)
        return widget

    def set_running(self, running):
        super().set_running(running)
        self.progress_bar.setRange(0, 0 if running else 100)
        if running:
            self.progress_bar.setFormat("SAM: проверка данных и загрузка весов…")

    def _path(self, label, callback, placeholder):
        edit, button = QLineEdit(), QPushButton("…")
        edit.setPlaceholderText(placeholder)
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(button)
        button.clicked.connect(callback)
        self.form_layout.addRow(label, row)
        return edit

    def _choose_source(self, action):
        path, _ = sources.get_open_file(self, action)
        if path:
            sources.set_dataset_path(self.source, path)

    def _choose_validation(self):
        path, _ = sources.get_open_file(self, "dataset_any", confirm=False)
        if path:
            approved = sources.confirm_dataset_import(self, [path], "coco" if Path(path).suffix.lower() == ".json" else "yolo")
            if approved:
                sources.set_dataset_path(self.val_source, approved[0])

    def _choose_weights(self):
        path, _ = sources.get_open_file(self, "model_torch", directory=str(Path(__file__).resolve().parents[2] / "models"))
        if path:
            self.weights.setText(path)

    def _model_changed(self):
        self.resume.clear()
        self._find_weights()

    def _find_weights(self):
        plan = plan_model_download(self.model.currentData())
        weights = plan.files[0][1]
        self.weights.setText(str(weights) if weights.is_file() else "")
        self.weights.setToolTip(f"Ожидаемый файл: {weights}. Недостающие веса можно скачать в общем блоке SAM или через CLI sam_weights.")

    def _choose_resume(self):
        path, _ = sources.get_open_file(self, "model_torch", directory="runs/sam")
        if not path:
            return
        try:
            if not path.endswith(".resume.pt"):
                raise ValueError("Выберите last.resume.pt. Файлы best.pt/last.pt — начальные веса нового запуска.")
            meta = read_model_meta(path)
            cfg = meta["config"]
            if meta.get("kind") != "sam" or not meta.get("complete_epoch"):
                raise ValueError("Нужен last.resume.pt и его файл метаданных.")
            if self.model.findData(cfg["model_key"]) < 0:
                raise ValueError("Архитектура checkpoint не поддерживает обучение.")
            self._resume_defaults = cfg
            self.model.setCurrentIndex(self.model.findData(cfg["model_key"]))
            self.weights.setText(cfg["model_path"])
            for name in ("source", "val_source"):
                sources.set_dataset_path(getattr(self, name), cfg[name])
            for name in ("accum_steps", "lr", "positive_points", "negative_points", "max_objects", "max_side", "seed", "background_id"):
                getattr(self, name).setValue(cfg[name])
            self.ignore_index.setValue(cfg["ignore_index"] if cfg["ignore_index"] is not None else -1)
            for name in ("train_prompt_encoder", "use_amp"):
                getattr(self, name).setChecked(cfg[name])
            self.prompt_type.setCurrentIndex(self.prompt_type.findData(cfg["prompt_type"]))
            self.epochs.setValue(max(cfg["epochs"], meta["epoch"] + 10))
            self.resume.setText(path)
        except (KeyError, OSError, ValueError) as error:
            QMessageBox.warning(self, "Продолжение SAM", str(error))

    def get_params(self):
        params = dict(getattr(self, "_resume_defaults", {}) if self.resume.text().strip() else {})
        params.update(model_key=self.model.currentData(), model_path=self.weights.text().strip(),
            resume=self.resume.text().strip(), source=sources.dataset_path(self.source),
            val_source=sources.dataset_path(self.val_source), project_path=self.project_path.text().strip(),
            name=self.experiment_name.text().strip(), device=self.device_combo.currentText(),
            epochs=self.epochs.value(), prompt_type=self.prompt_type.currentData(),
            train_prompt_encoder=self.train_prompt_encoder.isChecked(), use_amp=self.use_amp.isChecked())
        for name in ("accum_steps", "lr", "positive_points", "negative_points", "max_objects", "max_side", "seed", "background_id", "ignore_index"):
            params[name] = getattr(self, name).value()
        if params["ignore_index"] == -1:
            params["ignore_index"] = None
        try:
            return sam_training_config(params)
        except ValueError as error:
            QMessageBox.warning(self, "Параметры SAM", str(error))
            return None
