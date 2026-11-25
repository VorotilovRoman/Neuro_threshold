"""SAM annotation panel, prompt overlay and asynchronous editor integration."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

from pathlib import Path

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QProgressBar, QPushButton, QSizePolicy, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from ui.image_view import SmartGraphicsViewNew
from ui import dataset_source as source_dialogs
from ui.segmentation.jobs import BackgroundTask
from utils_new.annotations.adapters import from_legacy
from utils_new.annotations.geometry import bounds
from utils_new.annotations.model import ImageSize
from utils_new.import_export.sam_weights import MODELS
from utils_new.segmentation.backends.sam import LocalSAM, MODELS_DIR, SamPrompt


class AutoLabelStack(QStackedWidget):
    """Size to the active model's controls, including when SAM is hidden."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _: self.updateGeometry())

    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() else super().sizeHint()

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() else super().minimumSizeHint()


class SamPromptView(SmartGraphicsViewNew):
    prompts_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sam_tool = None
        self.sam_points, self.sam_labels = [], []
        self.sam_box = self._sam_start = None
        self.sam_preview = []

    def clear_sam_prompts(self):
        self.sam_points, self.sam_labels = [], []
        self.sam_box = self._sam_start = None
        self.prompts_changed.emit()
        self.viewport().update()

    def _sam_point(self, event, *, dragging=False):
        point = self.mapToScene(event.pos())
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        if not dragging and not (0 <= point.x() < self.img_width and 0 <= point.y() < self.img_height):
            return None
        return (max(0, min(point.x(), self.img_width)), max(0, min(point.y(), self.img_height)))

    def mousePressEvent(self, event):
        if self.sam_tool in ("positive", "negative") and event.button() in (Qt.LeftButton, Qt.RightButton):
            point = self._sam_point(event)
            if point is not None:
                self.sam_points.append(point)
                self.sam_labels.append(0 if event.button() == Qt.RightButton or self.sam_tool == "negative" else 1)
                self.prompts_changed.emit()
                self.viewport().update()
            event.accept()
            return
        if self.sam_tool == "box" and event.button() == Qt.LeftButton:
            self._sam_start = self._sam_point(event)
            if self._sam_start is not None:
                self.sam_box = (*self._sam_start, *self._sam_start)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.sam_tool == "box" and self._sam_start is not None:
            point = self._sam_point(event, dragging=True)
            if point is not None:
                x, y = self._sam_start
                self.sam_box = (min(x, point[0]), min(y, point[1]), max(x, point[0]), max(y, point[1]))
                self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.sam_tool == "box" and event.button() == Qt.LeftButton and self._sam_start is not None:
            self.mouseMoveEvent(event)
            self._sam_start = None
            if self.sam_box[2] <= self.sam_box[0] or self.sam_box[3] <= self.sam_box[1]:
                self.sam_box = None
            self.prompts_changed.emit()
            self.viewport().update()
            event.accept()
            return
        if self.sam_tool in ("positive", "negative") and event.button() in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        if self.sam_tool in ("positive", "negative"):
            event.accept()
            return
        super().contextMenuEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.sam_tool is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.sam_tool is not None:
            self.sam_tool = self._sam_start = None
            self.set_drawing_tool(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not hasattr(self, "sam_preview"):
            return
        painter.save()
        try:
            pen = QPen(QColor("#00c8ff"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 200, 255, 50))
            if self.img_width > 0 and self.img_height > 0:
                size = ImageSize(self.img_width, self.img_height)
                for ann in self.sam_preview:
                    polygon = from_legacy(ann, size).points_px
                    painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in polygon]))
            painter.setBrush(Qt.NoBrush)
            if self.sam_box is not None:
                x1, y1, x2, y2 = self.sam_box
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            radius = 5 / max(abs(self.transform().m11()), 1e-6)
            for (x, y), label in zip(self.sam_points, self.sam_labels):
                color = QColor("#26d84b" if label else "#ff4858")
                pen.setColor(color)
                painter.setPen(pen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, y), radius, radius)
        finally:
            painter.restore()


class SamPanel(QWidget):
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
        form.addRow(self.load)
        self.mode = QComboBox()
        self.mode.addItem("Точки и рамка", "points")
        self.mode.addItem("Текст и рамка (SAM3)", "text")
        self.mode.setToolTip("При смене режима SAM3 загрузите модель заново: в памяти хранится один backend.")
        form.addRow("Подсказки:", self.mode)
        self.mode_label = form.labelForField(self.mode)
        self.text = QLineEdit()
        self.text.setPlaceholderText("Например: metal screw")
        self.text.setToolTip("Только SAM3. Короткое английское название объекта; класс датасета задаётся отдельно.")
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
        self.score = QDoubleSpinBox()
        self.score.setRange(0, 1)
        self.score.setSingleStep(0.05)
        self.score.setValue(0.5)
        self.score.setToolTip("SAM/SAM2: предсказанный IoU; SAM3 с текстом: уверенность. Это разные оценки.")
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
        sam3 = self.model.currentData() == "sam3"
        self.mode.setEnabled(sam3)
        self.mode.setVisible(sam3)
        self.mode_label.setVisible(sam3)
        if not sam3:
            self.mode.setCurrentIndex(0)
        text_mode = sam3 and self.mode.currentData() == "text"
        self.tokenizer_row.setVisible(text_mode)
        self.tokenizer_label.setVisible(text_mode)
        self.text.setEnabled(text_mode)
        self.text.setVisible(text_mode)
        self.text_label.setVisible(text_mode)
        self.positive.setEnabled(not text_mode)
        self.negative.setEnabled(not text_mode)


class SamAutoLabelController:
    def __init__(self, owner):
        self.owner, self.panel, self.view = owner, owner.sam_panel, owner.image_view
        self.backend = self.job = None
        self.loading = False
        self.closed = False
        self.revision = 0
        self.frame = None
        self.preview_key = None
        panel = self.panel
        panel.browse_weights.clicked.connect(self.browse_weights)
        panel.browse_tokenizer.clicked.connect(self.browse_tokenizer)
        panel.load.clicked.connect(self.load)
        panel.run.clicked.connect(self.run)
        panel.accept.clicked.connect(self.accept)
        panel.cancel.clicked.connect(self.cancel)
        panel.clear.clicked.connect(self.view.clear_sam_prompts)
        panel.selected_box.clicked.connect(self.selected_box)
        panel.positive.clicked.connect(lambda: self.tool("positive"))
        panel.negative.clicked.connect(lambda: self.tool("negative"))
        panel.box.clicked.connect(lambda: self.tool("box"))
        panel.model.currentIndexChanged.connect(self.model_changed)
        panel.mode.currentIndexChanged.connect(self.mode_changed)
        for edit in (panel.weights, panel.tokenizer):
            edit.textChanged.connect(self.model_changed)
        panel.device.currentIndexChanged.connect(self.model_changed)
        for widget, signal in ((panel.text, "textChanged"), (panel.class_id, "valueChanged"),
                               (panel.kind, "currentIndexChanged"), (panel.area, "valueChanged"),
                               (panel.score, "valueChanged"), (panel.replace, "toggled")):
            getattr(widget, signal).connect(self.invalidate)
        self.view.prompts_changed.connect(self.invalidate)
        owner.auto_backend_combo.currentIndexChanged.connect(self.provider_changed)

    def identity(self):
        owner = self.owner
        path = owner.image_paths[owner.current_index] if 0 <= owner.current_index < len(owner.image_paths) else None
        return id(owner._lazy_cache), owner.current_index, path

    def invalidate(self, *_, cancel_load=False):
        self.revision += 1
        self.preview_key = None
        self.view.sam_preview = []
        self.view.viewport().update()
        self.panel.accept.setEnabled(False)
        if self.job is not None and (cancel_load or not self.loading):
            self.job.cancel()

    def frame_changed(self):
        if self.identity() != self.frame:
            self.frame = self.identity()
            self.stop_tool()
            self.view.clear_sam_prompts()

    def stop_tool(self):
        self.view.sam_tool = self.view._sam_start = None
        self.view.set_drawing_tool(None)

    def tool(self, name):
        if self.job is not None or not self.owner._current_frame_ready():
            return
        self.owner.reset_drawing_tool()
        self.view.set_edit_mode(False)
        self.view.setDragMode(self.view.NoDrag)
        self.view.setCursor(Qt.CrossCursor)
        self.view.sam_tool = name
        self.panel.status.setText({"positive": "Кликните внутри объекта; ПКМ добавляет фон.",
                                   "negative": "Кликните по области фона.",
                                   "box": "Протяните рамку вокруг объекта."}[name])

    def provider_changed(self):
        self.invalidate(cancel_load=True)
        self.stop_tool()
        self.view.clear_sam_prompts()
        self.owner.auto_stack.setCurrentIndex(self.owner.auto_backend_combo.currentIndex())

    def mode_changed(self):
        self.invalidate(cancel_load=True)
        self.stop_tool()
        self.view.clear_sam_prompts()
        self.panel.capabilities()
        if self.panel.model.currentData() == "sam3":
            self.panel.status.setText("Режим SAM3 изменён. Нажмите «Загрузить SAM».")

    def model_changed(self, *_):
        self.invalidate(cancel_load=True)
        self.panel.capabilities()

    def settings_key(self):
        panel = self.panel
        return (panel.model.currentData(), panel.weights.text().strip(),
                panel.device.currentText(), panel.tokenizer.text().strip(),
                panel.mode.currentData() if panel.model.currentData() == "sam3" else "points")

    def browse_weights(self):
        path, _ = source_dialogs.get_open_file(self.owner, "model_torch", directory=str(MODELS_DIR))
        if path:
            self.panel.weights.setText(path)
            for key, model in MODELS.items():
                if Path(path).name == Path(model.assets[0].path).name:
                    self.panel.model.setCurrentIndex(self.panel.model.findData(key))
                    break
            bpe = Path(path).parent / "bpe_simple_vocab_16e6.txt.gz"
            if bpe.is_file():
                self.panel.tokenizer.setText(str(bpe))

    def browse_tokenizer(self):
        path, _ = source_dialogs.get_open_file(self.owner, "sam_tokenizer", directory=str(MODELS_DIR))
        if path:
            self.panel.tokenizer.setText(path)

    def selected_box(self):
        index = self.view.selected_index
        if not 0 <= index < len(self.owner.current_annotations):
            self.panel.status.setText("Сначала выделите объект в списке или на снимке.")
            return
        ann = from_legacy(self.owner.current_annotations[index], ImageSize(self.owner.img_w, self.owner.img_h))
        self.view.sam_box = tuple(bounds(ann.points_px))
        self.panel.class_id.setValue(ann.class_id)
        self.view.prompts_changed.emit()
        self.view.viewport().update()

    def _start(self, compute, accept, status, *, loading=False):
        if self.closed or self.job is not None:
            return
        self.stop_tool()
        job = self.job = BackgroundTask(compute)
        self.loading = loading
        self.panel.controls.setEnabled(False)
        self.owner.auto_group.setEnabled(False)
        self.panel.progress.setRange(0, 0)
        self.panel.progress.setVisible(True)
        self.panel.cancel.setEnabled(True)
        self.panel.cancel.setVisible(True)
        self.panel.status.setText(status)

        def result(value):
            if not self.closed and self.job is job and not job.cancelled:
                try:
                    accept(value)
                except Exception as error:
                    failed(str(error))

        def failed(message):
            if not self.closed and self.job is job and not job.cancelled:
                self.panel.status.setText(message)
                self.owner.log(f"[SAM] {message}")
                QMessageBox.warning(self.owner, "SAM", message)

        def settled():
            if self.closed or self.job is not job:
                return
            self.job = None
            self.loading = False
            self.panel.controls.setEnabled(True)
            self.owner.auto_group.setEnabled(True)
            self.panel.capabilities()
            self.panel.progress.setVisible(False)
            self.panel.cancel.setEnabled(False)
            self.panel.cancel.setVisible(False)
            if job.cancelled:
                self.panel.status.setText("Вычисление отменено; результат не применён.")
            job.compute = None  # Release the worker closure's model/image references.
            job.deleteLater()

        job.result_ready.connect(result, Qt.QueuedConnection)
        job.failed.connect(failed, Qt.QueuedConnection)
        job.settled.connect(settled, Qt.QueuedConnection)
        job.start()

    def load(self):
        if self.job is not None:
            return
        self.invalidate()
        key = self.settings_key()
        self.backend = None

        def accept(backend):
            if key == self.settings_key():
                self.backend = backend
                self.loaded_key = key
                self.panel.status.setText(f"Модель загружена: {key[0]} ({backend.device}). Укажите подсказки.")
                self.owner.log(f"[SAM] Локальные веса: {key[1]}; архитектура: {key[0]}; устройство: {backend.device}")

        self._start(lambda cancel: LocalSAM.load(key[0], key[1], device=key[2], tokenizer=key[3] or None, prompt_mode=key[4]),
                    accept, "Загрузка локальной модели SAM…", loading=True)

    def run(self):
        if self.job is not None:
            return
        if self.backend is None or self.settings_key() != getattr(self, "loaded_key", None):
            self.panel.status.setText("Сначала загрузите модель с выбранными параметрами.")
            return
        if not self.owner._current_frame_ready():
            self.panel.status.setText("Дождитесь загрузки текущего снимка.")
            return
        image = self.owner._lazy_cache.get(self.owner.current_index)["image"]
        panel = self.panel
        prompt = SamPrompt(tuple(self.view.sam_points), tuple(self.view.sam_labels), self.view.sam_box,
                           panel.text.text() if panel.mode.currentData() == "text" else "")
        if panel.mode.currentData() == "text" and not prompt.text.strip():
            panel.status.setText("Введите текстовый запрос или выберите режим «Точки и рамка» и загрузите модель заново.")
            return
        try:
            prompt.validated(image.shape[1], image.shape[0], supports_text=self.backend.model_key == "sam3")
        except ValueError as error:
            panel.status.setText(str(error))
            return
        self.invalidate()
        key = (self.identity(), self.revision, id(image))
        backend = self.backend
        options = dict(class_id=panel.class_id.value(), kind=panel.kind.currentData(), min_area=panel.area.value())
        min_score = panel.score.value()

        def compute(cancel):
            result = backend.predict(image, prompt, image_key=key[0], min_score=min_score, cancel=cancel)
            return result.annotations(**options), result.scores

        def accept(value):
            if key[:2] != (self.identity(), self.revision):
                return
            annotations, scores = value
            self.preview_key = key
            self.view.sam_preview = annotations
            self.view.viewport().update()
            panel.accept.setEnabled(bool(annotations))
            score_text = ", ".join(f"{score:.3f}" for score in scores[:10])
            panel.status.setText(f"Предпросмотр: {len(annotations)} объектов. Оценки: {score_text or '—'}. "
                                 "Проверьте контуры и нажмите «Принять результат».")

        self._start(compute, accept, "Вычисление маски SAM…")

    def accept(self):
        if (self.preview_key is None or self.preview_key[:2] != (self.identity(), self.revision)
                or not self.owner._current_frame_ready()):
            return
        image = self.owner._lazy_cache.get(self.owner.current_index)["image"]
        if id(image) != self.preview_key[2]:
            self.invalidate()
            return
        annotations = list(self.view.sam_preview)
        if self.panel.replace.isChecked() and QMessageBox.question(
                self.owner, "SAM", "Заменить все существующие аннотации текущего снимка?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.owner._apply_auto_annotations(annotations, self.panel.replace.isChecked())
        self.owner.log(f"[SAM] Принято объектов: {len(annotations)}; класс: {self.panel.class_id.value()}")
        self.view.clear_sam_prompts()
        self.stop_tool()
        self.panel.status.setText(f"Принято объектов: {len(annotations)}. Сохраните разметку кнопкой Save Labels.")

    def cancel(self):
        self.invalidate(cancel_load=True)
        self.panel.status.setText("Отмена… Текущая операция модели завершится в фоне.")

    def close(self):
        self.closed = True
        if self.job is not None:
            self.job.close()
        self.backend = None
