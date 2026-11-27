"""Shared Pose / Classification editor and viewer on the canonical graphics view."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QBrush, QKeySequence
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QListWidget,
    QAbstractItemView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QPlainTextEdit,
    QSpinBox,
    QStackedWidget,
    QShortcut,
    QProgressBar,
    QScrollArea,
    QHeaderView,
    QDoubleSpinBox,
)

from ui.image_view import SmartGraphicsViewNew
from ui.image_conversion import numpy_to_qpixmap
from ui.image_navigation import ImageNavigationWidget
from ui.log_widget import LogWidget
from ui.segmentation.jobs import BackgroundTask
from ui import dataset_source as sources
from utils_new.annotations.model import (
    Annotation,
    PoseAnnotation,
    PoseSchema,
    ImageClassification,
    ImageSize,
)
from utils_new.annotations.adapters import to_legacy, from_legacy
from utils_new.annotations.geometry import rectangle_points
from utils_new.import_export.task_datasets import (
    TaskDataset,
    TaskSample,
    load_task_dataset,
    save_task_project,
    export_task_dataset,
    classes_dict,
    merge_task_datasets,
    check,
)
from utils_new.import_export.images import read_image_with_fallback, collect_image_paths


class PoseGraphicsView(SmartGraphicsViewNew):
    point_started = pyqtSignal()
    point_moved = pyqtSignal(int, float, float, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_mode = False
        self.pose_data = []
        self.schema = None
        self.active_point = 0
        self._drag_point = None
        self.show_links = True
        self.show_names = True

    def mousePressEvent(self, event):
        if (
            self.point_mode
            and event.button() == Qt.LeftButton
            and self.selected_index >= 0
        ):
            pos = self.mapToScene(event.pos())
            if not (0 <= pos.x() <= self.img_width and 0 <= pos.y() <= self.img_height):
                return
            points = self.pose_data[self.selected_index].keypoints
            hits = [
                (np.hypot(x - pos.x(), y - pos.y()), i)
                for i, (x, y, v) in enumerate(points)
                if v > 0
            ]
            hit = min(hits, default=(float("inf"), 0))
            self._placing = hit[0] > 10 * self._scene_per_px()
            self._drag_point = self.active_point if self._placing else hit[1]
            self.point_started.emit()
            self.point_moved.emit(self._drag_point, pos.x(), pos.y(), False)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_point is not None:
            pos = self.mapToScene(event.pos())
            self.point_moved.emit(
                self._drag_point,
                float(np.clip(pos.x(), 0, self.img_width)),
                float(np.clip(pos.y(), 0, self.img_height)),
                False,
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_point is not None and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            index = self._drag_point
            self._drag_point = None
            self.point_moved.emit(
                index,
                float(np.clip(pos.x(), 0, self.img_width)),
                float(np.clip(pos.y(), 0, self.img_height)),
                self._placing,
            )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self.schema is None:
            return
        scale = self._scene_per_px()
        r = 4 * scale
        for obj, pose in enumerate(self.pose_data):
            pen = QPen(QColor("#00bcd4"))
            pen.setCosmetic(True)
            pen.setWidth(2)
            painter.setPen(pen)
            if self.show_links:
                for a, b in self.schema.edges:
                    if pose.keypoints[a, 2] > 0 and pose.keypoints[b, 2] > 0:
                        from PyQt5.QtCore import QLineF

                        painter.drawLine(
                            QLineF(*pose.keypoints[a, :2], *pose.keypoints[b, :2])
                        )
            for i, (x, y, v) in enumerate(pose.keypoints):
                if not v:
                    continue
                color = QColor(
                    "#ff5252"
                    if obj == self.selected_index and i == self.active_point
                    else "#ffc107" if v == 1 else "#00bcd4"
                )
                pen = QPen(color)
                pen.setCosmetic(True)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QBrush(color) if v == 2 else QBrush(Qt.NoBrush))
                from PyQt5.QtCore import QRectF, QPointF

                painter.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
                if self.show_names:
                    font = painter.font()
                    font.setPixelSize(max(1, round(11 * scale)))
                    painter.setFont(font)
                    painter.drawText(
                        QPointF(x + r, y - r), f"{i+1}: {self.schema.names[i]}"
                    )


class PoseSchemaDialog(QDialog):
    def __init__(self, schema, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Шаблон Pose — любые ориентиры объекта")
        self.resize(620, 520)
        layout = QFormLayout(self)
        self.count = QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(len(schema.names))
        self.table = QTableWidget(len(schema.names), 2)
        self.table.setHorizontalHeaderLabels(
            ["Имя точки", "Зеркальная точка (номер с 1)"]
        )
        self.mirror = QCheckBox("Использовать отражение с перестановкой точек")
        self.mirror.setChecked(schema.flip_idx is not None)
        for i, name in enumerate(schema.names):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(
                i,
                1,
                QTableWidgetItem(
                    str((schema.flip_idx[i] if schema.flip_idx else i) + 1)
                ),
            )
        self.links = QLineEditCompat(", ".join(f"{a+1}-{b+1}" for a, b in schema.edges))
        self.dims = QComboBox()
        self.dims.addItems(["3 — x,y,видимость", "2 — только x,y"])
        self.dims.setCurrentIndex(0 if schema.dims == 3 else 1)
        layout.addRow("Число точек:", self.count)
        layout.addRow(self.table)
        layout.addRow("Связи, например 1-2, 2-3:", self.links)
        layout.addRow(self.mirror)
        layout.addRow("Формат YOLO:", self.dims)
        note = QLabel(
            "Одна схема на датасет. Для разных объектов можно создавать отдельные шаблоны/проекты. Без flip_idx горизонтальное отражение при обучении нужно выключить."
        )
        note.setWordWrap(True)
        layout.addRow(note)
        self.count.valueChanged.connect(self.resize_points)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def resize_points(self, n):
        self.table.setRowCount(n)
        for i in range(n):
            if not self.table.item(i, 0):
                self.table.setItem(i, 0, QTableWidgetItem(f"point_{i+1}"))
            if not self.table.item(i, 1):
                self.table.setItem(i, 1, QTableWidgetItem(str(i + 1)))

    def value(self):
        names = tuple(
            self.table.item(i, 0).text().strip() for i in range(self.count.value())
        )
        flip = (
            tuple(int(self.table.item(i, 1).text()) - 1 for i in range(len(names)))
            if self.mirror.isChecked()
            else None
        )
        edges = tuple(
            tuple(int(i.strip()) - 1 for i in pair.split("-"))
            for pair in self.links.text().split(",")
            if pair.strip()
        )
        return PoseSchema(names, edges, flip, 3 if self.dims.currentIndex() == 0 else 2)

    def validate_accept(self):
        try:
            self.value()
            self.accept()
        except (ValueError, AttributeError) as error:
            QMessageBox.warning(self, "Шаблон", str(error))


from PyQt5.QtWidgets import QLineEdit as QLineEditCompat


class TaskAnnotationWidget(QWidget):
    progress = pyqtSignal(int, int)

    def __init__(self, task, editable=True, parent=None):
        super().__init__(parent)
        self.task = task
        self.dataset = TaskDataset(
            task,
            {0: "object"} if task == "pose" else {0: "normal", 1: "defect"},
            schema=(
                PoseSchema(("point_1", "point_2", "point_3"))
                if task == "pose"
                else None
            ),
        )
        self.index = 0
        self.image = None
        self.job = None
        self.model = None
        self.dirty = False
        self._closed = False
        self._history = {}
        self._redo = {}
        self.project_path = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.nav = ImageNavigationWidget(
            source_actions=(
                ("images", "folder", "labels", "task_project")
                if task == "pose"
                else ("images", "folder", "classification_dataset", "task_project")
            )
        )
        if task == "pose":
            self.nav.source_widget.button("labels").clicked.connect(self.choose_labels)
        else:
            self.nav.source_widget.button("classification_dataset").clicked.connect(
                self.choose_classification_dataset
            )
        self.nav._resize_cb.hide()  # Native source coordinates; graphics handles display scaling.
        layout.addWidget(self.nav)
        row = QHBoxLayout()
        layout.addLayout(row)
        model_row = QHBoxLayout()
        layout.addLayout(model_row)
        self.append = QCheckBox("Добавлять к набору")
        row.addWidget(self.append)
        self.edit = QCheckBox("Редактировать")
        self.edit.setChecked(editable)
        row.addWidget(self.edit)
        self.buttons = {}
        for key, text, callback in [
            ("new", "Новый", self.new_dataset),
            ("classes", "Классы…", self.edit_classes),
            ("schema", "Шаблон Pose…", self.edit_schema),
            ("save", "Сохранить проект…", self.save_project),
            ("export", "Экспорт YOLO…", self.export_yolo),
            ("coco", "Экспорт COCO…", lambda: self.export_yolo(True)),
            ("model", "Модель YOLO…", self.load_model),
            ("predict", "Предложить разметку", self.predict),
            ("adopt", "Классы / шаблон из модели", self.adopt_model_schema),
        ]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            (model_row if key in ("model", "predict", "adopt") else row).addWidget(
                button
            )
            self.buttons[key] = button
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "0"])
        self.device.currentTextChanged.connect(self.invalidate_model)
        model_row.addWidget(QLabel("Устройство:"))
        model_row.addWidget(self.device)
        self.kpt_conf = QDoubleSpinBox()
        self.kpt_conf.setRange(0, 1)
        self.kpt_conf.setSingleStep(0.05)
        self.kpt_conf.setValue(0.5)
        self.kpt_conf.setToolTip(
            "Порог достоверности предсказанной точки. Это не ground-truth visibility."
        )
        model_row.addWidget(self.kpt_conf)
        self.kpt_conf.setVisible(task == "pose")
        self.buttons["schema"].setVisible(task == "pose")
        self.buttons["coco"].setVisible(task == "pose")
        body = QSplitter()
        layout.addWidget(body, 1)
        self.view = PoseGraphicsView()
        self.view.set_class_name_provider(
            lambda cls: self.dataset.classes.get(cls, str(cls))
        )
        body.addWidget(self.view)
        right = QWidget()
        panel = QVBoxLayout(right)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        body.addWidget(right_scroll)
        body.setSizes([850, 400])
        self.class_combo = QComboBox()
        panel.addWidget(
            QLabel("Класс объекта" if task == "pose" else "Класс всего снимка")
        )
        panel.addWidget(self.class_combo)
        self.assign = QPushButton(
            "Назначить выбранным снимкам" if task == "classify" else "Назначить объекту"
        )
        panel.addWidget(self.assign)
        self.clear_class = QPushButton("Снять класс")
        self.clear_class.setVisible(task == "classify")
        panel.addWidget(self.clear_class)
        self.review = QCheckBox("Разметка проверена")
        panel.addWidget(self.review)
        self.split = QComboBox()
        self.split.addItems(["train", "val", "test"])
        panel.addWidget(QLabel("Split выбранных снимков"))
        panel.addWidget(self.split)
        self.assign_split = QPushButton("Назначить split")
        panel.addWidget(self.assign_split)
        self.filter = QComboBox()
        panel.addWidget(self.filter)
        self.filter.currentIndexChanged.connect(self.apply_filter)
        self.images = QListWidget()
        self.images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        panel.addWidget(self.images, 1)
        tools = QHBoxLayout()
        panel.addLayout(tools)
        self.box = QPushButton("Рамка")
        self.points = QPushButton("Точки")
        self.pan = QPushButton("Панорама")
        self.bbox_edit = QPushButton("Правка BBox")
        tools.addWidget(self.bbox_edit)
        self.bbox_edit.setVisible(task == "pose")
        self.bbox_edit.clicked.connect(lambda: self.set_tool("bbox_edit"))
        self.remove = QPushButton("Удалить")
        for button in (self.box, self.points, self.pan, self.remove):
            tools.addWidget(button)
            button.setVisible(task == "pose")
        self.objects = QListWidget()
        self.objects.setMaximumHeight(120)
        panel.addWidget(self.objects)
        self.objects.setVisible(task == "pose")
        self.table = QTableWidget(0, 4)
        self.table.setToolTip(
            "Видимость: 0 — не размечена, 1 — координата известна, но точка перекрыта, 2 — видна."
        )
        self.table.setHorizontalHeaderLabels(["Точка", "X px", "Y px", "Видимость"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        panel.addWidget(self.table)
        self.table.setVisible(task == "pose")
        point_actions = QHBoxLayout()
        panel.addLayout(point_actions)
        self.visibility_buttons = []
        for visibility, caption in ((0, "Пропустить"), (1, "Перекрыта"), (2, "Видна")):
            button = QPushButton(caption)
            button.setVisible(task == "pose")
            point_actions.addWidget(button)
            self.visibility_buttons.append(button)
            button.clicked.connect(lambda _, v=visibility: self.set_point_visibility(v))
        self.links = QCheckBox("Показывать связи")
        self.links.setChecked(True)
        panel.addWidget(self.links)
        self.links.setVisible(task == "pose")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.current_label = QLabel()
        layout.addWidget(self.current_label)
        self.status = QLabel(
            "Загрузите снимки, датасет или проект. Изменения сохраняются в JSON проекта; исходные снимки и TXT не перезаписываются."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
        self.log_widget.setMaximumHeight(140)
        self.log_widget.hide()
        layout.addWidget(self.log_widget)
        self.log_button = QPushButton("Лог")
        self.log_button.setCheckable(True)
        self.log_button.toggled.connect(self.log_widget.setVisible)
        model_row.addWidget(self.log_button)
        foot = QHBoxLayout()
        layout.addLayout(foot)
        self.bar = QProgressBar()
        foot.addWidget(self.bar)
        self.cancel = QPushButton("Отмена")
        foot.addWidget(self.cancel)
        self.cancel.clicked.connect(self.cancel_job)
        self.cancel.setEnabled(False)
        self.progress.connect(self.show_progress)
        self.nav.load_images.connect(self.choose_images)
        self.nav.load_folder.connect(self.choose_folder)
        self.nav.source_widget.button("task_project").clicked.connect(
            self.choose_source
        )
        self.nav.prev.connect(lambda: self.step(-1))
        self.nav.next.connect(lambda: self.step(1))
        self.nav.goto_page.connect(lambda n: self.navigate(n - 1))
        self.images.currentRowChanged.connect(self.navigate)
        self.assign.clicked.connect(self.assign_class)
        self.clear_class.clicked.connect(lambda: self.assign_class(None, True))
        self.assign_split.clicked.connect(self.set_split)
        self.review.toggled.connect(self.set_review)
        self.objects.currentRowChanged.connect(self.select_object)
        self.table.cellChanged.connect(self.point_edited)
        self.table.currentCellChanged.connect(lambda row, *_: self.select_point(row))
        self.box.clicked.connect(lambda: self.set_tool("rect"))
        self.points.clicked.connect(lambda: self.set_tool("points"))
        self.pan.clicked.connect(lambda: self.set_tool(None))
        self.remove.clicked.connect(self.delete_pose)
        self.point_names = QCheckBox("Подписи точек")
        self.point_names.setChecked(True)
        self.point_names.setVisible(task == "pose")
        panel.addWidget(self.point_names)
        self.point_names.toggled.connect(
            lambda value: setattr(self.view, "show_names", value)
            or self.view.viewport().update()
        )
        self.links.toggled.connect(self.toggle_links)
        self.edit.toggled.connect(self.update_editability)
        self.view.set_callbacks(
            on_rect_drawn=self.add_bbox,
            on_selection_changed=self.select_object,
            on_annotation_modified=self.modify_bbox,
        )
        self.view.point_started.connect(self.remember)
        self.view.point_moved.connect(self.move_point)
        for key, callback in [
            ("Ctrl+Z", self.undo),
            ("Ctrl+Y", lambda: self.undo(True)),
            ("Delete", self.delete_pose),
        ]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
        self.refresh_classes()
        self.update_editability()

    def error(self, error):
        self.status.setText(str(error))
        self.log_widget.log(str(error))
        QMessageBox.warning(self, "Разметка", str(error))

    def show_progress(self, n, total):
        self.bar.setRange(0, total if total else 0)
        self.bar.setValue(n)

    def run_job(self, label, compute, accept):
        self.cancel_job()
        worker = BackgroundTask(compute)
        self.job = worker
        self.setEnabled(True)
        self.cancel.setEnabled(True)
        self.bar.setRange(0, 0)
        self.status.setText(label)
        self.nav.setEnabled(False)
        self.edit.setEnabled(False)
        self.append.setEnabled(False)
        self.device.setEnabled(False)
        self.kpt_conf.setEnabled(False)
        for b in self.buttons.values():
            b.setEnabled(False)
        self.images.setEnabled(False)
        self.view.setEnabled(False)
        self.right_enabled(False)

        def finish():
            if self.job is not worker or self._closed:
                return
            self.job = None
            self.bar.setRange(0, 100)
            self.cancel.setEnabled(False)
            self.nav.setEnabled(True)
            self.images.setEnabled(True)
            self.view.setEnabled(True)
            self.update_editability()

        def result(value):
            if self.job is not worker or self._closed:
                return
            finish()
            try:
                accept(value)
            except Exception as error:
                self.error(error)

        def fail(message):
            if self.job is not worker or self._closed:
                return
            finish()
            self.error(message)

        worker.result_ready.connect(result)
        worker.failed.connect(fail)
        worker.settled.connect(lambda: finish())
        worker.start()

    def right_enabled(self, enabled):
        for w in (
            self.assign,
            self.clear_class,
            self.assign_split,
            self.review,
            self.class_combo,
            self.split,
            self.box,
            self.points,
            self.bbox_edit,
            self.remove,
            self.table,
            *self.visibility_buttons,
        ):
            w.setEnabled(enabled)

    def cancel_job(self):
        if self.job:
            self.job.close()
            self.job = None
        self.bar.setRange(0, 100)
        self.cancel.setEnabled(False)
        self.nav.setEnabled(True)
        self.images.setEnabled(True)
        self.view.setEnabled(True)
        self.update_editability()

    def close_jobs(self):
        self._closed = True
        self.cancel_job()

    def update_editability(self):
        self.edit.setEnabled(self.job is None)
        self.append.setEnabled(self.job is None)
        self.device.setEnabled(self.job is None)
        self.kpt_conf.setEnabled(self.job is None)
        enabled = self.edit.isChecked() and self.job is None
        self.right_enabled(enabled)
        for key, b in self.buttons.items():
            b.setEnabled(
                self.job is None
                and (enabled or key in ("save", "export", "coco", "model"))
            )
        self.buttons["adopt"].setEnabled(enabled and self.model is not None)
        self.buttons["predict"].setEnabled(
            enabled and self.model is not None and self.image is not None
        )
        self.pan.setEnabled(self.job is None)
        if self.task == "pose":
            for w in (self.box, self.points, self.bbox_edit, self.remove, self.table):
                w.setEnabled(enabled and self.image is not None)
        if not enabled:
            self.set_tool(None)

    def refresh_classes(self):
        self.class_combo.clear()
        for i, name in self.dataset.classes.items():
            self.class_combo.addItem(f"{i}: {name}", i)
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.addItem("Все снимки", None)
        self.filter.addItem("Без разметки", "unlabelled")
        self.filter.addItem("Не проверены", "unreviewed")
        if self.task == "classify":
            for i, name in self.dataset.classes.items():
                self.filter.addItem(name, i)
        self.filter.blockSignals(False)

    def accept_dataset(self, data):
        self.dataset = data
        self.index = 0
        self.image = None
        self.view.pose_data = []
        self.view.set_annotations([], 0, 0)
        self.view.set_pixmap(numpy_to_qpixmap(None))
        self.objects.clear()
        self.table.setRowCount(0)
        self.nav.set_current_index(0, len(data.samples))
        self.project_path = ""
        self._history.clear()
        self._redo.clear()
        self.dirty = False
        self.refresh_classes()
        self.refresh_images()
        self.navigate(0)
        self.log_widget.log(
            f'Источник: {data.source or "снимки"}; задача: {data.task}; файлов: {len(data.samples)}; классы: {data.classes}; схема: {data.schema}'
        )
        self.status.setText(
            f'Источник: {data.source or "снимки"}. Загружено: {len(data.samples)}; классов: {len(data.classes)}. Pose TXT читаются при открытии снимка.'
        )

    def discard_ok(self):
        return (
            not self.dirty
            or QMessageBox.question(
                self,
                "Несохранённая разметка",
                "Продолжить без сохранения изменений проекта?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            == QMessageBox.Yes
        )

    def new_dataset(self):
        if self.discard_ok():
            self.accept_dataset(
                TaskDataset(self.task, self.dataset.classes, schema=self.dataset.schema)
            )

    def choose_images(self):
        paths, _ = sources.get_open_file(self, "images", multiple=True)
        if paths:
            self.load_images(paths)

    def choose_folder(self):
        path = sources.get_directory(self, "dataset_root")
        if not path:
            return
        self.run_job(
            "Поиск снимков…",
            lambda cancel: collect_image_paths(path, recursive=False),
            self.load_images,
        )

    def choose_classification_dataset(self):
        path = sources.get_directory(self, "classification_dataset")
        if path:
            self.load_source(path)

    def load_images(self, paths):
        if not paths:
            self.status.setText(
                "Снимки не найдены; текущий набор сохранён. Для папок классов используйте «Датасет CLS…»."
            )
            return
        if not self.append.isChecked() and not self.discard_ok():
            return
        incoming = TaskDataset(
            self.task,
            self.dataset.classes,
            [TaskSample(str(Path(p).resolve())) for p in dict.fromkeys(paths)],
            self.dataset.schema,
        )

        def accepted(data):
            self.accept_dataset(data)
            self.dirty = True

        if self.append.isChecked():
            current = deepcopy(self.dataset)
            self.run_job(
                "Добавление снимков…",
                lambda cancel: merge_task_datasets(current, incoming, cancel=cancel),
                accepted,
            )
        else:
            accepted(incoming)

    def choose_source(self):
        path, _ = sources.get_open_file(self, "task_project")
        if not path:
            return
        if self.task == "pose" and Path(path).suffix.lower() in (".yaml", ".yml"):
            approved = sources.confirm_dataset_import(self, [path], "yolo")
            if not approved:
                return
            path = approved[0]
        if self.task == "pose" and Path(path).suffix.lower() == ".json":
            with open(path, encoding="utf-8-sig") as stream:
                header = stream.read(256)
            if "neuro-task-dataset-v1" not in header:
                approved = sources.confirm_dataset_import(self, [path], "coco")
                if not approved:
                    return
                path = approved[0]
        split = None
        if self.task == "pose" and Path(path).suffix.lower() == ".json":
            with open(path, encoding="utf-8-sig") as stream:
                header = stream.read(256)
            if "neuro-task-dataset-v1" not in header:
                split, ok = QInputDialog.getItem(
                    self,
                    "Split COCO",
                    "Назначить снимкам загружаемого COCO:",
                    ["train", "val", "test"],
                    0,
                    False,
                )
                if not ok:
                    return
        self.load_source(path, split)

    def load_source(self, path, split=None):
        append = self.append.isChecked() and bool(self.dataset.samples)
        if not append and not self.discard_ok():
            return
        current = deepcopy(self.dataset) if append else None

        def compute(cancel):
            data = load_task_dataset(
                path, self.task, cancel=cancel, progress=self.progress.emit
            )
            if split is not None:
                for sample in data.samples:
                    sample.split = split
            return (
                merge_task_datasets(current, data, cancel=cancel) if current else data
            )

        def accepted(data):
            self.accept_dataset(data)
            if append:
                self.dirty = True

        self.run_job("Загрузка структуры датасета…", compute, accepted)

    def refresh_images(self):
        selected = [self.images.row(item) for item in self.images.selectedItems()]
        current = self.index
        self.images.blockSignals(True)
        self.images.clear()
        for sample in self.dataset.samples:
            label = (
                self.dataset.classes.get(sample.classification.class_id, "не размечен")
                if self.task == "classify"
                else (
                    f"{len(sample.poses)} объектов"
                    if sample.poses is not None
                    else "TXT" if sample.label_path else "без разметки"
                )
            )
            self.images.addItem(
                f"{Path(sample.path).name} · {sample.split} · {label}"
                + (" ✓" if sample.classification.reviewed else "")
            )
        from PyQt5.QtCore import QItemSelectionModel

        self.images.setCurrentRow(current, QItemSelectionModel.NoUpdate)
        for row in selected:
            if row < self.images.count():
                self.images.item(row).setSelected(True)
        self.images.blockSignals(False)
        self.apply_filter()
        from collections import Counter

        splits = Counter(s.split for s in self.dataset.samples)
        reviewed = sum(s.classification.reviewed for s in self.dataset.samples)
        self.update_current_label()
        self.summary.setText(
            f'Снимков: {len(self.dataset.samples)} · train: {splits["train"]} · val: {splits["val"]} · test: {splits["test"]} · проверено: {reviewed}'
        )

    def update_current_label(self):
        if self.dataset.samples:
            self.review.blockSignals(True)
            self.review.setChecked(
                self.dataset.samples[self.index].classification.reviewed
            )
            self.review.blockSignals(False)
        if self.task == "classify" and self.dataset.samples:
            sample = self.dataset.samples[self.index]
            self.current_label.setText(
                "Класс снимка: "
                + self.dataset.classes.get(
                    sample.classification.class_id, "НЕ РАЗМЕЧЕН"
                )
            )
        else:
            self.current_label.clear()

    def apply_filter(self):
        if not hasattr(self, "images"):
            return
        value = self.filter.currentData()
        for i, s in enumerate(self.dataset.samples):
            unlabelled = (
                (s.classification.class_id is None)
                if self.task == "classify"
                else not s.has_pose_labels
            )
            show = (
                value is None
                or value == "unlabelled"
                and unlabelled
                or value == "unreviewed"
                and not s.classification.reviewed
                or type(value) is int
                and s.classification.class_id == value
            )
            self.images.item(i).setHidden(not show)
            if not show:
                self.images.item(i).setSelected(False)

    def step(self, direction):
        visible = [
            i
            for i in range(len(self.dataset.samples))
            if not self.images.item(i).isHidden() and (i - self.index) * direction > 0
        ]
        if visible:
            self.navigate(min(visible) if direction > 0 else max(visible))

    def navigate(self, index):
        if self.job is not None or not 0 <= index < len(self.dataset.samples):
            return
        self.index = index
        sample = self.dataset.samples[index]
        self.image = None
        self.view.pose_data = []
        self.view.set_pixmap(numpy_to_qpixmap(None))
        self.view.set_annotations([], 0, 0)
        self.objects.clear()
        self.table.setRowCount(0)

        snapshot = deepcopy(sample)
        dataset = self.dataset

        def compute(cancel):
            image = read_image_with_fallback(snapshot.path)
            if image is None:
                raise ValueError(f"Не удалось прочитать {sample.path}")
            poses = dataset.load_poses(snapshot) if self.task == "pose" else []
            return image, poses

        def accept(result):
            if self.task == "pose":
                sample.poses = result[1]
            self.image = result[0]
            sample.size = ImageSize(self.image.shape[1], self.image.shape[0])
            self.view.set_pixmap(numpy_to_qpixmap(self.image))
            self.nav.set_current_index(index, len(self.dataset.samples))
            self.images.blockSignals(True)
            self.images.setCurrentRow(index)
            self.images.blockSignals(False)
            self.review.blockSignals(True)
            self.review.setChecked(sample.classification.reviewed)
            self.review.blockSignals(False)
            self.split.setCurrentText(sample.split)
            if sample.classification.class_id is not None:
                self.class_combo.setCurrentIndex(
                    self.class_combo.findData(sample.classification.class_id)
                )
            self.render(0)
            self.update_editability()
            self.update_current_label()
            self.status.setText(sample.path)

        self.run_job("Чтение снимка…", compute, accept)

    def render(self, selected=None):
        if not self.dataset.samples or self.image is None:
            self.view.set_pixmap(numpy_to_qpixmap(None))
            self.view.pose_data = []
            return
        sample = self.dataset.samples[self.index]
        poses = sample.poses or []
        if selected is None:
            selected = self.objects.currentRow()
        self.objects.blockSignals(True)
        self.objects.clear()
        for i, p in enumerate(poses):
            self.objects.addItem(
                f"{i+1}: {self.dataset.classes[p.bbox.class_id]} · {(p.keypoints[:,2]>0).sum()}/{len(p.keypoints)} точек"
            )
        self.objects.setCurrentRow(
            selected if selected is not None and selected < len(poses) else -1
        )
        self.objects.blockSignals(False)
        self.view.pose_data = poses
        self.view.schema = self.dataset.schema
        self.view.set_annotations(
            [to_legacy(p.bbox, sample.size) for p in poses],
            sample.size.width,
            sample.size.height,
        )
        self.view.set_selected_index(self.objects.currentRow())
        self.fill_points()
        self.update_current_label()
        self.view.viewport().update()

    def select_object(self, index):
        self.objects.blockSignals(True)
        self.objects.setCurrentRow(index)
        self.objects.blockSignals(False)
        self.view.set_selected_index(index)
        if self.task == "pose" and 0 <= index < len(self.view.pose_data):
            self.class_combo.setCurrentIndex(
                self.class_combo.findData(self.view.pose_data[index].bbox.class_id)
            )
        self.fill_points()
        self.view.viewport().update()

    def fill_points(self):
        row = max(0, self.table.currentRow())
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        i = self.objects.currentRow()
        if (
            self.task == "pose"
            and self.dataset.samples
            and 0 <= i < len(self.view.pose_data)
        ):
            pose = self.view.pose_data[i]
            self.table.setRowCount(len(pose.keypoints))
            for k, (x, y, v) in enumerate(pose.keypoints):
                for col, text in enumerate(
                    (self.dataset.schema.names[k], f"{x:.3f}", f"{y:.3f}", str(int(v)))
                ):
                    item = QTableWidgetItem(text)
                    if col == 0:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(k, col, item)
            self.table.setCurrentCell(min(row, len(pose.keypoints) - 1), 0)
        self.table.blockSignals(False)

    def select_point(self, row):
        self.view.active_point = max(0, row)
        self.view.viewport().update()

    def toggle_links(self, value):
        self.view.show_links = value
        self.view.viewport().update()

    def set_tool(self, tool):
        self.view.point_mode = tool == "points" and self.edit.isChecked()
        self.view.set_drawing_tool(
            "rect" if tool == "rect" and self.edit.isChecked() else None
        )
        self.view.set_edit_mode(tool == "bbox_edit" and self.edit.isChecked())

    def remember(self, sample=None):
        if (
            self.job is not None
            or not self.dataset.samples
            or not self.edit.isChecked()
        ):
            return
        sample = sample or self.dataset.samples[self.index]
        self._history.setdefault(sample.path, []).append(
            deepcopy(
                (sample.poses, sample.classification, sample.split, sample.pose_labeled)
            )
        )
        self._history[sample.path] = self._history[sample.path][-100:]
        self._redo[sample.path] = []
        sample.classification = replace(sample.classification, reviewed=False)
        self.dirty = True

    def undo(self, redo=False):
        if not self.edit.isChecked() or not self.dataset.samples or self.job:
            return
        sample = self.dataset.samples[self.index]
        src = self._redo if redo else self._history
        dst = self._history if redo else self._redo
        stack = src.get(sample.path, [])
        if not stack:
            return
        dst.setdefault(sample.path, []).append(
            deepcopy(
                (sample.poses, sample.classification, sample.split, sample.pose_labeled)
            )
        )
        sample.poses, sample.classification, sample.split, sample.pose_labeled = (
            stack.pop()
        )
        self.dirty = True
        self.refresh_images()
        self.render()
        self.review.blockSignals(True)
        self.review.setChecked(sample.classification.reviewed)
        self.review.blockSignals(False)

    def add_bbox(self, rect):
        if not self.edit.isChecked() or not self.dataset.samples or self.image is None:
            return
        sample = self.dataset.samples[self.index]
        self.remember()
        pose = PoseAnnotation(
            Annotation(
                "detect", self.class_combo.currentData(), rectangle_points(*rect)
            ),
            np.zeros((len(self.dataset.schema.names), 3)),
        )
        sample.poses.append(pose)
        sample.pose_labeled = True
        self.render(len(sample.poses) - 1)
        self.set_tool("points")

    def modify_bbox(self, index, value):
        if not self.edit.isChecked():
            return
        sample = self.dataset.samples[self.index]
        self.remember()
        old = sample.poses[index]
        sample.poses[index] = PoseAnnotation(
            from_legacy(value, sample.size), old.keypoints
        )
        self.render(index)

    def move_point(self, index, x, y, advance):
        obj = self.objects.currentRow()
        if obj < 0 or not self.edit.isChecked():
            return
        sample = self.dataset.samples[self.index]
        pose = sample.poses[obj]
        points = pose.keypoints.copy()
        points[index] = [x, y, 2]
        sample.poses[obj] = PoseAnnotation(pose.bbox, points)
        self.render(obj)
        self.table.setCurrentCell((index + 1) % len(points) if advance else index, 0)
        self.select_point(self.table.currentRow())
        self.dirty = True

    def point_edited(self, row, column):
        obj = self.objects.currentRow()
        if obj < 0 or not self.edit.isChecked():
            return
        try:
            values = [float(self.table.item(row, c).text()) for c in (1, 2, 3)]
            sample = self.dataset.samples[self.index]
            if not (
                0 <= values[0] <= sample.size.width
                and 0 <= values[1] <= sample.size.height
            ):
                raise ValueError("Точка вне снимка.")
            pose = sample.poses[obj]
            points = pose.keypoints.copy()
            points[row] = values
            updated = PoseAnnotation(pose.bbox, points)
            self.remember()
            sample.poses[obj] = updated
            self.render(obj)
        except (ValueError, AttributeError) as error:
            self.error(error)
            self.fill_points()

    def set_point_visibility(self, value):
        row = self.table.currentRow()
        obj = self.objects.currentRow()
        if row < 0 or obj < 0 or not self.edit.isChecked() or self.job:
            return
        self.remember()
        sample = self.dataset.samples[self.index]
        pose = sample.poses[obj]
        pts = pose.keypoints.copy()
        pts[row, 2] = value
        sample.poses[obj] = PoseAnnotation(pose.bbox, pts)
        self.render(obj)
        self.table.setCurrentCell((row + 1) % len(pts), 0)

    def delete_pose(self):
        obj = self.objects.currentRow()
        if self.task != "pose" or obj < 0 or not self.edit.isChecked() or self.job:
            return
        self.remember()
        self.dataset.samples[self.index].poses.pop(obj)
        self.dataset.samples[self.index].pose_labeled = True
        self.render(obj - 1)

    def selected_samples(self):
        rows = [self.images.row(item) for item in self.images.selectedItems()]
        return [
            self.dataset.samples[i]
            for i in (rows or ([self.index] if self.dataset.samples else []))
        ]

    def assign_class(self, value=False, clear=False):
        if not self.edit.isChecked() or self.job:
            return
        cls = None if clear else self.class_combo.currentData()
        if self.task == "pose":
            i = self.objects.currentRow()
            if i < 0:
                return
            self.remember()
            s = self.dataset.samples[self.index]
            p = s.poses[i]
            s.poses[i] = PoseAnnotation(
                Annotation("detect", cls, p.bbox.points_px), p.keypoints
            )
            self.render(i)
        else:
            for s in self.selected_samples():
                self.remember(s)
                s.classification = ImageClassification(cls, False)
            self.dirty = True
            self.refresh_images()

    def set_review(self, value):
        if self.dataset.samples and self.edit.isChecked() and self.job is None:
            for s in self.selected_samples():
                self.remember(s)
                s.classification = replace(s.classification, reviewed=value)
            self.refresh_images()

    def set_split(self):
        if not self.edit.isChecked() or self.job:
            return
        for s in self.selected_samples():
            self.remember(s)
            s.split = self.split.currentText()
        self.refresh_images()
        self.dirty = True

    def edit_classes(self):
        value, ok = QInputDialog.getMultiLineText(
            self,
            "Классы",
            "JSON: ID → имя",
            json.dumps(self.dataset.classes, ensure_ascii=False, indent=2),
        )
        if not ok:
            return
        try:
            classes = classes_dict(json.loads(value))
            if (
                any(
                    s.classification.class_id is not None
                    and s.classification.class_id not in classes
                    for s in self.dataset.samples
                )
                or any(s.label_path for s in self.dataset.samples)
                and not set(self.dataset.classes) <= set(classes)
                or any(
                    p.bbox.class_id not in classes
                    for s in self.dataset.samples
                    for p in (s.poses or [])
                )
            ):
                raise ValueError("Нельзя удалить класс, используемый разметкой.")
            self.dataset.classes = classes
            self.refresh_classes()
            self.refresh_images()
            self.render()
            self.dirty = True
        except (ValueError, TypeError, AttributeError) as error:
            self.error(error)

    def edit_schema(self):
        dialog = PoseSchemaDialog(self.dataset.schema, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            schema = dialog.value()
            if any(s.label_path or s.poses for s in self.dataset.samples) and len(
                schema.names
            ) != len(self.dataset.schema.names):
                raise ValueError(
                    "Для изменения числа точек создайте новый проект. Уже размеченные точки не удаляются автоматически."
                )
            self.dataset.schema = schema
            self.render()
            self.dirty = True
        except ValueError as error:
            self.error(error)

    def choose_labels(self):
        if not self.dataset.samples:
            return
        folder = sources.get_directory(self, "labels")
        if not folder:
            return
        from utils_new.import_export.images import sidecar_matches

        snapshot = deepcopy(self.dataset)

        def compute(cancel):
            cache = {}
            for sample in snapshot.samples:
                check(cancel)
                matches = sidecar_matches(
                    folder, Path(sample.path).name, cache, basename_fallback=True
                )
                if len(matches) > 1:
                    raise ValueError(
                        f"Несколько TXT для {sample.path}; используйте YAML с путями."
                    )
                sample.label_path = str(matches[0]) if matches else None
                sample.label_dims = snapshot.schema.dims
                sample.poses = None
                sample.pose_labeled = False
            return snapshot

        if self.discard_ok():

            def accepted(data):
                self.accept_dataset(data)
                self.dirty = True

            self.run_job("Поиск TXT…", compute, accepted)

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Проект разметки",
            self.project_path or f"{self.task}_project.json",
            "JSON (*.json)",
        )
        if not path:
            return
        snapshot = deepcopy(self.dataset)

        def saved(_):
            self.project_path = path
            self.dirty = False
            self.status.setText(f"Проект сохранён: {path}")

        self.run_job(
            "Сохранение проекта…",
            lambda cancel: save_task_project(
                snapshot, path, cancel=cancel, progress=self.progress.emit
            ),
            saved,
        )

    def export_yolo(self, coco=False):
        path, _ = QFileDialog.getSaveFileName(
            self, "Новая папка экспорта", f"{self.task}_export", "Папка (*)"
        )
        if not path:
            return
        snapshot = deepcopy(self.dataset)
        self.run_job(
            "Проверка и экспорт…",
            lambda cancel: export_task_dataset(
                snapshot, path, cancel=cancel, progress=self.progress.emit, coco=coco
            ),
            lambda value: self.status.setText(f"Экспорт завершён: {value}"),
        )

    def load_model(self):
        path, _ = sources.get_open_file(self, "model_yolo")
        if not path:
            return

        device = (
            None if self.device.currentText() == "auto" else self.device.currentText()
        )

        def compute(cancel):
            from utils_new.import_export.model_load import load_yolo

            model = load_yolo(path)
            model.overrides["device"] = device
            if model.task != self.task:
                raise ValueError(f"Нужна модель {self.task}, получена {model.task}.")
            # Exported runtimes may initialize on names access. Do this off-thread.
            _ = model.names
            return model

        def loaded(model):
            self.model = model
            self.update_editability()
            self.status.setText(
                f'Загружена {model.task}: {path}; классы: {model.names}; схема: {getattr(model,"neuro_metadata",{}).get("pose_schema")}'
            )
            self.buttons["model"].setStyleSheet("background: #27843b; color: white;")

        self.run_job("Загрузка YOLO…", compute, loaded)

    def invalidate_model(self, *_):
        if self.model is not None:
            self.model = None
            self.buttons["model"].setStyleSheet("")
            self.status.setText("Устройство изменено. Загрузите модель повторно.")
            self.update_editability()

    def adopt_model_schema(self):
        if not self.edit.isChecked() or self.job or self.model is None:
            return
        if any(
            s.label_path or s.poses or s.classification.class_id is not None
            for s in self.dataset.samples
        ):
            self.error(
                "Классы/шаблон модели можно принять только до разметки. Уже созданные аннотации не преобразуются автоматически."
            )
            return
        try:
            schema = None
            if self.task == "pose":
                value = getattr(self.model, "neuro_metadata", {}).get(
                    "pose_schema"
                ) or getattr(self.model.model, "neuro_pose_schema", None)
                if value:
                    schema = PoseSchema(**value)
                else:
                    shape = getattr(self.model.model, "kpt_shape", None)
                    if shape is None:
                        shape = getattr(self.model, "neuro_metadata", {}).get(
                            "kpt_shape"
                        )
                        if isinstance(shape, str):
                            import ast

                            shape = ast.literal_eval(shape)
                    if shape is None:
                        raise ValueError(
                            "Файл не содержит форму ключевых точек; задайте шаблон вручную."
                        )
                    names = getattr(self.model.model, "kpt_names", {})
                    names = (
                        next(iter(names.values()))
                        if names
                        else [f"point_{i+1}" for i in range(shape[0])]
                    )
                    schema = PoseSchema(tuple(names), dims=shape[1])
            self.dataset.classes = classes_dict(self.model.names)
            self.dataset.schema = schema
            self.refresh_classes()
            self.refresh_images()
            self.render()
            self.dirty = True
            self.status.setText(
                "Приняты классы и порядок точек модели. Связи можно настроить в шаблоне."
            )
        except (ValueError, AttributeError, TypeError) as error:
            self.error(error)

    def predict(self):
        if self.model is None or self.image is None or not self.edit.isChecked():
            return
        image = self.image.copy()
        model = self.model
        schema = self.dataset.schema
        classes = dict(self.dataset.classes)
        threshold = self.kpt_conf.value()
        device = (
            None if self.device.currentText() == "auto" else self.device.currentText()
        )

        def compute(cancel):
            meta_schema = getattr(model, "neuro_metadata", {}).get(
                "pose_schema"
            ) or getattr(getattr(model, "model", None), "neuro_pose_schema", None)
            if (
                self.task == "pose"
                and meta_schema
                and tuple(meta_schema["names"]) != schema.names
            ):
                raise ValueError(
                    "Имена / порядок точек модели не соответствуют шаблону проекта."
                )
            from utils_new.images.processing import ImageStats

            prepared = ImageStats.to_bgr3(ImageStats.to_uint8(image))
            result = model.predict(prepared, verbose=False, device=device)[0]
            if self.task == "classify":
                return (
                    "classify",
                    int(result.probs.top1),
                    float(result.probs.top1conf),
                    model.names,
                )
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.cls.cpu().numpy().astype(int)
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf
            scores = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2])
            if xy.shape[1] != len(schema.names):
                raise ValueError(
                    "Модель имеет другое число точек. Выберите совместимый шаблон."
                )
            name_map = {n: i for i, n in classes.items()}
            if any(model.names[int(cls)] not in name_map for cls in ids):
                raise ValueError(
                    "Имена классов модели не соответствуют проекту. Настройте классы перед применением."
                )
            ids = [name_map[model.names[int(cls)]] for cls in ids]
            return [
                PoseAnnotation(
                    Annotation("detect", int(cls), rectangle_points(*bbox)),
                    np.column_stack((points, np.where(score >= threshold, 2, 0))),
                )
                for cls, bbox, points, score in zip(ids, boxes, xy, scores)
            ]

        def proposed(value):
            if isinstance(value, tuple):
                _, cls, score, names = value
                name = names[cls]
                match = next(
                    (i for i, n in self.dataset.classes.items() if n == name), None
                )
                if match is None:
                    self.error(f"Класс модели «{name}» отсутствует в проекте.")
                    return
                message = f"Предложение: {name}, score={score:.3f}. Принять для снимка?"
            else:
                message = (
                    f"Найдено объектов: {len(value)}. Заменить Pose текущего снимка?"
                )
            if (
                QMessageBox.question(
                    self,
                    "Авторазметка",
                    message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
            self.remember()
            sample = self.dataset.samples[self.index]
            if isinstance(value, tuple):
                sample.classification = ImageClassification(match, False)
            else:
                sample.poses = value
                sample.pose_labeled = True
                sample.classification = replace(sample.classification, reviewed=False)
            self.refresh_images()
            self.render(0)

        self.run_job("Инференс…", compute, proposed)


class AnnotationTaskModes(QWidget):
    def __init__(self, owner, editable):
        super().__init__(owner)
        self.owner = owner
        geometry = owner.takeCentralWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.selector = QComboBox()
        self.selector.addItems(
            [
                "Объекты: BBox / OBB / Segment",
                "Pose — ключевые точки",
                "Classification — класс снимка",
            ]
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Задача:"))
        row.addWidget(self.selector, 1)
        layout.addLayout(row)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.stack.addWidget(geometry)
        self.pose = TaskAnnotationWidget("pose", editable, self)
        self.classify = TaskAnnotationWidget("classify", editable, self)
        self.stack.addWidget(self.pose)
        self.stack.addWidget(self.classify)
        self.selector.currentIndexChanged.connect(self.stack.setCurrentIndex)
        owner.setCentralWidget(self)

    def can_close(self):
        return all(pane.discard_ok() for pane in (self.pose, self.classify))

    def close_jobs(self):
        self.pose.close_jobs()
        self.classify.close_jobs()

    def route_source(self, path):
        try:
            import yaml

            cfg = (
                json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if Path(path).suffix.lower() == ".json"
                else yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
            )
            pose = isinstance(cfg, dict) and (
                "kpt_shape" in cfg
                or any(c.get("keypoints") for c in cfg.get("categories", []))
            )
        except (OSError, ValueError):
            return False
        if pose:
            self.selector.setCurrentIndex(1)
            self.pose.load_source(path)
        return pose
