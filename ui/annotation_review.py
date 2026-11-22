"""One configurable overlap review, used by dataset lists and object lists."""

from pathlib import Path

from PyQt5.QtCore import QSettings, QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QStyle, QTabWidget, QVBoxLayout, QWidget

from ui.dataset_audit.workers import AuditWorker
from utils_new.annotations.processing import find_strong_overlaps, format_annotation_display, normalized_annotation
from utils_new.dataset_audit.annotation_review import AnnotationReviewJob


def update_annotation_list(list_widget, annotations, image_width, image_height):
    """Shared editor/viewer rows; preserve source indices and signal blocking."""
    blocker = QSignalBlocker(list_widget)
    try:
        list_widget.clear()
        for index, annotation in enumerate(annotations):
            try:
                text = format_annotation_display(
                    index, normalized_annotation(annotation), image_width, image_height
                )
            except (TypeError, ValueError, IndexError, OverflowError):
                text = f"Obj {index + 1}: некорректная аннотация"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, index)
            list_widget.addItem(item)
    finally:
        blocker.unblock()


class ReviewWorker(AuditWorker):
    job_type = AnnotationReviewJob


class AnnotationReviewWidget(QWidget):
    image_selected = pyqtSignal(int)
    changed = pyqtSignal()

    def __init__(self, objects=None, external_images=None, parent=None):
        super().__init__(parent)
        self.objects = objects
        self.images = external_images if external_images is not None else QListWidget()
        self.external = external_images is not None
        self._pairs, self._records, self._errors, self._memory = [], {}, {}, {}
        self._worker = None
        self._current = None
        self._current_key = None
        self._settings = QSettings("NeuroThreshold", "annotation-review")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.metric = QComboBox()
        self.metric.addItem("Пересечение / меньший объект", "smaller")
        self.metric.addItem("IoU (пересечение / объединение)", "iou")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(1, 100)
        self.threshold.setSuffix(" %")
        self.threshold.setValue(float(self._settings.value("threshold", 70)))
        self.same_class = QCheckBox("Только одинаковые классы")
        self.same_class.setToolTip(
            "Сравнивать только пары аннотаций с одинаковым ID класса."
        )
        self.threshold.setToolTip(
            "Пара считается кандидатом, если выбранная доля пересечения не меньше порога."
        )
        self.metric.setToolTip(
            "Пересечение/меньший объект обнаруживает вложенные дубли. IoU строже: пересечение делится на объединение площадей."
        )
        self.same_class.setChecked(self._settings.value("same_class", False, type=bool))
        self.metric.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.metric.setMinimumContentsLength(18)
        self.metric.setCurrentIndex(
            max(0, self.metric.findData(self._settings.value("metric", "smaller")))
        )
        row.addWidget(self.metric, 1)
        row.addWidget(self.threshold)
        layout.addLayout(row)
        layout.addWidget(self.same_class)
        hint = QLabel(
            "Кандидат — не ошибка: вложенные объекты могут быть правильной разметкой. Проверка ничего не удаляет."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.summary = QLabel("Загрузите снимки для проверки перекрытий.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        if not self.external:
            self.tabs = QTabWidget()
            if objects is not None:
                self.tabs.addTab(objects, "Объекты")
            page = QWidget()
            box = QVBoxLayout(page)
            self.only_candidates = QCheckBox("Только кандидаты")
            box.addWidget(self.only_candidates)
            box.addWidget(self.images)
            self.tabs.addTab(page, "Снимки")
            layout.addWidget(self.tabs)
            self.only_candidates.toggled.connect(self._filter_images)
            self.images.currentRowChanged.connect(
                lambda row: self.image_selected.emit(row) if row >= 0 else None
            )
        self.metric.currentIndexChanged.connect(self._settings_changed)
        self.threshold.valueChanged.connect(self._settings_changed)
        self.same_class.toggled.connect(self._settings_changed)
        self.destroyed.connect(self.close_review)

    def options(self):
        return dict(
            threshold=self.threshold.value() / 100,
            metric=self.metric.currentData(),
            same_class=self.same_class.isChecked(),
        )

    def close_review(self, *_):
        worker = self._worker
        self._worker = None
        if worker:
            worker.close()

    def _settings_changed(self, *_):
        self._settings.setValue("threshold", self.threshold.value())
        self._settings.setValue("metric", self.metric.currentData())
        self._settings.setValue("same_class", self.same_class.isChecked())
        self._current_key = None
        self._records = {}
        self._errors = {}
        if self._current:
            self.set_current(*self._current)
        self._start_scan()
        self.changed.emit()

    def set_dataset(self, pairs, dataset_type=0, input_format="auto"):
        self.close_review()
        self._pairs = [tuple(pair) for pair in pairs]
        self.dataset_type = dataset_type
        self.input_format = input_format
        self._memory = {}
        self._current_key = None
        self._records = {}
        self._errors = {}
        self._current = None
        if not self.external:
            self.images.blockSignals(True)
            self.images.clear()
            for pair in pairs:
                item = QListWidgetItem(Path(pair[0]).name)
                item.setData(Qt.UserRole, str(pair[0]))
                item.setToolTip(str(pair[0]))
                self.images.addItem(item)
            self.images.blockSignals(False)
        self._start_scan()

    def _start_scan(self):
        self.close_review()
        if not self._pairs:
            self.summary.setText("Загрузите снимки для проверки перекрытий.")
            return
        self._worker = ReviewWorker(
            self._pairs,
            getattr(self, "dataset_type", 0),
            getattr(self, "input_format", "auto"),
            self.options(),
            self._memory,
        )
        self._worker.finished_signal.connect(self._receive)
        self._worker.progress.connect(self._progress)
        self._worker.start()

    def _progress(self, done, total):
        if self.sender() is self._worker:
            self.summary.setText(f"Проверка перекрытий: {done}/{total}")

    def _receive(self, result):
        if self.sender() is not self._worker or result.get("cancelled"):
            return
        if result.get("error"):
            self.summary.setText(
                "Проверка перекрытий не завершена: " + str(result["error"])
            )
            return
        self._records = result.get("records", {})
        self._errors = result.get("errors", {})
        for path, (annotations, size) in self._memory.items():
            self._records[path] = find_strong_overlaps(
                annotations, *size, **self.options()
            )
        self._decorate_images()
        self.summary.setText(
            f"Кандидаты: {sum(bool(v) for v in self._records.values())} снимков из {len(self._pairs)}. Не проверено: {len(self._errors)}."
        )

    def set_current(self, index, annotations, width, height):
        self._current = (index, list(annotations), width, height)
        key = (index, repr(annotations), width, height, tuple(self.options().values()))
        if self._current_key != key:
            self._current_key = key
            warnings = (
                find_strong_overlaps(annotations, width, height, **self.options())
                if width and height
                else {}
            )
            if 0 <= index < len(self._pairs):
                path = str(self._pairs[index][0])
                self._memory[path] = (list(annotations), (width, height))
                self._records[path] = warnings
            self.current_warnings = warnings
        self.decorate_objects(getattr(self, "current_warnings", {}))
        self._decorate_images()
        if not self._worker or not self._worker.isRunning():
            self.summary.setText(
                f"Кандидаты: {sum(bool(v) for v in self._records.values())} снимков из {len(self._pairs)}. Не проверено: {len(self._errors)}."
            )
        if not self.external and 0 <= index < self.images.count():
            self.images.blockSignals(True)
            self.images.setCurrentRow(index)
            self.images.blockSignals(False)
        return getattr(self, "current_warnings", {})

    def decorate_objects(self, warnings):
        if self.objects is None:
            return
        for index in range(self.objects.count()):
            item = self.objects.item(index)
            text = item.text().split("  [OVERLAP:")[0]
            if text.startswith("⚠ ") and "[INVALID:" not in text:
                text = text[2:]
            item.setText(text)
            if index in warnings:
                text += f"  [OVERLAP: ≥{self.threshold.value():g}%]"
                item.setText(("⚠ " if not text.startswith("⚠ ") else "") + text)
                if "[INVALID:" not in text:
                    item.setForeground(QColor(180, 65, 0))
                previous = item.toolTip() if "[INVALID:" in text else ""
                item.setToolTip(
                    (previous + "\n" if previous else "")
                    + "Кандидат в дубли: "
                    + ", ".join(
                        f"#{j + 1}: {score * 100:.1f}%" for j, score in warnings[index]
                    )
                )
            elif "[INVALID:" not in text:
                item.setForeground(
                    self.objects.palette().brush(self.objects.palette().Text)
                )
                item.setToolTip("")

    def _decorate_images(self):
        icon = self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        from PyQt5.QtGui import QIcon

        for row in range(self.images.count()):
            item = self.images.item(row)
            path = (
                str(item.data(Qt.UserRole))
                if self.external
                else str(self._pairs[row][0])
                if row < len(self._pairs)
                else ""
            )
            warnings = self._records.get(path, {})
            # Keep existing validation/split text and colours. An icon is independent.
            note = (
                "\nКандидаты в дубли: " + str(len(warnings)) + " объектов."
                if warnings
                else ""
            )
            tip = (
                item.toolTip()
                .split("\nКандидаты в дубли:")[0]
                .split("\nПроверка перекрытий:")[0]
            )
            if path in self._errors:
                note = "\nПроверка перекрытий: " + self._errors[path]
            item.setToolTip(tip + note)
            item.setIcon(icon if warnings or path in self._errors else QIcon())
        self._filter_images()

    def _filter_images(self):
        if self.external:
            return
        for row in range(self.images.count()):
            self.images.item(row).setHidden(
                self.only_candidates.isChecked()
                and not self._records.get(str(self._pairs[row][0]))
            )
