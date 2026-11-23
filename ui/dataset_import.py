"""One reviewed import dialog for YAML, COCO and indexed-mask datasets."""

from copy import deepcopy
from threading import Event
import tempfile

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QPlainTextEdit,
    QProgressBar,
    QAbstractItemView,
)

from utils_new.import_export import dataset_sources as sources


class _Signals(QObject):
    finished = pyqtSignal(int, object, object)
    progress = pyqtSignal(int, int, int, str)


class _PathEdit(QPlainTextEdit):
    """Wrapped, selectable paths remain readable in fixed-width table columns."""

    def text(self):
        return self.toPlainText()

    def setText(self, text):
        if text != self.toPlainText():
            self.setPlainText(text)
        self.setToolTip(text)


class _Preview(QRunnable):
    def __init__(self, serial, spec, mode):
        super().__init__()
        self.serial, self.spec, self.mode = serial, spec, mode
        self.cancel = Event()
        self.signals = _Signals()

    def run(self):
        try:
            self.signals.finished.emit(
                self.serial, sources.preview(self.spec, self.mode, self.cancel,
                    lambda done, total, stage: self.signals.progress.emit(self.serial, done, total, stage)), None
            )
        except Exception as exc:
            self.signals.finished.emit(self.serial, None, str(exc))


class DatasetImportDialog(QDialog):
    def __init__(self, paths, kind, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Источник датасета — проверьте пути")
        self.resize(1280, 560)
        self.base = sources.defaults(paths, kind)
        for index, row in enumerate(self.base["rows"]):
            row["row_index"] = index
        self.plan = None
        self._serial = 0
        self._jobs = {}
        self._closed = False
        layout = QVBoxLayout(self)
        source = QLabel("Источник: " + ", ".join(self.base["sources"]))
        source.setWordWrap(True)
        layout.addWidget(source)
        row = QHBoxLayout()
        row.addWidget(QLabel("Основной path:"))
        self.root = QLineEdit(self.base["root"])
        self.root.setObjectName("dataset_root")
        row.addWidget(self.root, 1)
        browse = QPushButton("…")
        browse.clicked.connect(self._browse_root)
        row.addWidget(browse)
        layout.addLayout(row)
        self.coco_choice = None
        if self.base["kind"] == "coco" and len(paths) > 1:
            self.coco_choice = QComboBox()
            self.coco_choice.addItem("Использовать все выбранные COCO JSON", -1)
            for index, path in enumerate(paths):
                self.coco_choice.addItem(str(path), index)
            self.coco_choice.currentIndexChanged.connect(self._schedule)
            layout.addWidget(self.coco_choice)
        row = QHBoxLayout()
        row.addWidget(QLabel("Источник разметки:"))
        self.mode = QComboBox()
        modes = (
            ("json",)
            if self.base["kind"] == "coco"
            else (
                ("declared", "beside", "labels", "other", "none")
                if self.base["kind"] == "unet"
                else ("labels", "beside", "other", "none")
            )
        )
        for key in modes:
            self.mode.addItem(sources.MODES[key], key)
        initial = self.mode.findData(self.base.get("initial_mode"))
        if initial >= 0:
            self.mode.setCurrentIndex(initial)
        self.mode.setObjectName("annotation_source")
        row.addWidget(self.mode, 1)
        layout.addLayout(row)
        self.table = QTableWidget(len(self.base["rows"]), 7)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(78)
        self.table.setHorizontalHeaderLabels(
            [
                "Split / JSON",
                "Снимки: папка / TXT-список",
                "Метки / маски / JSON",
                "Снимки",
                "С разметкой",
                "Без разметки",
                "Конфликты",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for column in range(3, 7):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents
            )
        self.table.setMinimumHeight(150)
        layout.addWidget(self.table, 1)
        self.image_fields, self.annotation_fields = [], []
        self._render_rows(self.base)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        help_text = (
            "COCO: разметка читается только из указанных JSON. Снимки без объектов сохраняются."
            if self.base["kind"] == "coco"
            else "Пути можно редактировать; несколько путей разделяйте точкой с запятой. "
            "Снимки без меток/масок загружаются. Пустой TXT — размеченный фон; отсутствующий файл — без разметки. "
            "При обучении U-Net/DeepLab снимки без масок не используются."
        )
        hint = QLabel(help_text)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(180)
        self.timer.timeout.connect(self._scan)
        self.root.textChanged.connect(self._root_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _path_cell(self, value, key):
        from ui import dataset_source as dialogs

        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        edit = _PathEdit()
        edit.setText("; ".join(value))
        button = QPushButton("…")
        button.setFixedWidth(26)

        def browse():
            if key == "coco":
                path, _ = dialogs.get_open_file(self, "coco", confirm=False)
            else:
                path = dialogs.get_directory(self, key)
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        row.addWidget(edit)
        row.addWidget(button)
        edit.textChanged.connect(self._schedule)
        return widget, edit

    def _render_rows(self, spec):
        for i, row in enumerate(spec["rows"]):
            self.table.setItem(i, 0, QTableWidgetItem(row["split"]))
            widget, edit = self._path_cell(row["images"], "folder")
            self.table.setCellWidget(i, 1, widget)
            self.image_fields.append(edit)
            widget, edit = self._path_cell(
                row["annotations"],
                "coco"
                if spec["kind"] == "coco"
                else ("masks" if spec["kind"] == "unet" else "labels"),
            )
            self.table.setCellWidget(i, 2, widget)
            self.annotation_fields.append(edit)

    def _browse_root(self):
        from ui import dataset_source as dialogs

        path = dialogs.get_directory(self, "dataset_root", directory=self.root.text())
        if path:
            self.root.setText(path)

    def _root_changed(self):
        spec = sources.relocate(self.base, self.root.text())
        for i, row in enumerate(spec["rows"]):
            self.image_fields[i].setText("; ".join(row["images"]))
            self.annotation_fields[i].setText("; ".join(row["annotations"]))
        self._mode_changed()

    def _mode_changed(self):
        mode = self.mode.currentData()
        previous = self.plan
        spec = sources.relocate(self.base, self.root.text())
        for i, row in enumerate(spec["rows"]):
            row["images"] = self._split(self.image_fields[i].text())
            paths = sources.proposed_annotations(spec, row, mode)
            if mode == "other" and previous:
                from pathlib import Path

                detected = sorted(
                    {
                        str(Path(path).parent)
                        for entry in previous["entries"]
                        if entry["split"] == row["split"]
                        for path in entry["alternatives"].get("other", [])
                    }
                )
                paths = detected or paths
            self.annotation_fields[i].setText("; ".join(paths))
            self.annotation_fields[i].setReadOnly(
                mode not in ("other", "declared", "json")
            )
            for button in self.table.cellWidget(i, 2).findChildren(QPushButton):
                button.setEnabled(mode not in ("none", "beside", "labels"))
        self._schedule()

    @staticmethod
    def _split(text):
        return [
            value.strip()
            for value in text.replace("\n", ";").split(";")
            if value.strip()
        ]

    def _schedule(self, *args):
        if not hasattr(self, "timer"):
            return
        self._serial += 1
        self.plan = None
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.notice.setText("Подсчёт файлов…")
        self.progress.setRange(0, 0)
        self.progress.show()
        for job in self._jobs.values():
            job.cancel.set()
        self.timer.start()

    def _scan(self):
        spec = sources.relocate(self.base, self.root.text())
        for i, row in enumerate(spec["rows"]):
            row["images"] = self._split(self.image_fields[i].text())
            row["annotations"] = self._split(self.annotation_fields[i].text())
        if self.coco_choice is not None and self.coco_choice.currentData() != -1:
            spec["rows"] = [spec["rows"][self.coco_choice.currentData()]]
        job = _Preview(self._serial, deepcopy(spec), self.mode.currentData())
        self._jobs[job.serial] = job
        job.signals.finished.connect(self._finished)
        job.signals.progress.connect(self._progress)
        QThreadPool.globalInstance().start(job)

    def _progress(self, serial, done, total, stage):
        if serial == self._serial and not self._closed:
            self.progress.setRange(0, total)
            if total:
                self.progress.setValue(done)
            self.notice.setText(f'{stage}: {done} файлов')

    def _finished(self, serial, plan, error):
        self._jobs.pop(serial, None)
        if serial != self._serial or self._closed:
            return
        self.progress.hide()
        if error:
            self.notice.setText(error)
            return
        self.plan = plan
        for i in range(self.table.rowCount()):
            for column in range(3, 7):
                self.table.setItem(i, column, QTableWidgetItem("0"))
        for row in plan["summaries"]:
            i = row["row_index"]
            for column, key in enumerate(
                ("images", "matched", "missing", "conflicts"), 3
            ):
                self.table.setItem(i, column, QTableWidgetItem(str(row[key])))
        if plan["mode"] in ("beside", "labels"):
            for row in plan["spec"]["rows"]:
                field = self.annotation_fields[row["row_index"]]
                field.blockSignals(True)
                field.setText("; ".join(row.get("actual_annotations", [])))
                field.blockSignals(False)
        counts = {
            key: sum(row[key] for row in plan["summaries"])
            for key in ("images", "matched", "missing", "files", "conflicts")
        }
        self.notice.setText(
            f"Снимков: {counts['images']}; с разметкой: {counts['matched']}; файлов меток / записей COCO: {counts['files']}; без разметки: {counts['missing']}. "
            + (
                f"Разметка найдена в нескольких местах для {counts['conflicts']} снимков. Будет использован только выбранный источник."
                if counts["conflicts"]
                else "Проверьте пути и нажмите OK."
            )
        )
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(counts["images"] > 0)

    def done(self, code):
        self._closed = True
        self.timer.stop()
        for job in self._jobs.values():
            job.cancel.set()
        super().done(code)

    def accept(self):
        if (
            self.plan is not None
            and self.buttons.button(QDialogButtonBox.Ok).isEnabled()
        ):
            super().accept()


def confirm_dataset_import(owner, paths, kind):
    dialog = DatasetImportDialog(paths, kind, owner)
    panel = getattr(owner, "dataset_source", None) or getattr(
        owner, "reference_source", None
    )
    if (
        dialog.base["kind"] == "unet"
        and panel is not None
        and "unet_yaml" not in getattr(panel, "actions", ())
    ):
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.warning(
            owner,
            "Источник датасета",
            "Эта операция принимает YOLO TXT, а не индексные маски.",
        )
        dialog.setParent(None)
        dialog.deleteLater()
        return []
    if dialog.exec_() != QDialog.Accepted:
        dialog.setParent(None)
        dialog.deleteLater()
        return []
    temporary = tempfile.TemporaryDirectory(prefix="neuro-dataset-")
    if owner is not None:
        resources = getattr(owner, "_dataset_import_resources", [])
        resources.append(temporary)
        owner._dataset_import_resources = resources
    else:
        # Non-widget entry points retain the view for the application lifetime.
        _resources.append(temporary)
    result = sources.write_manifest(dialog.plan, temporary.name)
    target = owner
    while target is not None and not callable(getattr(target, "log", None)):
        target = target.parent() if hasattr(target, "parent") else None
    log = target.log if target is not None else print
    sources.log_plan(dialog.plan, log)
    dialog.setParent(None)
    dialog.deleteLater()
    return result


_resources = []
