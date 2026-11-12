"""Search buttons and GUI-owned asynchronous controller. Numerical code never sees Qt."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from utils_new.autoreseach_utils.search_space import combination_count
from utils_new.autoreseach_utils.threshold_search import run_search


class AutoResearchPanel(QWidget):
    def __init__(self, owner):
        super().__init__()
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        owner.auto_button = QPushButton("Запустить автоподбор")
        owner.configure_auto_button = QPushButton("Диапазоны автоподбора")
        row.addWidget(owner.auto_button)
        row.addWidget(owner.configure_auto_button)
        layout.addLayout(row)
        owner.search_all_checkbox = QCheckBox("Все загруженные изображения")
        layout.addWidget(owner.search_all_checkbox)


class SearchController(QObject):
    progress = pyqtSignal(object, object, str)
    finished = pyqtSignal(object, object)
    settled = pyqtSignal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.busy = False
        self.closed = False
        self.cancel = Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="threshold-search"
        )
        self.dialog = None
        self.progress.connect(self._progress, Qt.QueuedConnection)
        self.finished.connect(self._finished, Qt.QueuedConnection)

    def start(self):
        if self.busy or self.closed:
            return
        owner = self.owner
        try:
            paths = (
                list(owner.image_paths)
                if owner.search_all_checkbox.isChecked()
                else owner.image_paths[owner.current_index : owner.current_index + 1]
            )
            if not paths:
                raise ValueError("Загрузите изображения.")
            settings = deepcopy(owner.auto_settings)
            count = combination_count(settings) * len(paths)
            if not count:
                raise ValueError("Нет включённых методов.")
            if (
                QMessageBox.question(
                    owner,
                    "Автоподбор",
                    f"Вычислений: {count}. Отчёты будут записаны рядом с изображениями; готовые отчёты с тем же именем будут заменены. Продолжить?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                return
        except Exception as exc:
            QMessageBox.warning(owner, "Автоподбор", str(exc))
            return
        self.busy = True
        self.cancel = Event()
        self._pending = owner.update_timer.isActive()
        owner.update_timer.stop()
        self.dialog = QProgressDialog("Автоподбор…", "Отмена", 0, 1000, owner)
        self.dialog.setWindowModality(Qt.WindowModal)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.canceled.connect(self.cancel.set)
        self.dialog.show()
        resize = owner.nav_widget.is_resize_enabled()
        cancel = self.cancel

        def report(done, total, name):
            if not self.closed:
                try:
                    self.progress.emit(done, total, name)
                except RuntimeError:
                    pass

        future = self.executor.submit(
            run_search, paths, settings, cancel, report, resize_enabled=resize
        )

        def finished(future):
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            if not self.closed:
                try:
                    self.finished.emit(result, error)
                except RuntimeError:
                    pass

        future.add_done_callback(finished)

    def _progress(self, done, total, name):
        if self.closed or not self.busy:
            return
        self.dialog.setValue(round(done * 1000 / total))
        self.dialog.setLabelText(f"{name}: {done}/{total}")

    def _finished(self, result, error):
        if self.closed:
            return
        self.busy = False
        self.dialog.close()
        self.dialog.deleteLater()
        self.dialog = None
        if error:
            QMessageBox.critical(self.owner, "Автоподбор", str(error))
        else:
            self.owner.log(
                f"Автоподбор: {'отменён' if result['canceled'] else 'завершён'}, вычислений {result['processed']}, ошибок вычисления {result['failed']}. Отчёты: {result['reports']}"
            )
            for message in result["errors"]:
                self.owner.log(message)
        if self._pending:
            self.owner.schedule_update()
        self.settled.emit()

    def close(self):
        self.closed = True
        self.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
