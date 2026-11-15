"""Cancelable, GUI-owned reference evaluation; discard replies after input changes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from utils_new.autoreseach_utils.evaluation import evaluate_dataset


class EvaluationController(QObject):
    finished = pyqtSignal(object, object, int)
    progress = pyqtSignal(int, int, str, int)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.busy = False
        self.closed = False
        self.cancel = Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="report-evaluation"
        )
        self.finished.connect(self._finished, Qt.QueuedConnection)
        self.progress.connect(self._progress, Qt.QueuedConnection)

    def start(self, frame, source, format_hint, threshold, class_aware):
        if self.busy or self.closed:
            return False
        self.busy = True
        self.cancel = Event()
        revision = self.owner._evaluation_revision
        cancel = self.cancel
        last = 0.0

        def progress(done, total, name):
            nonlocal last
            now = monotonic()
            if not self.closed and (done == total or now - last >= 0.1):
                last = now
                try:
                    self.progress.emit(done, total, name, revision)
                except RuntimeError:
                    pass

        future = self.executor.submit(
            evaluate_dataset,
            frame.copy(deep=True),
            source,
            format_hint=format_hint,
            threshold=threshold,
            class_aware=class_aware,
            cancel_check=cancel.is_set,
            progress=progress,
        )

        def finished(future):
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            if not self.closed:
                try:
                    self.finished.emit(result, error, revision)
                except RuntimeError:
                    pass

        future.add_done_callback(finished)
        return True

    def _progress(self, done, total, name, revision):
        if not self.closed and revision == self.owner._evaluation_revision:
            self.owner.lbl_evaluation_status.setText(f"Оценка: {done}/{total} — {name}")

    def _finished(self, result, error, revision):
        self.busy = False
        if self.closed:
            return
        self.owner.btn_stop_evaluation.setEnabled(False)
        self.owner.btn_analyze.setEnabled(True)
        if revision == self.owner._evaluation_revision:
            self.owner._evaluation_finished(result, error)

    def stop(self):
        self.cancel.set()

    def close(self):
        self.closed = True
        self.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
