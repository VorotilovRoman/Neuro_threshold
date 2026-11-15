"""Single cancelable executor for report loading and dataset comparison."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from utils_new.autoreseach_utils.evaluation import evaluate_dataset
from utils_new.autoreseach_utils.workflow import load_collection, load_folder_collection


class EvaluationController(QObject):
    finished = pyqtSignal(str, object, object, int)
    progress = pyqtSignal(str, int, int, str, int)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.busy = False
        self.closed = False
        self.task = None
        self.cancel = Event()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="report-comparison"
        )
        self.finished.connect(self._finished, Qt.QueuedConnection)
        self.progress.connect(self._progress, Qt.QueuedConnection)

    def _submit(self, task, function, *args, **kwargs):
        if self.busy or self.closed:
            return False
        self.busy = True
        self.task = task
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
                    self.progress.emit(task, done, total, name, revision)
                except RuntimeError:
                    pass

        future = self.executor.submit(
            function, *args, cancel_check=cancel.is_set, progress=progress, **kwargs
        )

        def finished(future):
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, exc
            if not self.closed:
                try:
                    self.finished.emit(task, result, error, revision)
                except RuntimeError:
                    pass

        future.add_done_callback(finished)
        return True

    def load(self, paths):
        return self._submit("load", load_collection, tuple(paths))

    def load_folder(self, folder, existing):
        return self._submit(
            "load", load_folder_collection, folder, existing=tuple(existing)
        )

    def start(self, frame, source, format_hint, threshold, class_aware):
        return self._submit(
            "compare",
            evaluate_dataset,
            frame.copy(deep=True),
            source,
            format_hint=format_hint,
            threshold=threshold,
            class_aware=class_aware,
        )

    def _progress(self, task, done, total, name, revision):
        if not self.closed and revision == self.owner._evaluation_revision:
            self.owner._on_job_progress(task, done, total, name)

    def _finished(self, task, result, error, revision):
        self.busy = False
        self.task = None
        if self.closed:
            return
        if revision == self.owner._evaluation_revision:
            self.owner._task_finished(task, result, error)
        self.owner._update_workflow()

    def stop(self):
        self.cancel.set()

    def close(self):
        self.closed = True
        self.cancel.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
