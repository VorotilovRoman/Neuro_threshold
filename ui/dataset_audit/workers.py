"""Qt delivery around pure numerical jobs, without destructible running QThreads."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from utils_new.dataset_audit.jobs import (
    LossInspectionJob,
    EmbeddingClusterJob,
    ReportJob,
    YoloValidationJob,
)


class AuditWorker(QObject):
    progress = pyqtSignal(int, int)
    _progress_ready = pyqtSignal(int, int)
    _log_ready = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    job_type = None

    def __init__(self, *args, parent=None, job=None, **kwargs):
        super().__init__(parent)
        self.job = job if job is not None else self.job_type(*deepcopy(args), **deepcopy(kwargs))
        self.closed = False
        self._busy = False
        self.revision = 0
        self._executor = None
        self._progress_ready.connect(self._deliver_progress, Qt.QueuedConnection)
        self._log_ready.connect(self._deliver_log, Qt.QueuedConnection)
        self.job.progress = lambda c, t: self._emit(self._progress_ready, c, t)
        self.job.log = lambda text: self._emit(self._log_ready, str(text))

    def _deliver_progress(self, c, t):
        self._emit(self.progress, c, t)

    def _deliver_log(self, text):
        self._emit(self.log_signal, text)

    def _emit(self, signal, *args):
        if not self.closed:
            try:
                signal.emit(*args)
            except RuntimeError:
                pass

    def run(self):
        try:
            value = self.job.run()
            if getattr(self.job, "cancelled", False) is True:
                value = {"cancelled": True}
            if value is None:
                value = {"error": "Задача завершилась без результата."}
        except Exception as exc:
            value = {"error": str(exc)}
        self._busy = False
        value["_dataset_revision"] = self.revision
        self._emit(self.finished_signal, value)
        return value

    def start(self):
        if self.closed or self._busy:
            return
        self._busy = True
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dataset-audit"
        )
        future = self._executor.submit(self.run)

        def settled(_):
            self._busy = False
            self._executor.shutdown(wait=False)

        future.add_done_callback(settled)

    def isRunning(self):
        return self._busy

    def cancel(self):
        if hasattr(self.job, "cancel"):
            self.job.cancel()
        elif hasattr(self.job, "request_stop"):
            self.job.request_stop()

    request_stop = cancel

    def close(self):
        self.closed = True
        self.cancel()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)


class LossInspectionWorker(AuditWorker):
    job_type = LossInspectionJob


class EmbeddingClusterWorker(AuditWorker):
    job_type = EmbeddingClusterJob


class ReportWorker(AuditWorker):
    job_type = ReportJob


class YoloValidationWorker(AuditWorker):
    job_type = YoloValidationJob
