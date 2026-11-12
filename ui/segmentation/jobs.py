"""One serial numerical worker per window, with queued GUI delivery and stale guards."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

from PyQt5.QtCore import QObject, Qt, pyqtSignal

from ui.segmentation import errors, presentation
from utils_new.segmentation.result import ResultStatus


class TaskController(QObject):
    completed = pyqtSignal(int, object, object)
    settled = pyqtSignal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.busy = False
        self.closed = False
        self._serial = 0
        self._epoch = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="segmentation"
        )
        self._cancel = Event()
        self._request = None
        self.completed.connect(self._deliver, Qt.QueuedConnection)

    def key(self, frame_sensitive=True, dataset_sensitive=True):
        owner = self.owner
        key = (
            (owner._data_revision,)
            if dataset_sensitive
            else (owner.model_combo.currentText(), getattr(owner, "device", None))
        )
        if frame_sensitive:
            key += (
                owner.current_index,
                tuple(owner.image_paths),
                id(getattr(owner, "segmentor", None)),
                id(getattr(owner, "model", None)),
            )
        return key

    def start(
        self, context, compute, accept, *, frame_sensitive=True, dataset_sensitive=True
    ):
        if self.closed or self.busy:
            return False
        self.busy = True
        self.owner._reported_error = None
        self._serial += 1
        serial = self._serial
        self._cancel = Event()
        cancel = self._cancel
        self._request = (
            self._epoch,
            self.key(frame_sensitive, dataset_sensitive),
            frame_sensitive,
            dataset_sensitive,
            accept,
            context,
        )
        self._controls_enabled = self.owner.controls_panel.isEnabled()
        self.owner.controls_panel.setEnabled(False)
        presentation.show_unavailable(self.owner, ResultStatus.RUNNING)
        progress = getattr(self.owner, "progress_bar", None)
        if progress is not None:
            progress.setRange(0, 0)

        def run():
            if cancel.is_set():
                raise CancelledError()
            return compute(cancel)

        future = self._executor.submit(run)

        def done(future):
            try:
                value, error = future.result(), None
            except BaseException as exc:
                value, error = None, exc
            if not self.closed:
                try:
                    self.completed.emit(serial, value, error)
                except RuntimeError:
                    pass  # QObject was destroyed after the window closed.

        future.add_done_callback(done)
        return True

    def _deliver(self, serial, value, error):
        if self.closed or serial != self._serial:
            return
        epoch, key, frame_sensitive, dataset_sensitive, accept, context = self._request
        self.busy = False
        self._request = None
        self.owner.controls_panel.setEnabled(self._controls_enabled)
        progress = getattr(self.owner, "progress_bar", None)
        if progress is not None:
            progress.setRange(0, 100)
            progress.setValue(0 if error else 100)
        try:
            if (
                epoch != self._epoch
                or key != self.key(frame_sensitive, dataset_sensitive)
                or isinstance(error, CancelledError)
            ):
                if self.owner.result_state.status == ResultStatus.RUNNING:
                    presentation.show_unavailable(
                        self.owner,
                        ResultStatus.NOT_RUN
                        if self.owner.image_paths
                        else ResultStatus.NO_IMAGE,
                    )
                self.owner.schedule_update()
            elif error is not None:
                errors.report(self.owner, context, error)
            else:
                accept(value)
        except Exception as exc:
            errors.report(self.owner, context, exc)
        finally:
            self.settled.emit()

    def invalidate(self, *, data_only=False):
        if data_only and self._request is not None and not self._request[3]:
            return
        self._epoch += 1
        self._cancel.set()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.invalidate()
        self._executor.shutdown(wait=False, cancel_futures=True)
