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


class BackgroundTask(QObject):
    """Cancellable callable/worker without a destructible running QThread.

    Subclasses may keep domain signals and override run(). Owner callbacks must
    also reject detached senders: a signal can already be queued when closed.
    An optional serial executor lets a window avoid concurrent model access.
    """

    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    settled = pyqtSignal()

    def __init__(self, compute=None):
        super().__init__()
        self.compute = compute
        self.cancel_event = Event()
        self.closed = False
        self._running = False
        self._executor = None
        self._owns_executor = False
        self._future = None

    @property
    def cancelled(self):
        return self.cancel_event.is_set()

    def publish(self, signal, *args):
        if not self.closed:
            try:
                signal.emit(*args)
            except RuntimeError:
                pass

    def run(self):
        try:
            if self.cancelled:
                return
            value = self.compute(self.cancel_event)
            if not self.cancelled:
                self.publish(self.result_ready, value)
        except CancelledError:
            pass
        except Exception as exc:
            self.publish(self.failed, str(exc))

    def start(self, executor=None):
        if self.closed or self._running:
            return
        self._running = True
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="image-model"
        )

        def execute():
            try:
                self.run()
            except Exception as exc:
                self.publish(self.failed, str(exc))
            finally:
                self._running = False
                self.publish(self.settled)
                if self._owns_executor:
                    self._executor.shutdown(wait=False)

        self._future = self._executor.submit(execute)

    def isRunning(self):
        return self._running

    def cancel(self):
        self.cancel_event.set()

    def wait(self, milliseconds=None):
        from concurrent.futures import TimeoutError

        if self._future is None:
            return True
        try:
            self._future.result(
                timeout=None if milliseconds is None else milliseconds / 1000
            )
            return True
        except TimeoutError:
            return False

    def close(self):
        self.closed = True
        self.cancel()
        if self._future is not None:
            self._future.cancel()
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
