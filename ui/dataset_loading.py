"""Cancellable dataset tasks with GUI-thread delivery and one progress row per task."""

from time import monotonic

from PyQt5.QtCore import QEvent, QObject, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from ui.dataset_audit.workers import AuditWorker


class DatasetTasks(QObject):
    settled = pyqtSignal(str)

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.jobs = {}
        self.closed = False
        self.resources = []
        self.requests = {}
        self.rows = {}
        owner.destroyed.connect(self.close)
        owner.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "owner", None) and event.type() == QEvent.Close:
            self.close()
        return False

    def busy(self, key=None):
        return bool(self.jobs) if key is None else key in self.jobs

    def start(self, key, job_type, *args, label, finished):
        if self.closed:
            return
        self.invalidate(key)
        self.requests[key] = (job_type, args, label, finished)
        worker = AuditWorker(job=job_type(*args))
        last_progress = [0.0]

        def progress(done, total):
            now = monotonic()
            if (total > 0 and done >= total) or now - last_progress[0] >= 0.1:
                last_progress[0] = now
                worker._emit(worker._progress_ready, done, total)

        worker.job.progress = progress
        self.jobs[key] = (worker, finished)
        if key not in self.rows:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            title, bar, cancel = QLabel(), QProgressBar(), QPushButton("Отменить")
            bar.setMaximumWidth(180)
            layout.addWidget(title, 1)
            layout.addWidget(bar)
            layout.addWidget(cancel)
            cancel.clicked.connect(lambda _=False, key=key: self.cancel(key))
            self.owner.statusBar().addPermanentWidget(row, 1)
            self.rows[key] = (row, title, bar, cancel)
        row, title, bar, cancel = self.rows[key]
        title.setText(label)
        count = len(getattr(worker.job, "pairs", getattr(worker.job, "images", ())))
        bar.setRange(0, count)
        bar.setValue(0)
        bar.setFormat(f"0/{count} (%p%)" if count else "%p%")
        cancel.setEnabled(True)
        cancel.setText("Отменить")
        row.show()
        self.owner.statusBar().show()
        worker.progress.connect(self._progress)
        worker.log_signal.connect(self._log)
        worker.finished_signal.connect(self._finished)
        worker.start()
        return worker

    def _key(self):
        return next(
            (key for key, (worker, _) in self.jobs.items() if self.sender() is worker),
            None,
        )

    def _progress(self, done, total):
        key = self._key()
        if key is None or self.closed:
            return
        _, title, bar, _ = self.rows[key]
        bar.setRange(0, max(0, total))
        if total:
            bar.setValue(done)
            bar.setFormat(f"{done}/{total} (%p%)")

    def _log(self, message):
        if self._key() is not None and not self.closed:
            self.owner.log(message)

    def _finished(self, result):
        key = self._key()
        if key is None or self.closed:
            return
        worker, callback = self.jobs.pop(key)
        row, title, bar, cancel = self.rows[key]
        cancel.setEnabled(False)
        bar.setRange(0, 100)
        bar.setValue(100)
        if result.get("resource") is not None:
            self.resources.append(result["resource"])
        try:
            if result.get("error"):
                title.setText("Ошибка: " + result["error"])
                self.owner.log(title.text())
            elif result.get("cancelled"):
                title.setText("Операция отменена")
            else:
                callback(result)
                # Callback may start the next task under the same key.
                if key not in self.jobs:
                    row.hide()
        except Exception as exc:
            title.setText(f"Ошибка обработки результата: {exc}")
            self.owner.log(title.text())
        finally:
            if key not in self.jobs and (
                result.get("error") or result.get("cancelled")
            ):
                cancel.setText("Повторить")
                cancel.setEnabled(True)
            self.settled.emit(key)

    def cancel(self, key):
        if key not in self.jobs and key in self.requests and not self.closed:
            job_type, args, label, finished = self.requests[key]
            self.start(key, job_type, *args, label=label, finished=finished)
        elif key in self.jobs:
            self.jobs[key][0].cancel()
            _, title, _, cancel = self.rows[key]
            title.setText("Остановка…")
            cancel.setEnabled(False)

    def invalidate(self, key=None):
        names = set(self.jobs) | set(self.rows)
        for name in names:
            if key is None or name == key:
                if name in self.jobs:
                    self.jobs.pop(name)[0].close()
                self.requests.pop(name, None)
                try:
                    self.rows[name][0].hide()
                except RuntimeError:
                    pass

    def close(self, *_):
        if not getattr(self, "closed", True):
            self.closed = True
            self.invalidate()


def dataset_tasks(owner):
    if not hasattr(owner, "_dataset_tasks"):
        owner._dataset_tasks = DatasetTasks(owner)
    return owner._dataset_tasks


def load_editor_dataset(owner, kind, paths, *, overlay=False):
    """Shared background import; editor caches still decode only current/nearby frames."""
    from utils_new.import_export.catalog import DatasetImportJob

    tasks = dataset_tasks(owner)

    def loaded(result):
        catalog = result["catalog"]
        if not catalog.pairs:
            owner.log(
                "В выбранном источнике не найдено снимков; текущий набор сохранён."
            )
            return
        kwargs = dict(label_map=dict(catalog.pairs))
        if overlay:
            kwargs.update(
                mask_map=catalog.metadata.get("mask_map", {}),
                mask_pixel_offset=catalog.metadata.get("mask_class_offset", 1),
            )
        owner._load_images([pair[0] for pair in catalog.pairs], **kwargs)
        owner._save_label_map = result["save_targets"]
        owner._save_source_kind = (
            "unet" if result["kind"] == "unet_editor" else result["kind"]
        )
        owner.log(
            f"Источник: {', '.join(paths)}; path={catalog.root}; снимков={len(catalog.pairs)}, "
            f"с метками={sum(label is not None for _, label in catalog.pairs)}. Снимки загружаются лениво."
        )

    tasks.start(
        "load",
        DatasetImportJob,
        kind,
        paths,
        dict(editor=True, overlay=overlay),
        label="Загрузка списка / преобразование разметки…",
        finished=loaded,
    )


def load_editor_folder(owner, source, kwargs):
    """Enumerate folder names away from the event loop; preserve the bounded cache."""
    from utils_new.import_export.catalog import DatasetImportJob

    tasks = dataset_tasks(owner)
    tasks.start(
        "load",
        DatasetImportJob,
        "images",
        [str(source)],
        {},
        label="Поиск снимков…",
        finished=lambda result: owner._load_images(
            [p[0] for p in result["catalog"].pairs], **kwargs
        ),
    )


def request_cached_frame(owner, cache, index, display):
    """True means a frame is ready; otherwise deliver only the latest request."""
    from utils_new.images.cache import CachedImageJob

    if cache.is_loaded(index) or cache.get_error(index) is not None:
        return True
    tasks = dataset_tasks(owner)
    pending = tasks.requests.get("frame") if tasks.busy("frame") else None
    if pending and pending[1] == (cache, index):
        return False
    cache.trim_around(index)

    def loaded(result):
        if owner._lazy_cache is cache and result["item"] is not None:
            display()
        elif result["item"] is None:
            owner.log(f"Не удалось загрузить снимок: {cache.paths[index]}")

    tasks.start(
        "frame",
        CachedImageJob,
        cache,
        index,
        label=f"Чтение снимка {index + 1}/{len(cache)}…",
        finished=loaded,
    )
    return False
