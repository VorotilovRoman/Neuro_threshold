"""Bounded image cache with one load per index and deterministic shutdown."""

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError


class LazyImageCache:
    def __init__(
        self, paths, loader, window=8, prefetch_ahead=4, max_resident=16, max_workers=2
    ):
        for name, value, minimum in [
            ("window", window, 1),
            ("prefetch_ahead", prefetch_ahead, 0),
            ("max_resident", max_resident, 1),
            ("max_workers", max_workers, 1),
        ]:
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.paths = list(paths)
        self.loader = loader
        self.window = window
        self.prefetch_ahead = prefetch_ahead
        self.max_resident = max_resident
        self._cache = OrderedDict()
        self._errors = {}
        self._pending = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="img-load"
        )
        self._shutdown = False

    def __len__(self):
        return len(self.paths)

    def _load(self, idx, future):
        item, error = None, None
        try:
            with self._lock:
                if self._shutdown or self._pending.get(idx) is not future:
                    return
            item = self.loader(self.paths[idx])
            if item is None:
                error = OSError(f"Loader returned None for {self.paths[idx]}")
        except Exception as exc:
            error = exc
        finally:
            with self._lock:
                active = self._pending.get(idx) is future
                if active:
                    self._pending.pop(idx, None)
                if not self._shutdown and active:
                    if error is not None:
                        self._errors[idx] = error
                    elif item is not None:
                        self._cache[idx] = item
                        self._cache.move_to_end(idx)
                        while len(self._cache) > self.max_resident:
                            self._cache.popitem(last=False)
                        self._errors.pop(idx, None)
                if not future.done():
                    future.set_result(
                        item if not self._shutdown and error is None else None
                    )

    def is_loaded(self, idx):
        with self._lock:
            return idx in self._cache

    def get_error(self, idx):
        with self._lock:
            return self._errors.get(idx)

    def retry_error(self, idx):
        with self._lock:
            self._errors.pop(idx, None)

    def get(self, idx, *, cancel_check=None):
        with self._lock:
            if self._shutdown or not 0 <= idx < len(self.paths):
                return None
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return self._cache[idx]
            if idx in self._errors:
                return None
            future = self._pending.get(idx)
            owner = future is None
            if owner:
                future = self._pending[idx] = Future()
        if owner:
            self._load(idx, future)
        if cancel_check is None:
            return future.result()
        while not cancel_check():
            try:
                return future.result(timeout=0.05)
            except TimeoutError:
                pass
        return None

    def prefetch(self, indices):
        with self._lock:
            if self._shutdown:
                return
            for idx in indices:
                if (
                    not 0 <= idx < len(self.paths)
                    or idx in self._cache
                    or idx in self._pending
                    or idx in self._errors
                ):
                    continue
                future = self._pending[idx] = Future()
                # Submission and shutdown are serialized by this lock.
                self._pool.submit(self._load, idx, future)

    def trim_around(self, center, keep_behind=3, keep_ahead=None):
        keep_ahead = self.prefetch_ahead + 4 if keep_ahead is None else keep_ahead
        with self._lock:
            low, high = max(0, center - keep_behind), center + keep_ahead
            for values in (self._cache, self._errors):
                for idx in list(values):
                    if not low <= idx <= high:
                        del values[idx]
            for idx in list(self._pending):
                if not low <= idx <= high:
                    future = self._pending.pop(idx)
                    if not future.done():
                        future.set_result(None)

    def shutdown(self):
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._cache.clear()
            self._errors.clear()
            for future in self._pending.values():
                if not future.done():
                    future.set_result(None)
            self._pending.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


class CachedImageJob:
    """Wait for a requested cache entry away from the Qt event loop."""

    def __init__(self, cache, index):
        self.cache, self.index = cache, index
        self.cancelled = False
        self.progress = lambda *_: None
        self.log = lambda *_: None

    def cancel(self):
        self.cancelled = True

    def run(self):
        self.cache.retry_error(self.index)
        self.cache.prefetch([self.index])
        item = (
            None
            if self.cancelled
            else self.cache.get(self.index, cancel_check=lambda: self.cancelled)
        )
        if self.cancelled:
            return {"cancelled": True}
        if item is None:
            raise OSError(
                f"Не удалось прочитать снимок: {self.cache.paths[self.index]}"
            )
        return {"item": item}
