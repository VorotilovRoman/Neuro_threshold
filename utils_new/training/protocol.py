"""Worker names, UTF-8 line framing and cooperative stop requests; no Qt/Torch."""

import codecs
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
from threading import Event

WORKERS = {
    kind: f"utils_new.training.{kind}"
    for kind in ("yolo", "unet", "deeplab", "autoencoder")
}
CANCELLED = 130


def configure_worker_streams():
    """Use the GUI's wire encoding, even with redirected Windows console output."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def worker_arguments(kind, *, params_file=None, stop_file=None):
    if kind not in WORKERS:
        raise ValueError(f"Неизвестный воркер: {kind}")
    result = ["-u", "-m", WORKERS[kind]]
    if params_file:
        result += [
            "--config" if kind == "autoencoder" else "--params_file",
            str(params_file),
        ]
    if stop_file:
        result += ["--stop_file", str(stop_file)]
    return result


class LineDecoder:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.buffer = ""

    def feed(self, data=b"", *, final=False):
        self.buffer += self.decoder.decode(data, final=final).replace("\r", "\n")
        values = self.buffer.split("\n")
        self.buffer = values.pop()
        if final and self.buffer:
            values.append(self.buffer)
            self.buffer = ""
        return [line.strip() for line in values if line.strip()]


class StopRequest:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.event = Event()

    def is_set(self):
        return self.event.is_set() or (self.path is not None and self.path.exists())

    def set(self, *args):
        self.event.set()

    @contextmanager
    def signals(self):
        previous = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, self.set)
            yield self
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
