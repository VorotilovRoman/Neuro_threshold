"""Atomic file replacement shared by settings, labels, reports and images."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(destination, *, suffix=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=destination.parent, suffix=destination.suffix if suffix is None else suffix
        )
        os.close(descriptor)
        temporary = Path(name)
        yield temporary
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_text_atomic(path, text):
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding='utf-8')


def write_bytes_atomic(path, data):
    with atomic_path(path) as temporary:
        temporary.write_bytes(data)
