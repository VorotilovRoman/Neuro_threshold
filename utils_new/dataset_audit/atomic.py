"""Atomic UTF-8 report replacement."""

import os
import tempfile
from pathlib import Path


def write_text_atomic(path, text):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, suffix=".tmp", encoding="utf-8", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
