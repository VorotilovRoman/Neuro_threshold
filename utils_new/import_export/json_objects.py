"""Atomic UTF-8 JSON object persistence, without UI dependencies."""

import json
from pathlib import Path
from utils_new.import_export.atomic import write_text_atomic


def read_json_object(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON file must contain an object")
    return value


def write_json_atomic(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
    write_text_atomic(path, text)
