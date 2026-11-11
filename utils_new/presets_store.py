"""JSON preset persistence independent of widgets and algorithms."""
import json
import os
import tempfile
from pathlib import Path


def read_presets(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError('Preset file must contain a JSON object')
    return value


def write_presets(path, presets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(presets, stream, indent=4)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
