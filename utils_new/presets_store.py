"""Preset collection persistence; method schemas belong to segmentation.presets."""

from utils_new.json_store import read_json_object, write_json_atomic


def read_presets(path):
    return read_json_object(path)


def write_presets(path, presets):
    if not isinstance(presets, dict):
        raise TypeError("Preset collection must be a JSON object")
    write_json_atomic(path, presets)
