"""Current application settings, independent of QObject and Qt palette code."""

from copy import deepcopy

from utils_new.json_store import read_json_object, write_json_atomic

SETTINGS_VERSION = 1


def normalize_settings(data, defaults):
    if not isinstance(data, dict) or data.get("schema_version") != SETTINGS_VERSION:
        raise ValueError("Нужен файл настроек schema_version=1 с размерами в пикселях.")

    def merge(values, template):
        if not isinstance(values, dict):
            raise ValueError("Секция настроек должна быть словарём.")
        return {
            key: merge(values.get(key, {}), value)
            if isinstance(value, dict)
            else deepcopy(values.get(key, value))
            for key, value in template.items()
        }

    result = merge(data, defaults)
    for field, minimum, maximum in [
        ("font_size_px", 8, 48),
        ("line_thickness_px", 1, 12),
    ]:
        value = result[field]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{field}: требуется целое число {minimum}..{maximum} px.")
    return result


def load_settings(path, defaults):
    return normalize_settings(read_json_object(path), defaults)


def save_settings(path, data, defaults):
    normalized = normalize_settings(data, defaults)
    write_json_atomic(path, normalized)
    return normalized
