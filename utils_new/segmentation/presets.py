"""Current segmentation preset schema. Relative morphology is not supported."""

from copy import deepcopy
from utils_new.annotations.modes import validate_mode

from utils_new.morphology import SUPPORTED_OPERATIONS, SUPPORTED_SHAPES

SCHEMA_VERSION = 3
KINDS = ("threshold", "gradient", "interactive", "traditional_ml", "deep_learning")


def default_morphology():
    return {
        "version": 1,
        "units": "px",
        "shape": "Rectangle",
        "operations": [
            {"operation": op, "kernel_px": 0} for op in SUPPORTED_OPERATIONS
        ],
        "contour_only": False,
        "contour_width_px": None,
    }


def validate_morphology(state):
    if (
        not isinstance(state, dict)
        or state.get("version") != 1
        or state.get("units") != "px"
    ):
        raise ValueError("Морфология должна использовать формат version=1, units=px.")
    if state.get("shape") not in SUPPORTED_SHAPES:
        raise ValueError("Неизвестная форма ядра морфологии.")
    steps = state.get("operations")
    if (
        not isinstance(steps, list)
        or len(steps) != 4
        or not all(isinstance(s, dict) for s in steps)
    ):
        raise ValueError("Нужны четыре операции морфологии.")
    names = [s.get("operation") for s in steps]
    if len(set(names)) != 4 or set(names) != set(SUPPORTED_OPERATIONS):
        raise ValueError(
            "close/open/erode/dilate должны присутствовать по одному разу."
        )
    for step in steps:
        size = step.get("kernel_px")
        if type(size) is not int or not 0 <= size <= 4096:
            raise ValueError("Размер ядра должен быть целым числом 0..4096 px.")
        if set(step) != {"operation", "kernel_px"}:
            raise ValueError("Операция панели содержит только operation и kernel_px.")
    if (
        state.get("contour_only") is not False
        or state.get("contour_width_px") is not None
    ):
        raise ValueError("Морфология методов применяется ко всей бинарной маске.")
    return state


def validate_preset(state, kind=None):
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Нужен пресет schema_version=3. Старые относительные пресеты не поддерживаются."
        )
    if state.get("kind") not in KINDS or (kind is not None and state["kind"] != kind):
        raise ValueError("Пресет принадлежит другой вкладке методов.")
    if {"close_factor", "open_factor", "kernel_shape"} & state.keys():
        raise ValueError("Относительные параметры морфологии больше не поддерживаются.")
    validate_morphology(state.get("morphology"))
    if not isinstance(state.get("invert"), bool) or not isinstance(
        state.get("use_hull"), bool
    ):
        raise ValueError("invert и use_hull должны быть логическими значениями.")
    validate_mode(state.get('annotation_mode'))
    if 'draw_mode' in state:
        raise ValueError('draw_mode больше не поддерживается; используйте annotation_mode.')
    key = "model" if state["kind"] in ("traditional_ml", "deep_learning") else "method"
    if not isinstance(state.get(key), str) or not state[key]:
        raise ValueError(f"Не задан {key}.")
    if state["kind"] == "threshold" and not isinstance(state.get("params"), dict):
        raise ValueError("params должен содержать параметры порогового метода.")
    return state


def make_preset(kind, **parameters):
    state = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "morphology": default_morphology(),
        "invert": False,
        "annotation_mode": "none",
        "use_hull": False,
        **deepcopy(parameters),
    }
    if kind == "threshold":
        state.setdefault("params", {})
    validate_preset(state, kind)
    return state
