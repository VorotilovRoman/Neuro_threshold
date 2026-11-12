"""Validated finite ranges and lazy canonical threshold presets; no Qt."""

from copy import deepcopy
from decimal import Decimal
from itertools import product
from math import prod, isfinite

from utils_new.segmentation.presets import (
    default_morphology,
    make_preset,
    validate_morphology,
)

# Parameter key: default start, end, step, lower bound, upper bound, integer, odd.
SPECS = {
    "Simple Threshold": {"threshold": (1, 254, 1, 0, 255, True, False)},
    "Otsu": {},
    "Triangle": {},
    "Adaptive Mean": {
        "window": (3, 51, 2, 3, 4095, True, True),
        "c": (-10, 10, 5, -255, 255, True, False),
    },
    "Adaptive Gauss": {
        "window": (3, 51, 2, 3, 4095, True, True),
        "c": (-10, 10, 5, -255, 255, True, False),
    },
    "Niblack": {
        "window": (3, 51, 2, 3, 4095, True, True),
        "k": (-0.2, 0.2, 0.1, -1, 1, False, False),
    },
    "Sauvola": {
        "window": (3, 51, 2, 3, 4095, True, True),
        "k": (0.1, 0.3, 0.1, -1, 1, False, False),
        "r": (128, 128, 1, 1, 255, True, False),
    },
    "ISODATA": {"init": (0, 255, 1, 0, 255, True, False)},
    "Background Symmetry": {"excess": (0.01, 1, 0.01, 0.01, 1, False, False)},
    "Row Adaptive": {
        "window": (10, 100, 10, 1, 4096, True, False),
        "k": (0, 1, 0.1, 0, 1, False, False),
    },
}


def default_settings():
    return {
        "schema_version": 1,
        "methods": {
            name: {
                "enabled": name == "Simple Threshold",
                "ranges": {key: list(spec[:3]) for key, spec in params.items()},
            }
            for name, params in SPECS.items()
        },
        "morphology": default_morphology(),
        "kernel_ranges": {op: [0, 0, 1] for op in ("close", "open", "erode", "dilate")},
        "inversions": [False],
        "annotation_mode": "detect",
        "use_hull": False,
    }


def range_values(values, spec):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Диапазон: [начало, конец, шаг].")
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("Диапазон должен содержать конечные числа.")
    start, end, step = map(lambda v: Decimal(str(v)), values)
    lo, hi, integer, odd = spec[3:]
    if step <= 0 or start > end or start < Decimal(str(lo)) or end > Decimal(str(hi)):
        raise ValueError(f"Неверный диапазон {values}; границы {lo}..{hi}, шаг > 0.")
    if integer and any(v != v.to_integral_value() for v in (start, end, step)):
        raise ValueError("Требуются целые значения диапазона.")
    if odd and (int(start) % 2 != 1 or (end > start and int(step) % 2 != 0)):
        raise ValueError(
            "Окно должно оставаться нечётным: нечётное начало и чётный шаг."
        )
    count = int((end - start) // step) + 1
    if count > 100000:
        raise ValueError("Слишком много значений в одном диапазоне.")
    return [
        int(start + i * step) if integer else float(start + i * step)
        for i in range(count)
    ]


def dimensions(settings):
    if settings.get("schema_version") != 1:
        raise ValueError("Нужны новые настройки автоподбора schema_version=1.")
    validate_morphology(settings["morphology"])
    if settings["annotation_mode"] not in ("detect", "obb", "segment"):
        raise ValueError("Выберите BBox, OBB или Segment.")
    if type(settings["use_hull"]) is not bool:
        raise ValueError("use_hull должен быть bool.")
    inversions = settings["inversions"]
    if (
        not inversions
        or any(type(v) is not bool for v in inversions)
        or len(set(inversions)) != len(inversions)
    ):
        raise ValueError("Неверный перебор инверсии.")
    methods = []
    if set(settings["methods"]) != set(SPECS):
        raise ValueError("Неверный набор методов.")
    for method, specs in SPECS.items():
        state = settings["methods"][method]
        if type(state["enabled"]) is not bool:
            raise ValueError("enabled должен быть bool.")
        if not state["enabled"]:
            continue
        if set(state["ranges"]) != set(specs):
            raise ValueError(f"Неверные параметры {method}.")
        methods.append(
            (
                method,
                list(specs),
                [
                    range_values(state["ranges"][key], spec)
                    for key, spec in specs.items()
                ],
            )
        )
    kernels = [
        range_values(settings["kernel_ranges"][op], [0, 0, 1, 0, 4096, True, False])
        for op in settings["kernel_ranges"]
    ]
    if set(settings["kernel_ranges"]) != {"close", "open", "erode", "dilate"}:
        raise ValueError("Нужны четыре диапазона морфологии.")
    return methods, kernels


def combination_count(settings):
    methods, kernels = dimensions(settings)
    return (
        sum(prod(len(v) for v in values) for _, _, values in methods)
        * prod(map(len, kernels))
        * len(settings["inversions"])
    )


def iter_presets(settings):
    methods, kernels = dimensions(settings)
    for method, keys, values in methods:
        for params in product(*values):
            for sizes in product(*kernels):
                morphology = deepcopy(settings["morphology"])
                mapping = dict(zip(settings["kernel_ranges"], sizes))
                for operation in morphology["operations"]:
                    operation["kernel_px"] = mapping[operation["operation"]]
                for invert in settings["inversions"]:
                    yield make_preset(
                        "threshold",
                        method=method,
                        params=dict(zip(keys, params)),
                        morphology=morphology,
                        invert=invert,
                        annotation_mode=settings["annotation_mode"],
                        use_hull=settings["use_hull"],
                    )


def validate_params(method, params):
    specs = SPECS[method]
    if set(params) != set(specs):
        raise ValueError("Неверный набор параметров метода.")
    for key, spec in specs.items():
        range_values([params[key], params[key], 1], spec)
