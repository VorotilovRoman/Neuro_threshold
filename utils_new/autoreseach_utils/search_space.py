"""Validated finite ranges and lazy canonical threshold/gradient presets; no Qt."""

from copy import deepcopy
from decimal import Decimal
from itertools import product
from math import prod, isfinite

from utils_new.segmentation.presets import (
    default_morphology,
    make_preset,
    validate_morphology,
)

# Parameter key: start, end, step, lower, upper, integer, odd, optional quantum.
SPECS = {
    "Simple Threshold": {"threshold": (1, 254, 1, 0, 255, True, False)},
    "Otsu": {},
    "Triangle": {},
    "Adaptive Mean": {
        "window": (3, 51, 2, 3, 201, True, True),
        "c": (-10, 10, 5, -20, 20, True, False),
    },
    "Adaptive Gauss": {
        "window": (3, 51, 2, 3, 201, True, True),
        "c": (-10, 10, 5, -20, 20, True, False),
    },
    "Niblack": {
        "window": (3, 51, 2, 3, 201, True, True),
        "k": (-0.2, 0.2, 0.1, -1, 1, False, False, 0.01),
    },
    "Sauvola": {
        "window": (3, 51, 2, 3, 201, True, True),
        "k": (0.1, 0.3, 0.1, -1, 1, False, False, 0.01),
        "r": (128, 128, 1, 1, 255, True, False),
    },
    "ISODATA": {"init": (0, 255, 1, 0, 255, True, False)},
    "Background Symmetry": {"excess": (0.01, 1, 0.01, 0.01, 1, False, False, 0.01)},
    "Row Adaptive": {
        "window": (10, 100, 10, 10, 500, True, False),
        "k": (0, 1, 0.1, 0, 1, False, False, 0.01),
    },
}


GRADIENT_SPECS = {
    "Sobel": {
        "ksize": (3, 7, 2, 1, 31, True, True),
        "scale": (1, 3, 1, 1, 10, True, False),
    },
    "Scharr": {"scale": (1, 3, 1, 1, 10, True, False)},
    "Laplacian": {
        "ksize": (3, 7, 2, 1, 31, True, True),
        "scale": (1, 3, 1, 1, 10, True, False),
    },
    "Canny": {
        "threshold1": (50, 50, 1, 0, 255, True, False),
        "threshold2": (150, 150, 1, 0, 255, True, False),
        "aperture": (3, 7, 2, 3, 7, True, True),
    },
    "Prewitt": {},
    "Roberts": {},
    "Kirsch": {},
}
POST_SPECS = {
    "threshold": (127, 127, 1, 0, 255, True, False),
    "min_area_percent": (0, 0, 1, 0, 100, False, False, 0.000001),
}


def search_kind(settings):
    kind = settings.get("kind", "threshold")
    if kind not in ("threshold", "gradient"):
        raise ValueError("Неизвестный вид автоподбора.")
    return kind


def specs_for(kind):
    if kind not in ("threshold", "gradient"):
        raise ValueError("Неизвестный вид автоподбора.")
    return SPECS if kind == "threshold" else GRADIENT_SPECS


def default_settings(kind="threshold"):
    specs = specs_for(kind)
    state = {
        "schema_version": 1,
        "kind": kind,
        "methods": {
            name: {
                "enabled": name
                == ("Simple Threshold" if kind == "threshold" else "Sobel"),
                "ranges": {key: list(spec[:3]) for key, spec in params.items()},
            }
            for name, params in specs.items()
        },
        "morphology": default_morphology(),
        "kernel_ranges": {op: [0, 0, 1] for op in ("close", "open", "erode", "dilate")},
        "inversions": [False],
        "annotation_mode": "detect",
        "use_hull": False,
    }
    if kind == "gradient":
        state["post_ranges"] = {key: list(spec[:3]) for key, spec in POST_SPECS.items()}
        state["fill_contours"] = [True]
    return state


def range_values(values, spec):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("Диапазон: [начало, конец, шаг].")
    if any(type(v) not in (int, float) or not isfinite(v) for v in values):
        raise ValueError("Диапазон должен содержать конечные числа.")
    start, end, step = map(lambda v: Decimal(str(v)), values)
    lo, hi, integer, odd = spec[3:7]
    if len(spec) > 7:
        quantum = Decimal(str(spec[7]))
        if any(v % quantum != 0 for v in (start, end, step)):
            raise ValueError(
                f"Точность рабочего регулятора: {quantum}. Диапазон не должен требовать округления."
            )
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
    kind = search_kind(settings)
    specs_map = specs_for(kind)
    gradient_axes(settings)
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
    if set(settings["methods"]) != set(specs_map):
        raise ValueError("Неверный набор методов.")
    for method, specs in specs_map.items():
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


def gradient_axes(settings):
    if search_kind(settings) == "threshold":
        return [[None], [None], [None]]
    if set(settings["post_ranges"]) != set(POST_SPECS):
        raise ValueError("Неверные параметры постобработки градиента.")
    fills = settings["fill_contours"]
    if (
        not fills
        or any(type(v) is not bool for v in fills)
        or len(set(fills)) != len(fills)
    ):
        raise ValueError("Неверный перебор заливки.")
    return [
        range_values(settings["post_ranges"][key], spec)
        for key, spec in POST_SPECS.items()
    ] + [fills]


def _parameter_count(method, keys, values):
    if method != "Canny":
        return prod(map(len, values))
    from bisect import bisect_left

    first, second, apertures = values
    return sum(len(second) - bisect_left(second, t1) for t1 in first) * len(apertures)


def combination_count(settings):
    methods, kernels = dimensions(settings)
    return (
        sum(_parameter_count(method, keys, values) for method, keys, values in methods)
        * prod(map(len, kernels))
        * len(settings["inversions"])
        * prod(map(len, gradient_axes(settings)))
    )


def gradient_preset(method, params, **extras):
    fields = dict(
        sobel_kernel=3,
        sobel_scale=1,
        lap_kernel=3,
        lap_scale=1,
        canny_thresh1=50,
        canny_thresh2=150,
        canny_aperture=3,
    )
    if method in ("Sobel", "Scharr"):
        fields["sobel_scale"] = params["scale"]
        if method == "Sobel":
            fields["sobel_kernel"] = params["ksize"]
    elif method == "Laplacian":
        fields.update(lap_kernel=params["ksize"], lap_scale=params["scale"])
    elif method == "Canny":
        fields.update(
            canny_thresh1=params["threshold1"],
            canny_thresh2=params["threshold2"],
            canny_aperture=params["aperture"],
        )
    return make_preset("gradient", method=method, **fields, **extras)


def method_params(preset):
    if preset["kind"] == "threshold":
        return preset["params"]
    method = preset["method"]
    if method in ("Sobel", "Scharr"):
        result = {"scale": preset["sobel_scale"]}
        if method == "Sobel":
            result["ksize"] = preset["sobel_kernel"]
        return result
    if method == "Laplacian":
        return dict(ksize=preset["lap_kernel"], scale=preset["lap_scale"])
    if method == "Canny":
        return dict(
            threshold1=preset["canny_thresh1"],
            threshold2=preset["canny_thresh2"],
            aperture=preset["canny_aperture"],
        )
    return {}


def iter_presets(settings):
    methods, kernels = dimensions(settings)
    kind = search_kind(settings)
    post_axes = gradient_axes(settings)
    for method, keys, values in methods:
        for params_values in product(*values):
            params = dict(zip(keys, params_values))
            if method == "Canny" and params["threshold1"] > params["threshold2"]:
                continue
            for sizes in product(*kernels):
                morphology = deepcopy(settings["morphology"])
                mapping = dict(zip(settings["kernel_ranges"], sizes))
                for operation in morphology["operations"]:
                    operation["kernel_px"] = mapping[operation["operation"]]
                for threshold, area, fill in product(*post_axes):
                    for invert in settings["inversions"]:
                        extras = dict(
                            morphology=morphology,
                            invert=invert,
                            annotation_mode=settings["annotation_mode"],
                            use_hull=settings["use_hull"],
                        )
                        if kind == "threshold":
                            yield make_preset(
                                kind, method=method, params=params, **extras
                            )
                        else:
                            yield gradient_preset(
                                method,
                                params,
                                threshold=threshold,
                                min_area_percent=area,
                                fill_contours=fill,
                                **extras,
                            )


def validate_params(method, params, kind="threshold"):
    specs = specs_for(kind)[method]
    if set(params) != set(specs):
        raise ValueError("Неверный набор параметров метода.")
    for key, spec in specs.items():
        range_values([params[key], params[key], 1], spec)
    if method == "Canny" and params["threshold1"] > params["threshold2"]:
        raise ValueError("Canny: threshold1 должен быть ≤ threshold2.")


def validate_search_preset(preset):
    from utils_new.segmentation.presets import validate_preset

    validate_preset(preset)
    kind = preset["kind"]
    specs_for(kind)
    validate_params(preset["method"], method_params(preset), kind)
    if kind == "gradient":
        for key, spec in POST_SPECS.items():
            range_values([preset[key], preset[key], 1], spec)
        if type(preset["fill_contours"]) is not bool:
            raise ValueError("fill_contours должен быть bool.")
    return preset
