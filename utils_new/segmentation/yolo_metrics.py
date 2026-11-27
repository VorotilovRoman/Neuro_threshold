"""Scalar Ultralytics validation metrics: distinguish masks from boxes."""

import math


def validation_metrics(result):
    metrics = {}
    for key, value in (getattr(result, "results_dict", None) or {}).items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scalar):
            metrics[key] = scalar
    primary = None
    for name, attribute in (("box", "box"), ("mask", "seg"), ("pose", "pose")):
        summary = getattr(result, attribute, None)
        if summary is None:
            continue
        values = {}
        for key, field in (
            ("precision", "mp"),
            ("recall", "mr"),
            ("mAP50", "map50"),
            ("mAP50-95", "map"),
        ):
            value = getattr(summary, field, None)
            if value is not None and math.isfinite(float(value)):
                values[key] = float(value)
                metrics[f"{name}/{key}"] = float(value)
        if values:
            primary = values
    # A segmentation model's primary score is MASK AP; box scores stay separate.
    if primary:
        metrics.update(primary)
    return metrics
