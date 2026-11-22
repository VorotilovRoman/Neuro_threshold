"""Decode one semantic-segmentation output; preserve class IDs and reject ambiguity."""

import cv2
import numpy as np


def decode_output(output, *, output_kind="logits", threshold=0.5, target_shape=None):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Порог вероятности должен быть в диапазоне [0, 1].")
    values = np.asarray(output)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Модель вернула пустой выход, NaN или Infinity.")
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if output_kind == "labels":
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if (
            values.ndim != 2
            or np.any(values < 0)
            or np.any(values != np.floor(values))
            or values.max() > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Выход labels должен быть двумерной картой неотрицательных целых классов."
            )
        labels = values.astype(np.int32)
        if target_shape is not None:
            labels = cv2.resize(
                labels,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return labels
    if output_kind not in ("logits", "probabilities"):
        raise ValueError("Укажите output_kind: logits, probabilities или labels.")
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(
            "Ожидается выход NCHW (один снимок) или CHW/HW. Detection-выходы не поддерживаются."
        )
    # Preserve float64 outputs: forcing large finite logits to float32 can turn
    # them into Inf and silently classify an entire image as background.
    values = values.astype(np.result_type(values.dtype, np.float32))
    if output_kind == "probabilities" and (values.min() < 0 or values.max() > 1):
        raise ValueError(
            "Вероятности должны быть в диапазоне [0, 1]. Выберите logits для необработанного выхода."
        )
    if target_shape is not None and values.shape[1:] != tuple(target_shape):
        values = np.stack(
            [
                cv2.resize(
                    channel,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for channel in values
            ]
        )
    channels = values.shape[0]
    if channels == 1:
        probability = (
            1 / (1 + np.exp(-np.clip(values[0], -80, 80)))
            if output_kind == "logits"
            else values[0]
        )
        return (probability > threshold).astype(np.uint8) * 255
    if channels == 2:
        if output_kind == "logits":
            with np.errstate(over="ignore", under="ignore"):
                values = np.exp(values - values.max(axis=0, keepdims=True))
            values /= values.sum(axis=0, keepdims=True)
        return (values[1] > threshold).astype(np.uint8) * 255
    return values.argmax(axis=0).astype(np.int32)
