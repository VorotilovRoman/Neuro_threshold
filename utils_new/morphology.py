# -*- coding: utf-8 -*-
"""Общая морфология изображений для Threshold-Researcher.

Модуль не зависит от Qt, YOLO, PyTorch и формата аннотаций.
Вход — изображение NumPy (включая уже вырезанный ROI). При наличии
бинарной маски можно ограничить изменения объектом или его внутренним
контуром. Операции выполняются последовательно, в указанном порядке.

Публичный API:
    MorphologyStep(operation, kernel_px, iterations=1, shape=None)
    apply_morphology(image, operations, *, shape='Rectangle',
                     mask=None, contour_only=False, contour_width_px=None,
                     contour_shape=None)
    make_contour_mask(mask, width_px, shape='Rectangle')
    build_kernel(size_px, shape='Rectangle')

Принципы:
* kernel_px — РЕАЛЬНЫЙ размер квадратного ядра в пикселях; 4 означает 4x4,
  без неявного округления до нечётного числа. Для симметричной обработки
  рекомендуется выбирать нечётные значения. 0 — отключено, 1 — no-op.
* Для Open/Close используется cv2.morphologyEx; для Erode/Dilate —
  соответствующие функции OpenCV.
* Маска ограничивает ЗАПИСЬ результата после КАЖДОЙ операции. Пиксели
  за её пределами всегда остаются исходными; сами ядра читают соседние
  пиксели изображения, в том числе за пределами области маски.
* Контур — внутренняя полоса вокруг края маски; ширина задаётся именно
  в пикселях. Если не задана, берётся максимальный радиус ядра.
* Входные image и mask не изменяются, результат — новый ndarray.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import cv2
import numpy as np

SUPPORTED_OPERATIONS = ("close", "open", "erode", "dilate")
SUPPORTED_SHAPES = ("Rectangle", "Ellipse", "Cross")

_OPERATION_ALIASES = {
    "close": "close", "closing": "close", "закрытие": "close",
    "open": "open", "opening": "open", "открытие": "open",
    "erode": "erode", "erosion": "erode", "эрозия": "erode",
    "dilate": "dilate", "dilation": "dilate", "расширение": "dilate",
    "дилатация": "dilate",
}
_SHAPE_ALIASES = {
    "rectangle": cv2.MORPH_RECT, "rect": cv2.MORPH_RECT,
    "прямоугольник": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE, "elliptical": cv2.MORPH_ELLIPSE,
    "эллипс": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS, "крест": cv2.MORPH_CROSS,
}


@dataclass(frozen=True)
class MorphologyStep:
    """Одна операция. ``shape=None`` — наследовать общую форму ядра.

    ``operation``: close/open/erode/dilate (допускаются русские названия).
    ``kernel_px``: целое число пикселей (0/1 — нет эффекта).
    ``iterations``: число итераций операции OpenCV, по умолчанию 1.
    """

    operation: str
    kernel_px: int
    iterations: int = 1
    shape: Optional[str] = None


def _int_value(value, field: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field}: требуется целое число, получено {value!r}")
    converted = int(value)
    if converted < minimum:
        raise ValueError(f"{field}: требуется значение >= {minimum}, получено {converted}")
    return converted


def _shape_code(shape: str) -> int:
    if not isinstance(shape, str):
        raise TypeError("Форма ядра должна быть строкой")
    code = _SHAPE_ALIASES.get(shape.strip().lower())
    if code is None:
        raise ValueError(f"Неизвестная форма ядра {shape!r}; варианты: {SUPPORTED_SHAPES}")
    return code


def build_kernel(size_px: int, shape: str = "Rectangle") -> np.ndarray:
    """Квадратное структурирующее ядро OpenCV точного размера ``size_px``.

    В отличие от параметра операции, здесь размер должен быть >=1.
    """
    size = _int_value(size_px, "size_px", 1)
    return cv2.getStructuringElement(_shape_code(shape), (size, size))


def _normalized_steps(operations: Sequence[Union[MorphologyStep, Mapping]]) -> list[MorphologyStep]:
    if isinstance(operations, (str, bytes, Mapping)) or operations is None:
        raise TypeError("operations: требуется последовательность MorphologyStep или dict")
    result = []
    for i, value in enumerate(operations):
        if isinstance(value, MorphologyStep):
            step = value
        elif isinstance(value, Mapping):
            # Словари удобно сериализовать в JSON-профилях будущего UI.
            try:
                step = MorphologyStep(**value)
            except TypeError as exc:
                raise TypeError(f"Некорректная операция #{i + 1}: {exc}") from exc
        else:
            raise TypeError(f"Операция #{i + 1}: нужен MorphologyStep или dict")
        if not isinstance(step.operation, str):
            raise TypeError(f"Операция #{i + 1}: имя операции должно быть строкой")
        canonical = _OPERATION_ALIASES.get(step.operation.strip().lower())
        if canonical is None:
            raise ValueError(f"Неизвестная операция {step.operation!r}; варианты: {SUPPORTED_OPERATIONS}")
        size = _int_value(step.kernel_px, f"kernel_px для операции #{i + 1}", 0)
        count = _int_value(step.iterations, f"iterations для операции #{i + 1}", 1)
        if step.shape is not None:
            _shape_code(step.shape)
        result.append(MorphologyStep(canonical, size, count, step.shape))
    return result


def _region_mask(mask: np.ndarray, image_shape: tuple) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != image_shape[:2]:
        raise ValueError(
            f"mask: требуется одноканальный массив HxW={image_shape[:2]}, "
            f"получено {array.shape}"
        )
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
        raise TypeError("mask: ожидается числовой или логический массив")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("mask: запрещены NaN и бесконечность")
    return array > 0


def make_contour_mask(mask: np.ndarray, width_px: int,
                      shape: str = "Rectangle") -> np.ndarray:
    """Внутренняя полоса шириной ``width_px`` вдоль границы объекта.

    ``mask`` — 2D массив, где положительные значения обозначают объект.
    Возвращает bool-маску; внешняя часть объекта никогда не включается.
    Граница изображения считается границей объекта: если маска касается
    края изображения, контур вдоль края тоже учитывается.
    """
    width = _int_value(width_px, "width_px", 1)
    array = np.asarray(mask)
    region = _region_mask(array, array.shape)
    if not region.any():
        return np.zeros(region.shape, dtype=bool)
    # Радиус width достигается ядром (2*width+1) по каждому измерению.
    kernel = build_kernel(2 * width + 1, shape)
    eroded = cv2.erode(region.astype(np.uint8), kernel,
                       iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return region & (eroded == 0)


def apply_morphology(
    image: np.ndarray,
    operations: Sequence[Union[MorphologyStep, Mapping]],
    *,
    shape: str = "Rectangle",
    mask: Optional[np.ndarray] = None,
    contour_only: bool = False,
    contour_width_px: Optional[int] = None,
    contour_shape: Optional[str] = None,
) -> np.ndarray:
    """Применить операции в указанной последовательности к изображению или ROI.

    Args:
        image: HxW или HxWxC NumPy, поддерживаемый OpenCV тип (обычно uint8).
        operations: список MorphologyStep или dict со свойствами
            operation/kernel_px/iterations/shape. Порядок списка существенен.
            0 и 1 px отключают соответствующий шаг.
        shape: форма ядра по умолчанию, Rectangle / Ellipse / Cross.
        mask: необязательная 2D-маска HxW; положительное = область обработки.
            При отсутствии маски обрабатывается всё изображение.
        contour_only: обрабатывать только внутреннюю полосу края маски.
            Требует ``mask`` (для бинаризации её можно строить через
            ``mask = (binary == 0)`` либо ``mask = (binary > 0)``).
        contour_width_px: толщина полосы (пиксели); None = максимальный
            радиус ядра из активных шагов (но не меньше 1 px).
        contour_shape: форма ядра для построения полосы; None = shape.

    Returns:
        Новый ndarray исходной формы и типа. Исходные image и mask не меняются.

    Notes:
        Счёт ``iterations`` для Opening/Closing соответствует OpenCV:
        это N последовательных эрозий, затем N расширений (или наоборот),
        а не N повторений готовой пары операций.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image: требуется непустой ndarray HxW или HxWxC")
    if image.ndim == 3 and image.shape[2] < 1:
        raise ValueError("image: требуется хотя бы один канал")
    if image.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        raise TypeError(f"image: OpenCV морфология не поддерживает dtype {image.dtype}")
    default_shape_code = _shape_code(shape)
    steps = _normalized_steps(operations)
    if contour_shape is not None:
        _shape_code(contour_shape)
    if contour_width_px is not None:
        _int_value(contour_width_px, "contour_width_px", 1)
    if mask is None and contour_only:
        raise ValueError("contour_only=True требует маску области: mask=...")

    region = None if mask is None else _region_mask(mask, image.shape)
    active = [s for s in steps if s.kernel_px > 1]
    # Возвращается копия даже при выключенной обработке или пустой маске.
    current = np.array(image, copy=True, order="C")
    if not active or (region is not None and not region.any()):
        return current

    if contour_only:
        width = (int(contour_width_px) if contour_width_px is not None
                 else max(1, max(s.kernel_px // 2 for s in active)))
        region = make_contour_mask(region, width, shape=contour_shape or shape)
        if not region.any():
            return current

    for step in active:
        kernel = cv2.getStructuringElement(
            default_shape_code if step.shape is None else _shape_code(step.shape),
            (step.kernel_px, step.kernel_px),
        )
        if step.operation == "close":
            candidate = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel,
                                         iterations=step.iterations)
        elif step.operation == "open":
            candidate = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel,
                                         iterations=step.iterations)
        elif step.operation == "erode":
            candidate = cv2.erode(current, kernel, iterations=step.iterations)
        else:
            candidate = cv2.dilate(current, kernel, iterations=step.iterations)

        # OpenCV сворачивает HxWx1 в HxW; восстанавливаем исходную размерность.
        if current.ndim == 3 and current.shape[2] == 1 and candidate.ndim == 2:
            candidate = candidate[:, :, None]

        if region is None:
            current = candidate
        else:
            # Комбинация происходит на КАЖДОМ шаге, как в старой реализации
            # морфологии внутри маски объекта YOLO+AE/VAE.
            if current.ndim == 3:
                np.copyto(current, candidate, where=region[:, :, None])
            else:
                np.copyto(current, candidate, where=region)

    return current


__all__ = [
    "MorphologyStep", "SUPPORTED_OPERATIONS", "SUPPORTED_SHAPES",
    "apply_morphology", "build_kernel", "make_contour_mask",
]
