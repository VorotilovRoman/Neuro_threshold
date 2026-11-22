# -*- coding: utf-8 -*-
"""Пороговая бинаризация изображений (10 методов) без зависимостей от Qt.

Единая реализация для вкладок, ансамблей и автоподбора.
Белые пиксели всегда означают интенсивность выше вычисленного порога.

apply_threshold_method(gray_img, method_name, params) -> (binary, threshold)
threshold = None у локально адаптивных методов; изображение входа uint8 grayscale.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "simple_threshold",
    "otsu_threshold",
    "triangle_threshold",
    "niblack_threshold",
    "sauvola_threshold",
    "isodata_threshold",
    "background_symmetry_threshold",
    "row_adaptive_threshold",
    "normalize_method_name",
    "apply_threshold_method",
]


def simple_threshold(img, thresh):
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary, thresh


def otsu_threshold(img):
    th, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, th


def triangle_threshold(img):
    th, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary, th


def niblack_threshold(img, window_size=15, k=-0.2):
    if window_size < 1:
        raise ValueError("Window должен быть положительным.")
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter(
        (img.astype(np.float32)) ** 2, cv2.CV_32F, (window_size, window_size)
    )
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean + k * std
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None


def sauvola_threshold(img, window_size=25, k=0.2, R=128):
    if not np.isfinite(R) or R <= 0:
        raise ValueError("Sauvola R должен быть положительным.")
    if window_size < 1:
        raise ValueError("Window должен быть положительным.")
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter(
        (img.astype(np.float32)) ** 2, cv2.CV_32F, (window_size, window_size)
    )
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean * (1 + k * (std / R - 1))
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None


def isodata_threshold(img, initial_threshold=128):
    threshold = initial_threshold
    img_flat = img.ravel().astype(np.float32)
    for _ in range(256):
        bg = img_flat[img_flat <= threshold]
        fg = img_flat[img_flat > threshold]
        if len(bg) == 0 or len(fg) == 0:
            break
        new_threshold = float((np.mean(bg) + np.mean(fg)) / 2)
        converged = abs(new_threshold - threshold) < 0.5
        threshold = new_threshold
        if converged:
            break
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary, float(threshold)


def background_symmetry_threshold(img, excess_threshold=0.2):
    """Project histogram heuristic; symmetric background peak, excess tail.

    Reflect a high-valued background before estimating the tail. If there is
    no asymmetric excess, use Otsu rather than an arbitrary peak+20 threshold.
    """
    if not 0 <= excess_threshold <= 1:
        raise ValueError("Excess должен быть в диапазоне [0, 1].")
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    reflected = int(np.argmax(hist)) > 127
    if reflected:
        hist = hist[::-1]
    peak = int(np.argmax(hist))
    excess = np.zeros(256)
    for value in range(peak + 1, 256):
        mirror = 2 * peak - value
        excess[value] = max(0, hist[value] - (hist[mirror] if mirror >= 0 else 0))
    candidates = np.flatnonzero(excess > excess.max() * excess_threshold)
    if not len(candidates):
        return otsu_threshold(img)
    threshold = (peak + int(candidates[0])) / 2
    if reflected:
        threshold = 255 - threshold
    return simple_threshold(img, threshold)


def row_adaptive_threshold(img, window_size=50, k=0.5):
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("Window должен быть положительным целым числом.")
    h, w = img.shape
    binary = np.zeros_like(img)
    for row in range(h):
        row_data = img[row, :].astype(np.float32)
        for col in range(0, w, window_size):
            end = min(col + window_size, w)
            window = row_data[col:end]
            if len(window) > 0:
                mean = np.mean(window)
                std = np.std(window)
                thresh = max(0, mean - k * std)
                binary[row, col:end] = (row_data[col:end] > thresh).astype(
                    np.uint8
                ) * 255
    return binary, None


def normalize_method_name(method_name):
    """Приводит имя метода к каноническому виду для UI."""
    method_map = {
        "otsu": "Otsu",
        "otsu (opencv)": "Otsu",
        "triangle": "Triangle",
        "triangle (opencv)": "Triangle",
        "simple threshold": "Simple Threshold",
        "adaptive mean": "Adaptive Mean",
        "adaptive gauss": "Adaptive Gauss",
        "niblack": "Niblack",
        "sauvola": "Sauvola",
        "isodata": "ISODATA",
        "background symmetry": "Background Symmetry",
        "row adaptive": "Row Adaptive",
    }
    lower_name = method_name.lower().strip()
    if lower_name in method_map:
        return method_map[lower_name]
    if lower_name.endswith(" (opencv)"):
        base = lower_name[:-9]
        if base in method_map:
            return method_map[base]
    return method_name


def apply_threshold_method(gray_img, method_name, params):
    """Применяет метод бинаризации по имени и параметрам."""
    if gray_img.dtype != np.uint8 or gray_img.ndim != 2 or not gray_img.size:
        raise ValueError("Нужно непустое grayscale изображение uint8.")
    if any(not np.isfinite(value) for value in params.values()):
        raise ValueError("Параметры должны быть конечными числами.")
    method_name = normalize_method_name(method_name)
    if method_name in ("Adaptive Mean", "Adaptive Gauss", "Niblack", "Sauvola"):
        window = params.get("window", 25)
        if int(window) != window or window < 3 or window % 2 != 1:
            raise ValueError(
                "Размер локального окна должен быть нечётным целым числом ≥ 3."
            )
    if method_name == "Simple Threshold":
        thresh = params.get("threshold", 128)
        binary, _ = simple_threshold(gray_img, thresh)
        return binary, thresh
    elif method_name == "Otsu":
        binary, thresh = otsu_threshold(gray_img)
        return binary, thresh
    elif method_name == "Triangle":
        binary, thresh = triangle_threshold(gray_img)
        return binary, thresh
    elif method_name == "Adaptive Mean":
        ws = params.get("window", 25)
        c = params.get("c", 3)
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ws, c
        )
        return binary, None
    elif method_name == "Adaptive Gauss":
        ws = params.get("window", 25)
        c = params.get("c", 3)
        binary = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ws, c
        )
        return binary, None
    elif method_name == "Niblack":
        ws = params.get("window", 25)
        k = params.get("k", -0.2)
        binary, _ = niblack_threshold(gray_img, ws, k)
        return binary, None
    elif method_name == "Sauvola":
        ws = params.get("window", 25)
        k = params.get("k", 0.2)
        r = params.get("r", 128)
        binary, _ = sauvola_threshold(gray_img, ws, k, r)
        return binary, None
    elif method_name == "ISODATA":
        init = params.get("init", 128)
        binary, thresh = isodata_threshold(gray_img, init)
        return binary, thresh
    elif method_name == "Background Symmetry":
        excess = params.get("excess", 0.2)
        binary, thresh = background_symmetry_threshold(gray_img, excess)
        return binary, thresh
    elif method_name == "Row Adaptive":
        ws = params.get("window", 50)
        k = params.get("k", 0.5)
        binary, _ = row_adaptive_threshold(gray_img, ws, k)
        return binary, None
    else:
        raise ValueError(f"Unknown method: {method_name}")


def apply_threshold_preset(image, gray, preset):
    """Canonical preset pipeline shared by preview, ensemble and research."""
    from utils_new.images.morphology import apply_morphology
    from utils_new.segmentation.postprocessing import extract_objects
    from utils_new.segmentation.presets import validate_preset

    validate_preset(preset, "threshold")
    if image.shape[:2] != gray.shape:
        raise ValueError("Размер снимка и grayscale различается.")
    binary, threshold = apply_threshold_method(gray, preset["method"], preset["params"])
    morphology = preset["morphology"]
    mask = apply_morphology(
        cv2.bitwise_not(binary), morphology["operations"], shape=morphology["shape"]
    )
    if preset["invert"]:
        mask = cv2.bitwise_not(mask)
    objects = extract_objects(
        image, mask, preset["annotation_mode"], preset["use_hull"]
    )[1]
    return mask, objects, threshold


def get_detections_from_preset(image, gray, preset, confidence=0.7, *, class_id=0):
    """Return enclosing Detect boxes with a user-assigned, uncalibrated score."""
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Оценка пресета должна быть в диапазоне 0..1.")
    if int(class_id) != class_id or class_id < 0:
        raise ValueError("ID класса должен быть целым ≥0.")
    _, objects, _ = apply_threshold_preset(image, gray, preset)
    detections = []
    for obj in objects:
        if obj[0] == "detect":
            cx, cy, w, h = obj[2:]
        else:
            points = np.asarray(obj[2], dtype=float).reshape(-1, 2)
            low, high = points.min(axis=0), points.max(axis=0)
            cx, cy = (low + high) / 2
            w, h = high - low
        if w > 0 and h > 0:
            detections.append(
                (
                    int(class_id),
                    float(cx),
                    float(cy),
                    float(w),
                    float(h),
                    float(confidence),
                )
            )
    return detections
