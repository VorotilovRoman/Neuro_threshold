# -*- coding: utf-8 -*-
"""Пороговая бинаризация изображений (10 методов) без зависимостей от Qt.

Алгоритмы перенесены из utils_ops/segmentation_ops.py без изменения
математических операций, порогов, порядка и возвращаемых значений.

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
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter((img.astype(np.float32))**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean + k * std
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None

def sauvola_threshold(img, window_size=25, k=0.2, R=128):
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter((img.astype(np.float32))**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean * (1 + k * (std / R - 1))
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None

def isodata_threshold(img, initial_threshold=128):
    threshold = initial_threshold
    img_flat = img.ravel().astype(np.float32)
    while True:
        bg = img_flat[img_flat < threshold]
        fg = img_flat[img_flat >= threshold]
        if len(bg) == 0 or len(fg) == 0:
            break
        mean_bg = np.mean(bg)
        mean_fg = np.mean(fg)
        new_threshold = (mean_bg + mean_fg) / 2
        if abs(new_threshold - threshold) < 0.5:
            break
        threshold = new_threshold
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary, int(threshold)

def background_symmetry_threshold(img, excess_threshold=0.2):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    bg_peak = np.argmax(hist)
    bg_hist_expected = np.zeros_like(hist)
    bg_hist_expected[:bg_peak + 1] = hist[:bg_peak + 1]
    for i in range(1, bg_peak + 1):
        if bg_peak + i < 256:
            bg_hist_expected[bg_peak + i] = hist[bg_peak - i]
    excess = hist - bg_hist_expected
    excess = np.maximum(excess, 0)
    if excess.max() > 0:
        excess_norm = excess / excess.max()
        threshold_candidates = np.where(excess_norm > excess_threshold)[0]
        if len(threshold_candidates) > 0:
            threshold = threshold_candidates[0]
        else:
            threshold = bg_peak + 20
    else:
        threshold = bg_peak + 20
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary, threshold

def row_adaptive_threshold(img, window_size=50, k=0.5):
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
                binary[row, col:end] = (row_data[col:end] < thresh).astype(np.uint8) * 255
    return binary, None

def normalize_method_name(method_name):
    """Приводит имя метода к каноническому виду для UI."""
    method_map = {
        'otsu': 'Otsu',
        'otsu (opencv)': 'Otsu',
        'triangle': 'Triangle',
        'triangle (opencv)': 'Triangle',
        'simple threshold': 'Simple Threshold',
        'adaptive mean': 'Adaptive Mean',
        'adaptive gauss': 'Adaptive Gauss',
        'niblack': 'Niblack',
        'sauvola': 'Sauvola',
        'isodata': 'ISODATA',
        'background symmetry': 'Background Symmetry',
        'row adaptive': 'Row Adaptive',
    }
    lower_name = method_name.lower().strip()
    if lower_name in method_map:
        return method_map[lower_name]
    if lower_name.endswith(' (opencv)'):
        base = lower_name[:-9]
        if base in method_map:
            return method_map[base]
    return method_name

def apply_threshold_method(gray_img, method_name, params):
    """Применяет метод бинаризации по имени и параметрам."""
    method_name = normalize_method_name(method_name)
    if method_name == "Simple Threshold":
        thresh = params.get('threshold', 128)
        binary, _ = simple_threshold(gray_img, thresh)
        return binary, thresh
    elif method_name == "Otsu":
        binary, thresh = otsu_threshold(gray_img)
        return binary, thresh
    elif method_name == "Triangle":
        binary, thresh = triangle_threshold(gray_img)
        return binary, thresh
    elif method_name == "Adaptive Mean":
        ws = params.get('window', 25)
        if ws % 2 == 0:
            ws += 1
        if ws < 3:
            ws = 3
        c = params.get('c', 3)
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, ws, c)
        return binary, None
    elif method_name == "Adaptive Gauss":
        ws = params.get('window', 25)
        if ws % 2 == 0:
            ws += 1
        if ws < 3:
            ws = 3
        c = params.get('c', 3)
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ws, c)
        return binary, None
    elif method_name == "Niblack":
        ws = params.get('window', 25)
        k = params.get('k', -0.2)
        binary, _ = niblack_threshold(gray_img, ws, k)
        return binary, None
    elif method_name == "Sauvola":
        ws = params.get('window', 25)
        k = params.get('k', 0.2)
        r = params.get('r', 128)
        binary, _ = sauvola_threshold(gray_img, ws, k, r)
        return binary, None
    elif method_name == "ISODATA":
        init = params.get('init', 128)
        binary, thresh = isodata_threshold(gray_img, init)
        return binary, thresh
    elif method_name == "Background Symmetry":
        excess = params.get('excess', 0.2)
        binary, thresh = background_symmetry_threshold(gray_img, excess)
        return binary, thresh
    elif method_name == "Row Adaptive":
        ws = params.get('window', 50)
        k = params.get('k', 0.5)
        binary, _ = row_adaptive_threshold(gray_img, ws, k)
        return binary, None
    else:
        raise ValueError(f"Unknown method: {method_name}")
