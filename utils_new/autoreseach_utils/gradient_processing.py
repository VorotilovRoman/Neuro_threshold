"""Gradient stages shared by the live tab and numerical research."""

import cv2
import numpy as np
from utils_ops.gradient_ops import apply_gradient_method


def gradient_binary(gray, method, params, threshold):
    gradient = apply_gradient_method(cv2.bitwise_not(gray), method, params)
    _, binary = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)
    return gradient, binary


def finalize_gradient_mask(processed, min_area_percent, fill_contours):
    processed = processed.copy()
    min_area = min_area_percent / 100.0 * processed.size
    if min_area > 0:
        contours, _ = cv2.findContours(
            processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(processed, [contour], -1, 0, thickness=cv2.FILLED)
    if not fill_contours:
        return processed
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(processed)
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled
