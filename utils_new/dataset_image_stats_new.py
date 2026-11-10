# -*- coding: utf-8 -*-
"""Image color/depth classification and conversion without Qt."""
import os

import cv2
import numpy as np


class ImageStats:
    @staticmethod
    def empty_color_stats():
        return {'gray_real': 0, 'gray_visual': 0, 'color': 0, 'unknown': 0}

    @staticmethod
    def empty_bit_stats():
        return {'uint8_png': 0, 'uint8_other': 0, 'uint16': 0,
                'int16': 0, 'float': 0, 'other': 0, 'unknown': 0}

    @staticmethod
    def detect_color_type(img):
        if img is None:
            return 'unknown'
        try:
            if len(img.shape) == 2:
                return 'gray_real'
            if img.shape[2] == 1:
                return 'gray_real'
            bgr = img[:, :, :3] if img.shape[2] >= 3 else img
            b = bgr[:, :, 0]; g = bgr[:, :, 1]; r = bgr[:, :, 2]
            if np.array_equal(b, g) and np.array_equal(g, r):
                return 'gray_visual'
            return 'color'
        except Exception:
            return 'unknown'

    @staticmethod
    def detect_bit_type(img, img_path):
        if img is None:
            return 'unknown'
        try:
            ext = os.path.splitext(str(img_path))[1].lower()
            is_png = ext == '.png'
            dt = img.dtype
            if dt == np.uint8:
                return 'uint8_png' if is_png else 'uint8_other'
            if dt == np.uint16:
                return 'uint16'
            if dt == np.int16:
                return 'int16'
            if dt in (np.float32, np.float64):
                return 'float'
            return 'other'
        except Exception:
            return 'unknown'

    @staticmethod
    def to_uint8(img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img / 256.0).clip(0, 255).astype(np.uint8)
        if img.dtype in (np.float32, np.float64) and img.size > 0 \
                and img.max() <= 1.0 + 1e-6:
            return (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def to_display_bgr(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def to_gray(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 1:
            return img[:, :, 0]
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_bgr3(img):
        img = ImageStats.to_uint8(img)
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    @staticmethod
    def format_stats_text(color_stats, bit_stats, has_pairs):
        if not has_pairs:
            return "", False
        gr = color_stats.get('gray_real', 0)
        gv = color_stats.get('gray_visual', 0)
        c = color_stats.get('color', 0)
        u = color_stats.get('unknown', 0)
        p8 = bit_stats.get('uint8_png', 0)
        o8 = bit_stats.get('uint8_other', 0)
        b16 = bit_stats.get('uint16', 0)
        i16 = bit_stats.get('int16', 0)
        fl = bit_stats.get('float', 0)
        oth = bit_stats.get('other', 0)

        line1 = (f"⚫ Реально ЧБ: {gr}   |   "
                 f"🌫 Визуально ЧБ: {gv}   |   "
                 f"🎨 Реально цветных: {c}")
        if u:
            line1 += f"   |   ❓ Неизвестно: {u}"

        non_8bit = b16 + i16 + fl + oth
        line2 = f"🔵 8-bit PNG: {p8}   |   🟡 8-bit non-PNG: {o8}"
        if non_8bit:
            parts = []
            if b16: parts.append(f"16-bit: {b16}")
            if i16: parts.append(f"int16: {i16}")
            if fl: parts.append(f"float: {fl}")
            if oth: parts.append(f"other: {oth}")
            line2 += "   |   ⚠️ " + ", ".join(parts)
        return line1 + "\n" + line2, True

