# -*- coding: utf-8 -*-
"""Image color/depth classification and conversion without Qt."""

import os

import cv2
import numpy as np


class ImageStats:
    @staticmethod
    def empty_color_stats():
        return {"gray_real": 0, "gray_visual": 0, "color": 0, "unknown": 0}

    @staticmethod
    def empty_bit_stats():
        return {
            "uint8_png": 0,
            "uint8_other": 0,
            "uint16": 0,
            "int16": 0,
            "float": 0,
            "other": 0,
            "unknown": 0,
        }

    @staticmethod
    def detect_color_type(img):
        if img is None:
            return "unknown"
        try:
            if len(img.shape) == 2:
                return "gray_real"
            if img.shape[2] == 1:
                return "gray_real"
            bgr = img[:, :, :3] if img.shape[2] >= 3 else img
            b = bgr[:, :, 0]
            g = bgr[:, :, 1]
            r = bgr[:, :, 2]
            if np.array_equal(b, g) and np.array_equal(g, r):
                return "gray_visual"
            return "color"
        except Exception:
            return "unknown"

    @staticmethod
    def detect_bit_type(img, img_path):
        if img is None:
            return "unknown"
        try:
            ext = os.path.splitext(str(img_path))[1].lower()
            is_png = ext == ".png"
            dt = img.dtype
            if dt == np.uint8:
                return "uint8_png" if is_png else "uint8_other"
            if dt == np.uint16:
                return "uint16"
            if dt == np.int16:
                return "int16"
            if dt in (np.float32, np.float64):
                return "float"
            return "other"
        except Exception:
            return "unknown"

    @staticmethod
    def to_uint8(img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img / 256.0).clip(0, 255).astype(np.uint8)
        if (
            img.dtype in (np.float32, np.float64)
            and img.size > 0
            and img.max() <= 1.0 + 1e-6
        ):
            return (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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
    def to_rgb3(img):
        """Shared RGB uint8 input for prompt models, retaining the project's depth policy."""
        if (img is None or img.ndim not in (2, 3) or not img.size
                or (img.ndim == 3 and img.shape[2] not in (1, 3, 4))):
            raise ValueError("Ожидается непустой серый, BGR или BGRA снимок")
        return cv2.cvtColor(ImageStats.to_bgr3(img), cv2.COLOR_BGR2RGB)

    @staticmethod
    def format_stats_text(color_stats, bit_stats, has_pairs):
        if not has_pairs:
            return "", False
        gr = color_stats.get("gray_real", 0)
        gv = color_stats.get("gray_visual", 0)
        c = color_stats.get("color", 0)
        u = color_stats.get("unknown", 0)
        p8 = bit_stats.get("uint8_png", 0)
        o8 = bit_stats.get("uint8_other", 0)
        b16 = bit_stats.get("uint16", 0)
        i16 = bit_stats.get("int16", 0)
        fl = bit_stats.get("float", 0)
        oth = bit_stats.get("other", 0)

        line1 = (
            f"⚫ Реально ЧБ: {gr}   |   "
            f"🌫 Визуально ЧБ: {gv}   |   "
            f"🎨 Реально цветных: {c}"
        )
        if u:
            line1 += f"   |   ❓ Неизвестно: {u}"

        non_8bit = b16 + i16 + fl + oth
        line2 = f"🔵 8-bit PNG: {p8}   |   🟡 8-bit non-PNG: {o8}"
        if non_8bit:
            parts = []
            if b16:
                parts.append(f"16-bit: {b16}")
            if i16:
                parts.append(f"int16: {i16}")
            if fl:
                parts.append(f"float: {fl}")
            if oth:
                parts.append(f"other: {oth}")
            line2 += "   |   ⚠️ " + ", ".join(parts)
        return line1 + "\n" + line2, True


def resize_to_max_side(img, max_side=1024, *, rounding="round"):
    """Preserve aspect ratio with an explicit rounding policy for trained pipelines."""
    if img is None:
        return None
    if rounding not in ("round", "floor"):
        raise ValueError("rounding must be round or floor")
    if max_side is not None and (int(max_side) != max_side or max_side < 1):
        raise ValueError("max_side must be a positive integer or None")
    if max_side is None or max(img.shape[:2]) <= max_side:
        return img
    h, w = img.shape[:2]
    factor = float(max_side) / max(w, h)
    convert = round if rounding == "round" else int
    return cv2.resize(
        img,
        (max(1, convert(w * factor)), max(1, convert(h * factor))),
        interpolation=cv2.INTER_AREA,
    )


def segmentation_uint8(image):
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image / 256).astype(np.uint8)
    if image.dtype in (np.float32, np.float64) and image.size and image.max() <= 1:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def resize_for_segmentation(image, max_side):
    return resize_to_max_side(image, max_side, rounding="floor")


normalize_to_uint8 = ImageStats.to_uint8
convert_to_grayscale = ImageStats.to_gray


def resize_segmentation_pair(
    image, mask, size, mode="hard", *, is_train=False, mask_fill=0
):
    """OpenCV linear image resize and exact nearest index sampling, shared with inference."""
    from utils_new.annotations import ImageSize
    from utils_new.annotations.masks import mask_roi_at_image_scale

    target_h, target_w = (size, size) if isinstance(size, int) else size
    height, width = image.shape[:2]
    if mode not in ("hard", "keep_aspect_pad", "random_crop"):
        raise ValueError("Неизвестный режим ресайза.")
    if mode == "hard":
        nh, nw = target_h, target_w
    else:
        scale = (min if mode == "keep_aspect_pad" else max)(
            target_h / height, target_w / width
        )
        nh, nw = max(1, round(height * scale)), max(1, round(width * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    labels = (
        mask_roi_at_image_scale(mask, ImageSize(nw, nh), (0, 0, nw, nh))
        if mask is not None
        else None
    )
    if mode == "keep_aspect_pad":
        top, left = (target_h - nh) // 2, (target_w - nw) // 2
        result = np.zeros((target_h, target_w, *image.shape[2:]), dtype=image.dtype)
        result[top : top + nh, left : left + nw] = resized
        if labels is not None:
            padded = np.full((target_h, target_w), mask_fill, dtype=labels.dtype)
            padded[top : top + nh, left : left + nw] = labels
            labels = padded
        return result, labels, (left, top, nh, nw)
    if mode == "random_crop":
        top = (
            int(np.random.randint(nh - target_h + 1))
            if is_train
            else (nh - target_h) // 2
        )
        left = (
            int(np.random.randint(nw - target_w + 1))
            if is_train
            else (nw - target_w) // 2
        )
        resized = resized[top : top + target_h, left : left + target_w]
        if labels is not None:
            labels = labels[top : top + target_h, left : left + target_w]
    return resized, labels, (0, 0, target_h, target_w)
