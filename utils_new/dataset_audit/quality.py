"""Per-image quality measurements and explicit review candidates; no Qt/models."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from utils_new.dataset_files_new import read_image_with_fallback
from utils_new.dataset_image_stats_new import ImageStats


@dataclass(frozen=True)
class QualityOptions:
    dark: float = 40
    bright: float = 215
    min_contrast: float = 10
    min_sharpness: float = 20

    def __post_init__(self):
        if not all(
            math.isfinite(v)
            for v in (self.dark, self.bright, self.min_contrast, self.min_sharpness)
        ):
            raise ValueError("Пороги качества должны быть конечными.")
        if (
            not 0 <= self.dark < self.bright <= 255
            or min(self.min_contrast, self.min_sharpness) < 0
        ):
            raise ValueError(
                "Нужны 0 ≤ тёмный порог < светлый порог ≤ 255 и неотрицательные контраст/резкость."
            )


class QualityJob:
    def __init__(self, pairs, options=None):
        self.pairs = pairs
        self.options = QualityOptions(**(options or {}))
        self._cancelled = False
        self.progress = lambda *args: None
        self.log = lambda message: None

    def cancel(self):
        self._cancelled = True

    def run(self):
        rows = []
        for index, pair in enumerate(self.pairs):
            if self._cancelled:
                return {"cancelled": True, "rows": []}
            path = str(pair[0])
            row = dict(path=path, issues=[], error=None)
            try:
                image = read_image_with_fallback(path)
                if image is None or not image.size or not np.isfinite(image).all():
                    raise ValueError(
                        "Изображение не читается или содержит некорректные значения."
                    )
                gray = ImageStats.to_gray(image)
                if gray is None or not gray.size or not np.isfinite(gray).all():
                    raise ValueError("Не удалось получить grayscale.")
                # The same 0..255 conversion as dataset brightness filters.
                h, w = gray.shape
                row.update(
                    width=w,
                    height=h,
                    brightness=float(gray.mean()),
                    contrast=float(gray.std()),
                    sharpness=float(
                        cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var()
                    ),
                )
                if row["brightness"] < self.options.dark:
                    row["issues"].append("Тёмный")
                if row["brightness"] > self.options.bright:
                    row["issues"].append("Светлый")
                if row["contrast"] < self.options.min_contrast:
                    row["issues"].append("Низкий контраст")
                if row["sharpness"] < self.options.min_sharpness:
                    row["issues"].append("Низкая резкость")
            except (OSError, ValueError, TypeError, cv2.error) as exc:
                row["error"] = str(exc)
                row["issues"].append("Не читается")
            rows.append(row)
            self.progress(index + 1, len(self.pairs))
        valid = [row for row in rows if not row["error"]]
        resolution = None
        brightness = None
        if valid:
            widths = [r["width"] for r in valid]
            heights = [r["height"] for r in valid]
            means = [r["brightness"] for r in valid]
            resolution = dict(
                widths=widths,
                heights=heights,
                aspects=[w / h for w, h in zip(widths, heights)],
                median_w=float(np.median(widths)),
                median_h=float(np.median(heights)),
                recommended_imgsz=min(
                    1280,
                    max(
                        32,
                        32 * math.ceil(max(np.median(widths), np.median(heights)) / 32),
                    ),
                ),
            )
            brightness = dict(
                means=means,
                dark_count=sum(v < self.options.dark for v in means),
                bright_count=sum(v > self.options.bright for v in means),
                mean_brightness=float(np.mean(means)),
            )
        return dict(
            cancelled=False,
            rows=rows,
            resolution=resolution,
            brightness=brightness,
            checked=len(valid),
            unreadable=len(rows) - len(valid),
            candidates=sum(bool(row["issues"]) for row in rows),
            thresholds=dict(
                dark=self.options.dark,
                bright=self.options.bright,
                min_contrast=self.options.min_contrast,
                min_sharpness=self.options.min_sharpness,
            ),
        )
