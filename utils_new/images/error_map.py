"""AE/VAE reconstruction-error visualization without model inference."""

import cv2
import numpy as np


def render_ae_error_map(
    error,
    base,
    *,
    valid_region=None,
    overlay=True,
    alpha=0.5,
    show_mask=False,
    threshold=float("nan"),
):
    """Render an AE/VAE error heatmap with the original percentile stretch.

    Neural inference and torch pre-processing are kept in the YOLO+AE tab;
    this routine only handles NumPy/OpenCV post-processing and visualization.
    """
    if error is None or base is None:
        return None
    h, w = base.shape[:2]
    if valid_region is not None:
        x1, y1, x2, y2 = valid_region
        roi = error[y1:y2, x1:x2]
        if roi.size == 0:
            roi = error
    else:
        roi = error
    full = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(np.percentile(full, 1)), float(np.percentile(full, 99))
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    norm = np.clip((full - lo) / (hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if overlay:
        a = max(0.0, min(1.0, float(alpha)))
        image = cv2.addWeighted(base, 1 - a, heat, a, 0)
    else:
        image = heat
    if show_mask and np.isfinite(threshold):
        bw = (roi > threshold).astype(np.uint8)
        full_bw = cv2.resize(bw, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            full_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
    return image
