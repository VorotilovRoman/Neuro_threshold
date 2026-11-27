"""Display-only conversion: Qt belongs to the UI layer."""
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap


def numpy_to_qpixmap(img_bgr, max_side=8192, max_pixels=16_000_000):
    """Display-only conversion. Geometry stays in source-image coordinates."""
    from PyQt5.QtGui import QImage
    if img_bgr is None:
        return QPixmap()
    img = np.asarray(img_bgr)
    if img.ndim not in (2, 3) or min(img.shape[:2]) < 1:
        raise ValueError(f"Unsupported display shape: {img.shape}")
    if max_side < 1 or max_pixels < 1:
        raise ValueError("Display limits must be positive")
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w), (max_pixels / (h * w)) ** 0.5)
    if scale < 1:
        img = cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        from utils_new.images.processing import ImageStats
        img = ImageStats.to_uint8(img)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        fmt = QImage.Format_Grayscale8
        stride = w
    elif img.ndim == 3 and img.shape[2] == 3:
        fmt = QImage.Format_BGR888
        stride = 3*w
    elif img.ndim == 3 and img.shape[2] == 4:
        fmt = QImage.Format_ARGB32
        stride = 4*w
    else:
        raise ValueError(f'Unsupported display shape: {img.shape}')
    # Deep-copy QImage: numpy data must not be freed while Qt holds the pixels.
    qimage = QImage(img.data, w, h, stride, fmt).copy()
    if qimage.isNull():
        import logging
        logging.getLogger(__name__).error("Недостаточно памяти для экранной копии снимка.")
        return QPixmap()
    pixmap = QPixmap.fromImage(qimage)
    if pixmap.isNull():
        import logging
        logging.getLogger(__name__).error("Недостаточно памяти для отображения снимка.")
        return QPixmap()
    return pixmap
