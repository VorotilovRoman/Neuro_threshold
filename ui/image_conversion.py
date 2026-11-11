"""Display-only conversion: Qt belongs to the UI layer."""
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap


def numpy_to_qpixmap(img_bgr, max_side=8192):
    """Display-only conversion. Geometry stays in source-image coordinates."""
    from PyQt5.QtGui import QImage
    if img_bgr is None:
        return QPixmap()
    img = np.asarray(img_bgr)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = float(max_side) / max(h, w)
        img = cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        from utils_new.dataset_image_stats_new import ImageStats
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
    return QPixmap.fromImage(qimage)
