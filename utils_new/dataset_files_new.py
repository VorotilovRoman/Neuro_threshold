# -*- coding: utf-8 -*-
"""Filesystem image I/O, Unicode-safe and independent of Qt."""
from pathlib import Path
import cv2
import numpy as np

def _read_with_fallback(path, flags):
    """Read image/mask with Unicode-path fallback (notably Windows Cyrillic)."""
    path = str(path)
    image = cv2.imread(path, flags)
    if image is not None:
        return image
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw, flags) if raw.size else None
    except (OSError, ValueError, cv2.error):
        return None



def read_image(path):
    """Read BGR image, keeping existing generator's 3-channel convention."""
    image = _read_with_fallback(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f'Cannot read image: {path}')
    return image



def read_index_mask(path):
    mask = _read_with_fallback(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f'Cannot read mask: {path}')
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f'Expected single-channel integer class mask: {path} ({mask.shape})')
    return mask



def write_image(path, image):
    """Write image safely also when cv2.imwrite cannot handle the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(path), image):
        return
    try:
        success, payload = cv2.imencode(path.suffix or '.png', image)
        if success:
            payload.tofile(str(path))
            return
    except (OSError, ValueError, cv2.error):
        pass
    raise OSError(f'Failed writing image: {path}')

