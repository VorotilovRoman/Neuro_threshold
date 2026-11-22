# -*- coding: utf-8 -*-
"""Filesystem image I/O, Unicode-safe and independent of Qt."""

from pathlib import Path

import cv2
import numpy as np

from utils_new.annotations.masks import normalize_index_mask_shape


def _read_with_fallback(path, flags):
    """Read image/mask with Unicode-path fallback (notably Windows Cyrillic)."""
    path = str(path)
    try:
        image = cv2.imread(path, flags)
    except cv2.error:
        image = None
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
        raise OSError(f"Cannot read image: {path}")
    return image


def read_index_mask(path, *, ignore_index=None):
    """Read indexed PNG as 2D integer array; accept decoder's (H,W,1)."""
    # Palette PNG stores class IDs; decoding its RGB palette would destroy them.
    try:
        from PIL import Image

        with Image.open(path) as source:
            if source.mode in ("P", "I", "I;16", "1"):
                value = np.array(source)
                if value.dtype == np.bool_:
                    value = value.astype(np.uint8)
                return normalize_index_mask_shape(
                    value, path=path, ignore_index=ignore_index
                )
    except (OSError, ImportError):
        pass
    mask = _read_with_fallback(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f"Cannot read mask: {path}")
    return normalize_index_mask_shape(mask, path=path, ignore_index=ignore_index)


def write_image(path, image):
    """Encode before replacing a file; Unicode paths and failed writes are safe."""
    from utils_new.import_export.atomic import write_bytes_atomic

    path = Path(path)
    try:
        success, payload = cv2.imencode(path.suffix.lower() or ".png", image)
        if success:
            write_bytes_atomic(path, payload.tobytes())
            return
    except (OSError, ValueError, cv2.error) as exc:
        raise OSError(f"Failed writing image: {path}") from exc
    raise OSError(f"Failed writing image: {path}")


# Read/display compatibility required by the viewer and the annotation editor.
# Unlike read_image(), these functions do not discard source bit depth.


def read_image_with_fallback(path):
    """Read image with original depth, TIFF and Unicode-path support.

    Viewer API returns None on failed reads; the generation API read_image()
    intentionally raises and returns three-channel BGR on success.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            from PIL import Image

            with Image.open(path) as source:
                arr = np.asarray(source)
            if arr.ndim == 2:
                return arr
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except (ImportError, OSError, ValueError, cv2.error):
            pass
    result = _read_with_fallback(path, cv2.IMREAD_UNCHANGED)
    if result is not None:
        return result
    try:
        from PIL import Image

        with Image.open(path) as source:
            arr = np.asarray(source)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except (ImportError, OSError, ValueError, cv2.error):
        return None


def read_mask_with_fallback(path):
    """Nullable indexed-mask reader; never squeeze a real RGB mask."""
    try:
        return read_index_mask(path)
    except (OSError, ValueError, TypeError):
        return None


def image_size_from_file(path):
    """Full-resolution (width,height), without decoding image pixels if possible."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except (ImportError, OSError, ValueError):
        image = read_image_with_fallback(path)
        if image is None:
            raise OSError(f"Cannot read image dimensions: {path}")
        return int(image.shape[1]), int(image.shape[0])


IMAGE_EXTENSIONS = frozenset(
    (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
)


def collect_image_paths(source, *, recursive=False):
    """Ordered, deduplicated image inputs; recursion is explicit, never implicit."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            entries = path.rglob("*") if recursive else path.iterdir()
            values = sorted(
                (
                    str(p)
                    for p in entries
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=str.casefold,
            )
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            values = [str(path)]
        else:
            return []
    elif isinstance(source, (tuple, list)):
        values = [str(p) for p in source if Path(p).suffix.lower() in IMAGE_EXTENSIONS]
    else:
        return []
    # Preserve user-selected order and the original spelling of the first path.
    import os

    unique = {}
    for path in values:
        unique.setdefault(os.path.normcase(os.path.abspath(path)), path)
    return list(unique.values())


def index_mask_files(folder):
    """Index lossless mask paths by relative stem; reject ambiguous sidecars."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Нет папки масок: {folder}")
    result = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".png", ".tif", ".tiff", ".bmp"):
            key = path.relative_to(folder).with_suffix("").as_posix().casefold()
            if key in result:
                raise ValueError(f"Неоднозначная маска: {path}")
            result[key] = path
    return result
