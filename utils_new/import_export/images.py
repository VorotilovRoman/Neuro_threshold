# -*- coding: utf-8 -*-
"""Filesystem image I/O, Unicode-safe and independent of Qt."""

from pathlib import Path
from functools import wraps
import threading

import cv2
import numpy as np

from utils_new.annotations.masks import normalize_index_mask_shape


# Header inspection is cheap even for very large local radiographs. Keep a
# finite project limit and restore Pillow's process setting immediately.
MAX_IMAGE_PIXELS = 512_000_000
MAX_DECODE_BYTES = 1024 * 1024 * 1024
_pillow_lock = threading.RLock()
_decode_lock = threading.RLock()


def open_pillow_image(path):
    from PIL import Image

    with _pillow_lock:
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
            image = Image.open(path)
        finally:
            Image.MAX_IMAGE_PIXELS = previous
    if image.width * image.height > MAX_IMAGE_PIXELS:
        image.close()
        raise OSError(f"Снимок превышает лимит {MAX_IMAGE_PIXELS:,} пикселей: {path}")
    return image


def _check_decode_size(path, color=False):
    try:
        with open_pillow_image(path) as image:
            channels = 3 if color else len(image.getbands())
            itemsize = 1 if color else (4 if image.mode in ('I', 'F') else 2 if '16' in image.mode else 1)
            required = image.width * image.height * channels * itemsize
    except (ImportError, SyntaxError, ValueError):
        return  # Let OpenCV try formats not understood by Pillow.
    except OSError as error:
        if str(error).startswith('Снимок превышает'):
            raise
        return
    if required > MAX_DECODE_BYTES:
        raise OSError(f"Для декодирования снимка нужно около {required / 2**20:.0f} МиБ; "
                      f"лимит одного снимка — {MAX_DECODE_BYTES / 2**20:.0f} МиБ. "
                      f"Уменьшите снимок или разделите его на тайлы: {path}")


def _serialized_decode(function):
    @wraps(function)
    def read(path, *args, **kwargs):
        # Different tabs/caches must not decode several huge sources at once.
        with _decode_lock:
            _check_decode_size(path, color=function.__name__ == 'read_image')
            return function(path, *args, **kwargs)
    return read


def _read_with_fallback(path, flags):
    """Read image/mask with Unicode-path fallback (notably Windows Cyrillic)."""
    path = str(path)
    try:
        # OpenCV filename APIs on Windows cannot reliably open Unicode paths.
        import os
        image = None if os.name == "nt" and not path.isascii() else cv2.imread(path, flags)
    except cv2.error:
        image = None
    if image is not None:
        return image
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw, flags) if raw.size else None
    except (OSError, ValueError, cv2.error):
        return None


@_serialized_decode
def read_image(path):
    """Read BGR image, keeping existing generator's 3-channel convention."""
    image = _read_with_fallback(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read image: {path}")
    return image


@_serialized_decode
def read_index_mask(path, *, ignore_index=None):
    """Read indexed PNG as 2D integer array; accept decoder's (H,W,1)."""
    # Palette PNG stores class IDs; decoding its RGB palette would destroy them.
    try:
        from PIL import Image

        with open_pillow_image(path) as source:
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


@_serialized_decode
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

            with open_pillow_image(path) as source:
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

        with open_pillow_image(path) as source:
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

        with open_pillow_image(path) as image:
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


def index_sidecars(folder, extensions=('.txt',), *, recursive=True, cancel=None):
    """Index whole relative stems once; dots/spaces are preserved, case is ignored.

    Keep multiple candidates so callers can report ambiguous sources explicitly.
    No image pixels or annotation contents are read.
    """
    import unicodedata
    folder = Path(folder)
    result = {}
    if not folder.is_dir():
        return result
    paths = folder.rglob('*') if recursive else folder.iterdir()
    for path in paths:
        if cancel is not None and cancel.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError()
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        stem = path.relative_to(folder).with_suffix('').as_posix()
        key = unicodedata.normalize('NFC', stem).casefold()
        result.setdefault(key, []).append(path)
    return result


def sidecar_matches(folder, relative, cache, *, recursive=True, basename_fallback=False):
    import unicodedata
    folder = Path(folder)
    key = (str(folder), recursive)
    if key not in cache:
        cache[key] = index_sidecars(folder, recursive=recursive)
    stem = Path(relative).with_suffix('').as_posix()
    normalized = unicodedata.normalize('NFC', stem).casefold()
    matches = cache[key].get(normalized, [])
    if matches or not basename_fallback:
        return matches
    # Build this second index once. Prefer relative paths, then accept only an
    # unambiguous basename across nested label folders (the caller checks ties).
    basename_key = ('basenames', *key)
    if basename_key not in cache:
        names = {}
        for name, paths in cache[key].items():
            names.setdefault(name.rsplit('/', 1)[-1], []).extend(paths)
        cache[basename_key] = names
    return cache[basename_key].get(normalized.rsplit('/', 1)[-1], [])
