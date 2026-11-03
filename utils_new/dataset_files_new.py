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



def normalize_index_mask_shape(mask, *, path=None):
    """Normalize equivalent OpenCV single-channel shapes to (H, W).

    cv2 normally decodes gray PNG to (H, W), but certain decoders/builds
    return (H, W, 1). Only a trailing singleton channel is removed: actual
    RGB/RGBA masks are *not* silently converted to labels. Pixel values and
    dtype (including uint16 IDs) are preserved exactly.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        suffix = f': {path}' if path is not None else ''
        raise ValueError(
            f'Expected single-channel integer class mask{suffix} '
            f'(shape={mask.shape}, dtype={mask.dtype})'
        )
    return mask


def read_index_mask(path):
    """Read indexed PNG as 2D integer array; accept decoder's (H,W,1)."""
    mask = _read_with_fallback(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise OSError(f'Cannot read mask: {path}')
    return normalize_index_mask_shape(mask, path=path)



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



# Read/display compatibility required by the viewer and the annotation editor.
# Unlike read_image(), these functions do not discard source bit depth.
HARD_MAX_SIDE = 16384

def read_image_with_fallback(path):
    """Read image with original depth, TIFF and Unicode-path support.

    Viewer API returns None on failed reads; the generation API read_image()
    intentionally raises and returns three-channel BGR on success.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in ('.tif', '.tiff'):
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
            raise OSError(f'Cannot read image dimensions: {path}')
        return int(image.shape[1]), int(image.shape[0])


def resize_to_max_side(img, max_side=1024):
    if max_side is None or max(img.shape[:2]) <= max_side:
        return img
    h, w = img.shape[:2]
    factor = float(max_side) / max(w, h)
    return cv2.resize(img, (max(1, round(w*factor)), max(1, round(h*factor))),
                      interpolation=cv2.INTER_AREA)


def normalize_to_uint8(img):
    """Use the already-shared ImageStats conversion, not a second algorithm."""
    from utils_new.dataset_image_stats_new import ImageStats
    return ImageStats.to_uint8(img)


def convert_to_grayscale(img):
    from utils_new.dataset_image_stats_new import ImageStats
    return ImageStats.to_gray(img)


def load_one_image_item(image_path, resize_enabled=True, max_side=1024,
                        safe_max_side=None, load_annotations_flag=True,
                        format_hint='auto'):
    """Qt-free lazy cache record used by both dataset tabs."""
    from utils_new.dataset_yolo_labels_new import load_annotations
    image = read_image_with_fallback(image_path)
    if image is None:
        return None
    image = normalize_to_uint8(image)
    if max(image.shape[:2]) > HARD_MAX_SIDE:
        image = resize_to_max_side(image, HARD_MAX_SIDE)
    if resize_enabled:
        image = resize_to_max_side(image, max_side)
    if safe_max_side is not None:
        image = resize_to_max_side(image, safe_max_side)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] != 3:
        image = image[:, :, :3]
    annotations = []
    if load_annotations_flag:
        label_path = Path(image_path).with_suffix('.txt')
        if label_path.exists():
            width, height = image_size_from_file(image_path)
            annotations = load_annotations(label_path, width, height,
                                           format_hint=format_hint)
    return {'image': image, 'gray': convert_to_grayscale(image),
            'annotations': annotations, 'path': str(image_path)}


from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading

class LazyImageCache:
    def __init__(self, paths, loader, window=8, prefetch_ahead=4,
                 max_resident=16, max_workers=2):
        self.paths = list(paths)
        self.loader = loader
        self.window = window
        self.prefetch_ahead = prefetch_ahead
        self.max_resident = max_resident
        self._cache = OrderedDict()
        self._errors = {}
        self._lock = threading.RLock()
        self._in_flight = set()
        self._events = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="img-load")
        self._shutdown = False

    def __len__(self):
        return len(self.paths)

    def _trim_locked(self):
        while len(self._cache) > self.max_resident:
            self._cache.popitem(last=False)

    def _load_sync(self, idx):
        try:
            item = self.loader(self.paths[idx])
        except Exception as e:
            with self._lock:
                self._errors[idx] = e
            raise
        with self._lock:
            if item is not None:
                self._cache[idx] = item
                self._cache.move_to_end(idx)
                self._trim_locked()
                self._errors.pop(idx, None)
            else:
                self._errors[idx] = IOError(f"Loader вернул None для {self.paths[idx]}")
        return item

    def is_loaded(self, idx):
        with self._lock:
            return idx in self._cache

    def get_error(self, idx):
        with self._lock:
            return self._errors.get(idx)

    def get(self, idx):
        if idx < 0 or idx >= len(self.paths):
            return None
        with self._lock:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                return self._cache[idx]
            ev = self._events.get(idx)
        if ev is not None:
            ev.wait()
            with self._lock:
                if idx in self._cache:
                    self._cache.move_to_end(idx)
                    return self._cache[idx]
                if idx in self._errors:
                    return None
        return self._load_sync(idx)

    def _load_bg(self, idx):
        if self._shutdown:
            return
        item = None
        error = None
        try:
            item = self.loader(self.paths[idx])
            if item is None:
                error = IOError(f"Loader вернул None для {self.paths[idx]}")
        except Exception as e:
            error = e
            print(f"[lazy] background error loading {self.paths[idx]}: {e}")
        if self._shutdown:
            return
        with self._lock:
            self._in_flight.discard(idx)
            if item is not None:
                self._cache[idx] = item
                self._cache.move_to_end(idx)
                self._trim_locked()
                self._errors.pop(idx, None)
            elif error is not None:
                self._errors[idx] = error
            ev = self._events.pop(idx, None)
        if ev is not None:
            ev.set()

    def prefetch(self, indices):
        if self._shutdown:
            return
        for idx in indices:
            if idx < 0 or idx >= len(self.paths):
                continue
            with self._lock:
                if idx in self._cache or idx in self._in_flight:
                    continue
                if idx in self._errors:
                    continue
                self._in_flight.add(idx)
                self._events[idx] = threading.Event()
            self._pool.submit(self._load_bg, idx)

    def trim_around(self, center, keep_behind=3, keep_ahead=None):
        if keep_ahead is None:
            keep_ahead = self.prefetch_ahead + 4
        with self._lock:
            lo = max(0, center - keep_behind)
            hi = center + keep_ahead
            to_remove = [i for i in self._cache.keys() if not (lo <= i <= hi)]
            for i in to_remove:
                del self._cache[i]
            err_remove = [i for i in self._errors.keys() if not (lo <= i <= hi)]
            for i in err_remove:
                del self._errors[i]

    def shutdown(self):
        self._shutdown = True
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._pool.shutdown(wait=False)
        with self._lock:
            self._cache.clear()
            self._errors.clear()
            self._in_flight.clear()
            for ev in self._events.values():
                ev.set()
            self._events.clear()

