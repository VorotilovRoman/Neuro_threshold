"""Dataset duplicate, split and spatial checks; optional packages are lazy."""

import hashlib
import os
from collections import Counter, defaultdict
from importlib.util import find_spec
import cv2
import numpy as np
from PIL import Image
from .objects import read_objects
from .export import infer_split

IMAGEHASH_AVAILABLE = find_spec("imagehash") is not None
CLIP_AVAILABLE = find_spec("torch") is not None and (
    find_spec("open_clip") is not None or find_spec("transformers") is not None
)
_CLIP_BACKEND = (
    "open_clip"
    if find_spec("open_clip") is not None
    else "transformers"
    if CLIP_AVAILABLE
    else None
)
UMAP_AVAILABLE = find_spec("umap") is not None
HDBSCAN_AVAILABLE = find_spec("hdbscan") is not None


def _safe_progress(progress_cb, i, total, step=25):
    if progress_cb and ((i + 1) % step == 0 or i == total - 1):
        progress_cb(i + 1, total)


def _check_cancel(cancel_check):
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def detect_device():
    """Автоопределение устройства: 'cuda:0', если доступно, иначе 'cpu'."""
    from runtime_torch import get_torch_runtime

    return get_torch_runtime().device


class DuplicateDetector:
    @staticmethod
    def _file_md5(path, chunk_size=1 << 20):
        h = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _phash(path, hash_size=8):
        if not IMAGEHASH_AVAILABLE:
            return None
        try:
            with Image.open(path) as img:
                import imagehash

                return imagehash.phash(img.convert("RGB"), hash_size=hash_size)
        except Exception:
            return None

    @staticmethod
    def find_duplicates(pairs, phash_threshold=8, progress_cb=None, cancel_check=None):
        unique = {}
        for pair in pairs:
            unique.setdefault(os.path.normcase(os.path.realpath(pair[0])), str(pair[0]))
        paths = list(unique.values())
        total = len(paths)
        md5_map = defaultdict(list)
        hashes = {}
        file_hash = {}
        errors = []
        cancelled = False
        for i, path in enumerate(paths):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            digest = DuplicateDetector._file_md5(path)
            if digest is None:
                errors.append(path)
            else:
                md5_map[digest].append(path)
                file_hash[path] = digest
            value = DuplicateDetector._phash(path)
            if value is not None:
                hashes[path] = value
            _safe_progress(progress_cb, i, total)
        exact = [group for group in md5_map.values() if len(group) > 1]
        parent = {path: path for path in hashes}

        def root(path):
            while parent[path] != path:
                parent[path] = parent[parent[path]]
                path = parent[path]
            return path

        keys = list(hashes)
        for i, first in enumerate(keys):
            if cancelled or _check_cancel(cancel_check):
                cancelled = True
                break
            for j, second in enumerate(keys[i + 1 :]):
                if j % 256 == 0 and _check_cancel(cancel_check):
                    cancelled = True
                    break
                if hashes[first] - hashes[second] <= phash_threshold:
                    a, b = root(first), root(second)
                    if a != b:
                        parent[b] = a
        groups = defaultdict(list)
        for path in keys:
            groups[root(path)].append(path)
        near = [
            group
            for group in groups.values()
            if len(group) > 1 and len({file_hash.get(path, path) for path in group}) > 1
        ]
        return dict(exact=exact, near=near, cancelled=cancelled, errors=errors)


class LeakageChecker:
    SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}

    _infer_split = staticmethod(infer_split)

    @staticmethod
    def check(duplicate_groups, pairs, split_by_image=None):
        memberships = {
            str(pair[0]): set(
                (split_by_image or {}).get(str(pair[0]), [infer_split(pair[0])])
            )
            for pair in pairs
        }
        groups = [list(dict.fromkeys(map(str, group))) for group in duplicate_groups]
        covered = {path for group in groups for path in group}
        groups += [
            [path]
            for path, splits in memberships.items()
            if len(splits) > 1 and path not in covered
        ]
        rows = []
        for group in groups:
            splits = Counter(
                split for path in group for split in memberships.get(path, {"unknown"})
            )
            rows.append(
                dict(
                    paths=group,
                    splits=dict(splits),
                    leak=len(set(splits) - {"unknown"}) > 1,
                )
            )
        return dict(groups=rows, leaked_count=sum(row["leak"] for row in rows))


class ObjectDistributionHeatmap:
    @staticmethod
    def compute(
        pairs,
        dataset_type=0,
        grid_size=16,
        progress_cb=None,
        cancel_check=None,
        input_format="auto",
    ):
        grid_size = max(2, min(128, int(grid_size)))
        heatmap = np.zeros((grid_size, grid_size), np.float32)
        total_objects = 0
        errors = []
        cancelled = False
        for index, pair in enumerate(pairs):
            if _check_cancel(cancel_check):
                cancelled = True
                break
            try:
                _, objects = read_objects(pair, dataset_type, input_format)
                for obj in objects:
                    x, y = obj.center
                    gx = min(grid_size - 1, max(0, int(x * grid_size)))
                    gy = min(grid_size - 1, max(0, int(y * grid_size)))
                    heatmap[gy, gx] += 1
                    total_objects += 1
            except (OSError, ValueError, TypeError, cv2.error) as exc:
                errors.append(f"{pair[0]}: {exc}")
            finally:
                _safe_progress(progress_cb, index, len(pairs))
        return dict(
            heatmap=heatmap,
            total_objects=total_objects,
            grid_size=grid_size,
            cancelled=cancelled,
            errors=errors,
        )
