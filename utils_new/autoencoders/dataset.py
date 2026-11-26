"""Clean targets, derived edge channels and paired geometry for AE training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils_new.autoencoders.preprocessing import append_edge, preprocess_image_for_ae
from utils_new.import_export.atomic import atomic_path
from utils_new.import_export.images import collect_image_paths, read_image_with_fallback


def collect_training_images(root, *, exclude=()):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Не найдена папка снимков: {root}")
    excluded = [Path(p).resolve() for p in exclude]
    files = [
        str(Path(p).resolve())
        for p in collect_image_paths(root, recursive=True)
        if not any(Path(p).resolve().is_relative_to(e) for e in excluded)
        and Path(p).name.lower() not in ("loss.png", "recon_examples.png")
    ]
    if len(files) < 2:
        raise ValueError("Нужны минимум два разных снимка для раздельных train и val.")
    return files


def dataset_manifest(paths, cancel=None):
    """Content identity detects changed data even when size/mtime were preserved."""
    result = []
    for path in paths:
        _check_cancel(cancel)
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                _check_cancel(cancel)
                digest.update(block)
        result.append(dict(path=str(Path(path).resolve()), sha256=digest.hexdigest()))
    return result


def split_images(paths, fraction, seed, *, records=None):
    """Identical files remain in one split, preventing exact duplicate leakage."""
    groups = {}
    for item in records if records is not None else dataset_manifest(paths):
        groups.setdefault(item["sha256"], []).append(item["path"])
    if len(groups) < 2:
        raise ValueError("Для независимого val нужны хотя бы два различающихся снимка.")
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
    val = [p for key in keys[:count] for p in groups[key]]
    train = [p for key in keys[count:] for p in groups[key]]
    return train, val


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from concurrent.futures import CancelledError

        raise CancelledError()


def training_sources(source, fraction, seed, *, exclude=(), cancel=None):
    from utils_new.import_export.yolo_yaml import load_image_splits

    data = load_image_splits(source, exclude=exclude, cancel=cancel)
    splits = data['splits']
    records = {name: dataset_manifest(paths, cancel) for name, paths in splits.items()}
    if data['auto_split']:
        splits['train'], splits['val'] = split_images(splits['train'], fraction, seed, records=records['train'])
        lookup = {item['path']: item for item in records['train']}
        for name in ('train', 'val'):
            records[name] = [lookup[path] for path in splits[name]]
    seen = {}
    for name, values in records.items():
        for item in values:
            previous = seen.get(item['sha256'])
            if previous and previous[0] != name:
                raise ValueError(f'Одинаковое содержимое снимков в {previous[0]}/{name}: {previous[1]} / {item["path"]}')
            seen[item['sha256']] = (name, item['path'])
    return dict(data, manifest=records)


class AutoencoderDataset(Dataset):
    def __init__(self, paths, config, *, training=False, edge_cache=None):
        self.paths = list(paths)
        self.config = config
        self.model = config.model
        self.training = training
        self.edge_cache = Path(edge_cache) if edge_cache else None
        self.jitter = transforms.ColorJitter(
            brightness=config.brightness, contrast=config.contrast
        )

    def __len__(self):
        return len(self.paths)

    def _cache_edge(self, image, path):
        """Only deterministic inputs are cached; cached and fresh values are float32."""
        if self.edge_cache is None or not self.model["use_edge_channel"]:
            return append_edge(image, self.model)
        metadata = {
            k: self.model[k]
            for k in (
                "preprocessing_version",
                "edge_kernel",
                "median_kernel",
                "edge_lo",
                "edge_hi",
            )
        }
        digest = hashlib.sha256(str(Path(path).resolve()).encode())
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        digest.update(image.numpy().tobytes())
        target = self.edge_cache / (digest.hexdigest() + ".npy")
        try:
            edge = np.load(target, allow_pickle=False)
            if (
                edge.dtype == np.float32
                and edge.shape == (1, *image.shape[-2:])
                and np.isfinite(edge).all()
            ):
                return torch.cat((image, torch.from_numpy(edge)), dim=0)
        except (OSError, ValueError, EOFError):
            pass
        combined = append_edge(image, self.model)
        with atomic_path(target) as temporary:
            with temporary.open("wb") as stream:
                np.save(stream, combined[-1:].numpy(), allow_pickle=False)
        return combined

    def __getitem__(self, index):
        path = self.paths[index]
        image = read_image_with_fallback(path)
        if image is None:
            raise ValueError(f"Не удалось прочитать снимок: {path}")
        # Derive edges after geometry, photometry and noise, exactly from the input.
        options = dict(
            self.model,
            use_edge_channel=False,
            model_in_channels=self.model["in_channels"],
        )
        clean, region = preprocess_image_for_ae(image, options)
        valid = torch.zeros((1, *clean.shape[-2:]), dtype=torch.float32)
        x0, y0, x1, y1 = region
        valid[:, y0:y1, x0:x1] = 1
        cfg = self.config
        augmented = self.training and cfg.augmentation
        if augmented:
            parameters = transforms.RandomAffine.get_params(
                (-cfg.degrees, cfg.degrees),
                (cfg.translate, cfg.translate),
                None,
                None,
                img_size=[clean.shape[-1], clean.shape[-2]],
            )
            clean = TF.affine(
                clean,
                *parameters,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=cfg.model["pad_value"] / 255.0,
            )
            valid = TF.affine(
                valid, *parameters, interpolation=TF.InterpolationMode.NEAREST, fill=0.0
            )
            for enabled, axis in ((cfg.hflip, -1), (cfg.vflip, -2)):
                if enabled and torch.rand(()).item() < 0.5:
                    clean = clean.flip((axis,))
                    valid = valid.flip((axis,))
            clean = self.jitter(clean)
        target = clean.clone()
        noisy = self.training and cfg.noise_std > 0
        if noisy:
            clean = (clean + torch.randn_like(clean) * cfg.noise_std).clamp(0, 1)
        inputs = (
            append_edge(clean, self.model)
            if (augmented or noisy)
            else self._cache_edge(clean, path)
        )
        if not valid.any():
            raise ValueError(f"Аугментация удалила всю полезную область: {path}")
        return dict(input=inputs, target=target, valid=valid)
