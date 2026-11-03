# -*- coding: utf-8 -*-
"""Index-mask inspection and optional COLOR previews, independent of Qt.

Training masks must remain a single-channel integer PNG with pixel values
0 (background), 1 (class 0), 2 (class 1), ... . Small values look black in
ordinary grayscale viewers even when the annotations are present.

Do not train on color previews; they are visualization artifacts only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# OpenCV BGR colors. Visible and deterministic for low class IDs.
_DEFAULT_BGR = (
    (51, 160, 255), (70, 210, 65), (240, 105, 95), (197, 101, 240),
    (30, 210, 220), (225, 182, 45), (255, 87, 185), (90, 215, 170),
    (125, 130, 245), (225, 225, 75),
)


@dataclass(frozen=True)
class MaskStatistics:
    width: int
    height: int
    dtype: str
    counts: dict[int, int]
    background_value: int = 0

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    @property
    def foreground_pixels(self) -> int:
        return self.total_pixels - self.counts.get(self.background_value, 0)

    @property
    def foreground_fraction(self) -> float:
        return self.foreground_pixels / self.total_pixels if self.total_pixels else 0.0

    @property
    def is_empty(self) -> bool:
        return self.foreground_pixels == 0

    def brief(self, max_classes=15) -> str:
        counts = sorted(self.counts.items())
        class_text = ', '.join(f'{value}:{count}' for value, count in counts[:max_classes])
        if len(counts) > max_classes:
            class_text += f', ... ({len(counts)} values)'
        return (f'{self.width}x{self.height}, dtype={self.dtype}, '
                f'nonzero={self.foreground_pixels}/{self.total_pixels} '
                f'({100 * self.foreground_fraction:.3f}%), values={{ {class_text} }}')


def mask_statistics(mask: np.ndarray, *, background_value=0) -> MaskStatistics:
    """Return exact histogram without modifying or reinterpreting label pixels."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError('Index mask must be a two-dimensional numpy array')
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError('Index mask must contain integer class values')
    if mask.size == 0:
        raise ValueError('Mask is empty (zero-size array)')
    # Indexed masks are most often uint8/uint16 with a tiny palette. Avoid
    # np.unique sorting millions of pixels on every large training image.
    if (mask.dtype in (np.dtype('uint8'), np.dtype('uint16'))):
        frequency = np.bincount(mask.reshape(-1))
        values = np.flatnonzero(frequency)
        counts = frequency[values]
    else:
        values, counts = np.unique(mask, return_counts=True)
    return MaskStatistics(
        width=int(mask.shape[1]), height=int(mask.shape[0]),
        dtype=str(mask.dtype),
        counts={int(value): int(count) for value, count in zip(values, counts)},
        background_value=int(background_value))


def colorize_index_mask(mask: np.ndarray, *, pixel_offset=1, background_value=0,
                        color_for_class=None) -> np.ndarray:
    """Return 8-bit 3-channel BGR preview, never alter original numeric mask.

    Optional color_for_class(class_id) must return 3 BGR values (0-255).
    Class IDs are pixel values minus pixel_offset. Pixel 0 stays black.
    """
    info = mask_statistics(mask, background_value=background_value)
    out = np.zeros((info.height, info.width, 3), dtype=np.uint8)
    for value in info.counts:
        if value == background_value:
            continue
        class_id = int(value) - int(pixel_offset)
        if class_id < 0:
            raise ValueError(f'Unexpected pixel {value} below class offset {pixel_offset}')
        color = (color_for_class(class_id) if color_for_class is not None
                 else _DEFAULT_BGR[class_id % len(_DEFAULT_BGR)])
        color = np.asarray(color)
        if color.shape != (3,) or not np.isfinite(color).all() or (color < 0).any() or (color > 255).any():
            raise ValueError(f'Invalid BGR color for class {class_id}: {color!r}')
        out[mask == value] = np.asarray(color, dtype=np.uint8)
    return out


def save_mask_preview(mask_path, output_path, *, pixel_offset=1, background_value=0,
                      color_for_class=None) -> MaskStatistics:
    """Read a REAL saved mask and write a distinct color preview PNG."""
    from utils_new.dataset_files_new import read_index_mask, write_image

    mask = read_index_mask(mask_path)
    stats = mask_statistics(mask, background_value=background_value)
    preview = colorize_index_mask(mask, pixel_offset=pixel_offset,
                                  background_value=background_value,
                                  color_for_class=color_for_class)
    output_path = Path(output_path)
    if Path(mask_path).resolve() == output_path.resolve():
        raise ValueError('Refusing to overwrite the training mask with a color preview')
    write_image(output_path, preview)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description='Inspect indexed class masks (0=background).')
    parser.add_argument('mask', help='Path to PNG mask or a directory of masks')
    parser.add_argument('--preview-dir', type=Path,
                        help='Optional separate directory for colorized previews')
    parser.add_argument('--offset', type=int, default=1,
                        help='Mask class pixel offset (default 1)')
    args = parser.parse_args(argv)
    from utils_new.dataset_files_new import read_index_mask

    source = Path(args.mask)
    paths = sorted(source.rglob('*.png')) if source.is_dir() else [source]
    if not paths:
        parser.error('No PNG masks found')
    if args.preview_dir is not None and source.is_dir():
        try:
            args.preview_dir.resolve().relative_to(source.resolve())
        except ValueError:
            pass
        else:
            parser.error('--preview-dir must NOT be inside the masks source directory')
    for path in paths:
        stats = mask_statistics(read_index_mask(path))
        print(f'{path}: {stats.brief()}')
        if stats.is_empty:
            print('  WARNING: mask truly contains only background pixels')
        if args.preview_dir is not None:
            rel = path.relative_to(source) if source.is_dir() else Path(path.name)
            preview_path = args.preview_dir / rel.parent / (rel.stem + '_preview.png')
            save_mask_preview(path, preview_path, pixel_offset=args.offset)
            print('  preview:', preview_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
