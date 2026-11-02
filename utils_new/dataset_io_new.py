# -*- coding: utf-8 -*-
"""Filesystem formats for DatasetGenerator_new, independent of legacy utils.

Uses the canonical pixel-coordinate Annotation API. Input YOLO tuples are
converted only at file boundaries; internal processing never manipulates
normalized coordinates. COCO/U-Net exporters receive canonical geometry.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils_new.annotations import (Annotation, ImageSize, MaskEncoding,
                                   from_legacy, to_legacy, bounds, polygon_area,
                                   rasterize_annotations, mask_to_annotations)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
LABEL_HINTS = {'auto', 'mixed', 'detect', 'obb_corners', 'obb_xywhr', 'segment'}


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


def _xywhr_points(vals):
    cx, cy, w, h, radians = map(float, vals)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    v1 = np.array((w*cos_a/2, w*sin_a/2), dtype=float)
    v2 = np.array((-h*sin_a/2, h*cos_a/2), dtype=float)
    c = np.array((cx, cy), dtype=float)
    return np.array([c+v1+v2, c+v1-v2, c-v1-v2, c-v1+v2]).flatten().tolist()


def read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None):
    """Read old YOLO txt variants into canonical pixel-space annotations.

    9 tokens are OBB in auto mode. Explicit 'segment' disambiguates a 4-point
    polygon. Invalid rows are logged and skipped as in legacy image_io.
    """
    if format_hint not in LABEL_HINTS:
        raise ValueError(f'Unsupported YOLO input format {format_hint!r}')
    if path is None or not Path(path).exists():
        return []
    result = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            parts = line.split()
            cls = int(parts[0])
            numbers = list(map(float, parts[1:]))
            count = len(parts)
            if format_hint == 'detect':
                kind = 'detect' if count == 5 else None
            elif format_hint == 'obb_corners':
                kind = 'obb' if count == 9 else None
            elif format_hint == 'obb_xywhr':
                kind = 'obb_xywhr' if count == 6 else None
            elif format_hint == 'segment':
                kind = 'segment' if count >= 7 and (count-1)%2 == 0 else None
            else:
                kind = ('detect' if count == 5 else 'obb_xywhr' if count == 6
                        else 'obb' if count == 9 else 'segment' if count >= 7 and (count-1)%2 == 0
                        else None)
            if kind is None:
                raise ValueError(f'{count} tokens incompatible with format_hint={format_hint}')
            if kind == 'obb_xywhr':
                kind, numbers = 'obb', _xywhr_points(numbers)
            value = ((kind, cls, *numbers) if kind == 'detect' else (kind, cls, numbers))
            result.append(from_legacy(value, size))
        except (TypeError, ValueError, IndexError) as exc:
            if log:
                log(f'[labels] {path}:{line_no}: {exc}; запись пропущена')
    return result


def _normalized_xywhr(ann, size):
    """Legacy-compatible xywhr: angle in radians of normalized corner edges."""
    normalized = np.asarray(to_legacy(ann, size)[2], dtype=np.float64).reshape(4, 2)
    center = normalized.mean(axis=0)
    edges = np.roll(normalized, -1, axis=0) - normalized
    lengths = np.linalg.norm(edges, axis=1)
    idx = int(np.argmax(lengths))
    width = float(lengths[idx])
    angle = math.atan2(float(edges[idx, 1]), float(edges[idx, 0]))
    height = (float(lengths[(idx + 1) % 4]) + float(lengths[(idx + 3) % 4])) / 2
    if height > width:
        width, height = height, width
        angle += math.pi / 2
    angle = angle % math.pi
    if angle >= 3*math.pi/4:
        angle -= math.pi
    return [float(center[0]), float(center[1]), width, height, angle]


def write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners'):
    if obb_format not in ('corners', 'xywhr'):
        raise ValueError('obb_format must be corners/xywhr')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ann in annotations:
        if not isinstance(ann, Annotation):
            raise TypeError('Expected canonical Annotation objects')
        legacy = to_legacy(ann, size)
        kind = legacy[0]
        if kind == 'detect':
            coords = legacy[2:]
        elif kind == 'obb' and obb_format == 'xywhr':
            coords = _normalized_xywhr(ann, size)
        else:
            coords = legacy[2]
        # Four-point Segment has 9 tokens (class+8), ambiguous with OBB.
        # Insert a collinear midpoint to preserve the exact polygon shape and
        # make AUTO / mixed readers identify Segment unambiguously.
        if kind == 'segment' and len(coords) == 8:
            coords = list(coords)
            midpoint = [(float(coords[0])+float(coords[2]))/2,
                        (float(coords[1])+float(coords[3]))/2]
            coords = coords[:2] + midpoint + coords[2:]
        rows.append(f'{ann.class_id} ' + ' '.join(f'{float(v):.6f}' for v in coords))
    path.write_text(('\n'.join(rows) + '\n') if rows else '', encoding='utf-8')


def write_index_mask(path, mask):
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError('Expected a 2D indexed class mask')
    if np.min(mask) < 0 or np.max(mask) > 65535:
        raise ValueError('PNG class indices must fit uint16')
    if mask.dtype not in (np.uint8, np.uint16):
        mask = mask.astype(np.uint8 if int(mask.max()) <= 255 else np.uint16)
    write_image(path, mask)


def make_index_mask(annotations, size, *, offset=1):
    if int(offset) < 1:
        raise ValueError('Output mask offset must be >=1 (class 0 cannot equal background)')
    return rasterize_annotations(list(annotations), size,
                                 encoding=MaskEncoding(offset=int(offset), background_value=0))


def write_classes_and_yaml(output_dir, class_names, *, out_yolo_txt=True,
                           out_unet_yaml=False, out_masks_offset=1, input_type=0, dataset_path=None):
    output_dir = Path(output_dir)
    if input_type == 2 or not class_names:
        return
    ordered = [str(class_names[k]) for k in sorted(class_names)]
    (output_dir/'classes.txt').write_text('\n'.join(ordered)+'\n', encoding='utf-8')
    if out_yolo_txt:
        data = {'path': str(Path(dataset_path or output_dir).absolute()).replace('\\', '/'),
                'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
                'nc': len(ordered), 'names': ordered}
        (output_dir/'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
    if out_unet_yaml:
        data = {'path': str(Path(dataset_path or output_dir).absolute()).replace('\\', '/'),
                'train_images': 'images/train', 'train_masks': 'masks/train',
                'val_images': 'images/val', 'val_masks': 'masks/val',
                'test_images': 'images/test', 'test_masks': 'masks/test',
                'num_classes': len(ordered)+1,
                'class_names': ['background', *ordered],
                'mask_class_offset': int(out_masks_offset)}
        (output_dir/'masks_data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')


def _coco_record(ann, image_id, ann_id):
    x0, y0, x1, y1 = bounds(ann.points_px)
    return {'id': ann_id, 'image_id': image_id,
            'category_id': int(ann.class_id)+1,
            'bbox': [float(x0), float(y0), float(x1-x0), float(y1-y0)],
            'area': float(polygon_area(ann.points_px)),
            'iscrowd': 0,
            'segmentation': [] if ann.kind == 'detect' else
                [ann.points_px.reshape(-1).astype(float).tolist()]}


def write_coco_split(output_dir, split, rows, class_names, *, masks_offset=1,
                     log=None):
    """rows: [(image_name, image_size, annotations, optional_mask), ...].

    For mask source, annotations may be None: conversion via contour extraction
    is explicitly lossy (holes dropped) to match old COCO semantics.
    """
    output_dir = Path(output_dir)
    images = []
    objects = []
    counter = 0
    for image_id, (name, size, annotations, mask) in enumerate(rows):
        images.append({'id': image_id, 'file_name': f'images/{split}/{name}',
                       'width': size.width, 'height': size.height, 'license': 1})
        if annotations is None:
            if mask is None:
                annotations = []
            else:
                annotations = mask_to_annotations(mask, encoding=MaskEncoding(offset=int(masks_offset)),
                                                  kind='segment', min_area=1.0, holes='drop')
                if log:
                    log(f'[COCO] {name}: mask→polygon — lossy (контуры, без отверстий)')
        for ann in annotations:
            objects.append(_coco_record(ann, image_id, counter))
            counter += 1
    data = {'info': {'description': 'Threshold-Researcher generated dataset'},
            'licenses': [{'id': 1, 'name': ''}],
            'images': images,
            'annotations': objects,
            'categories': [{'id': int(cid)+1, 'name': str(name), 'supercategory': 'none'}
                           for cid, name in sorted(class_names.items())]}
    out = output_dir/f'annotations_{split}.json'
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return counter
