# -*- coding: utf-8 -*-
"""YOLO/U-Net dataset metadata, indexed masks, and COCO export. Qt-free."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from utils_new.annotations import (
    ImageSize,
    MaskEncoding,
    bounds,
    mask_to_annotations,
    polygon_area,
    rasterize_annotations,
)
from utils_new.dataset_files_new import write_image


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

