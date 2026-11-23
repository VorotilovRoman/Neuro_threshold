# -*- coding: utf-8 -*-
"""YOLO/U-Net dataset metadata, indexed masks, and COCO export. Qt-free."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import cv2

from utils_new.annotations import (
    MaskEncoding,
    bounds,
    polygon_area,
    rasterize_annotations,
)
from utils_new.import_export.images import write_image


def write_index_mask(path, mask):
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError("Expected a 2D indexed class mask")
    if np.min(mask) < 0 or np.max(mask) > 65535:
        raise ValueError("PNG class indices must fit uint16")
    if mask.dtype not in (np.uint8, np.uint16):
        mask = mask.astype(np.uint8 if int(mask.max()) <= 255 else np.uint16)
    write_image(path, mask)


def make_index_mask(annotations, size, *, offset=1):
    if int(offset) < 1:
        raise ValueError(
            "Output mask offset must be >=1 (class 0 cannot equal background)"
        )
    return rasterize_annotations(
        list(annotations),
        size,
        encoding=MaskEncoding(offset=int(offset), background_value=0),
    )


def write_classes_and_yaml(
    output_dir,
    class_names,
    *,
    out_yolo_txt=True,
    out_unet_yaml=False,
    out_masks_offset=1,
    input_type=0,
    yolo_channels=3,
):
    output_dir = Path(output_dir)
    if type(yolo_channels) is not int or yolo_channels not in (1, 3):
        raise ValueError("YOLO channels must be 1 or 3")
    if input_type == 2:
        data = dict(
            dataset_type="images_only",
            channels=yolo_channels,
            train="images/train",
            val="images/val",
            test="images/test",
        )
        (output_dir / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
        return
    if not class_names:
        return
    if sorted(class_names) != list(range(len(class_names))):
        raise ValueError(
            "Output class IDs must be dense 0..N-1; remap selected classes first"
        )
    if type(out_masks_offset) is not int or not 1 <= out_masks_offset <= 65535:
        raise ValueError("Output mask offset must be an integer 1..65535")
    if out_unet_yaml and len(class_names) + out_masks_offset > 65536:
        raise ValueError("PNG class indices must fit uint16")
    ordered = [str(class_names[k]) for k in sorted(class_names)]
    (output_dir / "classes.txt").write_text("\n".join(ordered) + "\n", encoding="utf-8")
    if out_yolo_txt:
        data = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "channels": yolo_channels,
            "nc": len(ordered),
            "names": ordered,
        }
        (output_dir / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    if out_unet_yaml:
        data = {
            "train_images": "images/train",
            "train_masks": "masks/train",
            "val_images": "images/val",
            "val_masks": "masks/val",
            "test_images": "images/test",
            "test_masks": "masks/test",
            "num_classes": len(ordered) + out_masks_offset,
            "class_names": [
                "background",
                *[f"reserved_{i}" for i in range(1, out_masks_offset)],
                *ordered,
            ],
            "mask_class_offset": int(out_masks_offset),
        }
        (output_dir / "masks_data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )


def _coco_record(ann, image_id, ann_id):
    x0, y0, x1, y1 = bounds(ann.points_px)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(ann.class_id) + 1,
        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
        "area": float(polygon_area(ann.points_px)),
        "iscrowd": 0,
        "segmentation": []
        if ann.kind == "detect"
        else [ann.points_px.reshape(-1).astype(float).tolist()],
    }


def write_coco_split(output_dir, split, rows, class_names, *, masks_offset=1, log=None):
    """rows: [(image_name, image_size, annotations, optional_mask), ...].

    Indexed masks are encoded as RLE components without losing holes or pixels.
    An optional fifth row value distinguishes unlabelled images from background.
    """
    output_dir = Path(output_dir)
    images = []
    objects = []
    counter = 0
    for image_id, row in enumerate(rows):
        name, size, row_annotations, mask = row[:4]
        status = row[4] if len(row) > 4 else "labelled"
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{split}/{name}",
                "width": size.width,
                "height": size.height,
                "license": 1,
                "annotation_status": status,
            }
        )
        if row_annotations is None:
            if mask is None:
                row_annotations = []
            else:
                from utils_new.import_export.coco import encode_rle

                for value in np.unique(mask):
                    if value == 0:
                        continue
                    cls = int(value) - int(masks_offset)
                    if cls not in class_names:
                        raise ValueError(f"Unknown mask class {cls}")
                    count, components, stats, _ = cv2.connectedComponentsWithStats(
                        (mask == value).astype(np.uint8), connectivity=8
                    )
                    for index in range(1, count):
                        x, y, width, height, area = stats[index].tolist()
                        objects.append(
                            dict(
                                id=counter,
                                image_id=image_id,
                                category_id=cls + 1,
                                bbox=[x, y, width, height],
                                area=area,
                                iscrowd=0,
                                segmentation=encode_rle(components == index),
                            )
                        )
                        counter += 1
                continue
        for ann in row_annotations:
            if ann.class_id not in class_names:
                raise ValueError(f"Unknown annotation class {ann.class_id}")
            objects.append(_coco_record(ann, image_id, counter))
            counter += 1
    data = {
        "info": {"description": "Threshold-Researcher generated dataset"},
        "licenses": [{"id": 1, "name": ""}],
        "images": images,
        "annotations": objects,
        "categories": [
            {"id": int(cid) + 1, "name": str(name), "supercategory": "none"}
            for cid, name in sorted(class_names.items())
        ],
    }
    out = output_dir / f"annotations_{split}.json"
    from utils_new.import_export.json_objects import write_json_atomic

    write_json_atomic(out, data)
    return counter


def merge_coco(existing, incoming):
    """Append an export with fresh IDs; reject file/category collisions."""
    from copy import deepcopy

    old, new = deepcopy(existing), deepcopy(incoming)
    categories = lambda data: sorted(
        (int(c["id"]), str(c["name"])) for c in data.get("categories", [])
    )
    if categories(old) != categories(new):
        raise ValueError("COCO categories differ; cannot merge exports")
    names = {str(i["file_name"]).casefold() for i in old["images"]}
    mapping = {}
    next_image = max((i["id"] for i in old["images"]), default=-1) + 1
    for image in new["images"]:
        if str(image["file_name"]).casefold() in names or image["id"] in mapping:
            raise ValueError("COCO image name/ID collision")
        names.add(str(image["file_name"]).casefold())
        mapping[image["id"]] = next_image
        image["id"] = next_image
        next_image += 1
    next_ann = max((a["id"] for a in old["annotations"]), default=-1) + 1
    for ann in new["annotations"]:
        if ann["image_id"] not in mapping:
            raise ValueError("COCO annotation references a missing image")
        ann["image_id"] = mapping[ann["image_id"]]
        ann["id"] = next_ann
        next_ann += 1
    old["images"].extend(new["images"])
    old["annotations"].extend(new["annotations"])
    return old
