# -*- coding: utf-8 -*-
"""Backward-compatible dataset I/O import facade (V19).

For new code import from dataset_files_new, dataset_yolo_labels_new or
 dataset_output_new. This facade retains V15–V18 signatures and constants.
"""
import json
import math
from pathlib import Path

import cv2  # retained: legacy monkeypatches access dataset_io_new.cv2
import numpy as np
import yaml

from utils_new.annotations import (
    Annotation,
    ImageSize,
    MaskEncoding,
    bounds,
    from_legacy,
    mask_to_annotations,
    polygon_area,
    rasterize_annotations,
    to_legacy,
)
from utils_new.dataset_files_new import (
    _read_with_fallback,
    read_image,
    read_index_mask,
    write_image,
)
from utils_new.dataset_output_new import (
    _coco_record,
    make_index_mask,
    write_classes_and_yaml,
    write_coco_split,
    write_index_mask,
)
from utils_new.dataset_yolo_labels_new import (
    LABEL_HINTS,
    _normalized_xywhr,
    _xywhr_points,
    read_yolo_labels,
    write_yolo_labels,
)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
