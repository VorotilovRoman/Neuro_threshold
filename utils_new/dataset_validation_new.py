# -*- coding: utf-8 -*-
"""Backward-compatible public validation facade (V19), with no Qt dependency.

New consumers can use dataset_pair_validator_new, dataset_pair_tools_new and
 dataset_image_stats_new independently. Public V17–V18 names remain stable.
"""
import os

import cv2
import numpy as np

from utils_new.annotations import ImageSize, to_legacy
from utils_new.dataset_files_new import (
    _read_with_fallback,
    read_index_mask,
    write_image,
)
from utils_new.dataset_image_stats_new import ImageStats
from utils_new.dataset_io_new import read_yolo_labels
from utils_new.dataset_pair_tools_new import (
    _list_images,
    build_pair_display,
    convert_pairs_color,
    convert_pairs_to_8bit_png,
    scan_detection_pairs,
    scan_image_only_pairs,
    scan_segmentation_pairs,
    select_pairs_for_8bit_png,
    select_pairs_for_conversion,
)
from utils_new.dataset_pair_validator_new import (
    _VALID_INPUT_FORMATS,
    IMG_EXTENSIONS,
    INPUT_FORMAT_AUTO,
    INPUT_FORMAT_DETECT,
    INPUT_FORMAT_MIXED,
    INPUT_FORMAT_OBB_CORNERS,
    INPUT_FORMAT_OBB_XYWHR,
    INPUT_FORMAT_SEGMENT,
    PairValidator,
    load_annotations,
    read_mask_with_fallback,
)
