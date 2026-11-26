"""Object extraction using the same annotation kinds as the dataset editor."""
import numpy as np
from utils_new.annotations.processing import extract_annotations_from_mask
from utils_new.annotations.modes import supports_hull, validate_mode
from utils_new.annotations.adapters import to_legacy_many
from utils_new.annotations.geometry import enclose_annotations
from utils_new.annotations.masks import mask_to_annotations
from utils_new.annotations.model import ImageSize


def extract_objects(image, binary, mode, use_hull=False, merge_all=False):
    validate_mode(mode)
    if merge_all:
        if mode == 'none':
            return image, []
        if binary.ndim != 2 or binary.dtype != np.uint8 or binary.shape != image.shape[:2]:
            raise ValueError('Нужна маска uint8, совпадающая по размеру с изображением')
        # Group fragments before the usual per-object area cutoff; otherwise
        # splitting a valid object into small islands would discard all of it.
        objects = mask_to_annotations((binary > 0).astype('uint8'), kind=mode, min_area=0, holes='drop')
        size = ImageSize(image.shape[1], image.shape[0])
        return image, to_legacy_many(enclose_annotations(objects), size)
    objects = extract_annotations_from_mask(binary, mode=mode, image_shape=image.shape,
                                             use_hull=bool(use_hull) and supports_hull(mode))
    return image, objects
