"""Object extraction using the same annotation kinds as the dataset editor."""
from utils_new.annotation_processing import extract_annotations_from_mask
from utils_new.annotations.modes import supports_hull, validate_mode


def extract_objects(image, binary, mode, use_hull=False):
    validate_mode(mode)
    return image, extract_annotations_from_mask(binary, mode=mode, image_shape=image.shape,
                                                use_hull=bool(use_hull) and supports_hull(mode))
