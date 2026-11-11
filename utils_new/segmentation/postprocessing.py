"""Compatibility morphology and object extraction without Qt/settings."""

from utils_new.annotation_processing import extract_annotations_from_mask
from utils_new.morphology import MorphologyStep, apply_morphology


def apply_relative_morphology(binary, close_factor, open_factor, kernel_shape, img_shape=None):
    shape = binary.shape if img_shape is None else img_shape
    base = min(shape[:2])
    steps = [MorphologyStep(name, max(1, int(factor*base)) if factor > 0 else 0)
             for name, factor in [('close', close_factor), ('open', open_factor)]]
    return apply_morphology(binary, steps, shape=kernel_shape)


def extract_objects(image, binary, mode, use_hull=False):
    mode = {'Contours (simple)': 'segment', 'Projections (connected components)': 'detect',
            'Min area rectangle': 'obb'}.get(mode, mode)
    if mode == 'None':
        return image, []
    return image, extract_annotations_from_mask(binary, mode=mode, image_shape=image.shape,
                                                use_hull=use_hull)
