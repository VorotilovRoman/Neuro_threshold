"""Compatibility morphology and object extraction without Qt/settings."""

from utils_new.annotation_processing import extract_annotations_from_mask


def extract_objects(image, binary, mode, use_hull=False):
    mode = {'Contours (simple)': 'segment', 'Projections (connected components)': 'detect',
            'Min area rectangle': 'obb'}.get(mode, mode)
    if mode == 'None':
        return image, []
    return image, extract_annotations_from_mask(binary, mode=mode, image_shape=image.shape,
                                                use_hull=use_hull)
