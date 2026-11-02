"""Canonical annotation-processing core for Threshold-Researcher.

Independent of Qt, YOLO inference, global settings and filesystem side effects.
This package is additive: existing tabs continue using current utilities until
migrated deliberately, with per-tab regression tests.
"""
from .model import Annotation, AnnotationSample, ImageSize, MaskEncoding
from .adapters import (from_legacy, to_legacy, from_legacy_many, to_legacy_many,
                       from_detection_dict, to_detection_dict, from_ultralytics_result, legacy_convert_annotations)
from .geometry import (GeometryError, LossyConversionError, polygon_area, bounds,
                       rectangle_points, canonical_obb_points, convert_annotation,
                       clip_polygon_to_rect, clip_annotation, overlap_area,
                       overlap_ratio_of_smaller)
from .transforms import (Flip, Rotate90, Rotate, Shear, Crop, Resize, Translate,
                         TransformPlan, TransformReport, apply_affine_to_points,
                         compose_affine_matrices, transform_sample, extract_roi)
from .augmentation import AugmentationPlan, build_augmentation_plan, apply_augmentation_plan
from .masks import (annotation_binary_mask, rasterize_annotations, remap_mask_indices,
                    mask_to_annotations, mask_roi_at_image_scale)
from .rendering import render_annotations, overlay_class_mask

__all__ = [
    'Annotation', 'AnnotationSample', 'ImageSize', 'MaskEncoding',
    'from_legacy', 'to_legacy', 'from_legacy_many', 'to_legacy_many',
    'from_detection_dict', 'to_detection_dict', 'from_ultralytics_result', 'legacy_convert_annotations',
    'GeometryError', 'LossyConversionError', 'polygon_area', 'bounds',
    'rectangle_points', 'canonical_obb_points', 'convert_annotation',
    'clip_polygon_to_rect', 'clip_annotation', 'overlap_area', 'overlap_ratio_of_smaller',
    'Flip', 'Rotate90', 'Rotate', 'Shear', 'Crop', 'Resize', 'Translate',
    'TransformPlan', 'TransformReport', 'apply_affine_to_points',
    'compose_affine_matrices', 'transform_sample', 'extract_roi',
    'AugmentationPlan', 'build_augmentation_plan', 'apply_augmentation_plan',
    'annotation_binary_mask', 'rasterize_annotations', 'remap_mask_indices', 'mask_to_annotations',
    'mask_roi_at_image_scale', 'render_annotations', 'overlay_class_mask',
]
