"""Canonical annotation-processing core for Threshold-Researcher.

Independent of Qt, YOLO inference, global settings and filesystem side effects.
This package is additive: existing tabs continue using current utilities until
migrated deliberately, with per-tab regression tests.
"""
from .adapters import (
                       from_detection_dict,
                       from_legacy,
                       from_legacy_many,
                       from_ultralytics_result,
                       legacy_convert_annotations,
                       to_detection_dict,
                       to_legacy,
                       to_legacy_many,
)
from .augmentation import (
                       AugmentationPlan,
                       apply_augmentation_plan,
                       build_augmentation_plan,
)
from .geometry import (
                       GeometryError,
                       LossyConversionError,
                       bounds,
                       canonical_obb_points,
                       clip_annotation,
                       clip_polygon_to_rect,
                       convert_annotation,
                       overlap_area,
                       overlap_ratio_of_smaller,
                       polygon_area,
                       rectangle_points,
)
from .masks import (
                       annotation_binary_mask,
                       mask_roi_at_image_scale,
                       mask_to_annotations,
                       rasterize_annotations,
                       remap_mask_indices,
)
from .model import Annotation, AnnotationSample, ImageSize, MaskEncoding
from .rendering import overlay_class_mask, render_annotations
from .transforms import (
                       Crop,
                       Flip,
                       Resize,
                       Rotate,
                       Rotate90,
                       Shear,
                       TransformPlan,
                       TransformReport,
                       Translate,
                       apply_affine_to_points,
                       compose_affine_matrices,
                       extract_roi,
                       transform_sample,
)

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
