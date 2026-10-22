# import_libs_internal.py
# -*- coding: utf-8 -*-

from import_libs_external import *

# ========== Утилиты проекта (общие) ==========
from utils.image_io import (
    load_images_universal, save_coordinates, numpy_to_qpixmap, save_annotations,
    read_image_with_fallback, read_mask_with_fallback,      # NEW
    resize_to_max_side, normalize_to_uint8,
    load_dataset_from_yaml_with_masks, load_dataset_from_yaml,
    convert_to_grayscale, load_annotations,
    read_image_with_fallback_find, convert_segment_masks_to_yolo_seg_manual,
    load_one_image_item, LazyImageCache, canonicalize_obb_corners,
    # NEW — константы форматов
    FORMAT_HINT_AUTO, FORMAT_HINT_MIXED, FORMAT_HINT_DETECT,
    FORMAT_HINT_OBB_CORNERS, FORMAT_HINT_OBB_XYWHR, FORMAT_HINT_SEGMENT,
)
from utils.yaml_io import (
    load_yolo_yaml, load_unet_yaml_pairs, find_mask_for_image,
)

# ========== НОВЫЕ модули (генерация датасетов) ==========
from utils.annotation_converter import (
    convert_annotations,
    to_detect, to_obb, to_segment,
    SUBTYPE_KEEP, SUBTYPE_DETECT, SUBTYPE_OBB, SUBTYPE_SEGMENT,
)
from utils.coco_export import (
    export_coco_from_labels, export_coco_from_masks,
)
from utils.coco_io import (
    load_coco, write_yolo_txt_for_coco_items,
)
from utils.unet_export import (
    export_masks_from_annotations, write_unet_yaml,
)
from utils.unet_io import (
    load_unet_yaml, is_unet_yaml,
)

# ========== Остальные утилиты проекта ==========
from utils_ops.segmentation_ops import (
    segment_contours, segment_projections, segment_min_area_rect,
    get_display_params, normalize_method_name,
    apply_threshold_method, apply_morphology, draw_selected_objects,
    format_object_for_list, update_annotation_list, delete_annotation_by_index,
    draw_yolo_annotations, draw_label, draw_label_rotated, get_class_color,
    draw_class_label,
)
from utils.settings import settings, apply_theme
from utils.presets import (
    PresetManager, GradientPresetManager, InteractivePresetManager,
    TraditionalMLPresetManager, DeepLearningPresetManager,
)
from utils.analysis import analyze, global_top_diverse
from utils.prepare import (
    sample_rows, aggregate_by_params,
    filter_full_image_object, filter_invalid_params, filter_anomaly_count,
    filter_zero_true, filter_oversegmentation, filter_duplicates,
    add_cluster_labels, save_unique_params, add_meanap_metrics,
    cluster_open_close, optimize_invert, filter_no_true_objects,
)
from utils.metrics_core import DEFAULT_IOU_THRESHOLDS, yolo_to_xyxy, xyxy_to_yolo, iou_xyxy

from utils.smart_view import SmartGraphicsView

# ========== Градиентные методы ==========
from utils_ops.gradient_ops import apply_gradient_method
from utils_auto_research.gradient_auto_settings_dialog import GradientAutoSettingsDialog
from utils_auto_research.gradient_auto_research import GradientAutoResearch

# ========== Интерактивные методы ==========
from utils_ops.interactive_ops import (
    grabcut_segmentation, watershed_segmentation, lazy_snapping,
    supercut_segmentation, onecut_segmentation,
    random_walker_segmentation, active_contour_segmentation,
)

# ========== Традиционные ML ==========
from utils_ops.traditional_ml_ops import (
    extract_pixel_features, extract_superpixel_features, create_superpixels,
    train_kmeans, train_meanshift, train_svm, train_random_forest,
    train_xgboost, train_decision_tree,
    predict_pixelwise, predict_superpixel,
    compute_elbow_kmeans, find_optimal_k_elbow,
)

# ========== Глубокое обучение ==========
from utils_ops.deep_learning_ops import create_segmentor

# ========== Автоисследование ==========
from utils_auto_research.threshold_auto_research import AutoResearch
from utils_auto_research.threshold_auto_settings_dialog import AutoSettingsDialog

# ========== Ансамбли ==========
from utils.ensemble_utils import (
    run_ensemble, get_detections_from_preset,
    extract_threshold_params, ensemble_detections,
)

# ========== Аугментации ==========
from utils.aug_utils import (
    repair_obb_to_rectangle, log_annotations_summary,
    resize_image_and_annotations, clip_annotations,
    compose_affine_matrices, resize_image_and_annotations_stretch,
    apply_affine_to_points_np,
)
from utils.augmentation_base import (
    apply_simple_augmentations, apply_simple_augmentations_mask,
)
from utils.augmentation_random import (
    apply_complex_augmentations, apply_complex_augmentations_mask,
)

# ========== UI компоненты ==========
from ui.image_navigation import ImageNavigationWidget, NavigationShortcutInstaller
from ui.log_widget import LogWidget
from ui.control_buttons import ControlButtons

# ========== Генератор датасетов ==========
from utils.DatasetGenerator import (
    DatasetGeneratorThread,
    INPUT_LABELS, INPUT_MASKS, INPUT_IMAGES,
)

# ========== Тренеры DL ==========
from ui.DL_train_settings_base import YOLOTrainWidget, UNetTrainWidget
from ui.DL_train_settings_new_model import (
    DeepLabV3TrainWidget, SegFormerTrainWidget, SAMTrainWidget,
)
from ui.DL_predict_settings import (
    YOLOInferenceSettings,
    UNetInferenceSettings,
    DeepLabV3InferenceSettings,
    SegFormerInferenceSettings,
    SAMInferenceSettings,
    CustomONNXInferenceSettings,
)


from utils.dataset_validation import (
    ImageStats,
    PairValidator,
    PairValidationDialog,
    scan_detection_pairs,
    scan_segmentation_pairs,
    scan_image_only_pairs,
    select_pairs_for_conversion,
    convert_pairs_color,
    select_pairs_for_8bit_png,
    convert_pairs_to_8bit_png,
    build_pair_display,
    DatasetLoaderMixin,
)




from utils.dataset_analysis_logic import (
    DatasetAnalyzer, FilterParams,
    ClassImbalanceAnalyzer, ResolutionAnalyzer, BrightnessAnalyzer,
    TrainingResultsLoader, clear_brightness_cache,
)
from utils.dataset_audit_logic import detect_device