# libs_internal.py
# -*- coding: utf-8 -*-

# Импортируем все внешние библиотеки
from import_libs_external import *

# ========== Утилиты проекта (общие) ==========
from utils.image_io import (
    load_images_universal, save_coordinates, numpy_to_qpixmap, save_annotations,
    read_image_with_fallback, resize_to_max_side, normalize_to_uint8, load_dataset_from_yaml_with_masks, load_dataset_from_yaml,
    convert_to_grayscale, load_annotations, read_image_with_fallback_find, convert_segment_masks_to_yolo_seg_manual
)

from utils_ops.segmentation_ops import (
    segment_contours, segment_projections, segment_min_area_rect, get_display_params, normalize_method_name,
    apply_threshold_method, apply_morphology, draw_selected_objects,
    format_object_for_list, update_annotation_list, delete_annotation_by_index,
    draw_yolo_annotations, draw_label, draw_label_rotated
)
from utils.settings import settings, apply_theme
from utils.presets import (
    PresetManager, GradientPresetManager, InteractivePresetManager,
    TraditionalMLPresetManager, DeepLearningPresetManager
)

from utils.analysis import analyze, global_top_diverse
from utils.prepare import (
    sample_rows,
    aggregate_by_params,
    filter_full_image_object,
    filter_invalid_params,
    filter_anomaly_count,
    filter_zero_true,
    filter_oversegmentation,
    filter_duplicates,
    add_cluster_labels,
    save_unique_params,
    add_meanap_metrics,
    cluster_open_close,
    optimize_invert,
    filter_no_true_objects      # новая функция
)
from utils.metrics_core import DEFAULT_IOU_THRESHOLDS




from utils.smart_view import SmartGraphicsView # Новый метод с зумом и рисованием

# ========== Утилиты для градиентных методов ==========
from utils_ops.gradient_ops import apply_gradient_method
from utils_auto_research.gradient_auto_settings_dialog import GradientAutoSettingsDialog
from utils_auto_research.gradient_auto_research import GradientAutoResearch

# ========== Утилиты для интерактивных методов ==========
from utils_ops.interactive_ops import (
    grabcut_segmentation, watershed_segmentation, lazy_snapping,
    supercut_segmentation, onecut_segmentation,
    random_walker_segmentation, active_contour_segmentation
)

# ========== Утилиты для традиционных ML ==========
from utils_ops.traditional_ml_ops import (
    extract_pixel_features, extract_superpixel_features, create_superpixels,
    train_kmeans, train_meanshift, train_svm, train_random_forest,
    train_xgboost, train_decision_tree, predict_pixelwise, predict_superpixel,
    compute_elbow_kmeans, find_optimal_k_elbow
)

# ========== Утилиты для глубокого обучения ==========
from utils_ops.deep_learning_ops import create_segmentor

# ========== Утилиты автоисследования ==========
from utils_auto_research.threshold_auto_research import AutoResearch
from utils_auto_research.threshold_auto_settings_dialog import AutoSettingsDialog


# Импорты из общего модуля метрик
from utils.metrics_core import yolo_to_xyxy, xyxy_to_yolo, iou_xyxy


# ========== Утилиты ансамблей и доп. операции ==========
from utils.ensemble_utils import run_ensemble, get_detections_from_preset, extract_threshold_params, ensemble_detections

from utils.aug_utils import (
    repair_obb_to_rectangle, log_annotations_summary, resize_image_and_annotations, clip_annotations, compose_affine_matrices, resize_image_and_annotations_stretch, apply_affine_to_points_np
)

from albumentations import PadIfNeeded, LongestMaxSize

from utils.augmentation_base import apply_simple_augmentations, apply_simple_augmentations_mask
from utils.augmentation_random import apply_complex_augmentations, apply_complex_augmentations_mask


# ========== UI компоненты ==========
from ui.image_navigation import ImageNavigationWidget
from ui.log_widget import LogWidget
from ui.control_buttons import ControlButtons


from utils.DatasetGenerator import DatasetGeneratorThread


from ui.DL_train_settings_base import YOLOTrainWidget, UNetTrainWidget
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget, SegFormerTrainWidget, SAMTrainWidget

from ui.DL_predict_settings import (
    YOLOInferenceSettings,
    UNetInferenceSettings,
    DeepLabV3InferenceSettings,
    SegFormerInferenceSettings,
    SAMInferenceSettings,
    CustomONNXInferenceSettings
)


# Импорты из соседних модулей пакета utils
from utils_ops.segmentation_ops import (
    apply_threshold_method,
    apply_morphology,
    segment_contours,
    segment_projections,
    segment_min_area_rect
)
