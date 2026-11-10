# Подтверждённые структурные копии

[Разбор и рекомендации](Duplication.md) · [Справочник](Module_Reference.md)

Автоматическое приложение: одинаковые AST функций/методов в разных файлах, от 9 строк исходника. Имя функции, docstring и номера строк игнорируются; аргументы и тело сохраняются. Сигнатуры и окружающие импорты проверяйте перед объединением. Это не поиск похожих алгоритмов и не доказательство взаимозаменяемости классов.

Групп совпадений: **144**.

## 1. mouseMoveEvent

- [utils/smart_view.py:916](../utils/smart_view.py#L916) — `mouseMoveEvent`, 353 строк
- [utils_new/smart_view_new.py:1169](../utils_new/smart_view_new.py#L1169) — `mouseMoveEvent`, 353 строк

## 2. main

- [utils_aeb/auto-aeb_def.py:372](../utils_aeb/auto-aeb_def.py#L372) — `main`, 193 строк
- [utils_aeb/auto-aeb_old.py:375](../utils_aeb/auto-aeb_old.py#L375) — `main`, 193 строк

## 3. _try_segment_double_click

- [utils/smart_view.py:1332](../utils/smart_view.py#L1332) — `_try_segment_double_click`, 105 строк
- [utils_new/smart_view_new.py:1585](../utils_new/smart_view_new.py#L1585) — `_try_segment_double_click`, 105 строк

## 4. load_yolo_yaml

- [utils/yaml_io.py:162](../utils/yaml_io.py#L162) — `load_yolo_yaml`, 77 строк
- [utils_new/dataset_yaml_new.py:150](../utils_new/dataset_yaml_new.py#L150) — `load_yolo_yaml`, 77 строк

## 5. _rebuild_file_list_display

- [utils/dataset_validation.py:1455](../utils/dataset_validation.py#L1455) — `_rebuild_file_list_display`, 75 строк
- [utils_new/dataset_loader_new.py:617](../utils_new/dataset_loader_new.py#L617) — `_rebuild_file_list_display`, 75 строк

## 6. __getitem__

- [utils_training/train_unet_worker.py:264](../utils_training/train_unet_worker.py#L264) — `__getitem__`, 72 строк
- [utils_training/train_unet_worker_рабочий.py:266](../utils_training/train_unet_worker_рабочий.py#L266) — `__getitem__`, 75 строк

## 7. compute_metrics

- [utils_training/train_unet_worker.py:404](../utils_training/train_unet_worker.py#L404) — `compute_metrics`, 72 строк
- [utils_training/train_unet_worker_рабочий.py:409](../utils_training/train_unet_worker_рабочий.py#L409) — `compute_metrics`, 72 строк

## 8. _resize_pair

- [utils_training/train_unet_worker.py:208](../utils_training/train_unet_worker.py#L208) — `_resize_pair`, 55 строк
- [utils_training/train_unet_worker_рабочий.py:201](../utils_training/train_unet_worker_рабочий.py#L201) — `_resize_pair`, 64 строк

## 9. scan_pairs

- [utils/dataset_validation.py:1303](../utils/dataset_validation.py#L1303) — `scan_pairs`, 63 строк
- [utils_new/dataset_loader_new.py:465](../utils_new/dataset_loader_new.py#L465) — `scan_pairs`, 63 строк

## 10. _get_obb_handles

- [utils/smart_view.py:576](../utils/smart_view.py#L576) — `_get_obb_handles`, 59 строк
- [utils_new/smart_view_new.py:829](../utils_new/smart_view_new.py#L829) — `_get_obb_handles`, 59 строк

## 11. apply_threshold_method

- [utils_new/threshold_processing.py:135](../utils_new/threshold_processing.py#L135) — `apply_threshold_method`, 59 строк
- [utils_ops/segmentation_ops.py:564](../utils_ops/segmentation_ops.py#L564) — `apply_threshold_method`, 59 строк

## 12. save_confusion_matrix

- [utils_training/train_unet_worker.py:645](../utils_training/train_unet_worker.py#L645) — `save_confusion_matrix`, 59 строк
- [utils_training/train_unet_worker_рабочий.py:606](../utils_training/train_unet_worker_рабочий.py#L606) — `save_confusion_matrix`, 55 строк

## 13. save_pr_curve

- [utils_training/train_unet_worker.py:586](../utils_training/train_unet_worker.py#L586) — `save_pr_curve`, 57 строк
- [utils_training/train_unet_worker_рабочий.py:563](../utils_training/train_unet_worker_рабочий.py#L563) — `save_pr_curve`, 42 строк

## 14. build_pair_display

- [utils/dataset_validation.py:616](../utils/dataset_validation.py#L616) — `build_pair_display`, 54 строк
- [utils_new/dataset_pair_tools_new.py:183](../utils_new/dataset_pair_tools_new.py#L183) — `build_pair_display`, 54 строк

## 15. _apply_loaded_pairs

- [utils/dataset_validation.py:1249](../utils/dataset_validation.py#L1249) — `_apply_loaded_pairs`, 52 строк
- [utils_new/dataset_loader_new.py:411](../utils_new/dataset_loader_new.py#L411) — `_apply_loaded_pairs`, 52 строк

## 16. check_dataset_masks_consistency

- [utils_training/train_unet_worker.py:820](../utils_training/train_unet_worker.py#L820) — `check_dataset_masks_consistency`, 47 строк
- [utils_training/train_unet_worker_рабочий.py:751](../utils_training/train_unet_worker_рабочий.py#L751) — `check_dataset_masks_consistency`, 52 строк

## 17. train_one_epoch

- [utils_aeb/auto-aeb_def.py:280](../utils_aeb/auto-aeb_def.py#L280) — `train_one_epoch`, 49 строк
- [utils_aeb/auto-aeb_old.py:283](../utils_aeb/auto-aeb_old.py#L283) — `train_one_epoch`, 49 строк

## 18. save_current_annotations

- [tabs/deep_learning_methods.py:971](../tabs/deep_learning_methods.py#L971) — `save_current_annotations`, 47 строк
- [tabs/gradient_methods.py:737](../tabs/gradient_methods.py#L737) — `save_current_annotations`, 47 строк
- [tabs/interactive_methods.py:685](../tabs/interactive_methods.py#L685) — `save_current_annotations`, 47 строк
- [tabs/threshold_methods.py:704](../tabs/threshold_methods.py#L704) — `save_current_annotations`, 48 строк

## 19. _resolve_split_dirs

- [utils/yaml_io.py:32](../utils/yaml_io.py#L32) — `_resolve_split_dirs`, 47 строк
- [utils_new/dataset_yaml_new.py:20](../utils_new/dataset_yaml_new.py#L20) — `_resolve_split_dirs`, 47 строк

## 20. _default_settings

- [utils_auto_research/gradient_auto_research.py:163](../utils_auto_research/gradient_auto_research.py#L163) — `_default_settings`, 32 строк
- [utils_auto_research/gradient_auto_settings_dialog.py:69](../utils_auto_research/gradient_auto_settings_dialog.py#L69) — `_default_settings`, 44 строк

## 21. _fill_class_table_from_dict

- [utils/dataset_validation.py:1576](../utils/dataset_validation.py#L1576) — `_fill_class_table_from_dict`, 43 строк
- [utils_new/dataset_loader_new.py:738](../utils_new/dataset_loader_new.py#L738) — `_fill_class_table_from_dict`, 43 строк

## 22. _finish_segment_line

- [utils/smart_view.py:1657](../utils/smart_view.py#L1657) — `_finish_segment_line`, 43 строк
- [utils_new/smart_view_new.py:1913](../utils_new/smart_view_new.py#L1913) — `_finish_segment_line`, 43 строк

## 23. _compute_obb_preview_params

- [utils/smart_view.py:703](../utils/smart_view.py#L703) — `_compute_obb_preview_params`, 42 строк
- [utils_new/smart_view_new.py:956](../utils_new/smart_view_new.py#L956) — `_compute_obb_preview_params`, 42 строк

## 24. _obb_to_params

- [utils/smart_view.py:479](../utils/smart_view.py#L479) — `_obb_to_params`, 42 строк
- [utils_new/smart_view_new.py:732](../utils_new/smart_view_new.py#L732) — `_obb_to_params`, 42 строк

## 25. __init__

- [utils_training/train_unet_worker.py:163](../utils_training/train_unet_worker.py#L163) — `__init__`, 41 строк
- [utils_training/train_unet_worker_рабочий.py:156](../utils_training/train_unet_worker_рабочий.py#L156) — `__init__`, 41 строк

## 26. _validate_unified_annotation

- [tabs/layout_dataset.py:33](../tabs/layout_dataset.py#L33) — `_validate_unified_annotation`, 40 строк
- [tabs/viewing_dataset.py:42](../tabs/viewing_dataset.py#L42) — `_validate_unified_annotation`, 40 строк

## 27. detect_annotation_type

- [utils/dataset_validation.py:398](../utils/dataset_validation.py#L398) — `detect_annotation_type`, 40 строк
- [utils_new/dataset_pair_validator_new.py:285](../utils_new/dataset_pair_validator_new.py#L285) — `detect_annotation_type`, 40 строк

## 28. _obb_to_points

- [utils/smart_view.py:522](../utils/smart_view.py#L522) — `_obb_to_points`, 39 строк
- [utils_new/smart_view_new.py:775](../utils_new/smart_view_new.py#L775) — `_obb_to_points`, 39 строк

## 29. find_mask_for_image

- [utils/yaml_io.py:120](../utils/yaml_io.py#L120) — `find_mask_for_image`, 39 строк
- [utils_new/dataset_yaml_new.py:109](../utils_new/dataset_yaml_new.py#L109) — `find_mask_for_image`, 39 строк

## 30. _find_label_for_image

- [utils/yaml_io.py:81](../utils/yaml_io.py#L81) — `_find_label_for_image`, 38 строк
- [utils_new/dataset_yaml_new.py:69](../utils_new/dataset_yaml_new.py#L69) — `_find_label_for_image`, 38 строк

## 31. evaluate

- [utils_aeb/auto-aeb_def.py:332](../utils_aeb/auto-aeb_def.py#L332) — `evaluate`, 38 строк
- [utils_aeb/auto-aeb_old.py:335](../utils_aeb/auto-aeb_old.py#L335) — `evaluate`, 38 строк

## 32. update_coordinates_display

- [tabs/deep_learning_methods.py:796](../tabs/deep_learning_methods.py#L796) — `update_coordinates_display`, 34 строк
- [tabs/gradient_methods.py:426](../tabs/gradient_methods.py#L426) — `update_coordinates_display`, 34 строк
- [tabs/interactive_methods.py:650](../tabs/interactive_methods.py#L650) — `update_coordinates_display`, 34 строк
- [tabs/threshold_methods.py:525](../tabs/threshold_methods.py#L525) — `update_coordinates_display`, 36 строк

## 33. get_loss_function

- [utils_training/train_unet_worker.py:339](../utils_training/train_unet_worker.py#L339) — `get_loss_function`, 34 строк
- [utils_training/train_unet_worker_рабочий.py:344](../utils_training/train_unet_worker_рабочий.py#L344) — `get_loss_function`, 34 строк

## 34. __init__

- [utils_aeb/auto-aeb_unet.py:249](../utils_aeb/auto-aeb_unet.py#L249) — `__init__`, 33 строк
- [utils_aeb/yolo_aeb_utils.py:256](../utils_aeb/yolo_aeb_utils.py#L256) — `__init__`, 33 строк

## 35. save_training_plots

- [utils_training/train_unet_worker.py:542](../utils_training/train_unet_worker.py#L542) — `save_training_plots`, 33 строк
- [utils_training/train_unet_worker_рабочий.py:527](../utils_training/train_unet_worker_рабочий.py#L527) — `save_training_plots`, 27 строк

## 36. scan_segmentation_pairs

- [utils/dataset_validation.py:475](../utils/dataset_validation.py#L475) — `scan_segmentation_pairs`, 32 строк
- [utils_new/dataset_pair_tools_new.py:42](../utils_new/dataset_pair_tools_new.py#L42) — `scan_segmentation_pairs`, 32 строк

## 37. _hit_test_obb_handles

- [utils/smart_view.py:636](../utils/smart_view.py#L636) — `_hit_test_obb_handles`, 32 строк
- [utils_new/smart_view_new.py:889](../utils_new/smart_view_new.py#L889) — `_hit_test_obb_handles`, 32 строк

## 38. format_stats_text

- [utils/dataset_validation.py:121](../utils/dataset_validation.py#L121) — `format_stats_text`, 30 строк
- [utils_new/dataset_image_stats_new.py:107](../utils_new/dataset_image_stats_new.py#L107) — `format_stats_text`, 30 строк

## 39. _ensure_min_polygon_points

- [utils/smart_view.py:1626](../utils/smart_view.py#L1626) — `_ensure_min_polygon_points`, 30 строк
- [utils_new/smart_view_new.py:1882](../utils_new/smart_view_new.py#L1882) — `_ensure_min_polygon_points`, 30 строк

## 40. _max_height_inside

- [utils/smart_view.py:672](../utils/smart_view.py#L672) — `_max_height_inside`, 30 строк
- [utils_new/smart_view_new.py:925](../utils_new/smart_view_new.py#L925) — `_max_height_inside`, 30 строк

## 41. mouseDoubleClickEvent

- [utils/smart_view.py:1301](../utils/smart_view.py#L1301) — `mouseDoubleClickEvent`, 30 строк
- [utils_new/smart_view_new.py:1554](../utils_new/smart_view_new.py#L1554) — `mouseDoubleClickEvent`, 30 строк

## 42. __init__

- [utils_aeb/auto-aeb_unet.py:211](../utils_aeb/auto-aeb_unet.py#L211) — `__init__`, 28 строк
- [utils_aeb/yolo_aeb_utils.py:218](../utils_aeb/yolo_aeb_utils.py#L218) — `__init__`, 28 строк

## 43. format_annotation_display

- [utils_new/annotation_processing.py:1185](../utils_new/annotation_processing.py#L1185) — `format_annotation_display`, 28 строк
- [utils_ops/segmentation_ops.py:415](../utils_ops/segmentation_ops.py#L415) — `format_annotation_display`, 28 строк

## 44. read_image_bgr

- [cluster/canonicalize_crops.py:73](../cluster/canonicalize_crops.py#L73) — `read_image_bgr`, 21 строк
- [cluster/canonicalize_crops_method2.py:90](../cluster/canonicalize_crops_method2.py#L90) — `read_image_bgr`, 21 строк
- [cluster/save_yolo_object_to_path.py:49](../cluster/save_yolo_object_to_path.py#L49) — `read_image`, 27 строк

## 45. _is_polygon_rectangle

- [utils/coco_io.py:48](../utils/coco_io.py#L48) — `_is_polygon_rectangle`, 27 строк
- [utils_new/dataset_coco_new.py:50](../utils_new/dataset_coco_new.py#L50) — `_is_polygon_rectangle`, 27 строк

## 46. mouseReleaseEvent

- [utils/smart_view.py:1270](../utils/smart_view.py#L1270) — `mouseReleaseEvent`, 27 строк
- [utils_new/smart_view_new.py:1523](../utils_new/smart_view_new.py#L1523) — `mouseReleaseEvent`, 27 строк

## 47. _load_bg

- [utils/image_io.py:1036](../utils/image_io.py#L1036) — `_load_bg`, 26 строк
- [utils_new/dataset_files_new.py:263](../utils_new/dataset_files_new.py#L263) — `_load_bg`, 26 строк

## 48. _apply_segment_change

- [utils/smart_view.py:1438](../utils/smart_view.py#L1438) — `_apply_segment_change`, 26 строк
- [utils_new/smart_view_new.py:1691](../utils_new/smart_view_new.py#L1691) — `_apply_segment_change`, 26 строк

## 49. _default_settings

- [utils_auto_research/threshold_auto_research.py:152](../utils_auto_research/threshold_auto_research.py#L152) — `_default_settings`, 26 строк
- [utils_auto_research/threshold_auto_settings_dialog.py:39](../utils_auto_research/threshold_auto_settings_dialog.py#L39) — `_default_settings`, 25 строк

## 50. _load_coco_into_labeler

- [tabs/layout_dataset.py:601](../tabs/layout_dataset.py#L601) — `_load_coco_into_labeler`, 24 строк
- [tabs/viewing_dataset.py:1461](../tabs/viewing_dataset.py#L1461) — `_load_coco_into_viewer`, 23 строк

## 51. _open_pair_validation

- [utils/dataset_validation.py:1405](../utils/dataset_validation.py#L1405) — `_open_pair_validation`, 24 строк
- [utils_new/dataset_loader_new.py:567](../utils_new/dataset_loader_new.py#L567) — `_open_pair_validation`, 24 строк

## 52. normalize_method_name

- [utils_new/threshold_processing.py:110](../utils_new/threshold_processing.py#L110) — `normalize_method_name`, 24 строк
- [utils_ops/segmentation_ops.py:538](../utils_ops/segmentation_ops.py#L538) — `normalize_method_name`, 24 строк

## 53. read_image_rgb

- [cluster/cluster_embeddings.py:90](../cluster/cluster_embeddings.py#L90) — `read_image_rgb`, 23 строк
- [cluster/extract_embeddings.py:59](../cluster/extract_embeddings.py#L59) — `read_image_rgb`, 23 строк

## 54. update_annotated_view

- [tabs/deep_learning_methods.py:774](../tabs/deep_learning_methods.py#L774) — `update_annotated_view`, 20 строк
- [tabs/gradient_methods.py:402](../tabs/gradient_methods.py#L402) — `update_annotated_view`, 23 строк
- [tabs/interactive_methods.py:627](../tabs/interactive_methods.py#L627) — `update_annotated_view`, 22 строк
- [tabs/traditional_ml_methods.py:934](../tabs/traditional_ml_methods.py#L934) — `update_annotated_view`, 20 строк

## 55. _make_list_item

- [utils/dataset_validation.py:1431](../utils/dataset_validation.py#L1431) — `_make_list_item`, 23 строк
- [utils_new/dataset_loader_new.py:593](../utils_new/dataset_loader_new.py#L593) — `_make_list_item`, 23 строк

## 56. _on_dataset_type_changed

- [utils/dataset_validation.py:1540](../utils/dataset_validation.py#L1540) — `_on_dataset_type_changed`, 23 строк
- [utils_new/dataset_loader_new.py:702](../utils_new/dataset_loader_new.py#L702) — `_on_dataset_type_changed`, 23 строк

## 57. _finish_segment

- [utils/smart_view.py:1602](../utils/smart_view.py#L1602) — `_finish_segment`, 23 строк
- [utils_new/smart_view_new.py:1858](../utils_new/smart_view_new.py#L1858) — `_finish_segment`, 23 строк

## 58. __init__

- [utils_aeb/auto-aeb_def.py:112](../utils_aeb/auto-aeb_def.py#L112) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_def_new.py:126](../utils_aeb/auto-aeb_def_new.py#L126) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_old.py:115](../utils_aeb/auto-aeb_old.py#L115) — `__init__`, 23 строк

## 59. _detect_annotation_to_obb

- [tabs/layout_dataset.py:78](../tabs/layout_dataset.py#L78) — `_detect_annotation_to_obb`, 22 строк
- [tabs/viewing_dataset.py:87](../tabs/viewing_dataset.py#L87) — `_detect_annotation_to_obb`, 22 строк

## 60. update_object_list

- [tabs/deep_learning_methods.py:754](../tabs/deep_learning_methods.py#L754) — `update_object_list`, 19 строк
- [tabs/gradient_methods.py:380](../tabs/gradient_methods.py#L380) — `update_object_list`, 21 строк
- [tabs/interactive_methods.py:605](../tabs/interactive_methods.py#L605) — `update_object_list`, 21 строк
- [tabs/threshold_methods.py:484](../tabs/threshold_methods.py#L484) — `update_object_list`, 21 строк

## 61. background_symmetry_threshold

- [utils_new/threshold_processing.py:73](../utils_new/threshold_processing.py#L73) — `background_symmetry_threshold`, 21 строк
- [utils_ops/segmentation_ops.py:200](../utils_ops/segmentation_ops.py#L200) — `background_symmetry_threshold`, 21 строк

## 62. _start_annotation_validation

- [tabs/layout_dataset.py:790](../tabs/layout_dataset.py#L790) — `_start_annotation_validation`, 20 строк
- [tabs/viewing_dataset.py:797](../tabs/viewing_dataset.py#L797) — `_start_annotation_validation`, 20 строк

## 63. _validate_all_pairs

- [utils/dataset_validation.py:1368](../utils/dataset_validation.py#L1368) — `_validate_all_pairs`, 20 строк
- [utils_new/dataset_loader_new.py:530](../utils_new/dataset_loader_new.py#L530) — `_validate_all_pairs`, 20 строк

## 64. _handle_load_coco

- [utils/dataset_validation.py:1011](../utils/dataset_validation.py#L1011) — `_handle_load_coco`, 19 строк
- [utils_new/dataset_loader_new.py:241](../utils_new/dataset_loader_new.py#L241) — `_handle_load_coco`, 19 строк

## 65. select_labels_folder

- [utils/dataset_validation.py:922](../utils/dataset_validation.py#L922) — `select_labels_folder`, 19 строк
- [utils_new/dataset_loader_new.py:148](../utils_new/dataset_loader_new.py#L148) — `select_labels_folder`, 19 строк

## 66. _finish_obb

- [utils/smart_view.py:1700](../utils/smart_view.py#L1700) — `_finish_obb`, 19 строк
- [utils_new/smart_view_new.py:1956](../utils_new/smart_view_new.py#L1956) — `_finish_obb`, 19 строк

## 67. _update_segment_line

- [utils/smart_view.py:1544](../utils/smart_view.py#L1544) — `_update_segment_line`, 19 строк
- [utils_new/smart_view_new.py:1800](../utils_new/smart_view_new.py#L1800) — `_update_segment_line`, 19 строк

## 68. is_unet_yaml

- [utils/unet_io.py:127](../utils/unet_io.py#L127) — `is_unet_yaml`, 19 строк
- [utils_new/dataset_unet_new.py:120](../utils_new/dataset_unet_new.py#L120) — `is_unet_yaml`, 19 строк

## 69. save_as_preset

- [tabs/deep_learning_methods.py:391](../tabs/deep_learning_methods.py#L391) — `save_as_preset`, 18 строк
- [tabs/gradient_methods.py:112](../tabs/gradient_methods.py#L112) — `save_as_preset`, 18 строк
- [tabs/interactive_methods.py:136](../tabs/interactive_methods.py#L136) — `save_as_preset`, 16 строк
- [tabs/traditional_ml_methods.py:306](../tabs/traditional_ml_methods.py#L306) — `save_as_preset`, 18 строк

## 70. run

- [tabs/layout_dataset.py:132](../tabs/layout_dataset.py#L132) — `run`, 18 строк
- [tabs/viewing_dataset.py:141](../tabs/viewing_dataset.py#L141) — `run`, 18 строк

## 71. save_as_preset

- [tabs/threshold_methods.py:978](../tabs/threshold_methods.py#L978) — `save_as_preset`, 18 строк
- [tabs/yolo_test_aeb.py:1331](../tabs/yolo_test_aeb.py#L1331) — `save_as_preset`, 18 строк

## 72. _segmentation_to_segment

- [utils/coco_io.py:28](../utils/coco_io.py#L28) — `_segmentation_to_segment`, 18 строк
- [utils_new/dataset_coco_new.py:30](../utils_new/dataset_coco_new.py#L30) — `_segmentation_to_segment`, 18 строк

## 73. detect_bit_type

- [utils/dataset_validation.py:49](../utils/dataset_validation.py#L49) — `detect_bit_type`, 18 строк
- [utils_new/dataset_image_stats_new.py:35](../utils_new/dataset_image_stats_new.py#L35) — `detect_bit_type`, 18 строк

## 74. point_in_polygon

- [utils/smart_view.py:378](../utils/smart_view.py#L378) — `point_in_polygon`, 18 строк
- [utils_new/smart_view_new.py:631](../utils_new/smart_view_new.py#L631) — `point_in_polygon`, 18 строк

## 75. set_drawing_tool

- [utils/smart_view.py:207](../utils/smart_view.py#L207) — `set_drawing_tool`, 18 строк
- [utils_new/smart_view_new.py:290](../utils_new/smart_view_new.py#L290) — `set_drawing_tool`, 18 строк

## 76. format_object_for_list

- [utils_new/annotation_processing.py:1214](../utils_new/annotation_processing.py#L1214) — `format_object_for_list`, 18 строк
- [utils_ops/segmentation_ops.py:517](../utils_ops/segmentation_ops.py#L517) — `format_object_for_list`, 18 строк

## 77. scan_detection_pairs

- [utils/dataset_validation.py:456](../utils/dataset_validation.py#L456) — `scan_detection_pairs`, 17 строк
- [utils_new/dataset_pair_tools_new.py:22](../utils_new/dataset_pair_tools_new.py#L22) — `scan_detection_pairs`, 17 строк

## 78. get

- [utils/image_io.py:1018](../utils/image_io.py#L1018) — `get`, 17 строк
- [utils_new/dataset_files_new.py:245](../utils_new/dataset_files_new.py#L245) — `get`, 17 строк

## 79. _find_hit

- [utils/smart_view.py:458](../utils/smart_view.py#L458) — `_find_hit`, 17 строк
- [utils_new/smart_view_new.py:711](../utils_new/smart_view_new.py#L711) — `_find_hit`, 17 строк

## 80. load_from_yaml

- [utils/dataset_validation.py:943](../utils/dataset_validation.py#L943) — `load_from_yaml`, 16 строк
- [utils_new/dataset_loader_new.py:169](../utils_new/dataset_loader_new.py#L169) — `load_from_yaml`, 16 строк

## 81. _load_sync

- [utils/image_io.py:993](../utils/image_io.py#L993) — `_load_sync`, 16 строк
- [utils_new/dataset_files_new.py:220](../utils_new/dataset_files_new.py#L220) — `_load_sync`, 16 строк

## 82. _setup_drawing_mode

- [utils/smart_view.py:171](../utils/smart_view.py#L171) — `_setup_drawing_mode`, 16 строк
- [utils_new/smart_view_new.py:254](../utils_new/smart_view_new.py#L254) — `_setup_drawing_mode`, 16 строк

## 83. set_callbacks

- [utils/smart_view.py:285](../utils/smart_view.py#L285) — `set_callbacks`, 16 строк
- [utils_new/smart_view_new.py:493](../utils_new/smart_view_new.py#L493) — `set_callbacks`, 16 строк

## 84. isodata_threshold

- [utils_new/threshold_processing.py:56](../utils_new/threshold_processing.py#L56) — `isodata_threshold`, 16 строк
- [utils_ops/segmentation_ops.py:182](../utils_ops/segmentation_ops.py#L182) — `isodata_threshold`, 16 строк

## 85. _ask_coco_source

- [utils/dataset_validation.py:1031](../utils/dataset_validation.py#L1031) — `_ask_coco_source`, 15 строк
- [utils_new/dataset_loader_new.py:261](../utils_new/dataset_loader_new.py#L261) — `_ask_coco_source`, 15 строк

## 86. _sync_all_states

- [utils/dataset_validation.py:839](../utils/dataset_validation.py#L839) — `_sync_all_states`, 15 строк
- [utils_new/dataset_loader_new.py:65](../utils_new/dataset_loader_new.py#L65) — `_sync_all_states`, 15 строк

## 87. detect_color_type

- [utils/dataset_validation.py:32](../utils/dataset_validation.py#L32) — `detect_color_type`, 15 строк
- [utils_new/dataset_image_stats_new.py:18](../utils_new/dataset_image_stats_new.py#L18) — `detect_color_type`, 15 строк

## 88. __init__

- [utils/image_io.py:970](../utils/image_io.py#L970) — `__init__`, 15 строк
- [utils_new/dataset_files_new.py:197](../utils_new/dataset_files_new.py#L197) — `__init__`, 15 строк

## 89. _clear_edit_state

- [utils/smart_view.py:242](../utils/smart_view.py#L242) — `_clear_edit_state`, 15 строк
- [utils_new/smart_view_new.py:325](../utils_new/smart_view_new.py#L325) — `_clear_edit_state`, 15 строк

## 90. _point_to_segment_dist

- [utils/smart_view.py:398](../utils/smart_view.py#L398) — `_point_to_segment_dist`, 15 строк
- [utils_new/smart_view_new.py:651](../utils_new/smart_view_new.py#L651) — `_point_to_segment_dist`, 15 строк

## 91. set_edit_mode

- [utils/smart_view.py:226](../utils/smart_view.py#L226) — `set_edit_mode`, 15 строк
- [utils_new/smart_view_new.py:309](../utils_new/smart_view_new.py#L309) — `set_edit_mode`, 15 строк

## 92. signal_handler

- [utils_training/train_unet_worker.py:140](../utils_training/train_unet_worker.py#L140) — `signal_handler`, 15 строк
- [utils_training/train_unet_worker_рабочий.py:134](../utils_training/train_unet_worker_рабочий.py#L134) — `signal_handler`, 15 строк

## 93. _resolve_label_path

- [tabs/layout_dataset.py:117](../tabs/layout_dataset.py#L117) — `_resolve_label_path`, 14 строк
- [tabs/viewing_dataset.py:126](../tabs/viewing_dataset.py#L126) — `_resolve_label_path`, 14 строк

## 94. validate_all

- [utils/dataset_validation.py:194](../utils/dataset_validation.py#L194) — `validate_all`, 14 строк
- [utils_new/dataset_pair_validator_new.py:78](../utils_new/dataset_pair_validator_new.py#L78) — `validate_all`, 14 строк

## 95. prefetch

- [utils/image_io.py:1063](../utils/image_io.py#L1063) — `prefetch`, 14 строк
- [utils_new/dataset_files_new.py:290](../utils_new/dataset_files_new.py#L290) — `prefetch`, 14 строк

## 96. _continue_drawing

- [utils/smart_view.py:1572](../utils/smart_view.py#L1572) — `_continue_drawing`, 14 строк
- [utils_new/smart_view_new.py:1828](../utils_new/smart_view_new.py#L1828) — `_continue_drawing`, 14 строк

## 97. forward

- [utils_aeb/auto-aeb_def.py:92](../utils_aeb/auto-aeb_def.py#L92) — `forward`, 14 строк
- [utils_aeb/auto-aeb_def_new.py:106](../utils_aeb/auto-aeb_def_new.py#L106) — `forward`, 14 строк
- [utils_aeb/auto-aeb_old.py:95](../utils_aeb/auto-aeb_old.py#L95) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet.py:98](../utils_aeb/auto-aeb_unet.py#L98) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:190](../utils_aeb/auto-aeb_unet_with_grad.py#L190) — `forward`, 14 строк

## 98. row_adaptive_threshold

- [utils_new/threshold_processing.py:95](../utils_new/threshold_processing.py#L95) — `row_adaptive_threshold`, 14 строк
- [utils_ops/segmentation_ops.py:223](../utils_ops/segmentation_ops.py#L223) — `row_adaptive_threshold`, 14 строк

## 99. collect_images

- [cluster/canonicalize_crops.py:108](../cluster/canonicalize_crops.py#L108) — `collect_images`, 13 строк
- [cluster/canonicalize_crops_method2.py:125](../cluster/canonicalize_crops_method2.py#L125) — `collect_images`, 13 строк

## 100. on_resize_mode_changed

- [tabs/gradient_methods.py:603](../tabs/gradient_methods.py#L603) — `on_resize_mode_changed`, 13 строк
- [tabs/interactive_methods.py:880](../tabs/interactive_methods.py#L880) — `on_resize_mode_changed`, 13 строк
- [tabs/layout_dataset.py:464](../tabs/layout_dataset.py#L464) — `on_resize_mode_changed`, 13 строк
- [tabs/threshold_methods.py:758](../tabs/threshold_methods.py#L758) — `on_resize_mode_changed`, 13 строк

## 101. _log_exception

- [tabs/layout_dataset.py:574](../tabs/layout_dataset.py#L574) — `_log_exception`, 13 строк
- [tabs/viewing_dataset.py:265](../tabs/viewing_dataset.py#L265) — `_log_exception`, 13 строк

## 102. shutdown

- [utils/image_io.py:1091](../utils/image_io.py#L1091) — `shutdown`, 13 строк
- [utils_new/dataset_files_new.py:318](../utils_new/dataset_files_new.py#L318) — `shutdown`, 13 строк

## 103. select_pairs_for_8bit_png

- [utils/dataset_validation.py:557](../utils/dataset_validation.py#L557) — `select_pairs_for_8bit_png`, 12 строк
- [utils_new/dataset_pair_tools_new.py:125](../utils_new/dataset_pair_tools_new.py#L125) — `select_pairs_for_8bit_png`, 12 строк

## 104. select_pairs_for_conversion

- [utils/dataset_validation.py:516](../utils/dataset_validation.py#L516) — `select_pairs_for_conversion`, 12 строк
- [utils_new/dataset_pair_tools_new.py:82](../utils_new/dataset_pair_tools_new.py#L82) — `select_pairs_for_conversion`, 12 строк

## 105. trim_around

- [utils/image_io.py:1078](../utils/image_io.py#L1078) — `trim_around`, 12 строк
- [utils_new/dataset_files_new.py:305](../utils_new/dataset_files_new.py#L305) — `trim_around`, 12 строк

## 106. _finish_drawing

- [utils/smart_view.py:1587](../utils/smart_view.py#L1587) — `_finish_drawing`, 12 строк
- [utils_new/smart_view_new.py:1843](../utils_new/smart_view_new.py#L1843) — `_finish_drawing`, 12 строк

## 107. _setup_edit_mode

- [utils/smart_view.py:188](../utils/smart_view.py#L188) — `_setup_edit_mode`, 12 строк
- [utils_new/smart_view_new.py:271](../utils_new/smart_view_new.py#L271) — `_setup_edit_mode`, 12 строк

## 108. __init__

- [utils_aeb/auto-aeb_def.py:177](../utils_aeb/auto-aeb_def.py#L177) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:180](../utils_aeb/auto-aeb_old.py#L180) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:36](../utils_aeb/yolo_aeb_utils.py#L36) — `__init__`, 12 строк

## 109. __init__

- [utils_aeb/auto-aeb_def.py:195](../utils_aeb/auto-aeb_def.py#L195) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:198](../utils_aeb/auto-aeb_old.py#L198) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:54](../utils_aeb/yolo_aeb_utils.py#L54) — `__init__`, 12 строк

## 110. __init__

- [utils_aeb/auto-aeb_def_new.py:201](../utils_aeb/auto-aeb_def_new.py#L201) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:131](../utils_aeb/yolo_aeb_utils.py#L131) — `__init__`, 12 строк

## 111. __init__

- [utils_aeb/auto-aeb_def_new.py:219](../utils_aeb/auto-aeb_def_new.py#L219) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:149](../utils_aeb/yolo_aeb_utils.py#L149) — `__init__`, 12 строк

## 112. __init__

- [utils_aeb/auto-aeb_unet.py:312](../utils_aeb/auto-aeb_unet.py#L312) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:319](../utils_aeb/yolo_aeb_utils.py#L319) — `__init__`, 12 строк

## 113. get_optimizer

- [utils_training/train_unet_worker.py:376](../utils_training/train_unet_worker.py#L376) — `get_optimizer`, 12 строк
- [utils_training/train_unet_worker_рабочий.py:381](../utils_training/train_unet_worker_рабочий.py#L381) — `get_optimizer`, 12 строк

## 114. delete_current_preset

- [tabs/deep_learning_methods.py:410](../tabs/deep_learning_methods.py#L410) — `delete_current_preset`, 11 строк
- [tabs/gradient_methods.py:131](../tabs/gradient_methods.py#L131) — `delete_current_preset`, 11 строк
- [tabs/threshold_methods.py:997](../tabs/threshold_methods.py#L997) — `delete_current_preset`, 11 строк
- [tabs/traditional_ml_methods.py:325](../tabs/traditional_ml_methods.py#L325) — `delete_current_preset`, 11 строк
- [tabs/yolo_test_aeb.py:1350](../tabs/yolo_test_aeb.py#L1350) — `delete_current_preset`, 11 строк

## 115. closeEvent

- [tabs/gradient_methods.py:172](../tabs/gradient_methods.py#L172) — `closeEvent`, 11 строк
- [tabs/threshold_methods.py:293](../tabs/threshold_methods.py#L293) — `closeEvent`, 11 строк

## 116. to_display_bgr

- [utils/dataset_validation.py:82](../utils/dataset_validation.py#L82) — `to_display_bgr`, 11 строк
- [utils/dataset_validation.py:108](../utils/dataset_validation.py#L108) — `to_bgr3`, 11 строк
- [utils_new/dataset_image_stats_new.py:68](../utils_new/dataset_image_stats_new.py#L68) — `to_display_bgr`, 11 строк
- [utils_new/dataset_image_stats_new.py:94](../utils_new/dataset_image_stats_new.py#L94) — `to_bgr3`, 11 строк

## 117. to_gray

- [utils/dataset_validation.py:95](../utils/dataset_validation.py#L95) — `to_gray`, 11 строк
- [utils_new/dataset_image_stats_new.py:81](../utils_new/dataset_image_stats_new.py#L81) — `to_gray`, 11 строк

## 118. to_uint8

- [utils/dataset_validation.py:69](../utils/dataset_validation.py#L69) — `to_uint8`, 11 строк
- [utils_new/dataset_image_stats_new.py:55](../utils_new/dataset_image_stats_new.py#L55) — `to_uint8`, 11 строк

## 119. _hit_test_any

- [utils/smart_view.py:446](../utils/smart_view.py#L446) — `_hit_test_any`, 11 строк
- [utils_new/smart_view_new.py:699](../utils_new/smart_view_new.py#L699) — `_hit_test_any`, 11 строк

## 120. _hit_test_detect

- [utils/smart_view.py:422](../utils/smart_view.py#L422) — `_hit_test_detect`, 11 строк
- [utils_new/smart_view_new.py:675](../utils_new/smart_view_new.py#L675) — `_hit_test_detect`, 11 строк

## 121. _scene_per_px

- [utils/smart_view.py:342](../utils/smart_view.py#L342) — `_scene_per_px`, 11 строк
- [utils_new/smart_view_new.py:592](../utils_new/smart_view_new.py#L592) — `_scene_per_px`, 11 строк

## 122. __getitem__

- [utils_aeb/auto-aeb_def.py:151](../utils_aeb/auto-aeb_def.py#L151) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:165](../utils_aeb/auto-aeb_def_new.py#L165) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_old.py:154](../utils_aeb/auto-aeb_old.py#L154) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_unet.py:183](../utils_aeb/auto-aeb_unet.py#L183) — `__getitem__`, 11 строк

## 123. __init__

- [utils_aeb/auto-aeb_def.py:232](../utils_aeb/auto-aeb_def.py#L232) — `__init__`, 11 строк
- [utils_aeb/auto-aeb_old.py:235](../utils_aeb/auto-aeb_old.py#L235) — `__init__`, 11 строк
- [utils_aeb/yolo_aeb_utils.py:91](../utils_aeb/yolo_aeb_utils.py#L91) — `__init__`, 11 строк

## 124. _fit

- [utils_aeb/auto-aeb_def.py:136](../utils_aeb/auto-aeb_def.py#L136) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:150](../utils_aeb/auto-aeb_def_new.py#L150) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_old.py:139](../utils_aeb/auto-aeb_old.py#L139) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet.py:168](../utils_aeb/auto-aeb_unet.py#L168) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:263](../utils_aeb/auto-aeb_unet_with_grad.py#L263) — `_fit`, 11 строк

## 125. __init__

- [utils_aeb/auto-aeb_unet.py:291](../utils_aeb/auto-aeb_unet.py#L291) — `__init__`, 11 строк
- [utils_aeb/yolo_aeb_utils.py:298](../utils_aeb/yolo_aeb_utils.py#L298) — `__init__`, 11 строк

## 126. imwrite_safe

- [cluster/canonicalize_crops.py:96](../cluster/canonicalize_crops.py#L96) — `imwrite_safe`, 10 строк
- [cluster/canonicalize_crops_method2.py:113](../cluster/canonicalize_crops_method2.py#L113) — `imwrite_safe`, 10 строк

## 127. closeEvent

- [tabs/deep_learning_methods.py:1122](../tabs/deep_learning_methods.py#L1122) — `closeEvent`, 10 строк
- [tabs/interactive_methods.py:1001](../tabs/interactive_methods.py#L1001) — `closeEvent`, 10 строк

## 128. on_object_selection_changed

- [tabs/deep_learning_methods.py:1079](../tabs/deep_learning_methods.py#L1079) — `on_object_selection_changed`, 10 строк
- [tabs/gradient_methods.py:689](../tabs/gradient_methods.py#L689) — `on_object_selection_changed`, 10 строк
- [tabs/interactive_methods.py:946](../tabs/interactive_methods.py#L946) — `on_object_selection_changed`, 10 строк
- [tabs/threshold_methods.py:937](../tabs/threshold_methods.py#L937) — `on_object_selection_changed`, 10 строк
- [tabs/traditional_ml_methods.py:1176](../tabs/traditional_ml_methods.py#L1176) — `on_object_selection_changed`, 10 строк

## 129. _log_exception

- [utils/dataset_validation.py:881](../utils/dataset_validation.py#L881) — `_log_exception`, 10 строк
- [utils_new/dataset_loader_new.py:107](../utils_new/dataset_loader_new.py#L107) — `_log_exception`, 10 строк

## 130. _on_file_list_double_clicked

- [utils/dataset_validation.py:901](../utils/dataset_validation.py#L901) — `_on_file_list_double_clicked`, 10 строк
- [utils_new/dataset_loader_new.py:127](../utils_new/dataset_loader_new.py#L127) — `_on_file_list_double_clicked`, 10 строк

## 131. _on_input_format_changed

- [utils/dataset_validation.py:1564](../utils/dataset_validation.py#L1564) — `_on_input_format_changed`, 10 строк
- [utils_new/dataset_loader_new.py:726](../utils_new/dataset_loader_new.py#L726) — `_on_input_format_changed`, 10 строк

## 132. _finite

- [utils/smart_view.py:523](../utils/smart_view.py#L523) — `_finite`, 10 строк
- [utils_new/smart_view_new.py:776](../utils_new/smart_view_new.py#L776) — `_finite`, 10 строк

## 133. _resolve_relative_to_yaml

- [utils/yaml_io.py:20](../utils/yaml_io.py#L20) — `_resolve_relative_to_yaml`, 10 строк
- [utils_new/dataset_yaml_new.py:8](../utils_new/dataset_yaml_new.py#L8) — `_resolve_relative_to_yaml`, 10 строк

## 134. get_scheduler

- [utils_training/train_unet_worker.py:391](../utils_training/train_unet_worker.py#L391) — `get_scheduler`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:396](../utils_training/train_unet_worker_рабочий.py#L396) — `get_scheduler`, 10 строк

## 135. save_args_yaml

- [utils_training/train_unet_worker.py:522](../utils_training/train_unet_worker.py#L522) — `save_args_yaml`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:508](../utils_training/train_unet_worker_рабочий.py#L508) — `save_args_yaml`, 10 строк

## 136. apply_transform

- [cluster/canonicalize_crops.py:142](../cluster/canonicalize_crops.py#L142) — `apply_transform`, 9 строк
- [cluster/canonicalize_crops_method2.py:162](../cluster/canonicalize_crops_method2.py#L162) — `apply_transform`, 9 строк

## 137. apply_transform

- [cluster/extract_embeddings.py:114](../cluster/extract_embeddings.py#L114) — `apply_transform`, 9 строк
- [cluster/test_new_image.py:187](../cluster/test_new_image.py#L187) — `apply_transform`, 9 строк

## 138. load_images_from_dialog

- [tabs/gradient_methods.py:496](../tabs/gradient_methods.py#L496) — `load_images_from_dialog`, 9 строк
- [tabs/interactive_methods.py:778](../tabs/interactive_methods.py#L778) — `load_images_from_dialog`, 9 строк
- [tabs/layout_dataset.py:445](../tabs/layout_dataset.py#L445) — `load_images`, 9 строк
- [tabs/threshold_methods.py:599](../tabs/threshold_methods.py#L599) — `load_images_from_dialog`, 9 строк

## 139. _update_color_stats_label

- [utils/dataset_validation.py:1395](../utils/dataset_validation.py#L1395) — `_update_color_stats_label`, 9 строк
- [utils_new/dataset_loader_new.py:557](../utils_new/dataset_loader_new.py#L557) — `_update_color_stats_label`, 9 строк

## 140. _update_pair_count_label

- [utils/dataset_validation.py:1842](../utils/dataset_validation.py#L1842) — `_update_pair_count_label`, 9 строк
- [utils_new/dataset_loader_new.py:961](../utils_new/dataset_loader_new.py#L961) — `_update_pair_count_label`, 9 строк

## 141. __init__

- [utils_aeb/auto-aeb_def.py:213](../utils_aeb/auto-aeb_def.py#L213) — `__init__`, 9 строк
- [utils_aeb/auto-aeb_old.py:216](../utils_aeb/auto-aeb_old.py#L216) — `__init__`, 9 строк
- [utils_aeb/yolo_aeb_utils.py:72](../utils_aeb/yolo_aeb_utils.py#L72) — `__init__`, 9 строк

## 142. collect_images

- [utils_aeb/auto-aeb_def.py:164](../utils_aeb/auto-aeb_def.py#L164) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_def_new.py:178](../utils_aeb/auto-aeb_def_new.py#L178) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_old.py:167](../utils_aeb/auto-aeb_old.py#L167) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet.py:196](../utils_aeb/auto-aeb_unet.py#L196) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:387](../utils_aeb/auto-aeb_unet_with_grad.py#L387) — `collect_images`, 9 строк

## 143. forward

- [utils_aeb/auto-aeb_unet.py:125](../utils_aeb/auto-aeb_unet.py#L125) — `forward`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:216](../utils_aeb/auto-aeb_unet_with_grad.py#L216) — `forward`, 9 строк

## 144. median_blur

- [utils_aeb/auto-aeb_unet_with_grad.py:125](../utils_aeb/auto-aeb_unet_with_grad.py#L125) — `median_blur`, 9 строк
- [utils_aeb/yolo_aeb_utils.py:534](../utils_aeb/yolo_aeb_utils.py#L534) — `_median_blur_torch`, 9 строк
