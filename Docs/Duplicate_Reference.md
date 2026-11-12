# Подтверждённые структурные копии

[Разбор и рекомендации](Duplication.md) · [Справочник](Module_Reference.md)

Автоматическое приложение: одинаковые AST функций/методов в разных файлах, от 9 строк исходника. Имя функции, docstring и номера строк игнорируются; аргументы и тело сохраняются. Сигнатуры и окружающие импорты проверяйте перед объединением. Это не поиск похожих алгоритмов и не доказательство взаимозаменяемости классов.

Групп совпадений: **57**.

## 1. main

- [utils_aeb/auto-aeb_def.py:370](../utils_aeb/auto-aeb_def.py#L370) — `main`, 197 строк
- [utils_aeb/auto-aeb_old.py:372](../utils_aeb/auto-aeb_old.py#L372) — `main`, 197 строк

## 2. __getitem__

- [utils_training/train_unet_worker.py:258](../utils_training/train_unet_worker.py#L258) — `__getitem__`, 72 строк
- [utils_training/train_unet_worker_рабочий.py:260](../utils_training/train_unet_worker_рабочий.py#L260) — `__getitem__`, 75 строк

## 3. compute_metrics

- [utils_training/train_unet_worker.py:399](../utils_training/train_unet_worker.py#L399) — `compute_metrics`, 72 строк
- [utils_training/train_unet_worker_рабочий.py:404](../utils_training/train_unet_worker_рабочий.py#L404) — `compute_metrics`, 72 строк

## 4. _resize_pair

- [utils_training/train_unet_worker.py:202](../utils_training/train_unet_worker.py#L202) — `_resize_pair`, 55 строк
- [utils_training/train_unet_worker_рабочий.py:195](../utils_training/train_unet_worker_рабочий.py#L195) — `_resize_pair`, 64 строк

## 5. apply_threshold_method

- [utils_new/threshold_processing.py:135](../utils_new/threshold_processing.py#L135) — `apply_threshold_method`, 59 строк
- [utils_ops/segmentation_ops.py:568](../utils_ops/segmentation_ops.py#L568) — `apply_threshold_method`, 59 строк

## 6. save_confusion_matrix

- [utils_training/train_unet_worker.py:640](../utils_training/train_unet_worker.py#L640) — `save_confusion_matrix`, 59 строк
- [utils_training/train_unet_worker_рабочий.py:601](../utils_training/train_unet_worker_рабочий.py#L601) — `save_confusion_matrix`, 55 строк

## 7. save_pr_curve

- [utils_training/train_unet_worker.py:581](../utils_training/train_unet_worker.py#L581) — `save_pr_curve`, 57 строк
- [utils_training/train_unet_worker_рабочий.py:558](../utils_training/train_unet_worker_рабочий.py#L558) — `save_pr_curve`, 42 строк

## 8. check_dataset_masks_consistency

- [utils_training/train_unet_worker.py:815](../utils_training/train_unet_worker.py#L815) — `check_dataset_masks_consistency`, 47 строк
- [utils_training/train_unet_worker_рабочий.py:746](../utils_training/train_unet_worker_рабочий.py#L746) — `check_dataset_masks_consistency`, 52 строк

## 9. train_one_epoch

- [utils_aeb/auto-aeb_def.py:278](../utils_aeb/auto-aeb_def.py#L278) — `train_one_epoch`, 49 строк
- [utils_aeb/auto-aeb_old.py:280](../utils_aeb/auto-aeb_old.py#L280) — `train_one_epoch`, 49 строк

## 10. _default_settings

- [utils_auto_research/gradient_auto_research.py:167](../utils_auto_research/gradient_auto_research.py#L167) — `_default_settings`, 32 строк
- [utils_auto_research/gradient_auto_settings_dialog.py:85](../utils_auto_research/gradient_auto_settings_dialog.py#L85) — `_default_settings`, 44 строк

## 11. __init__

- [utils_training/train_unet_worker.py:157](../utils_training/train_unet_worker.py#L157) — `__init__`, 41 строк
- [utils_training/train_unet_worker_рабочий.py:150](../utils_training/train_unet_worker_рабочий.py#L150) — `__init__`, 41 строк

## 12. _validate_unified_annotation

- [tabs/layout_dataset.py:82](../tabs/layout_dataset.py#L82) — `_validate_unified_annotation`, 40 строк
- [tabs/viewing_dataset.py:94](../tabs/viewing_dataset.py#L94) — `_validate_unified_annotation`, 40 строк

## 13. evaluate

- [utils_aeb/auto-aeb_def.py:330](../utils_aeb/auto-aeb_def.py#L330) — `evaluate`, 38 строк
- [utils_aeb/auto-aeb_old.py:332](../utils_aeb/auto-aeb_old.py#L332) — `evaluate`, 38 строк

## 14. get_loss_function

- [utils_training/train_unet_worker.py:333](../utils_training/train_unet_worker.py#L333) — `get_loss_function`, 35 строк
- [utils_training/train_unet_worker_рабочий.py:338](../utils_training/train_unet_worker_рабочий.py#L338) — `get_loss_function`, 35 строк

## 15. __init__

- [utils_aeb/auto-aeb_unet.py:246](../utils_aeb/auto-aeb_unet.py#L246) — `__init__`, 33 строк
- [utils_aeb/yolo_aeb_utils.py:259](../utils_aeb/yolo_aeb_utils.py#L259) — `__init__`, 33 строк

## 16. save_training_plots

- [utils_training/train_unet_worker.py:537](../utils_training/train_unet_worker.py#L537) — `save_training_plots`, 33 строк
- [utils_training/train_unet_worker_рабочий.py:522](../utils_training/train_unet_worker_рабочий.py#L522) — `save_training_plots`, 27 строк

## 17. __init__

- [utils_aeb/auto-aeb_unet.py:208](../utils_aeb/auto-aeb_unet.py#L208) — `__init__`, 28 строк
- [utils_aeb/yolo_aeb_utils.py:221](../utils_aeb/yolo_aeb_utils.py#L221) — `__init__`, 28 строк

## 18. format_annotation_display

- [utils_new/annotation_processing.py:1187](../utils_new/annotation_processing.py#L1187) — `format_annotation_display`, 28 строк
- [utils_ops/segmentation_ops.py:419](../utils_ops/segmentation_ops.py#L419) — `format_annotation_display`, 28 строк

## 19. read_image_bgr

- [cluster/canonicalize_crops.py:74](../cluster/canonicalize_crops.py#L74) — `read_image_bgr`, 21 строк
- [cluster/canonicalize_crops_method2.py:91](../cluster/canonicalize_crops_method2.py#L91) — `read_image_bgr`, 21 строк
- [cluster/save_yolo_object_to_path.py:58](../cluster/save_yolo_object_to_path.py#L58) — `read_image`, 27 строк

## 20. _load_coco_into_labeler

- [tabs/layout_dataset.py:668](../tabs/layout_dataset.py#L668) — `_load_coco_into_labeler`, 26 строк
- [tabs/viewing_dataset.py:1560](../tabs/viewing_dataset.py#L1560) — `_load_coco_into_viewer`, 25 строк

## 21. normalize_method_name

- [utils_new/threshold_processing.py:110](../utils_new/threshold_processing.py#L110) — `normalize_method_name`, 24 строк
- [utils_ops/segmentation_ops.py:542](../utils_ops/segmentation_ops.py#L542) — `normalize_method_name`, 24 строк

## 22. read_image_rgb

- [cluster/cluster_embeddings.py:90](../cluster/cluster_embeddings.py#L90) — `read_image_rgb`, 23 строк
- [cluster/extract_embeddings.py:58](../cluster/extract_embeddings.py#L58) — `read_image_rgb`, 23 строк

## 23. __init__

- [utils_aeb/auto-aeb_def.py:110](../utils_aeb/auto-aeb_def.py#L110) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_def_new.py:123](../utils_aeb/auto-aeb_def_new.py#L123) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_old.py:112](../utils_aeb/auto-aeb_old.py#L112) — `__init__`, 23 строк

## 24. _detect_annotation_to_obb

- [tabs/layout_dataset.py:127](../tabs/layout_dataset.py#L127) — `_detect_annotation_to_obb`, 22 строк
- [tabs/viewing_dataset.py:139](../tabs/viewing_dataset.py#L139) — `_detect_annotation_to_obb`, 22 строк

## 25. background_symmetry_threshold

- [utils_new/threshold_processing.py:73](../utils_new/threshold_processing.py#L73) — `background_symmetry_threshold`, 21 строк
- [utils_ops/segmentation_ops.py:204](../utils_ops/segmentation_ops.py#L204) — `background_symmetry_threshold`, 21 строк

## 26. _start_annotation_validation

- [tabs/layout_dataset.py:861](../tabs/layout_dataset.py#L861) — `_start_annotation_validation`, 20 строк
- [tabs/viewing_dataset.py:855](../tabs/viewing_dataset.py#L855) — `_start_annotation_validation`, 20 строк

## 27. run

- [tabs/layout_dataset.py:181](../tabs/layout_dataset.py#L181) — `run`, 18 строк
- [tabs/viewing_dataset.py:193](../tabs/viewing_dataset.py#L193) — `run`, 18 строк

## 28. format_object_for_list

- [utils_new/annotation_processing.py:1216](../utils_new/annotation_processing.py#L1216) — `format_object_for_list`, 18 строк
- [utils_ops/segmentation_ops.py:521](../utils_ops/segmentation_ops.py#L521) — `format_object_for_list`, 18 строк

## 29. isodata_threshold

- [utils_new/threshold_processing.py:56](../utils_new/threshold_processing.py#L56) — `isodata_threshold`, 16 строк
- [utils_ops/segmentation_ops.py:186](../utils_ops/segmentation_ops.py#L186) — `isodata_threshold`, 16 строк

## 30. signal_handler

- [utils_training/train_unet_worker.py:134](../utils_training/train_unet_worker.py#L134) — `signal_handler`, 15 строк
- [utils_training/train_unet_worker_рабочий.py:128](../utils_training/train_unet_worker_рабочий.py#L128) — `signal_handler`, 15 строк

## 31. _confirm_label_save

- [tabs/layout_dataset.py:2074](../tabs/layout_dataset.py#L2074) — `_confirm_label_save`, 14 строк
- [tabs/viewing_dataset.py:988](../tabs/viewing_dataset.py#L988) — `_confirm_label_save`, 14 строк

## 32. _resolve_label_path

- [tabs/layout_dataset.py:166](../tabs/layout_dataset.py#L166) — `_resolve_label_path`, 14 строк
- [tabs/viewing_dataset.py:178](../tabs/viewing_dataset.py#L178) — `_resolve_label_path`, 14 строк

## 33. forward

- [utils_aeb/auto-aeb_def.py:90](../utils_aeb/auto-aeb_def.py#L90) — `forward`, 14 строк
- [utils_aeb/auto-aeb_def_new.py:103](../utils_aeb/auto-aeb_def_new.py#L103) — `forward`, 14 строк
- [utils_aeb/auto-aeb_old.py:92](../utils_aeb/auto-aeb_old.py#L92) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet.py:95](../utils_aeb/auto-aeb_unet.py#L95) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:185](../utils_aeb/auto-aeb_unet_with_grad.py#L185) — `forward`, 14 строк

## 34. row_adaptive_threshold

- [utils_new/threshold_processing.py:95](../utils_new/threshold_processing.py#L95) — `row_adaptive_threshold`, 14 строк
- [utils_ops/segmentation_ops.py:227](../utils_ops/segmentation_ops.py#L227) — `row_adaptive_threshold`, 14 строк

## 35. collect_images

- [cluster/canonicalize_crops.py:109](../cluster/canonicalize_crops.py#L109) — `collect_images`, 13 строк
- [cluster/canonicalize_crops_method2.py:126](../cluster/canonicalize_crops_method2.py#L126) — `collect_images`, 13 строк

## 36. _log_exception

- [tabs/layout_dataset.py:641](../tabs/layout_dataset.py#L641) — `_log_exception`, 13 строк
- [tabs/viewing_dataset.py:319](../tabs/viewing_dataset.py#L319) — `_log_exception`, 13 строк

## 37. __getattr__

- [import_libs_external.py:138](../import_libs_external.py#L138) — `__getattr__`, 12 строк
- [import_libs_internal.py:268](../import_libs_internal.py#L268) — `__getattr__`, 12 строк

## 38. __init__

- [utils_aeb/auto-aeb_def.py:175](../utils_aeb/auto-aeb_def.py#L175) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:177](../utils_aeb/auto-aeb_old.py#L177) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:39](../utils_aeb/yolo_aeb_utils.py#L39) — `__init__`, 12 строк

## 39. __init__

- [utils_aeb/auto-aeb_def.py:193](../utils_aeb/auto-aeb_def.py#L193) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:195](../utils_aeb/auto-aeb_old.py#L195) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:57](../utils_aeb/yolo_aeb_utils.py#L57) — `__init__`, 12 строк

## 40. __init__

- [utils_aeb/auto-aeb_def_new.py:198](../utils_aeb/auto-aeb_def_new.py#L198) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:134](../utils_aeb/yolo_aeb_utils.py#L134) — `__init__`, 12 строк

## 41. __init__

- [utils_aeb/auto-aeb_def_new.py:216](../utils_aeb/auto-aeb_def_new.py#L216) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:152](../utils_aeb/yolo_aeb_utils.py#L152) — `__init__`, 12 строк

## 42. __init__

- [utils_aeb/auto-aeb_unet.py:309](../utils_aeb/auto-aeb_unet.py#L309) — `__init__`, 12 строк
- [utils_aeb/yolo_aeb_utils.py:322](../utils_aeb/yolo_aeb_utils.py#L322) — `__init__`, 12 строк

## 43. get_optimizer

- [utils_training/train_unet_worker.py:371](../utils_training/train_unet_worker.py#L371) — `get_optimizer`, 12 строк
- [utils_training/train_unet_worker_рабочий.py:376](../utils_training/train_unet_worker_рабочий.py#L376) — `get_optimizer`, 12 строк

## 44. __getitem__

- [utils_aeb/auto-aeb_def.py:149](../utils_aeb/auto-aeb_def.py#L149) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:162](../utils_aeb/auto-aeb_def_new.py#L162) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_old.py:151](../utils_aeb/auto-aeb_old.py#L151) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_unet.py:180](../utils_aeb/auto-aeb_unet.py#L180) — `__getitem__`, 11 строк

## 45. __init__

- [utils_aeb/auto-aeb_def.py:230](../utils_aeb/auto-aeb_def.py#L230) — `__init__`, 11 строк
- [utils_aeb/auto-aeb_old.py:232](../utils_aeb/auto-aeb_old.py#L232) — `__init__`, 11 строк
- [utils_aeb/yolo_aeb_utils.py:94](../utils_aeb/yolo_aeb_utils.py#L94) — `__init__`, 11 строк

## 46. _fit

- [utils_aeb/auto-aeb_def.py:134](../utils_aeb/auto-aeb_def.py#L134) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:147](../utils_aeb/auto-aeb_def_new.py#L147) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_old.py:136](../utils_aeb/auto-aeb_old.py#L136) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet.py:165](../utils_aeb/auto-aeb_unet.py#L165) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:258](../utils_aeb/auto-aeb_unet_with_grad.py#L258) — `_fit`, 11 строк

## 47. __init__

- [utils_aeb/auto-aeb_unet.py:288](../utils_aeb/auto-aeb_unet.py#L288) — `__init__`, 11 строк
- [utils_aeb/yolo_aeb_utils.py:301](../utils_aeb/yolo_aeb_utils.py#L301) — `__init__`, 11 строк

## 48. imwrite_safe

- [cluster/canonicalize_crops.py:97](../cluster/canonicalize_crops.py#L97) — `imwrite_safe`, 10 строк
- [cluster/canonicalize_crops_method2.py:114](../cluster/canonicalize_crops_method2.py#L114) — `imwrite_safe`, 10 строк

## 49. _resolve_save_path_for_img

- [tabs/layout_dataset.py:2063](../tabs/layout_dataset.py#L2063) — `_resolve_save_path_for_img`, 10 строк
- [tabs/viewing_dataset.py:977](../tabs/viewing_dataset.py#L977) — `_resolve_save_path_for_img`, 10 строк

## 50. get_scheduler

- [utils_training/train_unet_worker.py:386](../utils_training/train_unet_worker.py#L386) — `get_scheduler`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:391](../utils_training/train_unet_worker_рабочий.py#L391) — `get_scheduler`, 10 строк

## 51. save_args_yaml

- [utils_training/train_unet_worker.py:517](../utils_training/train_unet_worker.py#L517) — `save_args_yaml`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:503](../utils_training/train_unet_worker_рабочий.py#L503) — `save_args_yaml`, 10 строк

## 52. apply_transform

- [cluster/canonicalize_crops.py:143](../cluster/canonicalize_crops.py#L143) — `apply_transform`, 9 строк
- [cluster/canonicalize_crops_method2.py:163](../cluster/canonicalize_crops_method2.py#L163) — `apply_transform`, 9 строк

## 53. apply_transform

- [cluster/extract_embeddings.py:113](../cluster/extract_embeddings.py#L113) — `apply_transform`, 9 строк
- [cluster/test_new_image.py:197](../cluster/test_new_image.py#L197) — `apply_transform`, 9 строк

## 54. __init__

- [utils_aeb/auto-aeb_def.py:211](../utils_aeb/auto-aeb_def.py#L211) — `__init__`, 9 строк
- [utils_aeb/auto-aeb_old.py:213](../utils_aeb/auto-aeb_old.py#L213) — `__init__`, 9 строк
- [utils_aeb/yolo_aeb_utils.py:75](../utils_aeb/yolo_aeb_utils.py#L75) — `__init__`, 9 строк

## 55. collect_images

- [utils_aeb/auto-aeb_def.py:162](../utils_aeb/auto-aeb_def.py#L162) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_def_new.py:175](../utils_aeb/auto-aeb_def_new.py#L175) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_old.py:164](../utils_aeb/auto-aeb_old.py#L164) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet.py:193](../utils_aeb/auto-aeb_unet.py#L193) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:382](../utils_aeb/auto-aeb_unet_with_grad.py#L382) — `collect_images`, 9 строк

## 56. forward

- [utils_aeb/auto-aeb_unet.py:122](../utils_aeb/auto-aeb_unet.py#L122) — `forward`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:211](../utils_aeb/auto-aeb_unet_with_grad.py#L211) — `forward`, 9 строк

## 57. median_blur

- [utils_aeb/auto-aeb_unet_with_grad.py:120](../utils_aeb/auto-aeb_unet_with_grad.py#L120) — `median_blur`, 9 строк
- [utils_aeb/yolo_aeb_utils.py:537](../utils_aeb/yolo_aeb_utils.py#L537) — `_median_blur_torch`, 9 строк
