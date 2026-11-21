# Подтверждённые структурные копии

[Разбор и рекомендации](Duplication.md) · [Справочник](Module_Reference.md)

Автоматическое приложение: одинаковые AST функций/методов в разных файлах, от 9 строк исходника. Имя функции, docstring и номера строк игнорируются; аргументы и тело сохраняются. Сигнатуры и окружающие импорты проверяйте перед объединением. Это не поиск похожих алгоритмов и не доказательство взаимозаменяемости классов.

Групп совпадений: **50**.

## 1. main

- [utils_aeb/auto-aeb_def.py:370](../utils_aeb/auto-aeb_def.py#L370) — `main`, 197 строк
- [utils_aeb/auto-aeb_old.py:372](../utils_aeb/auto-aeb_old.py#L372) — `main`, 197 строк

## 2. __getitem__

- [utils_training/train_unet_worker.py:265](../utils_training/train_unet_worker.py#L265) — `__getitem__`, 64 строк
- [utils_training/train_unet_worker_рабочий.py:267](../utils_training/train_unet_worker_рабочий.py#L267) — `__getitem__`, 67 строк

## 3. _resize_pair

- [utils_training/train_unet_worker.py:209](../utils_training/train_unet_worker.py#L209) — `_resize_pair`, 55 строк
- [utils_training/train_unet_worker_рабочий.py:202](../utils_training/train_unet_worker_рабочий.py#L202) — `_resize_pair`, 64 строк

## 4. save_pr_curve

- [utils_training/train_unet_worker.py:510](../utils_training/train_unet_worker.py#L510) — `save_pr_curve`, 57 строк
- [utils_training/train_unet_worker_рабочий.py:487](../utils_training/train_unet_worker_рабочий.py#L487) — `save_pr_curve`, 42 строк

## 5. check_dataset_masks_consistency

- [utils_training/train_unet_worker.py:727](../utils_training/train_unet_worker.py#L727) — `check_dataset_masks_consistency`, 47 строк
- [utils_training/train_unet_worker_рабочий.py:658](../utils_training/train_unet_worker_рабочий.py#L658) — `check_dataset_masks_consistency`, 52 строк

## 6. train_one_epoch

- [utils_aeb/auto-aeb_def.py:278](../utils_aeb/auto-aeb_def.py#L278) — `train_one_epoch`, 49 строк
- [utils_aeb/auto-aeb_old.py:280](../utils_aeb/auto-aeb_old.py#L280) — `train_one_epoch`, 49 строк

## 7. save_confusion_matrix

- [utils_training/train_unet_worker.py:569](../utils_training/train_unet_worker.py#L569) — `save_confusion_matrix`, 44 строк
- [utils_training/train_unet_worker_рабочий.py:530](../utils_training/train_unet_worker_рабочий.py#L530) — `save_confusion_matrix`, 40 строк

## 8. __init__

- [utils_training/train_unet_worker.py:162](../utils_training/train_unet_worker.py#L162) — `__init__`, 43 строк
- [utils_training/train_unet_worker_рабочий.py:155](../utils_training/train_unet_worker_рабочий.py#L155) — `__init__`, 43 строк

## 9. __init__

- [utils_aeb/auto-aeb_unet.py:246](../utils_aeb/auto-aeb_unet.py#L246) — `__init__`, 33 строк
- [utils_new/segmentation/autoencoder.py:294](../utils_new/segmentation/autoencoder.py#L294) — `__init__`, 41 строк

## 10. _validate_unified_annotation

- [tabs/layout_dataset.py:76](../tabs/layout_dataset.py#L76) — `_validate_unified_annotation`, 40 строк
- [tabs/viewing_dataset.py:86](../tabs/viewing_dataset.py#L86) — `_validate_unified_annotation`, 40 строк

## 11. evaluate

- [utils_aeb/auto-aeb_def.py:330](../utils_aeb/auto-aeb_def.py#L330) — `evaluate`, 38 строк
- [utils_aeb/auto-aeb_old.py:332](../utils_aeb/auto-aeb_old.py#L332) — `evaluate`, 38 строк

## 12. _load_coco_into_labeler

- [tabs/layout_dataset.py:662](../tabs/layout_dataset.py#L662) — `_load_coco_into_labeler`, 37 строк
- [tabs/viewing_dataset.py:1649](../tabs/viewing_dataset.py#L1649) — `_load_coco_into_viewer`, 36 строк

## 13. __init__

- [utils_aeb/auto-aeb_unet.py:208](../utils_aeb/auto-aeb_unet.py#L208) — `__init__`, 28 строк
- [utils_new/segmentation/autoencoder.py:248](../utils_new/segmentation/autoencoder.py#L248) — `__init__`, 36 строк

## 14. get_loss_function

- [utils_training/train_unet_worker.py:332](../utils_training/train_unet_worker.py#L332) — `get_loss_function`, 35 строк
- [utils_training/train_unet_worker_рабочий.py:337](../utils_training/train_unet_worker_рабочий.py#L337) — `get_loss_function`, 35 строк

## 15. _start_annotation_validation

- [tabs/layout_dataset.py:889](../tabs/layout_dataset.py#L889) — `_start_annotation_validation`, 32 строк
- [tabs/viewing_dataset.py:875](../tabs/viewing_dataset.py#L875) — `_start_annotation_validation`, 32 строк

## 16. save_training_plots

- [utils_training/train_unet_worker.py:467](../utils_training/train_unet_worker.py#L467) — `save_training_plots`, 32 строк
- [utils_training/train_unet_worker_рабочий.py:452](../utils_training/train_unet_worker_рабочий.py#L452) — `save_training_plots`, 26 строк

## 17. format_annotation_display

- [utils_new/annotation_processing.py:1204](../utils_new/annotation_processing.py#L1204) — `format_annotation_display`, 28 строк
- [utils_ops/segmentation_ops.py:340](../utils_ops/segmentation_ops.py#L340) — `format_annotation_display`, 28 строк

## 18. read_image_bgr

- [cluster/canonicalize_crops.py:74](../cluster/canonicalize_crops.py#L74) — `read_image_bgr`, 21 строк
- [cluster/canonicalize_crops_method2.py:91](../cluster/canonicalize_crops_method2.py#L91) — `read_image_bgr`, 21 строк
- [cluster/save_yolo_object_to_path.py:58](../cluster/save_yolo_object_to_path.py#L58) — `read_image`, 27 строк

## 19. read_image_rgb

- [cluster/cluster_embeddings.py:90](../cluster/cluster_embeddings.py#L90) — `read_image_rgb`, 23 строк
- [cluster/extract_embeddings.py:58](../cluster/extract_embeddings.py#L58) — `read_image_rgb`, 23 строк

## 20. __init__

- [utils_aeb/auto-aeb_def.py:110](../utils_aeb/auto-aeb_def.py#L110) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_def_new.py:123](../utils_aeb/auto-aeb_def_new.py#L123) — `__init__`, 23 строк
- [utils_aeb/auto-aeb_old.py:112](../utils_aeb/auto-aeb_old.py#L112) — `__init__`, 23 строк

## 21. _detect_annotation_to_obb

- [tabs/layout_dataset.py:121](../tabs/layout_dataset.py#L121) — `_detect_annotation_to_obb`, 22 строк
- [tabs/viewing_dataset.py:131](../tabs/viewing_dataset.py#L131) — `_detect_annotation_to_obb`, 22 строк

## 22. _confirm_label_save

- [tabs/layout_dataset.py:2156](../tabs/layout_dataset.py#L2156) — `_confirm_label_save`, 20 строк
- [tabs/viewing_dataset.py:1026](../tabs/viewing_dataset.py#L1026) — `_confirm_label_save`, 20 строк

## 23. __init__

- [utils_aeb/auto-aeb_def.py:175](../utils_aeb/auto-aeb_def.py#L175) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:177](../utils_aeb/auto-aeb_old.py#L177) — `__init__`, 12 строк
- [utils_new/segmentation/autoencoder.py:40](../utils_new/segmentation/autoencoder.py#L40) — `__init__`, 20 строк

## 24. __init__

- [utils_aeb/auto-aeb_def_new.py:198](../utils_aeb/auto-aeb_def_new.py#L198) — `__init__`, 12 строк
- [utils_new/segmentation/autoencoder.py:148](../utils_new/segmentation/autoencoder.py#L148) — `__init__`, 20 строк

## 25. run

- [tabs/layout_dataset.py:176](../tabs/layout_dataset.py#L176) — `run`, 18 строк
- [tabs/viewing_dataset.py:186](../tabs/viewing_dataset.py#L186) — `run`, 18 строк

## 26. format_object_for_list

- [utils_new/annotation_processing.py:1233](../utils_new/annotation_processing.py#L1233) — `format_object_for_list`, 18 строк
- [utils_ops/segmentation_ops.py:442](../utils_ops/segmentation_ops.py#L442) — `format_object_for_list`, 18 строк

## 27. __init__

- [utils_aeb/auto-aeb_def.py:193](../utils_aeb/auto-aeb_def.py#L193) — `__init__`, 12 строк
- [utils_aeb/auto-aeb_old.py:195](../utils_aeb/auto-aeb_old.py#L195) — `__init__`, 12 строк
- [utils_new/segmentation/autoencoder.py:66](../utils_new/segmentation/autoencoder.py#L66) — `__init__`, 15 строк

## 28. __init__

- [utils_aeb/auto-aeb_def_new.py:216](../utils_aeb/auto-aeb_def_new.py#L216) — `__init__`, 12 строк
- [utils_new/segmentation/autoencoder.py:174](../utils_new/segmentation/autoencoder.py#L174) — `__init__`, 15 строк

## 29. signal_handler

- [utils_training/train_unet_worker.py:139](../utils_training/train_unet_worker.py#L139) — `signal_handler`, 15 строк
- [utils_training/train_unet_worker_рабочий.py:133](../utils_training/train_unet_worker_рабочий.py#L133) — `signal_handler`, 15 строк

## 30. _resolve_label_path

- [tabs/layout_dataset.py:161](../tabs/layout_dataset.py#L161) — `_resolve_label_path`, 14 строк
- [tabs/viewing_dataset.py:171](../tabs/viewing_dataset.py#L171) — `_resolve_label_path`, 14 строк

## 31. forward

- [utils_aeb/auto-aeb_def.py:90](../utils_aeb/auto-aeb_def.py#L90) — `forward`, 14 строк
- [utils_aeb/auto-aeb_def_new.py:103](../utils_aeb/auto-aeb_def_new.py#L103) — `forward`, 14 строк
- [utils_aeb/auto-aeb_old.py:92](../utils_aeb/auto-aeb_old.py#L92) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet.py:95](../utils_aeb/auto-aeb_unet.py#L95) — `forward`, 14 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:185](../utils_aeb/auto-aeb_unet_with_grad.py#L185) — `forward`, 14 строк

## 32. collect_images

- [cluster/canonicalize_crops.py:109](../cluster/canonicalize_crops.py#L109) — `collect_images`, 13 строк
- [cluster/canonicalize_crops_method2.py:126](../cluster/canonicalize_crops_method2.py#L126) — `collect_images`, 13 строк

## 33. _log_exception

- [tabs/layout_dataset.py:638](../tabs/layout_dataset.py#L638) — `_log_exception`, 13 строк
- [tabs/viewing_dataset.py:315](../tabs/viewing_dataset.py#L315) — `_log_exception`, 13 строк

## 34. __getattr__

- [import_libs_external.py:138](../import_libs_external.py#L138) — `__getattr__`, 12 строк
- [import_libs_internal.py:224](../import_libs_internal.py#L224) — `__getattr__`, 12 строк

## 35. __init__

- [utils_aeb/auto-aeb_unet.py:309](../utils_aeb/auto-aeb_unet.py#L309) — `__init__`, 12 строк
- [utils_new/segmentation/autoencoder.py:371](../utils_new/segmentation/autoencoder.py#L371) — `__init__`, 12 строк

## 36. get_optimizer

- [utils_training/train_unet_worker.py:370](../utils_training/train_unet_worker.py#L370) — `get_optimizer`, 12 строк
- [utils_training/train_unet_worker_рабочий.py:375](../utils_training/train_unet_worker_рабочий.py#L375) — `get_optimizer`, 12 строк

## 37. __getitem__

- [utils_aeb/auto-aeb_def.py:149](../utils_aeb/auto-aeb_def.py#L149) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:162](../utils_aeb/auto-aeb_def_new.py#L162) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_old.py:151](../utils_aeb/auto-aeb_old.py#L151) — `__getitem__`, 11 строк
- [utils_aeb/auto-aeb_unet.py:180](../utils_aeb/auto-aeb_unet.py#L180) — `__getitem__`, 11 строк

## 38. __init__

- [utils_aeb/auto-aeb_def.py:230](../utils_aeb/auto-aeb_def.py#L230) — `__init__`, 11 строк
- [utils_aeb/auto-aeb_old.py:232](../utils_aeb/auto-aeb_old.py#L232) — `__init__`, 11 строк
- [utils_new/segmentation/autoencoder.py:106](../utils_new/segmentation/autoencoder.py#L106) — `__init__`, 11 строк

## 39. _fit

- [utils_aeb/auto-aeb_def.py:134](../utils_aeb/auto-aeb_def.py#L134) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_def_new.py:147](../utils_aeb/auto-aeb_def_new.py#L147) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_old.py:136](../utils_aeb/auto-aeb_old.py#L136) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet.py:165](../utils_aeb/auto-aeb_unet.py#L165) — `_fit`, 11 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:258](../utils_aeb/auto-aeb_unet_with_grad.py#L258) — `_fit`, 11 строк

## 40. __init__

- [utils_aeb/auto-aeb_unet.py:288](../utils_aeb/auto-aeb_unet.py#L288) — `__init__`, 11 строк
- [utils_new/segmentation/autoencoder.py:350](../utils_new/segmentation/autoencoder.py#L350) — `__init__`, 11 строк

## 41. imwrite_safe

- [cluster/canonicalize_crops.py:97](../cluster/canonicalize_crops.py#L97) — `imwrite_safe`, 10 строк
- [cluster/canonicalize_crops_method2.py:114](../cluster/canonicalize_crops_method2.py#L114) — `imwrite_safe`, 10 строк

## 42. _resolve_save_path_for_img

- [tabs/layout_dataset.py:2145](../tabs/layout_dataset.py#L2145) — `_resolve_save_path_for_img`, 10 строк
- [tabs/viewing_dataset.py:1015](../tabs/viewing_dataset.py#L1015) — `_resolve_save_path_for_img`, 10 строк

## 43. get_scheduler

- [utils_training/train_unet_worker.py:385](../utils_training/train_unet_worker.py#L385) — `get_scheduler`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:390](../utils_training/train_unet_worker_рабочий.py#L390) — `get_scheduler`, 10 строк

## 44. save_args_yaml

- [utils_training/train_unet_worker.py:447](../utils_training/train_unet_worker.py#L447) — `save_args_yaml`, 10 строк
- [utils_training/train_unet_worker_рабочий.py:433](../utils_training/train_unet_worker_рабочий.py#L433) — `save_args_yaml`, 10 строк

## 45. apply_transform

- [cluster/canonicalize_crops.py:143](../cluster/canonicalize_crops.py#L143) — `apply_transform`, 9 строк
- [cluster/canonicalize_crops_method2.py:163](../cluster/canonicalize_crops_method2.py#L163) — `apply_transform`, 9 строк

## 46. apply_transform

- [cluster/extract_embeddings.py:113](../cluster/extract_embeddings.py#L113) — `apply_transform`, 9 строк
- [cluster/test_new_image.py:197](../cluster/test_new_image.py#L197) — `apply_transform`, 9 строк

## 47. __init__

- [utils_aeb/auto-aeb_def.py:211](../utils_aeb/auto-aeb_def.py#L211) — `__init__`, 9 строк
- [utils_aeb/auto-aeb_old.py:213](../utils_aeb/auto-aeb_old.py#L213) — `__init__`, 9 строк
- [utils_new/segmentation/autoencoder.py:87](../utils_new/segmentation/autoencoder.py#L87) — `__init__`, 9 строк

## 48. collect_images

- [utils_aeb/auto-aeb_def.py:162](../utils_aeb/auto-aeb_def.py#L162) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_def_new.py:175](../utils_aeb/auto-aeb_def_new.py#L175) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_old.py:164](../utils_aeb/auto-aeb_old.py#L164) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet.py:193](../utils_aeb/auto-aeb_unet.py#L193) — `collect_images`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:382](../utils_aeb/auto-aeb_unet_with_grad.py#L382) — `collect_images`, 9 строк

## 49. forward

- [utils_aeb/auto-aeb_unet.py:122](../utils_aeb/auto-aeb_unet.py#L122) — `forward`, 9 строк
- [utils_aeb/auto-aeb_unet_with_grad.py:211](../utils_aeb/auto-aeb_unet_with_grad.py#L211) — `forward`, 9 строк

## 50. median_blur

- [utils_aeb/auto-aeb_unet_with_grad.py:120](../utils_aeb/auto-aeb_unet_with_grad.py#L120) — `median_blur`, 9 строк
- [utils_new/segmentation/autoencoder.py:614](../utils_new/segmentation/autoencoder.py#L614) — `_median_blur_torch`, 9 строк
