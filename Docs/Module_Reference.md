# Справочник модулей, импортов и API

[Навигация](README.md) · [Карта архитектуры](Architecture.md) · [Дублирование](Duplication.md)

Сгенерировано `python Docs/generate_reference.py` из текущих исходников. Каталог исключает саму документацию и скрытые каталоги.

Проанализировано Python-модулей: **128**. Импорты учитываются на всех уровнях AST, включая функции и try/except; наличие записи не означает, что ветка выполняется при запуске. Звёздочные импорты не разворачиваются в список фактических вызовов. QProcess, строковые динамические импорты и файловые зависимости описаны отдельно в карте.

Для всех модулей перечислены верхнеуровневые функции и классы. Для всех каталогов утилит дополнительно перечислены сигнатуры методов классов. Вложенные локальные функции не являются отдельным API и в список не включены. Приватные имена сохранены для поиска реализации.

## Содержание

- [cluster.canonicalize_crops](#cluster-canonicalize-crops)
- [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2)
- [cluster.cluster_embeddings](#cluster-cluster-embeddings)
- [cluster.extract_embeddings](#cluster-extract-embeddings)
- [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path)
- [cluster.test_new_image](#cluster-test-new-image)
- [generate](#generate)
- [import_libs_external](#import-libs-external)
- [import_libs_internal](#import-libs-internal)
- [import_libs_methods_ui](#import-libs-methods-ui)
- [install](#install)
- [main](#main)
- [main_ui](#main-ui)
- [path_setup](#path-setup)
- [rename_data](#rename-data)
- [tabs.dataset_analysis](#tabs-dataset-analysis)
- [tabs.deep_learning_methods](#tabs-deep-learning-methods)
- [tabs.deep_learning_training](#tabs-deep-learning-training)
- [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods)
- [tabs.gradient_methods](#tabs-gradient-methods)
- [tabs.interactive_methods](#tabs-interactive-methods)
- [tabs.layout_dataset](#tabs-layout-dataset)
- [tabs.model_research](#tabs-model-research)
- [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml)
- [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi)
- [tabs.threshold_methods](#tabs-threshold-methods)
- [tabs.traditional_ml_methods](#tabs-traditional-ml-methods)
- [tabs.viewing_dataset](#tabs-viewing-dataset)
- [tabs.yolo_find_img](#tabs-yolo-find-img)
- [tabs.yolo_test_aeb](#tabs-yolo-test-aeb)
- [ui.DL_predict_settings](#ui-DL-predict-settings)
- [ui.DL_train_settings_base](#ui-DL-train-settings-base)
- [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model)
- [ui.control_buttons](#ui-control-buttons)
- [ui.image_navigation](#ui-image-navigation)
- [ui.interactive_params_ui](#ui-interactive-params-ui)
- [ui.log_widget](#ui-log-widget)
- [ui.morphology_ui](#ui-morphology-ui)
- [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new)
- [ui.settings_ui](#ui-settings-ui)
- [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui)
- [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui)
- [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui)
- [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui)
- [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui)
- [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui)
- [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui)
- [ui_tabs.model_research_ui](#ui-tabs-model-research-ui)
- [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui)
- [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui)
- [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui)
- [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui)
- [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui)
- [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui)
- [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui)
- [utils.DatasetGenerator](#utils-DatasetGenerator)
- [utils.analysis](#utils-analysis)
- [utils.annotation_converter](#utils-annotation-converter)
- [utils.aug_utils](#utils-aug-utils)
- [utils.augmentation_base](#utils-augmentation-base)
- [utils.augmentation_random](#utils-augmentation-random)
- [utils.coco_export](#utils-coco-export)
- [utils.coco_io](#utils-coco-io)
- [utils.dataset_analysis_logic](#utils-dataset-analysis-logic)
- [utils.dataset_audit_logic](#utils-dataset-audit-logic)
- [utils.dataset_validation](#utils-dataset-validation)
- [utils.ensemble_utils](#utils-ensemble-utils)
- [utils.image_io](#utils-image-io)
- [utils.metrics_core](#utils-metrics-core)
- [utils.prepare](#utils-prepare)
- [utils.presets](#utils-presets)
- [utils.settings](#utils-settings)
- [utils.smart_view](#utils-smart-view)
- [utils.unet_export](#utils-unet-export)
- [utils.unet_io](#utils-unet-io)
- [utils.yaml_io](#utils-yaml-io)
- [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def)
- [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new)
- [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old)
- [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet)
- [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad)
- [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils)
- [utils_auto_research.gradient_auto_research](#utils-auto-research-gradient-auto-research)
- [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog)
- [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research)
- [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog)
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new)
- [utils_new.annotation_processing](#utils-new-annotation-processing)
- [utils_new.annotations.__init__](#utils-new-annotations---init--)
- [utils_new.annotations.adapters](#utils-new-annotations-adapters)
- [utils_new.annotations.augmentation](#utils-new-annotations-augmentation)
- [utils_new.annotations.geometry](#utils-new-annotations-geometry)
- [utils_new.annotations.masks](#utils-new-annotations-masks)
- [utils_new.annotations.model](#utils-new-annotations-model)
- [utils_new.annotations.rendering](#utils-new-annotations-rendering)
- [utils_new.annotations.transforms](#utils-new-annotations-transforms)
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new)
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new)
- [utils_new.dataset_files_new](#utils-new-dataset-files-new)
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new)
- [utils_new.dataset_io_new](#utils-new-dataset-io-new)
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new)
- [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new)
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new)
- [utils_new.dataset_output_new](#utils-new-dataset-output-new)
- [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new)
- [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new)
- [utils_new.dataset_preparation_bridge](#utils-new-dataset-preparation-bridge)
- [utils_new.dataset_review_view](#utils-new-dataset-review-view)
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new)
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new)
- [utils_new.dataset_view_new](#utils-new-dataset-view-new)
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new)
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new)
- [utils_new.morphology](#utils-new-morphology)
- [utils_new.smart_view_new](#utils-new-smart-view-new)
- [utils_new.threshold_processing](#utils-new-threshold-processing)
- [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops)
- [utils_ops.gradient_ops](#utils-ops-gradient-ops)
- [utils_ops.interactive_ops](#utils-ops-interactive-ops)
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops)
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops)
- [utils_training.test_train_worker](#utils-training-test-train-worker)
- [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker)
- [utils_training.train_unet_worker](#utils-training-train-unet-worker)
- [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий)
- [utils_training.train_yolo_worker](#utils-training-train-yolo-worker)
- [utils_training.yolo2voc](#utils-training-yolo2voc)

## Внешние импортируемые модули

Имена слева — Python imports, а не обязательно имена пакетов PyPI. Версии и назначение — в [зависимостях](Dependencies.md). Стандартная библиотека не включена в эту таблицу.

| Import | Файлы-потребители (включая условные импорты) |
| --- | --- |
| `PIL` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [generate](#generate), [main](#main), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.image_io](#utils-image-io), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `PyQt5` | [generate](#generate), [import_libs_external](#import-libs-external), [main](#main), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui.control_buttons](#ui-control-buttons), [ui.image_navigation](#ui-image-navigation), [ui.interactive_params_ui](#ui-interactive-params-ui), [ui.morphology_ui](#ui-morphology-ui), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.settings_ui](#ui-settings-ui), [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_review_view](#utils-new-dataset-review-view), [utils_new.smart_view_new](#utils-new-smart-view-new) |
| `albumentations` | [import_libs_external](#import-libs-external), [main](#main) |
| `cv2` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [utils.coco_export](#utils-coco-export), [utils.coco_io](#utils-coco-io), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.unet_export](#utils-unet-export), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.smart_view_new](#utils-new-smart-view-new), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `dataset_analysis_ui` | [tabs.dataset_analysis](#tabs-dataset-analysis) |
| `hdbscan` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `imagehash` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `joblib` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `matplotlib` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.prepare](#utils-prepare), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `networkx` | [import_libs_external](#import-libs-external), [utils_ops.interactive_ops](#utils-ops-interactive-ops) |
| `numpy` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [utils.annotation_converter](#utils-annotation-converter), [utils.coco_export](#utils-coco-export), [utils.coco_io](#utils-coco-io), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.prepare](#utils-prepare), [utils.unet_export](#utils-unet-export), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.model](#utils-new-annotations-model), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.smart_view_new](#utils-new-smart-view-new), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `onnx` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `onnxruntime` | [import_libs_external](#import-libs-external), [main](#main), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker) |
| `open_clip` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `pandas` | [import_libs_external](#import-libs-external), [utils.prepare](#utils-prepare) |
| `pascal_voc_writer` | [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `pip` | [install](#install) |
| `psutil` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `sahi` | [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi) |
| `seaborn` | [import_libs_external](#import-libs-external), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `segment_anything` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops) |
| `segmentation_models_pytorch` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `shapely` | [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.geometry](#utils-new-annotations-geometry) |
| `skimage` | [import_libs_external](#import-libs-external), [main](#main), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `sklearn` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [import_libs_external](#import-libs-external), [main](#main), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.prepare](#utils-prepare), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `tifffile` | [utils.image_io](#utils-image-io) |
| `torch` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [install](#install), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `torchvision` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `transformers` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `ultralytics` | [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [import_libs_external](#import-libs-external), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_training](#tabs-deep-learning-training), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.smart_view](#utils-smart-view), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `umap` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `xgboost` | [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `yaml` | [import_libs_external](#import-libs-external), [main](#main), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [utils.unet_export](#utils-unet-export), [utils.unet_io](#utils-unet-io), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |

<a id="cluster-canonicalize-crops"></a>

## cluster.canonicalize_crops

Исходник: [cluster/canonicalize_crops.py](../cluster/canonicalize_crops.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `csv`, `cv2`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sys`, `warnings`.

**Объявления:**

- [`read_image_bgr(path: str)`](../cluster/canonicalize_crops.py#L73)
- [`imwrite_safe(path: str, img_bgr)`](../cluster/canonicalize_crops.py#L96)
- [`collect_images(folder: str, recursive: bool=False)`](../cluster/canonicalize_crops.py#L108)
- [`prepare_compare(img_bgr, size: int)`](../cluster/canonicalize_crops.py#L123) — Grayscale → ресайз с сохранением пропорций + паддинг → zero-mean/unit-std.
- [`apply_transform(img, angle_deg: int, flip: bool)`](../cluster/canonicalize_crops.py#L142)
- [`pearson(a, b)`](../cluster/canonicalize_crops.py#L153)
- [`find_best_transform(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/canonicalize_crops.py#L157)
- [`make_debug_montage(records, out_path, cols=8, max_items=60)`](../cluster/canonicalize_crops.py#L172) — records: list of (img_bgr, angle, flip, score).
- [`parse_args()`](../cluster/canonicalize_crops.py#L202)
- [`main()`](../cluster/canonicalize_crops.py#L225)

<a id="cluster-canonicalize-crops-method2"></a>

## cluster.canonicalize_crops_method2

Исходник: [cluster/canonicalize_crops_method2.py](../cluster/canonicalize_crops_method2.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `csv`, `cv2`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sys`, `warnings`.

**Объявления:**

- [`read_image_bgr(path: str)`](../cluster/canonicalize_crops_method2.py#L90)
- [`imwrite_safe(path: str, img_bgr)`](../cluster/canonicalize_crops_method2.py#L113)
- [`collect_images(folder: str, recursive: bool=False)`](../cluster/canonicalize_crops_method2.py#L125)
- [`prepare_compare(img_bgr, size: int=224)`](../cluster/canonicalize_crops_method2.py#L143) — Grayscale → ресайз → паддинг → zero-mean/unit-std.
- [`apply_transform(img, angle_deg: int, flip: bool)`](../cluster/canonicalize_crops_method2.py#L162)
- [`find_best_by_correlation(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/canonicalize_crops_method2.py#L173) — Перебор дискретных углов и зеркала, оценка — Pearson.
- [`get_detector(method: str)`](../cluster/canonicalize_crops_method2.py#L190) — Возвращает (detector, norm_type).
- [`extract_features(img_bgr, detector)`](../cluster/canonicalize_crops_method2.py#L202)
- [`match_good(des1, des2, norm_type, ratio=0.75)`](../cluster/canonicalize_crops_method2.py#L208) — Lowe ratio test, возвращает список good matches.
- [`estimate_rotation(img_bgr, ref_data, apply_flip, ratio, min_good, ransac_thresh)`](../cluster/canonicalize_crops_method2.py#L228) — Пытается оценить угол поворота кропа относительно референса
- [`rotate_image(img_bgr, angle_deg, flip)`](../cluster/canonicalize_crops_method2.py#L289) — Применяет зеркало (если надо) и поворот на произвольный угол.
- [`save_fig_via_cv2(fig, out_path, dpi=110)`](../cluster/canonicalize_crops_method2.py#L321) — fig → numpy → cv2.imencode → файл. Обходит savefig/PIL на Windows.
- [`make_debug_montage(records, out_path, cols=8, max_items=60)`](../cluster/canonicalize_crops_method2.py#L335) — records: list of (img_bgr, info_str).
- [`parse_args()`](../cluster/canonicalize_crops_method2.py#L364)
- [`main()`](../cluster/canonicalize_crops_method2.py#L391)

<a id="cluster-cluster-embeddings"></a>

## cluster.cluster_embeddings

Исходник: [cluster/cluster_embeddings.py](../cluster/cluster_embeddings.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `csv`, `cv2`, `hdbscan`, `joblib`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sklearn.cluster.HDBSCAN`, `sklearn.cluster.KMeans`, `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `sklearn.metrics.silhouette_score`, `sklearn.preprocessing.normalize`, `sys`, `umap`, `warnings`.

**Объявления:**

- [`read_image_rgb(path: str)`](../cluster/cluster_embeddings.py#L90)
- [`collect_embeddings(folder: str, recursive: bool=False)`](../cluster/cluster_embeddings.py#L115)
- [`load_embedding(path: str)`](../cluster/cluster_embeddings.py#L130)
- [`find_image_for_embedding(emb_path: str)`](../cluster/cluster_embeddings.py#L142)
- [`load_all_embeddings(folder: str, recursive: bool=False)`](../cluster/cluster_embeddings.py#L152)
- [`build_reducer(method: str, n_components: int, X: np.ndarray)`](../cluster/cluster_embeddings.py#L171) — Обучает reducer. Для маленьких выборок корректирует n_components.
- [`apply_reducer(reducer, X)`](../cluster/cluster_embeddings.py#L204)
- [`fit_hdbscan(X, min_cluster_size=3, min_samples=None)`](../cluster/cluster_embeddings.py#L213)
- [`fit_kmeans(X, n_clusters=8, n_init=10, random_state=42)`](../cluster/cluster_embeddings.py#L235)
- [`auto_kmeans_k(X, k_min=2, k_max=12, random_state=42)`](../cluster/cluster_embeddings.py#L240)
- [`reduce_2d_for_plot(X)`](../cluster/cluster_embeddings.py#L264)
- [`plot_scatter(X2d, labels, out_path, title)`](../cluster/cluster_embeddings.py#L290)
- [`make_montage(image_paths, title, out_path, cols=6, max_items=60)`](../cluster/cluster_embeddings.py#L314)
- [`compute_centroids(X, labels)`](../cluster/cluster_embeddings.py#L340) — Центроиды в том пространстве, в котором выполнялась кластеризация.
- [`save_state(out_dir, centroids, cluster_ids, reducer, reducer_kind, meta, emb_paths, labels, image_paths)`](../cluster/cluster_embeddings.py#L356)
- [`load_state(model_dir)`](../cluster/cluster_embeddings.py#L379)
- [`run_train(args)`](../cluster/cluster_embeddings.py#L393)
- [`predict_centroid(X_new_red, centroids, cluster_ids)`](../cluster/cluster_embeddings.py#L557)
- [`run_predict(args)`](../cluster/cluster_embeddings.py#L567)
- [`parse_args()`](../cluster/cluster_embeddings.py#L643)
- [`main()`](../cluster/cluster_embeddings.py#L675)

<a id="cluster-extract-embeddings"></a>

## cluster.extract_embeddings

Исходник: [cluster/extract_embeddings.py](../cluster/extract_embeddings.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `cv2`, `numpy`, `os`, `sys`, `torch`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `warnings`.

**Объявления:**

- [`read_image_rgb(path: str)`](../cluster/extract_embeddings.py#L59)
- [`collect_images(folder: str, recursive: bool=True)`](../cluster/extract_embeddings.py#L84)
- [`resolve_device(device_str: str)`](../cluster/extract_embeddings.py#L99)
- [`embedding_path_for(img_path: str)`](../cluster/extract_embeddings.py#L110)
- [`apply_transform(img, angle_deg, flip)`](../cluster/extract_embeddings.py#L114)
- [`load_dinov2(model_name: str, device: str)`](../cluster/extract_embeddings.py#L128)
- [`extract_batch(model, batch_tensor, device)`](../cluster/extract_embeddings.py#L145)
- [`parse_args()`](../cluster/extract_embeddings.py#L155)
- [`main()`](../cluster/extract_embeddings.py#L178)

<a id="cluster-save-yolo-object-to-path"></a>

## cluster.save_yolo_object_to_path

Исходник: [cluster/save_yolo_object_to_path.py](../cluster/save_yolo_object_to_path.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `cv2`, `numpy`, `os`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- [`read_image(path: str)`](../cluster/save_yolo_object_to_path.py#L49) — Чтение изображения с поддержкой кириллицы в пути + нормализация к BGR uint8.
- [`imwrite_safe(path: str, img)`](../cluster/save_yolo_object_to_path.py#L78) — Запись с поддержкой кириллицы в пути.
- [`unique_path(path: str)`](../cluster/save_yolo_object_to_path.py#L91) — Если файл уже существует — добавляет _1, _2, ...
- [`collect_images(folder: str)`](../cluster/save_yolo_object_to_path.py#L104) — Возвращает список путей ко всем изображениям (рекурсивно).
- [`resolve_device(device_str: str)`](../cluster/save_yolo_object_to_path.py#L115)
- [`parse_args()`](../cluster/save_yolo_object_to_path.py#L137)
- [`main()`](../cluster/save_yolo_object_to_path.py#L166)

<a id="cluster-test-new-image"></a>

## cluster.test_new_image

Исходник: [cluster/test_new_image.py](../cluster/test_new_image.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `collections.Counter`, `csv`, `cv2`, `dataclasses.dataclass`, `dataclasses.field`, `joblib`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sys`, `torch`, `torch.nn.functional`, `torchvision.transforms`, `ultralytics.YOLO`, `warnings`.

**Объявления:**

- [`read_image_bgr(path)`](../cluster/test_new_image.py#L105)
- [`imwrite_safe(path, img_bgr)`](../cluster/test_new_image.py#L128)
- [`collect_images(path, recursive=False)`](../cluster/test_new_image.py#L140)
- [`resolve_device(device_str)`](../cluster/test_new_image.py#L157)
- [`prepare_compare(img_bgr, size=224)`](../cluster/test_new_image.py#L171)
- [`apply_transform(img, angle_deg, flip)`](../cluster/test_new_image.py#L187)
- [`find_best_alignment(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/test_new_image.py#L198)
- [`load_dinov2(model_name, device)`](../cluster/test_new_image.py#L214)
- [`embed_images(model, transform, images_bgr, device, batch_size=16)`](../cluster/test_new_image.py#L231) — images_bgr: list[np.ndarray BGR]. Возвращает np.ndarray [N, D] L2-норм.
- [`load_cluster_state(model_dir)`](../cluster/test_new_image.py#L254)
- [`predict_cluster(emb, centroids, reducer, cluster_ids, min_sim=0.0)`](../cluster/test_new_image.py#L266)
- Класс [`Record`](../cluster/test_new_image.py#L285)
- [`build_collage(records, out_path, cell=220, cols=8, title='')`](../cluster/test_new_image.py#L300) — Строит коллаж с группировкой по (class_id, cluster_id).
- [`parse_args()`](../cluster/test_new_image.py#L393)
- [`main()`](../cluster/test_new_image.py#L409)

<a id="generate"></a>

## generate

Исходник: [generate.py](../generate.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `datetime.datetime`, `os`, `sys`, `traceback`.

**Объявления:**

- [`log_event(message)`](../generate.py#L15)
- [`index_to_label(idx)`](../generate.py#L20) — 0 -> 'а)', 1 -> 'б)', ..., 32 -> 'я)', 33 -> 'а1)', 34 -> 'б1)', ...
- Класс [`CollageApp`](../generate.py#L34) (`QMainWindow`)

<a id="import-libs-external"></a>

## import_libs_external

Исходник: [import_libs_external.py](../import_libs_external.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [main_ui](#main-ui), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.DL_predict_settings](#ui-DL-predict-settings), [ui.image_navigation](#ui-image-navigation), [ui.log_widget](#ui-log-widget), [utils.analysis](#utils-analysis), [utils.annotation_converter](#utils-annotation-converter), [utils.coco_export](#utils-coco-export), [utils.coco_io](#utils-coco-io), [utils.image_io](#utils-image-io), [utils.metrics_core](#utils-metrics-core), [utils.settings](#utils-settings), [utils.unet_export](#utils-unet-export), [utils.unet_io](#utils-unet-io), [utils.yaml_io](#utils-yaml-io), [utils_auto_research.gradient_auto_research](#utils-auto-research-gradient-auto-research), [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog), [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research), [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_ops.gradient_ops](#utils-ops-gradient-ops), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.QDir`, `PyQt5.QtCore.QObject`, `PyQt5.QtCore.QPointF`, `PyQt5.QtCore.QProcess`, `PyQt5.QtCore.QRectF`, `PyQt5.QtCore.QSettings`, `PyQt5.QtCore.QThread`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.QUrl`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QBrush`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QDesktopServices`, `PyQt5.QtGui.QFont`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QMouseEvent`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPainterPath`, `PyQt5.QtGui.QPalette`, `PyQt5.QtGui.QPen`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtGui.QTransform`, `PyQt5.QtGui.QWheelEvent`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QAction`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QButtonGroup`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFileSystemModel`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGraphicsPixmapItem`, `PyQt5.QtWidgets.QGraphicsScene`, `PyQt5.QtWidgets.QGraphicsView`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QInputDialog`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplashScreen`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QStackedWidget`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QTableWidgetItem`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QToolBox`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QTreeView`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `albumentations`, `ast.literal_eval`, `collections.OrderedDict`, `colorsys`, `concurrent.futures.ThreadPoolExecutor`, `cv2`, `datetime.datetime`, `hashlib`, `joblib`, `json`, `logging`, `math`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.backends.backend_qt5agg.NavigationToolbar2QT`, `matplotlib.figure.Figure`, `matplotlib.pyplot`, `networkx`, `numpy`, `onnxruntime`, `os`, `pandas`, `pathlib.Path`, `queue`, `random`, `re`, `seaborn`, `shutil`, `skimage.segmentation`, `skimage.segmentation.find_boundaries`, `skimage.segmentation.slic`, `skimage.util.img_as_float`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.r2_score`, `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.StandardScaler`, `sys`, `threading`, `time`, `torch`, `torch.nn`, `torchvision.transforms`, `traceback`, `ultralytics`, `ultralytics.YOLO`, `ultralytics.data.converter.convert_segment_masks_to_yolo_seg`, `warnings`, `yaml`.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="import-libs-internal"></a>

## import_libs_internal

Исходник: [import_libs_internal.py](../import_libs_internal.py#L1).

**Прямые потребители:** [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.model_research_ui](#ui-tabs-model-research-ui), [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.aug_utils](#utils-aug-utils), [utils.augmentation_base](#utils-augmentation-base), [utils.augmentation_random](#utils-augmentation-random), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.dataset_validation](#utils-dataset-validation), [utils.ensemble_utils](#utils-ensemble-utils), [utils.presets](#utils-presets), [utils.smart_view](#utils-smart-view), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [ui.DL_predict_settings](#ui-DL-predict-settings): `CustomONNXInferenceSettings`, `DeepLabV3InferenceSettings`, `SAMInferenceSettings`, `SegFormerInferenceSettings`, `UNetInferenceSettings`, `YOLOInferenceSettings`
- [ui.DL_train_settings_base](#ui-DL-train-settings-base): `UNetTrainWidget`, `YOLOTrainWidget`
- [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model): `DeepLabV3TrainWidget`, `SAMTrainWidget`, `SegFormerTrainWidget`
- [ui.control_buttons](#ui-control-buttons): `ControlButtons`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`, `NavigationShortcutInstaller`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [utils.DatasetGenerator](#utils-DatasetGenerator): `DatasetGeneratorThread`, `INPUT_IMAGES`, `INPUT_LABELS`, `INPUT_MASKS`
- [utils.analysis](#utils-analysis): `analyze`, `global_top_diverse`
- [utils.annotation_converter](#utils-annotation-converter): `SUBTYPE_DETECT`, `SUBTYPE_KEEP`, `SUBTYPE_OBB`, `SUBTYPE_SEGMENT`, `convert_annotations`, `to_detect`, `to_obb`, `to_segment`
- [utils.aug_utils](#utils-aug-utils): `apply_affine_to_points_np`, `clip_annotations`, `compose_affine_matrices`, `log_annotations_summary`, `repair_obb_to_rectangle`, `resize_image_and_annotations`, `resize_image_and_annotations_stretch`
- [utils.augmentation_base](#utils-augmentation-base): `apply_simple_augmentations`, `apply_simple_augmentations_mask`
- [utils.augmentation_random](#utils-augmentation-random): `apply_complex_augmentations`, `apply_complex_augmentations_mask`
- [utils.coco_export](#utils-coco-export): `export_coco_from_labels`, `export_coco_from_masks`
- [utils.coco_io](#utils-coco-io): `load_coco`, `write_yolo_txt_for_coco_items`
- [utils.dataset_analysis_logic](#utils-dataset-analysis-logic): `BrightnessAnalyzer`, `ClassImbalanceAnalyzer`, `DatasetAnalyzer`, `FilterParams`, `ResolutionAnalyzer`, `TrainingResultsLoader`, `clear_brightness_cache`
- [utils.dataset_audit_logic](#utils-dataset-audit-logic): `detect_device`
- [utils.dataset_validation](#utils-dataset-validation): `DatasetLoaderMixin`, `ImageStats`, `PairValidationDialog`, `PairValidator`, `build_pair_display`, `convert_pairs_color`, `convert_pairs_to_8bit_png`, `scan_detection_pairs`, `scan_image_only_pairs`, `scan_segmentation_pairs`, `select_pairs_for_8bit_png`, `select_pairs_for_conversion`
- [utils.ensemble_utils](#utils-ensemble-utils): `ensemble_detections`, `extract_threshold_params`, `get_detections_from_preset`, `run_ensemble`
- [utils.image_io](#utils-image-io): `FORMAT_HINT_AUTO`, `FORMAT_HINT_DETECT`, `FORMAT_HINT_MIXED`, `FORMAT_HINT_OBB_CORNERS`, `FORMAT_HINT_OBB_XYWHR`, `FORMAT_HINT_SEGMENT`, `LazyImageCache`, `canonicalize_obb_corners`, `convert_segment_masks_to_yolo_seg_manual`, `convert_to_grayscale`, `load_annotations`, `load_dataset_from_yaml`, `load_dataset_from_yaml_with_masks`, `load_images_universal`, `load_one_image_item`, `normalize_to_uint8`, `numpy_to_qpixmap`, `read_image_with_fallback`, `read_image_with_fallback_find`, `read_mask_with_fallback`, `resize_to_max_side`, `save_annotations`, `save_coordinates`
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`, `iou_xyxy`, `xyxy_to_yolo`, `yolo_to_xyxy`
- [utils.prepare](#utils-prepare): `add_cluster_labels`, `add_meanap_metrics`, `aggregate_by_params`, `cluster_open_close`, `filter_anomaly_count`, `filter_duplicates`, `filter_full_image_object`, `filter_invalid_params`, `filter_no_true_objects`, `filter_oversegmentation`, `filter_zero_true`, `optimize_invert`, `sample_rows`, `save_unique_params`
- [utils.presets](#utils-presets): `DeepLearningPresetManager`, `GradientPresetManager`, `InteractivePresetManager`, `PresetManager`, `TraditionalMLPresetManager`
- [utils.settings](#utils-settings): `apply_theme`, `settings`
- [utils.smart_view](#utils-smart-view): `SmartGraphicsView`
- [utils.unet_export](#utils-unet-export): `export_masks_from_annotations`, `write_unet_yaml`
- [utils.unet_io](#utils-unet-io): `is_unet_yaml`, `load_unet_yaml`
- [utils.yaml_io](#utils-yaml-io): `find_mask_for_image`, `load_unet_yaml_pairs`, `load_yolo_yaml`
- [utils_auto_research.gradient_auto_research](#utils-auto-research-gradient-auto-research): `GradientAutoResearch`
- [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog): `GradientAutoSettingsDialog`
- [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research): `AutoResearch`
- [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog): `AutoSettingsDialog`
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new): `DatasetGeneratorThread`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_preparation_bridge](#utils-new-dataset-preparation-bridge): `mask_classes_from_rows`, `normalize_generation_pairs`
- [utils_new.dataset_review_view](#utils-new-dataset-review-view): `DatasetReviewGraphicsView`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`, `convert_pairs_color`, `convert_pairs_to_8bit_png`, `select_pairs_for_8bit_png`, `select_pairs_for_conversion`
- [utils_new.smart_view_new](#utils-new-smart-view-new): `SmartGraphicsViewNew`
- [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops): `create_segmentor`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`
- [utils_ops.interactive_ops](#utils-ops-interactive-ops): `active_contour_segmentation`, `grabcut_segmentation`, `lazy_snapping`, `onecut_segmentation`, `random_walker_segmentation`, `supercut_segmentation`, `watershed_segmentation`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `apply_morphology`, `apply_threshold_method`, `delete_annotation_by_index`, `draw_class_label`, `draw_label`, `draw_label_rotated`, `draw_selected_objects`, `draw_yolo_annotations`, `format_object_for_list`, `get_class_color`, `get_display_params`, `normalize_method_name`, `segment_contours`, `segment_min_area_rect`, `segment_projections`, `update_annotation_list`
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops): `compute_elbow_kmeans`, `create_superpixels`, `extract_pixel_features`, `extract_superpixel_features`, `find_optimal_k_elbow`, `predict_pixelwise`, `predict_superpixel`, `train_decision_tree`, `train_kmeans`, `train_meanshift`, `train_random_forest`, `train_svm`, `train_xgboost`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="import-libs-methods-ui"></a>

## import_libs_methods_ui

Исходник: [import_libs_methods_ui.py](../import_libs_methods_ui.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui): `setup_dataset_analysis_ui`
- [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui): `setup_deep_learning_methods_ui`
- [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui): `setup_deep_learning_training_ui`
- [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui): `setup_demo_ensemble_methods_ui`
- [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui): `setup_gradient_ui`
- [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui): `setup_interactive_ui`
- [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui): `setup_layout_dataset_ui`
- [ui_tabs.model_research_ui](#ui-tabs-model-research-ui): `setup_model_research_ui`
- [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui): `setup_preparing_dataset_yaml_ui`
- [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui): `setup_sahi_slicer_ui`
- [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui): `setup_threshold_ui`
- [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui): `setup_traditional_ml_ui`
- [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui): `setup_viewing_dataset_ui`
- [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui): `setup_yolo_find_img_ui`
- [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui): `setup_yolo_ae_vae_ui`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="install"></a>

## install

Исходник: [install.py](../install.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `importlib`, `pip`, `platform`, `subprocess`, `sys`, `torch`.

**Объявления:**

- [`run_pip_install(package, extra_index=None)`](../install.py#L12) — Устанавливает пакет через pip и возвращает True/False.
- [`get_pip()`](../install.py#L26)
- [`install_pytorch()`](../install.py#L36)
- [`install_other_packages()`](../install.py#L63)
- [`verify_installation()`](../install.py#L78)
- [`main()`](../install.py#L102)

<a id="main"></a>

## main

Исходник: [main.py](../main.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [main_ui](#main-ui): `StartupDialog`
- [tabs.dataset_analysis](#tabs-dataset-analysis): `DatasetAnalysisWindow`
- [tabs.deep_learning_methods](#tabs-deep-learning-methods): `DeepLearningWindow`
- [tabs.deep_learning_training](#tabs-deep-learning-training): `YoloTrainWindow`
- [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods): `YoloDemoWindow`
- [tabs.gradient_methods](#tabs-gradient-methods): `GradientMethodsWindow`
- [tabs.interactive_methods](#tabs-interactive-methods): `InteractiveMethodsWindow`
- [tabs.layout_dataset](#tabs-layout-dataset): `Labeler`
- [tabs.model_research](#tabs-model-research): `ModelResearchApp`
- [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml): `DatasetPreparationWindow`
- [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi): `SahiSlicerWindow`
- [tabs.threshold_methods](#tabs-threshold-methods): `ThresholdWindow`
- [tabs.traditional_ml_methods](#tabs-traditional-ml-methods): `TraditionalMLWindow`
- [tabs.viewing_dataset](#tabs-viewing-dataset): `ViewingDataset`
- [tabs.yolo_find_img](#tabs-yolo-find-img): `FindImagesWindow`
- [tabs.yolo_test_aeb](#tabs-yolo-test-aeb): `YoloInspectWindow`
- [ui.settings_ui](#ui-settings-ui): `SettingsWidget`

**Внешние/stdlib импорты:** `PIL`, `PyQt5.QtCore.PYQT_VERSION_STR`, `PyQt5.QtCore.QT_VERSION_STR`, `albumentations`, `cv2`, `matplotlib`, `numpy`, `onnxruntime`, `platform`, `skimage`, `sklearn`, `torch`, `ultralytics`, `yaml`.

**Объявления:**

- Класс [`ImagePlayer`](../main.py#L8) (`QMainWindow`)
- [`main()`](../main.py#L340)

<a id="main-ui"></a>

## main_ui

Исходник: [main_ui.py](../main_ui.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`StartupDialog`](../main_ui.py#L24) (`QDialog`)

<a id="path-setup"></a>

## path_setup

Исходник: [path_setup.py](../path_setup.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tabs.model_research](#tabs-model-research), [tabs.threshold_methods](#tabs-threshold-methods), [ui.settings_ui](#ui-settings-ui), [utils.presets](#utils-presets), [utils.settings](#utils-settings).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `sys`.

**Объявления:**

- [`setup_project_path()`](../path_setup.py#L7)
  Явно импортируют: [ui.settings_ui](#ui-settings-ui).
- [`get_project_root()`](../path_setup.py#L20)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.model_research](#tabs-model-research), [tabs.threshold_methods](#tabs-threshold-methods), [utils.presets](#utils-presets), [utils.settings](#utils-settings).

<a id="rename-data"></a>

## rename_data

Исходник: [rename_data.py](../rename_data.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `pathlib.Path`, `re`.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="tabs-dataset-analysis"></a>

## tabs.dataset_analysis

Исходник: [tabs/dataset_analysis.py](../tabs/dataset_analysis.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `TrainingImageLabel`, `setup_dataset_analysis_ui`
- [utils.dataset_audit_logic](#utils-dataset-audit-logic): `CLIP_AVAILABLE`, `DuplicateDetector`, `EmbeddingClusterThread`, `LeakageChecker`, `LossInspectionThread`, `ObjectDistributionHeatmap`, `ReportThread`

**Внешние/stdlib импорты:** `collections.Counter`, `cv2`, `dataset_analysis_ui.TrainingImageLabel`, `io`, `logging`, `matplotlib.pyplot`, `numpy`, `os`, `shutil`, `subprocess`, `sys`, `torch`, `traceback`, `ultralytics.YOLO`, `ultralytics.utils.LOGGER`.

**Объявления:**

- Класс [`YoloValThread`](../tabs/dataset_analysis.py#L27) (`QThread`)
- Класс [`DatasetAnalysisWindow`](../tabs/dataset_analysis.py#L253) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-deep-learning-methods"></a>

## tabs.deep_learning_methods

Исходник: [tabs/deep_learning_methods.py](../tabs/deep_learning_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_deep_learning_methods_ui`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QSpinBox`.

**Объявления:**

- Класс [`DeepLearningWindow`](../tabs/deep_learning_methods.py#L15) (`QMainWindow`)
- [`center_window(window, width=1200, height=800)`](../tabs/deep_learning_methods.py#L1134)

<a id="tabs-deep-learning-training"></a>

## tabs.deep_learning_training

Исходник: [tabs/deep_learning_training.py](../tabs/deep_learning_training.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_deep_learning_training_ui`

**Внешние/stdlib импорты:** `ultralytics.YOLO`.

**Объявления:**

- Класс [`YoloTrainWindow`](../tabs/deep_learning_training.py#L19) (`QMainWindow`)

<a id="tabs-demo-ensemble-methods"></a>

## tabs.demo_ensemble_methods

Исходник: [tabs/demo_ensemble_methods.py](../tabs/demo_ensemble_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_demo_ensemble_methods_ui`

**Внешние/stdlib импорты:** `traceback`.

**Объявления:**

- [`print_flush(*args, **kwargs)`](../tabs/demo_ensemble_methods.py#L6)
- Класс [`YoloInferThread`](../tabs/demo_ensemble_methods.py#L21) (`QThread`)
- Класс [`EnsembleThread`](../tabs/demo_ensemble_methods.py#L94) (`QThread`)
- Класс [`YoloDemoWindow`](../tabs/demo_ensemble_methods.py#L159) (`QMainWindow`)

<a id="tabs-gradient-methods"></a>

## tabs.gradient_methods

Исходник: [tabs/gradient_methods.py](../tabs/gradient_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_gradient_ui`
- [path_setup](#path-setup): `get_project_root`
- [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog): `GradientAutoSettingsDialog`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`GradientMethodsWindow`](../tabs/gradient_methods.py#L7) (`QMainWindow`)

<a id="tabs-interactive-methods"></a>

## tabs.interactive_methods

Исходник: [tabs/interactive_methods.py](../tabs/interactive_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_interactive_ui`

**Внешние/stdlib импорты:** `traceback`.

**Объявления:**

- Класс [`InteractiveMethodsWindow`](../tabs/interactive_methods.py#L7) (`QMainWindow`)

<a id="tabs-layout-dataset"></a>

## tabs.layout_dataset

Исходник: [tabs/layout_dataset.py](../tabs/layout_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_layout_dataset_ui`
- [utils.settings](#utils-settings): `settings`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `convert_to_grayscale`, `image_size_from_file`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `read_mask_with_fallback`, `resize_to_max_side`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `find_mask_for_image`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `canonicalize_obb_corners`, `load_annotations`, `load_annotations_for_image`, `save_annotations`, `save_annotations_for_image`
- [utils_new.smart_view_new](#utils-new-smart-view-new): `numpy_to_qpixmap`, `set_fill_color_button_preview`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `update_annotation_list`

**Внешние/stdlib импорты:** `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QColorDialog`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/layout_dataset.py#L33) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/layout_dataset.py#L78) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/layout_dataset.py#L102) (`QThread`)
- Класс [`Labeler`](../tabs/layout_dataset.py#L152) (`QMainWindow`)

<a id="tabs-model-research"></a>

## tabs.model_research

Исходник: [tabs/model_research.py](../tabs/model_research.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_model_research_ui`
- [path_setup](#path-setup): `get_project_root`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`ModelResearchApp`](../tabs/model_research.py#L8) (`QMainWindow`)

<a id="tabs-preparing-dataset-yaml"></a>

## tabs.preparing_dataset_yaml

Исходник: [tabs/preparing_dataset_yaml.py](../tabs/preparing_dataset_yaml.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_preparing_dataset_yaml_ui`
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new): `DatasetGeneratorThread`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_preparation_bridge](#utils-new-dataset-preparation-bridge): `mask_classes_from_rows`, `normalize_generation_pairs`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`, `convert_pairs_color`, `convert_pairs_to_8bit_png`, `select_pairs_for_8bit_png`, `select_pairs_for_conversion`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QTableWidgetItem`, `os`, `sys`, `yaml`.

**Объявления:**

- Класс [`DatasetPreparationWindow`](../tabs/preparing_dataset_yaml.py#L27) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-slice-dataset-sahi"></a>

## tabs.slice_dataset_sahi

Исходник: [tabs/slice_dataset_sahi.py](../tabs/slice_dataset_sahi.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_sahi_slicer_ui`

**Внешние/stdlib импорты:** `contextlib.redirect_stdout`, `cv2`, `json`, `math`, `numpy`, `os`, `random`, `sahi.slicing.slice_coco`, `shutil`, `sys`, `tempfile`, `traceback`.

**Объявления:**

- [`yolo_labels_to_coco_annotations(label_path, img_w, img_h, input_format='auto')`](../tabs/slice_dataset_sahi.py#L25)
- [`slice_dataset(coco_json_path, image_dir, output_dir, slice_size=640, overlap=0.2, min_area_ratio=0.3)`](../tabs/slice_dataset_sahi.py#L117)
- [`detect_dataset_format(coco_dict)`](../tabs/slice_dataset_sahi.py#L154)
- [`describe_segmentation(coco_dict)`](../tabs/slice_dataset_sahi.py#L161)
- [`subsample_negative_tiles(coco_dict, image_dir, keep_ratio=0.1, seed=42)`](../tabs/slice_dataset_sahi.py#L183)
- [`split_and_move(coco_dict, output_dir, train_ratio=0.8, seed=42, strip_segmentation=False)`](../tabs/slice_dataset_sahi.py#L224) — strip_segmentation: если True — удалить поле 'segmentation' из всех
- [`coco_to_yolo(coco_json_path, output_dir, class_names=None, annotation_format='bbox', verbose=True)`](../tabs/slice_dataset_sahi.py#L290)
- [`write_yolo_yaml(output_dir, class_names, channels=3)`](../tabs/slice_dataset_sahi.py#L410)
- [`visualize_samples(coco_json_path, output_dir, n=3)`](../tabs/slice_dataset_sahi.py#L425)
- Класс [`SliceWorker`](../tabs/slice_dataset_sahi.py#L477) (`QThread`)
- Класс [`SahiSlicerWindow`](../tabs/slice_dataset_sahi.py#L601) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-threshold-methods"></a>

## tabs.threshold_methods

Исходник: [tabs/threshold_methods.py](../tabs/threshold_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_threshold_ui`
- [path_setup](#path-setup): `get_project_root`
- [utils.image_io](#utils-image-io): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`, `save_annotations`
- [utils.presets](#utils-presets): `PresetManager`
- [utils.settings](#utils-settings): `settings`
- [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research): `AutoResearch`
- [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog): `AutoSettingsDialog`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `extract_annotations_from_mask`, `format_object_for_list`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology as apply_shared_morphology`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`, `normalize_method_name`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QInputDialog`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSlider`, `cv2`, `json`, `os`, `sys`.

**Объявления:**

- Класс [`_LegacyMorphologyRatioSlider`](../tabs/threshold_methods.py#L33) (`QSlider`)
- Класс [`ThresholdMorphologyPresetManager`](../tabs/threshold_methods.py#L54) (`PresetManager`)
- Класс [`ThresholdWindow`](../tabs/threshold_methods.py#L159) (`QMainWindow`)

<a id="tabs-traditional-ml-methods"></a>

## tabs.traditional_ml_methods

Исходник: [tabs/traditional_ml_methods.py](../tabs/traditional_ml_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_traditional_ml_ui`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`TraditionalMLWindow`](../tabs/traditional_ml_methods.py#L6) (`QMainWindow`)

<a id="tabs-viewing-dataset"></a>

## tabs.viewing_dataset

Исходник: [tabs/viewing_dataset.py](../tabs/viewing_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_viewing_dataset_ui`
- [utils.settings](#utils-settings): `settings`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `annotation_in_crop`, `class_id_for_annotation`, `crop_for_annotation`, `find_strong_overlaps`, `mask_roi_at_image_scale`, `normalized_annotation`, `prepare_annotation_preview`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `convert_to_grayscale`, `image_size_from_file`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `read_index_mask`, `read_mask_with_fallback`, `resize_to_max_side`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `find_mask_for_image`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `canonicalize_obb_corners`, `load_annotations`, `load_annotations_for_image`, `save_annotations`, `save_annotations_for_image`
- [utils_new.smart_view_new](#utils-new-smart-view-new): `numpy_to_qpixmap`, `set_fill_color_button_preview`

**Внешние/stdlib импорты:** `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPalette`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QShortcut`, `collections.OrderedDict`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/viewing_dataset.py#L42) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/viewing_dataset.py#L87) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/viewing_dataset.py#L111) (`QThread`)
- Класс [`ViewingDataset`](../tabs/viewing_dataset.py#L161) (`QMainWindow`)

<a id="tabs-yolo-find-img"></a>

## tabs.yolo_find_img

Исходник: [tabs/yolo_find_img.py](../tabs/yolo_find_img.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_yolo_find_img_ui`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`ClickableLabel`](../tabs/yolo_find_img.py#L11) (`QLabel`)
- Класс [`ScanThread`](../tabs/yolo_find_img.py#L75) (`QThread`)
- Класс [`FindImagesWindow`](../tabs/yolo_find_img.py#L252) (`QMainWindow, setup_yolo_find_img_ui`)

<a id="tabs-yolo-test-aeb"></a>

## tabs.yolo_test_aeb

Исходник: [tabs/yolo_test_aeb.py](../tabs/yolo_test_aeb.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [import_libs_methods_ui](#import-libs-methods-ui): `setup_yolo_ae_vae_ui`
- [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils): `ConvAutoencoder`, `ConvAutoencoderGN`, `ConvDecoder`, `ConvDecoderGN`, `ConvEncoder`, `ConvEncoderGN`, `ConvVAE`, `ConvVAEGN`, `UNetAE`, `UNetDecoder`, `UNetEncoder`, `UNetVAE`, `build_ae_model_from_cfg`, `detect_arch`, `preprocess_image_for_ae`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `aeb_transform_image`, `apply_object_morphology`, `cls_suffix_from_detection`, `crop_to_detection`, `detection_to_unified`, `detections_to_unified`, `longest_edge_angle_deg`, `render_ae_error_map`, `render_unified_annotations`, `rotate_with_optional_crop`, `yolo_result_to_detections`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`YoloAEBPresetManager`](../tabs/yolo_test_aeb.py#L55) (`PresetManager`)
- Класс [`YoloInspectWindow`](../tabs/yolo_test_aeb.py#L151) (`QMainWindow`)

<a id="ui-DL-predict-settings"></a>

## ui.DL_predict_settings

Исходник: [ui/DL_predict_settings.py](../ui/DL_predict_settings.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`BaseInferenceSettings`](../ui/DL_predict_settings.py#L3) (`QGroupBox`)
- Класс [`YOLOInferenceSettings`](../ui/DL_predict_settings.py#L27) (`BaseInferenceSettings`)
- Класс [`UNetInferenceSettings`](../ui/DL_predict_settings.py#L106) (`BaseInferenceSettings`)
- Класс [`DeepLabV3InferenceSettings`](../ui/DL_predict_settings.py#L447) (`BaseInferenceSettings`)
- Класс [`SegFormerInferenceSettings`](../ui/DL_predict_settings.py#L832) (`BaseInferenceSettings`)
- Класс [`SAMInferenceSettings`](../ui/DL_predict_settings.py#L869) (`BaseInferenceSettings`)
- Класс [`CustomONNXInferenceSettings`](../ui/DL_predict_settings.py#L912) (`BaseInferenceSettings`)

<a id="ui-DL-train-settings-base"></a>

## ui.DL_train_settings_base

Исходник: [ui/DL_train_settings_base.py](../ui/DL_train_settings_base.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.*`, `os`, `sys`, `yaml`.

**Объявления:**

- Класс [`BaseTrainWidget`](../ui/DL_train_settings_base.py#L8) (`QWidget`)
- Класс [`YOLOTrainWidget`](../ui/DL_train_settings_base.py#L149) (`BaseTrainWidget`)
- Класс [`UNetTrainWidget`](../ui/DL_train_settings_base.py#L386) (`BaseTrainWidget`)

<a id="ui-DL-train-settings-new-model"></a>

## ui.DL_train_settings_new_model

Исходник: [ui/DL_train_settings_new_model.py](../ui/DL_train_settings_new_model.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [ui.DL_train_settings_base](#ui-DL-train-settings-base): `BaseTrainWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.*`, `ast`, `os`, `sys`.

**Объявления:**

- Класс [`DeepLabV3TrainWidget`](../ui/DL_train_settings_new_model.py#L14) (`BaseTrainWidget`)
- Класс [`SegFormerTrainWidget`](../ui/DL_train_settings_new_model.py#L966) (`BaseTrainWidget`)
- Класс [`SAMTrainWidget`](../ui/DL_train_settings_new_model.py#L1101) (`BaseTrainWidget`)

<a id="ui-control-buttons"></a>

## ui.control_buttons

Исходник: [ui/control_buttons.py](../ui/control_buttons.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`ControlButtons`](../ui/control_buttons.py#L5) (`QWidget`)

<a id="ui-image-navigation"></a>

## ui.image_navigation

Исходник: [ui/image_navigation.py](../ui/image_navigation.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.settings](#utils-settings): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QEvent`, `PyQt5.QtCore.QObject`, `PyQt5.QtWidgets.QAbstractSpinBox`.

**Объявления:**

- Класс [`ImageNavigationWidget`](../ui/image_navigation.py#L6) (`QWidget`)
- [`parse_key_spec(spec)`](../ui/image_navigation.py#L95) — Парсит строку вида:
- Класс [`NavigationShortcutInstaller`](../ui/image_navigation.py#L133) (`QObject`)

<a id="ui-interactive-params-ui"></a>

## ui.interactive_params_ui

Исходник: [ui/interactive_params_ui.py](../ui/interactive_params_ui.py#L1).

**Прямые потребители:** [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`InteractiveParametersUI`](../ui/interactive_params_ui.py#L8) (`QWidget`)

<a id="ui-log-widget"></a>

## ui.log_widget

Исходник: [ui/log_widget.py](../ui/log_widget.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`LogWidget`](../ui/log_widget.py#L4) (`QWidget`)

<a id="ui-morphology-ui"></a>

## ui.morphology_ui

Исходник: [ui/morphology_ui.py](../ui/morphology_ui.py#L1).

**Прямые потребители:** [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `MorphologyStep`, `SUPPORTED_OPERATIONS`, `SUPPORTED_SHAPES`

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `__future__.annotations`.

**Объявления:**

- Класс [`MorphologyWidget`](../ui/morphology_ui.py#L35) (`QGroupBox`)

<a id="ui-pair-validation-dialog-new"></a>

## ui.pair_validation_dialog_new

Исходник: [ui/pair_validation_dialog_new.py](../ui/pair_validation_dialog_new.py#L1).

**Прямые потребители:** [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [utils.settings](#utils-settings): `settings`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `_read_with_fallback`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_review_view](#utils-new-dataset-review-view): `DatasetReviewGraphicsView`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `load_annotations`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `cv2`, `numpy`, `os`.

**Объявления:**

- [`_to_qpixmap(bgr)`](../ui/pair_validation_dialog_new.py#L23) — Convert a BGR preview to an owned QPixmap without scaling its pixels.
- [`_palette_color(class_id)`](../ui/pair_validation_dialog_new.py#L33) — Read current application class palette, with deterministic fallback.
- Класс [`AutoFitReviewView`](../ui/pair_validation_dialog_new.py#L44) (`DatasetReviewGraphicsView`)
- Класс [`PairValidationDialog`](../ui/pair_validation_dialog_new.py#L87) (`QDialog`)

<a id="ui-settings-ui"></a>

## ui.settings_ui

Исходник: [ui/settings_ui.py](../ui/settings_ui.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `setup_project_path`
- [utils.settings](#utils-settings): `DEFAULT_DATASET_CLASS_PALETTE_BGR`, `FONT_SIZE_PX_MAX`, `FONT_SIZE_PX_MIN`, `LINE_THICKNESS_PX_MAX`, `LINE_THICKNESS_PX_MIN`, `apply_theme`, `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `__future__.annotations`, `copy.deepcopy`.

**Объявления:**

- Класс [`SettingsWidget`](../ui/settings_ui.py#L28) (`QWidget`)

<a id="ui-tabs-dataset-analysis-ui"></a>

## ui_tabs.dataset_analysis_ui

Исходник: [ui_tabs/dataset_analysis_ui.py](../ui_tabs/dataset_analysis_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSize`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QPixmap`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.backends.backend_qt5agg.NavigationToolbar2QT`, `matplotlib.figure.Figure`.

**Объявления:**

- Класс [`TrainingImageLabel`](../ui_tabs/dataset_analysis_ui.py#L12) (`QLabel`)
- [`_make_page(toolbox, title, description, content, buttons, min_height=240, content_stretch=1)`](../ui_tabs/dataset_analysis_ui.py#L49)
- [`_mk_min_spin(rng, suffix='', special='Не задано', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L83)
- [`_mk_max_spin(rng, suffix='', special='Без ограничений', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L94)
- [`_mk_min_dspin(rng, decimals=2, suffix='', special='Не задано', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L105)
- [`_mk_max_dspin(rng, decimals=2, suffix='', special='Без ограничений', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L118)
- [`setup_dataset_analysis_ui(parent)`](../ui_tabs/dataset_analysis_ui.py#L131)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-deep-learning-methods-ui"></a>

## ui_tabs.deep_learning_methods_ui

Исходник: [ui_tabs/deep_learning_methods_ui.py](../ui_tabs/deep_learning_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_deep_learning_methods_ui(parent)`](../ui_tabs/deep_learning_methods_ui.py#L5)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-deep-learning-training-ui"></a>

## ui_tabs.deep_learning_training_ui

Исходник: [ui_tabs/deep_learning_training_ui.py](../ui_tabs/deep_learning_training_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_deep_learning_training_ui(parent)`](../ui_tabs/deep_learning_training_ui.py#L4)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-demo-ensemble-methods-ui"></a>

## ui_tabs.demo_ensemble_methods_ui

Исходник: [ui_tabs/demo_ensemble_methods_ui.py](../ui_tabs/demo_ensemble_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_demo_ensemble_methods_ui(parent)`](../ui_tabs/demo_ensemble_methods_ui.py#L3)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-gradient-methods-ui"></a>

## ui_tabs.gradient_methods_ui

Исходник: [ui_tabs/gradient_methods_ui.py](../ui_tabs/gradient_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_gradient_ui(parent)`](../ui_tabs/gradient_methods_ui.py#L3)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-interactive-methods-ui"></a>

## ui_tabs.interactive_methods_ui

Исходник: [ui_tabs/interactive_methods_ui.py](../ui_tabs/interactive_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [ui.interactive_params_ui](#ui-interactive-params-ui): `InteractiveParametersUI`

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_interactive_ui(parent)`](../ui_tabs/interactive_methods_ui.py#L4) — Создаёт UI для вкладки интерактивных методов сегментации.
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-layout-dataset-ui"></a>

## ui_tabs.layout_dataset_ui

Исходник: [ui_tabs/layout_dataset_ui.py](../ui_tabs/layout_dataset_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_layout_dataset_ui(parent)`](../ui_tabs/layout_dataset_ui.py#L6)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-model-research-ui"></a>

## ui_tabs.model_research_ui

Исходник: [ui_tabs/model_research_ui.py](../ui_tabs/model_research_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`ModelResearchWidget`](../ui_tabs/model_research_ui.py#L4) (`QWidget`)
- [`setup_model_research_ui(parent)`](../ui_tabs/model_research_ui.py#L78) — Настройка UI для ModelResearchApp.
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-preparing-dataset-yaml-ui"></a>

## ui_tabs.preparing_dataset_yaml_ui

Исходник: [ui_tabs/preparing_dataset_yaml_ui.py](../ui_tabs/preparing_dataset_yaml_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [ui.control_buttons](#ui-control-buttons): `ControlButtons`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`_section(title, description=None)`](../ui_tabs/preparing_dataset_yaml_ui.py#L14)
- [`_page()`](../ui_tabs/preparing_dataset_yaml_ui.py#L28)
- [`_wrap_row(*widgets)`](../ui_tabs/preparing_dataset_yaml_ui.py#L41)
- [`setup_preparing_dataset_yaml_ui(parent)`](../ui_tabs/preparing_dataset_yaml_ui.py#L50)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-slice-dataset-sahi-ui"></a>

## ui_tabs.slice_dataset_sahi_ui

Исходник: [ui_tabs/slice_dataset_sahi_ui.py](../ui_tabs/slice_dataset_sahi_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_sahi_slicer_ui(parent)`](../ui_tabs/slice_dataset_sahi_ui.py#L5) — SAHI slicer UI: слева — пары/валидация, справа — SAHI.
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-threshold-methods-ui"></a>

## ui_tabs.threshold_methods_ui

Исходник: [ui_tabs/threshold_methods_ui.py](../ui_tabs/threshold_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [utils_new.dataset_review_view](#utils-new-dataset-review-view): `DatasetReviewGraphicsView`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_threshold_ui(parent)`](../ui_tabs/threshold_methods_ui.py#L17) — Создаёт UI для пороговых методов внутри QMainWindow.
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-traditional-ml-methods-ui"></a>

## ui_tabs.traditional_ml_methods_ui

Исходник: [ui_tabs/traditional_ml_methods_ui.py](../ui_tabs/traditional_ml_methods_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_traditional_ml_ui(parent)`](../ui_tabs/traditional_ml_methods_ui.py#L4)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-viewing-dataset-ui"></a>

## ui_tabs.viewing_dataset_ui

Исходник: [ui_tabs/viewing_dataset_ui.py](../ui_tabs/viewing_dataset_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_viewing_dataset_ui(parent)`](../ui_tabs/viewing_dataset_ui.py#L4) — Создаёт UI для просмотра датасета с поддержкой аннотаций, масок и YAML.
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="ui-tabs-yolo-find-img-ui"></a>

## ui_tabs.yolo_find_img_ui

Исходник: [ui_tabs/yolo_find_img_ui.py](../ui_tabs/yolo_find_img_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`setup_yolo_find_img_ui`](../ui_tabs/yolo_find_img_ui.py#L4) (`object`)

<a id="ui-tabs-yolo-test-aeb-ui"></a>

## ui_tabs.yolo_test_aeb_ui

Исходник: [ui_tabs/yolo_test_aeb_ui.py](../ui_tabs/yolo_test_aeb_ui.py#L1).

**Прямые потребители:** [import_libs_methods_ui](#import-libs-methods-ui).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [utils_new.dataset_review_view](#utils-new-dataset-review-view): `DatasetReviewGraphicsView`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_yolo_ae_vae_ui(parent)`](../ui_tabs/yolo_test_aeb_ui.py#L19)
  Явно импортируют: [import_libs_methods_ui](#import-libs-methods-ui).

<a id="utils-DatasetGenerator"></a>

## utils.DatasetGenerator

Исходник: [utils/DatasetGenerator.py](../utils/DatasetGenerator.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [utils.annotation_converter](#utils-annotation-converter): `SUBTYPE_KEEP`, `convert_annotations`
- [utils.coco_export](#utils-coco-export): `export_coco_from_labels`, `export_coco_from_masks`
- [utils.image_io](#utils-image-io): `FORMAT_HINT_AUTO`, `convert_segment_masks_to_yolo_seg_manual`, `load_annotations`, `read_mask_with_fallback`, `save_annotations`
- [utils.unet_export](#utils-unet-export): `export_masks_from_annotations`, `write_unet_yaml`

**Внешние/stdlib импорты:** `traceback`, `ultralytics`.

**Объявления:**

- Класс [`DatasetGeneratorThread`](../utils/DatasetGenerator.py#L64) (`QThread`)
  - [`DatasetGeneratorThread.__init__(self, images_folder, labels_folder, output_folder, pairs, train_pct, val_pct, test_pct, target_size, augmentations, has_augmentations, aug_multiplier, class_names, class_mapping, bg_color, input_type, masks_folder, num_classes, mask_class_remap, included_orig_ids=None, included_class_ids=None, total_orig_classes=1, resize_mode='fixed', include_empty_annotations=False, delete_source_after_success=False, input_format=FORMAT_HINT_AUTO, out_yolo_txt=True, out_yolo_subtype=SUBTYPE_KEEP, out_yolo_obb_format='corners', out_masks=False, out_masks_offset=1, out_unet_yaml=False, out_coco=False, mask_pixel_offset_in=1, min_ann_area_px=0)`](../utils/DatasetGenerator.py#L69)
  - [`DatasetGeneratorThread.cancel(self)`](../utils/DatasetGenerator.py#L140)
  - [`DatasetGeneratorThread.log(self, message)`](../utils/DatasetGenerator.py#L143)
  - [`DatasetGeneratorThread._create_transforms(self)`](../utils/DatasetGenerator.py#L146)
  - [`DatasetGeneratorThread._compute_image_hash(self, img)`](../utils/DatasetGenerator.py#L166)
  - [`DatasetGeneratorThread._annotation_area_px(self, ann, w, h)`](../utils/DatasetGenerator.py#L169)
  - [`DatasetGeneratorThread._filter_small_annotations(self, anns, w, h)`](../utils/DatasetGenerator.py#L198)
  - [`DatasetGeneratorThread._load_mask(self, path)`](../utils/DatasetGenerator.py#L214)
  - [`DatasetGeneratorThread._resize_mask(self, mask, target_size, resize_mode)`](../utils/DatasetGenerator.py#L238)
  - [`DatasetGeneratorThread._dense_class_mapping(self)`](../utils/DatasetGenerator.py#L262)
  - [`DatasetGeneratorThread._dense_mask_class_mapping(self)`](../utils/DatasetGenerator.py#L273)
  - [`DatasetGeneratorThread._apply_class_mapping(self, annotations)`](../utils/DatasetGenerator.py#L285)
  - [`DatasetGeneratorThread._remap_mask_pixels(self, mask)`](../utils/DatasetGenerator.py#L302)
  - [`DatasetGeneratorThread._write_label(self, anns, out_path, w, h)`](../utils/DatasetGenerator.py#L335)
  - [`DatasetGeneratorThread._generate_all_pairs(self, output_path)`](../utils/DatasetGenerator.py#L344)
  - [`DatasetGeneratorThread._process_pair_labels(self, pair, temp_images, temp_labels, all_pairs, total_to_generate)`](../utils/DatasetGenerator.py#L404)
  - [`DatasetGeneratorThread._process_pair_masks(self, pair, temp_images, temp_masks, all_pairs, total_to_generate)`](../utils/DatasetGenerator.py#L527)
  - [`DatasetGeneratorThread._process_pair_images(self, pair, temp_images, all_pairs, total_to_generate)`](../utils/DatasetGenerator.py#L622)
  - [`DatasetGeneratorThread._resize_with_annotations(self, img, anns)`](../utils/DatasetGenerator.py#L704)
  - [`DatasetGeneratorThread._convert_masks_to_yolo_labels(self, temp_masks, temp_labels, all_pairs)`](../utils/DatasetGenerator.py#L718)
  - [`DatasetGeneratorThread._split_and_move(self, output_path, all_pairs)`](../utils/DatasetGenerator.py#L781)
  - [`DatasetGeneratorThread._write_classes_txt(self, output_path)`](../utils/DatasetGenerator.py#L831)
  - [`DatasetGeneratorThread._write_data_yaml(self, output_path)`](../utils/DatasetGenerator.py#L840)
  - [`DatasetGeneratorThread._export_masks(self, output_path)`](../utils/DatasetGenerator.py#L857)
  - [`DatasetGeneratorThread._export_unet_yaml(self, output_path)`](../utils/DatasetGenerator.py#L876)
  - [`DatasetGeneratorThread._export_coco(self, output_path)`](../utils/DatasetGenerator.py#L886)
  - [`DatasetGeneratorThread._delete_processed_sources(self)`](../utils/DatasetGenerator.py#L909)
  - [`DatasetGeneratorThread.run(self)`](../utils/DatasetGenerator.py#L947)

<a id="utils-analysis"></a>

## utils.analysis

Исходник: [utils/analysis.py](../utils/analysis.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`
- [utils.prepare](#utils-prepare): `add_meanap_metrics`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`distance_between_presets(p1, p2)`](../utils/analysis.py#L8) — Расстояние между двумя пресетами (Series). Учитывает метод, cluster, cluster_co.
- [`diverse_top_k(df, k=5, score_col='pred_f1', candidate_pool_size=None)`](../utils/analysis.py#L17) — Жадный выбор топ-K разнообразных пресетов.
- [`analyze(df, output_dir=None, skip_metrics=False, iou_thresholds=None, use_diversity=True, use_iou_penalty=False, penalty_alpha=0.5, per_method=True, cluster_col='cluster', cluster_co_col='cluster_co')`](../utils/analysis.py#L45) — Анализ для каждого метода отдельно.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`global_top_diverse(df, k=10, score_col='actual_f1', method_col='method', params_col='params', invert_col='invert', close_col='close_factor', open_col='open_factor')`](../utils/analysis.py#L205) — Жадный выбор топ-k разнообразных пресетов из глобального датасета.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-annotation-converter"></a>

## utils.annotation_converter

Исходник: [utils/annotation_converter.py](../utils/annotation_converter.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.image_io](#utils-image-io): `canonicalize_obb_corners`

**Внешние/stdlib импорты:** `numpy`.

**Объявления:**

- [`to_detect(ann)`](../utils/annotation_converter.py#L21) — Любой тип → detect (axis-aligned bbox).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`to_obb(ann)`](../utils/annotation_converter.py#L42) — Любой тип → OBB (4 канонических угла).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`to_segment(ann)`](../utils/annotation_converter.py#L70) — Любой тип → segment (полигон).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`convert_annotations(annotations, subtype)`](../utils/annotation_converter.py#L93) — Применяет конвертер subtype ко всем аннотациям.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

<a id="utils-aug-utils"></a>

## utils.aug_utils

Исходник: [utils/aug_utils.py](../utils/aug_utils.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`repair_obb_to_rectangle(pts_norm_flat)`](../utils/aug_utils.py#L18) — Принимает список из 8 чисел (4 точки x1,y1,x2,y2,x3,y3,x4,y4) в нормализованных координатах.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`log_annotations_summary(prefix, annotations, max_items=5)`](../utils/aug_utils.py#L41) — Выводит краткую информацию об аннотациях (тип, класс, кол-во точек)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`_log_verbose_annotations(prefix, annotations)`](../utils/aug_utils.py#L66) — Выводит полные данные аннотаций только при VERBOSE_ANNOTATION_LOG=True
- [`_get_fill_params(fill_color)`](../utils/aug_utils.py#L79)
- [`apply_affine_to_points_np(points, M_3x3, orig_w, orig_h, new_w, new_h)`](../utils/aug_utils.py#L87) — Применяет аффинное преобразование (3x3) к массиву точек (Nx2) в пикселях
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`compose_affine_matrices(matrices_3x3)`](../utils/aug_utils.py#L95)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`resize_image_and_annotations(image, annotations, target_size, fill_color=(0, 0, 0))`](../utils/aug_utils.py#L102) — Ресайз с сохранением пропорций (LongestMaxSize) + паддинг до target_size.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`resize_image_and_annotations_stretch(image, annotations, target_size)`](../utils/aug_utils.py#L169) — Растягивает изображение до квадрата target_size x target_size без сохранения пропорций.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`clip_annotations(annotations, epsilon=1e-07)`](../utils/aug_utils.py#L209)
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-augmentation-base"></a>

## utils.augmentation_base

Исходник: [utils/augmentation_base.py](../utils/augmentation_base.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`apply_simple_augmentations(image, annotations, aug_config)`](../utils/augmentation_base.py#L16) — Применяет горизонтальный/вертикальный флип и повороты на 90°.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`_flip_horizontal(img, ann)`](../utils/augmentation_base.py#L29)
- [`_flip_vertical(img, ann)`](../utils/augmentation_base.py#L51)
- [`_rotate_90_multiple(img, ann, k)`](../utils/augmentation_base.py#L73)
- [`_rotate_90(img, ann)`](../utils/augmentation_base.py#L78)
- [`apply_simple_augmentations_mask(image, mask, aug_config)`](../utils/augmentation_base.py#L107) — Применяет простые аугментации к изображению и маске (флип, поворот 90°) с помощью OpenCV.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-augmentation-random"></a>

## utils.augmentation_random

Исходник: [utils/augmentation_random.py](../utils/augmentation_random.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`_random_crop_factor(aug_config)`](../utils/augmentation_random.py#L24) — Возвращает случайный scale для crop в [min_scale, 1.0].
- [`_pick_crop_box(w, h, aug_config)`](../utils/augmentation_random.py#L37) — Случайный прямоугольник кропа внутри (w, h).
- [`_crop_annotations(annotations, x0, y0, cw, ch, prev_w, prev_h)`](../utils/augmentation_random.py#L52) — Пересчитывает нормализованные аннотации после кропа.
- [`apply_complex_augmentations(image, annotations, aug_config, border_color=(0, 0, 0))`](../utils/augmentation_random.py#L118) — Применяет случайный поворот, shear и кроп к изображению и аннотациям.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`apply_complex_augmentations_mask(image, mask, aug_config, border_color=(0, 0, 0))`](../utils/augmentation_random.py#L265) — Применяет случайный поворот, shear и кроп к изображению и маске.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-coco-export"></a>

## utils.coco_export

Исходник: [utils/coco_export.py](../utils/coco_export.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.image_io](#utils-image-io): `FORMAT_HINT_AUTO`, `load_annotations`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `cv2`, `json`, `numpy`, `os`, `pathlib.Path`.

**Объявления:**

- [`_polygon_area_abs(pts_flat)`](../utils/coco_export.py#L37) — Площадь полигона (шнуровка), pts_flat — [x1,y1,x2,y2,...] в абс. пикселях.
- [`_ann_to_coco_entry(ann, image_id, ann_id, w, h)`](../utils/coco_export.py#L50) — canonical-аннотация → словарь COCO. Возвращает dict или None.
- [`_categories_from_class_names(class_names)`](../utils/coco_export.py#L119) — class_names: {yolo_cls_id: name} — плотная нумерация 0..N-1.
- [`_images_for_split(output_path, split)`](../utils/coco_export.py#L134) — Список изображений в images/{split}. Возвращает [(img_path, w, h)].
- [`export_coco_from_labels(output_path, split, class_names, log_cb=None)`](../utils/coco_export.py#L152) — Читает labels/{split}/*.txt, пишет annotations_{split}.json.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).
- [`_mask_to_segment_polygons(mask, pixel_value, w, h)`](../utils/coco_export.py#L220) — Возвращает список полигонов (каждый — плоский список абс. координат)
- [`export_coco_from_masks(output_path, split, class_names, mask_pixel_offset=1, log_cb=None)`](../utils/coco_export.py#L246) — Читает masks/{split}/*.png, делает findContours, пишет
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

<a id="utils-coco-io"></a>

## utils.coco_io

Исходник: [utils/coco_io.py](../utils/coco_io.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.image_io](#utils-image-io): `canonicalize_obb_corners`, `save_annotations`

**Внешние/stdlib импорты:** `cv2`, `json`, `numpy`, `os`, `pathlib.Path`.

**Объявления:**

- [`_bbox_to_detect(bbox, w, h)`](../utils/coco_io.py#L18) — COCO bbox [x, y, bw, bh] в абс. пикселях → canonical detect.
- [`_segmentation_to_segment(seg, w, h)`](../utils/coco_io.py#L28) — COCO segmentation: [poly1, poly2, ...], каждый poly — плоский список абс. координат.
- [`_is_polygon_rectangle(poly_abs, tol=2.0)`](../utils/coco_io.py#L48) — Эвристика: 4 точки, стороны примерно параллельны друг другу,
- [`load_coco(json_path, log_cb=None)`](../utils/coco_io.py#L77) — Загружает COCO JSON.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`write_yolo_txt_for_coco_items(items, output_labels_dir, class_remap=None, log_cb=None, name_prefix='')`](../utils/coco_io.py#L207) — Записывает .txt-файлы для items, полученных из load_coco.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-dataset-analysis-logic"></a>

## utils.dataset_analysis_logic

Исходник: [utils/dataset_analysis_logic.py](../utils/dataset_analysis_logic.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `PIL.Image`, `cv2`, `matplotlib.pyplot`, `numpy`, `os`.

**Объявления:**

- [`get_image_brightness(path)`](../utils/dataset_analysis_logic.py#L34) — Средняя яркость изображения (grayscale mean, 0..255).
- [`clear_brightness_cache()`](../utils/dataset_analysis_logic.py#L61) — Сброс кэша яркости (например, после удаления файлов).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`_safe_progress(progress_cb, i, total, step=25)`](../utils/dataset_analysis_logic.py#L69)
- [`_check_cancel(cancel_check)`](../utils/dataset_analysis_logic.py#L74)
- Класс [`FilterParams`](../utils/dataset_analysis_logic.py#L84)
  - [`FilterParams.__init__(self, min_w=0, max_w=0, min_h=0, max_h=0, min_brightness=-1, max_brightness=-1, min_objs=0, max_objs=0, min_area=0, max_area=0, min_area_pct=0.0, max_area_pct=0.0, min_aspect=0.0, max_aspect=0.0)`](../utils/dataset_analysis_logic.py#L104)
- Класс [`DatasetAnalyzer`](../utils/dataset_analysis_logic.py#L136)
  - [`DatasetAnalyzer.__init__(self, dataset_type=0, input_format='auto')`](../utils/dataset_analysis_logic.py#L137)
  - [`DatasetAnalyzer.set_dataset_type(self, dt)`](../utils/dataset_analysis_logic.py#L141)
  - [`DatasetAnalyzer.set_input_format(self, fmt)`](../utils/dataset_analysis_logic.py#L144)
  - [`DatasetAnalyzer.get_included_class_ids(class_table)`](../utils/dataset_analysis_logic.py#L151)
  - [`DatasetAnalyzer.get_class_names_by_orig_id(class_table)`](../utils/dataset_analysis_logic.py#L163)
  - [`DatasetAnalyzer.get_class_names_included(class_table)`](../utils/dataset_analysis_logic.py#L175)
  - [`DatasetAnalyzer.filter_pairs(self, pairs, included_class_ids, total_classes, params, progress_cb=None, include_empty=False)`](../utils/dataset_analysis_logic.py#L193)
  - [`DatasetAnalyzer._load_annotations_for_pair(self, lbl_path, included_class_ids)`](../utils/dataset_analysis_logic.py#L342)
  - [`DatasetAnalyzer._annotation_area(self, ann, w, h)`](../utils/dataset_analysis_logic.py#L365)
  - [`DatasetAnalyzer.render_metrics(metrics_summary_label, class_stats_table, ax_all, ax_classes, canvas, filtered_pairs, all_pairs, class_counts, total_objects, object_areas, class_areas, name_by_orig, ax_pie=None, ax_bar=None, ax_area_pct=None, object_areas_pct=None)`](../utils/dataset_analysis_logic.py#L402)
  - [`DatasetAnalyzer.export_subset(filtered_pairs, out_dir, class_names, progress_cb=None, log_cb=None, included_class_ids=None)`](../utils/dataset_analysis_logic.py#L578)
- Класс [`ClassImbalanceAnalyzer`](../utils/dataset_analysis_logic.py#L680)
  - [`ClassImbalanceAnalyzer.analyze(class_counts)`](../utils/dataset_analysis_logic.py#L682)
- Класс [`ResolutionAnalyzer`](../utils/dataset_analysis_logic.py#L736)
  - [`ResolutionAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils/dataset_analysis_logic.py#L738)
- Класс [`BrightnessAnalyzer`](../utils/dataset_analysis_logic.py#L775)
  - [`BrightnessAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils/dataset_analysis_logic.py#L780)
- Класс [`TrainingResultsLoader`](../utils/dataset_analysis_logic.py#L811)
  - [`TrainingResultsLoader.scan(folder)`](../utils/dataset_analysis_logic.py#L839)
  - [`TrainingResultsLoader.summary_text(data)`](../utils/dataset_analysis_logic.py#L898)

<a id="utils-dataset-audit-logic"></a>

## utils.dataset_audit_logic

Исходник: [utils/dataset_audit_logic.py](../utils/dataset_audit_logic.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** `PIL.Image`, `collections.Counter`, `collections.defaultdict`, `cv2`, `datetime.datetime`, `hashlib`, `hdbscan`, `imagehash`, `numpy`, `open_clip`, `os`, `sklearn.cluster.DBSCAN`, `sklearn.decomposition.PCA`, `torch`, `transformers.CLIPModel`, `transformers.CLIPProcessor`, `ultralytics.YOLO`, `umap`.

**Объявления:**

- [`_safe_progress(progress_cb, i, total, step=25)`](../utils/dataset_audit_logic.py#L61)
- [`_check_cancel(cancel_check)`](../utils/dataset_audit_logic.py#L66)
- [`detect_device()`](../utils/dataset_audit_logic.py#L73) — Автоопределение устройства: 'cuda:0', если доступно, иначе 'cpu'.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- Класс [`DuplicateDetector`](../utils/dataset_audit_logic.py#L87)
  - [`DuplicateDetector._file_md5(path, chunk_size=1 << 20)`](../utils/dataset_audit_logic.py#L89)
  - [`DuplicateDetector._phash(path, hash_size=8)`](../utils/dataset_audit_logic.py#L100)
  - [`DuplicateDetector.find_duplicates(pairs, phash_threshold=8, progress_cb=None, cancel_check=None)`](../utils/dataset_audit_logic.py#L111)
- Класс [`LeakageChecker`](../utils/dataset_audit_logic.py#L166)
  - [`LeakageChecker._infer_split(path)`](../utils/dataset_audit_logic.py#L170)
  - [`LeakageChecker.check(duplicate_groups, pairs)`](../utils/dataset_audit_logic.py#L181)
- Класс [`ObjectDistributionHeatmap`](../utils/dataset_audit_logic.py#L205)
  - [`ObjectDistributionHeatmap.compute(pairs, dataset_type=0, grid_size=16, progress_cb=None, cancel_check=None)`](../utils/dataset_audit_logic.py#L207)
- Класс [`LossInspectionThread`](../utils/dataset_audit_logic.py#L280) (`QThread`)
  - [`LossInspectionThread.__init__(self, pairs, model_path, dataset_type=0, iou_threshold=0.3, conf_threshold=0.25, imgsz=640, device=None, input_format='auto', parent=None)`](../utils/dataset_audit_logic.py#L285)
  - [`LossInspectionThread.cancel(self)`](../utils/dataset_audit_logic.py#L299)
  - [`LossInspectionThread._iou_xyxy(a, b)`](../utils/dataset_audit_logic.py#L303)
  - [`LossInspectionThread.run(self)`](../utils/dataset_audit_logic.py#L314)
- Класс [`EmbeddingClusterThread`](../utils/dataset_audit_logic.py#L536) (`QThread`)
  - [`EmbeddingClusterThread.__init__(self, pairs, batch_size=16, device=None, parent=None)`](../utils/dataset_audit_logic.py#L541)
  - [`EmbeddingClusterThread.cancel(self)`](../utils/dataset_audit_logic.py#L548)
  - [`EmbeddingClusterThread.run(self)`](../utils/dataset_audit_logic.py#L551)
- Класс [`ReportThread`](../utils/dataset_audit_logic.py#L695) (`QThread`)
  - [`ReportThread.__init__(self, data, save_path, sections, parent=None)`](../utils/dataset_audit_logic.py#L698)
  - [`ReportThread.run(self)`](../utils/dataset_audit_logic.py#L704)

<a id="utils-dataset-validation"></a>

## utils.dataset_validation

Исходник: [utils/dataset_validation.py](../utils/dataset_validation.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [utils.smart_view](#utils-smart-view): `SmartGraphicsView`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`ImageStats`](../utils/dataset_validation.py#L21)
  - [`ImageStats.empty_color_stats()`](../utils/dataset_validation.py#L23)
  - [`ImageStats.empty_bit_stats()`](../utils/dataset_validation.py#L27)
  - [`ImageStats.detect_color_type(img)`](../utils/dataset_validation.py#L32)
  - [`ImageStats.detect_bit_type(img, img_path)`](../utils/dataset_validation.py#L49)
  - [`ImageStats.to_uint8(img)`](../utils/dataset_validation.py#L69)
  - [`ImageStats.to_display_bgr(img)`](../utils/dataset_validation.py#L82)
  - [`ImageStats.to_gray(img)`](../utils/dataset_validation.py#L95)
  - [`ImageStats.to_bgr3(img)`](../utils/dataset_validation.py#L108)
  - [`ImageStats.format_stats_text(color_stats, bit_stats, has_pairs)`](../utils/dataset_validation.py#L121)
- Класс [`PairValidator`](../utils/dataset_validation.py#L153)
  - [`PairValidator.__init__(self, dataset_type=0, input_format=INPUT_FORMAT_AUTO)`](../utils/dataset_validation.py#L165)
  - [`PairValidator.set_dataset_type(self, dataset_type)`](../utils/dataset_validation.py#L171)
  - [`PairValidator.set_input_format(self, fmt)`](../utils/dataset_validation.py#L175)
  - [`PairValidator.clear_cache(self)`](../utils/dataset_validation.py#L179)
  - [`PairValidator.validate_full(self, pair)`](../utils/dataset_validation.py#L182)
  - [`PairValidator.validate(self, pair)`](../utils/dataset_validation.py#L190)
  - [`PairValidator.validate_all(self, pairs, progress_cb=None)`](../utils/dataset_validation.py#L194)
  - [`PairValidator.recompute_stats(self, pairs)`](../utils/dataset_validation.py#L209)
  - [`PairValidator._validate_impl(self, pair)`](../utils/dataset_validation.py#L218)
  - [`PairValidator.detect_annotation_type(label_path, format_hint=INPUT_FORMAT_AUTO)`](../utils/dataset_validation.py#L398)
  - [`PairValidator.check_pair_type_consistency()`](../utils/dataset_validation.py#L440)
- [`_list_images(folder)`](../utils/dataset_validation.py#L447)
- [`scan_detection_pairs(images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO)`](../utils/dataset_validation.py#L456)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`scan_segmentation_pairs(images_folder, masks_folder, log_cb=None)`](../utils/dataset_validation.py#L475)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`scan_image_only_pairs(images_folder)`](../utils/dataset_validation.py#L509)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`select_pairs_for_conversion(pairs, validator, target)`](../utils/dataset_validation.py#L516)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`convert_pairs_color(pairs, target, progress_cb=None, log_cb=None)`](../utils/dataset_validation.py#L530)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`select_pairs_for_8bit_png(pairs, validator)`](../utils/dataset_validation.py#L557) — Возвращает (all_pairs, need_convert).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None, skip_masks=False)`](../utils/dataset_validation.py#L571) — skip_masks: если True — pairs трактуются как pairs из (img, mask),
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`build_pair_display(pair, validator, dataset_type, include_empty)`](../utils/dataset_validation.py#L616)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`draw_annotations_on_image(img, annotations)`](../utils/dataset_validation.py#L672)
- [`apply_mask_overlay(img, mask)`](../utils/dataset_validation.py#L699) — Накладывает цветной overlay.
- Класс [`PairValidationDialog`](../utils/dataset_validation.py#L722) (`QDialog`)
  - [`PairValidationDialog.__init__(self, parent, pair, dataset_type, validator)`](../utils/dataset_validation.py#L723)
- Класс [`DatasetLoaderMixin`](../utils/dataset_validation.py#L805)
  - [`DatasetLoaderMixin._after_load_hook(self)`](../utils/dataset_validation.py#L827)
  - [`DatasetLoaderMixin._after_type_change_hook(self, new_type)`](../utils/dataset_validation.py#L834)
  - [`DatasetLoaderMixin._sync_all_states(self, *_)`](../utils/dataset_validation.py#L839)
  - [`DatasetLoaderMixin._refresh_mask_pixel_column(self)`](../utils/dataset_validation.py#L855)
  - [`DatasetLoaderMixin.log(self, message)`](../utils/dataset_validation.py#L865)
  - [`DatasetLoaderMixin._log_debug(self, msg)`](../utils/dataset_validation.py#L874)
  - [`DatasetLoaderMixin._log_exception(self, context, exc=None)`](../utils/dataset_validation.py#L881)
  - [`DatasetLoaderMixin._copy_selected_rows_to_clipboard(self)`](../utils/dataset_validation.py#L893)
  - [`DatasetLoaderMixin._on_file_list_double_clicked(self, item)`](../utils/dataset_validation.py#L901)
  - [`DatasetLoaderMixin.select_images_folder(self)`](../utils/dataset_validation.py#L913)
  - [`DatasetLoaderMixin.select_labels_folder(self)`](../utils/dataset_validation.py#L922)
  - [`DatasetLoaderMixin.load_from_yaml(self)`](../utils/dataset_validation.py#L943)
  - [`DatasetLoaderMixin._load_yolo_yaml_from_path(self, file_path)`](../utils/dataset_validation.py#L960)
  - [`DatasetLoaderMixin.load_from_coco(self)`](../utils/dataset_validation.py#L1005)
  - [`DatasetLoaderMixin.add_from_coco(self)`](../utils/dataset_validation.py#L1008)
  - [`DatasetLoaderMixin._handle_load_coco(self, replace)`](../utils/dataset_validation.py#L1011)
  - [`DatasetLoaderMixin._ask_coco_source(self)`](../utils/dataset_validation.py#L1031)
  - [`DatasetLoaderMixin._find_coco_json_in_folder(self, folder)`](../utils/dataset_validation.py#L1047)
  - [`DatasetLoaderMixin._get_coco_tmp_labels_dir(self, first_json_path=None)`](../utils/dataset_validation.py#L1063)
  - [`DatasetLoaderMixin._load_coco_from_paths(self, json_paths, replace=True)`](../utils/dataset_validation.py#L1073)
  - [`DatasetLoaderMixin.load_from_unet_yaml(self)`](../utils/dataset_validation.py#L1203)
  - [`DatasetLoaderMixin._load_unet_from_path(self, yaml_path)`](../utils/dataset_validation.py#L1210)
  - [`DatasetLoaderMixin._apply_loaded_pairs(self, pairs, input_type, input_format, class_names_from_yaml=None, class_names_override=None, source_label='')`](../utils/dataset_validation.py#L1249)
  - [`DatasetLoaderMixin.scan_pairs(self)`](../utils/dataset_validation.py#L1303)
  - [`DatasetLoaderMixin._validate_all_pairs(self)`](../utils/dataset_validation.py#L1368)
  - [`DatasetLoaderMixin._recompute_color_stats(self)`](../utils/dataset_validation.py#L1389)
  - [`DatasetLoaderMixin._update_color_stats_label(self)`](../utils/dataset_validation.py#L1395)
  - [`DatasetLoaderMixin._open_pair_validation(self, pair)`](../utils/dataset_validation.py#L1405)
  - [`DatasetLoaderMixin._make_list_item(self, text, tag=None, error_msg='')`](../utils/dataset_validation.py#L1431)
  - [`DatasetLoaderMixin._rebuild_file_list_display(self, pairs=None)`](../utils/dataset_validation.py#L1455)
  - [`DatasetLoaderMixin._is_include_empty(self)`](../utils/dataset_validation.py#L1531)
  - [`DatasetLoaderMixin._on_include_empty_toggled(self, *_)`](../utils/dataset_validation.py#L1534)
  - [`DatasetLoaderMixin._on_dataset_type_changed(self, idx)`](../utils/dataset_validation.py#L1540)
  - [`DatasetLoaderMixin._on_input_format_changed(self, idx)`](../utils/dataset_validation.py#L1564)
  - [`DatasetLoaderMixin._fill_class_table_from_dict(self, class_names_dict, preserve_include=True)`](../utils/dataset_validation.py#L1576)
  - [`DatasetLoaderMixin._get_yaml_name_for_id(self, cid)`](../utils/dataset_validation.py#L1620)
  - [`DatasetLoaderMixin.collect_classes(self)`](../utils/dataset_validation.py#L1637)
  - [`DatasetLoaderMixin.update_included_classes(self)`](../utils/dataset_validation.py#L1756)
  - [`DatasetLoaderMixin.on_class_table_item_changed(self, item)`](../utils/dataset_validation.py#L1836)
  - [`DatasetLoaderMixin._update_pair_count_label(self, extra_text, error=False)`](../utils/dataset_validation.py#L1842)

<a id="utils-ensemble-utils"></a>

## utils.ensemble_utils

Исходник: [utils/ensemble_utils.py](../utils/ensemble_utils.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`contains_center(box1, box2)`](../utils/ensemble_utils.py#L4) — Проверяет, лежит ли центр box2 внутри box1.
- [`boxes_contain_each_other(box1, box2)`](../utils/ensemble_utils.py#L14) — Проверяет, содержит ли один бокс другой (центр одного внутри другого).
- [`non_max_suppression(boxes, scores, iou_threshold=0.5, conf_threshold=0.1)`](../utils/ensemble_utils.py#L24) — Классический NMS: сортировка по уверенности, выбор лучшего, подавление по IoU.
- [`weighted_boxes_fusion(boxes, scores, weights, iou_threshold=0.5, conf_threshold=0.1, use_containment=False, verbose=False)`](../utils/ensemble_utils.py#L76) — Weighted Boxes Fusion (WBF) – взвешенное усреднение координат и уверенностей.
- [`ensemble_detections(detections_list, weights, iou_threshold=0.5, conf_threshold=0.1, use_containment=False, method='wbf', verbose=False)`](../utils/ensemble_utils.py#L132) — Объединяет детекции из нескольких источников с помощью выбранного метода.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`extract_threshold_params(preset, method)`](../utils/ensemble_utils.py#L209) — Извлекает параметры для apply_threshold_method из плоского словаря пресета.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`get_detections_from_preset(image, gray, preset, confidence=0.7)`](../utils/ensemble_utils.py#L236) — Применяет пресет к изображению и возвращает список детекций.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`run_ensemble(image, gray, yolo_detections, yolo_weight, presets_with_weights, iou_threshold, conf_threshold, use_containment=False, method='wbf', verbose=True)`](../utils/ensemble_utils.py#L317) — Выполняет полный ансамбль: собирает детекции из YOLO и всех пресетов,
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-image-io"></a>

## utils.image_io

Исходник: [utils/image_io.py](../utils/image_io.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.annotation_converter](#utils-annotation-converter), [utils.coco_export](#utils-coco-export), [utils.coco_io](#utils-coco-io), [utils.unet_export](#utils-unet-export).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** `PIL.Image`, `math`, `pathlib.Path`, `tifffile`.

**Объявления:**

- [`_xywhr_to_corners(cx, cy, w, h, angle_rad)`](../utils/image_io.py#L16)
- [`_corners_to_xywhr(corners)`](../utils/image_io.py#L31)
- [`canonicalize_obb_corners(norm_pts)`](../utils/image_io.py#L64)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.annotation_converter](#utils-annotation-converter), [utils.coco_io](#utils-coco-io).
- [`_read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)`](../utils/image_io.py#L105)
- [`_read_with_tifffile(image_path)`](../utils/image_io.py#L127)
- [`read_image_with_fallback(image_path)`](../utils/image_io.py#L149)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`read_mask_with_fallback(mask_path)`](../utils/image_io.py#L223) — Читает маску, сохраняя её dtype (uint8/uint16).
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.coco_export](#utils-coco-export).
- [`resize_to_max_side(img, max_side=1024)`](../utils/image_io.py#L258)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`normalize_to_uint8(img)`](../utils/image_io.py#L268)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`convert_to_grayscale(img)`](../utils/image_io.py#L281)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`numpy_to_qpixmap(img_bgr, max_side=MAX_QPIXMAP_SIDE)`](../utils/image_io.py#L293)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods).
- [`_classify_line(n_parts, format_hint)`](../utils/image_io.py#L340)
- [`load_annotations(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L366)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.coco_export](#utils-coco-export), [utils.unet_export](#utils-unet-export).
- [`save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False)`](../utils/image_io.py#L425)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods), [utils.DatasetGenerator](#utils-DatasetGenerator), [utils.coco_io](#utils-coco-io).
- [`load_annotations_obb(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L453)
- [`load_annotations_for_image(img_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L458)
- [`load_images_universal(source, require_annotations=False, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L466)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`load_dataset_from_yaml(yaml_path, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L550) — Загружает YOLO-датасет из data.yaml.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`load_dataset_from_yaml_with_masks(yaml_path, resize_enabled=True, max_side=1024, progress_callback=None, parent=None)`](../utils/image_io.py#L645) — Загружает YAML с масками.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`save_coordinates(main_window)`](../utils/image_io.py#L764)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`find_image_label_pairs(images_folder, labels_folder, img_extensions=None)`](../utils/image_io.py#L798)
- [`collect_unique_classes_from_labels(label_paths, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L814)
- [`read_image_with_fallback_find(image_path)`](../utils/image_io.py#L824)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`convert_segment_masks_to_yolo_seg_manual(masks_dir: str, output_dir: str, pixel_to_class: dict, mask_offset: int=0)`](../utils/image_io.py#L869) — Конвертация масок в YOLO-seg .txt (lossy, findContours).
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L930)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods).
- Класс [`LazyImageCache`](../utils/image_io.py#L969)
  - [`LazyImageCache.__init__(self, paths, loader, window=8, prefetch_ahead=4, max_resident=16, max_workers=2)`](../utils/image_io.py#L970)
  - [`LazyImageCache.__len__(self)`](../utils/image_io.py#L986)
  - [`LazyImageCache._trim_locked(self)`](../utils/image_io.py#L989)
  - [`LazyImageCache._load_sync(self, idx)`](../utils/image_io.py#L993)
  - [`LazyImageCache.is_loaded(self, idx)`](../utils/image_io.py#L1010)
  - [`LazyImageCache.get_error(self, idx)`](../utils/image_io.py#L1014)
  - [`LazyImageCache.get(self, idx)`](../utils/image_io.py#L1018)
  - [`LazyImageCache._load_bg(self, idx)`](../utils/image_io.py#L1036)
  - [`LazyImageCache.prefetch(self, indices)`](../utils/image_io.py#L1063)
  - [`LazyImageCache.trim_around(self, center, keep_behind=3, keep_ahead=None)`](../utils/image_io.py#L1078)
  - [`LazyImageCache.shutdown(self)`](../utils/image_io.py#L1091)

<a id="utils-metrics-core"></a>

## utils.metrics_core

Исходник: [utils/metrics_core.py](../utils/metrics_core.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.analysis](#utils-analysis), [utils.prepare](#utils-prepare).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.prepare](#utils-prepare): `add_meanap_metrics`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`yolo_to_xyxy(cx, cy, w, h)`](../utils/metrics_core.py#L7) — Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`xyxy_to_yolo(x1, y1, x2, y2)`](../utils/metrics_core.py#L19) — Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`iou(box1, box2)`](../utils/metrics_core.py#L31) — Вычисляет IoU для двух боксов в YOLO-формате (cls, cx, cy, w, h).
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`iou_xyxy(box1, box2)`](../utils/metrics_core.py#L61) — Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`parse_objects(s)`](../utils/metrics_core.py#L80) — Преобразует строковое представление списка объектов в список кортежей.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_precision_recall_f1(tp, fp, fn)`](../utils/metrics_core.py#L94) — Вычисляет precision, recall и F1-меру по количеству TP, FP, FN.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_metrics(pred_boxes, gt_boxes, iou_threshold=0.5)`](../utils/metrics_core.py#L104) — Для одного изображения вычисляет F1, TP, FP, FN.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_ap(precision, recall)`](../utils/metrics_core.py#L139) — Вычисляет Average Precision (AP) по кривой precision-recall.
- [`compute_map(df, iou_thresholds=None)`](../utils/metrics_core.py#L156) — Вычисляет mAP (mean Average Precision) для датасета.
- [`extract_numeric_params(method: str, params_str: str)`](../utils/metrics_core.py#L172) — Извлекает числовые параметры из строки params (поддерживает кириллицу и латиницу).
  Явно импортируют: [utils.prepare](#utils-prepare).

<a id="utils-prepare"></a>

## utils.prepare

Исходник: [utils/prepare.py](../utils/prepare.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.analysis](#utils-analysis), [utils.metrics_core](#utils-metrics-core).

**Прямые локальные импорты:**
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`, `compute_metrics as compute_metrics_core`, `compute_precision_recall_f1`, `extract_numeric_params`, `iou`, `parse_objects`

**Внешние/stdlib импорты:** `ast.literal_eval`, `json`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `re`, `sklearn.cluster.KMeans`, `sklearn.decomposition.PCA`, `sklearn.preprocessing.StandardScaler`.

**Объявления:**

- [`compute_metrics_detailed(row, iou_thresholds)`](../utils/prepare.py#L28) — row: dict-like с ключами 'objects' и 'true_objects'
- [`add_meanap_metrics(df, iou_thresholds=None)`](../utils/prepare.py#L113) — Добавляет в DataFrame колонки:
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.analysis](#utils-analysis), [utils.metrics_core](#utils-metrics-core).
- [`compute_f1(df: pd.DataFrame, iou_threshold: float=0.5)`](../utils/prepare.py#L166) — Оставлена для обратной совместимости, использует старую метрику.
- [`filter_by_f1_threshold(df: pd.DataFrame, threshold: float)`](../utils/prepare.py#L200)
- [`sample_rows(df: pd.DataFrame, n: int, random_state: int=42)`](../utils/prepare.py#L212)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_full_image_object(df: pd.DataFrame)`](../utils/prepare.py#L223) — Удаляет строки, где среди предсказанных объектов есть объект, покрывающий почти всё изображение.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_invalid_params(df: pd.DataFrame)`](../utils/prepare.py#L252)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_anomaly_count(df: pd.DataFrame, max_objects: int=15)`](../utils/prepare.py#L326)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_zero_true(df: pd.DataFrame)`](../utils/prepare.py#L341)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_oversegmentation(df: pd.DataFrame, ratio: float=2.0)`](../utils/prepare.py#L356)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_duplicates(df: pd.DataFrame)`](../utils/prepare.py#L371)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`filter_no_true_objects(df: pd.DataFrame)`](../utils/prepare.py#L381) — Удаляет строки, в которых нет истинных объектов (num_true_objects == 0).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`aggregate_by_params(df: pd.DataFrame, group_by=None, score_col='mean_f1', top_k=1)`](../utils/prepare.py#L400) — Агрегирует датасет, оставляя топ-K строк с наибольшим score_col в каждой группе.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`save_cluster_report(df, method, cluster_ids, param_names, clusters_dir, method_name)`](../utils/prepare.py#L441) — Сохраняет текстовый отчёт о кластерах для одного метода.
- [`add_cluster_labels(df: pd.DataFrame, output_dir: str=None)`](../utils/prepare.py#L477) — Для каждого метода выполняет кластеризацию параметров и добавляет колонку 'cluster'.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`save_unique_params(df: pd.DataFrame, output_dir: str=None)`](../utils/prepare.py#L572) — Для каждого метода сохраняет уникальные значения params в текстовый файл.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`optimize_invert(df: pd.DataFrame)`](../utils/prepare.py#L601) — Для каждого метода оставляет только строки с лучшим значением invert (True/False)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`cluster_open_close(df: pd.DataFrame, n_clusters: int=3)`](../utils/prepare.py#L636) — Для каждого уникального сочетания (method, cluster) выполняет кластеризацию
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-presets"></a>

## utils.presets

Исходник: [utils/presets.py](../utils/presets.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [path_setup](#path-setup): `get_project_root`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`PresetManager`](../utils/presets.py#L5)
  - [`PresetManager.__init__(self, main_window=None, preset_file='presets.json')`](../utils/presets.py#L6)
  - [`PresetManager.load_presets(self)`](../utils/presets.py#L15)
  - [`PresetManager.save_presets(self)`](../utils/presets.py#L27)
  - [`PresetManager.get_current_settings(self)`](../utils/presets.py#L35)
  - [`PresetManager.apply_preset(self, name)`](../utils/presets.py#L76)
  - [`PresetManager.add_preset(self, name)`](../utils/presets.py#L173)
  - [`PresetManager.add_preset_dict(self, name, preset_dict)`](../utils/presets.py#L182)
  - [`PresetManager.delete_preset(self, name)`](../utils/presets.py#L189)
  - [`PresetManager.get_preset_names(self)`](../utils/presets.py#L199)
- Класс [`GradientPresetManager`](../utils/presets.py#L203) (`PresetManager`)
  - [`GradientPresetManager.get_current_settings(self)`](../utils/presets.py#L204)
  - [`GradientPresetManager.apply_preset(self, name)`](../utils/presets.py#L226)
- Класс [`TraditionalMLPresetManager`](../utils/presets.py#L273) (`PresetManager`)
  - [`TraditionalMLPresetManager.get_current_settings(self)`](../utils/presets.py#L274)
  - [`TraditionalMLPresetManager.apply_preset(self, name)`](../utils/presets.py#L294)
- Класс [`DeepLearningPresetManager`](../utils/presets.py#L352) (`PresetManager`)
  - [`DeepLearningPresetManager.get_current_settings(self)`](../utils/presets.py#L353)
  - [`DeepLearningPresetManager.apply_preset(self, name)`](../utils/presets.py#L376)
- Класс [`InteractivePresetManager`](../utils/presets.py#L467) (`PresetManager`)
  - [`InteractivePresetManager.get_current_settings(self)`](../utils/presets.py#L468)
  - [`InteractivePresetManager.apply_preset(self, name)`](../utils/presets.py#L478)

<a id="utils-settings"></a>

## utils.settings

Исходник: [utils/settings.py](../utils/settings.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.image_navigation](#ui-image-navigation), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.settings_ui](#ui-settings-ui), [utils.smart_view](#utils-smart-view), [utils_new.smart_view_new](#utils-new-smart-view-new), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [path_setup](#path-setup): `get_project_root`

**Внешние/stdlib импорты:** `copy.deepcopy`.

**Объявления:**

- Класс [`SettingsManager`](../utils/settings.py#L21) (`QObject`)
  - [`SettingsManager.__new__(cls, *args, **kwargs)`](../utils/settings.py#L26)
  - [`SettingsManager.__init__(self)`](../utils/settings.py#L31)
  - [`SettingsManager._with_defaults(data, defaults)`](../utils/settings.py#L71)
  - [`SettingsManager._migrate_legacy_size_settings(data)`](../utils/settings.py#L88)
  - [`SettingsManager.load(self)`](../utils/settings.py#L103)
  - [`SettingsManager.save(self, settings=None)`](../utils/settings.py#L119)
  - [`SettingsManager.get_color(self, name)`](../utils/settings.py#L133)
  - [`SettingsManager.get_primary_color(self)`](../utils/settings.py#L137)
  - [`SettingsManager.get_theme(self)`](../utils/settings.py#L140)
  - [`SettingsManager._safe_int(value, fallback, low, high)`](../utils/settings.py#L144)
  - [`SettingsManager.get_font_size_px(self)`](../utils/settings.py#L150)
  - [`SettingsManager.get_line_thickness_px(self)`](../utils/settings.py#L156)
  - [`SettingsManager.get_dataset_class_palette(self)`](../utils/settings.py#L162)
  - [`SettingsManager.get_dataset_class_color(self, class_id)`](../utils/settings.py#L181)
  - [`SettingsManager.get_font_scale_factor(self)`](../utils/settings.py#L194)
  - [`SettingsManager.get_line_thickness_factor(self)`](../utils/settings.py#L198)
- [`adjust_color(color, h_shift=0, s_shift=0, l_shift=0)`](../utils/settings.py#L204)
- [`generate_palette_for_theme(theme, primary_color)`](../utils/settings.py#L214)
- [`apply_theme(app)`](../utils/settings.py#L285)
  Явно импортируют: [import_libs_internal](#import-libs-internal), [ui.settings_ui](#ui-settings-ui).

<a id="utils-smart-view"></a>

## utils.smart_view

Исходник: [utils/smart_view.py](../utils/smart_view.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.dataset_validation](#utils-dataset-validation).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [utils.settings](#utils-settings): `settings`

**Внешние/stdlib импорты:** `math`, `ultralytics.utils.ops.xywhr2xyxyxyxy`, `ultralytics.utils.ops.xyxyxyxy2xywhr`.

**Объявления:**

- Класс [`SmartGraphicsView`](../utils/smart_view.py#L16) (`QGraphicsView`)
  - [`SmartGraphicsView.__init__(self, parent=None)`](../utils/smart_view.py#L80)
  - [`SmartGraphicsView._setup_pan_mode(self)`](../utils/smart_view.py#L167)
  - [`SmartGraphicsView._setup_drawing_mode(self)`](../utils/smart_view.py#L171)
  - [`SmartGraphicsView._setup_edit_mode(self)`](../utils/smart_view.py#L188)
  - [`SmartGraphicsView.set_suggested_save_name(self, name)`](../utils/smart_view.py#L201)
  - [`SmartGraphicsView.set_drawing_tool(self, tool)`](../utils/smart_view.py#L207)
  - [`SmartGraphicsView.set_edit_mode(self, enabled)`](../utils/smart_view.py#L226)
  - [`SmartGraphicsView._clear_edit_state(self)`](../utils/smart_view.py#L242)
  - [`SmartGraphicsView.set_annotations(self, annotations, img_width, img_height)`](../utils/smart_view.py#L258)
  - [`SmartGraphicsView.set_selected_index(self, idx)`](../utils/smart_view.py#L263)
  - [`SmartGraphicsView.set_callbacks(self, on_rect_drawn=None, on_obb_drawn=None, on_segment_drawn=None, on_scribble_added=None, on_display_update=None, on_reset_tool=None, on_annotation_modified=None, on_selection_changed=None, on_annotation_right_clicked=None, on_log=None)`](../utils/smart_view.py#L285)
  - [`SmartGraphicsView.set_image_data(self, width, height, fg_scribbles=None, bg_scribbles=None)`](../utils/smart_view.py#L302)
  - [`SmartGraphicsView.set_pixmap(self, pixmap, preserve_view=True)`](../utils/smart_view.py#L306)
  - [`SmartGraphicsView.reset_view(self)`](../utils/smart_view.py#L321)
  - [`SmartGraphicsView.save_current_image(self)`](../utils/smart_view.py#L327)
  - [`SmartGraphicsView._scene_per_px(self)`](../utils/smart_view.py#L342)
  - [`SmartGraphicsView._get_points_pixel(self, ann)`](../utils/smart_view.py#L354)
  - [`SmartGraphicsView.point_in_polygon(self, pt, poly)`](../utils/smart_view.py#L378)
  - [`SmartGraphicsView._point_to_segment_dist(px, py, x1, y1, x2, y2)`](../utils/smart_view.py#L398)
  - [`SmartGraphicsView._hit_test_point(self, point, pos_pixel, radius=None)`](../utils/smart_view.py#L414)
  - [`SmartGraphicsView._hit_test_detect(self, ann, pos_pixel)`](../utils/smart_view.py#L422)
  - [`SmartGraphicsView._hit_test_segment(self, ann, pos_pixel)`](../utils/smart_view.py#L434)
  - [`SmartGraphicsView._hit_test_any(self, ann, pos_pixel)`](../utils/smart_view.py#L446)
  - [`SmartGraphicsView._find_hit(self, pos_pixel, prefer_idx=-1)`](../utils/smart_view.py#L458)
  - [`SmartGraphicsView._obb_to_params(self, pts_pixel)`](../utils/smart_view.py#L479)
  - [`SmartGraphicsView._obb_to_points(self, cx, cy, w, h, angle)`](../utils/smart_view.py#L522)
  - [`SmartGraphicsView._obb_points_in_bounds(self, points, eps=0.5)`](../utils/smart_view.py#L562)
  - [`SmartGraphicsView._obb_world_half_extent(self, w, h, angle)`](../utils/smart_view.py#L569)
  - [`SmartGraphicsView._get_obb_handles(self, pts_pixel)`](../utils/smart_view.py#L576)
  - [`SmartGraphicsView._hit_test_obb_handles(self, pts_pixel, pos_pixel)`](../utils/smart_view.py#L636)
  - [`SmartGraphicsView._max_height_inside(self, x1, y1, x2, y2, dir_x, dir_y)`](../utils/smart_view.py#L672)
  - [`SmartGraphicsView._compute_obb_preview_params(self)`](../utils/smart_view.py#L703)
  - [`SmartGraphicsView.wheelEvent(self, event)`](../utils/smart_view.py#L749)
  - [`SmartGraphicsView.mousePressEvent(self, event)`](../utils/smart_view.py#L755)
  - [`SmartGraphicsView.mouseMoveEvent(self, event)`](../utils/smart_view.py#L916)
  - [`SmartGraphicsView.mouseReleaseEvent(self, event)`](../utils/smart_view.py#L1270)
  - [`SmartGraphicsView.mouseDoubleClickEvent(self, event)`](../utils/smart_view.py#L1301)
  - [`SmartGraphicsView._try_segment_double_click(self, pos_pixel)`](../utils/smart_view.py#L1332)
  - [`SmartGraphicsView._apply_segment_change(self, idx, norm_flat, log_msg)`](../utils/smart_view.py#L1438)
  - [`SmartGraphicsView.leaveEvent(self, event)`](../utils/smart_view.py#L1466)
  - [`SmartGraphicsView.contextMenuEvent(self, event)`](../utils/smart_view.py#L1471)
  - [`SmartGraphicsView._start_drawing(self, event)`](../utils/smart_view.py#L1486)
  - [`SmartGraphicsView._update_segment_cursor(self, event)`](../utils/smart_view.py#L1536)
  - [`SmartGraphicsView._update_segment_line(self, event)`](../utils/smart_view.py#L1544)
  - [`SmartGraphicsView._update_obb_draw(self, event)`](../utils/smart_view.py#L1564)
  - [`SmartGraphicsView._continue_drawing(self, event)`](../utils/smart_view.py#L1572)
  - [`SmartGraphicsView._finish_drawing(self, event)`](../utils/smart_view.py#L1587)
  - [`SmartGraphicsView._finish_segment(self)`](../utils/smart_view.py#L1602)
  - [`SmartGraphicsView._ensure_min_polygon_points(self, pts)`](../utils/smart_view.py#L1626)
  - [`SmartGraphicsView._finish_segment_line(self)`](../utils/smart_view.py#L1657)
  - [`SmartGraphicsView._finish_obb(self)`](../utils/smart_view.py#L1700)
  - [`SmartGraphicsView._reset_obb_draw(self)`](../utils/smart_view.py#L1720)
  - [`SmartGraphicsView._add_scribble(self, x, y, temp=False)`](../utils/smart_view.py#L1729)
  - [`SmartGraphicsView.normalize_angle_rad(self, angle)`](../utils/smart_view.py#L1735)
  - [`SmartGraphicsView.drawForeground(self, painter: QPainter, rect)`](../utils/smart_view.py#L1746)
  - [`SmartGraphicsView._draw_foreground_impl(self, painter: QPainter, rect)`](../utils/smart_view.py#L1752)

<a id="utils-unet-export"></a>

## utils.unet_export

Исходник: [utils/unet_export.py](../utils/unet_export.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.image_io](#utils-image-io): `FORMAT_HINT_AUTO`, `load_annotations`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`_choose_dtype(max_pixel_value)`](../utils/unet_export.py#L38)
- [`export_masks_from_annotations(output_path, split, class_names, mask_class_offset=1, progress_cb=None, log_cb=None)`](../utils/unet_export.py#L42) — Читает labels/{split}/*.txt, пишет masks/{split}/*.png.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).
- [`write_unet_yaml(output_path, class_names, mask_class_offset=1, log_cb=None)`](../utils/unet_export.py#L156) — Пишет masks_data.yaml в корень output_path.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.DatasetGenerator](#utils-DatasetGenerator).

<a id="utils-unet-io"></a>

## utils.unet_io

Исходник: [utils/unet_io.py](../utils/unet_io.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [utils.yaml_io](#utils-yaml-io).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`load_unet_yaml(yaml_path, log_cb=None)`](../utils/unet_io.py#L14) — Загружает masks_data.yaml (или data.yaml с masks-полями).
  Явно импортируют: [import_libs_internal](#import-libs-internal), [utils.yaml_io](#utils-yaml-io).
- [`is_unet_yaml(yaml_path)`](../utils/unet_io.py#L127) — Эвристика: является ли YAML U-Net-структурой.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-yaml-io"></a>

## utils.yaml_io

Исходник: [utils/yaml_io.py](../utils/yaml_io.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.unet_io](#utils-unet-io): `load_unet_yaml`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`_resolve_relative_to_yaml(raw_path, yaml_path)`](../utils/yaml_io.py#L20) — Абсолютный путь: если raw_path абсолютный — берём как есть,
- [`_resolve_split_dirs(split_val, base_path)`](../utils/yaml_io.py#L32) — split_val может быть:
- [`_find_label_for_image(img_path, base_path, split, rel_dir)`](../utils/yaml_io.py#L81) — Ищет .txt для изображения в стандартных местах YOLO-структуры.
- [`find_mask_for_image(img_path, base_path=None)`](../utils/yaml_io.py#L120) — Ищет маску для изображения в стандартных местах YAML-структуры.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`load_yolo_yaml(yaml_path, log_cb=None)`](../utils/yaml_io.py#L162) — Парсит YOLO data.yaml.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`load_unet_yaml_pairs(yaml_path, log_cb=None)`](../utils/yaml_io.py#L244) — Парсит U-Net masks_data.yaml.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-aeb-auto-aeb-def"></a>

## utils_aeb.auto-aeb_def

Исходник: [utils_aeb/auto-aeb_def.py](../utils_aeb/auto-aeb_def.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_def.py#L71) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_def.py#L72)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_def.py#L80)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_def.py#L87)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_def.py#L92)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_def.py#L109) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_def.py#L112)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_def.py#L136)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_def.py#L148)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_def.py#L151)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_def.py#L164)
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_def.py#L176) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def.py#L177)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L190)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_def.py#L194) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def.py#L195)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_def.py#L208)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_def.py#L212) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L213)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L223)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_def.py#L231) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L232)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_def.py#L244)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L251)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L261)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_def.py#L267)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_def.py#L273)
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def.py#L280)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def.py#L332)
- [`main()`](../utils_aeb/auto-aeb_def.py#L372)

<a id="utils-aeb-auto-aeb-def-new"></a>

## utils_aeb.auto-aeb_def_new

Исходник: [utils_aeb/auto-aeb_def_new.py](../utils_aeb/auto-aeb_def_new.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_def_new.py#L85) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_def_new.py#L86)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_def_new.py#L94)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_def_new.py#L101)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_def_new.py#L106)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_def_new.py#L123) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_def_new.py#L126)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_def_new.py#L150)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_def_new.py#L162)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_def_new.py#L165)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_def_new.py#L178)
- [`_gn(channels, max_groups=8)`](../utils_aeb/auto-aeb_def_new.py#L190) — GroupNorm с автоматическим подбором числа групп (устойчив к малому batch).
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_def_new.py#L198) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def_new.py#L201)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L214)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_def_new.py#L218) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def_new.py#L219)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_def_new.py#L232)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_def_new.py#L236) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L237)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L247)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_def_new.py#L255) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L256)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_def_new.py#L268)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L275)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L286)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_def_new.py#L292)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_def_new.py#L298) — FIX: линейный KL-annealing 0 → β_target за warmup_epochs эпох.
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def_new.py#L306)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def_new.py#L359)
- [`main()`](../utils_aeb/auto-aeb_def_new.py#L400)

<a id="utils-aeb-auto-aeb-old"></a>

## utils_aeb.auto-aeb_old

Исходник: [utils_aeb/auto-aeb_old.py](../utils_aeb/auto-aeb_old.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_old.py#L74) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_old.py#L75)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_old.py#L83)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_old.py#L90)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_old.py#L95)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_old.py#L112) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_old.py#L115)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_old.py#L139)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_old.py#L151)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_old.py#L154)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_old.py#L167)
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_old.py#L179) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_old.py#L180)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L193)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_old.py#L197) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_old.py#L198)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_old.py#L211)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_old.py#L215) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L216)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L226)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_old.py#L234) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L235)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_old.py#L247)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L254)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L264)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_old.py#L270)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_old.py#L276)
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_old.py#L283)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_old.py#L335)
- [`main()`](../utils_aeb/auto-aeb_old.py#L375)

<a id="utils-aeb-auto-aeb-unet"></a>

## utils_aeb.auto-aeb_unet

Исходник: [utils_aeb/auto-aeb_unet.py](../utils_aeb/auto-aeb_unet.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_unet.py#L77) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_unet.py#L78)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_unet.py#L86)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_unet.py#L93)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_unet.py#L98)
- Класс [`GradientLoss`](../utils_aeb/auto-aeb_unet.py#L115) (`nn.Module`)
  - [`GradientLoss.__init__(self)`](../utils_aeb/auto-aeb_unet.py#L116)
  - [`GradientLoss.forward(self, pred, target)`](../utils_aeb/auto-aeb_unet.py#L125)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_unet.py#L137) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_unet.py#L140)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_unet.py#L168)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_unet.py#L180)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_unet.py#L183)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_unet.py#L196)
- Класс [`UNetEncoder`](../utils_aeb/auto-aeb_unet.py#L210) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/auto-aeb_unet.py#L211)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L240)
- Класс [`UNetDecoder`](../utils_aeb/auto-aeb_unet.py#L248) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/auto-aeb_unet.py#L249)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/auto-aeb_unet.py#L283)
- Класс [`UNetAE`](../utils_aeb/auto-aeb_unet.py#L290) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet.py#L291)
  - [`UNetAE.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L303)
- Класс [`UNetVAE`](../utils_aeb/auto-aeb_unet.py#L311) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet.py#L312)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_unet.py#L325)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L332)
- [`build_model(model_type, in_channels, latent_channels, img_h, img_w)`](../utils_aeb/auto-aeb_unet.py#L342)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_unet.py#L348)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_unet.py#L355)
- [`_compute_recon_loss(recon, x, criterion_l1, criterion_grad, criterion_ssim)`](../utils_aeb/auto-aeb_unet.py#L362)
- [`train_one_epoch(model, loader, optimizer, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet.py#L370)
- [`evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet.py#L417)
- [`main()`](../utils_aeb/auto-aeb_unet.py#L453)

<a id="utils-aeb-auto-aeb-unet-with-grad"></a>

## utils_aeb.auto-aeb_unet_with_grad

Исходник: [utils_aeb/auto-aeb_unet_with_grad.py](../utils_aeb/auto-aeb_unet_with_grad.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `hashlib`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `time`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`.

**Объявления:**

- [`_to_gray(x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L119)
- [`median_blur(x, k=3)`](../utils_aeb/auto-aeb_unet_with_grad.py#L125) — x: [B,C,H,W]. Медиана через unfold.
- [`_edge_kernels(kind)`](../utils_aeb/auto-aeb_unet_with_grad.py#L144)
- [`edge_map(x, lo=EDGE_LO, hi=EDGE_HI, kind=EDGE_KERNEL, med=MEDIAN_KERNEL)`](../utils_aeb/auto-aeb_unet_with_grad.py#L152) — x: [B,C,H,W] в [0,1] → [B,1,H,W] мягкая карта рёбер.
- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_unet_with_grad.py#L169) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_unet_with_grad.py#L170)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_unet_with_grad.py#L178)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_unet_with_grad.py#L185)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L190)
- Класс [`GradientLoss`](../utils_aeb/auto-aeb_unet_with_grad.py#L206) (`nn.Module`)
  - [`GradientLoss.__init__(self)`](../utils_aeb/auto-aeb_unet_with_grad.py#L207)
  - [`GradientLoss.forward(self, pred, target)`](../utils_aeb/auto-aeb_unet_with_grad.py#L216)
- [`d_hinge_loss(d_real, d_fake)`](../utils_aeb/auto-aeb_unet_with_grad.py#L227)
- [`g_hinge_loss(d_fake)`](../utils_aeb/auto-aeb_unet_with_grad.py#L231)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_unet_with_grad.py#L236) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0, use_edge=False, edge_cache_dir=None)`](../utils_aeb/auto-aeb_unet_with_grad.py#L239)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_unet_with_grad.py#L263)
  - [`ImageFolderDataset._load_resized_tensor(self, path)`](../utils_aeb/auto-aeb_unet_with_grad.py#L275)
  - [`ImageFolderDataset._edge_cache_path(self, img_path)`](../utils_aeb/auto-aeb_unet_with_grad.py#L282)
  - [`ImageFolderDataset._load_or_build_edge(self, img_path, img_t)`](../utils_aeb/auto-aeb_unet_with_grad.py#L302)
  - [`ImageFolderDataset._augment_geom(self, img_t, edge_t)`](../utils_aeb/auto-aeb_unet_with_grad.py#L326)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_unet_with_grad.py#L358)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_unet_with_grad.py#L361)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_unet_with_grad.py#L387)
- [`prewarm_edge_cache(dataset)`](../utils_aeb/auto-aeb_unet_with_grad.py#L399) — Считает и сохраняет edge-карты для всех сэмплов датасета (только новые).
- [`_norm(c, kind=NORM_TYPE)`](../utils_aeb/auto-aeb_unet_with_grad.py#L426) — GroupNorm с безопасным числом групп (c % g == 0).
- [`_act()`](../utils_aeb/auto-aeb_unet_with_grad.py#L436)
- Класс [`ConvBlock`](../utils_aeb/auto-aeb_unet_with_grad.py#L440) (`nn.Module`)
  - [`ConvBlock.__init__(self, in_c, out_c, stride=1, n_convs=2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L441)
  - [`ConvBlock.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L453)
- Класс [`DilatedResidual`](../utils_aeb/auto-aeb_unet_with_grad.py#L457) (`nn.Module`)
  - [`DilatedResidual.__init__(self, c, dilations=(2, 4))`](../utils_aeb/auto-aeb_unet_with_grad.py#L458)
  - [`DilatedResidual.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L470)
- Класс [`UNetEncoder`](../utils_aeb/auto-aeb_unet_with_grad.py#L476) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L477)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L486)
- Класс [`DecoderBlock`](../utils_aeb/auto-aeb_unet_with_grad.py#L495) (`nn.Module`)
  - [`DecoderBlock.__init__(self, in_c, skip_c, out_c)`](../utils_aeb/auto-aeb_unet_with_grad.py#L496)
  - [`DecoderBlock.forward(self, x, skip)`](../utils_aeb/auto-aeb_unet_with_grad.py#L506)
- Класс [`UNetDecoder`](../utils_aeb/auto-aeb_unet_with_grad.py#L516) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L517)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L528)
- Класс [`UNetAE`](../utils_aeb/auto-aeb_unet_with_grad.py#L535) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L537)
  - [`UNetAE.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L548)
- Класс [`UNetVAE`](../utils_aeb/auto-aeb_unet_with_grad.py#L556) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L558)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_unet_with_grad.py#L570)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L576)
- Класс [`PatchDiscriminator`](../utils_aeb/auto-aeb_unet_with_grad.py#L586) (`nn.Module`)
  - [`PatchDiscriminator.__init__(self, in_ch, base=D_BASE)`](../utils_aeb/auto-aeb_unet_with_grad.py#L587)
  - [`PatchDiscriminator.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L603)
- [`build_model(model_type, in_channels, out_channels, latent_channels, img_h, img_w)`](../utils_aeb/auto-aeb_unet_with_grad.py#L607)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_unet_with_grad.py#L614)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_unet_with_grad.py#L620)
- [`_edge_weighted_l1(pred, target, w=EDGE_L1_EXTRA_W)`](../utils_aeb/auto-aeb_unet_with_grad.py#L627)
- [`_compute_recon_loss(recon, target, criterion_l1, criterion_grad, criterion_ssim)`](../utils_aeb/auto-aeb_unet_with_grad.py#L635)
- [`train_one_epoch(model, loader, optimizer, D, opt_d, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta, use_gan, gan_weight)`](../utils_aeb/auto-aeb_unet_with_grad.py#L647)
- [`evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet_with_grad.py#L724)
- [`main()`](../utils_aeb/auto-aeb_unet_with_grad.py#L761)

<a id="utils-aeb-yolo-aeb-utils"></a>

## utils_aeb.yolo_aeb_utils

Исходник: [utils_aeb/yolo_aeb_utils.py](../utils_aeb/yolo_aeb_utils.py#L1).

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `*`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `apply_object_morphology`, `build_detection_mask`, `cls_suffix_from_detection`, `crop_to_detection`, `fill_outside_object`, `limit_image_size`, `longest_edge_angle_deg`, `morph_image_in_mask`, `rotate_with_optional_crop`

**Внешние/stdlib импорты:** `torch.nn.functional`.

**Объявления:**

- Класс [`ConvEncoder`](../utils_aeb/yolo_aeb_utils.py#L35) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L36)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L49)
- Класс [`ConvDecoder`](../utils_aeb/yolo_aeb_utils.py#L53) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L54)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/yolo_aeb_utils.py#L67)
- Класс [`ConvAutoencoder`](../utils_aeb/yolo_aeb_utils.py#L71) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L72)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L82)
- Класс [`ConvVAE`](../utils_aeb/yolo_aeb_utils.py#L90) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L91)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L103)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L109)
- [`_gn(channels, max_groups=8)`](../utils_aeb/yolo_aeb_utils.py#L122) — GroupNorm с автоматическим подбором числа групп — как в обучении.
- Класс [`ConvEncoderGN`](../utils_aeb/yolo_aeb_utils.py#L130) (`nn.Module`)
  - [`ConvEncoderGN.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L131)
  - [`ConvEncoderGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L144)
- Класс [`ConvDecoderGN`](../utils_aeb/yolo_aeb_utils.py#L148) (`nn.Module`)
  - [`ConvDecoderGN.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L149)
  - [`ConvDecoderGN.forward(self, z)`](../utils_aeb/yolo_aeb_utils.py#L162)
- Класс [`ConvAutoencoderGN`](../utils_aeb/yolo_aeb_utils.py#L166) (`nn.Module`)
  - [`ConvAutoencoderGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L167)
  - [`ConvAutoencoderGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L177)
- Класс [`ConvVAEGN`](../utils_aeb/yolo_aeb_utils.py#L185) (`nn.Module`)
  - [`ConvVAEGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L186)
  - [`ConvVAEGN.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L198)
  - [`ConvVAEGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L204)
- Класс [`UNetEncoder`](../utils_aeb/yolo_aeb_utils.py#L217) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/yolo_aeb_utils.py#L218)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L247)
- Класс [`UNetDecoder`](../utils_aeb/yolo_aeb_utils.py#L255) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/yolo_aeb_utils.py#L256)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/yolo_aeb_utils.py#L290)
- Класс [`UNetAE`](../utils_aeb/yolo_aeb_utils.py#L297) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/yolo_aeb_utils.py#L298)
  - [`UNetAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L310)
- Класс [`UNetVAE`](../utils_aeb/yolo_aeb_utils.py#L318) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/yolo_aeb_utils.py#L319)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L332)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L338)
- [`_norm_v2(c, kind='gn')`](../utils_aeb/yolo_aeb_utils.py#L353) — GroupNorm с тем же подбором групп, что и в тренировочном скрипте.
- [`_act_v2()`](../utils_aeb/yolo_aeb_utils.py#L363)
- Класс [`ConvBlockV2`](../utils_aeb/yolo_aeb_utils.py#L367) (`nn.Module`)
  - [`ConvBlockV2.__init__(self, in_c, out_c, stride=1, n_convs=2, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L368)
  - [`ConvBlockV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L380)
- Класс [`DilatedResidualV2`](../utils_aeb/yolo_aeb_utils.py#L384) (`nn.Module`)
  - [`DilatedResidualV2.__init__(self, c, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L385)
  - [`DilatedResidualV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L397)
- Класс [`UNetEncoderV2`](../utils_aeb/yolo_aeb_utils.py#L403) (`nn.Module`)
  - [`UNetEncoderV2.__init__(self, in_channels, base=32, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L404)
  - [`UNetEncoderV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L413)
- Класс [`DecoderBlockV2`](../utils_aeb/yolo_aeb_utils.py#L422) (`nn.Module`)
  - [`DecoderBlockV2.__init__(self, in_c, skip_c, out_c, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L423)
  - [`DecoderBlockV2.forward(self, x, skip)`](../utils_aeb/yolo_aeb_utils.py#L433)
- Класс [`UNetDecoderV2`](../utils_aeb/yolo_aeb_utils.py#L444) (`nn.Module`)
  - [`UNetDecoderV2.__init__(self, out_channels, base=32, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L445)
  - [`UNetDecoderV2.forward(self, h, f0, f1, f2)`](../utils_aeb/yolo_aeb_utils.py#L456)
- Класс [`UNetAEV2`](../utils_aeb/yolo_aeb_utils.py#L463) (`nn.Module`)
  - [`UNetAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L464)
  - [`UNetAEV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L479)
- Класс [`UNetVAEV2`](../utils_aeb/yolo_aeb_utils.py#L487) (`nn.Module`)
  - [`UNetVAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn', logvar_min=-10.0, logvar_max=10.0)`](../utils_aeb/yolo_aeb_utils.py#L488)
  - [`UNetVAEV2.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L507)
  - [`UNetVAEV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L513)
- [`_median_blur_torch(x, k=3)`](../utils_aeb/yolo_aeb_utils.py#L534) — x: [B,C,H,W]. Медианный фильтр через unfold (совместим с обучением).
- [`compute_edge_map_torch(img_tensor, edge_kernel='scharr', median_kernel=3, edge_lo=0.1, edge_hi=0.35)`](../utils_aeb/yolo_aeb_utils.py#L545) — img_tensor: [C,H,W] или [B,C,H,W], float32 в [0,1].
- [`detect_arch_from_state_dict(state_dict)`](../utils_aeb/yolo_aeb_utils.py#L586) — Определяет архитектуру по ключам state_dict.
- [`detect_arch(cfg, state_dict=None)`](../utils_aeb/yolo_aeb_utils.py#L603) — Определяет архитектуру AE/VAE.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`build_ae_model_from_cfg(cfg, arch=None)`](../utils_aeb/yolo_aeb_utils.py#L649) — Собирает модель AE/VAE по config чекпойнта.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`preprocess_image_for_ae(img_bgr, cfg)`](../utils_aeb/yolo_aeb_utils.py#L709) — Приводит произвольный BGR-снимок к тензору [C, H, W] float32 [0..1],
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

<a id="utils-auto-research-gradient-auto-research"></a>

## utils_auto_research.gradient_auto_research

Исходник: [utils_auto_research/gradient_auto_research.py](../utils_auto_research/gradient_auto_research.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`GradientAutoResearch`](../utils_auto_research/gradient_auto_research.py#L3)
  - [`GradientAutoResearch.__init__(self, parent_window, search_all=False)`](../utils_auto_research/gradient_auto_research.py#L4)
  - [`GradientAutoResearch.run(self)`](../utils_auto_research/gradient_auto_research.py#L8)
  - [`GradientAutoResearch._merge_with_defaults(self, settings)`](../utils_auto_research/gradient_auto_research.py#L118)
  - [`GradientAutoResearch._save_current_ui_state(self)`](../utils_auto_research/gradient_auto_research.py#L134)
  - [`GradientAutoResearch._restore_ui_state(self, state)`](../utils_auto_research/gradient_auto_research.py#L149)
  - [`GradientAutoResearch._default_settings(self)`](../utils_auto_research/gradient_auto_research.py#L163)
  - [`GradientAutoResearch._generate_combinations(self, settings)`](../utils_auto_research/gradient_auto_research.py#L196)
  - [`GradientAutoResearch._generate_method_params(self, method, method_settings)`](../utils_auto_research/gradient_auto_research.py#L291)
  - [`GradientAutoResearch._range_values(self, start, end, step)`](../utils_auto_research/gradient_auto_research.py#L313)
  - [`GradientAutoResearch._params_desc(self, params)`](../utils_auto_research/gradient_auto_research.py#L328)
  - [`GradientAutoResearch._apply_method_params(self, method, params)`](../utils_auto_research/gradient_auto_research.py#L336)

<a id="utils-auto-research-gradient-auto-settings-dialog"></a>

## utils_auto_research.gradient_auto_settings_dialog

Исходник: [utils_auto_research/gradient_auto_settings_dialog.py](../utils_auto_research/gradient_auto_settings_dialog.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.gradient_methods](#tabs-gradient-methods).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`GradientAutoSettingsDialog`](../utils_auto_research/gradient_auto_settings_dialog.py#L4) (`QDialog`)
  - [`GradientAutoSettingsDialog.__init__(self, parent=None, current_settings=None)`](../utils_auto_research/gradient_auto_settings_dialog.py#L5)
  - [`GradientAutoSettingsDialog.log(self, message)`](../utils_auto_research/gradient_auto_settings_dialog.py#L57)
  - [`GradientAutoSettingsDialog._safe_name(self, name)`](../utils_auto_research/gradient_auto_settings_dialog.py#L66)
  - [`GradientAutoSettingsDialog._default_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L69)
  - [`GradientAutoSettingsDialog._setup_tabs(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L114)
  - [`GradientAutoSettingsDialog._add_method_tab(self, method_name)`](../utils_auto_research/gradient_auto_settings_dialog.py#L133)
  - [`GradientAutoSettingsDialog._create_param_widget(self, value, is_step=False)`](../utils_auto_research/gradient_auto_settings_dialog.py#L176)
  - [`GradientAutoSettingsDialog._add_binarization_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L196)
  - [`GradientAutoSettingsDialog._add_object_filtering_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L220)
  - [`GradientAutoSettingsDialog._add_morphology_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L269)
  - [`GradientAutoSettingsDialog._add_invert_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L349)
  - [`GradientAutoSettingsDialog._load_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L367)
  - [`GradientAutoSettingsDialog.get_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L436)

<a id="utils-auto-research-threshold-auto-research"></a>

## utils_auto_research.threshold_auto_research

Исходник: [utils_auto_research/threshold_auto_research.py](../utils_auto_research/threshold_auto_research.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`AutoResearch`](../utils_auto_research/threshold_auto_research.py#L3)
  - [`AutoResearch.__init__(self, main_window, search_all=False)`](../utils_auto_research/threshold_auto_research.py#L4)
  - [`AutoResearch.run(self)`](../utils_auto_research/threshold_auto_research.py#L8)
  - [`AutoResearch._default_settings(self)`](../utils_auto_research/threshold_auto_research.py#L152)
  - [`AutoResearch._generate_combinations(self, settings)`](../utils_auto_research/threshold_auto_research.py#L179)
  - [`AutoResearch._generate_binarization_combos(self, settings)`](../utils_auto_research/threshold_auto_research.py#L231)
  - [`AutoResearch._apply_method_params(self, method, params)`](../utils_auto_research/threshold_auto_research.py#L351)

<a id="utils-auto-research-threshold-auto-settings-dialog"></a>

## utils_auto_research.threshold_auto_settings_dialog

Исходник: [utils_auto_research/threshold_auto_settings_dialog.py](../utils_auto_research/threshold_auto_settings_dialog.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`AutoSettingsDialog`](../utils_auto_research/threshold_auto_settings_dialog.py#L3) (`QDialog`)
  - [`AutoSettingsDialog.__init__(self, parent=None, current_settings=None)`](../utils_auto_research/threshold_auto_settings_dialog.py#L4)
  - [`AutoSettingsDialog._safe_name(self, name)`](../utils_auto_research/threshold_auto_settings_dialog.py#L35)
  - [`AutoSettingsDialog._default_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L39)
  - [`AutoSettingsDialog._setup_tabs(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L65)
  - [`AutoSettingsDialog._add_simple_threshold_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L78)
  - [`AutoSettingsDialog._add_otsu_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L102)
  - [`AutoSettingsDialog._add_triangle_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L112)
  - [`AutoSettingsDialog._add_adaptive_tab(self, method_name)`](../utils_auto_research/threshold_auto_settings_dialog.py#L122)
  - [`AutoSettingsDialog._add_niblack_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L157)
  - [`AutoSettingsDialog._add_sauvola_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L191)
  - [`AutoSettingsDialog._add_isodata_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L234)
  - [`AutoSettingsDialog._add_background_symmetry_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L257)
  - [`AutoSettingsDialog._add_row_adaptive_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L280)
  - [`AutoSettingsDialog._add_morphology_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L312)
  - [`AutoSettingsDialog._load_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L384)
  - [`AutoSettingsDialog.get_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L480)

<a id="utils-new-DatasetGenerator-new"></a>

## utils_new.DatasetGenerator_new

Исходник: [utils_new/DatasetGenerator_new.py](../utils_new/DatasetGenerator_new.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `AnnotationSample`, `AugmentationPlan`, `Flip`, `ImageSize`, `MaskEncoding`, `Resize`, `TransformPlan`, `apply_augmentation_plan`, `build_augmentation_plan`, `convert_annotation`, `mask_to_annotations`, `polygon_area`, `remap_mask_indices`, `transform_sample`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `sanitize_annotation_topology`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `make_index_mask`, `read_image`, `read_index_mask`, `read_yolo_labels`, `write_classes_and_yaml`, `write_coco_split`, `write_image`, `write_index_mask`, `write_yolo_labels`
- [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new): `mask_statistics`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.pyqtSignal`, `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `hashlib`, `json`, `numpy`, `os`, `pathlib.Path`, `random`, `shutil`, `tempfile`, `traceback`, `typing.Callable`.

**Объявления:**

- Класс [`_Generated`](../utils_new/DatasetGenerator_new.py#L43)
- Класс [`DatasetGeneratorCoreNew`](../utils_new/DatasetGenerator_new.py#L54)
  - [`DatasetGeneratorCoreNew.__init__(self, images_folder, labels_folder, output_folder, pairs, train_pct, val_pct, test_pct, target_size, augmentations, has_augmentations, aug_multiplier, class_names, class_mapping, bg_color, input_type, masks_folder, num_classes, mask_class_remap, included_orig_ids=None, included_class_ids=None, total_orig_classes=1, resize_mode='fixed', include_empty_annotations=False, delete_source_after_success=False, input_format=FORMAT_HINT_AUTO, out_yolo_txt=True, out_yolo_subtype=SUBTYPE_KEEP, out_yolo_obb_format='corners', out_masks=False, out_masks_offset=1, out_unet_yaml=False, out_coco=False, mask_pixel_offset_in=1, min_ann_area_px=0, *, random_seed=None, split_by_source=False, obb_policy='promote_to_segment', hole_policy='error', mixed_output_policy='normalize_segment', strict_complete=False, existing_output_policy='merge', log_cb: Callable[[str], None] &#124; None=None, progress_cb: Callable[[int, int], None] &#124; None=None)`](../utils_new/DatasetGenerator_new.py#L65)
  - [`DatasetGeneratorCoreNew.log(self, message)`](../utils_new/DatasetGenerator_new.py#L141)
  - [`DatasetGeneratorCoreNew.cancel(self)`](../utils_new/DatasetGenerator_new.py#L144)
  - [`DatasetGeneratorCoreNew._check_cancelled(self)`](../utils_new/DatasetGenerator_new.py#L147)
  - [`DatasetGeneratorCoreNew._validate_config(self)`](../utils_new/DatasetGenerator_new.py#L151)
  - [`DatasetGeneratorCoreNew._dense_map(mapping)`](../utils_new/DatasetGenerator_new.py#L190)
  - [`DatasetGeneratorCoreNew._map_annotations(self, annotations)`](../utils_new/DatasetGenerator_new.py#L197)
  - [`DatasetGeneratorCoreNew._map_mask(self, mask)`](../utils_new/DatasetGenerator_new.py#L207)
  - [`DatasetGeneratorCoreNew._verify_mask_export(self, mask, mask_path, *, has_objects: bool)`](../utils_new/DatasetGenerator_new.py#L217)
  - [`DatasetGeneratorCoreNew._filter_annotations(self, anns)`](../utils_new/DatasetGenerator_new.py#L249)
  - [`DatasetGeneratorCoreNew._filter_mask(self, mask)`](../utils_new/DatasetGenerator_new.py#L260)
  - [`DatasetGeneratorCoreNew._resize(self, sample)`](../utils_new/DatasetGenerator_new.py#L275)
  - [`DatasetGeneratorCoreNew._augmentation_impossible(self, size)`](../utils_new/DatasetGenerator_new.py#L284)
  - [`DatasetGeneratorCoreNew._make_variant(self, sample, *, augmented=False, require_operation=False)`](../utils_new/DatasetGenerator_new.py#L301)
  - [`DatasetGeneratorCoreNew._same_sample(a, b)`](../utils_new/DatasetGenerator_new.py#L330)
  - [`DatasetGeneratorCoreNew._variant_digest(sample)`](../utils_new/DatasetGenerator_new.py#L345)
  - [`DatasetGeneratorCoreNew._get_name(self, img_path)`](../utils_new/DatasetGenerator_new.py#L366)
  - [`DatasetGeneratorCoreNew._reserve_stem(self, proposed)`](../utils_new/DatasetGenerator_new.py#L376)
  - [`DatasetGeneratorCoreNew._sanitize_input_annotations(self, annotations, image_path)`](../utils_new/DatasetGenerator_new.py#L385)
  - [`DatasetGeneratorCoreNew._prepare_sample(self, pair)`](../utils_new/DatasetGenerator_new.py#L406)
  - [`DatasetGeneratorCoreNew._generate_pair(self, pair, stage, generated, target_count)`](../utils_new/DatasetGenerator_new.py#L444)
  - [`DatasetGeneratorCoreNew._ensure_consistent_yolo_kinds(self, records)`](../utils_new/DatasetGenerator_new.py#L619)
  - [`DatasetGeneratorCoreNew._assign_splits(self, records)`](../utils_new/DatasetGenerator_new.py#L654)
  - [`DatasetGeneratorCoreNew._distribute_and_export(self, stage, records)`](../utils_new/DatasetGenerator_new.py#L679)
  - [`DatasetGeneratorCoreNew._publish(self, stage, destination)`](../utils_new/DatasetGenerator_new.py#L717)
  - [`DatasetGeneratorCoreNew._delete_processed_sources(self)`](../utils_new/DatasetGenerator_new.py#L792)
  - [`DatasetGeneratorCoreNew.execute(self)`](../utils_new/DatasetGenerator_new.py#L813)
- Класс [`_Cancelled`](../utils_new/DatasetGenerator_new.py#L917) (`Exception`)
- Класс [`DatasetGeneratorThread`](../utils_new/DatasetGenerator_new.py#L928) (`QThread`)
  - [`DatasetGeneratorThread.__init__(self, *args, **kwargs)`](../utils_new/DatasetGenerator_new.py#L934)
  - [`DatasetGeneratorThread.cancel(self)`](../utils_new/DatasetGenerator_new.py#L940)
  - [`DatasetGeneratorThread.run(self)`](../utils_new/DatasetGenerator_new.py#L943)
- Класс [`DatasetGeneratorThread`](../utils_new/DatasetGenerator_new.py#L947)
  - [`DatasetGeneratorThread.__init__(self, *args, **kwargs)`](../utils_new/DatasetGenerator_new.py#L948)

<a id="utils-new-annotation-processing"></a>

## utils_new.annotation_processing

Исходник: [utils_new/annotation_processing.py](../utils_new/annotation_processing.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_new.dataset_view_new](#utils-new-dataset-view-new).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `MorphologyStep`, `apply_morphology as apply_shared_morphology`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `math`, `numpy`, `shapely.geometry.Polygon`, `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`.

**Объявления:**

- [`class_id_for_annotation(annotation)`](../utils_new/annotation_processing.py#L33) — Класс отсутствует/None/отрицательный -> 0 (общая политика V4).
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`normalized_annotation(annotation)`](../utils_new/annotation_processing.py#L49) — Добавляет класс 0 к устаревшим записям без класса, не меняя геометрию.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`polygon_for_annotation(annotation, width: int, height: int)`](../utils_new/annotation_processing.py#L75) — N×2 float64 вершины в пикселях исходного изображения, без округления.
- [`crop_bounds_for_annotation(annotation, image_w: int, image_h: int, padding: int=10)`](../utils_new/annotation_processing.py#L109) — Pixel-exclusive (x1,y1,x2,y2), обрезанные по границам снимка.
- [`crop_for_annotation(image: np.ndarray, annotation, padding: int=10)`](../utils_new/annotation_processing.py#L125) — Возвращает (copy ROI без ресайза, bounds); не меняет входной массив.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`annotation_in_crop(annotation, original_w: int, original_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L137) — Пересчитывает аннотацию в координаты кропа для векторного Qt-слоя.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`mask_roi_at_image_scale(mask: np.ndarray, image_w: int, image_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L167) — Извлекает маску точно для ROI без ресайза всей маски к размеру снимка.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`apply_class_mask_overlay(image: np.ndarray, mask: Optional[np.ndarray], opacity: float, color_for_class, *, pixel_offset: int=1, background_value: int=0, annotations=None, restrict_to_annotations=False)`](../utils_new/annotation_processing.py#L187) — Показывает индексную маску цветами палитры классов (BGR).
- [`_shapely_polygon(points)`](../utils_new/annotation_processing.py#L283)
- [`_raster_intersection_area(a: np.ndarray, b: np.ndarray, rect: Tuple[float, float, float, float], max_pixels=2000000)`](../utils_new/annotation_processing.py#L295) — Fallback для невыпуклых полигонов без shapely.
- [`overlap_ratio_of_smaller(annotation_a, annotation_b, width: int, height: int)`](../utils_new/annotation_processing.py#L319) — Пересечение / площадь МЕНЬШЕГО объекта, в интервале [0,1].
- [`find_strong_overlaps(annotations, width: int, height: int, threshold: float=0.7)`](../utils_new/annotation_processing.py#L364) — Сопоставляет индексы аннотаций с перекрывающими их соседями.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`detection_to_unified(detection, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L398) — Return one normalized annotation for a YOLO+AE/VAE detection dict.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`detections_to_unified(detections, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L443) — Convert all detections, keeping list positions stable for Qt selection.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`_name_for_class(names, class_id)`](../utils_new/annotation_processing.py#L459)
- [`_as_numpy(value)`](../utils_new/annotation_processing.py#L468) — Accept Ultralytics tensors and plain numpy arrays for standalone tests.
- [`yolo_result_to_detections(result, names=None, *, angle_fn=None)`](../utils_new/annotation_processing.py#L479) — Decode YOLO detect/OBB/segment into the existing AE/VAE dict contract.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`prepare_annotation_preview(image, annotation, padding=10, *, mask=None, opacity=0.5, color_for_class=None, pixel_offset=1, background_value=0)`](../utils_new/annotation_processing.py#L562) — Single crop pipeline used by the dataset viewer and YOLO-side previews.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`render_unified_annotations(image, annotations, color_for_class, *, selected_index=-1, selected_color=(0, 255, 255), text_color=(255, 255, 255), thickness=1, font_scale=0.5)`](../utils_new/annotation_processing.py#L587) — Optional OpenCV renderer for exporting detection imagery.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`render_ae_error_map(error, base, *, valid_region=None, overlay=True, alpha=0.5, show_mask=False, threshold=float('nan'))`](../utils_new/annotation_processing.py#L619) — Render an AE/VAE error heatmap with the original percentile stretch.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`longest_edge_angle_deg(corners)`](../utils_new/annotation_processing.py#L665)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`build_detection_mask(det, W, H)`](../utils_new/annotation_processing.py#L680) — Бинарная маска (uint8, 0/255) для детекции.
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`morph_image_in_mask(img, mask, erosion_px, dilation_px, shape_name='Rectangle', contour_only=True)`](../utils_new/annotation_processing.py#L702) — Применяет эрозию и/или дилатацию к изображению, сохраняя результат
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`apply_object_morphology(img, det, erosion_px=1, dilation_px=1, shape_name='Rectangle', contour_only=True, *, operations=None, contour_width_px=None)`](../utils_new/annotation_processing.py#L755) — Морфология изображения внутри выбранной YOLO-детекции.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`fill_outside_object(img, det, fill_mode, rotation_matrix=None, crop_offset=(0, 0))`](../utils_new/annotation_processing.py#L809) — Заливает пиксели вне маски объекта (fill_mode: 'none' / 'black' / 'white').
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`crop_to_detection(img, det, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L852) — Обрезает изображение по xyxy объекта с учётом padding и смещений.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`rotate_with_optional_crop(img, det, auto_crop, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L869) — Доворачивает изображение так, чтобы длинная сторона объекта стала
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`cls_suffix_from_detection(det, index=None)`](../utils_new/annotation_processing.py#L911) — Безопасный суффикс имени файла по классу объекта и его номеру в списке.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`limit_image_size(img, max_side)`](../utils_new/annotation_processing.py#L935) — Пропорционально уменьшает изображение, если длинная сторона > max_side.
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`aeb_transform_image(image, detection, *, erode=1, dilate=1, morphology_steps=None, contour_width_px=None, shape_name='Rectangle', contour_only=True, auto_crop=False, auto_rotate=False, padding=0, off_x=0, off_y=0, fine_angle=0.0, fill_mode='none', manual_ops=(), invert=False)`](../utils_new/annotation_processing.py#L949) — Common sequential image pipeline for live preview and save-all.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`annotation_to_detection(annotation, image_w, image_h, *, class_name=None)`](../utils_new/annotation_processing.py#L1012) — Build an AEB-compatible detection dict from normalized YOLO annotation.
- [`transform_annotation_image(image, annotation, **kwargs)`](../utils_new/annotation_processing.py#L1056) — Run the same AEB pipeline for a label from a saved dataset.
- [`_mask_extraction_dimensions(binary: np.ndarray, image_shape=None, min_area=None)`](../utils_new/annotation_processing.py#L1073) — Площадь отсечения — та же, что в старом get_display_params().
- [`extract_segments_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1091) — Segment: внешние контуры CHAIN_APPROX_SIMPLE, нормализованные вершины.
- [`extract_boxes_from_mask(binary: np.ndarray, *, image_shape=None, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1118) — Detect: connectedComponentsWithStats, 8-связность, как в старом коде.
- [`extract_obb_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1141) — OBB: minAreaRect + boxPoints, с исходным усечением до int32.
- [`extract_annotations_from_mask(binary: np.ndarray, mode='detect', *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1163) — Единая точка получения Detect/OBB/Segment из бинарной маски.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_new/annotation_processing.py#L1185) — Форматирует аннотацию для отображения в QListWidget.
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_new/annotation_processing.py#L1214) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).

<a id="utils-new-annotations---init--"></a>

## utils_new.annotations.__init__

Исходник: [utils_new/annotations/__init__.py](../utils_new/annotations/__init__.py#L1).

**Прямые потребители:** [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

**Прямые локальные импорты:**
- [utils_new.annotations.adapters](#utils-new-annotations-adapters): `from_detection_dict`, `from_legacy`, `from_legacy_many`, `from_ultralytics_result`, `legacy_convert_annotations`, `to_detection_dict`, `to_legacy`, `to_legacy_many`
- [utils_new.annotations.augmentation](#utils-new-annotations-augmentation): `AugmentationPlan`, `apply_augmentation_plan`, `build_augmentation_plan`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `LossyConversionError`, `bounds`, `canonical_obb_points`, `clip_annotation`, `clip_polygon_to_rect`, `convert_annotation`, `overlap_area`, `overlap_ratio_of_smaller`, `polygon_area`, `rectangle_points`
- [utils_new.annotations.masks](#utils-new-annotations-masks): `annotation_binary_mask`, `mask_roi_at_image_scale`, `mask_to_annotations`, `rasterize_annotations`, `remap_mask_indices`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `AnnotationSample`, `ImageSize`, `MaskEncoding`
- [utils_new.annotations.rendering](#utils-new-annotations-rendering): `overlay_class_mask`, `render_annotations`
- [utils_new.annotations.transforms](#utils-new-annotations-transforms): `Crop`, `Flip`, `Resize`, `Rotate`, `Rotate90`, `Shear`, `TransformPlan`, `TransformReport`, `Translate`, `apply_affine_to_points`, `compose_affine_matrices`, `extract_roi`, `transform_sample`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-annotations-adapters"></a>

## utils_new.annotations.adapters

Исходник: [utils_new/annotations/adapters.py](../utils_new/annotations/adapters.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `bounds`, `convert_annotation`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`from_legacy(value, size: ImageSize, *, allow_classless=False)`](../utils_new/annotations/adapters.py#L8) — Legacy tuple -> pixel Annotation; strict by default for dataset export.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_legacy(ann: Annotation, size: ImageSize)`](../utils_new/annotations/adapters.py#L54) — Canonical Annotation -> original normalized tuple; does not round/clamp.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_legacy_many(values, size, *, allow_classless=False)`](../utils_new/annotations/adapters.py#L69)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_legacy_many(values, size)`](../utils_new/annotations/adapters.py#L73)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_detection_dict(detection: dict, size: ImageSize)`](../utils_new/annotations/adapters.py#L77) — YOLO+AE detection dictionary (pixel xyxy, corners or mask_xy) -> Annotation.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`legacy_convert_annotations(values, size: ImageSize, target: str, *, allow_lossy=False, obb_policy='refit')`](../utils_new/annotations/adapters.py#L100) — Explicit bridge for the old converter when migrating DatasetGenerator.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_detection_dict(ann: Annotation, *, class_name=None, confidence=None)`](../utils_new/annotations/adapters.py#L113) — Canonical annotation -> YOLO+AE/VAE detection dictionary in pixels.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_ultralytics_result(result, size: ImageSize)`](../utils_new/annotations/adapters.py#L143) — Duck-typed Ultralytics result -> canonical annotations; no torch import.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-augmentation"></a>

## utils_new.annotations.augmentation

Исходник: [utils_new/annotations/augmentation.py](../utils_new/annotations/augmentation.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `AnnotationSample`, `ImageSize`
- [utils_new.annotations.transforms](#utils-new-annotations-transforms): `Crop`, `Flip`, `Resize`, `Rotate`, `Rotate90`, `Shear`, `TransformPlan`, `transform_sample`

**Внешние/stdlib импорты:** `__future__.annotations`, `dataclasses.dataclass`, `math`, `numpy`, `random`.

**Объявления:**

- Класс [`AugmentationPlan`](../utils_new/annotations/augmentation.py#L15)
- [`_next_size(size: ImageSize, op)`](../utils_new/annotations/augmentation.py#L20)
- [`build_augmentation_plan(size: ImageSize, config: dict, *, seed=None, rng=None)`](../utils_new/annotations/augmentation.py#L42) — Old-style config accepted; repeatable when seed is fixed.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`apply_augmentation_plan(sample: AnnotationSample, plan: AugmentationPlan, **kwargs)`](../utils_new/annotations/augmentation.py#L86)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-geometry"></a>

## utils_new.annotations.geometry

Исходник: [utils_new/annotations/geometry.py](../utils_new/annotations/geometry.py#L1).

**Прямые потребители:** [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`, `shapely.geometry.Polygon`, `shapely.geometry.box`, `shapely.validation.make_valid`.

**Объявления:**

- Класс [`GeometryError`](../utils_new/annotations/geometry.py#L18) (`ValueError`)
- Класс [`LossyConversionError`](../utils_new/annotations/geometry.py#L22) (`GeometryError`)
- [`polygon_area(points)`](../utils_new/annotations/geometry.py#L26)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`bounds(points)`](../utils_new/annotations/geometry.py#L34)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`rectangle_points(x0, y0, x1, y1)`](../utils_new/annotations/geometry.py#L39)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`canonical_obb_points(points)`](../utils_new/annotations/geometry.py#L43) — Best-fitting minimum-area rectangle in *pixel* coordinates.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`is_rectangle(points, tol=1e-06)`](../utils_new/annotations/geometry.py#L61)
  Явно импортируют: [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).
- [`is_axis_aligned(points, atol=1e-07)`](../utils_new/annotations/geometry.py#L73)
  Явно импортируют: [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`convert_annotation(ann: Annotation, target: str, *, allow_lossy=False, obb_policy='refit')`](../utils_new/annotations/geometry.py#L79) — Detect/OBB/Segment. Never silently return a different target subtype.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters).
- [`_clip_convex_polygon(points, rect)`](../utils_new/annotations/geometry.py#L107) — Sutherland–Hodgman with float64 coordinates, convex polygons only.
- [`_iter_polygons(geom)`](../utils_new/annotations/geometry.py#L135)
- [`clip_polygon_to_rect(points, rect, *, hole_policy='error', min_area=1e-09)`](../utils_new/annotations/geometry.py#L145) — Exact polygon/ROI intersection; may return multiple components.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`clip_annotation(ann: Annotation, rect, *, min_area=1.0, multi_policy='all', obb_policy='promote_to_segment', hole_policy='error')`](../utils_new/annotations/geometry.py#L180) — Clip against pixel-edge ROI. Coordinate origin is NOT shifted.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`overlap_area(a: Annotation, b: Annotation)`](../utils_new/annotations/geometry.py#L222)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`overlap_ratio_of_smaller(a: Annotation, b: Annotation)`](../utils_new/annotations/geometry.py#L230)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`sanitize_annotation_topology(ann: Annotation, *, hole_policy='error')`](../utils_new/annotations/geometry.py#L235) — Return (valid annotations, diagnostics) before geometric augmentation.
  Явно импортируют: [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new).

<a id="utils-new-annotations-masks"></a>

## utils_new.annotations.masks

Исходник: [utils_new/annotations/masks.py](../utils_new/annotations/masks.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `bounds`, `canonical_obb_points`, `is_axis_aligned`, `rectangle_points`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`, `MaskEncoding`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`_dtype_for_values(values)`](../utils_new/annotations/masks.py#L11)
- [`annotation_binary_mask(ann: Annotation, size: ImageSize)`](../utils_new/annotations/masks.py#L16) — Raster footprint with half-open pixel-center semantics for rectangles.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new).
- [`rasterize_annotations(annotations: list[Annotation], size: ImageSize, *, encoding: MaskEncoding=MaskEncoding(), overlap='last', dtype=None)`](../utils_new/annotations/masks.py#L40) — Rasterize detect/obb/segment to one label per pixel.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`remap_mask_indices(mask: np.ndarray, class_map: dict[int, int], *, source_encoding: MaskEncoding=MaskEncoding(), target_encoding: MaskEncoding=MaskEncoding(), unmapped='background', dtype=None)`](../utils_new/annotations/masks.py#L72) — Remap index image without intermediate value collisions or input mutation.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`mask_roi_at_image_scale(mask: np.ndarray, size: ImageSize, bounds_xyxy)`](../utils_new/annotations/masks.py#L106) — Nearest-neighbor ROI from a mask whose resolution differs from image.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`mask_to_annotations(mask: np.ndarray, *, encoding: MaskEncoding=MaskEncoding(), kind='segment', min_area=1.0, holes='error')`](../utils_new/annotations/masks.py#L121) — Lossy contour extraction from class mask; explicit policy for holes.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-model"></a>

## utils_new.annotations.model

Исходник: [utils_new/annotations/model.py](../utils_new/annotations/model.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `dataclasses.dataclass`, `dataclasses.field`, `numpy`, `typing.Literal`.

**Объявления:**

- Класс [`ImageSize`](../utils_new/annotations/model.py#L16)
  - [`ImageSize.__post_init__(self)`](../utils_new/annotations/model.py#L20)
- Класс [`Annotation`](../utils_new/annotations/model.py#L26)
  - [`Annotation.__post_init__(self)`](../utils_new/annotations/model.py#L36)
  - [`Annotation.with_points(self, pts, *, kind=None)`](../utils_new/annotations/model.py#L56)
- Класс [`MaskEncoding`](../utils_new/annotations/model.py#L61)
  - [`MaskEncoding.__post_init__(self)`](../utils_new/annotations/model.py#L71)
- Класс [`AnnotationSample`](../utils_new/annotations/model.py#L81)
  - [`AnnotationSample.__post_init__(self)`](../utils_new/annotations/model.py#L87)
  - [`AnnotationSample.size(self)`](../utils_new/annotations/model.py#L103)
  - [`AnnotationSample.detached(self)`](../utils_new/annotations/model.py#L107)

<a id="utils-new-annotations-rendering"></a>

## utils_new.annotations.rendering

Исходник: [utils_new/annotations/rendering.py](../utils_new/annotations/rendering.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `bounds`
- [utils_new.annotations.masks](#utils-new-annotations-masks): `annotation_binary_mask`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`, `MaskEncoding`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`_bgr_color(color_for_class, cls)`](../utils_new/annotations/rendering.py#L12)
- [`render_annotations(image, annotations: list[Annotation], color_for_class, *, selected_index=None, selected_color=(0, 255, 255), thickness=2, draw_labels=True, text_color=(255, 255, 255))`](../utils_new/annotations/rendering.py#L19) — BGR raster preview; does not modify image.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`overlay_class_mask(image, mask, color_for_class, *, encoding: MaskEncoding=MaskEncoding(), opacity=0.5, annotations=None, restrict_to_annotations=False)`](../utils_new/annotations/rendering.py#L42) — Blend indexed mask with global class palette (BGR), preserving background.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-transforms"></a>

## utils_new.annotations.transforms

Исходник: [utils_new/annotations/transforms.py](../utils_new/annotations/transforms.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `bounds`, `canonical_obb_points`, `clip_annotation`, `is_axis_aligned`, `is_rectangle`, `rectangle_points`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `AnnotationSample`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `dataclasses.field`, `numpy`, `typing.Literal`.

**Объявления:**

- Класс [`Flip`](../utils_new/annotations/transforms.py#L20)
- Класс [`Rotate90`](../utils_new/annotations/transforms.py#L26)
- Класс [`Rotate`](../utils_new/annotations/transforms.py#L31)
- Класс [`Shear`](../utils_new/annotations/transforms.py#L37)
- Класс [`Crop`](../utils_new/annotations/transforms.py#L44)
- Класс [`Resize`](../utils_new/annotations/transforms.py#L52)
- Класс [`Translate`](../utils_new/annotations/transforms.py#L59)
- Класс [`TransformPlan`](../utils_new/annotations/transforms.py#L66)
  - [`TransformPlan.__init__(self, steps=())`](../utils_new/annotations/transforms.py#L69)
- Класс [`TransformReport`](../utils_new/annotations/transforms.py#L74)
- [`_translate_matrix(x, y)`](../utils_new/annotations/transforms.py#L79)
- [`apply_affine_to_points(points, matrix)`](../utils_new/annotations/transforms.py#L83)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`compose_affine_matrices(matrices)`](../utils_new/annotations/transforms.py#L92) — Input order == application order: [M1,M2] -> M2 @ M1.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`_map_annotations(annotations, M, new_size, *, detect_policy, obb_policy, multi_policy, hole_policy, min_area, report, step_name)`](../utils_new/annotations/transforms.py#L105)
- [`_build_canvas_matrix(M, size: ImageSize, expand_canvas)`](../utils_new/annotations/transforms.py#L142)
- [`_pixel_center_matrix(M)`](../utils_new/annotations/transforms.py#L152)
- [`_border_value_for_image(fill, img)`](../utils_new/annotations/transforms.py#L156)
- [`_warp(sample, M, new_size, fill_color)`](../utils_new/annotations/transforms.py#L166)
- [`_resize(sample, step)`](../utils_new/annotations/transforms.py#L182)
- [`_apply_step(sample, step, *, report, detect_policy, obb_policy, multi_policy, hole_policy, min_area, fill_color)`](../utils_new/annotations/transforms.py#L214)
- [`transform_sample(sample: AnnotationSample, plan, *, detect_policy='aabb', obb_policy='promote_to_segment', multi_policy='all', hole_policy='error', min_area=1e-09, fill_color=(0, 0, 0), return_report=False)`](../utils_new/annotations/transforms.py#L275) — Transform image, canonical annotations and optional mask identically.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation).
- [`extract_roi(sample: AnnotationSample, bounds_xyxy, **kwargs)`](../utils_new/annotations/transforms.py#L297)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-dataset-catalog-new"></a>

## utils_new.dataset_catalog_new

Исходник: [utils_new/dataset_catalog_new.py](../utils_new/dataset_catalog_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `load_coco`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `read_yolo_labels`

**Внешние/stdlib импорты:** `__future__.annotations`, `dataclasses.dataclass`, `dataclasses.field`, `json`, `numpy`, `os`, `pathlib.Path`, `typing.Callable`, `typing.Optional`.

**Объявления:**

- Класс [`DatasetCatalog`](../utils_new/dataset_catalog_new.py#L27)
- Класс [`CocoSource`](../utils_new/dataset_catalog_new.py#L38)
- Класс [`CocoCatalog`](../utils_new/dataset_catalog_new.py#L45)
- Класс [`CatalogSourceError`](../utils_new/dataset_catalog_new.py#L51) (`ValueError`)
- [`load_yolo_catalog(path, log_cb=None, *, require_labels=False)`](../utils_new/dataset_catalog_new.py#L55) — Read existing YOLO data.yaml into UI-independent pairs and class names.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`load_unet_catalog(path, log_cb=None)`](../utils_new/dataset_catalog_new.py#L66) — Source mask pixel values map to names; preserve actual input offset.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`discover_coco_json(folder)`](../utils_new/dataset_catalog_new.py#L76) — Find only JSON files containing COCO images and annotations fields.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`import_coco_catalog(json_paths, existing_name_to_id=None, *, log_cb=None)`](../utils_new/dataset_catalog_new.py#L94) — Merge COCO sources by category NAME and remap objects to global IDs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`collect_label_classes(pairs, format_hint='auto', log_cb=None)`](../utils_new/dataset_catalog_new.py#L142) — Read class IDs from YOLO labels without a QApplication or label rendering.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`collect_mask_classes(pairs, log_cb=None)`](../utils_new/dataset_catalog_new.py#L159) — Return (source pixel IDs, decodable pairs); masks are never modified.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`resolve_class_name(names, cid)`](../utils_new/dataset_catalog_new.py#L181)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`dense_class_remap(ids)`](../utils_new/dataset_catalog_new.py#L197) — Map selected SOURCE identifiers to zero-based dense IDs, sorted.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

<a id="utils-new-dataset-coco-new"></a>

## utils_new.dataset_coco_new

Исходник: [utils_new/dataset_coco_new.py](../utils_new/dataset_coco_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `canonical_obb_points`, `from_legacy`, `to_legacy`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `read_image`, `write_yolo_labels`

**Внешние/stdlib импорты:** `json`, `numpy`, `os`, `pathlib.Path`.

**Объявления:**

- [`canonicalize_obb_corners(norm_points, width, height)`](../utils_new/dataset_coco_new.py#L13) — Return deterministic corners after fitting in *pixel* coordinates.
- [`_bbox_to_detect(bbox, w, h)`](../utils_new/dataset_coco_new.py#L20) — COCO bbox [x, y, bw, bh] в абс. пикселях → canonical detect.
- [`_segmentation_to_segment(seg, w, h)`](../utils_new/dataset_coco_new.py#L30) — COCO segmentation: [poly1, poly2, ...], каждый poly — плоский список абс. координат.
- [`_is_polygon_rectangle(poly_abs, tol=2.0)`](../utils_new/dataset_coco_new.py#L50) — Эвристика: 4 точки, стороны примерно параллельны друг другу,
- [`load_coco(json_path, log_cb=None, *, preserve_category_ids=False)`](../utils_new/dataset_coco_new.py#L79) — Загружает COCO JSON.
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`write_yolo_txt_for_coco_items(items, output_labels_dir, class_remap=None, log_cb=None, name_prefix='')`](../utils_new/dataset_coco_new.py#L208) — COCO canonical tuples -> shared YOLO writer, preserving source image pairing.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

<a id="utils-new-dataset-files-new"></a>

## utils_new.dataset_files_new

Исходник: [utils_new/dataset_files_new.py](../utils_new/dataset_files_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

**Прямые локальные импорты:**
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`

**Внешние/stdlib импорты:** `PIL.Image`, `collections.OrderedDict`, `concurrent.futures.ThreadPoolExecutor`, `cv2`, `numpy`, `pathlib.Path`, `threading`.

**Объявления:**

- [`_read_with_fallback(path, flags)`](../utils_new/dataset_files_new.py#L7) — Read image/mask with Unicode-path fallback (notably Windows Cyrillic).
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image(path)`](../utils_new/dataset_files_new.py#L21) — Read BGR image, keeping existing generator's 3-channel convention.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`normalize_index_mask_shape(mask, *, path=None)`](../utils_new/dataset_files_new.py#L30) — Normalize equivalent OpenCV single-channel shapes to (H, W).
- [`read_index_mask(path)`](../utils_new/dataset_files_new.py#L50) — Read indexed PNG as 2D integer array; accept decoder's (H,W,1).
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`write_image(path, image)`](../utils_new/dataset_files_new.py#L59) — Write image safely also when cv2.imwrite cannot handle the path.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image_with_fallback(path)`](../utils_new/dataset_files_new.py#L80) — Read image with original depth, TIFF and Unicode-path support.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`read_mask_with_fallback(path)`](../utils_new/dataset_files_new.py#L117) — Nullable indexed-mask reader; never squeeze a real RGB mask.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`image_size_from_file(path)`](../utils_new/dataset_files_new.py#L125) — Full-resolution (width,height), without decoding image pixels if possible.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).
- [`resize_to_max_side(img, max_side=1024)`](../utils_new/dataset_files_new.py#L138)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`normalize_to_uint8(img)`](../utils_new/dataset_files_new.py#L147) — Use the already-shared ImageStats conversion, not a second algorithm.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`convert_to_grayscale(img)`](../utils_new/dataset_files_new.py#L153)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint='auto')`](../utils_new/dataset_files_new.py#L158) — Qt-free lazy cache record used by both dataset tabs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- Класс [`LazyImageCache`](../utils_new/dataset_files_new.py#L196)
  - [`LazyImageCache.__init__(self, paths, loader, window=8, prefetch_ahead=4, max_resident=16, max_workers=2)`](../utils_new/dataset_files_new.py#L197)
  - [`LazyImageCache.__len__(self)`](../utils_new/dataset_files_new.py#L213)
  - [`LazyImageCache._trim_locked(self)`](../utils_new/dataset_files_new.py#L216)
  - [`LazyImageCache._load_sync(self, idx)`](../utils_new/dataset_files_new.py#L220)
  - [`LazyImageCache.is_loaded(self, idx)`](../utils_new/dataset_files_new.py#L237)
  - [`LazyImageCache.get_error(self, idx)`](../utils_new/dataset_files_new.py#L241)
  - [`LazyImageCache.get(self, idx)`](../utils_new/dataset_files_new.py#L245)
  - [`LazyImageCache._load_bg(self, idx)`](../utils_new/dataset_files_new.py#L263)
  - [`LazyImageCache.prefetch(self, indices)`](../utils_new/dataset_files_new.py#L290)
  - [`LazyImageCache.trim_around(self, center, keep_behind=3, keep_ahead=None)`](../utils_new/dataset_files_new.py#L305)
  - [`LazyImageCache.shutdown(self)`](../utils_new/dataset_files_new.py#L318)

<a id="utils-new-dataset-image-stats-new"></a>

## utils_new.dataset_image_stats_new

Исходник: [utils_new/dataset_image_stats_new.py](../utils_new/dataset_image_stats_new.py#L1).

**Прямые потребители:** [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.smart_view_new](#utils-new-smart-view-new).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- Класс [`ImageStats`](../utils_new/dataset_image_stats_new.py#L7)
  - [`ImageStats.empty_color_stats()`](../utils_new/dataset_image_stats_new.py#L9)
  - [`ImageStats.empty_bit_stats()`](../utils_new/dataset_image_stats_new.py#L13)
  - [`ImageStats.detect_color_type(img)`](../utils_new/dataset_image_stats_new.py#L18)
  - [`ImageStats.detect_bit_type(img, img_path)`](../utils_new/dataset_image_stats_new.py#L35)
  - [`ImageStats.to_uint8(img)`](../utils_new/dataset_image_stats_new.py#L55)
  - [`ImageStats.to_display_bgr(img)`](../utils_new/dataset_image_stats_new.py#L68)
  - [`ImageStats.to_gray(img)`](../utils_new/dataset_image_stats_new.py#L81)
  - [`ImageStats.to_bgr3(img)`](../utils_new/dataset_image_stats_new.py#L94)
  - [`ImageStats.format_stats_text(color_stats, bit_stats, has_pairs)`](../utils_new/dataset_image_stats_new.py#L107)

<a id="utils-new-dataset-io-new"></a>

## utils_new.dataset_io_new

Исходник: [utils_new/dataset_io_new.py](../utils_new/dataset_io_new.py#L1).

**Прямые потребители:** [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `ImageSize`, `MaskEncoding`, `bounds`, `from_legacy`, `mask_to_annotations`, `polygon_area`, `rasterize_annotations`, `to_legacy`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `_read_with_fallback`, `read_image`, `read_index_mask`, `write_image`
- [utils_new.dataset_output_new](#utils-new-dataset-output-new): `_coco_record`, `make_index_mask`, `write_classes_and_yaml`, `write_coco_split`, `write_index_mask`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `LABEL_HINTS`, `_normalized_xywhr`, `_xywhr_points`, `read_yolo_labels`, `write_yolo_labels`

**Внешние/stdlib импорты:** `cv2`, `json`, `math`, `numpy`, `pathlib.Path`, `yaml`.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-dataset-loader-new"></a>

## utils_new.dataset_loader_new

Исходник: [utils_new/dataset_loader_new.py](../utils_new/dataset_loader_new.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new): `PairValidationDialog`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `CatalogSourceError`, `collect_label_classes`, `collect_mask_classes`, `dense_class_remap`, `discover_coco_json`, `import_coco_catalog`, `load_unet_catalog`, `load_yolo_catalog`, `resolve_class_name`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new): `build_pair_display`, `scan_detection_pairs`, `scan_image_only_pairs`, `scan_segmentation_pairs`
- [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new): `PairValidator`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `is_unet_yaml`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QTableWidgetItem`, `os`, `tempfile`, `traceback`.

**Объявления:**

- Класс [`DatasetLoaderMixin`](../utils_new/dataset_loader_new.py#L31)
  - [`DatasetLoaderMixin._after_load_hook(self)`](../utils_new/dataset_loader_new.py#L53)
  - [`DatasetLoaderMixin._after_type_change_hook(self, new_type)`](../utils_new/dataset_loader_new.py#L60)
  - [`DatasetLoaderMixin._sync_all_states(self, *_)`](../utils_new/dataset_loader_new.py#L65)
  - [`DatasetLoaderMixin._refresh_mask_pixel_column(self)`](../utils_new/dataset_loader_new.py#L81)
  - [`DatasetLoaderMixin.log(self, message)`](../utils_new/dataset_loader_new.py#L91)
  - [`DatasetLoaderMixin._log_debug(self, msg)`](../utils_new/dataset_loader_new.py#L100)
  - [`DatasetLoaderMixin._log_exception(self, context, exc=None)`](../utils_new/dataset_loader_new.py#L107)
  - [`DatasetLoaderMixin._copy_selected_rows_to_clipboard(self)`](../utils_new/dataset_loader_new.py#L119)
  - [`DatasetLoaderMixin._on_file_list_double_clicked(self, item)`](../utils_new/dataset_loader_new.py#L127)
  - [`DatasetLoaderMixin.select_images_folder(self)`](../utils_new/dataset_loader_new.py#L139)
  - [`DatasetLoaderMixin.select_labels_folder(self)`](../utils_new/dataset_loader_new.py#L148)
  - [`DatasetLoaderMixin.load_from_yaml(self)`](../utils_new/dataset_loader_new.py#L169)
  - [`DatasetLoaderMixin._load_yolo_yaml_from_path(self, file_path)`](../utils_new/dataset_loader_new.py#L186)
  - [`DatasetLoaderMixin.load_from_coco(self)`](../utils_new/dataset_loader_new.py#L235)
  - [`DatasetLoaderMixin.add_from_coco(self)`](../utils_new/dataset_loader_new.py#L238)
  - [`DatasetLoaderMixin._handle_load_coco(self, replace)`](../utils_new/dataset_loader_new.py#L241)
  - [`DatasetLoaderMixin._ask_coco_source(self)`](../utils_new/dataset_loader_new.py#L261)
  - [`DatasetLoaderMixin._find_coco_json_in_folder(self, folder)`](../utils_new/dataset_loader_new.py#L277)
  - [`DatasetLoaderMixin._get_coco_tmp_labels_dir(self, first_json_path=None)`](../utils_new/dataset_loader_new.py#L281)
  - [`DatasetLoaderMixin._load_coco_from_paths(self, json_paths, replace=True)`](../utils_new/dataset_loader_new.py#L289)
  - [`DatasetLoaderMixin.load_from_unet_yaml(self)`](../utils_new/dataset_loader_new.py#L367)
  - [`DatasetLoaderMixin._load_unet_from_path(self, yaml_path)`](../utils_new/dataset_loader_new.py#L374)
  - [`DatasetLoaderMixin._apply_loaded_pairs(self, pairs, input_type, input_format, class_names_from_yaml=None, class_names_override=None, source_label='')`](../utils_new/dataset_loader_new.py#L411)
  - [`DatasetLoaderMixin.scan_pairs(self)`](../utils_new/dataset_loader_new.py#L465)
  - [`DatasetLoaderMixin._validate_all_pairs(self)`](../utils_new/dataset_loader_new.py#L530)
  - [`DatasetLoaderMixin._recompute_color_stats(self)`](../utils_new/dataset_loader_new.py#L551)
  - [`DatasetLoaderMixin._update_color_stats_label(self)`](../utils_new/dataset_loader_new.py#L557)
  - [`DatasetLoaderMixin._open_pair_validation(self, pair)`](../utils_new/dataset_loader_new.py#L567)
  - [`DatasetLoaderMixin._make_list_item(self, text, tag=None, error_msg='')`](../utils_new/dataset_loader_new.py#L593)
  - [`DatasetLoaderMixin._rebuild_file_list_display(self, pairs=None)`](../utils_new/dataset_loader_new.py#L617)
  - [`DatasetLoaderMixin._is_include_empty(self)`](../utils_new/dataset_loader_new.py#L693)
  - [`DatasetLoaderMixin._on_include_empty_toggled(self, *_)`](../utils_new/dataset_loader_new.py#L696)
  - [`DatasetLoaderMixin._on_dataset_type_changed(self, idx)`](../utils_new/dataset_loader_new.py#L702)
  - [`DatasetLoaderMixin._on_input_format_changed(self, idx)`](../utils_new/dataset_loader_new.py#L726)
  - [`DatasetLoaderMixin._fill_class_table_from_dict(self, class_names_dict, preserve_include=True)`](../utils_new/dataset_loader_new.py#L738)
  - [`DatasetLoaderMixin._get_yaml_name_for_id(self, cid)`](../utils_new/dataset_loader_new.py#L782)
  - [`DatasetLoaderMixin.collect_classes(self)`](../utils_new/dataset_loader_new.py#L786)
  - [`DatasetLoaderMixin.update_included_classes(self)`](../utils_new/dataset_loader_new.py#L875)
  - [`DatasetLoaderMixin.on_class_table_item_changed(self, item)`](../utils_new/dataset_loader_new.py#L955)
  - [`DatasetLoaderMixin._update_pair_count_label(self, extra_text, error=False)`](../utils_new/dataset_loader_new.py#L961)

<a id="utils-new-dataset-mask-diagnostics-new"></a>

## utils_new.dataset_mask_diagnostics_new

Исходник: [utils_new/dataset_mask_diagnostics_new.py](../utils_new/dataset_mask_diagnostics_new.py#L1).

**Прямые потребители:** [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`, `write_image`

**Внешние/stdlib импорты:** `__future__.annotations`, `argparse`, `cv2`, `dataclasses.dataclass`, `numpy`, `pathlib.Path`.

**Объявления:**

- Класс [`MaskStatistics`](../utils_new/dataset_mask_diagnostics_new.py#L28)
  - [`MaskStatistics.total_pixels(self)`](../utils_new/dataset_mask_diagnostics_new.py#L36)
  - [`MaskStatistics.foreground_pixels(self)`](../utils_new/dataset_mask_diagnostics_new.py#L40)
  - [`MaskStatistics.foreground_fraction(self)`](../utils_new/dataset_mask_diagnostics_new.py#L44)
  - [`MaskStatistics.is_empty(self)`](../utils_new/dataset_mask_diagnostics_new.py#L48)
  - [`MaskStatistics.brief(self, max_classes=15)`](../utils_new/dataset_mask_diagnostics_new.py#L51)
- [`mask_statistics(mask: np.ndarray, *, background_value=0)`](../utils_new/dataset_mask_diagnostics_new.py#L61) — Return exact histogram without modifying or reinterpreting label pixels.
  Явно импортируют: [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new).
- [`colorize_index_mask(mask: np.ndarray, *, pixel_offset=1, background_value=0, color_for_class=None)`](../utils_new/dataset_mask_diagnostics_new.py#L91) — Return 8-bit 3-channel BGR preview, never alter original numeric mask.
- [`save_mask_preview(mask_path, output_path, *, pixel_offset=1, background_value=0, color_for_class=None)`](../utils_new/dataset_mask_diagnostics_new.py#L115) — Read a REAL saved mask and write a distinct color preview PNG.
- [`main(argv=None)`](../utils_new/dataset_mask_diagnostics_new.py#L132)

<a id="utils-new-dataset-mask-preview-new"></a>

## utils_new.dataset_mask_preview_new

Исходник: [utils_new/dataset_mask_preview_new.py](../utils_new/dataset_mask_preview_new.py#L1).

**Прямые потребители:** [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `from_legacy`
- [utils_new.annotations.masks](#utils-new-annotations-masks): `annotation_binary_mask`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `numpy`, `typing.Mapping`, `typing.Optional`.

**Объявления:**

- Класс [`MaskPreviewInfo`](../utils_new/dataset_mask_preview_new.py#L18)
  - [`MaskPreviewInfo.is_empty(self)`](../utils_new/dataset_mask_preview_new.py#L25)
  - [`MaskPreviewInfo.percentage(self)`](../utils_new/dataset_mask_preview_new.py#L29)
  - [`MaskPreviewInfo.description(self)`](../utils_new/dataset_mask_preview_new.py#L32)
- [`_validate_and_resize(mask: np.ndarray, width: int, height: int)`](../utils_new/dataset_mask_preview_new.py#L41)
- [`blend_index_mask(image_bgr: np.ndarray, mask: np.ndarray, color_for_class, *, opacity: float=0.5, pixel_offset: int=1, background_value: Optional[int]=0, pixel_to_class: Optional[Mapping[int, int]]=None, annotations=None, prefer_annotation_classes: bool=False)`](../utils_new/dataset_mask_preview_new.py#L49) — Return ``(BGR image, MaskPreviewInfo)`` without Qt, Ultralytics or writes.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new).
- [`_bgr(color_for_class, class_id)`](../utils_new/dataset_mask_preview_new.py#L129)

<a id="utils-new-dataset-output-new"></a>

## utils_new.dataset_output_new

Исходник: [utils_new/dataset_output_new.py](../utils_new/dataset_output_new.py#L1).

**Прямые потребители:** [utils_new.dataset_io_new](#utils-new-dataset-io-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `MaskEncoding`, `bounds`, `mask_to_annotations`, `polygon_area`, `rasterize_annotations`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `write_image`

**Внешние/stdlib импорты:** `__future__.annotations`, `json`, `numpy`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`write_index_mask(path, mask)`](../utils_new/dataset_output_new.py#L12)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`make_index_mask(annotations, size, *, offset=1)`](../utils_new/dataset_output_new.py#L23)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_classes_and_yaml(output_dir, class_names, *, out_yolo_txt=True, out_unet_yaml=False, out_masks_offset=1, input_type=0, dataset_path=None)`](../utils_new/dataset_output_new.py#L31)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`_coco_record(ann, image_id, ann_id)`](../utils_new/dataset_output_new.py#L55)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_coco_split(output_dir, split, rows, class_names, *, masks_offset=1, log=None)`](../utils_new/dataset_output_new.py#L67) — rows: [(image_name, image_size, annotations, optional_mask), ...].
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).

<a id="utils-new-dataset-pair-tools-new"></a>

## utils_new.dataset_pair_tools_new

Исходник: [utils_new/dataset_pair_tools_new.py](../utils_new/dataset_pair_tools_new.py#L1).

**Прямые потребители:** [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `_read_with_fallback`, `write_image`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new): `IMG_EXTENSIONS`, `INPUT_FORMAT_AUTO`, `PairValidator`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- [`_list_images(folder)`](../utils_new/dataset_pair_tools_new.py#L12)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_detection_pairs(images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_tools_new.py#L22)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_segmentation_pairs(images_folder, masks_folder, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L42)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_image_only_pairs(images_folder)`](../utils_new/dataset_pair_tools_new.py#L77)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_conversion(pairs, validator, target)`](../utils_new/dataset_pair_tools_new.py#L82)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_color(pairs, target, progress_cb=None, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L97)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_8bit_png(pairs, validator)`](../utils_new/dataset_pair_tools_new.py#L125) — Возвращает (all_pairs, need_convert).
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None, skip_masks=False)`](../utils_new/dataset_pair_tools_new.py#L140) — skip_masks: если True — pairs трактуются как pairs из (img, mask),
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`build_pair_display(pair, validator, dataset_type, include_empty)`](../utils_new/dataset_pair_tools_new.py#L183)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

<a id="utils-new-dataset-pair-validator-new"></a>

## utils_new.dataset_pair_validator_new

Исходник: [utils_new/dataset_pair_validator_new.py](../utils_new/dataset_pair_validator_new.py#L1).

**Прямые потребители:** [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `to_legacy`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `_read_with_fallback`, `read_index_mask`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `read_yolo_labels`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- [`read_mask_with_fallback(path)`](../utils_new/dataset_pair_validator_new.py#L21)
  Явно импортируют: [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`load_annotations(path, image_w, image_h, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L29) — Legacy tuple API backed by the canonical parser (no old image_io).
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- Класс [`PairValidator`](../utils_new/dataset_pair_validator_new.py#L36)
  - [`PairValidator.__init__(self, dataset_type=0, input_format=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L48)
  - [`PairValidator.set_dataset_type(self, dataset_type)`](../utils_new/dataset_pair_validator_new.py#L54)
  - [`PairValidator.set_input_format(self, fmt)`](../utils_new/dataset_pair_validator_new.py#L58)
  - [`PairValidator.clear_cache(self)`](../utils_new/dataset_pair_validator_new.py#L62)
  - [`PairValidator.validate_full(self, pair)`](../utils_new/dataset_pair_validator_new.py#L65)
  - [`PairValidator.validate(self, pair)`](../utils_new/dataset_pair_validator_new.py#L74)
  - [`PairValidator.validate_all(self, pairs, progress_cb=None)`](../utils_new/dataset_pair_validator_new.py#L78)
  - [`PairValidator.recompute_stats(self, pairs)`](../utils_new/dataset_pair_validator_new.py#L93)
  - [`PairValidator._validate_impl(self, pair)`](../utils_new/dataset_pair_validator_new.py#L102)
  - [`PairValidator.detect_annotation_type(label_path, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L285)
  - [`PairValidator.check_pair_type_consistency()`](../utils_new/dataset_pair_validator_new.py#L327)

<a id="utils-new-dataset-preparation-bridge"></a>

## utils_new.dataset_preparation_bridge

Исходник: [utils_new/dataset_preparation_bridge.py](../utils_new/dataset_preparation_bridge.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `typing.Iterable`.

**Объявления:**

- [`mask_classes_from_rows(rows: Iterable[tuple[int, str, bool]])`](../utils_new/dataset_preparation_bridge.py#L13) — Return (pixel->class_id, class_id->name) for selected positive pixels.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).
- [`normalize_generation_pairs(pairs, input_type)`](../utils_new/dataset_preparation_bridge.py#L36) — Convert legacy (image, label-or-mask) to (image, label, mask).
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

<a id="utils-new-dataset-review-view"></a>

## utils_new.dataset_review_view

Исходник: [utils_new/dataset_review_view.py](../utils_new/dataset_review_view.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [utils_new.smart_view_new](#utils-new-smart-view-new): `SmartGraphicsViewNew`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QGraphicsView`, `__future__.annotations`.

**Объявления:**

- Класс [`DatasetReviewGraphicsView`](../utils_new/dataset_review_view.py#L14) (`SmartGraphicsViewNew`)
  - [`DatasetReviewGraphicsView.__init__(self, parent=None)`](../utils_new/dataset_review_view.py#L17)
  - [`DatasetReviewGraphicsView.mousePressEvent(self, event)`](../utils_new/dataset_review_view.py#L28)
  - [`DatasetReviewGraphicsView.mouseMoveEvent(self, event)`](../utils_new/dataset_review_view.py#L60)
  - [`DatasetReviewGraphicsView.mouseReleaseEvent(self, event)`](../utils_new/dataset_review_view.py#L63)
  - [`DatasetReviewGraphicsView.mouseDoubleClickEvent(self, event)`](../utils_new/dataset_review_view.py#L78)
  - [`DatasetReviewGraphicsView.contextMenuEvent(self, event)`](../utils_new/dataset_review_view.py#L81)

<a id="utils-new-dataset-unet-new"></a>

## utils_new.dataset_unet_new

Исходник: [utils_new/dataset_unet_new.py](../utils_new/dataset_unet_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `MaskEncoding`, `mask_to_annotations`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `write_yolo_labels`

**Внешние/stdlib импорты:** `cv2`, `os`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`load_unet_yaml(yaml_path, log_cb=None)`](../utils_new/dataset_unet_new.py#L7) — Загружает masks_data.yaml (или data.yaml с masks-полями).
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`is_unet_yaml(yaml_path)`](../utils_new/dataset_unet_new.py#L120) — Эвристика: является ли YAML U-Net-структурой.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`load_unet_yaml_pairs(yaml_path, log_cb=None)`](../utils_new/dataset_unet_new.py#L143) — Compatibility tuple for tabs, delegating all YAML logic to load_unet_yaml.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`write_yolo_txt_for_unet_pairs(pairs, output_labels_dir, *, class_names, mask_offset=1, include_empty=True, log_cb=None)`](../utils_new/dataset_unet_new.py#L150) — Convert indexed masks to editable YOLO-Segment labels (lossy contours).
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).

<a id="utils-new-dataset-validation-new"></a>

## utils_new.dataset_validation_new

Исходник: [utils_new/dataset_validation_new.py](../utils_new/dataset_validation_new.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `to_legacy`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `_read_with_fallback`, `read_index_mask`, `write_image`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `read_yolo_labels`
- [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new): `_list_images`, `build_pair_display`, `convert_pairs_color`, `convert_pairs_to_8bit_png`, `scan_detection_pairs`, `scan_image_only_pairs`, `scan_segmentation_pairs`, `select_pairs_for_8bit_png`, `select_pairs_for_conversion`
- [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new): `IMG_EXTENSIONS`, `INPUT_FORMAT_AUTO`, `INPUT_FORMAT_DETECT`, `INPUT_FORMAT_MIXED`, `INPUT_FORMAT_OBB_CORNERS`, `INPUT_FORMAT_OBB_XYWHR`, `INPUT_FORMAT_SEGMENT`, `PairValidator`, `_VALID_INPUT_FORMATS`, `load_annotations`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-dataset-view-new"></a>

## utils_new.dataset_view_new

Исходник: [utils_new/dataset_view_new.py](../utils_new/dataset_view_new.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `annotation_processing as _implementation`

**Внешние/stdlib импорты:** `sys`.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-dataset-yaml-new"></a>

## utils_new.dataset_yaml_new

Исходник: [utils_new/dataset_yaml_new.py](../utils_new/dataset_yaml_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `yaml`.

**Объявления:**

- [`_resolve_relative_to_yaml(raw_path, yaml_path)`](../utils_new/dataset_yaml_new.py#L8) — Абсолютный путь: если raw_path абсолютный — берём как есть,
- [`_resolve_split_dirs(split_val, base_path)`](../utils_new/dataset_yaml_new.py#L20) — split_val может быть:
- [`_find_label_for_image(img_path, base_path, split, rel_dir)`](../utils_new/dataset_yaml_new.py#L69) — Ищет .txt для изображения в стандартных местах YOLO-структуры.
- [`find_mask_for_image(img_path, base_path=None)`](../utils_new/dataset_yaml_new.py#L109) — Ищет маску для изображения в стандартных местах YAML-структуры.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_yolo_yaml(yaml_path, log_cb=None)`](../utils_new/dataset_yaml_new.py#L150) — Парсит YOLO data.yaml.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).

<a id="utils-new-dataset-yolo-labels-new"></a>

## utils_new.dataset_yolo_labels_new

Исходник: [utils_new/dataset_yolo_labels_new.py](../utils_new/dataset_yolo_labels_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `ImageSize`, `from_legacy`, `to_legacy`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `is_rectangle`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `image_size_from_file`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `math`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`_xywhr_points(vals, size=None)`](../utils_new/dataset_yolo_labels_new.py#L12) — Decode normalized ``cx,cy,w,h,angle`` into rectangle corners.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L39) — Read YOLO txt variants into canonical pixel-space annotations.
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new).
- [`_normalized_xywhr(ann, size)`](../utils_new/dataset_yolo_labels_new.py#L111) — OBB as ``cx/W, cy/H, width_px/W, height_px/H, radians_px``.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners')`](../utils_new/dataset_yolo_labels_new.py#L134)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new).
- [`canonicalize_obb_corners(norm_points, width=1, height=1)`](../utils_new/dataset_yolo_labels_new.py#L174) — Order the original four vertices, without altering their coordinates.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_annotations_for_image(label_path, image_path, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L197) — Read normalized tuples using the associated image's *real* dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`save_annotations_for_image(annotations, label_path, image_path, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L205) — Prevent Save All from using the current (different) image dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`_tolerant_editor_rows(path, size, format_hint='auto')`](../utils_new/dataset_yolo_labels_new.py#L214) — Recover syntactically valid, geometrically invalid rows for manual repair.
- [`load_annotations(txt_path, img_w, img_h, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L256)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_files_new](#utils-new-dataset-files-new).
- [`save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L276)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).

<a id="utils-new-morphology"></a>

## utils_new.morphology

Исходник: [utils_new/morphology.py](../utils_new/morphology.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods), [ui.morphology_ui](#ui-morphology-ui), [utils_new.annotation_processing](#utils-new-annotation-processing).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `numpy`, `typing.Mapping`, `typing.Optional`, `typing.Sequence`, `typing.Union`.

**Объявления:**

- Класс [`MorphologyStep`](../utils_new/morphology.py#L59)
- [`_int_value(value, field: str, minimum: int)`](../utils_new/morphology.py#L73)
- [`_shape_code(shape: str)`](../utils_new/morphology.py#L82)
- [`build_kernel(size_px: int, shape: str='Rectangle')`](../utils_new/morphology.py#L91) — Квадратное структурирующее ядро OpenCV точного размера ``size_px``.
- [`_normalized_steps(operations: Sequence[Union[MorphologyStep, Mapping]])`](../utils_new/morphology.py#L100)
- [`_region_mask(mask: np.ndarray, image_shape: tuple)`](../utils_new/morphology.py#L128)
- [`make_contour_mask(mask: np.ndarray, width_px: int, shape: str='Rectangle')`](../utils_new/morphology.py#L142) — Внутренняя полоса шириной ``width_px`` вдоль границы объекта.
- [`apply_morphology(image: np.ndarray, operations: Sequence[Union[MorphologyStep, Mapping]], *, shape: str='Rectangle', mask: Optional[np.ndarray]=None, contour_only: bool=False, contour_width_px: Optional[int]=None, contour_shape: Optional[str]=None)`](../utils_new/morphology.py#L163) — Применить операции в указанной последовательности к изображению или ROI.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods), [utils_new.annotation_processing](#utils-new-annotation-processing).

<a id="utils-new-smart-view-new"></a>

## utils_new.smart_view_new

Исходник: [utils_new/smart_view_new.py](../utils_new/smart_view_new.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_review_view](#utils-new-dataset-review-view).

**Прямые локальные импорты:**
- [utils.settings](#utils-settings): `settings`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QPointF`, `PyQt5.QtCore.QRectF`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QBrush`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QFont`, `PyQt5.QtGui.QFontMetricsF`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPen`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtGui.QPolygonF`, `PyQt5.QtGui.QTransform`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QGraphicsPixmapItem`, `PyQt5.QtWidgets.QGraphicsScene`, `PyQt5.QtWidgets.QGraphicsView`, `PyQt5.QtWidgets.QMenu`, `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `dataclasses.replace`, `math`, `numpy`.

**Объявления:**

- Класс [`AnnotationDisplayStyle`](../utils_new/smart_view_new.py#L40)
- [`set_fill_color_button_preview(button, color)`](../utils_new/smart_view_new.py#L70) — Общий Qt-предпросмотр выбранного цвета без создания новой утилиты.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- Класс [`SmartGraphicsViewNew`](../utils_new/smart_view_new.py#L89) (`QGraphicsView`)
  - [`SmartGraphicsViewNew.__init__(self, parent=None)`](../utils_new/smart_view_new.py#L154)
  - [`SmartGraphicsViewNew._setup_pan_mode(self)`](../utils_new/smart_view_new.py#L250)
  - [`SmartGraphicsViewNew._setup_drawing_mode(self)`](../utils_new/smart_view_new.py#L254)
  - [`SmartGraphicsViewNew._setup_edit_mode(self)`](../utils_new/smart_view_new.py#L271)
  - [`SmartGraphicsViewNew.set_suggested_save_name(self, name)`](../utils_new/smart_view_new.py#L284)
  - [`SmartGraphicsViewNew.set_drawing_tool(self, tool)`](../utils_new/smart_view_new.py#L290)
  - [`SmartGraphicsViewNew.set_edit_mode(self, enabled)`](../utils_new/smart_view_new.py#L309)
  - [`SmartGraphicsViewNew._clear_edit_state(self)`](../utils_new/smart_view_new.py#L325)
  - [`SmartGraphicsViewNew.set_annotations(self, annotations, img_width, img_height)`](../utils_new/smart_view_new.py#L341)
  - [`SmartGraphicsViewNew._normalize_classless_annotation(ann)`](../utils_new/smart_view_new.py#L357)
  - [`SmartGraphicsViewNew.set_display_style(self, style=None, **overrides)`](../utils_new/smart_view_new.py#L390)
  - [`SmartGraphicsViewNew.apply_global_display_settings(self, _new_settings=None)`](../utils_new/smart_view_new.py#L408)
  - [`SmartGraphicsViewNew.set_annotation_visibility(self, *, show_frames=None, show_labels=None, fill_white=None, fill_color=None)`](../utils_new/smart_view_new.py#L425)
  - [`SmartGraphicsViewNew.set_class_color_provider(self, provider)`](../utils_new/smart_view_new.py#L456)
  - [`SmartGraphicsViewNew.set_class_name_provider(self, provider)`](../utils_new/smart_view_new.py#L462)
  - [`SmartGraphicsViewNew.refresh_annotations(self)`](../utils_new/smart_view_new.py#L467)
  - [`SmartGraphicsViewNew.set_selected_index(self, idx)`](../utils_new/smart_view_new.py#L471)
  - [`SmartGraphicsViewNew.set_callbacks(self, on_rect_drawn=None, on_obb_drawn=None, on_segment_drawn=None, on_scribble_added=None, on_display_update=None, on_reset_tool=None, on_annotation_modified=None, on_selection_changed=None, on_annotation_right_clicked=None, on_log=None)`](../utils_new/smart_view_new.py#L493)
  - [`SmartGraphicsViewNew.set_image_data(self, width, height, fg_scribbles=None, bg_scribbles=None)`](../utils_new/smart_view_new.py#L510)
  - [`SmartGraphicsViewNew._sync_scene_geometry(self)`](../utils_new/smart_view_new.py#L517)
  - [`SmartGraphicsViewNew.set_pixmap(self, pixmap, preserve_view=True)`](../utils_new/smart_view_new.py#L534)
  - [`SmartGraphicsViewNew.reset_view(self)`](../utils_new/smart_view_new.py#L553)
  - [`SmartGraphicsViewNew.save_current_image(self)`](../utils_new/smart_view_new.py#L560)
  - [`SmartGraphicsViewNew.save_source_image(self)`](../utils_new/smart_view_new.py#L577)
  - [`SmartGraphicsViewNew._scene_per_px(self)`](../utils_new/smart_view_new.py#L592)
  - [`SmartGraphicsViewNew._get_points_pixel(self, ann)`](../utils_new/smart_view_new.py#L604)
  - [`SmartGraphicsViewNew.point_in_polygon(self, pt, poly)`](../utils_new/smart_view_new.py#L631)
  - [`SmartGraphicsViewNew._point_to_segment_dist(px, py, x1, y1, x2, y2)`](../utils_new/smart_view_new.py#L651)
  - [`SmartGraphicsViewNew._hit_test_point(self, point, pos_pixel, radius=None)`](../utils_new/smart_view_new.py#L667)
  - [`SmartGraphicsViewNew._hit_test_detect(self, ann, pos_pixel)`](../utils_new/smart_view_new.py#L675)
  - [`SmartGraphicsViewNew._hit_test_segment(self, ann, pos_pixel)`](../utils_new/smart_view_new.py#L687)
  - [`SmartGraphicsViewNew._hit_test_any(self, ann, pos_pixel)`](../utils_new/smart_view_new.py#L699)
  - [`SmartGraphicsViewNew._find_hit(self, pos_pixel, prefer_idx=-1)`](../utils_new/smart_view_new.py#L711)
  - [`SmartGraphicsViewNew._obb_to_params(self, pts_pixel)`](../utils_new/smart_view_new.py#L732)
  - [`SmartGraphicsViewNew._obb_to_points(self, cx, cy, w, h, angle)`](../utils_new/smart_view_new.py#L775)
  - [`SmartGraphicsViewNew._obb_points_in_bounds(self, points, eps=0.5)`](../utils_new/smart_view_new.py#L815)
  - [`SmartGraphicsViewNew._obb_world_half_extent(self, w, h, angle)`](../utils_new/smart_view_new.py#L822)
  - [`SmartGraphicsViewNew._get_obb_handles(self, pts_pixel)`](../utils_new/smart_view_new.py#L829)
  - [`SmartGraphicsViewNew._hit_test_obb_handles(self, pts_pixel, pos_pixel)`](../utils_new/smart_view_new.py#L889)
  - [`SmartGraphicsViewNew._max_height_inside(self, x1, y1, x2, y2, dir_x, dir_y)`](../utils_new/smart_view_new.py#L925)
  - [`SmartGraphicsViewNew._compute_obb_preview_params(self)`](../utils_new/smart_view_new.py#L956)
  - [`SmartGraphicsViewNew.wheelEvent(self, event)`](../utils_new/smart_view_new.py#L1002)
  - [`SmartGraphicsViewNew.mousePressEvent(self, event)`](../utils_new/smart_view_new.py#L1008)
  - [`SmartGraphicsViewNew.mouseMoveEvent(self, event)`](../utils_new/smart_view_new.py#L1169)
  - [`SmartGraphicsViewNew.mouseReleaseEvent(self, event)`](../utils_new/smart_view_new.py#L1523)
  - [`SmartGraphicsViewNew.mouseDoubleClickEvent(self, event)`](../utils_new/smart_view_new.py#L1554)
  - [`SmartGraphicsViewNew._try_segment_double_click(self, pos_pixel)`](../utils_new/smart_view_new.py#L1585)
  - [`SmartGraphicsViewNew._apply_segment_change(self, idx, norm_flat, log_msg)`](../utils_new/smart_view_new.py#L1691)
  - [`SmartGraphicsViewNew.leaveEvent(self, event)`](../utils_new/smart_view_new.py#L1719)
  - [`SmartGraphicsViewNew.contextMenuEvent(self, event)`](../utils_new/smart_view_new.py#L1724)
  - [`SmartGraphicsViewNew._start_drawing(self, event)`](../utils_new/smart_view_new.py#L1742)
  - [`SmartGraphicsViewNew._update_segment_cursor(self, event)`](../utils_new/smart_view_new.py#L1792)
  - [`SmartGraphicsViewNew._update_segment_line(self, event)`](../utils_new/smart_view_new.py#L1800)
  - [`SmartGraphicsViewNew._update_obb_draw(self, event)`](../utils_new/smart_view_new.py#L1820)
  - [`SmartGraphicsViewNew._continue_drawing(self, event)`](../utils_new/smart_view_new.py#L1828)
  - [`SmartGraphicsViewNew._finish_drawing(self, event)`](../utils_new/smart_view_new.py#L1843)
  - [`SmartGraphicsViewNew._finish_segment(self)`](../utils_new/smart_view_new.py#L1858)
  - [`SmartGraphicsViewNew._ensure_min_polygon_points(self, pts)`](../utils_new/smart_view_new.py#L1882)
  - [`SmartGraphicsViewNew._finish_segment_line(self)`](../utils_new/smart_view_new.py#L1913)
  - [`SmartGraphicsViewNew._finish_obb(self)`](../utils_new/smart_view_new.py#L1956)
  - [`SmartGraphicsViewNew._reset_obb_draw(self)`](../utils_new/smart_view_new.py#L1976)
  - [`SmartGraphicsViewNew._add_scribble(self, x, y, temp=False)`](../utils_new/smart_view_new.py#L1985)
  - [`SmartGraphicsViewNew.normalize_angle_rad(self, angle)`](../utils_new/smart_view_new.py#L1991)
  - [`SmartGraphicsViewNew._color_from_bgr(bgr)`](../utils_new/smart_view_new.py#L2006)
  - [`SmartGraphicsViewNew._annotation_color(self, cls_id, selected=False)`](../utils_new/smart_view_new.py#L2009)
  - [`SmartGraphicsViewNew._annotation_label(self, cls_id)`](../utils_new/smart_view_new.py#L2019)
  - [`SmartGraphicsViewNew._cosmetic_pen(color, width_px=1.0, style=Qt.SolidLine)`](../utils_new/smart_view_new.py#L2028)
  - [`SmartGraphicsViewNew.drawForeground(self, painter: QPainter, rect)`](../utils_new/smart_view_new.py#L2035)
  - [`SmartGraphicsViewNew._draw_foreground_impl(self, painter: QPainter, rect)`](../utils_new/smart_view_new.py#L2041)
  - [`SmartGraphicsViewNew._draw_annotations_layer(self, painter, visible_rect)`](../utils_new/smart_view_new.py#L2052)
  - [`SmartGraphicsViewNew._label_is_visible(self, bounds, selected)`](../utils_new/smart_view_new.py#L2137)
  - [`SmartGraphicsViewNew._paint_screen_label(self, painter, label, bounds, color, background_color=None)`](../utils_new/smart_view_new.py#L2149)
  - [`SmartGraphicsViewNew._draw_preview_layer(self, painter)`](../utils_new/smart_view_new.py#L2206)
  - [`SmartGraphicsViewNew._draw_edit_handles(self, painter)`](../utils_new/smart_view_new.py#L2283)
  - [`SmartGraphicsViewNew._draw_crosshair(self, painter)`](../utils_new/smart_view_new.py#L2348)
- [`numpy_to_qpixmap(img_bgr, max_side=8192)`](../utils_new/smart_view_new.py#L2362) — Display-only conversion. Geometry stays in source-image coordinates.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).

<a id="utils-new-threshold-processing"></a>

## utils_new.threshold_processing

Исходник: [utils_new/threshold_processing.py](../utils_new/threshold_processing.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`simple_threshold(img, thresh)`](../utils_new/threshold_processing.py#L28)
- [`otsu_threshold(img)`](../utils_new/threshold_processing.py#L32)
- [`triangle_threshold(img)`](../utils_new/threshold_processing.py#L36)
- [`niblack_threshold(img, window_size=15, k=-0.2)`](../utils_new/threshold_processing.py#L40)
- [`sauvola_threshold(img, window_size=25, k=0.2, R=128)`](../utils_new/threshold_processing.py#L48)
- [`isodata_threshold(img, initial_threshold=128)`](../utils_new/threshold_processing.py#L56)
- [`background_symmetry_threshold(img, excess_threshold=0.2)`](../utils_new/threshold_processing.py#L73)
- [`row_adaptive_threshold(img, window_size=50, k=0.5)`](../utils_new/threshold_processing.py#L95)
- [`normalize_method_name(method_name)`](../utils_new/threshold_processing.py#L110) — Приводит имя метода к каноническому виду для UI.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).
- [`apply_threshold_method(gray_img, method_name, params)`](../utils_new/threshold_processing.py#L135) — Применяет метод бинаризации по имени и параметрам.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).

<a id="utils-ops-deep-learning-ops"></a>

## utils_ops.deep_learning_ops

Исходник: [utils_ops/deep_learning_ops.py](../utils_ops/deep_learning_ops.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** `json`, `segment_anything.SamPredictor`, `segment_anything.sam_model_registry`, `segmentation_models_pytorch`, `ultralytics.YOLO`.

**Объявления:**

- [`_log(msg)`](../utils_ops/deep_learning_ops.py#L5)
- Класс [`BaseSegmentor`](../utils_ops/deep_learning_ops.py#L28)
  - [`BaseSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L30)
  - [`BaseSegmentor.load(self, **kwargs)`](../utils_ops/deep_learning_ops.py#L36)
  - [`BaseSegmentor.predict(self, image, **kwargs)`](../utils_ops/deep_learning_ops.py#L39)
- Класс [`TorchSegmentor`](../utils_ops/deep_learning_ops.py#L43) (`BaseSegmentor`)
  - [`TorchSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L45)
  - [`TorchSegmentor._get_device(self)`](../utils_ops/deep_learning_ops.py#L50)
  - [`TorchSegmentor._resolve_resize_mode(self)`](../utils_ops/deep_learning_ops.py#L68)
  - [`TorchSegmentor._preprocess(self, image, input_size)`](../utils_ops/deep_learning_ops.py#L103)
  - [`TorchSegmentor._postprocess_mask(self, mask_np, orig_size, pad_info, resize_mode)`](../utils_ops/deep_learning_ops.py#L172)
- Класс [`UNetSegmentor`](../utils_ops/deep_learning_ops.py#L193) (`TorchSegmentor`)
  - [`UNetSegmentor.load(self, weights_path=None, encoder=None, input_size=None)`](../utils_ops/deep_learning_ops.py#L211)
  - [`UNetSegmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L353)
- Класс [`DeepLabV3Segmentor`](../utils_ops/deep_learning_ops.py#L409) (`TorchSegmentor`)
  - [`DeepLabV3Segmentor.load(self, weights_path=None, backbone=None, output_stride=None)`](../utils_ops/deep_learning_ops.py#L432)
  - [`DeepLabV3Segmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L601)
- Класс [`YOLOSegmentor`](../utils_ops/deep_learning_ops.py#L691) (`BaseSegmentor`)
  - [`YOLOSegmentor.load(self, model_path=None)`](../utils_ops/deep_learning_ops.py#L705)
  - [`YOLOSegmentor._otsu_inside_roi(image, x1, y1, x2, y2)`](../utils_ops/deep_learning_ops.py#L726)
  - [`YOLOSegmentor.predict(self, image, conf=0.25, iou=0.45, imgsz=640, save=False)`](../utils_ops/deep_learning_ops.py#L741)
- Класс [`SAMSegmentor`](../utils_ops/deep_learning_ops.py#L872) (`BaseSegmentor`)
  - [`SAMSegmentor.load(self, model_type='vit_h', model_path=None)`](../utils_ops/deep_learning_ops.py#L874)
  - [`SAMSegmentor.predict(self, image, prompt_type='everything')`](../utils_ops/deep_learning_ops.py#L887)
- Класс [`ONNXSegmentor`](../utils_ops/deep_learning_ops.py#L894) (`BaseSegmentor`)
  - [`ONNXSegmentor.load(self, model_path)`](../utils_ops/deep_learning_ops.py#L896)
  - [`ONNXSegmentor.predict(self, image)`](../utils_ops/deep_learning_ops.py#L905)
- [`create_segmentor(model_type, **kwargs)`](../utils_ops/deep_learning_ops.py#L937)
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-ops-gradient-ops"></a>

## utils_ops.gradient_ops

Исходник: [utils_ops/gradient_ops.py](../utils_ops/gradient_ops.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`apply_gradient_method(gray, method, params)`](../utils_ops/gradient_ops.py#L3)
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-ops-interactive-ops"></a>

## utils_ops.interactive_ops

Исходник: [utils_ops/interactive_ops.py](../utils_ops/interactive_ops.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`

**Внешние/stdlib импорты:** `networkx`, `skimage.segmentation.active_contour`, `skimage.segmentation.find_boundaries`, `skimage.segmentation.random_walker`, `skimage.segmentation.slic`, `sklearn.mixture.GaussianMixture`, `warnings`.

**Объявления:**

- [`grabcut_segmentation(img, rect=None, mask=None, fg_scribbles=None, bg_scribbles=None, iterations=5, mode='Both')`](../utils_ops/interactive_ops.py#L15) — GrabCut segmentation with mode selection.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`watershed_segmentation(gray, distance_threshold=50, min_area=20)`](../utils_ops/interactive_ops.py#L65)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`random_walker_segmentation(img, markers, beta=130, mode='cg_j')`](../utils_ops/interactive_ops.py#L90) — Random Walker segmentation.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`active_contour_segmentation(gray, rect, alpha=0.01, beta=0.1, gamma=0.01, max_iter=250, convergence=0.001, smooth=True, init_type='Ellipse')`](../utils_ops/interactive_ops.py#L113) — Active Contours (Snakes) segmentation.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`supercut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, lambda_val=0.5, sigma_color=10.0)`](../utils_ops/interactive_ops.py#L197) — SuperCut using graph cut (max-flow/min-cut) on superpixels.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`onecut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, spatial_weight=1.0, data_weight=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L302) — OneCut segmentation using superpixels and graph cut.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`lazy_snapping(img, fg_scribbles, bg_scribbles, superpixel_size=50, compactness=10, sigma=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L449) — Lazy Snapping using graph cut on superpixels.
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-ops-segmentation-ops"></a>

## utils_ops.segmentation_ops

Исходник: [utils_ops/segmentation_ops.py](../utils_ops/segmentation_ops.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal), [tabs.layout_dataset](#tabs-layout-dataset).

**Прямые локальные импорты:**
- [import_libs_external](#import-libs-external): `*`
- [utils.settings](#utils-settings): `settings`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`get_display_params(img_shape)`](../utils_ops/segmentation_ops.py#L7)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`draw_label(img_color, label, x, y, box_w, box_h, font_scale, font_thickness, thickness, color=None)`](../utils_ops/segmentation_ops.py#L21)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`draw_label_rotated(img_color, label, center, font_scale, font_thickness, color=None)`](../utils_ops/segmentation_ops.py#L44)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`get_class_color(cls_id)`](../utils_ops/segmentation_ops.py#L81) — Цвет (BGR) для класса. Для id >= 10 цвета циклически повторяются.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`draw_class_label(img, text, x, y, box_w, box_h, font_scale, font_thickness, bg_color, text_color=(255, 255, 255), padding=2, margin=5)`](../utils_ops/segmentation_ops.py#L92) — Рисует текстовую подпись поверх изображения на цветной плашке.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`simple_threshold(img, thresh)`](../utils_ops/segmentation_ops.py#L149)
- [`otsu_threshold(img)`](../utils_ops/segmentation_ops.py#L154)
- [`triangle_threshold(img)`](../utils_ops/segmentation_ops.py#L159)
- [`niblack_threshold(img, window_size=15, k=-0.2)`](../utils_ops/segmentation_ops.py#L164)
- [`sauvola_threshold(img, window_size=25, k=0.2, R=128)`](../utils_ops/segmentation_ops.py#L173)
- [`isodata_threshold(img, initial_threshold=128)`](../utils_ops/segmentation_ops.py#L182)
- [`background_symmetry_threshold(img, excess_threshold=0.2)`](../utils_ops/segmentation_ops.py#L200)
- [`row_adaptive_threshold(img, window_size=50, k=0.5)`](../utils_ops/segmentation_ops.py#L223)
- [`segment_contours(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L240) — Возвращает (img_color, objects) где objects = список ('segment', 0, points) в нормализованных координатах.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`segment_projections(img, binary, draw=True)`](../utils_ops/segmentation_ops.py#L280) — Возвращает (img_color, objects) где objects = список ('detect', 0, cx, cy, w, h)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`segment_min_area_rect(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L318) — Возвращает (img_color, objects) где objects = список ('obb', 0, points) где points = 8 нормализованных координат углов.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`segment_polygons(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L358) — Алиас для segment_contours (уже возвращает segment).
- [`draw_yolo_annotations(img, annotations, color=(0, 255, 0))`](../utils_ops/segmentation_ops.py#L364) — Рисует аннотации любого типа (detect, obb, segment) на изображении.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_ops/segmentation_ops.py#L415) — Форматирует аннотацию для отображения в QListWidget.
- [`update_annotation_list(list_widget: QListWidget, annotations, img_w: int, img_h: int)`](../utils_ops/segmentation_ops.py#L445) — Очищает и заполняет QListWidget строками аннотаций. Каждому элементу в UserRole записывается индекс.
  Явно импортируют: [import_libs_internal](#import-libs-internal), [tabs.layout_dataset](#tabs-layout-dataset).
- [`delete_annotation_by_index(current_annotations, idx, all_annotations=None, current_index=None)`](../utils_ops/segmentation_ops.py#L455) — Удаляет аннотацию с индексом idx. Возвращает (new_annotations, new_selected_index).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`draw_selected_objects(img, objects, selected_indices, draw_mode, use_hull, color_rect, color_label, thickness, font_scale, font_thickness)`](../utils_ops/segmentation_ops.py#L468) — Рисует выбранные объекты на изображении. Поддерживает ('detect',...), ('obb',...), ('segment',...).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_ops/segmentation_ops.py#L517) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`normalize_method_name(method_name)`](../utils_ops/segmentation_ops.py#L538) — Приводит имя метода к каноническому виду для UI.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`apply_threshold_method(gray_img, method_name, params)`](../utils_ops/segmentation_ops.py#L564) — Применяет метод бинаризации по имени и параметрам.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`get_morph_kernel(size, shape_name)`](../utils_ops/segmentation_ops.py#L625)
- [`apply_morphology(binary, close_factor, open_factor, kernel_shape, img_shape=None)`](../utils_ops/segmentation_ops.py#L635)
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-ops-traditional-ml-ops"></a>

## utils_ops.traditional_ml_ops

Исходник: [utils_ops/traditional_ml_ops.py](../utils_ops/traditional_ml_ops.py#L1).

**Прямые потребители:** [import_libs_internal](#import-libs-internal).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `joblib`, `numpy`, `skimage.feature`, `skimage.segmentation`, `sklearn.cluster.KMeans`, `sklearn.cluster.MeanShift`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.preprocessing.StandardScaler`, `sklearn.svm.SVC`, `sklearn.tree.DecisionTreeClassifier`, `warnings`, `xgboost`.

**Объявления:**

- [`extract_pixel_features(image, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L20)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`extract_superpixel_features(image, segments, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L47)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_kmeans(X, n_clusters=3)`](../utils_ops/traditional_ml_ops.py#L85)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_meanshift(X)`](../utils_ops/traditional_ml_ops.py#L90)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_svm(X, y, kernel='rbf', C=10)`](../utils_ops/traditional_ml_ops.py#L95)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_random_forest(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L100)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_xgboost(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L105)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`train_decision_tree(X, y)`](../utils_ops/traditional_ml_ops.py#L112)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`predict_pixelwise(model, X, original_shape)`](../utils_ops/traditional_ml_ops.py#L117)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`predict_superpixel(model, segments, X_superpixel)`](../utils_ops/traditional_ml_ops.py#L121)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`create_superpixels(image, superpixel_size=50)`](../utils_ops/traditional_ml_ops.py#L128)
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`compute_elbow_kmeans(X, max_k=10)`](../utils_ops/traditional_ml_ops.py#L134) — Вычисляет инерцию (сумму квадратов расстояний) для K от 1 до max_k.
  Явно импортируют: [import_libs_internal](#import-libs-internal).
- [`find_optimal_k_elbow(inertias)`](../utils_ops/traditional_ml_ops.py#L146) — Находит оптимальное K как точку максимального изгиба (метод локтя).
  Явно импортируют: [import_libs_internal](#import-libs-internal).

<a id="utils-training-test-train-worker"></a>

## utils_training.test_train_worker

Исходник: [utils_training/test_train_worker.py](../utils_training/test_train_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `itertools.product`, `json`, `os`, `subprocess`, `sys`, `time`.

**Объявления:**

- [`generate_combinations(params_dict)`](../utils_training/test_train_worker.py#L192)
- [`save_metrics(model_name, params, metrics)`](../utils_training/test_train_worker.py#L198)
- [`run_worker(worker_name, params)`](../utils_training/test_train_worker.py#L206)
- [`main()`](../utils_training/test_train_worker.py#L241)

<a id="utils-training-train-DeepLabV3-worker"></a>

## utils_training.train_DeepLabV3_worker

Исходник: [utils_training/train_DeepLabV3_worker.py](../utils_training/train_DeepLabV3_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `copy`, `csv`, `datetime.datetime`, `glob.glob`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `onnx`, `onnxruntime`, `os`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `traceback`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_DeepLabV3_worker.py#L192)
- [`signal_handler(signum, frame)`](../utils_training/train_DeepLabV3_worker.py#L196)
- [`find_image_file(jpeg_dir, basename)`](../utils_training/train_DeepLabV3_worker.py#L212)
- [`read_voc_pairs(txt_path, jpeg_dir, seg_dir)`](../utils_training/train_DeepLabV3_worker.py#L219)
- [`read_voc_file_list(txt_path, root_dir)`](../utils_training/train_DeepLabV3_worker.py#L244)
- [`_resolve_path(root, p)`](../utils_training/train_DeepLabV3_worker.py#L249)
- [`create_loaders_from_yaml(yaml_path, default_input_size, default_num_classes, augmentations, ignore_index, batch_size, num_workers, normalize, resize_mode)`](../utils_training/train_DeepLabV3_worker.py#L260)
- [`create_voc_loaders(root, input_size, num_classes, augmentations, ignore_index, batch_size, num_workers, normalize=False, resize_mode='hard')`](../utils_training/train_DeepLabV3_worker.py#L367)
- Класс [`VOCSegmentationPairs`](../utils_training/train_DeepLabV3_worker.py#L391) (`Dataset`)
  - [`VOCSegmentationPairs.__init__(self, image_mask_pairs, input_size, num_classes, ignore_index, augmentations, normalize=False, resize_mode='hard')`](../utils_training/train_DeepLabV3_worker.py#L392)
  - [`VOCSegmentationPairs.__len__(self)`](../utils_training/train_DeepLabV3_worker.py#L402)
  - [`VOCSegmentationPairs.__getitem__(self, idx)`](../utils_training/train_DeepLabV3_worker.py#L405)
- [`get_loss_function(loss_name, num_classes, ignore_index=None, class_weights=None, loss_weights=None, device='cpu')`](../utils_training/train_DeepLabV3_worker.py#L470) — Патч 1: weights_tensor создаётся на нужном устройстве (device).
- [`get_optimizer(model, opt_name, lr, momentum=0.9, weight_decay=0.0001)`](../utils_training/train_DeepLabV3_worker.py#L520)
- [`get_layerwise_optimizer(model, opt_name, base_lr, momentum=0.9, weight_decay=0.0001, decay_factor=0.1)`](../utils_training/train_DeepLabV3_worker.py#L533) — Layer-wise LR. Устойчив к тому, что get_stages() может возвращать
- [`get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-06)`](../utils_training/train_DeepLabV3_worker.py#L592)
- Класс [`PolyLRSchedulerIter`](../utils_training/train_DeepLabV3_worker.py#L605)
  - [`PolyLRSchedulerIter.__init__(self, optimizer, max_iter, power=0.9, min_lr=0.0)`](../utils_training/train_DeepLabV3_worker.py#L606)
  - [`PolyLRSchedulerIter.step(self)`](../utils_training/train_DeepLabV3_worker.py#L614)
  - [`PolyLRSchedulerIter.get_last_lr(self)`](../utils_training/train_DeepLabV3_worker.py#L620)
- Класс [`ModelEMA`](../utils_training/train_DeepLabV3_worker.py#L624)
  - [`ModelEMA.__init__(self, model, decay=0.999, device=None)`](../utils_training/train_DeepLabV3_worker.py#L625)
  - [`ModelEMA.update(self, model)`](../utils_training/train_DeepLabV3_worker.py#L636)
  - [`ModelEMA.state_dict(self)`](../utils_training/train_DeepLabV3_worker.py#L647)
- [`set_bn_eval(module)`](../utils_training/train_DeepLabV3_worker.py#L651)
- [`_pad_to_multiple(x, multiple=16)`](../utils_training/train_DeepLabV3_worker.py#L657)
- [`_crop_to_hw(x, hw)`](../utils_training/train_DeepLabV3_worker.py#L667)
- [`_forward_safe(model, x, use_amp, device, output_stride=None)`](../utils_training/train_DeepLabV3_worker.py#L672) — Патч 3: убран nan_to_num(nan=0.0) — он маскировал проблему,
- [`predict_batch_multiscale(model, images, device, scales, use_amp)`](../utils_training/train_DeepLabV3_worker.py#L702)
- [`compute_metrics(pred_logits, true_masks, num_classes, ignore_index=None, output_activation='argmax', threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L727)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_DeepLabV3_worker.py#L789)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L818)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_DeepLabV3_worker.py#L830)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_DeepLabV3_worker.py#L837)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L859)
- [`save_pr_curve(preds, masks, save_dir, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L867)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes, ignore_index=None, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L886)
- [`save_train_batch_examples(loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_DeepLabV3_worker.py#L924)
- [`save_val_examples(loader, model, device, save_dir, num_samples=8, num_classes=1, output_activation='argmax', normalize=False, use_amp=False, multi_scale=False, scales=None)`](../utils_training/train_DeepLabV3_worker.py#L956)
- [`main()`](../utils_training/train_DeepLabV3_worker.py#L1016)

<a id="utils-training-train-unet-worker"></a>

## utils_training.train_unet_worker

Исходник: [utils_training/train_unet_worker.py](../utils_training/train_unet_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `onnx`, `onnxruntime`, `os`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `traceback`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker.py#L135)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker.py#L140)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker.py#L162) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker.py#L163)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker.py#L205)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker.py#L208)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker.py#L264)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker.py#L339)
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker.py#L376)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker.py#L391)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L404)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker.py#L479)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker.py#L522)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker.py#L534)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker.py#L542)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker.py#L577)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker.py#L586)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L645)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_unet_worker.py#L706)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1, normalize=False)`](../utils_training/train_unet_worker.py#L749)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker.py#L820)
- [`main()`](../utils_training/train_unet_worker.py#L870)

<a id="utils-training-train-unet-worker-рабочий"></a>

## utils_training.train_unet_worker_рабочий

Исходник: [utils_training/train_unet_worker_рабочий.py](../utils_training/train_unet_worker_рабочий.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker_рабочий.py#L130)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker_рабочий.py#L134)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker_рабочий.py#L155) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker_рабочий.py#L156)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker_рабочий.py#L198)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker_рабочий.py#L201)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker_рабочий.py#L266)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L344)
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker_рабочий.py#L381)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker_рабочий.py#L396)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L409)
- [`auto_batch_size(device, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker_рабочий.py#L484)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L508)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker_рабочий.py#L519)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker_рабочий.py#L527)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L555)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L563)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L606)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L662)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L693)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker_рабочий.py#L751) — Проверяет первые max_check масок: собирает уникальные значения пикселей,
- [`main()`](../utils_training/train_unet_worker_рабочий.py#L806)

<a id="utils-training-train-yolo-worker"></a>

## utils_training.train_yolo_worker

Исходник: [utils_training/train_yolo_worker.py](../utils_training/train_yolo_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `datetime.datetime`, `json`, `onnx`, `os`, `shutil`, `signal`, `sys`, `traceback`, `ultralytics`, `ultralytics.YOLO`.

**Объявления:**

- [`log(msg)`](../utils_training/train_yolo_worker.py#L12) — Вывод сообщения с немедленным сбросом буфера.
- [`is_standard_yolo_model(model_path)`](../utils_training/train_yolo_worker.py#L16) — Определяет, является ли имя модели стандартным (без пути).
- [`signal_handler(signum, frame)`](../utils_training/train_yolo_worker.py#L27)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_yolo_worker.py#L34) — Сохраняет JSON-файл с метаданными рядом с .pt файлом.
- [`main()`](../utils_training/train_yolo_worker.py#L41)

<a id="utils-training-yolo2voc"></a>

## utils_training.yolo2voc

Исходник: [utils_training/yolo2voc.py](../utils_training/yolo2voc.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `os`, `pascal_voc_writer.Writer`, `shutil`, `sys`, `yaml`.

**Объявления:**

- [`create_binary_palette()`](../utils_training/yolo2voc.py#L45) — Создаёт палитру для 8-битного PNG: индекс 0 -> чёрный, индекс 1 -> белый.
- [`convert_mask_to_voc_format(src_mask_path, dst_mask_path)`](../utils_training/yolo2voc.py#L50) — Конвертирует бинарную маску (0 и 255) в P-режим с палитрой.
- [`copy_image(src_img_path, dst_img_path)`](../utils_training/yolo2voc.py#L63) — Копирует изображение без изменений.
- [`convert_yolo_txt_to_voc_xml(txt_path, img_path, class_names, output_xml_path)`](../utils_training/yolo2voc.py#L67) — Конвертирует один YOLO .txt файл (детекция) в VOC XML.
- [`process_set(set_name, base_dir, images_rel, masks_rel, labels_rel, class_names, voc_root)`](../utils_training/yolo2voc.py#L104) — Обрабатывает один набор данных (train/val/test).
- [`build_voc_from_yolo(yaml_file, output_root)`](../utils_training/yolo2voc.py#L166) — Основная функция конвертации, использующая YAML-файл датасета YOLO.
- [`interactive_mode()`](../utils_training/yolo2voc.py#L238) — Интерактивный режим с запросом параметров.
- [`main()`](../utils_training/yolo2voc.py#L294)
