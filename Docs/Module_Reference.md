# Справочник модулей, импортов и API

[Навигация](README.md) · [Карта архитектуры](Architecture.md) · [Дублирование](Duplication.md)

Сгенерировано `python Docs/generate_reference.py` из текущих исходников. Каталог исключает саму документацию и скрытые каталоги.

Проанализировано Python-модулей: **144**. Импорты учитываются на всех уровнях AST, включая функции и try/except; наличие записи не означает, что ветка выполняется при запуске. Звёздочные импорты не разворачиваются в список фактических вызовов. QProcess, строковые динамические импорты и файловые зависимости описаны отдельно в карте.

Для всех модулей перечислены верхнеуровневые функции и классы. Для всех каталогов утилит дополнительно перечислены сигнатуры методов классов. Вложенные локальные функции не являются отдельным API и в список не включены. Приватные имена сохранены для поиска реализации.

## Содержание

- [cluster.canonicalize_crops](#cluster-canonicalize-crops)
- [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2)
- [cluster.cluster_embeddings](#cluster-cluster-embeddings)
- [cluster.extract_embeddings](#cluster-extract-embeddings)
- [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path)
- [cluster.test_new_image](#cluster-test-new-image)
- [dependency_health](#dependency-health)
- [generate](#generate)
- [import_libs_external](#import-libs-external)
- [import_libs_internal](#import-libs-internal)
- [import_libs_methods_ui](#import-libs-methods-ui)
- [install](#install)
- [main](#main)
- [main_ui](#main-ui)
- [path_setup](#path-setup)
- [rename_data](#rename-data)
- [runtime_policy](#runtime-policy)
- [runtime_torch](#runtime-torch)
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
- [tests.test_dependency_installation](#tests-test-dependency-installation)
- [tests.test_import_architecture](#tests-test-import-architecture)
- [tests.test_label_save_merge](#tests-test-label-save-merge)
- [tests.test_pixel_morphology](#tests-test-pixel-morphology)
- [tests.test_processing_smoke](#tests-test-processing-smoke)
- [tests.test_runtime_imports](#tests-test-runtime-imports)
- [tests.test_segmentation_shared](#tests-test-segmentation-shared)
- [tests.test_torch_runtime](#tests-test-torch-runtime)
- [ui.DL_predict_settings](#ui-DL-predict-settings)
- [ui.DL_train_settings_base](#ui-DL-train-settings-base)
- [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model)
- [ui.annotation_review_view](#ui-annotation-review-view)
- [ui.application](#ui-application)
- [ui.control_buttons](#ui-control-buttons)
- [ui.image_conversion](#ui-image-conversion)
- [ui.image_navigation](#ui-image-navigation)
- [ui.image_view](#ui-image-view)
- [ui.interactive_params_ui](#ui-interactive-params-ui)
- [ui.log_widget](#ui-log-widget)
- [ui.morphology_ui](#ui-morphology-ui)
- [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new)
- [ui.segmentation.actions](#ui-segmentation-actions)
- [ui.segmentation.export](#ui-segmentation-export)
- [ui.segmentation.histogram](#ui-segmentation-histogram)
- [ui.segmentation.image_input](#ui-segmentation-image-input)
- [ui.segmentation.interactive_view](#ui-segmentation-interactive-view)
- [ui.segmentation.navigation](#ui-segmentation-navigation)
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings)
- [ui.segmentation.results](#ui-segmentation-results)
- [ui.settings_service](#ui-settings-service)
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
- [utils.analysis](#utils-analysis)
- [utils.dataset_analysis_logic](#utils-dataset-analysis-logic)
- [utils.dataset_audit_logic](#utils-dataset-audit-logic)
- [utils.ensemble_utils](#utils-ensemble-utils)
- [utils.image_io](#utils-image-io)
- [utils.metrics_core](#utils-metrics-core)
- [utils.prepare](#utils-prepare)
- [utils.smart_view](#utils-smart-view)
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
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new)
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new)
- [utils_new.dataset_view_new](#utils-new-dataset-view-new)
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new)
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new)
- [utils_new.json_store](#utils-new-json-store)
- [utils_new.morphology](#utils-new-morphology)
- [utils_new.presets_store](#utils-new-presets-store)
- [utils_new.segmentation.export](#utils-new-segmentation-export)
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session)
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing)
- [utils_new.segmentation.presets](#utils-new-segmentation-presets)
- [utils_new.settings_store](#utils-new-settings-store)
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
| `PIL` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [generate](#generate), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.image_io](#utils-image-io), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `PyQt5` | [generate](#generate), [main](#main), [main_ui](#main-ui), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.DL_predict_settings](#ui-DL-predict-settings), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui.annotation_review_view](#ui-annotation-review-view), [ui.application](#ui-application), [ui.control_buttons](#ui-control-buttons), [ui.image_conversion](#ui-image-conversion), [ui.image_navigation](#ui-image-navigation), [ui.image_view](#ui-image-view), [ui.interactive_params_ui](#ui-interactive-params-ui), [ui.log_widget](#ui-log-widget), [ui.morphology_ui](#ui-morphology-ui), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.actions](#ui-segmentation-actions), [ui.segmentation.export](#ui-segmentation-export), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.interactive_view](#ui-segmentation-interactive-view), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.segmentation.results](#ui-segmentation-results), [ui.settings_service](#ui-settings-service), [ui.settings_ui](#ui-settings-ui), [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.model_research_ui](#ui-tabs-model-research-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.image_io](#utils-image-io), [utils.smart_view](#utils-smart-view), [utils_auto_research.gradient_auto_research](#utils-auto-research-gradient-auto-research), [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog), [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research), [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops) |
| `cv2` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.image_conversion](#ui-image-conversion), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.ensemble_utils](#utils-ensemble-utils), [utils.image_io](#utils-image-io), [utils.smart_view](#utils-smart-view), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_ops.gradient_ops](#utils-ops-gradient-ops), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `hdbscan` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `imagehash` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `joblib` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `matplotlib` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [tabs.dataset_analysis](#tabs-dataset-analysis), [ui.segmentation.histogram](#ui-segmentation-histogram), [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.prepare](#utils-prepare), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `networkx` | [utils_ops.interactive_ops](#utils-ops-interactive-ops) |
| `numpy` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.image_conversion](#ui-image-conversion), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [utils.analysis](#utils-analysis), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.ensemble_utils](#utils-ensemble-utils), [utils.image_io](#utils-image-io), [utils.metrics_core](#utils-metrics-core), [utils.prepare](#utils-prepare), [utils.smart_view](#utils-smart-view), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.model](#utils-new-annotations-model), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_ops.gradient_ops](#utils-ops-gradient-ops), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `onnx` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `onnxruntime` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker) |
| `open_clip` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `pandas` | [tabs.model_research](#tabs-model-research), [ui_tabs.model_research_ui](#ui-tabs-model-research-ui), [utils.analysis](#utils-analysis), [utils.metrics_core](#utils-metrics-core), [utils.prepare](#utils-prepare) |
| `pascal_voc_writer` | [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `psutil` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `sahi` | [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi) |
| `seaborn` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `segment_anything` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops) |
| `segmentation_models_pytorch` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `shapely` | [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.geometry](#utils-new-annotations-geometry) |
| `skimage` | [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `sklearn` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [utils.analysis](#utils-analysis), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils.prepare](#utils-prepare), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `tifffile` | [utils.image_io](#utils-image-io) |
| `torch` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `torchvision` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `transformers` | [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `ultralytics` | [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `umap` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils.dataset_audit_logic](#utils-dataset-audit-logic) |
| `xgboost` | [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `yaml` | [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tests.test_processing_smoke](#tests-test-processing-smoke), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [utils.image_io](#utils-image-io), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |

<a id="cluster-canonicalize-crops"></a>

## cluster.canonicalize_crops

Исходник: [cluster/canonicalize_crops.py](../cluster/canonicalize_crops.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `csv`, `cv2`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sys`, `warnings`.

**Объявления:**

- [`read_image_bgr(path: str)`](../cluster/canonicalize_crops.py#L74)
- [`imwrite_safe(path: str, img_bgr)`](../cluster/canonicalize_crops.py#L97)
- [`collect_images(folder: str, recursive: bool=False)`](../cluster/canonicalize_crops.py#L109)
- [`prepare_compare(img_bgr, size: int)`](../cluster/canonicalize_crops.py#L124) — Grayscale → ресайз с сохранением пропорций + паддинг → zero-mean/unit-std.
- [`apply_transform(img, angle_deg: int, flip: bool)`](../cluster/canonicalize_crops.py#L143)
- [`pearson(a, b)`](../cluster/canonicalize_crops.py#L154)
- [`find_best_transform(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/canonicalize_crops.py#L158)
- [`make_debug_montage(records, out_path, cols=8, max_items=60)`](../cluster/canonicalize_crops.py#L173) — records: list of (img_bgr, angle, flip, score).
- [`parse_args()`](../cluster/canonicalize_crops.py#L203)
- [`main()`](../cluster/canonicalize_crops.py#L226)

<a id="cluster-canonicalize-crops-method2"></a>

## cluster.canonicalize_crops_method2

Исходник: [cluster/canonicalize_crops_method2.py](../cluster/canonicalize_crops_method2.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `csv`, `cv2`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `sys`, `warnings`.

**Объявления:**

- [`read_image_bgr(path: str)`](../cluster/canonicalize_crops_method2.py#L91)
- [`imwrite_safe(path: str, img_bgr)`](../cluster/canonicalize_crops_method2.py#L114)
- [`collect_images(folder: str, recursive: bool=False)`](../cluster/canonicalize_crops_method2.py#L126)
- [`prepare_compare(img_bgr, size: int=224)`](../cluster/canonicalize_crops_method2.py#L144) — Grayscale → ресайз → паддинг → zero-mean/unit-std.
- [`apply_transform(img, angle_deg: int, flip: bool)`](../cluster/canonicalize_crops_method2.py#L163)
- [`find_best_by_correlation(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/canonicalize_crops_method2.py#L174) — Перебор дискретных углов и зеркала, оценка — Pearson.
- [`get_detector(method: str)`](../cluster/canonicalize_crops_method2.py#L191) — Возвращает (detector, norm_type).
- [`extract_features(img_bgr, detector)`](../cluster/canonicalize_crops_method2.py#L203)
- [`match_good(des1, des2, norm_type, ratio=0.75)`](../cluster/canonicalize_crops_method2.py#L209) — Lowe ratio test, возвращает список good matches.
- [`estimate_rotation(img_bgr, ref_data, apply_flip, ratio, min_good, ransac_thresh)`](../cluster/canonicalize_crops_method2.py#L229) — Пытается оценить угол поворота кропа относительно референса
- [`rotate_image(img_bgr, angle_deg, flip)`](../cluster/canonicalize_crops_method2.py#L290) — Применяет зеркало (если надо) и поворот на произвольный угол.
- [`save_fig_via_cv2(fig, out_path, dpi=110)`](../cluster/canonicalize_crops_method2.py#L322) — fig → numpy → cv2.imencode → файл. Обходит savefig/PIL на Windows.
- [`make_debug_montage(records, out_path, cols=8, max_items=60)`](../cluster/canonicalize_crops_method2.py#L336) — records: list of (img_bgr, info_str).
- [`parse_args()`](../cluster/canonicalize_crops_method2.py#L365)
- [`main()`](../cluster/canonicalize_crops_method2.py#L392)

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

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `cv2`, `numpy`, `os`, `sys`, `torch`, `torch.nn.functional`, `torchvision.transforms`, `warnings`.

**Объявления:**

- [`read_image_rgb(path: str)`](../cluster/extract_embeddings.py#L58)
- [`collect_images(folder: str, recursive: bool=True)`](../cluster/extract_embeddings.py#L83)
- [`resolve_device(device_str: str)`](../cluster/extract_embeddings.py#L98)
- [`embedding_path_for(img_path: str)`](../cluster/extract_embeddings.py#L109)
- [`apply_transform(img, angle_deg, flip)`](../cluster/extract_embeddings.py#L113)
- [`load_dinov2(model_name: str, device: str)`](../cluster/extract_embeddings.py#L127)
- [`extract_batch(model, batch_tensor, device)`](../cluster/extract_embeddings.py#L144)
- [`parse_args()`](../cluster/extract_embeddings.py#L154)
- [`main()`](../cluster/extract_embeddings.py#L177)

<a id="cluster-save-yolo-object-to-path"></a>

## cluster.save_yolo_object_to_path

Исходник: [cluster/save_yolo_object_to_path.py](../cluster/save_yolo_object_to_path.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `argparse`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- [`read_image(path: str)`](../cluster/save_yolo_object_to_path.py#L58) — Чтение изображения с поддержкой кириллицы в пути + нормализация к BGR uint8.
- [`imwrite_safe(path: str, img)`](../cluster/save_yolo_object_to_path.py#L87) — Запись с поддержкой кириллицы в пути.
- [`unique_path(path: str)`](../cluster/save_yolo_object_to_path.py#L100) — Если файл уже существует — добавляет _1, _2, ...
- [`collect_images(folder: str)`](../cluster/save_yolo_object_to_path.py#L113) — Возвращает список путей ко всем изображениям (рекурсивно).
- [`resolve_device(device_str: str)`](../cluster/save_yolo_object_to_path.py#L124)
- [`parse_args()`](../cluster/save_yolo_object_to_path.py#L146)
- [`main()`](../cluster/save_yolo_object_to_path.py#L175)

<a id="cluster-test-new-image"></a>

## cluster.test_new_image

Исходник: [cluster/test_new_image.py](../cluster/test_new_image.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `collections.Counter`, `csv`, `cv2`, `dataclasses.dataclass`, `dataclasses.field`, `joblib`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `torch.nn.functional`, `torchvision.transforms`, `ultralytics.YOLO`, `warnings`.

**Объявления:**

- [`read_image_bgr(path)`](../cluster/test_new_image.py#L115)
- [`imwrite_safe(path, img_bgr)`](../cluster/test_new_image.py#L138)
- [`collect_images(path, recursive=False)`](../cluster/test_new_image.py#L150)
- [`resolve_device(device_str)`](../cluster/test_new_image.py#L167)
- [`prepare_compare(img_bgr, size=224)`](../cluster/test_new_image.py#L181)
- [`apply_transform(img, angle_deg, flip)`](../cluster/test_new_image.py#L197)
- [`find_best_alignment(img_bgr, ref_cmp, angles, apply_flip, compare_size)`](../cluster/test_new_image.py#L208)
- [`load_dinov2(model_name, device)`](../cluster/test_new_image.py#L224)
- [`embed_images(model, transform, images_bgr, device, batch_size=16)`](../cluster/test_new_image.py#L241) — images_bgr: list[np.ndarray BGR]. Возвращает np.ndarray [N, D] L2-норм.
- [`load_cluster_state(model_dir)`](../cluster/test_new_image.py#L264)
- [`predict_cluster(emb, centroids, reducer, cluster_ids, min_sim=0.0)`](../cluster/test_new_image.py#L276)
- Класс [`Record`](../cluster/test_new_image.py#L295)
- [`build_collage(records, out_path, cell=220, cols=8, title='')`](../cluster/test_new_image.py#L310) — Строит коллаж с группировкой по (class_id, cluster_id).
- [`parse_args()`](../cluster/test_new_image.py#L403)
- [`main()`](../cluster/test_new_image.py#L419)

<a id="dependency-health"></a>

## dependency_health

Исходник: [dependency_health.py](../dependency_health.py#L1).

**Прямые потребители:** [install](#install), [tests.test_dependency_installation](#tests-test-dependency-installation).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `argparse`, `importlib`, `importlib.metadata`, `json`, `pathlib.Path`, `re`, `subprocess`, `sys`.

**Объявления:**

- [`installed_opencv()`](../dependency_health.py#L18)
  Явно импортируют: [install](#install).
- [`direct_requirement_errors(requirements_file=None)`](../dependency_health.py#L28) — Validate the declared direct pins, accepting local Torch build suffixes.
- [`check_environment()`](../dependency_health.py#L48) — Return errors and explicitly allowed metadata exceptions.
- [`main(argv=None)`](../dependency_health.py#L77)

<a id="generate"></a>

## generate

Исходник: [generate.py](../generate.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [ui.application](#ui-application): `exec_window`

**Внешние/stdlib импорты:** `PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `datetime.datetime`, `os`, `sys`, `traceback`.

**Объявления:**

- [`log_event(message)`](../generate.py#L30)
- [`index_to_label(idx)`](../generate.py#L35) — 0 -> 'а)', 1 -> 'б)', ..., 32 -> 'я)', 33 -> 'а1)', 34 -> 'б1)', ...
- Класс [`CollageApp`](../generate.py#L49) (`QMainWindow`)

<a id="import-libs-external"></a>

## import_libs_external

Исходник: [import_libs_external.py](../import_libs_external.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `importlib.import_module`.

**Объявления:**

- [`__getattr__(name)`](../import_libs_external.py#L138)
- [`__dir__()`](../import_libs_external.py#L152)

<a id="import-libs-internal"></a>

## import_libs_internal

Исходник: [import_libs_internal.py](../import_libs_internal.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `importlib.import_module`.

**Объявления:**

- [`__getattr__(name)`](../import_libs_internal.py#L270)
- [`__dir__()`](../import_libs_internal.py#L284)

<a id="import-libs-methods-ui"></a>

## import_libs_methods_ui

Исходник: [import_libs_methods_ui.py](../import_libs_methods_ui.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `importlib.import_module`.

**Объявления:**

- [`__getattr__(name)`](../import_libs_methods_ui.py#L31)
- [`__dir__()`](../import_libs_methods_ui.py#L42)

<a id="install"></a>

## install

Исходник: [install.py](../install.py#L1).

**Прямые потребители:** [tests.test_dependency_installation](#tests-test-dependency-installation).

**Прямые локальные импорты:**
- [dependency_health](#dependency-health): `OPENCV_PROVIDERS`, `SELECTED_OPENCV`, `installed_opencv`

**Внешние/stdlib импорты:** `argparse`, `pathlib.Path`, `subprocess`, `sys`.

**Объявления:**

- [`run_pip(*args, dry_run=False)`](../install.py#L15)
- [`normalize_opencv(dry_run=False, force=False)`](../install.py#L22) — Repair shared cv2 files after removing competing OpenCV wheels.
- [`main(argv=None)`](../install.py#L39)

<a id="main"></a>

## main

Исходник: [main.py](../main.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [main_ui](#main-ui): `StartupDialog`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
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
- [ui.application](#ui-application): `exec_window`
- [ui.image_navigation](#ui-image-navigation): `NavigationShortcutInstaller`
- [ui.settings_service](#ui-settings-service): `apply_theme`, `settings`
- [ui.settings_ui](#ui-settings-ui): `SettingsWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSplashScreen`, `PyQt5.QtWidgets.QTabWidget`, `datetime.datetime`, `importlib.metadata.PackageNotFoundError`, `importlib.metadata.version`, `os`, `platform`, `sys`.

**Объявления:**

- Класс [`ImagePlayer`](../main.py#L29) (`QMainWindow`)
- [`main()`](../main.py#L278)

<a id="main-ui"></a>

## main_ui

Исходник: [main_ui.py](../main_ui.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSettings`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`.

**Объявления:**

- Класс [`StartupDialog`](../main_ui.py#L32) (`QDialog`)

<a id="path-setup"></a>

## path_setup

Исходник: [path_setup.py](../path_setup.py#L1).

**Прямые потребители:** [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.model_research](#tabs-model-research), [tabs.threshold_methods](#tabs-threshold-methods), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.settings_service](#ui-settings-service), [ui.settings_ui](#ui-settings-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `pathlib.Path`, `sys`.

**Объявления:**

- [`setup_project_path()`](../path_setup.py#L8) — Explicit compatibility helper; normal module imports need no mutation.
  Явно импортируют: [ui.settings_ui](#ui-settings-ui).
- [`get_project_root()`](../path_setup.py#L15) — Return the repository root without changing the import search path.
  Явно импортируют: [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.model_research](#tabs-model-research), [tabs.threshold_methods](#tabs-threshold-methods), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.settings_service](#ui-settings-service).

<a id="rename-data"></a>

## rename_data

Исходник: [rename_data.py](../rename_data.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `argparse`, `pathlib.Path`, `re`.

**Объявления:**

- [`rename_files(folder)`](../rename_data.py#L10) — Rename files in an explicitly selected directory; skip collisions.
- [`main()`](../rename_data.py#L39)

<a id="runtime-policy"></a>

## runtime_policy

Исходник: [runtime_policy.py](../runtime_policy.py#L1).

**Прямые потребители:** [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [dependency_health](#dependency-health), [import_libs_external](#import-libs-external), [import_libs_internal](#import-libs-internal), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `sys`.

**Объявления:**

- [`configure_ultralytics()`](../runtime_policy.py#L6) — Keep optional backend installation under install.py's control.
  Явно импортируют: [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [dependency_health](#dependency-health), [import_libs_external](#import-libs-external), [import_libs_internal](#import-libs-internal), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils.dataset_audit_logic](#utils-dataset-audit-logic), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

<a id="runtime-torch"></a>

## runtime_torch

Исходник: [runtime_torch.py](../runtime_torch.py#L1).

**Прямые потребители:** [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_torch_runtime](#tests-test-torch-runtime), [utils.dataset_audit_logic](#utils-dataset-audit-logic).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `dataclasses.dataclass`, `importlib`, `sys`, `threading`.

**Объявления:**

- Класс [`TorchRuntime`](../runtime_torch.py#L9)
- [`get_torch_runtime()`](../runtime_torch.py#L21) — Cache failure as well as success; never retry a broken native loader.
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis), [utils.dataset_audit_logic](#utils-dataset-audit-logic).
- [`prepare_windows_torch()`](../runtime_torch.py#L43) — Load Torch before GUI/native stacks on Windows; keep Linux loading lazy.
  Явно импортируют: [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`require_torch()`](../runtime_torch.py#L54) — Return the loaded module or report the cached failure without retrying.
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis), [utils.dataset_audit_logic](#utils-dataset-audit-logic).

<a id="tabs-dataset-analysis"></a>

## tabs.dataset_analysis

Исходник: [tabs/dataset_analysis.py](../tabs/dataset_analysis.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `get_torch_runtime`, `prepare_windows_torch`, `require_torch`
- [ui.application](#ui-application): `exec_window`
- [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui): `TrainingImageLabel`, `setup_dataset_analysis_ui`
- [utils.dataset_analysis_logic](#utils-dataset-analysis-logic): `BrightnessAnalyzer`, `ClassImbalanceAnalyzer`, `DatasetAnalyzer`, `FilterParams`, `ResolutionAnalyzer`, `TrainingResultsLoader`, `clear_brightness_cache`
- [utils.dataset_audit_logic](#utils-dataset-audit-logic): `CLIP_AVAILABLE`, `DuplicateDetector`, `EmbeddingClusterThread`, `LeakageChecker`, `LossInspectionThread`, `ObjectDistributionHeatmap`, `ReportThread`, `detect_device`
- [utils.image_io](#utils-image-io): `load_annotations`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `get_class_color`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QTableWidgetItem`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `collections.Counter`, `cv2`, `io`, `logging`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `shutil`, `subprocess`, `sys`, `traceback`, `ultralytics.YOLO`, `ultralytics.utils.LOGGER`.

**Объявления:**

- Класс [`YoloValThread`](../tabs/dataset_analysis.py#L67) (`QThread`)
- Класс [`DatasetAnalysisWindow`](../tabs/dataset_analysis.py#L297) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-deep-learning-methods"></a>

## tabs.deep_learning_methods

Исходник: [tabs/deep_learning_methods.py](../tabs/deep_learning_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `DeepLearningPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui): `setup_deep_learning_methods_ui`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops): `create_segmentor`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSpinBox`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`DeepLearningWindow`](../tabs/deep_learning_methods.py#L53) (`QMainWindow`)
- [`center_window(window, width=1200, height=800)`](../tabs/deep_learning_methods.py#L844)

<a id="tabs-deep-learning-training"></a>

## tabs.deep_learning_training

Исходник: [tabs/deep_learning_training.py](../tabs/deep_learning_training.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui): `setup_deep_learning_training_ui`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QProcess`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `importlib.util.find_spec`, `json`, `os`, `pathlib.Path`, `sys`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`YoloTrainWindow`](../tabs/deep_learning_training.py#L32) (`QMainWindow`)

<a id="tabs-demo-ensemble-methods"></a>

## tabs.demo_ensemble_methods

Исходник: [tabs/demo_ensemble_methods.py](../tabs/demo_ensemble_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `ThresholdPresetManager`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui): `setup_demo_ensemble_methods_ui`
- [utils.ensemble_utils](#utils-ensemble-utils): `ensemble_detections`, `get_detections_from_preset`
- [utils.image_io](#utils-image-io): `load_images_universal`, `numpy_to_qpixmap`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `validate_preset`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `apply_threshold_method`, `draw_selected_objects`, `draw_yolo_annotations`, `get_display_params`, `segment_contours`, `segment_min_area_rect`, `segment_projections`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `cv2`, `importlib.util.find_spec`, `os`, `pathlib.Path`, `sys`, `torch`, `traceback`, `ultralytics.YOLO`.

**Объявления:**

- [`print_flush(*args, **kwargs)`](../tabs/demo_ensemble_methods.py#L52)
- Класс [`YoloInferThread`](../tabs/demo_ensemble_methods.py#L63) (`QThread`)
- Класс [`EnsembleThread`](../tabs/demo_ensemble_methods.py#L139) (`QThread`)
- Класс [`YoloDemoWindow`](../tabs/demo_ensemble_methods.py#L204) (`QMainWindow`)

<a id="tabs-gradient-methods"></a>

## tabs.gradient_methods

Исходник: [tabs/gradient_methods.py](../tabs/gradient_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `GradientPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui): `setup_gradient_ui`
- [utils_auto_research.gradient_auto_research](#utils-auto-research-gradient-auto-research): `GradientAutoResearch`
- [utils_auto_research.gradient_auto_settings_dialog](#utils-auto-research-gradient-auto-settings-dialog): `GradientAutoSettingsDialog`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `cv2`, `json`, `numpy`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`GradientMethodsWindow`](../tabs/gradient_methods.py#L45) (`QMainWindow`)

<a id="tabs-interactive-methods"></a>

## tabs.interactive_methods

Исходник: [tabs/interactive_methods.py](../tabs/interactive_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `InteractivePresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui): `setup_interactive_ui`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.interactive_ops](#utils-ops-interactive-ops): `active_contour_segmentation`, `grabcut_segmentation`, `lazy_snapping`, `onecut_segmentation`, `random_walker_segmentation`, `supercut_segmentation`, `watershed_segmentation`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `traceback`.

**Объявления:**

- Класс [`InteractiveMethodsWindow`](../tabs/interactive_methods.py#L49) (`QMainWindow`)

<a id="tabs-layout-dataset"></a>

## tabs.layout_dataset

Исходник: [tabs/layout_dataset.py](../tabs/layout_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_view](#ui-image-view): `numpy_to_qpixmap`, `set_fill_color_button_preview`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui): `setup_layout_dataset_ui`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `convert_to_grayscale`, `image_size_from_file`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `read_mask_with_fallback`, `resize_to_max_side`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `build_yolo_save_target_map`, `find_mask_for_image`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `canonicalize_obb_corners`, `format_label_save_confirmation`, `load_annotations`, `load_annotations_for_image`, `save_annotations`, `save_annotations_for_image`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `update_annotation_list`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QAction`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `traceback`, `ultralytics.YOLO`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/layout_dataset.py#L82) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/layout_dataset.py#L127) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/layout_dataset.py#L151) (`QThread`)
- Класс [`Labeler`](../tabs/layout_dataset.py#L201) (`QMainWindow`)

<a id="tabs-model-research"></a>

## tabs.model_research

Исходник: [tabs/model_research.py](../tabs/model_research.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `ThresholdPresetManager`
- [ui_tabs.model_research_ui](#ui-tabs-model-research-ui): `setup_model_research_ui`
- [utils.analysis](#utils-analysis): `analyze`, `global_top_diverse`
- [utils.image_io](#utils-image-io): `load_annotations`
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`
- [utils.prepare](#utils-prepare): `add_cluster_labels`, `add_meanap_metrics`, `aggregate_by_params`, `cluster_open_close`, `filter_anomaly_count`, `filter_duplicates`, `filter_full_image_object`, `filter_invalid_params`, `filter_no_true_objects`, `filter_oversegmentation`, `filter_zero_true`, `optimize_invert`, `sample_rows`, `save_unique_params`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `make_preset`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QTableWidgetItem`, `ast.literal_eval`, `os`, `pandas`, `pathlib.Path`, `re`, `sys`.

**Объявления:**

- Класс [`ModelResearchApp`](../tabs/model_research.py#L53) (`QMainWindow`)

<a id="tabs-preparing-dataset-yaml"></a>

## tabs.preparing_dataset_yaml

Исходник: [tabs/preparing_dataset_yaml.py](../tabs/preparing_dataset_yaml.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui): `setup_preparing_dataset_yaml_ui`
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new): `DatasetGeneratorThread`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_preparation_bridge](#utils-new-dataset-preparation-bridge): `mask_classes_from_rows`, `normalize_generation_pairs`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`, `convert_pairs_color`, `convert_pairs_to_8bit_png`, `select_pairs_for_8bit_png`, `select_pairs_for_conversion`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QTableWidgetItem`, `os`, `pathlib.Path`, `sys`, `yaml`.

**Объявления:**

- Класс [`DatasetPreparationWindow`](../tabs/preparing_dataset_yaml.py#L48) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-slice-dataset-sahi"></a>

## tabs.slice_dataset_sahi

Исходник: [tabs/slice_dataset_sahi.py](../tabs/slice_dataset_sahi.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui): `setup_sahi_slicer_ui`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `contextlib.redirect_stdout`, `cv2`, `json`, `math`, `numpy`, `os`, `pathlib.Path`, `random`, `sahi.slicing.slice_coco`, `shutil`, `sys`, `tempfile`, `traceback`.

**Объявления:**

- [`yolo_labels_to_coco_annotations(label_path, img_w, img_h, input_format='auto')`](../tabs/slice_dataset_sahi.py#L48)
- [`slice_dataset(coco_json_path, image_dir, output_dir, slice_size=640, overlap=0.2, min_area_ratio=0.3)`](../tabs/slice_dataset_sahi.py#L140)
- [`detect_dataset_format(coco_dict)`](../tabs/slice_dataset_sahi.py#L177)
- [`describe_segmentation(coco_dict)`](../tabs/slice_dataset_sahi.py#L184)
- [`subsample_negative_tiles(coco_dict, image_dir, keep_ratio=0.1, seed=42)`](../tabs/slice_dataset_sahi.py#L206)
- [`split_and_move(coco_dict, output_dir, train_ratio=0.8, seed=42, strip_segmentation=False)`](../tabs/slice_dataset_sahi.py#L247) — strip_segmentation: если True — удалить поле 'segmentation' из всех
- [`coco_to_yolo(coco_json_path, output_dir, class_names=None, annotation_format='bbox', verbose=True)`](../tabs/slice_dataset_sahi.py#L313)
- [`write_yolo_yaml(output_dir, class_names, channels=3)`](../tabs/slice_dataset_sahi.py#L433)
- [`visualize_samples(coco_json_path, output_dir, n=3)`](../tabs/slice_dataset_sahi.py#L448)
- Класс [`SliceWorker`](../tabs/slice_dataset_sahi.py#L500) (`QThread`)
- Класс [`SahiSlicerWindow`](../tabs/slice_dataset_sahi.py#L624) (`DatasetLoaderMixin, QMainWindow`)

<a id="tabs-threshold-methods"></a>

## tabs.threshold_methods

Исходник: [tabs/threshold_methods.py](../tabs/threshold_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `ThresholdPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui): `setup_threshold_ui`
- [utils_auto_research.threshold_auto_research](#utils-auto-research-threshold-auto-research): `AutoResearch`
- [utils_auto_research.threshold_auto_settings_dialog](#utils-auto-research-threshold-auto-settings-dialog): `AutoSettingsDialog`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology as apply_shared_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `cv2`, `json`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`ThresholdWindow`](../tabs/threshold_methods.py#L49) (`QMainWindow`)

<a id="tabs-traditional-ml-methods"></a>

## tabs.traditional_ml_methods

Исходник: [tabs/traditional_ml_methods.py](../tabs/traditional_ml_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `TraditionalMLPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui): `setup_traditional_ml_ui`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops): `compute_elbow_kmeans`, `create_superpixels`, `extract_pixel_features`, `extract_superpixel_features`, `find_optimal_k_elbow`, `predict_superpixel`, `train_decision_tree`, `train_kmeans`, `train_meanshift`, `train_random_forest`, `train_svm`, `train_xgboost`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSpinBox`, `cv2`, `joblib`, `numpy`, `os`, `pathlib.Path`, `sklearn.preprocessing.StandardScaler`, `sys`.

**Объявления:**

- Класс [`TraditionalMLWindow`](../tabs/traditional_ml_methods.py#L60) (`QMainWindow`)

<a id="tabs-viewing-dataset"></a>

## tabs.viewing_dataset

Исходник: [tabs/viewing_dataset.py](../tabs/viewing_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_view](#ui-image-view): `numpy_to_qpixmap`, `set_fill_color_button_preview`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui): `setup_viewing_dataset_ui`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `annotation_in_crop`, `class_id_for_annotation`, `crop_for_annotation`, `find_strong_overlaps`, `mask_roi_at_image_scale`, `normalized_annotation`, `prepare_annotation_preview`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `convert_to_grayscale`, `image_size_from_file`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `read_index_mask`, `read_mask_with_fallback`, `resize_to_max_side`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `build_yolo_save_target_map`, `find_mask_for_image`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `canonicalize_obb_corners`, `format_label_save_confirmation`, `load_annotations`, `load_annotations_for_image`, `save_annotations`, `save_annotations_for_image`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPalette`, `PyQt5.QtWidgets.QAction`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `collections.OrderedDict`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `traceback`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/viewing_dataset.py#L94) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/viewing_dataset.py#L139) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/viewing_dataset.py#L163) (`QThread`)
- Класс [`ViewingDataset`](../tabs/viewing_dataset.py#L213) (`QMainWindow`)

<a id="tabs-yolo-find-img"></a>

## tabs.yolo_find_img

Исходник: [tabs/yolo_find_img.py](../tabs/yolo_find_img.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui): `setup_yolo_find_img_ui`
- [utils.image_io](#utils-image-io): `normalize_to_uint8`, `read_image_with_fallback_find`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QDir`, `PyQt5.QtCore.QThread`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.QUrl`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QDesktopServices`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFileSystemModel`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `cv2`, `hashlib`, `os`, `pathlib.Path`, `shutil`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`ClickableLabel`](../tabs/yolo_find_img.py#L48) (`QLabel`)
- Класс [`ScanThread`](../tabs/yolo_find_img.py#L112) (`QThread`)
- Класс [`FindImagesWindow`](../tabs/yolo_find_img.py#L292) (`QMainWindow, setup_yolo_find_img_ui`)

<a id="tabs-yolo-test-aeb"></a>

## tabs.yolo_test_aeb

Исходник: [tabs/yolo_test_aeb.py](../tabs/yolo_test_aeb.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `PresetManager`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui): `setup_yolo_ae_vae_ui`
- [utils.image_io](#utils-image-io): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`
- [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils): `ConvAutoencoder`, `ConvAutoencoderGN`, `ConvDecoder`, `ConvDecoderGN`, `ConvEncoder`, `ConvEncoderGN`, `ConvVAE`, `ConvVAEGN`, `UNetAE`, `UNetDecoder`, `UNetEncoder`, `UNetVAE`, `build_ae_model_from_cfg`, `detect_arch`, `preprocess_image_for_ae`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `aeb_transform_image`, `apply_object_morphology`, `cls_suffix_from_detection`, `crop_to_detection`, `detection_to_unified`, `detections_to_unified`, `longest_edge_angle_deg`, `render_ae_error_map`, `render_unified_annotations`, `rotate_with_optional_crop`, `yolo_result_to_detections`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QInputDialog`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`YoloAEBPresetManager`](../tabs/yolo_test_aeb.py#L98) (`PresetManager`)
- Класс [`YoloInspectWindow`](../tabs/yolo_test_aeb.py#L194) (`QMainWindow`)

<a id="tests-test-dependency-installation"></a>

## tests.test_dependency_installation

Исходник: [tests/test_dependency_installation.py](../tests/test_dependency_installation.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [dependency_health](#dependency-health): `(модуль)`
- [install](#install): `(модуль)`

**Внешние/stdlib импорты:** `pathlib.Path`, `subprocess`, `tempfile`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`InstallationTests`](../tests/test_dependency_installation.py#L12) (`unittest.TestCase`)

<a id="tests-test-import-architecture"></a>

## tests.test_import_architecture

Исходник: [tests/test_import_architecture.py](../tests/test_import_architecture.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `ast`, `importlib.util`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- [`python_sources()`](../tests/test_import_architecture.py#L15)
- [`eager_imports(nodes)`](../tests/test_import_architecture.py#L22)
- Класс [`ArchitectureTests`](../tests/test_import_architecture.py#L34) (`unittest.TestCase`)

<a id="tests-test-label-save-merge"></a>

## tests.test_label_save_merge

Исходник: [tests/test_label_save_merge.py](../tests/test_label_save_merge.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `build_yolo_save_target_map`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `format_label_save_confirmation`

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`LabelSaveMergeTests`](../tests/test_label_save_merge.py#L18) (`unittest.TestCase`)

<a id="tests-test-pixel-morphology"></a>

## tests.test_pixel_morphology

Исходник: [tests/test_pixel_morphology.py](../tests/test_pixel_morphology.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils.ensemble_utils](#utils-ensemble-utils): `get_detections_from_preset`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `make_preset`, `validate_preset`
- [utils_new.settings_store](#utils-new-settings-store): `load_settings`, `normalize_settings`, `save_settings`

**Внешние/stdlib импорты:** `ast`, `copy`, `cv2`, `json`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`PixelPresetTests`](../tests/test_pixel_morphology.py#L41) (`unittest.TestCase`)
- Класс [`PixelMorphologyGuiTests`](../tests/test_pixel_morphology.py#L141) (`unittest.TestCase`)

<a id="tests-test-processing-smoke"></a>

## tests.test_processing_smoke

Исходник: [tests/test_processing_smoke.py](../tests/test_processing_smoke.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new): `DatasetGeneratorCoreNew`, `INPUT_LABELS`
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `AnnotationSample`, `ImageSize`, `MaskEncoding`, `Resize`, `from_legacy`, `rasterize_annotations`, `to_legacy`, `transform_sample`
- [utils_new.annotations.augmentation](#utils-new-annotations-augmentation): `apply_augmentation_plan`, `build_augmentation_plan`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `apply_threshold_method`

**Внешние/stdlib импорты:** `cv2`, `importlib.util`, `json`, `numpy`, `os`, `pathlib.Path`, `tempfile`, `torch`, `unittest`, `yaml`.

**Объявления:**

- Класс [`ProcessingSmokeTests`](../tests/test_processing_smoke.py#L13) (`unittest.TestCase`)

<a id="tests-test-runtime-imports"></a>

## tests.test_runtime_imports

Исходник: [tests/test_runtime_imports.py](../tests/test_runtime_imports.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `concurrent.futures`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`RuntimeImportTests`](../tests/test_runtime_imports.py#L15) (`unittest.TestCase`)

<a id="tests-test-segmentation-shared"></a>

## tests.test_segmentation_shared

Исходник: [tests/test_segmentation_shared.py](../tests/test_segmentation_shared.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `(модуль)`
- [ui.image_view](#ui-image-view): `(модуль)`
- [ui.settings_service](#ui-settings-service): `(модуль)`
- [utils.image_io](#utils-image-io): `LazyImageCache`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache as canonical`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`
- [utils_new.presets_store](#utils-new-presets-store): `read_presets`, `write_presets`
- [utils_new.segmentation.export](#utils-new-segmentation-export): `write_annotations_atomic`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `image_session`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `postprocessing`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `segment_contours`, `segment_min_area_rect`, `segment_projections`

**Внешние/stdlib импорты:** `ast`, `cv2`, `hashlib`, `json`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`SegmentationCommonTests`](../tests/test_segmentation_shared.py#L24) (`unittest.TestCase`)
- Класс [`SegmentationSharedGuiTests`](../tests/test_segmentation_shared.py#L150) (`unittest.TestCase`)

<a id="tests-test-torch-runtime"></a>

## tests.test_torch_runtime

Исходник: [tests/test_torch_runtime.py](../tests/test_torch_runtime.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `(модуль)`

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `types.SimpleNamespace`, `unittest`, `unittest.mock.Mock`, `unittest.mock.patch`.

**Объявления:**

- Класс [`TorchRuntimeTests`](../tests/test_torch_runtime.py#L16) (`unittest.TestCase`)

<a id="ui-DL-predict-settings"></a>

## ui.DL_predict_settings

Исходник: [ui/DL_predict_settings.py](../ui/DL_predict_settings.py#L1).

**Прямые потребители:** [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`BaseInferenceSettings`](../ui/DL_predict_settings.py#L19) (`QGroupBox`)
- Класс [`YOLOInferenceSettings`](../ui/DL_predict_settings.py#L43) (`BaseInferenceSettings`)
- Класс [`UNetInferenceSettings`](../ui/DL_predict_settings.py#L122) (`BaseInferenceSettings`)
- Класс [`DeepLabV3InferenceSettings`](../ui/DL_predict_settings.py#L463) (`BaseInferenceSettings`)
- Класс [`SegFormerInferenceSettings`](../ui/DL_predict_settings.py#L848) (`BaseInferenceSettings`)
- Класс [`SAMInferenceSettings`](../ui/DL_predict_settings.py#L885) (`BaseInferenceSettings`)
- Класс [`CustomONNXInferenceSettings`](../ui/DL_predict_settings.py#L928) (`BaseInferenceSettings`)

<a id="ui-DL-train-settings-base"></a>

## ui.DL_train_settings_base

Исходник: [ui/DL_train_settings_base.py](../ui/DL_train_settings_base.py#L1).

**Прямые потребители:** [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `os`, `sys`, `yaml`.

**Объявления:**

- Класс [`BaseTrainWidget`](../ui/DL_train_settings_base.py#L26) (`QWidget`)
- Класс [`YOLOTrainWidget`](../ui/DL_train_settings_base.py#L167) (`BaseTrainWidget`)
- Класс [`UNetTrainWidget`](../ui/DL_train_settings_base.py#L404) (`BaseTrainWidget`)

<a id="ui-DL-train-settings-new-model"></a>

## ui.DL_train_settings_new_model

Исходник: [ui/DL_train_settings_new_model.py](../ui/DL_train_settings_new_model.py#L1).

**Прямые потребители:** [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui).

**Прямые локальные импорты:**
- [ui.DL_train_settings_base](#ui-DL-train-settings-base): `BaseTrainWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `ast`, `os`, `sys`.

**Объявления:**

- Класс [`DeepLabV3TrainWidget`](../ui/DL_train_settings_new_model.py#L31) (`BaseTrainWidget`)
- Класс [`SegFormerTrainWidget`](../ui/DL_train_settings_new_model.py#L983) (`BaseTrainWidget`)
- Класс [`SAMTrainWidget`](../ui/DL_train_settings_new_model.py#L1118) (`BaseTrainWidget`)

<a id="ui-annotation-review-view"></a>

## ui.annotation_review_view

Исходник: [ui/annotation_review_view.py](../ui/annotation_review_view.py#L1).

**Прямые потребители:** [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [ui.image_view](#ui-image-view): `SmartGraphicsViewNew`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QGraphicsView`, `__future__.annotations`.

**Объявления:**

- Класс [`DatasetReviewGraphicsView`](../ui/annotation_review_view.py#L15) (`SmartGraphicsViewNew`)

<a id="ui-application"></a>

## ui.application

Исходник: [ui/application.py](../ui/application.py#L1).

**Прямые потребители:** [generate](#generate), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.QCoreApplication`, `PyQt5.QtCore.QEvent`.

**Объявления:**

- [`exec_window(window, app)`](../ui/application.py#L5) — Destroy the widget tree before QApplication/native modules shut down.
  Явно импортируют: [generate](#generate), [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

<a id="ui-control-buttons"></a>

## ui.control_buttons

Исходник: [ui/control_buttons.py](../ui/control_buttons.py#L1).

**Прямые потребители:** [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`ControlButtons`](../ui/control_buttons.py#L6) (`QWidget`)

<a id="ui-image-conversion"></a>

## ui.image_conversion

Исходник: [ui/image_conversion.py](../ui/image_conversion.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.image_view](#ui-image-view), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.results](#ui-segmentation-results), [utils.image_io](#utils-image-io).

**Прямые локальные импорты:**
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`

**Внешние/stdlib импорты:** `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPixmap`, `cv2`, `numpy`.

**Объявления:**

- [`numpy_to_qpixmap(img_bgr, max_side=8192)`](../ui/image_conversion.py#L7) — Display-only conversion. Geometry stays in source-image coordinates.
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.image_view](#ui-image-view), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.results](#ui-segmentation-results), [utils.image_io](#utils-image-io).

<a id="ui-image-navigation"></a>

## ui.image_navigation

Исходник: [ui/image_navigation.py](../ui/image_navigation.py#L1).

**Прямые потребители:** [main](#main), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [ui.settings_service](#ui-settings-service): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QEvent`, `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QAbstractSpinBox`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`ImageNavigationWidget`](../ui/image_navigation.py#L18) (`QWidget`)
- [`parse_key_spec(spec)`](../ui/image_navigation.py#L107) — Парсит строку вида:
- Класс [`NavigationShortcutInstaller`](../ui/image_navigation.py#L145) (`QObject`)

<a id="ui-image-view"></a>

## ui.image_view

Исходник: [ui/image_view.py](../ui/image_view.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.annotation_review_view](#ui-annotation-review-view), [ui.segmentation.interactive_view](#ui-segmentation-interactive-view), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap as numpy_to_qpixmap`
- [ui.settings_service](#ui-settings-service): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QPointF`, `PyQt5.QtCore.QRectF`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QBrush`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QFont`, `PyQt5.QtGui.QFontMetricsF`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPen`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtGui.QPolygonF`, `PyQt5.QtGui.QTransform`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QGraphicsPixmapItem`, `PyQt5.QtWidgets.QGraphicsScene`, `PyQt5.QtWidgets.QGraphicsView`, `PyQt5.QtWidgets.QMenu`, `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `dataclasses.replace`, `math`, `numpy`.

**Объявления:**

- Класс [`AnnotationDisplayStyle`](../ui/image_view.py#L52)
- [`set_fill_color_button_preview(button, color)`](../ui/image_view.py#L82) — Общий Qt-предпросмотр выбранного цвета без создания новой утилиты.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- Класс [`SmartGraphicsViewNew`](../ui/image_view.py#L101) (`QGraphicsView`)

<a id="ui-interactive-params-ui"></a>

## ui.interactive_params_ui

Исходник: [ui/interactive_params_ui.py](../ui/interactive_params_ui.py#L1).

**Прямые потребители:** [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`InteractiveParametersUI`](../ui/interactive_params_ui.py#L18) (`QWidget`)

<a id="ui-log-widget"></a>

## ui.log_widget

Исходник: [ui/log_widget.py](../ui/log_widget.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.model_research_ui](#ui-tabs-model-research-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`LogWidget`](../ui/log_widget.py#L13) (`QWidget`)

<a id="ui-morphology-ui"></a>

## ui.morphology_ui

Исходник: [ui/morphology_ui.py](../ui/morphology_ui.py#L1).

**Прямые потребители:** [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `MorphologyStep`, `SUPPORTED_OPERATIONS`, `SUPPORTED_SHAPES`

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `__future__.annotations`.

**Объявления:**

- Класс [`MorphologyWidget`](../ui/morphology_ui.py#L41) (`QGroupBox`)

<a id="ui-pair-validation-dialog-new"></a>

## ui.pair_validation_dialog_new

Исходник: [ui/pair_validation_dialog_new.py](../ui/pair_validation_dialog_new.py#L1).

**Прямые потребители:** [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `_read_with_fallback`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `load_annotations`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `cv2`, `numpy`, `os`.

**Объявления:**

- [`_to_qpixmap(bgr)`](../ui/pair_validation_dialog_new.py#L38) — Convert a BGR preview to an owned QPixmap without scaling its pixels.
- [`_palette_color(class_id)`](../ui/pair_validation_dialog_new.py#L48) — Read current application class palette, with deterministic fallback.
- Класс [`AutoFitReviewView`](../ui/pair_validation_dialog_new.py#L59) (`DatasetReviewGraphicsView`)
- Класс [`PairValidationDialog`](../ui/pair_validation_dialog_new.py#L102) (`QDialog`)

<a id="ui-segmentation-actions"></a>

## ui.segmentation.actions

Исходник: [ui/segmentation/actions.py](../ui/segmentation/actions.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QInputDialog`, `PyQt5.QtWidgets.QMessageBox`, `copy.deepcopy`.

**Объявления:**

- [`update_preset_combo(owner)`](../ui/segmentation/actions.py#L8)
- [`apply_preset(owner, index)`](../ui/segmentation/actions.py#L19)
- [`save_as_preset(owner)`](../ui/segmentation/actions.py#L31)
- [`delete_current_preset(owner)`](../ui/segmentation/actions.py#L60)
- [`toggle_log(owner, checked)`](../ui/segmentation/actions.py#L84)
- [`toggle_histogram(owner, checked)`](../ui/segmentation/actions.py#L89)
- [`log(owner, message)`](../ui/segmentation/actions.py#L96)

<a id="ui-segmentation-export"></a>

## ui.segmentation.export

Исходник: [ui/segmentation/export.py](../ui/segmentation/export.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `image_size_from_file`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `format_label_save_confirmation`
- [utils_new.segmentation.export](#utils-new-segmentation-export): `write_annotations_atomic`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QMessageBox`, `pathlib.Path`.

**Объявления:**

- [`save_selected(owner)`](../ui/segmentation/export.py#L14)

<a id="ui-segmentation-histogram"></a>

## ui.segmentation.histogram

Исходник: [ui/segmentation/histogram.py](../ui/segmentation/histogram.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- Класс [`HistogramCanvas`](../ui/segmentation/histogram.py#L6) (`FigureCanvasQTAgg`)
- [`update_histogram(owner, gray, threshold=None)`](../ui/segmentation/histogram.py#L12)
- [`update_current(owner)`](../ui/segmentation/histogram.py#L25)

<a id="ui-segmentation-image-input"></a>

## ui.segmentation.image_input

Исходник: [ui/segmentation/image_input.py](../ui/segmentation/image_input.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.results](#ui-segmentation-results): `clear_views`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `create_cache`, `image_paths`, `load_one_image_item`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `pathlib.Path`.

**Объявления:**

- [`choose_images(owner, load)`](../ui/segmentation/image_input.py#L15)
- [`choose_folder(owner, load)`](../ui/segmentation/image_input.py#L23)
- [`load_cached(owner, source, view_names, *, max_side=1024, reset=None)`](../ui/segmentation/image_input.py#L30)
- [`load_images_universal(source, require_annotations=False, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint='auto')`](../ui/segmentation/image_input.py#L73) — Temporary eager adapter for ML training; no Qt in the actual reader.
- [`load_eager(owner, source, *, max_side=640)`](../ui/segmentation/image_input.py#L104)
- [`reload_images(owner, load)`](../ui/segmentation/image_input.py#L114)
- [`clear_session(owner, *, reset=None, empty_fields=(), none_fields=())`](../ui/segmentation/image_input.py#L122)

<a id="ui-segmentation-interactive-view"></a>

## ui.segmentation.interactive_view

Исходник: [ui/segmentation/interactive_view.py](../ui/segmentation/interactive_view.py#L1).

**Прямые потребители:** [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui).

**Прямые локальные импорты:**
- [ui.image_view](#ui-image-view): `SmartGraphicsViewNew`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`.

**Объявления:**

- Класс [`SegmentationInputView`](../ui/segmentation/interactive_view.py#L11) (`SmartGraphicsViewNew`)

<a id="ui-segmentation-navigation"></a>

## ui.segmentation.navigation

Исходник: [ui/segmentation/navigation.py](../ui/segmentation/navigation.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`navigate(owner, index, reset=None, only_if_changed=False)`](../ui/segmentation/navigation.py#L3)
- [`step(owner, delta, reset=None)`](../ui/segmentation/navigation.py#L17)
- [`reset_zooms(owner)`](../ui/segmentation/navigation.py#L22)
- [`update_state(owner)`](../ui/segmentation/navigation.py#L27)

<a id="ui-segmentation-preset-bindings"></a>

## ui.segmentation.preset_bindings

Исходник: [ui/segmentation/preset_bindings.py](../ui/segmentation/preset_bindings.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.model_research](#tabs-model-research), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [utils_new.presets_store](#utils-new-presets-store): `read_presets`, `write_presets`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `make_preset`, `validate_preset`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `contextlib.contextmanager`, `copy.deepcopy`, `pathlib.Path`.

**Объявления:**

- [`blocked_signals(widgets)`](../ui/segmentation/preset_bindings.py#L22)
- [`control_value(widget)`](../ui/segmentation/preset_bindings.py#L33)
- [`set_control_value(widget, value)`](../ui/segmentation/preset_bindings.py#L43)
- [`validate_control(widget, value)`](../ui/segmentation/preset_bindings.py#L54)
- Класс [`PresetManager`](../ui/segmentation/preset_bindings.py#L71)
- Класс [`MethodPresetManager`](../ui/segmentation/preset_bindings.py#L144) (`PresetManager`)
- Класс [`ThresholdPresetManager`](../ui/segmentation/preset_bindings.py#L222) (`MethodPresetManager`)
- Класс [`GradientPresetManager`](../ui/segmentation/preset_bindings.py#L275) (`MethodPresetManager`)
- Класс [`InteractivePresetManager`](../ui/segmentation/preset_bindings.py#L302) (`MethodPresetManager`)
- Класс [`TraditionalMLPresetManager`](../ui/segmentation/preset_bindings.py#L333) (`MethodPresetManager`)
- Класс [`DeepLearningPresetManager`](../ui/segmentation/preset_bindings.py#L361) (`MethodPresetManager`)

<a id="ui-segmentation-results"></a>

## ui.segmentation.results

Исходник: [ui/segmentation/results.py](../ui/segmentation/results.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.segmentation.image_input](#ui-segmentation-image-input).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `format_annotation_display`, `format_object_for_list`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QListWidgetItem`.

**Объявления:**

- [`current_image(owner)`](../ui/segmentation/results.py#L12)
- [`update_object_list(owner)`](../ui/segmentation/results.py#L19)
- [`update_overlay(owner)`](../ui/segmentation/results.py#L39)
- [`update_coordinates(owner)`](../ui/segmentation/results.py#L55)
- [`on_selection_changed(owner, item)`](../ui/segmentation/results.py#L67)
- [`prepare_frame(owner, image)`](../ui/segmentation/results.py#L78) — Scene geometry always uses working pixels, even if QPixmap is capped.
- [`clear_views(owner)`](../ui/segmentation/results.py#L85)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).

<a id="ui-settings-service"></a>

## ui.settings_service

Исходник: [ui/settings_service.py](../ui/settings_service.py#L1).

**Прямые потребители:** [main](#main), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.image_navigation](#ui-image-navigation), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.settings_ui](#ui-settings-ui), [utils.smart_view](#utils-smart-view), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [utils_new.settings_store](#utils-new-settings-store): `load_settings`, `save_settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QPalette`, `colorsys`, `copy.deepcopy`, `os`.

**Объявления:**

- Класс [`SettingsManager`](../ui/settings_service.py#L26) (`QObject`)
- [`adjust_color(color, h_shift=0, s_shift=0, l_shift=0)`](../ui/settings_service.py#L167)
- [`generate_palette_for_theme(theme, primary_color)`](../ui/settings_service.py#L177)
- [`apply_theme(app)`](../ui/settings_service.py#L248)
  Явно импортируют: [main](#main), [ui.settings_ui](#ui-settings-ui).

<a id="ui-settings-ui"></a>

## ui.settings_ui

Исходник: [ui/settings_ui.py](../ui/settings_ui.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `setup_project_path`
- [ui.settings_service](#ui-settings-service): `DEFAULT_DATASET_CLASS_PALETTE_BGR`, `FONT_SIZE_PX_MAX`, `FONT_SIZE_PX_MIN`, `LINE_THICKNESS_PX_MAX`, `LINE_THICKNESS_PX_MIN`, `apply_theme`, `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `__future__.annotations`, `copy.deepcopy`.

**Объявления:**

- Класс [`SettingsWidget`](../ui/settings_ui.py#L46) (`QWidget`)

<a id="ui-tabs-dataset-analysis-ui"></a>

## ui_tabs.dataset_analysis_ui

Исходник: [ui_tabs/dataset_analysis_ui.py](../ui_tabs/dataset_analysis_ui.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSize`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QToolBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.backends.backend_qt5agg.NavigationToolbar2QT`, `matplotlib.figure.Figure`.

**Объявления:**

- Класс [`TrainingImageLabel`](../ui_tabs/dataset_analysis_ui.py#L38) (`QLabel`)
- [`_make_page(toolbox, title, description, content, buttons, min_height=240, content_stretch=1)`](../ui_tabs/dataset_analysis_ui.py#L75)
- [`_mk_min_spin(rng, suffix='', special='Не задано', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L109)
- [`_mk_max_spin(rng, suffix='', special='Без ограничений', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L120)
- [`_mk_min_dspin(rng, decimals=2, suffix='', special='Не задано', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L131)
- [`_mk_max_dspin(rng, decimals=2, suffix='', special='Без ограничений', tooltip='')`](../ui_tabs/dataset_analysis_ui.py#L144)
- [`setup_dataset_analysis_ui(parent)`](../ui_tabs/dataset_analysis_ui.py#L157)
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis).

<a id="ui-tabs-deep-learning-methods-ui"></a>

## ui_tabs.deep_learning_methods_ui

Исходник: [ui_tabs/deep_learning_methods_ui.py](../ui_tabs/deep_learning_methods_ui.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods).

**Прямые локальные импорты:**
- [ui.DL_predict_settings](#ui-DL-predict-settings): `CustomONNXInferenceSettings`, `DeepLabV3InferenceSettings`, `SAMInferenceSettings`, `SegFormerInferenceSettings`, `UNetInferenceSettings`, `YOLOInferenceSettings`
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_deep_learning_methods_ui(parent)`](../ui_tabs/deep_learning_methods_ui.py#L38)
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods).

<a id="ui-tabs-deep-learning-training-ui"></a>

## ui_tabs.deep_learning_training_ui

Исходник: [ui_tabs/deep_learning_training_ui.py](../ui_tabs/deep_learning_training_ui.py#L1).

**Прямые потребители:** [tabs.deep_learning_training](#tabs-deep-learning-training).

**Прямые локальные импорты:**
- [ui.DL_train_settings_base](#ui-DL-train-settings-base): `UNetTrainWidget`, `YOLOTrainWidget`
- [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model): `DeepLabV3TrainWidget`, `SAMTrainWidget`, `SegFormerTrainWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_deep_learning_training_ui(parent)`](../ui_tabs/deep_learning_training_ui.py#L14)
  Явно импортируют: [tabs.deep_learning_training](#tabs-deep-learning-training).

<a id="ui-tabs-demo-ensemble-methods-ui"></a>

## ui_tabs.demo_ensemble_methods_ui

Исходник: [ui_tabs/demo_ensemble_methods_ui.py](../ui_tabs/demo_ensemble_methods_ui.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).

**Прямые локальные импорты:**
- [ui.DL_predict_settings](#ui-DL-predict-settings): `YOLOInferenceSettings`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [utils.smart_view](#utils-smart-view): `SmartGraphicsView`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_demo_ensemble_methods_ui(parent)`](../ui_tabs/demo_ensemble_methods_ui.py#L27)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).

<a id="ui-tabs-gradient-methods-ui"></a>

## ui_tabs.gradient_methods_ui

Исходник: [ui_tabs/gradient_methods_ui.py](../ui_tabs/gradient_methods_ui.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_gradient_ui(parent)`](../ui_tabs/gradient_methods_ui.py#L31)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods).

<a id="ui-tabs-interactive-methods-ui"></a>

## ui_tabs.interactive_methods_ui

Исходник: [ui_tabs/interactive_methods_ui.py](../ui_tabs/interactive_methods_ui.py#L1).

**Прямые потребители:** [tabs.interactive_methods](#tabs-interactive-methods).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.interactive_params_ui](#ui-interactive-params-ui): `InteractiveParametersUI`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`
- [ui.segmentation.interactive_view](#ui-segmentation-interactive-view): `SegmentationInputView`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_interactive_ui(parent)`](../ui_tabs/interactive_methods_ui.py#L31) — Создаёт UI для вкладки интерактивных методов сегментации.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).

<a id="ui-tabs-layout-dataset-ui"></a>

## ui_tabs.layout_dataset_ui

Исходник: [ui_tabs/layout_dataset_ui.py](../ui_tabs/layout_dataset_ui.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset).

**Прямые локальные импорты:**
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.image_view](#ui-image-view): `SmartGraphicsViewNew`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `get_class_color`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QButtonGroup`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_layout_dataset_ui(parent)`](../ui_tabs/layout_dataset_ui.py#L31)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset).

<a id="ui-tabs-model-research-ui"></a>

## ui_tabs.model_research_ui

Исходник: [ui_tabs/model_research_ui.py](../ui_tabs/model_research_ui.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research).

**Прямые локальные импорты:**
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QFont`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `pandas`.

**Объявления:**

- Класс [`ModelResearchWidget`](../ui_tabs/model_research_ui.py#L25) (`QWidget`)
- [`setup_model_research_ui(parent)`](../ui_tabs/model_research_ui.py#L99) — Настройка UI для ModelResearchApp.
  Явно импортируют: [tabs.model_research](#tabs-model-research).

<a id="ui-tabs-preparing-dataset-yaml-ui"></a>

## ui_tabs.preparing_dataset_yaml_ui

Исходник: [ui_tabs/preparing_dataset_yaml_ui.py](../ui_tabs/preparing_dataset_yaml_ui.py#L1).

**Прямые потребители:** [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- [ui.control_buttons](#ui-control-buttons): `ControlButtons`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`_section(title, description=None)`](../ui_tabs/preparing_dataset_yaml_ui.py#L34)
- [`_page()`](../ui_tabs/preparing_dataset_yaml_ui.py#L48)
- [`_wrap_row(*widgets)`](../ui_tabs/preparing_dataset_yaml_ui.py#L61)
- [`setup_preparing_dataset_yaml_ui(parent)`](../ui_tabs/preparing_dataset_yaml_ui.py#L70)
  Явно импортируют: [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

<a id="ui-tabs-slice-dataset-sahi-ui"></a>

## ui_tabs.slice_dataset_sahi_ui

Исходник: [ui_tabs/slice_dataset_sahi_ui.py](../ui_tabs/slice_dataset_sahi_ui.py#L1).

**Прямые потребители:** [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_sahi_slicer_ui(parent)`](../ui_tabs/slice_dataset_sahi_ui.py#L26) — SAHI slicer UI: слева — пары/валидация, справа — SAHI.
  Явно импортируют: [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi).

<a id="ui-tabs-threshold-methods-ui"></a>

## ui_tabs.threshold_methods_ui

Исходник: [ui_tabs/threshold_methods_ui.py](../ui_tabs/threshold_methods_ui.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_threshold_ui(parent)`](../ui_tabs/threshold_methods_ui.py#L32) — Создаёт UI для пороговых методов внутри QMainWindow.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).

<a id="ui-tabs-traditional-ml-methods-ui"></a>

## ui_tabs.traditional_ml_methods_ui

Исходник: [ui_tabs/traditional_ml_methods_ui.py](../ui_tabs/traditional_ml_methods_ui.py#L1).

**Прямые потребители:** [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_traditional_ml_ui(parent)`](../ui_tabs/traditional_ml_methods_ui.py#L29)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

<a id="ui-tabs-viewing-dataset-ui"></a>

## ui_tabs.viewing_dataset_ui

Исходник: [ui_tabs/viewing_dataset_ui.py](../ui_tabs/viewing_dataset_ui.py#L1).

**Прямые потребители:** [tabs.viewing_dataset](#tabs-viewing-dataset).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_viewing_dataset_ui(parent)`](../ui_tabs/viewing_dataset_ui.py#L24) — Создаёт UI для просмотра датасета с поддержкой аннотаций, масок и YAML.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).

<a id="ui-tabs-yolo-find-img-ui"></a>

## ui_tabs.yolo_find_img_ui

Исходник: [ui_tabs/yolo_find_img_ui.py](../ui_tabs/yolo_find_img_ui.py#L1).

**Прямые потребители:** [tabs.yolo_find_img](#tabs-yolo-find-img).

**Прямые локальные импорты:**
- [ui.DL_predict_settings](#ui-DL-predict-settings): `YOLOInferenceSettings`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QStackedWidget`, `PyQt5.QtWidgets.QTreeView`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`setup_yolo_find_img_ui`](../ui_tabs/yolo_find_img_ui.py#L28) (`object`)

<a id="ui-tabs-yolo-test-aeb-ui"></a>

## ui_tabs.yolo_test_aeb_ui

Исходник: [ui_tabs/yolo_test_aeb_ui.py](../ui_tabs/yolo_test_aeb_ui.py#L1).

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_yolo_ae_vae_ui(parent)`](../ui_tabs/yolo_test_aeb_ui.py#L42)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

<a id="utils-analysis"></a>

## utils.analysis

Исходник: [utils/analysis.py](../utils/analysis.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research).

**Прямые локальные импорты:**
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`
- [utils.prepare](#utils-prepare): `add_meanap_metrics`

**Внешние/stdlib импорты:** `numpy`, `os`, `pandas`, `sklearn.ensemble.RandomForestRegressor`, `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.r2_score`, `sklearn.model_selection.train_test_split`.

**Объявления:**

- [`distance_between_presets(p1, p2)`](../utils/analysis.py#L15) — Расстояние между двумя пресетами (Series). Учитывает метод, cluster, cluster_co.
- [`diverse_top_k(df, k=5, score_col='pred_f1', candidate_pool_size=None)`](../utils/analysis.py#L24) — Жадный выбор топ-K разнообразных пресетов.
- [`analyze(df, output_dir=None, skip_metrics=False, iou_thresholds=None, use_diversity=True, use_iou_penalty=False, penalty_alpha=0.5, per_method=True, cluster_col='cluster', cluster_co_col='cluster_co')`](../utils/analysis.py#L52) — Анализ для каждого метода отдельно.
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`global_top_diverse(df, k=10, score_col='actual_f1', method_col='method', params_col='params', invert_col='invert', close_col='close_factor', open_col='open_factor')`](../utils/analysis.py#L212) — Жадный выбор топ-k разнообразных пресетов из глобального датасета.
  Явно импортируют: [tabs.model_research](#tabs-model-research).

<a id="utils-dataset-analysis-logic"></a>

## utils.dataset_analysis_logic

Исходник: [utils/dataset_analysis_logic.py](../utils/dataset_analysis_logic.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [utils.image_io](#utils-image-io): `load_annotations`

**Внешние/stdlib импорты:** `PIL.Image`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QTableWidgetItem`, `cv2`, `matplotlib.pyplot`, `numpy`, `os`, `shutil`.

**Объявления:**

- [`get_image_brightness(path)`](../utils/dataset_analysis_logic.py#L38) — Средняя яркость изображения (grayscale mean, 0..255).
- [`clear_brightness_cache()`](../utils/dataset_analysis_logic.py#L65) — Сброс кэша яркости (например, после удаления файлов).
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis).
- [`_safe_progress(progress_cb, i, total, step=25)`](../utils/dataset_analysis_logic.py#L73)
- [`_check_cancel(cancel_check)`](../utils/dataset_analysis_logic.py#L78)
- Класс [`FilterParams`](../utils/dataset_analysis_logic.py#L88)
  - [`FilterParams.__init__(self, min_w=0, max_w=0, min_h=0, max_h=0, min_brightness=-1, max_brightness=-1, min_objs=0, max_objs=0, min_area=0, max_area=0, min_area_pct=0.0, max_area_pct=0.0, min_aspect=0.0, max_aspect=0.0)`](../utils/dataset_analysis_logic.py#L108)
- Класс [`DatasetAnalyzer`](../utils/dataset_analysis_logic.py#L140)
  - [`DatasetAnalyzer.__init__(self, dataset_type=0, input_format='auto')`](../utils/dataset_analysis_logic.py#L141)
  - [`DatasetAnalyzer.set_dataset_type(self, dt)`](../utils/dataset_analysis_logic.py#L145)
  - [`DatasetAnalyzer.set_input_format(self, fmt)`](../utils/dataset_analysis_logic.py#L148)
  - [`DatasetAnalyzer.get_included_class_ids(class_table)`](../utils/dataset_analysis_logic.py#L155)
  - [`DatasetAnalyzer.get_class_names_by_orig_id(class_table)`](../utils/dataset_analysis_logic.py#L167)
  - [`DatasetAnalyzer.get_class_names_included(class_table)`](../utils/dataset_analysis_logic.py#L179)
  - [`DatasetAnalyzer.filter_pairs(self, pairs, included_class_ids, total_classes, params, progress_cb=None, include_empty=False)`](../utils/dataset_analysis_logic.py#L197)
  - [`DatasetAnalyzer._load_annotations_for_pair(self, lbl_path, included_class_ids)`](../utils/dataset_analysis_logic.py#L346)
  - [`DatasetAnalyzer._annotation_area(self, ann, w, h)`](../utils/dataset_analysis_logic.py#L369)
  - [`DatasetAnalyzer.render_metrics(metrics_summary_label, class_stats_table, ax_all, ax_classes, canvas, filtered_pairs, all_pairs, class_counts, total_objects, object_areas, class_areas, name_by_orig, ax_pie=None, ax_bar=None, ax_area_pct=None, object_areas_pct=None)`](../utils/dataset_analysis_logic.py#L406)
  - [`DatasetAnalyzer.export_subset(filtered_pairs, out_dir, class_names, progress_cb=None, log_cb=None, included_class_ids=None)`](../utils/dataset_analysis_logic.py#L582)
- Класс [`ClassImbalanceAnalyzer`](../utils/dataset_analysis_logic.py#L684)
  - [`ClassImbalanceAnalyzer.analyze(class_counts)`](../utils/dataset_analysis_logic.py#L686)
- Класс [`ResolutionAnalyzer`](../utils/dataset_analysis_logic.py#L740)
  - [`ResolutionAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils/dataset_analysis_logic.py#L742)
- Класс [`BrightnessAnalyzer`](../utils/dataset_analysis_logic.py#L779)
  - [`BrightnessAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils/dataset_analysis_logic.py#L784)
- Класс [`TrainingResultsLoader`](../utils/dataset_analysis_logic.py#L815)
  - [`TrainingResultsLoader.scan(folder)`](../utils/dataset_analysis_logic.py#L843)
  - [`TrainingResultsLoader.summary_text(data)`](../utils/dataset_analysis_logic.py#L902)

<a id="utils-dataset-audit-logic"></a>

## utils.dataset_audit_logic

Исходник: [utils/dataset_audit_logic.py](../utils/dataset_audit_logic.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `get_torch_runtime`, `require_torch`
- [utils.image_io](#utils-image-io): `load_annotations`

**Внешние/stdlib импорты:** `PIL.Image`, `PyQt5.QtCore.QThread`, `PyQt5.QtCore.pyqtSignal`, `collections.Counter`, `collections.defaultdict`, `cv2`, `datetime.datetime`, `hashlib`, `hdbscan`, `imagehash`, `importlib.util.find_spec`, `numpy`, `open_clip`, `os`, `sklearn.cluster.DBSCAN`, `sklearn.decomposition.PCA`, `transformers.CLIPModel`, `transformers.CLIPProcessor`, `ultralytics.YOLO`, `umap`.

**Объявления:**

- [`_safe_progress(progress_cb, i, total, step=25)`](../utils/dataset_audit_logic.py#L38)
- [`_check_cancel(cancel_check)`](../utils/dataset_audit_logic.py#L43)
- [`detect_device()`](../utils/dataset_audit_logic.py#L50) — Автоопределение устройства: 'cuda:0', если доступно, иначе 'cpu'.
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis).
- Класс [`DuplicateDetector`](../utils/dataset_audit_logic.py#L59)
  - [`DuplicateDetector._file_md5(path, chunk_size=1 << 20)`](../utils/dataset_audit_logic.py#L61)
  - [`DuplicateDetector._phash(path, hash_size=8)`](../utils/dataset_audit_logic.py#L72)
  - [`DuplicateDetector.find_duplicates(pairs, phash_threshold=8, progress_cb=None, cancel_check=None)`](../utils/dataset_audit_logic.py#L84)
- Класс [`LeakageChecker`](../utils/dataset_audit_logic.py#L139)
  - [`LeakageChecker._infer_split(path)`](../utils/dataset_audit_logic.py#L143)
  - [`LeakageChecker.check(duplicate_groups, pairs)`](../utils/dataset_audit_logic.py#L154)
- Класс [`ObjectDistributionHeatmap`](../utils/dataset_audit_logic.py#L178)
  - [`ObjectDistributionHeatmap.compute(pairs, dataset_type=0, grid_size=16, progress_cb=None, cancel_check=None)`](../utils/dataset_audit_logic.py#L180)
- Класс [`LossInspectionThread`](../utils/dataset_audit_logic.py#L253) (`QThread`)
  - [`LossInspectionThread.__init__(self, pairs, model_path, dataset_type=0, iou_threshold=0.3, conf_threshold=0.25, imgsz=640, device=None, input_format='auto', parent=None)`](../utils/dataset_audit_logic.py#L258)
  - [`LossInspectionThread.cancel(self)`](../utils/dataset_audit_logic.py#L272)
  - [`LossInspectionThread._iou_xyxy(a, b)`](../utils/dataset_audit_logic.py#L276)
  - [`LossInspectionThread.run(self)`](../utils/dataset_audit_logic.py#L287)
- Класс [`EmbeddingClusterThread`](../utils/dataset_audit_logic.py#L513) (`QThread`)
  - [`EmbeddingClusterThread.__init__(self, pairs, batch_size=16, device=None, parent=None)`](../utils/dataset_audit_logic.py#L518)
  - [`EmbeddingClusterThread.cancel(self)`](../utils/dataset_audit_logic.py#L525)
  - [`EmbeddingClusterThread.run(self)`](../utils/dataset_audit_logic.py#L528)
- Класс [`ReportThread`](../utils/dataset_audit_logic.py#L679) (`QThread`)
  - [`ReportThread.__init__(self, data, save_path, sections, parent=None)`](../utils/dataset_audit_logic.py#L682)
  - [`ReportThread.run(self)`](../utils/dataset_audit_logic.py#L688)

<a id="utils-ensemble-utils"></a>

## utils.ensemble_utils

Исходник: [utils/ensemble_utils.py](../utils/ensemble_utils.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_pixel_morphology](#tests-test-pixel-morphology).

**Прямые локальные импорты:**
- [utils.metrics_core](#utils-metrics-core): `iou_xyxy`, `xyxy_to_yolo`, `yolo_to_xyxy`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `validate_preset`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `apply_threshold_method`, `segment_contours`, `segment_min_area_rect`, `segment_projections`

**Внешние/stdlib импорты:** `cv2`, `numpy`.

**Объявления:**

- [`contains_center(box1, box2)`](../utils/ensemble_utils.py#L16) — Проверяет, лежит ли центр box2 внутри box1.
- [`boxes_contain_each_other(box1, box2)`](../utils/ensemble_utils.py#L26) — Проверяет, содержит ли один бокс другой (центр одного внутри другого).
- [`non_max_suppression(boxes, scores, iou_threshold=0.5, conf_threshold=0.1)`](../utils/ensemble_utils.py#L36) — Классический NMS: сортировка по уверенности, выбор лучшего, подавление по IoU.
- [`weighted_boxes_fusion(boxes, scores, weights, iou_threshold=0.5, conf_threshold=0.1, use_containment=False, verbose=False)`](../utils/ensemble_utils.py#L88) — Weighted Boxes Fusion (WBF) – взвешенное усреднение координат и уверенностей.
- [`ensemble_detections(detections_list, weights, iou_threshold=0.5, conf_threshold=0.1, use_containment=False, method='wbf', verbose=False)`](../utils/ensemble_utils.py#L144) — Объединяет детекции из нескольких источников с помощью выбранного метода.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`extract_threshold_params(preset, method)`](../utils/ensemble_utils.py#L221) — Извлекает параметры для apply_threshold_method из плоского словаря пресета.
- [`get_detections_from_preset(image, gray, preset, confidence=0.7)`](../utils/ensemble_utils.py#L248) — Применяет пресет к изображению и возвращает список детекций.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_pixel_morphology](#tests-test-pixel-morphology).
- [`run_ensemble(image, gray, yolo_detections, yolo_weight, presets_with_weights, iou_threshold, conf_threshold, use_containment=False, method='wbf', verbose=True)`](../utils/ensemble_utils.py#L328) — Выполняет полный ансамбль: собирает детекции из YOLO и всех пресетов,

<a id="utils-image-io"></a>

## utils.image_io

Исходник: [utils/image_io.py](../utils/image_io.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.model_research](#tabs-model-research), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap as numpy_to_qpixmap`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache as LazyImageCache`

**Внешние/stdlib импорты:** `PIL.Image`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `collections.OrderedDict`, `concurrent.futures.ThreadPoolExecutor`, `cv2`, `math`, `numpy`, `os`, `pathlib.Path`, `threading`, `tifffile`, `yaml`.

**Объявления:**

- [`_xywhr_to_corners(cx, cy, w, h, angle_rad)`](../utils/image_io.py#L25)
- [`_corners_to_xywhr(corners)`](../utils/image_io.py#L40)
- [`canonicalize_obb_corners(norm_pts)`](../utils/image_io.py#L73)
- [`_read_with_pil_draft(image_path, max_side=HARD_MAX_SIDE)`](../utils/image_io.py#L114)
- [`_read_with_tifffile(image_path)`](../utils/image_io.py#L136)
- [`read_image_with_fallback(image_path)`](../utils/image_io.py#L158)
- [`read_mask_with_fallback(mask_path)`](../utils/image_io.py#L232) — Читает маску, сохраняя её dtype (uint8/uint16).
- [`resize_to_max_side(img, max_side=1024)`](../utils/image_io.py#L267)
- [`normalize_to_uint8(img)`](../utils/image_io.py#L277)
  Явно импортируют: [tabs.yolo_find_img](#tabs-yolo-find-img).
- [`convert_to_grayscale(img)`](../utils/image_io.py#L290)
- [`_classify_line(n_parts, format_hint)`](../utils/image_io.py#L321)
- [`load_annotations(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L347)
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.model_research](#tabs-model-research), [utils.dataset_analysis_logic](#utils-dataset-analysis-logic), [utils.dataset_audit_logic](#utils-dataset-audit-logic).
- [`save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False)`](../utils/image_io.py#L406)
- [`load_annotations_obb(txt_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L434)
- [`load_annotations_for_image(img_path, img_w, img_h, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L439)
- [`load_images_universal(source, require_annotations=False, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L447)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`load_dataset_from_yaml(yaml_path, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L531) — Загружает YOLO-датасет из data.yaml.
- [`load_dataset_from_yaml_with_masks(yaml_path, resize_enabled=True, max_side=1024, progress_callback=None, parent=None)`](../utils/image_io.py#L626) — Загружает YAML с масками.
- [`save_coordinates(main_window)`](../utils/image_io.py#L745)
- [`find_image_label_pairs(images_folder, labels_folder, img_extensions=None)`](../utils/image_io.py#L779)
- [`collect_unique_classes_from_labels(label_paths, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L795)
- [`read_image_with_fallback_find(image_path)`](../utils/image_io.py#L805)
  Явно импортируют: [tabs.yolo_find_img](#tabs-yolo-find-img).
- [`convert_segment_masks_to_yolo_seg_manual(masks_dir: str, output_dir: str, pixel_to_class: dict, mask_offset: int=0)`](../utils/image_io.py#L850) — Конвертация масок в YOLO-seg .txt (lossy, findContours).
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint=FORMAT_HINT_AUTO)`](../utils/image_io.py#L911)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

<a id="utils-metrics-core"></a>

## utils.metrics_core

Исходник: [utils/metrics_core.py](../utils/metrics_core.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research), [utils.analysis](#utils-analysis), [utils.ensemble_utils](#utils-ensemble-utils), [utils.prepare](#utils-prepare).

**Прямые локальные импорты:**
- [utils.prepare](#utils-prepare): `add_meanap_metrics`

**Внешние/stdlib импорты:** `ast.literal_eval`, `numpy`, `pandas`, `re`.

**Объявления:**

- [`yolo_to_xyxy(cx, cy, w, h)`](../utils/metrics_core.py#L11) — Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
  Явно импортируют: [utils.ensemble_utils](#utils-ensemble-utils).
- [`xyxy_to_yolo(x1, y1, x2, y2)`](../utils/metrics_core.py#L23) — Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
  Явно импортируют: [utils.ensemble_utils](#utils-ensemble-utils).
- [`iou(box1, box2)`](../utils/metrics_core.py#L35) — Вычисляет IoU для двух боксов в YOLO-формате (cls, cx, cy, w, h).
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`iou_xyxy(box1, box2)`](../utils/metrics_core.py#L65) — Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
  Явно импортируют: [utils.ensemble_utils](#utils-ensemble-utils).
- [`parse_objects(s)`](../utils/metrics_core.py#L84) — Преобразует строковое представление списка объектов в список кортежей.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_precision_recall_f1(tp, fp, fn)`](../utils/metrics_core.py#L98) — Вычисляет precision, recall и F1-меру по количеству TP, FP, FN.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_metrics(pred_boxes, gt_boxes, iou_threshold=0.5)`](../utils/metrics_core.py#L108) — Для одного изображения вычисляет F1, TP, FP, FN.
  Явно импортируют: [utils.prepare](#utils-prepare).
- [`compute_ap(precision, recall)`](../utils/metrics_core.py#L143) — Вычисляет Average Precision (AP) по кривой precision-recall.
- [`compute_map(df, iou_thresholds=None)`](../utils/metrics_core.py#L160) — Вычисляет mAP (mean Average Precision) для датасета.
- [`extract_numeric_params(method: str, params_str: str)`](../utils/metrics_core.py#L176) — Извлекает числовые параметры из строки params (поддерживает кириллицу и латиницу).
  Явно импортируют: [utils.prepare](#utils-prepare).

<a id="utils-prepare"></a>

## utils.prepare

Исходник: [utils/prepare.py](../utils/prepare.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research), [utils.analysis](#utils-analysis), [utils.metrics_core](#utils-metrics-core).

**Прямые локальные импорты:**
- [utils.metrics_core](#utils-metrics-core): `DEFAULT_IOU_THRESHOLDS`, `compute_metrics as compute_metrics_core`, `compute_precision_recall_f1`, `extract_numeric_params`, `iou`, `parse_objects`

**Внешние/stdlib импорты:** `ast.literal_eval`, `json`, `matplotlib.pyplot`, `numpy`, `os`, `pandas`, `re`, `sklearn.cluster.KMeans`, `sklearn.decomposition.PCA`, `sklearn.preprocessing.StandardScaler`.

**Объявления:**

- [`compute_metrics_detailed(row, iou_thresholds)`](../utils/prepare.py#L28) — row: dict-like с ключами 'objects' и 'true_objects'
- [`add_meanap_metrics(df, iou_thresholds=None)`](../utils/prepare.py#L113) — Добавляет в DataFrame колонки:
  Явно импортируют: [tabs.model_research](#tabs-model-research), [utils.analysis](#utils-analysis), [utils.metrics_core](#utils-metrics-core).
- [`compute_f1(df: pd.DataFrame, iou_threshold: float=0.5)`](../utils/prepare.py#L166) — Оставлена для обратной совместимости, использует старую метрику.
- [`filter_by_f1_threshold(df: pd.DataFrame, threshold: float)`](../utils/prepare.py#L200)
- [`sample_rows(df: pd.DataFrame, n: int, random_state: int=42)`](../utils/prepare.py#L212)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_full_image_object(df: pd.DataFrame)`](../utils/prepare.py#L223) — Удаляет строки, где среди предсказанных объектов есть объект, покрывающий почти всё изображение.
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_invalid_params(df: pd.DataFrame)`](../utils/prepare.py#L252)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_anomaly_count(df: pd.DataFrame, max_objects: int=15)`](../utils/prepare.py#L326)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_zero_true(df: pd.DataFrame)`](../utils/prepare.py#L341)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_oversegmentation(df: pd.DataFrame, ratio: float=2.0)`](../utils/prepare.py#L356)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_duplicates(df: pd.DataFrame)`](../utils/prepare.py#L371)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`filter_no_true_objects(df: pd.DataFrame)`](../utils/prepare.py#L381) — Удаляет строки, в которых нет истинных объектов (num_true_objects == 0).
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`aggregate_by_params(df: pd.DataFrame, group_by=None, score_col='mean_f1', top_k=1)`](../utils/prepare.py#L400) — Агрегирует датасет, оставляя топ-K строк с наибольшим score_col в каждой группе.
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`save_cluster_report(df, method, cluster_ids, param_names, clusters_dir, method_name)`](../utils/prepare.py#L441) — Сохраняет текстовый отчёт о кластерах для одного метода.
- [`add_cluster_labels(df: pd.DataFrame, output_dir: str=None)`](../utils/prepare.py#L477) — Для каждого метода выполняет кластеризацию параметров и добавляет колонку 'cluster'.
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`save_unique_params(df: pd.DataFrame, output_dir: str=None)`](../utils/prepare.py#L572) — Для каждого метода сохраняет уникальные значения params в текстовый файл.
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`optimize_invert(df: pd.DataFrame)`](../utils/prepare.py#L601) — Для каждого метода оставляет только строки с лучшим значением invert (True/False)
  Явно импортируют: [tabs.model_research](#tabs-model-research).
- [`cluster_open_close(df: pd.DataFrame, n_clusters: int=3)`](../utils/prepare.py#L636) — Для каждого уникального сочетания (method, cluster) выполняет кластеризацию
  Явно импортируют: [tabs.model_research](#tabs-model-research).

<a id="utils-smart-view"></a>

## utils.smart_view

Исходник: [utils/smart_view.py](../utils/smart_view.py#L1).

**Прямые потребители:** [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui).

**Прямые локальные импорты:**
- [ui.settings_service](#ui-settings-service): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QPointF`, `PyQt5.QtCore.QRectF`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QBrush`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPainterPath`, `PyQt5.QtGui.QPen`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QGraphicsPixmapItem`, `PyQt5.QtWidgets.QGraphicsScene`, `PyQt5.QtWidgets.QGraphicsView`, `PyQt5.QtWidgets.QMenu`, `cv2`, `math`, `numpy`.

**Объявления:**

- Класс [`SmartGraphicsView`](../utils/smart_view.py#L19) (`QGraphicsView`)
  - [`SmartGraphicsView.__init__(self, parent=None)`](../utils/smart_view.py#L83)
  - [`SmartGraphicsView._setup_pan_mode(self)`](../utils/smart_view.py#L170)
  - [`SmartGraphicsView._setup_drawing_mode(self)`](../utils/smart_view.py#L174)
  - [`SmartGraphicsView._setup_edit_mode(self)`](../utils/smart_view.py#L191)
  - [`SmartGraphicsView.set_suggested_save_name(self, name)`](../utils/smart_view.py#L204)
  - [`SmartGraphicsView.set_drawing_tool(self, tool)`](../utils/smart_view.py#L210)
  - [`SmartGraphicsView.set_edit_mode(self, enabled)`](../utils/smart_view.py#L229)
  - [`SmartGraphicsView._clear_edit_state(self)`](../utils/smart_view.py#L245)
  - [`SmartGraphicsView.set_annotations(self, annotations, img_width, img_height)`](../utils/smart_view.py#L261)
  - [`SmartGraphicsView.set_selected_index(self, idx)`](../utils/smart_view.py#L266)
  - [`SmartGraphicsView.set_callbacks(self, on_rect_drawn=None, on_obb_drawn=None, on_segment_drawn=None, on_scribble_added=None, on_display_update=None, on_reset_tool=None, on_annotation_modified=None, on_selection_changed=None, on_annotation_right_clicked=None, on_log=None)`](../utils/smart_view.py#L288)
  - [`SmartGraphicsView.set_image_data(self, width, height, fg_scribbles=None, bg_scribbles=None)`](../utils/smart_view.py#L305)
  - [`SmartGraphicsView.set_pixmap(self, pixmap, preserve_view=True)`](../utils/smart_view.py#L309)
  - [`SmartGraphicsView.reset_view(self)`](../utils/smart_view.py#L324)
  - [`SmartGraphicsView.save_current_image(self)`](../utils/smart_view.py#L330)
  - [`SmartGraphicsView._scene_per_px(self)`](../utils/smart_view.py#L345)
  - [`SmartGraphicsView._get_points_pixel(self, ann)`](../utils/smart_view.py#L357)
  - [`SmartGraphicsView.point_in_polygon(self, pt, poly)`](../utils/smart_view.py#L381)
  - [`SmartGraphicsView._point_to_segment_dist(px, py, x1, y1, x2, y2)`](../utils/smart_view.py#L401)
  - [`SmartGraphicsView._hit_test_point(self, point, pos_pixel, radius=None)`](../utils/smart_view.py#L417)
  - [`SmartGraphicsView._hit_test_detect(self, ann, pos_pixel)`](../utils/smart_view.py#L425)
  - [`SmartGraphicsView._hit_test_segment(self, ann, pos_pixel)`](../utils/smart_view.py#L437)
  - [`SmartGraphicsView._hit_test_any(self, ann, pos_pixel)`](../utils/smart_view.py#L449)
  - [`SmartGraphicsView._find_hit(self, pos_pixel, prefer_idx=-1)`](../utils/smart_view.py#L461)
  - [`SmartGraphicsView._obb_to_params(self, pts_pixel)`](../utils/smart_view.py#L482)
  - [`SmartGraphicsView._obb_to_points(self, cx, cy, w, h, angle)`](../utils/smart_view.py#L525)
  - [`SmartGraphicsView._obb_points_in_bounds(self, points, eps=0.5)`](../utils/smart_view.py#L565)
  - [`SmartGraphicsView._obb_world_half_extent(self, w, h, angle)`](../utils/smart_view.py#L572)
  - [`SmartGraphicsView._get_obb_handles(self, pts_pixel)`](../utils/smart_view.py#L579)
  - [`SmartGraphicsView._hit_test_obb_handles(self, pts_pixel, pos_pixel)`](../utils/smart_view.py#L639)
  - [`SmartGraphicsView._max_height_inside(self, x1, y1, x2, y2, dir_x, dir_y)`](../utils/smart_view.py#L675)
  - [`SmartGraphicsView._compute_obb_preview_params(self)`](../utils/smart_view.py#L706)
  - [`SmartGraphicsView.wheelEvent(self, event)`](../utils/smart_view.py#L752)
  - [`SmartGraphicsView.mousePressEvent(self, event)`](../utils/smart_view.py#L758)
  - [`SmartGraphicsView.mouseMoveEvent(self, event)`](../utils/smart_view.py#L919)
  - [`SmartGraphicsView.mouseReleaseEvent(self, event)`](../utils/smart_view.py#L1273)
  - [`SmartGraphicsView.mouseDoubleClickEvent(self, event)`](../utils/smart_view.py#L1304)
  - [`SmartGraphicsView._try_segment_double_click(self, pos_pixel)`](../utils/smart_view.py#L1335)
  - [`SmartGraphicsView._apply_segment_change(self, idx, norm_flat, log_msg)`](../utils/smart_view.py#L1441)
  - [`SmartGraphicsView.leaveEvent(self, event)`](../utils/smart_view.py#L1469)
  - [`SmartGraphicsView.contextMenuEvent(self, event)`](../utils/smart_view.py#L1474)
  - [`SmartGraphicsView._start_drawing(self, event)`](../utils/smart_view.py#L1489)
  - [`SmartGraphicsView._update_segment_cursor(self, event)`](../utils/smart_view.py#L1539)
  - [`SmartGraphicsView._update_segment_line(self, event)`](../utils/smart_view.py#L1547)
  - [`SmartGraphicsView._update_obb_draw(self, event)`](../utils/smart_view.py#L1567)
  - [`SmartGraphicsView._continue_drawing(self, event)`](../utils/smart_view.py#L1575)
  - [`SmartGraphicsView._finish_drawing(self, event)`](../utils/smart_view.py#L1590)
  - [`SmartGraphicsView._finish_segment(self)`](../utils/smart_view.py#L1605)
  - [`SmartGraphicsView._ensure_min_polygon_points(self, pts)`](../utils/smart_view.py#L1629)
  - [`SmartGraphicsView._finish_segment_line(self)`](../utils/smart_view.py#L1660)
  - [`SmartGraphicsView._finish_obb(self)`](../utils/smart_view.py#L1703)
  - [`SmartGraphicsView._reset_obb_draw(self)`](../utils/smart_view.py#L1723)
  - [`SmartGraphicsView._add_scribble(self, x, y, temp=False)`](../utils/smart_view.py#L1732)
  - [`SmartGraphicsView.normalize_angle_rad(self, angle)`](../utils/smart_view.py#L1738)
  - [`SmartGraphicsView.drawForeground(self, painter: QPainter, rect)`](../utils/smart_view.py#L1749)
  - [`SmartGraphicsView._draw_foreground_impl(self, painter: QPainter, rect)`](../utils/smart_view.py#L1755)

<a id="utils-aeb-auto-aeb-def"></a>

## utils_aeb.auto-aeb_def

Исходник: [utils_aeb/auto-aeb_def.py](../utils_aeb/auto-aeb_def.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_def.py#L69) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_def.py#L70)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_def.py#L78)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_def.py#L85)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_def.py#L90)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_def.py#L107) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_def.py#L110)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_def.py#L134)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_def.py#L146)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_def.py#L149)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_def.py#L162)
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_def.py#L174) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def.py#L175)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L188)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_def.py#L192) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def.py#L193)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_def.py#L206)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_def.py#L210) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L211)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L221)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_def.py#L229) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L230)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_def.py#L242)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_def.py#L249)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def.py#L259)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_def.py#L265)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_def.py#L271)
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def.py#L278)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def.py#L330)
- [`main()`](../utils_aeb/auto-aeb_def.py#L370)

<a id="utils-aeb-auto-aeb-def-new"></a>

## utils_aeb.auto-aeb_def_new

Исходник: [utils_aeb/auto-aeb_def_new.py](../utils_aeb/auto-aeb_def_new.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_def_new.py#L82) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_def_new.py#L83)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_def_new.py#L91)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_def_new.py#L98)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_def_new.py#L103)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_def_new.py#L120) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_def_new.py#L123)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_def_new.py#L147)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_def_new.py#L159)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_def_new.py#L162)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_def_new.py#L175)
- [`_gn(channels, max_groups=8)`](../utils_aeb/auto-aeb_def_new.py#L187) — GroupNorm с автоматическим подбором числа групп (устойчив к малому batch).
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_def_new.py#L195) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def_new.py#L198)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L211)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_def_new.py#L215) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_def_new.py#L216)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_def_new.py#L229)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_def_new.py#L233) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L234)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L244)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_def_new.py#L252) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L253)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_def_new.py#L265)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_def_new.py#L272)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_def_new.py#L283)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_def_new.py#L289)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_def_new.py#L295) — FIX: линейный KL-annealing 0 → β_target за warmup_epochs эпох.
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def_new.py#L303)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_def_new.py#L356)
- [`main()`](../utils_aeb/auto-aeb_def_new.py#L397)

<a id="utils-aeb-auto-aeb-old"></a>

## utils_aeb.auto-aeb_old

Исходник: [utils_aeb/auto-aeb_old.py](../utils_aeb/auto-aeb_old.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_old.py#L71) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_old.py#L72)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_old.py#L80)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_old.py#L87)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_old.py#L92)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_old.py#L109) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_old.py#L112)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_old.py#L136)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_old.py#L148)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_old.py#L151)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_old.py#L164)
- Класс [`ConvEncoder`](../utils_aeb/auto-aeb_old.py#L176) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_old.py#L177)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L190)
- Класс [`ConvDecoder`](../utils_aeb/auto-aeb_old.py#L194) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/auto-aeb_old.py#L195)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/auto-aeb_old.py#L208)
- Класс [`ConvAutoencoder`](../utils_aeb/auto-aeb_old.py#L212) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L213)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L223)
- Класс [`ConvVAE`](../utils_aeb/auto-aeb_old.py#L231) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L232)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_old.py#L244)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/auto-aeb_old.py#L251)
- [`build_model(model_type, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/auto-aeb_old.py#L261)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_old.py#L267)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_old.py#L273)
- [`train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_old.py#L280)
- [`evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_old.py#L332)
- [`main()`](../utils_aeb/auto-aeb_old.py#L372)

<a id="utils-aeb-auto-aeb-unet"></a>

## utils_aeb.auto-aeb_unet

Исходник: [utils_aeb/auto-aeb_unet.py](../utils_aeb/auto-aeb_unet.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`.

**Объявления:**

- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_unet.py#L74) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_unet.py#L75)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_unet.py#L83)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_unet.py#L90)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_unet.py#L95)
- Класс [`GradientLoss`](../utils_aeb/auto-aeb_unet.py#L112) (`nn.Module`)
  - [`GradientLoss.__init__(self)`](../utils_aeb/auto-aeb_unet.py#L113)
  - [`GradientLoss.forward(self, pred, target)`](../utils_aeb/auto-aeb_unet.py#L122)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_unet.py#L134) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0)`](../utils_aeb/auto-aeb_unet.py#L137)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_unet.py#L165)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_unet.py#L177)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_unet.py#L180)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_unet.py#L193)
- Класс [`UNetEncoder`](../utils_aeb/auto-aeb_unet.py#L207) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/auto-aeb_unet.py#L208)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L237)
- Класс [`UNetDecoder`](../utils_aeb/auto-aeb_unet.py#L245) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/auto-aeb_unet.py#L246)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/auto-aeb_unet.py#L280)
- Класс [`UNetAE`](../utils_aeb/auto-aeb_unet.py#L287) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet.py#L288)
  - [`UNetAE.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L300)
- Класс [`UNetVAE`](../utils_aeb/auto-aeb_unet.py#L308) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet.py#L309)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_unet.py#L322)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/auto-aeb_unet.py#L329)
- [`build_model(model_type, in_channels, latent_channels, img_h, img_w)`](../utils_aeb/auto-aeb_unet.py#L339)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_unet.py#L345)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_unet.py#L352)
- [`_compute_recon_loss(recon, x, criterion_l1, criterion_grad, criterion_ssim)`](../utils_aeb/auto-aeb_unet.py#L359)
- [`train_one_epoch(model, loader, optimizer, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet.py#L367)
- [`evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet.py#L414)
- [`main()`](../utils_aeb/auto-aeb_unet.py#L450)

<a id="utils-aeb-auto-aeb-unet-with-grad"></a>

## utils_aeb.auto-aeb_unet_with_grad

Исходник: [utils_aeb/auto-aeb_unet_with_grad.py](../utils_aeb/auto-aeb_unet_with_grad.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `glob`, `hashlib`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `random`, `time`, `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`.

**Объявления:**

- [`_to_gray(x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L114)
- [`median_blur(x, k=3)`](../utils_aeb/auto-aeb_unet_with_grad.py#L120) — x: [B,C,H,W]. Медиана через unfold.
- [`_edge_kernels(kind)`](../utils_aeb/auto-aeb_unet_with_grad.py#L139)
- [`edge_map(x, lo=EDGE_LO, hi=EDGE_HI, kind=EDGE_KERNEL, med=MEDIAN_KERNEL)`](../utils_aeb/auto-aeb_unet_with_grad.py#L147) — x: [B,C,H,W] в [0,1] → [B,1,H,W] мягкая карта рёбер.
- Класс [`SSIMLoss`](../utils_aeb/auto-aeb_unet_with_grad.py#L164) (`nn.Module`)
  - [`SSIMLoss.__init__(self, window_size=11, channel=None)`](../utils_aeb/auto-aeb_unet_with_grad.py#L165)
  - [`SSIMLoss._gaussian(window_size, sigma)`](../utils_aeb/auto-aeb_unet_with_grad.py#L173)
  - [`SSIMLoss._create_window(self, window_size, channel)`](../utils_aeb/auto-aeb_unet_with_grad.py#L180)
  - [`SSIMLoss.forward(self, img1, img2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L185)
- Класс [`GradientLoss`](../utils_aeb/auto-aeb_unet_with_grad.py#L201) (`nn.Module`)
  - [`GradientLoss.__init__(self)`](../utils_aeb/auto-aeb_unet_with_grad.py#L202)
  - [`GradientLoss.forward(self, pred, target)`](../utils_aeb/auto-aeb_unet_with_grad.py#L211)
- [`d_hinge_loss(d_real, d_fake)`](../utils_aeb/auto-aeb_unet_with_grad.py#L222)
- [`g_hinge_loss(d_fake)`](../utils_aeb/auto-aeb_unet_with_grad.py#L226)
- Класс [`ImageFolderDataset`](../utils_aeb/auto-aeb_unet_with_grad.py#L231) (`Dataset`)
  - [`ImageFolderDataset.__init__(self, paths, img_h, img_w, color_mode, augment, keep_aspect=True, pad_value=0, noise_std=0.0, use_edge=False, edge_cache_dir=None)`](../utils_aeb/auto-aeb_unet_with_grad.py#L234)
  - [`ImageFolderDataset._fit(self, img)`](../utils_aeb/auto-aeb_unet_with_grad.py#L258)
  - [`ImageFolderDataset._load_resized_tensor(self, path)`](../utils_aeb/auto-aeb_unet_with_grad.py#L270)
  - [`ImageFolderDataset._edge_cache_path(self, img_path)`](../utils_aeb/auto-aeb_unet_with_grad.py#L277)
  - [`ImageFolderDataset._load_or_build_edge(self, img_path, img_t)`](../utils_aeb/auto-aeb_unet_with_grad.py#L297)
  - [`ImageFolderDataset._augment_geom(self, img_t, edge_t)`](../utils_aeb/auto-aeb_unet_with_grad.py#L321)
  - [`ImageFolderDataset.__len__(self)`](../utils_aeb/auto-aeb_unet_with_grad.py#L353)
  - [`ImageFolderDataset.__getitem__(self, i)`](../utils_aeb/auto-aeb_unet_with_grad.py#L356)
- [`collect_images(root, exclude_names=('loss.png', 'recon_examples.png', 'best.pt'))`](../utils_aeb/auto-aeb_unet_with_grad.py#L382)
- [`prewarm_edge_cache(dataset)`](../utils_aeb/auto-aeb_unet_with_grad.py#L394) — Считает и сохраняет edge-карты для всех сэмплов датасета (только новые).
- [`_norm(c, kind=NORM_TYPE)`](../utils_aeb/auto-aeb_unet_with_grad.py#L421) — GroupNorm с безопасным числом групп (c % g == 0).
- [`_act()`](../utils_aeb/auto-aeb_unet_with_grad.py#L431)
- Класс [`ConvBlock`](../utils_aeb/auto-aeb_unet_with_grad.py#L435) (`nn.Module`)
  - [`ConvBlock.__init__(self, in_c, out_c, stride=1, n_convs=2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L436)
  - [`ConvBlock.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L448)
- Класс [`DilatedResidual`](../utils_aeb/auto-aeb_unet_with_grad.py#L452) (`nn.Module`)
  - [`DilatedResidual.__init__(self, c, dilations=(2, 4))`](../utils_aeb/auto-aeb_unet_with_grad.py#L453)
  - [`DilatedResidual.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L465)
- Класс [`UNetEncoder`](../utils_aeb/auto-aeb_unet_with_grad.py#L471) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L472)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L481)
- Класс [`DecoderBlock`](../utils_aeb/auto-aeb_unet_with_grad.py#L490) (`nn.Module`)
  - [`DecoderBlock.__init__(self, in_c, skip_c, out_c)`](../utils_aeb/auto-aeb_unet_with_grad.py#L491)
  - [`DecoderBlock.forward(self, x, skip)`](../utils_aeb/auto-aeb_unet_with_grad.py#L501)
- Класс [`UNetDecoder`](../utils_aeb/auto-aeb_unet_with_grad.py#L511) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L512)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/auto-aeb_unet_with_grad.py#L523)
- Класс [`UNetAE`](../utils_aeb/auto-aeb_unet_with_grad.py#L530) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L532)
  - [`UNetAE.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L543)
- Класс [`UNetVAE`](../utils_aeb/auto-aeb_unet_with_grad.py#L551) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/auto-aeb_unet_with_grad.py#L553)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/auto-aeb_unet_with_grad.py#L565)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L571)
- Класс [`PatchDiscriminator`](../utils_aeb/auto-aeb_unet_with_grad.py#L581) (`nn.Module`)
  - [`PatchDiscriminator.__init__(self, in_ch, base=D_BASE)`](../utils_aeb/auto-aeb_unet_with_grad.py#L582)
  - [`PatchDiscriminator.forward(self, x)`](../utils_aeb/auto-aeb_unet_with_grad.py#L598)
- [`build_model(model_type, in_channels, out_channels, latent_channels, img_h, img_w)`](../utils_aeb/auto-aeb_unet_with_grad.py#L602)
- [`vae_kl_loss(mu, logvar)`](../utils_aeb/auto-aeb_unet_with_grad.py#L609)
- [`get_beta(epoch, warmup_epochs, target_beta)`](../utils_aeb/auto-aeb_unet_with_grad.py#L615)
- [`_edge_weighted_l1(pred, target, w=EDGE_L1_EXTRA_W)`](../utils_aeb/auto-aeb_unet_with_grad.py#L622)
- [`_compute_recon_loss(recon, target, criterion_l1, criterion_grad, criterion_ssim)`](../utils_aeb/auto-aeb_unet_with_grad.py#L630)
- [`train_one_epoch(model, loader, optimizer, D, opt_d, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta, use_gan, gan_weight)`](../utils_aeb/auto-aeb_unet_with_grad.py#L642)
- [`evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim, model_type, device, beta)`](../utils_aeb/auto-aeb_unet_with_grad.py#L719)
- [`main()`](../utils_aeb/auto-aeb_unet_with_grad.py#L756)

<a id="utils-aeb-yolo-aeb-utils"></a>

## utils_aeb.yolo_aeb_utils

Исходник: [utils_aeb/yolo_aeb_utils.py](../utils_aeb/yolo_aeb_utils.py#L1).

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `apply_object_morphology`, `build_detection_mask`, `cls_suffix_from_detection`, `crop_to_detection`, `fill_outside_object`, `limit_image_size`, `longest_edge_angle_deg`, `morph_image_in_mask`, `rotate_with_optional_crop`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `torch`, `torch.nn`, `torch.nn.functional`.

**Объявления:**

- Класс [`ConvEncoder`](../utils_aeb/yolo_aeb_utils.py#L38) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L39)
  - [`ConvEncoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L52)
- Класс [`ConvDecoder`](../utils_aeb/yolo_aeb_utils.py#L56) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L57)
  - [`ConvDecoder.forward(self, z)`](../utils_aeb/yolo_aeb_utils.py#L70)
- Класс [`ConvAutoencoder`](../utils_aeb/yolo_aeb_utils.py#L74) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L75)
  - [`ConvAutoencoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L85)
- Класс [`ConvVAE`](../utils_aeb/yolo_aeb_utils.py#L93) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L94)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L106)
  - [`ConvVAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L112)
- [`_gn(channels, max_groups=8)`](../utils_aeb/yolo_aeb_utils.py#L125) — GroupNorm с автоматическим подбором числа групп — как в обучении.
- Класс [`ConvEncoderGN`](../utils_aeb/yolo_aeb_utils.py#L133) (`nn.Module`)
  - [`ConvEncoderGN.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L134)
  - [`ConvEncoderGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L147)
- Класс [`ConvDecoderGN`](../utils_aeb/yolo_aeb_utils.py#L151) (`nn.Module`)
  - [`ConvDecoderGN.__init__(self, in_channels)`](../utils_aeb/yolo_aeb_utils.py#L152)
  - [`ConvDecoderGN.forward(self, z)`](../utils_aeb/yolo_aeb_utils.py#L165)
- Класс [`ConvAutoencoderGN`](../utils_aeb/yolo_aeb_utils.py#L169) (`nn.Module`)
  - [`ConvAutoencoderGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L170)
  - [`ConvAutoencoderGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L180)
- Класс [`ConvVAEGN`](../utils_aeb/yolo_aeb_utils.py#L188) (`nn.Module`)
  - [`ConvVAEGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_aeb/yolo_aeb_utils.py#L189)
  - [`ConvVAEGN.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L201)
  - [`ConvVAEGN.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L207)
- Класс [`UNetEncoder`](../utils_aeb/yolo_aeb_utils.py#L220) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_aeb/yolo_aeb_utils.py#L221)
  - [`UNetEncoder.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L250)
- Класс [`UNetDecoder`](../utils_aeb/yolo_aeb_utils.py#L258) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_aeb/yolo_aeb_utils.py#L259)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_aeb/yolo_aeb_utils.py#L293)
- Класс [`UNetAE`](../utils_aeb/yolo_aeb_utils.py#L300) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/yolo_aeb_utils.py#L301)
  - [`UNetAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L313)
- Класс [`UNetVAE`](../utils_aeb/yolo_aeb_utils.py#L321) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_aeb/yolo_aeb_utils.py#L322)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L335)
  - [`UNetVAE.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L341)
- [`_norm_v2(c, kind='gn')`](../utils_aeb/yolo_aeb_utils.py#L356) — GroupNorm с тем же подбором групп, что и в тренировочном скрипте.
- [`_act_v2()`](../utils_aeb/yolo_aeb_utils.py#L366)
- Класс [`ConvBlockV2`](../utils_aeb/yolo_aeb_utils.py#L370) (`nn.Module`)
  - [`ConvBlockV2.__init__(self, in_c, out_c, stride=1, n_convs=2, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L371)
  - [`ConvBlockV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L383)
- Класс [`DilatedResidualV2`](../utils_aeb/yolo_aeb_utils.py#L387) (`nn.Module`)
  - [`DilatedResidualV2.__init__(self, c, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L388)
  - [`DilatedResidualV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L400)
- Класс [`UNetEncoderV2`](../utils_aeb/yolo_aeb_utils.py#L406) (`nn.Module`)
  - [`UNetEncoderV2.__init__(self, in_channels, base=32, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L407)
  - [`UNetEncoderV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L416)
- Класс [`DecoderBlockV2`](../utils_aeb/yolo_aeb_utils.py#L425) (`nn.Module`)
  - [`DecoderBlockV2.__init__(self, in_c, skip_c, out_c, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L426)
  - [`DecoderBlockV2.forward(self, x, skip)`](../utils_aeb/yolo_aeb_utils.py#L436)
- Класс [`UNetDecoderV2`](../utils_aeb/yolo_aeb_utils.py#L447) (`nn.Module`)
  - [`UNetDecoderV2.__init__(self, out_channels, base=32, norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L448)
  - [`UNetDecoderV2.forward(self, h, f0, f1, f2)`](../utils_aeb/yolo_aeb_utils.py#L459)
- Класс [`UNetAEV2`](../utils_aeb/yolo_aeb_utils.py#L466) (`nn.Module`)
  - [`UNetAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn')`](../utils_aeb/yolo_aeb_utils.py#L467)
  - [`UNetAEV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L482)
- Класс [`UNetVAEV2`](../utils_aeb/yolo_aeb_utils.py#L490) (`nn.Module`)
  - [`UNetVAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn', logvar_min=-10.0, logvar_max=10.0)`](../utils_aeb/yolo_aeb_utils.py#L491)
  - [`UNetVAEV2.reparameterize(self, mu, logvar)`](../utils_aeb/yolo_aeb_utils.py#L510)
  - [`UNetVAEV2.forward(self, x)`](../utils_aeb/yolo_aeb_utils.py#L516)
- [`_median_blur_torch(x, k=3)`](../utils_aeb/yolo_aeb_utils.py#L537) — x: [B,C,H,W]. Медианный фильтр через unfold (совместим с обучением).
- [`compute_edge_map_torch(img_tensor, edge_kernel='scharr', median_kernel=3, edge_lo=0.1, edge_hi=0.35)`](../utils_aeb/yolo_aeb_utils.py#L548) — img_tensor: [C,H,W] или [B,C,H,W], float32 в [0,1].
- [`detect_arch_from_state_dict(state_dict)`](../utils_aeb/yolo_aeb_utils.py#L589) — Определяет архитектуру по ключам state_dict.
- [`detect_arch(cfg, state_dict=None)`](../utils_aeb/yolo_aeb_utils.py#L606) — Определяет архитектуру AE/VAE.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`build_ae_model_from_cfg(cfg, arch=None)`](../utils_aeb/yolo_aeb_utils.py#L652) — Собирает модель AE/VAE по config чекпойнта.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`preprocess_image_for_ae(img_bgr, cfg)`](../utils_aeb/yolo_aeb_utils.py#L712) — Приводит произвольный BGR-снимок к тензору [C, H, W] float32 [0..1],
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

<a id="utils-auto-research-gradient-auto-research"></a>

## utils_auto_research.gradient_auto_research

Исходник: [utils_auto_research/gradient_auto_research.py](../utils_auto_research/gradient_auto_research.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QProgressDialog`, `os`.

**Объявления:**

- Класс [`GradientAutoResearch`](../utils_auto_research/gradient_auto_research.py#L7)
  - [`GradientAutoResearch.__init__(self, parent_window, search_all=False)`](../utils_auto_research/gradient_auto_research.py#L8)
  - [`GradientAutoResearch.run(self)`](../utils_auto_research/gradient_auto_research.py#L12)
  - [`GradientAutoResearch._merge_with_defaults(self, settings)`](../utils_auto_research/gradient_auto_research.py#L122)
  - [`GradientAutoResearch._save_current_ui_state(self)`](../utils_auto_research/gradient_auto_research.py#L138)
  - [`GradientAutoResearch._restore_ui_state(self, state)`](../utils_auto_research/gradient_auto_research.py#L153)
  - [`GradientAutoResearch._default_settings(self)`](../utils_auto_research/gradient_auto_research.py#L167)
  - [`GradientAutoResearch._generate_combinations(self, settings)`](../utils_auto_research/gradient_auto_research.py#L200)
  - [`GradientAutoResearch._generate_method_params(self, method, method_settings)`](../utils_auto_research/gradient_auto_research.py#L295)
  - [`GradientAutoResearch._range_values(self, start, end, step)`](../utils_auto_research/gradient_auto_research.py#L317)
  - [`GradientAutoResearch._params_desc(self, params)`](../utils_auto_research/gradient_auto_research.py#L332)
  - [`GradientAutoResearch._apply_method_params(self, method, params)`](../utils_auto_research/gradient_auto_research.py#L340)

<a id="utils-auto-research-gradient-auto-settings-dialog"></a>

## utils_auto_research.gradient_auto_settings_dialog

Исходник: [utils_auto_research/gradient_auto_settings_dialog.py](../utils_auto_research/gradient_auto_settings_dialog.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `traceback`.

**Объявления:**

- Класс [`GradientAutoSettingsDialog`](../utils_auto_research/gradient_auto_settings_dialog.py#L20) (`QDialog`)
  - [`GradientAutoSettingsDialog.__init__(self, parent=None, current_settings=None)`](../utils_auto_research/gradient_auto_settings_dialog.py#L21)
  - [`GradientAutoSettingsDialog.log(self, message)`](../utils_auto_research/gradient_auto_settings_dialog.py#L73)
  - [`GradientAutoSettingsDialog._safe_name(self, name)`](../utils_auto_research/gradient_auto_settings_dialog.py#L82)
  - [`GradientAutoSettingsDialog._default_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L85)
  - [`GradientAutoSettingsDialog._setup_tabs(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L130)
  - [`GradientAutoSettingsDialog._add_method_tab(self, method_name)`](../utils_auto_research/gradient_auto_settings_dialog.py#L149)
  - [`GradientAutoSettingsDialog._create_param_widget(self, value, is_step=False)`](../utils_auto_research/gradient_auto_settings_dialog.py#L192)
  - [`GradientAutoSettingsDialog._add_binarization_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L212)
  - [`GradientAutoSettingsDialog._add_object_filtering_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L236)
  - [`GradientAutoSettingsDialog._add_morphology_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L285)
  - [`GradientAutoSettingsDialog._add_invert_tab(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L365)
  - [`GradientAutoSettingsDialog._load_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L383)
  - [`GradientAutoSettingsDialog.get_settings(self)`](../utils_auto_research/gradient_auto_settings_dialog.py#L452)

<a id="utils-auto-research-threshold-auto-research"></a>

## utils_auto_research.threshold_auto_research

Исходник: [utils_auto_research/threshold_auto_research.py](../utils_auto_research/threshold_auto_research.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QProgressDialog`, `os`.

**Объявления:**

- Класс [`AutoResearch`](../utils_auto_research/threshold_auto_research.py#L7)
  - [`AutoResearch.__init__(self, main_window, search_all=False)`](../utils_auto_research/threshold_auto_research.py#L8)
  - [`AutoResearch.run(self)`](../utils_auto_research/threshold_auto_research.py#L12)
  - [`AutoResearch._default_settings(self)`](../utils_auto_research/threshold_auto_research.py#L156)
  - [`AutoResearch._generate_combinations(self, settings)`](../utils_auto_research/threshold_auto_research.py#L183)
  - [`AutoResearch._generate_binarization_combos(self, settings)`](../utils_auto_research/threshold_auto_research.py#L235)
  - [`AutoResearch._apply_method_params(self, method, params)`](../utils_auto_research/threshold_auto_research.py#L355)

<a id="utils-auto-research-threshold-auto-settings-dialog"></a>

## utils_auto_research.threshold_auto_settings_dialog

Исходник: [utils_auto_research/threshold_auto_settings_dialog.py](../utils_auto_research/threshold_auto_settings_dialog.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`AutoSettingsDialog`](../utils_auto_research/threshold_auto_settings_dialog.py#L16) (`QDialog`)
  - [`AutoSettingsDialog.__init__(self, parent=None, current_settings=None)`](../utils_auto_research/threshold_auto_settings_dialog.py#L17)
  - [`AutoSettingsDialog._safe_name(self, name)`](../utils_auto_research/threshold_auto_settings_dialog.py#L48)
  - [`AutoSettingsDialog._default_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L52)
  - [`AutoSettingsDialog._setup_tabs(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L78)
  - [`AutoSettingsDialog._add_simple_threshold_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L91)
  - [`AutoSettingsDialog._add_otsu_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L115)
  - [`AutoSettingsDialog._add_triangle_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L125)
  - [`AutoSettingsDialog._add_adaptive_tab(self, method_name)`](../utils_auto_research/threshold_auto_settings_dialog.py#L135)
  - [`AutoSettingsDialog._add_niblack_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L170)
  - [`AutoSettingsDialog._add_sauvola_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L204)
  - [`AutoSettingsDialog._add_isodata_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L247)
  - [`AutoSettingsDialog._add_background_symmetry_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L270)
  - [`AutoSettingsDialog._add_row_adaptive_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L293)
  - [`AutoSettingsDialog._add_morphology_tab(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L325)
  - [`AutoSettingsDialog._load_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L397)
  - [`AutoSettingsDialog.get_settings(self)`](../utils_auto_research/threshold_auto_settings_dialog.py#L493)

<a id="utils-new-DatasetGenerator-new"></a>

## utils_new.DatasetGenerator_new

Исходник: [utils_new/DatasetGenerator_new.py](../utils_new/DatasetGenerator_new.py#L1).

**Прямые потребители:** [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tests.test_processing_smoke](#tests-test-processing-smoke).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `AnnotationSample`, `AugmentationPlan`, `Flip`, `ImageSize`, `MaskEncoding`, `Resize`, `TransformPlan`, `apply_augmentation_plan`, `build_augmentation_plan`, `convert_annotation`, `mask_to_annotations`, `polygon_area`, `remap_mask_indices`, `transform_sample`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `sanitize_annotation_topology`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `make_index_mask`, `read_image`, `read_index_mask`, `read_yolo_labels`, `write_classes_and_yaml`, `write_coco_split`, `write_image`, `write_index_mask`, `write_yolo_labels`
- [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new): `mask_statistics`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.pyqtSignal`, `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `hashlib`, `json`, `numpy`, `os`, `pathlib.Path`, `random`, `shutil`, `tempfile`, `traceback`, `typing.Callable`.

**Объявления:**

- Класс [`_Generated`](../utils_new/DatasetGenerator_new.py#L62)
- Класс [`DatasetGeneratorCoreNew`](../utils_new/DatasetGenerator_new.py#L73)
  - [`DatasetGeneratorCoreNew.__init__(self, images_folder, labels_folder, output_folder, pairs, train_pct, val_pct, test_pct, target_size, augmentations, has_augmentations, aug_multiplier, class_names, class_mapping, bg_color, input_type, masks_folder, num_classes, mask_class_remap, included_orig_ids=None, included_class_ids=None, total_orig_classes=1, resize_mode='fixed', include_empty_annotations=False, delete_source_after_success=False, input_format=FORMAT_HINT_AUTO, out_yolo_txt=True, out_yolo_subtype=SUBTYPE_KEEP, out_yolo_obb_format='corners', out_masks=False, out_masks_offset=1, out_unet_yaml=False, out_coco=False, mask_pixel_offset_in=1, min_ann_area_px=0, *, random_seed=None, split_by_source=False, obb_policy='promote_to_segment', hole_policy='error', mixed_output_policy='normalize_segment', strict_complete=False, existing_output_policy='merge', log_cb: Callable[[str], None] &#124; None=None, progress_cb: Callable[[int, int], None] &#124; None=None)`](../utils_new/DatasetGenerator_new.py#L84)
  - [`DatasetGeneratorCoreNew.log(self, message)`](../utils_new/DatasetGenerator_new.py#L160)
  - [`DatasetGeneratorCoreNew.cancel(self)`](../utils_new/DatasetGenerator_new.py#L163)
  - [`DatasetGeneratorCoreNew._check_cancelled(self)`](../utils_new/DatasetGenerator_new.py#L166)
  - [`DatasetGeneratorCoreNew._validate_config(self)`](../utils_new/DatasetGenerator_new.py#L170)
  - [`DatasetGeneratorCoreNew._dense_map(mapping)`](../utils_new/DatasetGenerator_new.py#L209)
  - [`DatasetGeneratorCoreNew._map_annotations(self, annotations)`](../utils_new/DatasetGenerator_new.py#L216)
  - [`DatasetGeneratorCoreNew._map_mask(self, mask)`](../utils_new/DatasetGenerator_new.py#L226)
  - [`DatasetGeneratorCoreNew._verify_mask_export(self, mask, mask_path, *, has_objects: bool)`](../utils_new/DatasetGenerator_new.py#L236)
  - [`DatasetGeneratorCoreNew._filter_annotations(self, anns)`](../utils_new/DatasetGenerator_new.py#L268)
  - [`DatasetGeneratorCoreNew._filter_mask(self, mask)`](../utils_new/DatasetGenerator_new.py#L279)
  - [`DatasetGeneratorCoreNew._resize(self, sample)`](../utils_new/DatasetGenerator_new.py#L294)
  - [`DatasetGeneratorCoreNew._augmentation_impossible(self, size)`](../utils_new/DatasetGenerator_new.py#L303)
  - [`DatasetGeneratorCoreNew._make_variant(self, sample, *, augmented=False, require_operation=False)`](../utils_new/DatasetGenerator_new.py#L320)
  - [`DatasetGeneratorCoreNew._same_sample(a, b)`](../utils_new/DatasetGenerator_new.py#L349)
  - [`DatasetGeneratorCoreNew._variant_digest(sample)`](../utils_new/DatasetGenerator_new.py#L364)
  - [`DatasetGeneratorCoreNew._get_name(self, img_path)`](../utils_new/DatasetGenerator_new.py#L385)
  - [`DatasetGeneratorCoreNew._reserve_stem(self, proposed)`](../utils_new/DatasetGenerator_new.py#L395)
  - [`DatasetGeneratorCoreNew._sanitize_input_annotations(self, annotations, image_path)`](../utils_new/DatasetGenerator_new.py#L404)
  - [`DatasetGeneratorCoreNew._prepare_sample(self, pair)`](../utils_new/DatasetGenerator_new.py#L427)
  - [`DatasetGeneratorCoreNew._generate_pair(self, pair, stage, generated, target_count)`](../utils_new/DatasetGenerator_new.py#L465)
  - [`DatasetGeneratorCoreNew._ensure_consistent_yolo_kinds(self, records)`](../utils_new/DatasetGenerator_new.py#L640)
  - [`DatasetGeneratorCoreNew._assign_splits(self, records)`](../utils_new/DatasetGenerator_new.py#L675)
  - [`DatasetGeneratorCoreNew._distribute_and_export(self, stage, records)`](../utils_new/DatasetGenerator_new.py#L700)
  - [`DatasetGeneratorCoreNew._publish(self, stage, destination)`](../utils_new/DatasetGenerator_new.py#L738)
  - [`DatasetGeneratorCoreNew._delete_processed_sources(self)`](../utils_new/DatasetGenerator_new.py#L813)
  - [`DatasetGeneratorCoreNew.execute(self)`](../utils_new/DatasetGenerator_new.py#L834)
- Класс [`_Cancelled`](../utils_new/DatasetGenerator_new.py#L938) (`Exception`)
- Класс [`DatasetGeneratorThread`](../utils_new/DatasetGenerator_new.py#L949) (`QThread`)
  - [`DatasetGeneratorThread.__init__(self, *args, **kwargs)`](../utils_new/DatasetGenerator_new.py#L955)
  - [`DatasetGeneratorThread.cancel(self)`](../utils_new/DatasetGenerator_new.py#L961)
  - [`DatasetGeneratorThread.run(self)`](../utils_new/DatasetGenerator_new.py#L964)
- Класс [`DatasetGeneratorThread`](../utils_new/DatasetGenerator_new.py#L968)
  - [`DatasetGeneratorThread.__init__(self, *args, **kwargs)`](../utils_new/DatasetGenerator_new.py#L969)

<a id="utils-new-annotation-processing"></a>

## utils_new.annotation_processing

Исходник: [utils_new/annotation_processing.py](../utils_new/annotation_processing.py#L1).

**Прямые потребители:** [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.segmentation.results](#ui-segmentation-results), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils), [utils_new.dataset_view_new](#utils-new-dataset-view-new), [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `MorphologyStep`, `apply_morphology as apply_shared_morphology`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `math`, `numpy`, `shapely.geometry.Polygon`, `typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`.

**Объявления:**

- [`class_id_for_annotation(annotation)`](../utils_new/annotation_processing.py#L35) — Класс отсутствует/None/отрицательный -> 0 (общая политика V4).
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`normalized_annotation(annotation)`](../utils_new/annotation_processing.py#L51) — Добавляет класс 0 к устаревшим записям без класса, не меняя геометрию.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`polygon_for_annotation(annotation, width: int, height: int)`](../utils_new/annotation_processing.py#L77) — N×2 float64 вершины в пикселях исходного изображения, без округления.
- [`crop_bounds_for_annotation(annotation, image_w: int, image_h: int, padding: int=10)`](../utils_new/annotation_processing.py#L111) — Pixel-exclusive (x1,y1,x2,y2), обрезанные по границам снимка.
- [`crop_for_annotation(image: np.ndarray, annotation, padding: int=10)`](../utils_new/annotation_processing.py#L127) — Возвращает (copy ROI без ресайза, bounds); не меняет входной массив.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`annotation_in_crop(annotation, original_w: int, original_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L139) — Пересчитывает аннотацию в координаты кропа для векторного Qt-слоя.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`mask_roi_at_image_scale(mask: np.ndarray, image_w: int, image_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L169) — Извлекает маску точно для ROI без ресайза всей маски к размеру снимка.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`apply_class_mask_overlay(image: np.ndarray, mask: Optional[np.ndarray], opacity: float, color_for_class, *, pixel_offset: int=1, background_value: int=0, annotations=None, restrict_to_annotations=False)`](../utils_new/annotation_processing.py#L189) — Показывает индексную маску цветами палитры классов (BGR).
- [`_shapely_polygon(points)`](../utils_new/annotation_processing.py#L285)
- [`_raster_intersection_area(a: np.ndarray, b: np.ndarray, rect: Tuple[float, float, float, float], max_pixels=2000000)`](../utils_new/annotation_processing.py#L297) — Fallback для невыпуклых полигонов без shapely.
- [`overlap_ratio_of_smaller(annotation_a, annotation_b, width: int, height: int)`](../utils_new/annotation_processing.py#L321) — Пересечение / площадь МЕНЬШЕГО объекта, в интервале [0,1].
- [`find_strong_overlaps(annotations, width: int, height: int, threshold: float=0.7)`](../utils_new/annotation_processing.py#L366) — Сопоставляет индексы аннотаций с перекрывающими их соседями.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`detection_to_unified(detection, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L400) — Return one normalized annotation for a YOLO+AE/VAE detection dict.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`detections_to_unified(detections, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L445) — Convert all detections, keeping list positions stable for Qt selection.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`_name_for_class(names, class_id)`](../utils_new/annotation_processing.py#L461)
- [`_as_numpy(value)`](../utils_new/annotation_processing.py#L470) — Accept Ultralytics tensors and plain numpy arrays for standalone tests.
- [`yolo_result_to_detections(result, names=None, *, angle_fn=None)`](../utils_new/annotation_processing.py#L481) — Decode YOLO detect/OBB/segment into the existing AE/VAE dict contract.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`prepare_annotation_preview(image, annotation, padding=10, *, mask=None, opacity=0.5, color_for_class=None, pixel_offset=1, background_value=0)`](../utils_new/annotation_processing.py#L564) — Single crop pipeline used by the dataset viewer and YOLO-side previews.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`render_unified_annotations(image, annotations, color_for_class, *, selected_index=-1, selected_color=(0, 255, 255), text_color=(255, 255, 255), thickness=1, font_scale=0.5)`](../utils_new/annotation_processing.py#L589) — Optional OpenCV renderer for exporting detection imagery.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`render_ae_error_map(error, base, *, valid_region=None, overlay=True, alpha=0.5, show_mask=False, threshold=float('nan'))`](../utils_new/annotation_processing.py#L621) — Render an AE/VAE error heatmap with the original percentile stretch.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`longest_edge_angle_deg(corners)`](../utils_new/annotation_processing.py#L667)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`build_detection_mask(det, W, H)`](../utils_new/annotation_processing.py#L682) — Бинарная маска (uint8, 0/255) для детекции.
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`morph_image_in_mask(img, mask, erosion_px, dilation_px, shape_name='Rectangle', contour_only=True)`](../utils_new/annotation_processing.py#L704) — Применяет эрозию и/или дилатацию к изображению, сохраняя результат
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`apply_object_morphology(img, det, erosion_px=1, dilation_px=1, shape_name='Rectangle', contour_only=True, *, operations=None, contour_width_px=None)`](../utils_new/annotation_processing.py#L757) — Морфология изображения внутри выбранной YOLO-детекции.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`fill_outside_object(img, det, fill_mode, rotation_matrix=None, crop_offset=(0, 0))`](../utils_new/annotation_processing.py#L811) — Заливает пиксели вне маски объекта (fill_mode: 'none' / 'black' / 'white').
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`crop_to_detection(img, det, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L854) — Обрезает изображение по xyxy объекта с учётом padding и смещений.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`rotate_with_optional_crop(img, det, auto_crop, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L871) — Доворачивает изображение так, чтобы длинная сторона объекта стала
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`cls_suffix_from_detection(det, index=None)`](../utils_new/annotation_processing.py#L913) — Безопасный суффикс имени файла по классу объекта и его номеру в списке.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`limit_image_size(img, max_side)`](../utils_new/annotation_processing.py#L937) — Пропорционально уменьшает изображение, если длинная сторона > max_side.
  Явно импортируют: [utils_aeb.yolo_aeb_utils](#utils-aeb-yolo-aeb-utils).
- [`aeb_transform_image(image, detection, *, erode=1, dilate=1, morphology_steps=None, contour_width_px=None, shape_name='Rectangle', contour_only=True, auto_crop=False, auto_rotate=False, padding=0, off_x=0, off_y=0, fine_angle=0.0, fill_mode='none', manual_ops=(), invert=False)`](../utils_new/annotation_processing.py#L951) — Common sequential image pipeline for live preview and save-all.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`annotation_to_detection(annotation, image_w, image_h, *, class_name=None)`](../utils_new/annotation_processing.py#L1014) — Build an AEB-compatible detection dict from normalized YOLO annotation.
- [`transform_annotation_image(image, annotation, **kwargs)`](../utils_new/annotation_processing.py#L1058) — Run the same AEB pipeline for a label from a saved dataset.
- [`_mask_extraction_dimensions(binary: np.ndarray, image_shape=None, min_area=None)`](../utils_new/annotation_processing.py#L1075) — Площадь отсечения — та же, что в старом get_display_params().
- [`extract_segments_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1093) — Segment: внешние контуры CHAIN_APPROX_SIMPLE, нормализованные вершины.
- [`extract_boxes_from_mask(binary: np.ndarray, *, image_shape=None, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1120) — Detect: connectedComponentsWithStats, 8-связность, как в старом коде.
- [`extract_obb_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1143) — OBB: minAreaRect + boxPoints, с исходным усечением до int32.
- [`extract_annotations_from_mask(binary: np.ndarray, mode='detect', *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1165) — Единая точка получения Detect/OBB/Segment из бинарной маски.
  Явно импортируют: [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing).
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_new/annotation_processing.py#L1187) — Форматирует аннотацию для отображения в QListWidget.
  Явно импортируют: [ui.segmentation.results](#ui-segmentation-results).
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_new/annotation_processing.py#L1216) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).
  Явно импортируют: [ui.segmentation.results](#ui-segmentation-results).

<a id="utils-new-annotations---init--"></a>

## utils_new.annotations.__init__

Исходник: [utils_new/annotations/__init__.py](../utils_new/annotations/__init__.py#L1).

**Прямые потребители:** [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

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

- [`from_legacy(value, size: ImageSize, *, allow_classless=False)`](../utils_new/annotations/adapters.py#L9) — Legacy tuple -> pixel Annotation; strict by default for dataset export.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_legacy(ann: Annotation, size: ImageSize)`](../utils_new/annotations/adapters.py#L55) — Canonical Annotation -> original normalized tuple; does not round/clamp.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_legacy_many(values, size, *, allow_classless=False)`](../utils_new/annotations/adapters.py#L70)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_legacy_many(values, size)`](../utils_new/annotations/adapters.py#L74)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_detection_dict(detection: dict, size: ImageSize)`](../utils_new/annotations/adapters.py#L78) — YOLO+AE detection dictionary (pixel xyxy, corners or mask_xy) -> Annotation.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`legacy_convert_annotations(values, size: ImageSize, target: str, *, allow_lossy=False, obb_policy='refit')`](../utils_new/annotations/adapters.py#L101) — Explicit bridge for the old converter when migrating DatasetGenerator.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`to_detection_dict(ann: Annotation, *, class_name=None, confidence=None)`](../utils_new/annotations/adapters.py#L114) — Canonical annotation -> YOLO+AE/VAE detection dictionary in pixels.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`from_ultralytics_result(result, size: ImageSize)`](../utils_new/annotations/adapters.py#L145) — Duck-typed Ultralytics result -> canonical annotations; no torch import.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-augmentation"></a>

## utils_new.annotations.augmentation

Исходник: [utils_new/annotations/augmentation.py](../utils_new/annotations/augmentation.py#L1).

**Прямые потребители:** [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.annotations.__init__](#utils-new-annotations---init--).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `AnnotationSample`, `ImageSize`
- [utils_new.annotations.transforms](#utils-new-annotations-transforms): `Crop`, `Flip`, `Resize`, `Rotate`, `Rotate90`, `Shear`, `TransformPlan`, `transform_sample`

**Внешние/stdlib импорты:** `__future__.annotations`, `dataclasses.dataclass`, `math`, `numpy`, `random`.

**Объявления:**

- Класс [`AugmentationPlan`](../utils_new/annotations/augmentation.py#L24)
- [`_next_size(size: ImageSize, op)`](../utils_new/annotations/augmentation.py#L29)
- [`build_augmentation_plan(size: ImageSize, config: dict, *, seed=None, rng=None)`](../utils_new/annotations/augmentation.py#L51) — Old-style config accepted; repeatable when seed is fixed.
  Явно импортируют: [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`apply_augmentation_plan(sample: AnnotationSample, plan: AugmentationPlan, **kwargs)`](../utils_new/annotations/augmentation.py#L95)
  Явно импортируют: [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-geometry"></a>

## utils_new.annotations.geometry

Исходник: [utils_new/annotations/geometry.py](../utils_new/annotations/geometry.py#L1).

**Прямые потребители:** [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`, `shapely.geometry.Polygon`, `shapely.geometry.box`, `shapely.validation.make_valid`.

**Объявления:**

- Класс [`GeometryError`](../utils_new/annotations/geometry.py#L20) (`ValueError`)
- Класс [`LossyConversionError`](../utils_new/annotations/geometry.py#L24) (`GeometryError`)
- [`polygon_area(points)`](../utils_new/annotations/geometry.py#L28)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`bounds(points)`](../utils_new/annotations/geometry.py#L36)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`rectangle_points(x0, y0, x1, y1)`](../utils_new/annotations/geometry.py#L41)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`canonical_obb_points(points)`](../utils_new/annotations/geometry.py#L45) — Best-fitting minimum-area rectangle in *pixel* coordinates.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`is_rectangle(points, tol=1e-06)`](../utils_new/annotations/geometry.py#L63)
  Явно импортируют: [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).
- [`is_axis_aligned(points, atol=1e-07)`](../utils_new/annotations/geometry.py#L75)
  Явно импортируют: [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`convert_annotation(ann: Annotation, target: str, *, allow_lossy=False, obb_policy='refit')`](../utils_new/annotations/geometry.py#L81) — Detect/OBB/Segment. Never silently return a different target subtype.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters).
- [`_clip_convex_polygon(points, rect)`](../utils_new/annotations/geometry.py#L109) — Sutherland–Hodgman with float64 coordinates, convex polygons only.
- [`_iter_polygons(geom)`](../utils_new/annotations/geometry.py#L137)
- [`clip_polygon_to_rect(points, rect, *, hole_policy='error', min_area=1e-09)`](../utils_new/annotations/geometry.py#L147) — Exact polygon/ROI intersection; may return multiple components.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`clip_annotation(ann: Annotation, rect, *, min_area=1.0, multi_policy='all', obb_policy='promote_to_segment', hole_policy='error')`](../utils_new/annotations/geometry.py#L182) — Clip against pixel-edge ROI. Coordinate origin is NOT shifted.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
- [`overlap_area(a: Annotation, b: Annotation)`](../utils_new/annotations/geometry.py#L224)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`overlap_ratio_of_smaller(a: Annotation, b: Annotation)`](../utils_new/annotations/geometry.py#L232)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`sanitize_annotation_topology(ann: Annotation, *, hole_policy='error')`](../utils_new/annotations/geometry.py#L237) — Return (valid annotations, diagnostics) before geometric augmentation.
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

- [`_dtype_for_values(values)`](../utils_new/annotations/masks.py#L17)
- [`annotation_binary_mask(ann: Annotation, size: ImageSize)`](../utils_new/annotations/masks.py#L22) — Raster footprint with half-open pixel-center semantics for rectangles.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new).
- [`rasterize_annotations(annotations: list[Annotation], size: ImageSize, *, encoding: MaskEncoding=MaskEncoding(), overlap='last', dtype=None)`](../utils_new/annotations/masks.py#L46) — Rasterize detect/obb/segment to one label per pixel.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`remap_mask_indices(mask: np.ndarray, class_map: dict[int, int], *, source_encoding: MaskEncoding=MaskEncoding(), target_encoding: MaskEncoding=MaskEncoding(), unmapped='background', dtype=None)`](../utils_new/annotations/masks.py#L78) — Remap index image without intermediate value collisions or input mutation.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`mask_roi_at_image_scale(mask: np.ndarray, size: ImageSize, bounds_xyxy)`](../utils_new/annotations/masks.py#L112) — Nearest-neighbor ROI from a mask whose resolution differs from image.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`mask_to_annotations(mask: np.ndarray, *, encoding: MaskEncoding=MaskEncoding(), kind='segment', min_area=1.0, holes='error')`](../utils_new/annotations/masks.py#L127) — Lossy contour extraction from class mask; explicit policy for holes.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).

<a id="utils-new-annotations-model"></a>

## utils_new.annotations.model

Исходник: [utils_new/annotations/model.py](../utils_new/annotations/model.py#L1).

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `dataclasses.dataclass`, `dataclasses.field`, `numpy`, `typing.Literal`.

**Объявления:**

- Класс [`ImageSize`](../utils_new/annotations/model.py#L17)
  - [`ImageSize.__post_init__(self)`](../utils_new/annotations/model.py#L21)
- Класс [`Annotation`](../utils_new/annotations/model.py#L27)
  - [`Annotation.__post_init__(self)`](../utils_new/annotations/model.py#L37)
  - [`Annotation.with_points(self, pts, *, kind=None)`](../utils_new/annotations/model.py#L57)
- Класс [`MaskEncoding`](../utils_new/annotations/model.py#L62)
  - [`MaskEncoding.__post_init__(self)`](../utils_new/annotations/model.py#L72)
- Класс [`AnnotationSample`](../utils_new/annotations/model.py#L82)
  - [`AnnotationSample.__post_init__(self)`](../utils_new/annotations/model.py#L88)
  - [`AnnotationSample.size(self)`](../utils_new/annotations/model.py#L104)
  - [`AnnotationSample.detached(self)`](../utils_new/annotations/model.py#L108)

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

- Класс [`Flip`](../utils_new/annotations/transforms.py#L27)
- Класс [`Rotate90`](../utils_new/annotations/transforms.py#L33)
- Класс [`Rotate`](../utils_new/annotations/transforms.py#L38)
- Класс [`Shear`](../utils_new/annotations/transforms.py#L44)
- Класс [`Crop`](../utils_new/annotations/transforms.py#L51)
- Класс [`Resize`](../utils_new/annotations/transforms.py#L59)
- Класс [`Translate`](../utils_new/annotations/transforms.py#L66)
- Класс [`TransformPlan`](../utils_new/annotations/transforms.py#L73)
  - [`TransformPlan.__init__(self, steps=())`](../utils_new/annotations/transforms.py#L76)
- Класс [`TransformReport`](../utils_new/annotations/transforms.py#L81)
- [`_translate_matrix(x, y)`](../utils_new/annotations/transforms.py#L86)
- [`apply_affine_to_points(points, matrix)`](../utils_new/annotations/transforms.py#L90)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`compose_affine_matrices(matrices)`](../utils_new/annotations/transforms.py#L99) — Input order == application order: [M1,M2] -> M2 @ M1.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--).
- [`_map_annotations(annotations, M, new_size, *, detect_policy, obb_policy, multi_policy, hole_policy, min_area, report, step_name)`](../utils_new/annotations/transforms.py#L112)
- [`_build_canvas_matrix(M, size: ImageSize, expand_canvas)`](../utils_new/annotations/transforms.py#L149)
- [`_pixel_center_matrix(M)`](../utils_new/annotations/transforms.py#L159)
- [`_border_value_for_image(fill, img)`](../utils_new/annotations/transforms.py#L163)
- [`_warp(sample, M, new_size, fill_color)`](../utils_new/annotations/transforms.py#L173)
- [`_resize(sample, step)`](../utils_new/annotations/transforms.py#L189)
- [`_apply_step(sample, step, *, report, detect_policy, obb_policy, multi_policy, hole_policy, min_area, fill_color)`](../utils_new/annotations/transforms.py#L221)
- [`transform_sample(sample: AnnotationSample, plan, *, detect_policy='aabb', obb_policy='promote_to_segment', multi_policy='all', hole_policy='error', min_area=1e-09, fill_color=(0, 0, 0), return_report=False)`](../utils_new/annotations/transforms.py#L282) — Transform image, canonical annotations and optional mask identically.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation).
- [`extract_roi(sample: AnnotationSample, bounds_xyxy, **kwargs)`](../utils_new/annotations/transforms.py#L304)
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

- [`canonicalize_obb_corners(norm_points, width, height)`](../utils_new/dataset_coco_new.py#L18) — Return deterministic corners after fitting in *pixel* coordinates.
- [`_bbox_to_detect(bbox, w, h)`](../utils_new/dataset_coco_new.py#L25) — COCO bbox [x, y, bw, bh] в абс. пикселях → canonical detect.
- [`_segmentation_to_segment(seg, w, h)`](../utils_new/dataset_coco_new.py#L35) — COCO segmentation: [poly1, poly2, ...], каждый poly — плоский список абс. координат.
- [`_is_polygon_rectangle(poly_abs, tol=2.0)`](../utils_new/dataset_coco_new.py#L55) — Эвристика: 4 точки, стороны примерно параллельны друг другу,
- [`load_coco(json_path, log_cb=None, *, preserve_category_ids=False)`](../utils_new/dataset_coco_new.py#L84) — Загружает COCO JSON.
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`write_yolo_txt_for_coco_items(items, output_labels_dir, class_remap=None, log_cb=None, name_prefix='')`](../utils_new/dataset_coco_new.py#L213) — COCO canonical tuples -> shared YOLO writer, preserving source image pairing.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

<a id="utils-new-dataset-files-new"></a>

## utils_new.dataset_files_new

Исходник: [utils_new/dataset_files_new.py](../utils_new/dataset_files_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.export](#ui-segmentation-export), [utils.image_io](#utils-image-io), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).

**Прямые локальные импорты:**
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`

**Внешние/stdlib импорты:** `PIL.Image`, `collections.OrderedDict`, `concurrent.futures.ThreadPoolExecutor`, `cv2`, `numpy`, `pathlib.Path`, `threading`.

**Объявления:**

- [`_read_with_fallback(path, flags)`](../utils_new/dataset_files_new.py#L9) — Read image/mask with Unicode-path fallback (notably Windows Cyrillic).
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image(path)`](../utils_new/dataset_files_new.py#L23) — Read BGR image, keeping existing generator's 3-channel convention.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`normalize_index_mask_shape(mask, *, path=None)`](../utils_new/dataset_files_new.py#L32) — Normalize equivalent OpenCV single-channel shapes to (H, W).
- [`read_index_mask(path)`](../utils_new/dataset_files_new.py#L52) — Read indexed PNG as 2D integer array; accept decoder's (H,W,1).
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`write_image(path, image)`](../utils_new/dataset_files_new.py#L61) — Write image safely also when cv2.imwrite cannot handle the path.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image_with_fallback(path)`](../utils_new/dataset_files_new.py#L82) — Read image with original depth, TIFF and Unicode-path support.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).
- [`read_mask_with_fallback(path)`](../utils_new/dataset_files_new.py#L119) — Nullable indexed-mask reader; never squeeze a real RGB mask.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`image_size_from_file(path)`](../utils_new/dataset_files_new.py#L127) — Full-resolution (width,height), without decoding image pixels if possible.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.segmentation.export](#ui-segmentation-export), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).
- [`resize_to_max_side(img, max_side=1024)`](../utils_new/dataset_files_new.py#L140)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`normalize_to_uint8(img)`](../utils_new/dataset_files_new.py#L149) — Use the already-shared ImageStats conversion, not a second algorithm.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`convert_to_grayscale(img)`](../utils_new/dataset_files_new.py#L155)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint='auto')`](../utils_new/dataset_files_new.py#L160) — Qt-free lazy cache record used by both dataset tabs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- Класс [`LazyImageCache`](../utils_new/dataset_files_new.py#L199)
  - [`LazyImageCache.__init__(self, paths, loader, window=8, prefetch_ahead=4, max_resident=16, max_workers=2)`](../utils_new/dataset_files_new.py#L200)
  - [`LazyImageCache.__len__(self)`](../utils_new/dataset_files_new.py#L216)
  - [`LazyImageCache._trim_locked(self)`](../utils_new/dataset_files_new.py#L219)
  - [`LazyImageCache._load_sync(self, idx)`](../utils_new/dataset_files_new.py#L223)
  - [`LazyImageCache.is_loaded(self, idx)`](../utils_new/dataset_files_new.py#L240)
  - [`LazyImageCache.get_error(self, idx)`](../utils_new/dataset_files_new.py#L244)
  - [`LazyImageCache.get(self, idx)`](../utils_new/dataset_files_new.py#L248)
  - [`LazyImageCache._load_bg(self, idx)`](../utils_new/dataset_files_new.py#L266)
  - [`LazyImageCache.prefetch(self, indices)`](../utils_new/dataset_files_new.py#L293)
  - [`LazyImageCache.trim_around(self, center, keep_behind=3, keep_ahead=None)`](../utils_new/dataset_files_new.py#L308)
  - [`LazyImageCache.shutdown(self)`](../utils_new/dataset_files_new.py#L321)

<a id="utils-new-dataset-image-stats-new"></a>

## utils_new.dataset_image_stats_new

Исходник: [utils_new/dataset_image_stats_new.py](../utils_new/dataset_image_stats_new.py#L1).

**Прямые потребители:** [ui.image_conversion](#ui-image-conversion), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- Класс [`ImageStats`](../utils_new/dataset_image_stats_new.py#L9)
  - [`ImageStats.empty_color_stats()`](../utils_new/dataset_image_stats_new.py#L11)
  - [`ImageStats.empty_bit_stats()`](../utils_new/dataset_image_stats_new.py#L15)
  - [`ImageStats.detect_color_type(img)`](../utils_new/dataset_image_stats_new.py#L20)
  - [`ImageStats.detect_bit_type(img, img_path)`](../utils_new/dataset_image_stats_new.py#L37)
  - [`ImageStats.to_uint8(img)`](../utils_new/dataset_image_stats_new.py#L57)
  - [`ImageStats.to_display_bgr(img)`](../utils_new/dataset_image_stats_new.py#L70)
  - [`ImageStats.to_gray(img)`](../utils_new/dataset_image_stats_new.py#L83)
  - [`ImageStats.to_bgr3(img)`](../utils_new/dataset_image_stats_new.py#L96)
  - [`ImageStats.format_stats_text(color_stats, bit_stats, has_pairs)`](../utils_new/dataset_image_stats_new.py#L109)

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

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi).

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

- Класс [`DatasetLoaderMixin`](../utils_new/dataset_loader_new.py#L47)
  - [`DatasetLoaderMixin._after_load_hook(self)`](../utils_new/dataset_loader_new.py#L69)
  - [`DatasetLoaderMixin._after_type_change_hook(self, new_type)`](../utils_new/dataset_loader_new.py#L76)
  - [`DatasetLoaderMixin._sync_all_states(self, *_)`](../utils_new/dataset_loader_new.py#L81)
  - [`DatasetLoaderMixin._refresh_mask_pixel_column(self)`](../utils_new/dataset_loader_new.py#L97)
  - [`DatasetLoaderMixin.log(self, message)`](../utils_new/dataset_loader_new.py#L107)
  - [`DatasetLoaderMixin._log_debug(self, msg)`](../utils_new/dataset_loader_new.py#L116)
  - [`DatasetLoaderMixin._log_exception(self, context, exc=None)`](../utils_new/dataset_loader_new.py#L123)
  - [`DatasetLoaderMixin._copy_selected_rows_to_clipboard(self)`](../utils_new/dataset_loader_new.py#L135)
  - [`DatasetLoaderMixin._on_file_list_double_clicked(self, item)`](../utils_new/dataset_loader_new.py#L143)
  - [`DatasetLoaderMixin.select_images_folder(self)`](../utils_new/dataset_loader_new.py#L155)
  - [`DatasetLoaderMixin.select_labels_folder(self)`](../utils_new/dataset_loader_new.py#L164)
  - [`DatasetLoaderMixin.load_from_yaml(self)`](../utils_new/dataset_loader_new.py#L185)
  - [`DatasetLoaderMixin._load_yolo_yaml_from_path(self, file_path)`](../utils_new/dataset_loader_new.py#L202)
  - [`DatasetLoaderMixin.load_from_coco(self)`](../utils_new/dataset_loader_new.py#L251)
  - [`DatasetLoaderMixin.add_from_coco(self)`](../utils_new/dataset_loader_new.py#L254)
  - [`DatasetLoaderMixin._handle_load_coco(self, replace)`](../utils_new/dataset_loader_new.py#L257)
  - [`DatasetLoaderMixin._ask_coco_source(self)`](../utils_new/dataset_loader_new.py#L277)
  - [`DatasetLoaderMixin._find_coco_json_in_folder(self, folder)`](../utils_new/dataset_loader_new.py#L293)
  - [`DatasetLoaderMixin._get_coco_tmp_labels_dir(self, first_json_path=None)`](../utils_new/dataset_loader_new.py#L297)
  - [`DatasetLoaderMixin._load_coco_from_paths(self, json_paths, replace=True)`](../utils_new/dataset_loader_new.py#L305)
  - [`DatasetLoaderMixin.load_from_unet_yaml(self)`](../utils_new/dataset_loader_new.py#L383)
  - [`DatasetLoaderMixin._load_unet_from_path(self, yaml_path)`](../utils_new/dataset_loader_new.py#L390)
  - [`DatasetLoaderMixin._apply_loaded_pairs(self, pairs, input_type, input_format, class_names_from_yaml=None, class_names_override=None, source_label='')`](../utils_new/dataset_loader_new.py#L427)
  - [`DatasetLoaderMixin.scan_pairs(self)`](../utils_new/dataset_loader_new.py#L481)
  - [`DatasetLoaderMixin._validate_all_pairs(self)`](../utils_new/dataset_loader_new.py#L546)
  - [`DatasetLoaderMixin._recompute_color_stats(self)`](../utils_new/dataset_loader_new.py#L567)
  - [`DatasetLoaderMixin._update_color_stats_label(self)`](../utils_new/dataset_loader_new.py#L573)
  - [`DatasetLoaderMixin._open_pair_validation(self, pair)`](../utils_new/dataset_loader_new.py#L583)
  - [`DatasetLoaderMixin._make_list_item(self, text, tag=None, error_msg='')`](../utils_new/dataset_loader_new.py#L609)
  - [`DatasetLoaderMixin._rebuild_file_list_display(self, pairs=None)`](../utils_new/dataset_loader_new.py#L633)
  - [`DatasetLoaderMixin._is_include_empty(self)`](../utils_new/dataset_loader_new.py#L709)
  - [`DatasetLoaderMixin._on_include_empty_toggled(self, *_)`](../utils_new/dataset_loader_new.py#L712)
  - [`DatasetLoaderMixin._on_dataset_type_changed(self, idx)`](../utils_new/dataset_loader_new.py#L718)
  - [`DatasetLoaderMixin._on_input_format_changed(self, idx)`](../utils_new/dataset_loader_new.py#L742)
  - [`DatasetLoaderMixin._fill_class_table_from_dict(self, class_names_dict, preserve_include=True)`](../utils_new/dataset_loader_new.py#L754)
  - [`DatasetLoaderMixin._get_yaml_name_for_id(self, cid)`](../utils_new/dataset_loader_new.py#L798)
  - [`DatasetLoaderMixin.collect_classes(self)`](../utils_new/dataset_loader_new.py#L802)
  - [`DatasetLoaderMixin.update_included_classes(self)`](../utils_new/dataset_loader_new.py#L891)
  - [`DatasetLoaderMixin.on_class_table_item_changed(self, item)`](../utils_new/dataset_loader_new.py#L971)
  - [`DatasetLoaderMixin._update_pair_count_label(self, extra_text, error=False)`](../utils_new/dataset_loader_new.py#L977)

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

- [`write_index_mask(path, mask)`](../utils_new/dataset_output_new.py#L22)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`make_index_mask(annotations, size, *, offset=1)`](../utils_new/dataset_output_new.py#L33)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_classes_and_yaml(output_dir, class_names, *, out_yolo_txt=True, out_unet_yaml=False, out_masks_offset=1, input_type=0, dataset_path=None)`](../utils_new/dataset_output_new.py#L41)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`_coco_record(ann, image_id, ann_id)`](../utils_new/dataset_output_new.py#L65)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_coco_split(output_dir, split, rows, class_names, *, masks_offset=1, log=None)`](../utils_new/dataset_output_new.py#L77) — rows: [(image_name, image_size, annotations, optional_mask), ...].
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

- [`_list_images(folder)`](../utils_new/dataset_pair_tools_new.py#L18)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_detection_pairs(images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_tools_new.py#L28)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_segmentation_pairs(images_folder, masks_folder, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L48)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_image_only_pairs(images_folder)`](../utils_new/dataset_pair_tools_new.py#L83)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_conversion(pairs, validator, target)`](../utils_new/dataset_pair_tools_new.py#L88)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_color(pairs, target, progress_cb=None, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L103)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_8bit_png(pairs, validator)`](../utils_new/dataset_pair_tools_new.py#L131) — Возвращает (all_pairs, need_convert).
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None, skip_masks=False)`](../utils_new/dataset_pair_tools_new.py#L146) — skip_masks: если True — pairs трактуются как pairs из (img, mask),
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`build_pair_display(pair, validator, dataset_type, include_empty)`](../utils_new/dataset_pair_tools_new.py#L189)
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

- [`read_mask_with_fallback(path)`](../utils_new/dataset_pair_validator_new.py#L24)
  Явно импортируют: [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`load_annotations(path, image_w, image_h, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L32) — Legacy tuple API backed by the canonical parser (no old image_io).
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- Класс [`PairValidator`](../utils_new/dataset_pair_validator_new.py#L39)
  - [`PairValidator.__init__(self, dataset_type=0, input_format=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L51)
  - [`PairValidator.set_dataset_type(self, dataset_type)`](../utils_new/dataset_pair_validator_new.py#L57)
  - [`PairValidator.set_input_format(self, fmt)`](../utils_new/dataset_pair_validator_new.py#L61)
  - [`PairValidator.clear_cache(self)`](../utils_new/dataset_pair_validator_new.py#L65)
  - [`PairValidator.validate_full(self, pair)`](../utils_new/dataset_pair_validator_new.py#L68)
  - [`PairValidator.validate(self, pair)`](../utils_new/dataset_pair_validator_new.py#L77)
  - [`PairValidator.validate_all(self, pairs, progress_cb=None)`](../utils_new/dataset_pair_validator_new.py#L81)
  - [`PairValidator.recompute_stats(self, pairs)`](../utils_new/dataset_pair_validator_new.py#L96)
  - [`PairValidator._validate_impl(self, pair)`](../utils_new/dataset_pair_validator_new.py#L105)
  - [`PairValidator.detect_annotation_type(label_path, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_validator_new.py#L288)
  - [`PairValidator.check_pair_type_consistency()`](../utils_new/dataset_pair_validator_new.py#L330)

<a id="utils-new-dataset-preparation-bridge"></a>

## utils_new.dataset_preparation_bridge

Исходник: [utils_new/dataset_preparation_bridge.py](../utils_new/dataset_preparation_bridge.py#L1).

**Прямые потребители:** [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `typing.Iterable`.

**Объявления:**

- [`mask_classes_from_rows(rows: Iterable[tuple[int, str, bool]])`](../utils_new/dataset_preparation_bridge.py#L13) — Return (pixel->class_id, class_id->name) for selected positive pixels.
  Явно импортируют: [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).
- [`normalize_generation_pairs(pairs, input_type)`](../utils_new/dataset_preparation_bridge.py#L36) — Convert legacy (image, label-or-mask) to (image, label, mask).
  Явно импортируют: [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

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

- [`load_unet_yaml(yaml_path, log_cb=None)`](../utils_new/dataset_unet_new.py#L9) — Загружает masks_data.yaml (или data.yaml с masks-полями).
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`is_unet_yaml(yaml_path)`](../utils_new/dataset_unet_new.py#L122) — Эвристика: является ли YAML U-Net-структурой.
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`load_unet_yaml_pairs(yaml_path, log_cb=None)`](../utils_new/dataset_unet_new.py#L145) — Compatibility tuple for tabs, delegating all YAML logic to load_unet_yaml.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`write_yolo_txt_for_unet_pairs(pairs, output_labels_dir, *, class_names, mask_offset=1, include_empty=True, log_cb=None)`](../utils_new/dataset_unet_new.py#L152) — Convert indexed masks to editable YOLO-Segment labels (lossy contours).
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).

<a id="utils-new-dataset-validation-new"></a>

## utils_new.dataset_validation_new

Исходник: [utils_new/dataset_validation_new.py](../utils_new/dataset_validation_new.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new).

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

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`_resolve_relative_to_yaml(raw_path, yaml_path)`](../utils_new/dataset_yaml_new.py#L9) — Абсолютный путь: если raw_path абсолютный — берём как есть,
- [`_resolve_split_dirs(split_val, base_path)`](../utils_new/dataset_yaml_new.py#L21) — split_val может быть:
- [`_find_label_for_image(img_path, base_path, split, rel_dir)`](../utils_new/dataset_yaml_new.py#L70) — Ищет .txt для изображения в стандартных местах YOLO-структуры.
- [`find_mask_for_image(img_path, base_path=None)`](../utils_new/dataset_yaml_new.py#L110) — Ищет маску для изображения в стандартных местах YAML-структуры.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_yolo_yaml(yaml_path, log_cb=None)`](../utils_new/dataset_yaml_new.py#L151) — Парсит YOLO data.yaml.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`build_yolo_save_target_map(yaml_path, image_paths, *, root=None)`](../utils_new/dataset_yaml_new.py#L233) — Resolve canonical labels/<split>/<image>.txt destinations for YOLO YAML.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge).

<a id="utils-new-dataset-yolo-labels-new"></a>

## utils_new.dataset_yolo_labels_new

Исходник: [utils_new/dataset_yolo_labels_new.py](../utils_new/dataset_yolo_labels_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.export](#ui-segmentation-export), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.segmentation.export](#utils-new-segmentation-export), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `ImageSize`, `from_legacy`, `to_legacy`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `is_rectangle`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `image_size_from_file`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `math`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`_xywhr_points(vals, size=None)`](../utils_new/dataset_yolo_labels_new.py#L16) — Decode normalized ``cx,cy,w,h,angle`` into rectangle corners.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L43) — Read YOLO txt variants into canonical pixel-space annotations.
  Явно импортируют: [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new).
- [`_normalized_xywhr(ann, size)`](../utils_new/dataset_yolo_labels_new.py#L115) — OBB as ``cx/W, cy/H, width_px/W, height_px/H, radians_px``.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners')`](../utils_new/dataset_yolo_labels_new.py#L138)
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new).
- [`canonicalize_obb_corners(norm_points, width=1, height=1)`](../utils_new/dataset_yolo_labels_new.py#L178) — Order the original four vertices, without altering their coordinates.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_annotations_for_image(label_path, image_path, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L201) — Read normalized tuples using the associated image's *real* dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`save_annotations_for_image(annotations, label_path, image_path, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L209) — Prevent Save All from using the current (different) image dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`_tolerant_editor_rows(path, size, format_hint='auto')`](../utils_new/dataset_yolo_labels_new.py#L218) — Recover syntactically valid, geometrically invalid rows for manual repair.
- [`load_annotations(txt_path, img_w, img_h, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L260)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).
- [`save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L280)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.segmentation.export](#utils-new-segmentation-export).
- Класс [`LabelSavePlan`](../utils_new/dataset_yolo_labels_new.py#L327)
- [`build_label_save_plan(entries)`](../utils_new/dataset_yolo_labels_new.py#L335) — Refuse collisions; count only existing *destination* .txt files.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [ui.segmentation.export](#ui-segmentation-export).
- [`format_label_save_confirmation(plan, *, operation='Save', obb_format='corners', source_kind='images', unloaded_count=0)`](../utils_new/dataset_yolo_labels_new.py#L356) — Human-readable summary for the shared Qt Yes/No dialogs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [ui.segmentation.export](#ui-segmentation-export).

<a id="utils-new-json-store"></a>

## utils_new.json_store

Исходник: [utils_new/json_store.py](../utils_new/json_store.py#L1).

**Прямые потребители:** [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `json`, `os`, `pathlib.Path`, `tempfile`.

**Объявления:**

- [`read_json_object(path)`](../utils_new/json_store.py#L9)
  Явно импортируют: [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).
- [`write_json_atomic(path, value)`](../utils_new/json_store.py#L17)
  Явно импортируют: [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).

<a id="utils-new-morphology"></a>

## utils_new.morphology

Исходник: [utils_new/morphology.py](../utils_new/morphology.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.morphology_ui](#ui-morphology-ui), [utils.ensemble_utils](#utils-ensemble-utils), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.segmentation.presets](#utils-new-segmentation-presets).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `numpy`, `typing.Mapping`, `typing.Optional`, `typing.Sequence`, `typing.Union`.

**Объявления:**

- Класс [`MorphologyStep`](../utils_new/morphology.py#L58)
- [`_int_value(value, field: str, minimum: int)`](../utils_new/morphology.py#L72)
- [`_shape_code(shape: str)`](../utils_new/morphology.py#L81)
- [`build_kernel(size_px: int, shape: str='Rectangle')`](../utils_new/morphology.py#L90) — Квадратное структурирующее ядро OpenCV точного размера ``size_px``.
- [`_normalized_steps(operations: Sequence[Union[MorphologyStep, Mapping]])`](../utils_new/morphology.py#L99)
- [`_region_mask(mask: np.ndarray, image_shape: tuple)`](../utils_new/morphology.py#L127)
- [`make_contour_mask(mask: np.ndarray, width_px: int, shape: str='Rectangle')`](../utils_new/morphology.py#L141) — Внутренняя полоса шириной ``width_px`` вдоль границы объекта.
- [`apply_morphology(image: np.ndarray, operations: Sequence[Union[MorphologyStep, Mapping]], *, shape: str='Rectangle', mask: Optional[np.ndarray]=None, contour_only: bool=False, contour_width_px: Optional[int]=None, contour_shape: Optional[str]=None)`](../utils_new/morphology.py#L162) — Применить операции в указанной последовательности к изображению или ROI.
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [utils.ensemble_utils](#utils-ensemble-utils), [utils_new.annotation_processing](#utils-new-annotation-processing).

<a id="utils-new-presets-store"></a>

## utils_new.presets_store

Исходник: [utils_new/presets_store.py](../utils_new/presets_store.py#L1).

**Прямые потребители:** [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings).

**Прямые локальные импорты:**
- [utils_new.json_store](#utils-new-json-store): `read_json_object`, `write_json_atomic`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`read_presets(path)`](../utils_new/presets_store.py#L6)
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings).
- [`write_presets(path, presets)`](../utils_new/presets_store.py#L10)
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings).

<a id="utils-new-segmentation-export"></a>

## utils_new.segmentation.export

Исходник: [utils_new/segmentation/export.py](../utils_new/segmentation/export.py#L1).

**Прямые потребители:** [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.export](#ui-segmentation-export).

**Прямые локальные импорты:**
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `save_annotations`

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `tempfile`.

**Объявления:**

- [`write_annotations_atomic(annotations, destination, width, height, obb_as_xywhr=False)`](../utils_new/segmentation/export.py#L9)
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.export](#ui-segmentation-export).

<a id="utils-new-segmentation-image-session"></a>

## utils_new.segmentation.image_session

Исходник: [utils_new/segmentation/image_session.py](../utils_new/segmentation/image_session.py#L1).

**Прямые потребители:** [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.image_input](#ui-segmentation-image-input).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `HARD_MAX_SIDE`, `LazyImageCache`, `read_image_with_fallback`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`image_paths(source)`](../utils_new/segmentation/image_session.py#L21)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).
- [`legacy_uint8(image)`](../utils_new/segmentation/image_session.py#L29)
- [`resize_legacy(image, max_side)`](../utils_new/segmentation/image_session.py#L39)
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint='auto')`](../utils_new/segmentation/image_session.py#L48)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).
- [`create_cache(source, **loader_options)`](../utils_new/segmentation/image_session.py#L79)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).

<a id="utils-new-segmentation-postprocessing"></a>

## utils_new.segmentation.postprocessing

Исходник: [utils_new/segmentation/postprocessing.py](../utils_new/segmentation/postprocessing.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tests.test_segmentation_shared](#tests-test-segmentation-shared).

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `extract_annotations_from_mask`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`extract_objects(image, binary, mode, use_hull=False)`](../utils_new/segmentation/postprocessing.py#L6)
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

<a id="utils-new-segmentation-presets"></a>

## utils_new.segmentation.presets

Исходник: [utils_new/segmentation/presets.py](../utils_new/segmentation/presets.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.model_research](#tabs-model-research), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [utils.ensemble_utils](#utils-ensemble-utils).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `SUPPORTED_OPERATIONS`, `SUPPORTED_SHAPES`

**Внешние/stdlib импорты:** `copy.deepcopy`.

**Объявления:**

- [`default_morphology()`](../utils_new/segmentation/presets.py#L11)
- [`validate_morphology(state)`](../utils_new/segmentation/presets.py#L24)
- [`validate_preset(state, kind=None)`](../utils_new/segmentation/presets.py#L59)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [utils.ensemble_utils](#utils-ensemble-utils).
- [`make_preset(kind, **parameters)`](../utils_new/segmentation/presets.py#L83)
  Явно импортируют: [tabs.model_research](#tabs-model-research), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings).

<a id="utils-new-settings-store"></a>

## utils_new.settings_store

Исходник: [utils_new/settings_store.py](../utils_new/settings_store.py#L1).

**Прямые потребители:** [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.settings_service](#ui-settings-service).

**Прямые локальные импорты:**
- [utils_new.json_store](#utils-new-json-store): `read_json_object`, `write_json_atomic`

**Внешние/stdlib импорты:** `copy.deepcopy`.

**Объявления:**

- [`normalize_settings(data, defaults)`](../utils_new/settings_store.py#L10)
  Явно импортируют: [tests.test_pixel_morphology](#tests-test-pixel-morphology).
- [`load_settings(path, defaults)`](../utils_new/settings_store.py#L35)
  Явно импортируют: [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.settings_service](#ui-settings-service).
- [`save_settings(path, data, defaults)`](../utils_new/settings_store.py#L39)
  Явно импортируют: [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.settings_service](#ui-settings-service).

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
- [`apply_threshold_method(gray_img, method_name, params)`](../utils_new/threshold_processing.py#L135) — Применяет метод бинаризации по имени и параметрам.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).

<a id="utils-ops-deep-learning-ops"></a>

## utils_ops.deep_learning_ops

Исходник: [utils_ops/deep_learning_ops.py](../utils_ops/deep_learning_ops.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `cv2`, `importlib.util.find_spec`, `json`, `numpy`, `onnxruntime`, `os`, `segment_anything.SamPredictor`, `segment_anything.sam_model_registry`, `segmentation_models_pytorch`, `sys`, `time`, `torch`, `torchvision.transforms`, `ultralytics.YOLO`.

**Объявления:**

- [`_log(msg)`](../utils_ops/deep_learning_ops.py#L11)
- Класс [`BaseSegmentor`](../utils_ops/deep_learning_ops.py#L28)
  - [`BaseSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L30)
  - [`BaseSegmentor.load(self, **kwargs)`](../utils_ops/deep_learning_ops.py#L36)
  - [`BaseSegmentor.predict(self, image, **kwargs)`](../utils_ops/deep_learning_ops.py#L39)
- Класс [`TorchSegmentor`](../utils_ops/deep_learning_ops.py#L43) (`BaseSegmentor`)
  - [`TorchSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L45)
  - [`TorchSegmentor._get_device(self)`](../utils_ops/deep_learning_ops.py#L50)
  - [`TorchSegmentor._resolve_resize_mode(self)`](../utils_ops/deep_learning_ops.py#L69)
  - [`TorchSegmentor._preprocess(self, image, input_size)`](../utils_ops/deep_learning_ops.py#L104)
  - [`TorchSegmentor._postprocess_mask(self, mask_np, orig_size, pad_info, resize_mode)`](../utils_ops/deep_learning_ops.py#L174)
- Класс [`UNetSegmentor`](../utils_ops/deep_learning_ops.py#L195) (`TorchSegmentor`)
  - [`UNetSegmentor.load(self, weights_path=None, encoder=None, input_size=None)`](../utils_ops/deep_learning_ops.py#L213)
  - [`UNetSegmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L357)
- Класс [`DeepLabV3Segmentor`](../utils_ops/deep_learning_ops.py#L414) (`TorchSegmentor`)
  - [`DeepLabV3Segmentor.load(self, weights_path=None, backbone=None, output_stride=None)`](../utils_ops/deep_learning_ops.py#L437)
  - [`DeepLabV3Segmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L608)
- Класс [`YOLOSegmentor`](../utils_ops/deep_learning_ops.py#L699) (`BaseSegmentor`)
  - [`YOLOSegmentor.load(self, model_path=None)`](../utils_ops/deep_learning_ops.py#L713)
  - [`YOLOSegmentor._otsu_inside_roi(image, x1, y1, x2, y2)`](../utils_ops/deep_learning_ops.py#L736)
  - [`YOLOSegmentor.predict(self, image, conf=0.25, iou=0.45, imgsz=640, save=False)`](../utils_ops/deep_learning_ops.py#L751)
- Класс [`SAMSegmentor`](../utils_ops/deep_learning_ops.py#L882) (`BaseSegmentor`)
  - [`SAMSegmentor.load(self, model_type='vit_h', model_path=None)`](../utils_ops/deep_learning_ops.py#L884)
  - [`SAMSegmentor.predict(self, image, prompt_type='everything')`](../utils_ops/deep_learning_ops.py#L897)
- Класс [`ONNXSegmentor`](../utils_ops/deep_learning_ops.py#L904) (`BaseSegmentor`)
  - [`ONNXSegmentor.load(self, model_path)`](../utils_ops/deep_learning_ops.py#L906)
  - [`ONNXSegmentor.predict(self, image)`](../utils_ops/deep_learning_ops.py#L916)
- [`create_segmentor(model_type, **kwargs)`](../utils_ops/deep_learning_ops.py#L948)
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods).

<a id="utils-ops-gradient-ops"></a>

## utils_ops.gradient_ops

Исходник: [utils_ops/gradient_ops.py](../utils_ops/gradient_ops.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tests.test_processing_smoke](#tests-test-processing-smoke).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `numpy`.

**Объявления:**

- [`apply_gradient_method(gray, method, params)`](../utils_ops/gradient_ops.py#L5)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tests.test_processing_smoke](#tests-test-processing-smoke).

<a id="utils-ops-interactive-ops"></a>

## utils_ops.interactive_ops

Исходник: [utils_ops/interactive_ops.py](../utils_ops/interactive_ops.py#L1).

**Прямые потребители:** [tabs.interactive_methods](#tabs-interactive-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `networkx`, `numpy`, `skimage.segmentation.active_contour`, `skimage.segmentation.find_boundaries`, `skimage.segmentation.random_walker`, `skimage.segmentation.slic`, `sklearn.mixture.GaussianMixture`, `warnings`.

**Объявления:**

- [`grabcut_segmentation(img, rect=None, mask=None, fg_scribbles=None, bg_scribbles=None, iterations=5, mode='Both')`](../utils_ops/interactive_ops.py#L17) — GrabCut segmentation with mode selection.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`watershed_segmentation(gray, distance_threshold=50, min_area=20)`](../utils_ops/interactive_ops.py#L67)
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`random_walker_segmentation(img, markers, beta=130, mode='cg_j')`](../utils_ops/interactive_ops.py#L92) — Random Walker segmentation.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`active_contour_segmentation(gray, rect, alpha=0.01, beta=0.1, gamma=0.01, max_iter=250, convergence=0.001, smooth=True, init_type='Ellipse')`](../utils_ops/interactive_ops.py#L116) — Active Contours (Snakes) segmentation.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`supercut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, lambda_val=0.5, sigma_color=10.0)`](../utils_ops/interactive_ops.py#L200) — SuperCut using graph cut (max-flow/min-cut) on superpixels.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`onecut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, spatial_weight=1.0, data_weight=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L305) — OneCut segmentation using superpixels and graph cut.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).
- [`lazy_snapping(img, fg_scribbles, bg_scribbles, superpixel_size=50, compactness=10, sigma=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L452) — Lazy Snapping using graph cut on superpixels.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).

<a id="utils-ops-segmentation-ops"></a>

## utils_ops.segmentation_ops

Исходник: [utils_ops/segmentation_ops.py](../utils_ops/segmentation_ops.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [utils.ensemble_utils](#utils-ensemble-utils).

**Прямые локальные импорты:**
- [ui.settings_service](#ui-settings-service): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `cv2`, `numpy`.

**Объявления:**

- [`get_display_params(img_shape)`](../utils_ops/segmentation_ops.py#L11)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`draw_label(img_color, label, x, y, box_w, box_h, font_scale, font_thickness, thickness, color=None)`](../utils_ops/segmentation_ops.py#L25)
- [`draw_label_rotated(img_color, label, center, font_scale, font_thickness, color=None)`](../utils_ops/segmentation_ops.py#L48)
- [`get_class_color(cls_id)`](../utils_ops/segmentation_ops.py#L85) — Цвет (BGR) для класса. Для id >= 10 цвета циклически повторяются.
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui).
- [`draw_class_label(img, text, x, y, box_w, box_h, font_scale, font_thickness, bg_color, text_color=(255, 255, 255), padding=2, margin=5)`](../utils_ops/segmentation_ops.py#L96) — Рисует текстовую подпись поверх изображения на цветной плашке.
- [`simple_threshold(img, thresh)`](../utils_ops/segmentation_ops.py#L153)
- [`otsu_threshold(img)`](../utils_ops/segmentation_ops.py#L158)
- [`triangle_threshold(img)`](../utils_ops/segmentation_ops.py#L163)
- [`niblack_threshold(img, window_size=15, k=-0.2)`](../utils_ops/segmentation_ops.py#L168)
- [`sauvola_threshold(img, window_size=25, k=0.2, R=128)`](../utils_ops/segmentation_ops.py#L177)
- [`isodata_threshold(img, initial_threshold=128)`](../utils_ops/segmentation_ops.py#L186)
- [`background_symmetry_threshold(img, excess_threshold=0.2)`](../utils_ops/segmentation_ops.py#L204)
- [`row_adaptive_threshold(img, window_size=50, k=0.5)`](../utils_ops/segmentation_ops.py#L227)
- [`segment_contours(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L244) — Возвращает (img_color, objects) где objects = список ('segment', 0, points) в нормализованных координатах.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils.ensemble_utils](#utils-ensemble-utils).
- [`segment_projections(img, binary, draw=True)`](../utils_ops/segmentation_ops.py#L284) — Возвращает (img_color, objects) где objects = список ('detect', 0, cx, cy, w, h)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils.ensemble_utils](#utils-ensemble-utils).
- [`segment_min_area_rect(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L322) — Возвращает (img_color, objects) где objects = список ('obb', 0, points) где points = 8 нормализованных координат углов.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils.ensemble_utils](#utils-ensemble-utils).
- [`segment_polygons(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L362) — Алиас для segment_contours (уже возвращает segment).
- [`draw_yolo_annotations(img, annotations, color=(0, 255, 0))`](../utils_ops/segmentation_ops.py#L368) — Рисует аннотации любого типа (detect, obb, segment) на изображении.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_ops/segmentation_ops.py#L419) — Форматирует аннотацию для отображения в QListWidget.
- [`update_annotation_list(list_widget: QListWidget, annotations, img_w: int, img_h: int)`](../utils_ops/segmentation_ops.py#L449) — Очищает и заполняет QListWidget строками аннотаций. Каждому элементу в UserRole записывается индекс.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset).
- [`delete_annotation_by_index(current_annotations, idx, all_annotations=None, current_index=None)`](../utils_ops/segmentation_ops.py#L459) — Удаляет аннотацию с индексом idx. Возвращает (new_annotations, new_selected_index).
- [`draw_selected_objects(img, objects, selected_indices, draw_mode, use_hull, color_rect, color_label, thickness, font_scale, font_thickness)`](../utils_ops/segmentation_ops.py#L472) — Рисует выбранные объекты на изображении. Поддерживает ('detect',...), ('obb',...), ('segment',...).
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_ops/segmentation_ops.py#L521) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).
- [`normalize_method_name(method_name)`](../utils_ops/segmentation_ops.py#L542) — Приводит имя метода к каноническому виду для UI.
- [`apply_threshold_method(gray_img, method_name, params)`](../utils_ops/segmentation_ops.py#L568) — Применяет метод бинаризации по имени и параметрам.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_processing_smoke](#tests-test-processing-smoke), [utils.ensemble_utils](#utils-ensemble-utils).
- [`get_morph_kernel(size, shape_name)`](../utils_ops/segmentation_ops.py#L629)
- [`apply_morphology(binary, close_factor, open_factor, kernel_shape, img_shape=None)`](../utils_ops/segmentation_ops.py#L639)

<a id="utils-ops-traditional-ml-ops"></a>

## utils_ops.traditional_ml_ops

Исходник: [utils_ops/traditional_ml_ops.py](../utils_ops/traditional_ml_ops.py#L1).

**Прямые потребители:** [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `joblib`, `numpy`, `skimage.feature`, `skimage.segmentation`, `sklearn.cluster.KMeans`, `sklearn.cluster.MeanShift`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.svm.SVC`, `sklearn.tree.DecisionTreeClassifier`, `warnings`, `xgboost`.

**Объявления:**

- [`extract_pixel_features(image, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L20)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`extract_superpixel_features(image, segments, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L47)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_kmeans(X, n_clusters=3)`](../utils_ops/traditional_ml_ops.py#L85)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_meanshift(X)`](../utils_ops/traditional_ml_ops.py#L90)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_svm(X, y, kernel='rbf', C=10)`](../utils_ops/traditional_ml_ops.py#L95)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_random_forest(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L100)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_xgboost(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L105)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_decision_tree(X, y)`](../utils_ops/traditional_ml_ops.py#L112)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`predict_pixelwise(model, X, original_shape)`](../utils_ops/traditional_ml_ops.py#L117)
- [`predict_superpixel(model, segments, X_superpixel)`](../utils_ops/traditional_ml_ops.py#L121)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`create_superpixels(image, superpixel_size=50)`](../utils_ops/traditional_ml_ops.py#L128)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`compute_elbow_kmeans(X, max_k=10)`](../utils_ops/traditional_ml_ops.py#L134) — Вычисляет инерцию (сумму квадратов расстояний) для K от 1 до max_k.
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`find_optimal_k_elbow(inertias)`](../utils_ops/traditional_ml_ops.py#L146) — Находит оптимальное K как точку максимального изгиба (метод локтя).
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

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

- [`log(msg)`](../utils_training/train_DeepLabV3_worker.py#L186)
- [`signal_handler(signum, frame)`](../utils_training/train_DeepLabV3_worker.py#L190)
- [`find_image_file(jpeg_dir, basename)`](../utils_training/train_DeepLabV3_worker.py#L206)
- [`read_voc_pairs(txt_path, jpeg_dir, seg_dir)`](../utils_training/train_DeepLabV3_worker.py#L213)
- [`read_voc_file_list(txt_path, root_dir)`](../utils_training/train_DeepLabV3_worker.py#L238)
- [`_resolve_path(root, p)`](../utils_training/train_DeepLabV3_worker.py#L243)
- [`create_loaders_from_yaml(yaml_path, default_input_size, default_num_classes, augmentations, ignore_index, batch_size, num_workers, normalize, resize_mode)`](../utils_training/train_DeepLabV3_worker.py#L254)
- [`create_voc_loaders(root, input_size, num_classes, augmentations, ignore_index, batch_size, num_workers, normalize=False, resize_mode='hard')`](../utils_training/train_DeepLabV3_worker.py#L361)
- Класс [`VOCSegmentationPairs`](../utils_training/train_DeepLabV3_worker.py#L385) (`Dataset`)
  - [`VOCSegmentationPairs.__init__(self, image_mask_pairs, input_size, num_classes, ignore_index, augmentations, normalize=False, resize_mode='hard')`](../utils_training/train_DeepLabV3_worker.py#L386)
  - [`VOCSegmentationPairs.__len__(self)`](../utils_training/train_DeepLabV3_worker.py#L396)
  - [`VOCSegmentationPairs.__getitem__(self, idx)`](../utils_training/train_DeepLabV3_worker.py#L399)
- [`get_loss_function(loss_name, num_classes, ignore_index=None, class_weights=None, loss_weights=None, device='cpu')`](../utils_training/train_DeepLabV3_worker.py#L464) — Патч 1: weights_tensor создаётся на нужном устройстве (device).
- [`get_optimizer(model, opt_name, lr, momentum=0.9, weight_decay=0.0001)`](../utils_training/train_DeepLabV3_worker.py#L515)
- [`get_layerwise_optimizer(model, opt_name, base_lr, momentum=0.9, weight_decay=0.0001, decay_factor=0.1)`](../utils_training/train_DeepLabV3_worker.py#L528) — Layer-wise LR. Устойчив к тому, что get_stages() может возвращать
- [`get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-06)`](../utils_training/train_DeepLabV3_worker.py#L587)
- Класс [`PolyLRSchedulerIter`](../utils_training/train_DeepLabV3_worker.py#L600)
  - [`PolyLRSchedulerIter.__init__(self, optimizer, max_iter, power=0.9, min_lr=0.0)`](../utils_training/train_DeepLabV3_worker.py#L601)
  - [`PolyLRSchedulerIter.step(self)`](../utils_training/train_DeepLabV3_worker.py#L609)
  - [`PolyLRSchedulerIter.get_last_lr(self)`](../utils_training/train_DeepLabV3_worker.py#L615)
- Класс [`ModelEMA`](../utils_training/train_DeepLabV3_worker.py#L619)
  - [`ModelEMA.__init__(self, model, decay=0.999, device=None)`](../utils_training/train_DeepLabV3_worker.py#L620)
  - [`ModelEMA.update(self, model)`](../utils_training/train_DeepLabV3_worker.py#L631)
  - [`ModelEMA.state_dict(self)`](../utils_training/train_DeepLabV3_worker.py#L642)
- [`set_bn_eval(module)`](../utils_training/train_DeepLabV3_worker.py#L646)
- [`_pad_to_multiple(x, multiple=16)`](../utils_training/train_DeepLabV3_worker.py#L652)
- [`_crop_to_hw(x, hw)`](../utils_training/train_DeepLabV3_worker.py#L662)
- [`_forward_safe(model, x, use_amp, device, output_stride=None)`](../utils_training/train_DeepLabV3_worker.py#L667) — Патч 3: убран nan_to_num(nan=0.0) — он маскировал проблему,
- [`predict_batch_multiscale(model, images, device, scales, use_amp)`](../utils_training/train_DeepLabV3_worker.py#L697)
- [`compute_metrics(pred_logits, true_masks, num_classes, ignore_index=None, output_activation='argmax', threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L722)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_DeepLabV3_worker.py#L784)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L813)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_DeepLabV3_worker.py#L825)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_DeepLabV3_worker.py#L832)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L854)
- [`save_pr_curve(preds, masks, save_dir, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L862)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes, ignore_index=None, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L881)
- [`save_train_batch_examples(loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_DeepLabV3_worker.py#L919)
- [`save_val_examples(loader, model, device, save_dir, num_samples=8, num_classes=1, output_activation='argmax', normalize=False, use_amp=False, multi_scale=False, scales=None)`](../utils_training/train_DeepLabV3_worker.py#L951)
- [`main()`](../utils_training/train_DeepLabV3_worker.py#L1011)

<a id="utils-training-train-unet-worker"></a>

## utils_training.train_unet_worker

Исходник: [utils_training/train_unet_worker.py](../utils_training/train_unet_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `onnx`, `onnxruntime`, `os`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `traceback`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker.py#L129)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker.py#L134)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker.py#L156) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker.py#L157)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker.py#L199)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker.py#L202)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker.py#L258)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker.py#L333)
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker.py#L371)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker.py#L386)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L399)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker.py#L474)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker.py#L517)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker.py#L529)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker.py#L537)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker.py#L572)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker.py#L581)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L640)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_unet_worker.py#L701)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1, normalize=False)`](../utils_training/train_unet_worker.py#L744)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker.py#L815)
- [`main()`](../utils_training/train_unet_worker.py#L865)

<a id="utils-training-train-unet-worker-рабочий"></a>

## utils_training.train_unet_worker_рабочий

Исходник: [utils_training/train_unet_worker_рабочий.py](../utils_training/train_unet_worker_рабочий.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker_рабочий.py#L124)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker_рабочий.py#L128)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker_рабочий.py#L149) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker_рабочий.py#L150)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker_рабочий.py#L192)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker_рабочий.py#L195)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker_рабочий.py#L260)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L338)
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker_рабочий.py#L376)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker_рабочий.py#L391)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L404)
- [`auto_batch_size(device, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker_рабочий.py#L479)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L503)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker_рабочий.py#L514)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker_рабочий.py#L522)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L550)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L558)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L601)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L657)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L688)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker_рабочий.py#L746) — Проверяет первые max_check масок: собирает уникальные значения пикселей,
- [`main()`](../utils_training/train_unet_worker_рабочий.py#L801)

<a id="utils-training-train-yolo-worker"></a>

## utils_training.train_yolo_worker

Исходник: [utils_training/train_yolo_worker.py](../utils_training/train_yolo_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `argparse`, `datetime.datetime`, `json`, `onnx`, `os`, `pathlib.Path`, `shutil`, `signal`, `sys`, `traceback`, `ultralytics`, `ultralytics.YOLO`.

**Объявления:**

- [`log(msg)`](../utils_training/train_yolo_worker.py#L19) — Вывод сообщения с немедленным сбросом буфера.
- [`is_standard_yolo_model(model_path)`](../utils_training/train_yolo_worker.py#L23) — Определяет, является ли имя модели стандартным (без пути).
- [`signal_handler(signum, frame)`](../utils_training/train_yolo_worker.py#L34)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_yolo_worker.py#L41) — Сохраняет JSON-файл с метаданными рядом с .pt файлом.
- [`main()`](../utils_training/train_yolo_worker.py#L48)

<a id="utils-training-yolo2voc"></a>

## utils_training.yolo2voc

Исходник: [utils_training/yolo2voc.py](../utils_training/yolo2voc.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `os`, `pascal_voc_writer.Writer`, `shutil`, `sys`, `yaml`.

**Объявления:**

- [`create_binary_palette()`](../utils_training/yolo2voc.py#L39) — Создаёт палитру для 8-битного PNG: индекс 0 -> чёрный, индекс 1 -> белый.
- [`convert_mask_to_voc_format(src_mask_path, dst_mask_path)`](../utils_training/yolo2voc.py#L44) — Конвертирует бинарную маску (0 и 255) в P-режим с палитрой.
- [`copy_image(src_img_path, dst_img_path)`](../utils_training/yolo2voc.py#L57) — Копирует изображение без изменений.
- [`convert_yolo_txt_to_voc_xml(txt_path, img_path, class_names, output_xml_path)`](../utils_training/yolo2voc.py#L61) — Конвертирует один YOLO .txt файл (детекция) в VOC XML.
- [`process_set(set_name, base_dir, images_rel, masks_rel, labels_rel, class_names, voc_root)`](../utils_training/yolo2voc.py#L99) — Обрабатывает один набор данных (train/val/test).
- [`build_voc_from_yolo(yaml_file, output_root)`](../utils_training/yolo2voc.py#L161) — Основная функция конвертации, использующая YAML-файл датасета YOLO.
- [`interactive_mode()`](../utils_training/yolo2voc.py#L233) — Интерактивный режим с запросом параметров.
- [`main()`](../utils_training/yolo2voc.py#L289)
