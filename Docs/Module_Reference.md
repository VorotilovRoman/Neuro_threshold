# Справочник модулей, импортов и API

[Навигация](README.md) · [Карта архитектуры](Architecture.md) · [Дублирование](Duplication.md)

Сгенерировано `python Docs/generate_reference.py` из текущих исходников. Каталог исключает саму документацию и скрытые каталоги.

Проанализировано Python-модулей: **222**. Импорты учитываются на всех уровнях AST, включая функции и try/except; наличие записи не означает, что ветка выполняется при запуске. Звёздочные импорты не разворачиваются в список фактических вызовов. QProcess, строковые динамические импорты и файловые зависимости описаны отдельно в карте.

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
- [tests.test_annotation_objects](#tests-test-annotation-objects)
- [tests.test_dataset_audit](#tests-test-dataset-audit)
- [tests.test_dataset_workspace](#tests-test-dataset-workspace)
- [tests.test_dependency_installation](#tests-test-dependency-installation)
- [tests.test_ensemble_view](#tests-test-ensemble-view)
- [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch)
- [tests.test_import_architecture](#tests-test-import-architecture)
- [tests.test_label_save_merge](#tests-test-label-save-merge)
- [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration)
- [tests.test_method_algorithms](#tests-test-method-algorithms)
- [tests.test_model_contracts](#tests-test-model-contracts)
- [tests.test_pixel_morphology](#tests-test-pixel-morphology)
- [tests.test_processing_smoke](#tests-test-processing-smoke)
- [tests.test_report_evaluation](#tests-test-report-evaluation)
- [tests.test_report_workflow](#tests-test-report-workflow)
- [tests.test_runtime_imports](#tests-test-runtime-imports)
- [tests.test_segmentation_shared](#tests-test-segmentation-shared)
- [tests.test_segmentation_tasks](#tests-test-segmentation-tasks)
- [tests.test_segmentation_unification](#tests-test-segmentation-unification)
- [tests.test_source_unification](#tests-test-source-unification)
- [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch)
- [tests.test_torch_runtime](#tests-test-torch-runtime)
- [tests.test_yolo_worker](#tests-test-yolo-worker)
- [ui.DL_predict_settings](#ui-DL-predict-settings)
- [ui.DL_train_settings_base](#ui-DL-train-settings-base)
- [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model)
- [ui.annotation_objects](#ui-annotation-objects)
- [ui.annotation_review](#ui-annotation-review)
- [ui.annotation_review_view](#ui-annotation-review-view)
- [ui.application](#ui-application)
- [ui.autoreseach_ui.__init__](#ui-autoreseach-ui---init--)
- [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls)
- [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller)
- [ui.autoreseach_ui.evaluation_controls](#ui-autoreseach-ui-evaluation-controls)
- [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui)
- [ui.autoreseach_ui.result_models](#ui-autoreseach-ui-result-models)
- [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog)
- [ui.control_buttons](#ui-control-buttons)
- [ui.dataset_audit.__init__](#ui-dataset-audit---init--)
- [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates)
- [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings)
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle)
- [ui.dataset_audit.models](#ui-dataset-audit-models)
- [ui.dataset_audit.page_duplicates](#ui-dataset-audit-page-duplicates)
- [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings)
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers)
- [ui.dataset_audit.page_models](#ui-dataset-audit-page-models)
- [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality)
- [ui.dataset_audit.page_report](#ui-dataset-audit-page-report)
- [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial)
- [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics)
- [ui.dataset_audit.page_training](#ui-dataset-audit-page-training)
- [ui.dataset_audit.page_wiring](#ui-dataset-audit-page-wiring)
- [ui.dataset_audit.presentation](#ui-dataset-audit-presentation)
- [ui.dataset_audit.quality_controller](#ui-dataset-audit-quality-controller)
- [ui.dataset_audit.reporting](#ui-dataset-audit-reporting)
- [ui.dataset_audit.selection](#ui-dataset-audit-selection)
- [ui.dataset_audit.session](#ui-dataset-audit-session)
- [ui.dataset_audit.source_page](#ui-dataset-audit-source-page)
- [ui.dataset_audit.spatial](#ui-dataset-audit-spatial)
- [ui.dataset_audit.tasks](#ui-dataset-audit-tasks)
- [ui.dataset_audit.training_results](#ui-dataset-audit-training-results)
- [ui.dataset_audit.workers](#ui-dataset-audit-workers)
- [ui.dataset_audit.workspace](#ui-dataset-audit-workspace)
- [ui.dataset_source](#ui-dataset-source)
- [ui.image_conversion](#ui-image-conversion)
- [ui.image_navigation](#ui-image-navigation)
- [ui.image_view](#ui-image-view)
- [ui.interactive_params_ui](#ui-interactive-params-ui)
- [ui.log_widget](#ui-log-widget)
- [ui.morphology_ui](#ui-morphology-ui)
- [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new)
- [ui.segmentation.actions](#ui-segmentation-actions)
- [ui.segmentation.computation](#ui-segmentation-computation)
- [ui.segmentation.controller](#ui-segmentation-controller)
- [ui.segmentation.errors](#ui-segmentation-errors)
- [ui.segmentation.export](#ui-segmentation-export)
- [ui.segmentation.histogram](#ui-segmentation-histogram)
- [ui.segmentation.image_input](#ui-segmentation-image-input)
- [ui.segmentation.interactive_view](#ui-segmentation-interactive-view)
- [ui.segmentation.jobs](#ui-segmentation-jobs)
- [ui.segmentation.navigation](#ui-segmentation-navigation)
- [ui.segmentation.panels](#ui-segmentation-panels)
- [ui.segmentation.presentation](#ui-segmentation-presentation)
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings)
- [ui.segmentation.results](#ui-segmentation-results)
- [ui.segmentation.session](#ui-segmentation-session)
- [ui.settings_service](#ui-settings-service)
- [ui.settings_ui](#ui-settings-ui)
- [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui)
- [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui)
- [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui)
- [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui)
- [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui)
- [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui)
- [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui)
- [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui)
- [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui)
- [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui)
- [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui)
- [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui)
- [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui)
- [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui)
- [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def)
- [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new)
- [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old)
- [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet)
- [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad)
- [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new)
- [utils_new.annotation_processing](#utils-new-annotation-processing)
- [utils_new.annotations.__init__](#utils-new-annotations---init--)
- [utils_new.annotations.adapters](#utils-new-annotations-adapters)
- [utils_new.annotations.augmentation](#utils-new-annotations-augmentation)
- [utils_new.annotations.boxes](#utils-new-annotations-boxes)
- [utils_new.annotations.geometry](#utils-new-annotations-geometry)
- [utils_new.annotations.masks](#utils-new-annotations-masks)
- [utils_new.annotations.model](#utils-new-annotations-model)
- [utils_new.annotations.modes](#utils-new-annotations-modes)
- [utils_new.annotations.object_info](#utils-new-annotations-object-info)
- [utils_new.annotations.rendering](#utils-new-annotations-rendering)
- [utils_new.annotations.transforms](#utils-new-annotations-transforms)
- [utils_new.autoreseach_utils.__init__](#utils-new-autoreseach-utils---init--)
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation)
- [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export)
- [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics)
- [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing)
- [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth)
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports)
- [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search)
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space)
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow)
- [utils_new.dataset_audit.__init__](#utils-new-dataset-audit---init--)
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis)
- [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review)
- [utils_new.dataset_audit.atomic](#utils-new-dataset-audit-atomic)
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks)
- [utils_new.dataset_audit.export](#utils-new-dataset-audit-export)
- [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs)
- [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching)
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects)
- [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality)
- [utils_new.dataset_audit.scans](#utils-new-dataset-audit-scans)
- [utils_new.dataset_audit.splits](#utils-new-dataset-audit-splits)
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
- [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder)
- [utils_new.segmentation.export](#utils-new-segmentation-export)
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session)
- [utils_new.segmentation.model_export](#utils-new-segmentation-model-export)
- [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics)
- [utils_new.segmentation.model_output](#utils-new-segmentation-model-output)
- [utils_new.segmentation.operations](#utils-new-segmentation-operations)
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing)
- [utils_new.segmentation.presets](#utils-new-segmentation-presets)
- [utils_new.segmentation.result](#utils-new-segmentation-result)
- [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts)
- [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels)
- [utils_new.segmentation.yolo_metrics](#utils-new-segmentation-yolo-metrics)
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
| `PIL` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [generate](#generate), [tests.test_model_contracts](#tests-test-model-contracts), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `PyQt5` | [generate](#generate), [main](#main), [main_ui](#main-ui), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.DL_predict_settings](#ui-DL-predict-settings), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui.annotation_objects](#ui-annotation-objects), [ui.annotation_review](#ui-annotation-review), [ui.annotation_review_view](#ui-annotation-review-view), [ui.application](#ui-application), [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls), [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller), [ui.autoreseach_ui.evaluation_controls](#ui-autoreseach-ui-evaluation-controls), [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui), [ui.autoreseach_ui.result_models](#ui-autoreseach-ui-result-models), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [ui.control_buttons](#ui-control-buttons), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.page_duplicates](#ui-dataset-audit-page-duplicates), [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings), [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers), [ui.dataset_audit.page_models](#ui-dataset-audit-page-models), [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality), [ui.dataset_audit.page_report](#ui-dataset-audit-page-report), [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial), [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics), [ui.dataset_audit.page_training](#ui-dataset-audit-page-training), [ui.dataset_audit.presentation](#ui-dataset-audit-presentation), [ui.dataset_audit.quality_controller](#ui-dataset-audit-quality-controller), [ui.dataset_audit.reporting](#ui-dataset-audit-reporting), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [ui.dataset_audit.session](#ui-dataset-audit-session), [ui.dataset_audit.source_page](#ui-dataset-audit-source-page), [ui.dataset_audit.tasks](#ui-dataset-audit-tasks), [ui.dataset_audit.training_results](#ui-dataset-audit-training-results), [ui.dataset_audit.workers](#ui-dataset-audit-workers), [ui.dataset_audit.workspace](#ui-dataset-audit-workspace), [ui.dataset_source](#ui-dataset-source), [ui.image_conversion](#ui-image-conversion), [ui.image_navigation](#ui-image-navigation), [ui.image_view](#ui-image-view), [ui.interactive_params_ui](#ui-interactive-params-ui), [ui.log_widget](#ui-log-widget), [ui.morphology_ui](#ui-morphology-ui), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.actions](#ui-segmentation-actions), [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.controller](#ui-segmentation-controller), [ui.segmentation.errors](#ui-segmentation-errors), [ui.segmentation.export](#ui-segmentation-export), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.interactive_view](#ui-segmentation-interactive-view), [ui.segmentation.jobs](#ui-segmentation-jobs), [ui.segmentation.panels](#ui-segmentation-panels), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.segmentation.results](#ui-segmentation-results), [ui.settings_service](#ui-settings-service), [ui.settings_ui](#ui-settings-ui), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops) |
| `cv2` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_dataset_audit](#tests-test-dataset-audit), [tests.test_dataset_workspace](#tests-test-dataset-workspace), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.image_conversion](#ui-image-conversion), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session), [utils_new.segmentation.model_output](#utils-new-segmentation-model-output), [utils_new.segmentation.operations](#utils-new-segmentation-operations), [utils_new.segmentation.result](#utils-new-segmentation-result), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_ops.gradient_ops](#utils-ops-gradient-ops), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `hdbscan` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs) |
| `imagehash` | [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks) |
| `joblib` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [utils_new.segmentation.operations](#utils-new-segmentation-operations), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `matplotlib` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.test_new_image](#cluster-test-new-image), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings), [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality), [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial), [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics), [ui.dataset_audit.presentation](#ui-dataset-audit-presentation), [ui.segmentation.histogram](#ui-segmentation-histogram), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `networkx` | [utils_ops.interactive_ops](#utils-ops-interactive-ops) |
| `numpy` | [cluster.canonicalize_crops](#cluster-canonicalize-crops), [cluster.canonicalize_crops_method2](#cluster-canonicalize-crops-method2), [cluster.cluster_embeddings](#cluster-cluster-embeddings), [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_annotation_objects](#tests-test-annotation-objects), [tests.test_dataset_audit](#tests-test-dataset-audit), [tests.test_dataset_workspace](#tests-test-dataset-workspace), [tests.test_ensemble_view](#tests-test-ensemble-view), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_model_contracts](#tests-test-model-contracts), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [tests.test_segmentation_tasks](#tests-test-segmentation-tasks), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.annotation_objects](#ui-annotation-objects), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.reporting](#ui-dataset-audit-reporting), [ui.image_conversion](#ui-image-conversion), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.computation](#ui-segmentation-computation), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.boxes](#utils-new-annotations-boxes), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.model](#utils-new-annotations-model), [utils_new.annotations.object_info](#utils-new-annotations-object-info), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics), [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.morphology](#utils-new-morphology), [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session), [utils_new.segmentation.model_output](#utils-new-segmentation-model-output), [utils_new.segmentation.operations](#utils-new-segmentation-operations), [utils_new.segmentation.result](#utils-new-segmentation-result), [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels), [utils_new.threshold_processing](#utils-new-threshold-processing), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_ops.gradient_ops](#utils-ops-gradient-ops), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `onnx` | [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `onnxruntime` | [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker) |
| `open_clip` | [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs) |
| `pandas` | [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui), [ui.autoreseach_ui.result_models](#ui-autoreseach-ui-result-models), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports) |
| `pascal_voc_writer` | [utils_training.yolo2voc](#utils-training-yolo2voc) |
| `psutil` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `sahi` | [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi) |
| `scipy` | [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics) |
| `seaborn` | [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `segmentation_models_pytorch` | [tests.test_model_contracts](#tests-test-model-contracts), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `shapely` | [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics) |
| `skimage` | [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `sklearn` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching), [utils_new.segmentation.operations](#utils-new-segmentation-operations), [utils_ops.interactive_ops](#utils-ops-interactive-ops), [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `test_report_evaluation` | [tests.test_report_workflow](#tests-test-report-workflow) |
| `torch` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_dataset_audit](#tests-test-dataset-audit), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_model_contracts](#tests-test-model-contracts), [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder), [utils_new.segmentation.model_export](#utils-new-segmentation-model-export), [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics), [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `torchvision` | [cluster.extract_embeddings](#cluster-extract-embeddings), [cluster.test_new_image](#cluster-test-new-image), [utils_aeb.auto-aeb_def](#utils-aeb-auto-aeb-def), [utils_aeb.auto-aeb_def_new](#utils-aeb-auto-aeb-def-new), [utils_aeb.auto-aeb_old](#utils-aeb-auto-aeb-old), [utils_aeb.auto-aeb_unet](#utils-aeb-auto-aeb-unet), [utils_aeb.auto-aeb_unet_with_grad](#utils-aeb-auto-aeb-unet-with-grad), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий) |
| `transformers` | [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs) |
| `ultralytics` | [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker) |
| `umap` | [cluster.cluster_embeddings](#cluster-cluster-embeddings), [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching) |
| `xgboost` | [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops) |
| `yaml` | [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tests.test_dataset_audit](#tests-test-dataset-audit), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_report_evaluation](#tests-test-report-evaluation), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.splits](#utils-new-dataset-audit-splits), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий), [utils_training.yolo2voc](#utils-training-yolo2voc) |

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

**Прямые потребители:** [tests.test_report_workflow](#tests-test-report-workflow).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`

**Внешние/stdlib импорты:** `importlib.import_module`.

**Объявления:**

- [`__getattr__(name)`](../import_libs_internal.py#L224)
- [`__dir__()`](../import_libs_internal.py#L238)

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
- [ui.dataset_source](#ui-dataset-source): `TAB_DESCRIPTIONS`
- [ui.image_navigation](#ui-image-navigation): `NavigationShortcutInstaller`
- [ui.settings_service](#ui-settings-service): `apply_theme`, `settings`
- [ui.settings_ui](#ui-settings-ui): `SettingsWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPainter`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSplashScreen`, `PyQt5.QtWidgets.QTabWidget`, `datetime.datetime`, `importlib.metadata.PackageNotFoundError`, `importlib.metadata.version`, `os`, `platform`, `sys`.

**Объявления:**

- Класс [`ImagePlayer`](../main.py#L30) (`QMainWindow`)
- [`main()`](../main.py#L341)

<a id="main-ui"></a>

## main_ui

Исходник: [main_ui.py](../main_ui.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `TAB_DESCRIPTIONS`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSettings`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`.

**Объявления:**

- Класс [`StartupDialog`](../main_ui.py#L34) (`QDialog`)

<a id="path-setup"></a>

## path_setup

Исходник: [path_setup.py](../path_setup.py#L1).

**Прямые потребители:** [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.settings_service](#ui-settings-service), [ui.settings_ui](#ui-settings-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `pathlib.Path`, `sys`.

**Объявления:**

- [`setup_project_path()`](../path_setup.py#L8) — Explicit compatibility helper; normal module imports need no mutation.
  Явно импортируют: [ui.settings_ui](#ui-settings-ui).
- [`get_project_root()`](../path_setup.py#L15) — Return the repository root without changing the import search path.
  Явно импортируют: [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [ui.settings_service](#ui-settings-service).

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

**Прямые потребители:** [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [dependency_health](#dependency-health), [import_libs_external](#import-libs-external), [import_libs_internal](#import-libs-internal), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `sys`.

**Объявления:**

- [`configure_ultralytics()`](../runtime_policy.py#L6) — Keep optional backend installation under install.py's control.
  Явно импортируют: [cluster.save_yolo_object_to_path](#cluster-save-yolo-object-to-path), [cluster.test_new_image](#cluster-test-new-image), [dependency_health](#dependency-health), [import_libs_external](#import-libs-external), [import_libs_internal](#import-libs-internal), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

<a id="runtime-torch"></a>

## runtime_torch

Исходник: [runtime_torch.py](../runtime_torch.py#L1).

**Прямые потребители:** [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_torch_runtime](#tests-test-torch-runtime), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `dataclasses.dataclass`, `importlib`, `sys`, `threading`.

**Объявления:**

- Класс [`TorchRuntime`](../runtime_torch.py#L9)
- [`get_torch_runtime()`](../runtime_torch.py#L21) — Cache failure as well as success; never retry a broken native loader.
  Явно импортируют: [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks).
- [`prepare_windows_torch()`](../runtime_torch.py#L43) — Load Torch before GUI/native stacks on Windows; keep Linux loading lazy.
  Явно импортируют: [main](#main), [tabs.dataset_analysis](#tabs-dataset-analysis), [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.deep_learning_training](#tabs-deep-learning-training), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml), [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`require_torch()`](../runtime_torch.py#L54) — Return the loaded module or report the cached failure without retrying.
  Явно импортируют: [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

<a id="tabs-dataset-analysis"></a>

## tabs.dataset_analysis

Исходник: [tabs/dataset_analysis.py](../tabs/dataset_analysis.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates): `DatasetDuplicatesController`
- [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings): `DatasetEmbeddingController`
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle): `close_workers`
- [ui.dataset_audit.models](#ui-dataset-audit-models): `DatasetModelController`
- [ui.dataset_audit.quality_controller](#ui-dataset-audit-quality-controller): `DatasetQualityController`
- [ui.dataset_audit.reporting](#ui-dataset-audit-reporting): `DatasetReportController`
- [ui.dataset_audit.selection](#ui-dataset-audit-selection): `DatasetSelectionController`
- [ui.dataset_audit.session](#ui-dataset-audit-session): `DatasetSessionController`
- [ui.dataset_audit.spatial](#ui-dataset-audit-spatial): `DatasetSpatialController`
- [ui.dataset_audit.tasks](#ui-dataset-audit-tasks): `ScanController`
- [ui.dataset_audit.training_results](#ui-dataset-audit-training-results): `DatasetTrainingResultsController`
- [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui): `setup_dataset_analysis_ui`
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `DatasetAnalyzer`
- [utils_new.dataset_loader_new](#utils-new-dataset-loader-new): `DatasetLoaderMixin`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `PairValidator`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QShortcut`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`DatasetAnalysisWindow`](../tabs/dataset_analysis.py#L34) (`DatasetSessionController, DatasetSelectionController, DatasetDuplicatesController, DatasetSpatialController, DatasetModelController, DatasetEmbeddingController, DatasetTrainingResultsController, DatasetReportController, DatasetQualityController, DatasetLoaderMixin, QMainWindow`)

<a id="tabs-deep-learning-methods"></a>

## tabs.deep_learning_methods

Исходник: [tabs/deep_learning_methods.py](../tabs/deep_learning_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.computation](#ui-segmentation-computation): `computation as segmentation_computation`
- [ui.segmentation.controller](#ui-segmentation-controller): `controller as segmentation_controller`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation as segmentation_presentation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `DeepLearningPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui): `setup_deep_learning_methods_ui`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QSpinBox`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`DeepLearningWindow`](../tabs/deep_learning_methods.py#L48) (`QMainWindow`)
- [`center_window(window, width=1200, height=800)`](../tabs/deep_learning_methods.py#L511)

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

**Внешние/stdlib импорты:** `PyQt5.QtCore.QProcess`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `codecs`, `importlib.util.find_spec`, `json`, `os`, `pathlib.Path`, `sys`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`YoloTrainWindow`](../tabs/deep_learning_training.py#L33) (`QMainWindow`)

<a id="tabs-demo-ensemble-methods"></a>

## tabs.demo_ensemble_methods

Исходник: [tabs/demo_ensemble_methods.py](../tabs/demo_ensemble_methods.py#L1).

**Прямые потребители:** [main](#main), [tests.test_ensemble_view](#tests-test-ensemble-view).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `load_images_universal`
- [ui.segmentation.jobs](#ui-segmentation-jobs): `BackgroundTask`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `ThresholdPresetManager`
- [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui): `setup_demo_ensemble_methods_ui`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `yolo_result_to_detections`
- [utils_new.annotations.boxes](#utils-new-annotations-boxes): `ensemble_detections`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_preset`, `get_detections_from_preset`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QWidget`, `concurrent.futures.ThreadPoolExecutor`, `copy.deepcopy`, `cv2`, `importlib.util.find_spec`, `os`, `pathlib.Path`, `sys`, `torch`, `traceback`, `ultralytics.YOLO`.

**Объявления:**

- [`print_flush(*args, **kwargs)`](../tabs/demo_ensemble_methods.py#L45)
- Класс [`YoloInferThread`](../tabs/demo_ensemble_methods.py#L58) (`BackgroundTask`)
- Класс [`EnsembleThread`](../tabs/demo_ensemble_methods.py#L132) (`BackgroundTask`)
- Класс [`YoloDemoWindow`](../tabs/demo_ensemble_methods.py#L223) (`QMainWindow`)

<a id="tabs-gradient-methods"></a>

## tabs.gradient_methods

Исходник: [tabs/gradient_methods.py](../tabs/gradient_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [path_setup](#path-setup): `get_project_root`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls): `SearchController`
- [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog): `AutoSettingsDialog`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.controller](#ui-segmentation-controller): `controller as segmentation_controller`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation as segmentation_presentation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `GradientPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui): `setup_gradient_ui`
- [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing): `finalize_gradient_mask`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`, `default_settings`, `search_kind`
- [utils_new.json_store](#utils-new-json-store): `read_json_object`, `write_json_atomic`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `cv2`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`GradientMethodsWindow`](../tabs/gradient_methods.py#L43) (`QMainWindow`)

<a id="tabs-interactive-methods"></a>

## tabs.interactive_methods

Исходник: [tabs/interactive_methods.py](../tabs/interactive_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.computation](#ui-segmentation-computation): `computation as segmentation_computation`
- [ui.segmentation.controller](#ui-segmentation-controller): `controller as segmentation_controller`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation as segmentation_presentation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `InteractivePresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui): `setup_interactive_ui`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `cv2`, `numpy`, `pathlib.Path`, `sys`, `traceback`.

**Объявления:**

- Класс [`InteractiveMethodsWindow`](../tabs/interactive_methods.py#L39) (`QMainWindow`)

<a id="tabs-layout-dataset"></a>

## tabs.layout_dataset

Исходник: [tabs/layout_dataset.py](../tabs/layout_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_view](#ui-image-view): `numpy_to_qpixmap`, `set_fill_color_button_preview`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui): `setup_layout_dataset_ui`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `collect_image_paths`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `build_yolo_save_target_map`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `canonicalize_obb_corners`, `format_label_save_confirmation`, `load_annotations_for_image`, `save_annotations_for_image`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `update_annotation_list`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QAction`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `traceback`, `ultralytics.YOLO`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/layout_dataset.py#L76) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/layout_dataset.py#L121) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/layout_dataset.py#L145) (`QThread`)
- Класс [`Labeler`](../tabs/layout_dataset.py#L196) (`QMainWindow`)

<a id="tabs-model-research"></a>

## tabs.model_research

Исходник: [tabs/model_research.py](../tabs/model_research.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller): `EvaluationController`
- [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui): `setup_model_research_ui`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `GradientPresetManager`, `ThresholdPresetManager`
- [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export): `export_evaluation`
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow): `input_guidance`, `selected_presets`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`ModelResearchApp`](../tabs/model_research.py#L29) (`QMainWindow`)

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
- [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls): `SearchController`
- [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog): `AutoSettingsDialog`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.controller](#ui-segmentation-controller): `controller as segmentation_controller`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation as segmentation_presentation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `ThresholdPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui): `setup_threshold_ui`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`, `default_settings`, `search_kind`
- [utils_new.json_store](#utils-new-json-store): `read_json_object`, `write_json_atomic`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QMainWindow`, `cv2`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`ThresholdWindow`](../tabs/threshold_methods.py#L47) (`QMainWindow`)

<a id="tabs-traditional-ml-methods"></a>

## tabs.traditional_ml_methods

Исходник: [tabs/traditional_ml_methods.py](../tabs/traditional_ml_methods.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.segmentation.actions](#ui-segmentation-actions): `actions as segmentation_actions`
- [ui.segmentation.computation](#ui-segmentation-computation): `computation as segmentation_computation`
- [ui.segmentation.controller](#ui-segmentation-controller): `controller as segmentation_controller`
- [ui.segmentation.export](#ui-segmentation-export): `export as segmentation_export`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram as segmentation_histogram`
- [ui.segmentation.image_input](#ui-segmentation-image-input): `image_input as segmentation_input`
- [ui.segmentation.navigation](#ui-segmentation-navigation): `navigation as segmentation_navigation`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation as segmentation_presentation`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `TraditionalMLPresetManager`
- [ui.segmentation.results](#ui-segmentation-results): `results as segmentation_results`
- [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui): `setup_traditional_ml_ui`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops): `create_superpixels`, `extract_pixel_features`, `extract_superpixel_features`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSpinBox`, `joblib`, `numpy`, `os`, `pathlib.Path`, `sys`.

**Объявления:**

- Класс [`TraditionalMLWindow`](../tabs/traditional_ml_methods.py#L51) (`QMainWindow`)

<a id="tabs-viewing-dataset"></a>

## tabs.viewing_dataset

Исходник: [tabs/viewing_dataset.py](../tabs/viewing_dataset.py#L1).

**Прямые потребители:** [main](#main).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_view](#ui-image-view): `numpy_to_qpixmap`, `set_fill_color_button_preview`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui): `setup_viewing_dataset_ui`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `class_id_for_annotation`, `mask_roi_at_image_scale`, `normalized_annotation`, `prepare_annotation_preview`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `import_coco_catalog`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `collect_image_paths`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `read_index_mask`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `load_unet_yaml_pairs`, `write_yolo_txt_for_unet_pairs`
- [utils_new.dataset_yaml_new](#utils-new-dataset-yaml-new): `build_yolo_save_target_map`, `find_mask_for_image`, `load_yolo_yaml`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `canonicalize_obb_corners`, `format_label_save_confirmation`, `load_annotations_for_image`, `save_annotations_for_image`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QThread`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPalette`, `PyQt5.QtWidgets.QAction`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QColorDialog`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `collections.OrderedDict`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `traceback`.

**Объявления:**

- [`_validate_unified_annotation(ann)`](../tabs/viewing_dataset.py#L86) — Возвращает (is_valid, err_msg).
- [`_detect_annotation_to_obb(ann)`](../tabs/viewing_dataset.py#L131) — ('detect', cls, cx, cy, w, h) → ('obb', cls, [x1,y1,x2,y2,x3,y3,x4,y4])
- Класс [`LabelValidationThread`](../tabs/viewing_dataset.py#L155) (`QThread`)
- Класс [`ViewingDataset`](../tabs/viewing_dataset.py#L206) (`QMainWindow`)

<a id="tabs-yolo-find-img"></a>

## tabs.yolo_find_img

Исходник: [tabs/yolo_find_img.py](../tabs/yolo_find_img.py#L1).

**Прямые потребители:** [main](#main), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.jobs](#ui-segmentation-jobs): `BackgroundTask`
- [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui): `setup_yolo_find_img_ui`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `yolo_result_to_detections`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `collect_image_paths`, `load_one_image_item`, `normalize_to_uint8`, `read_image_with_fallback`, `resize_to_max_side`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QDir`, `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.QUrl`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QCursor`, `PyQt5.QtGui.QDesktopServices`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFileSystemModel`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `concurrent.futures.ThreadPoolExecutor`, `os`, `pathlib.Path`, `shutil`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`ClickableLabel`](../tabs/yolo_find_img.py#L54) (`QLabel`)
- Класс [`ScanThread`](../tabs/yolo_find_img.py#L119) (`BackgroundTask`)
- Класс [`FindImagesWindow`](../tabs/yolo_find_img.py#L290) (`QMainWindow, setup_yolo_find_img_ui`)

<a id="tabs-yolo-test-aeb"></a>

## tabs.yolo_test_aeb

Исходник: [tabs/yolo_test_aeb.py](../tabs/yolo_test_aeb.py#L1).

**Прямые потребители:** [main](#main), [tests.test_segmentation_shared](#tests-test-segmentation-shared).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `prepare_windows_torch`
- [ui.application](#ui-application): `exec_window`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.jobs](#ui-segmentation-jobs): `BackgroundTask`
- [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings): `PresetManager`
- [ui.settings_service](#ui-settings-service): `settings`
- [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui): `setup_yolo_ae_vae_ui`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `aeb_transform_image`, `cls_suffix_from_detection`, `detections_to_unified`, `longest_edge_angle_deg`, `render_ae_error_map`, `render_unified_annotations`, `yolo_result_to_detections`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache`, `load_one_image_item`, `write_image`
- [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder): `build_ae_model_from_cfg`, `detect_arch`, `infer_autoencoder`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `image_paths`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QInputDialog`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMainWindow`, `PyQt5.QtWidgets.QMessageBox`, `concurrent.futures.ThreadPoolExecutor`, `copy.deepcopy`, `cv2`, `numpy`, `os`, `pathlib.Path`, `sys`, `torch`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`YoloAEBPresetManager`](../tabs/yolo_test_aeb.py#L79) (`PresetManager`)
- Класс [`YoloInspectWindow`](../tabs/yolo_test_aeb.py#L177) (`QMainWindow`)

<a id="tests-test-annotation-objects"></a>

## tests.test_annotation_objects

Исходник: [tests/test_annotation_objects.py](../tests/test_annotation_objects.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `from_legacy`
- [utils_new.annotations.object_info](#utils-new-annotations-object-info): `ObjectInfo`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `serialize_yolo_labels`, `write_yolo_labels`

**Внешние/stdlib импорты:** `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`ObjectInfoTests`](../tests/test_annotation_objects.py#L19) (`unittest.TestCase`)
- Класс [`ObjectInspectorGuiTests`](../tests/test_annotation_objects.py#L63) (`unittest.TestCase`)

<a id="tests-test-dataset-audit"></a>

## tests.test_dataset_audit

Исходник: [tests/test_dataset_audit.py](../tests/test_dataset_audit.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `ClassImbalanceAnalyzer`, `DatasetAnalyzer`, `FilterParams`, `get_image_brightness`
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks): `DuplicateDetector`, `LeakageChecker`, `ObjectDistributionHeatmap`
- [utils_new.dataset_audit.export](#utils-new-dataset-audit-export): `export_subset`, `infer_split`
- [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs): `EmbeddingClusterJob`, `LossInspectionJob`, `ReportJob`, `YoloValidationJob`
- [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching): `match_detections`, `project_embeddings`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `load_pair_annotations`, `read_objects`
- [utils_new.dataset_audit.splits](#utils-new-dataset-audit-splits): `split_memberships`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `load_unet_catalog`

**Внешние/stdlib импорты:** `contextlib.contextmanager`, `cv2`, `logging`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `torch`, `types`, `unittest`, `unittest.mock.patch`, `yaml`.

**Объявления:**

- [`injected_module(name, module)`](../tests/test_dataset_audit.py#L34)
- Класс [`DatasetAuditTests`](../tests/test_dataset_audit.py#L46) (`unittest.TestCase`)
- Класс [`DatasetAuditGuiTests`](../tests/test_dataset_audit.py#L452) (`unittest.TestCase`)

<a id="tests-test-dataset-workspace"></a>

## tests.test_dataset_workspace

Исходник: [tests/test_dataset_workspace.py](../tests/test_dataset_workspace.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `find_strong_overlaps`
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `histogram_edges`
- [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review): `AnnotationReviewJob`
- [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality): `QualityJob`
- [utils_new.dataset_audit.scans](#utils-new-dataset-audit-scans): `ScanJob`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`DatasetWorkspaceTests`](../tests/test_dataset_workspace.py#L21) (`unittest.TestCase`)
- Класс [`DatasetWorkspaceGuiTests`](../tests/test_dataset_workspace.py#L109) (`unittest.TestCase`)

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

<a id="tests-test-ensemble-view"></a>

## tests.test_ensemble_view

Исходник: [tests/test_ensemble_view.py](../tests/test_ensemble_view.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods): `YoloInferThread`

**Внешние/stdlib импорты:** `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `types.SimpleNamespace`, `unittest`, `unittest.mock.Mock`, `unittest.mock.patch`.

**Объявления:**

- Класс [`EnsembleWorkerDisplayTests`](../tests/test_ensemble_view.py#L17) (`unittest.TestCase`)
- Класс [`EnsembleViewGuiTests`](../tests/test_ensemble_view.py#L62) (`unittest.TestCase`)

<a id="tests-test-gradient-autoresearch"></a>

## tests.test_gradient_autoresearch

Исходник: [tests/test_gradient_autoresearch.py](../tests/test_gradient_autoresearch.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation): `evaluate_dataset`
- [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing): `gradient_processing`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `parse_reports`, `preset_from_row`, `read_report`
- [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search): `RawMaskCache`, `evaluate`, `run_search`, `search`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`, `default_settings`, `iter_presets`, `method_params`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`

**Внешние/stdlib импорты:** `cv2`, `json`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `threading.Event`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`GradientSearchTests`](../tests/test_gradient_autoresearch.py#L26) (`unittest.TestCase`)
- Класс [`UnifiedSearchGuiTests`](../tests/test_gradient_autoresearch.py#L202) (`unittest.TestCase`)

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

<a id="tests-test-legacy-tabs-migration"></a>

## tests.test_legacy_tabs_migration

Исходник: [tests/test_legacy_tabs_migration.py](../tests/test_legacy_tabs_migration.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [tabs.yolo_find_img](#tabs-yolo-find-img): `ScanThread`
- [utils_new.annotations.boxes](#utils-new-annotations-boxes): `ensemble_detections`
- [utils_new.segmentation.autoencoder](#utils-new-segmentation-autoencoder): `build_ae_model_from_cfg`, `infer_autoencoder`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `make_preset`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `get_detections_from_preset`

**Внешние/stdlib импорты:** `ast`, `cv2`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `torch`, `types.SimpleNamespace`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`EnsembleAndSearchTests`](../tests/test_legacy_tabs_migration.py#L23) (`unittest.TestCase`)
- Класс [`AutoencoderTests`](../tests/test_legacy_tabs_migration.py#L173) (`unittest.TestCase`)
- Класс [`MigratedTabsGuiTests`](../tests/test_legacy_tabs_migration.py#L230) (`unittest.TestCase`)

<a id="tests-test-method-algorithms"></a>

## tests.test_method_algorithms

Исходник: [tests/test_method_algorithms.py](../tests/test_method_algorithms.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.segmentation.operations](#utils-new-segmentation-operations): `interactive`, `operations`
- [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels): `manual_target`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`, `niblack_threshold`, `row_adaptive_threshold`, `sauvola_threshold`, `threshold_processing`
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`
- [utils_ops.interactive_ops](#utils-ops-interactive-ops): `interactive_ops as ops`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `segmentation_ops`
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops): `compute_elbow_kmeans`, `create_superpixels`, `find_optimal_k_elbow`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `pathlib.Path`, `skimage.filters.threshold_niblack`, `skimage.filters.threshold_sauvola`, `tempfile`, `threading`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`InteractiveAlgorithms`](../tests/test_method_algorithms.py#L13) (`unittest.TestCase`)
- Класс [`ThresholdAndGradientAlgorithms`](../tests/test_method_algorithms.py#L135) (`unittest.TestCase`)
- Класс [`TraditionalLearningContracts`](../tests/test_method_algorithms.py#L221) (`unittest.TestCase`)

<a id="tests-test-model-contracts"></a>

## tests.test_model_contracts

Исходник: [tests/test_model_contracts.py](../tests/test_model_contracts.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.segmentation.model_export](#utils-new-segmentation-model-export): `stride_padded_model`
- [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics): `segmentation_metrics`
- [utils_new.segmentation.model_output](#utils-new-segmentation-model-output): `decode_output`
- [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts): `apply_decoder_dropout`, `validate_mask_files`
- [utils_new.segmentation.yolo_metrics](#utils-new-segmentation-yolo-metrics): `validation_metrics`
- [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops): `DeepLabV3Segmentor`, `ONNXSegmentor`, `UNetSegmentor`, `YOLOSegmentor`
- [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker): `VOCSegmentationPairs`, `get_loss_function`, `get_loss_function as deeplab_loss`
- [utils_training.train_unet_worker](#utils-training-train-unet-worker): `SegmentationDataset`, `get_loss_function`

**Внешние/stdlib импорты:** `PIL.Image`, `importlib.util.find_spec`, `json`, `numpy`, `onnx`, `onnx.TensorProto`, `onnx.helper`, `pathlib.Path`, `segmentation_models_pytorch`, `tempfile`, `torch`, `types.SimpleNamespace`, `unittest`.

**Объявления:**

- Класс [`OutputContracts`](../tests/test_model_contracts.py#L17) (`unittest.TestCase`)
- Класс [`RealTorchContracts`](../tests/test_model_contracts.py#L83) (`unittest.TestCase`)
- Класс [`SemanticMetrics`](../tests/test_model_contracts.py#L217) (`unittest.TestCase`)
- Класс [`RealONNXContracts`](../tests/test_model_contracts.py#L264) (`unittest.TestCase`)
- Класс [`YOLOContracts`](../tests/test_model_contracts.py#L303) (`unittest.TestCase`)
- Класс [`ExportGeometryContracts`](../tests/test_model_contracts.py#L365) (`unittest.TestCase`)
- Класс [`BinaryVoidContracts`](../tests/test_model_contracts.py#L428) (`unittest.TestCase`)

<a id="tests-test-pixel-morphology"></a>

## tests.test_pixel_morphology

Исходник: [tests/test_pixel_morphology.py](../tests/test_pixel_morphology.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `make_preset`, `validate_preset`
- [utils_new.settings_store](#utils-new-settings-store): `load_settings`, `normalize_settings`, `save_settings`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `get_detections_from_preset`

**Внешние/stdlib импорты:** `ast`, `copy`, `cv2`, `json`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`PixelPresetTests`](../tests/test_pixel_morphology.py#L42) (`unittest.TestCase`)
- Класс [`PixelMorphologyGuiTests`](../tests/test_pixel_morphology.py#L142) (`unittest.TestCase`)

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

<a id="tests-test-report-evaluation"></a>

## tests.test_report_evaluation

Исходник: [tests/test_report_evaluation.py](../tests/test_report_evaluation.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation): `evaluate_dataset`
- [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export): `export_evaluation`
- [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics): `evaluate_image`, `match_overlaps`, `normalize_annotations`, `parse_sequence`
- [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth): `digest_file`, `load_ground_truth`, `match_sample`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `parse_reports`, `read_report`
- [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search): `run_search`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `default_settings`, `iter_presets`

**Внешние/stdlib импорты:** `cv2`, `json`, `numpy`, `os`, `pandas`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `threading.Event`, `unittest`, `unittest.mock.patch`, `yaml`.

**Объявления:**

- [`preset(kind='threshold', value=80, mode='detect')`](../tests/test_report_evaluation.py#L37)
- Класс [`ReportEvaluationTests`](../tests/test_report_evaluation.py#L45) (`unittest.TestCase`)
- Класс [`ReportEvaluationGuiTests`](../tests/test_report_evaluation.py#L357) (`unittest.TestCase`)

<a id="tests-test-report-workflow"></a>

## tests.test_report_workflow

Исходник: [tests/test_report_workflow.py](../tests/test_report_workflow.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [import_libs_internal](#import-libs-internal): `(модуль)`
- [utils_new.annotations.boxes](#utils-new-annotations-boxes): `iou_xyxy`, `non_max_suppression`, `xyxy_to_yolo`, `yolo_to_xyxy`
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation): `evaluate_dataset`
- [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth): `load_ground_truth`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `read_report`
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow): `discover_reports`, `input_guidance`, `load_collection`, `result_guidance`, `selected_presets`

**Внешние/stdlib импорты:** `ast`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `test_report_evaluation`, `threading.Event`, `unittest`.

**Объявления:**

- Класс [`WorkflowTests`](../tests/test_report_workflow.py#L29) (`unittest.TestCase`)
- Класс [`WorkflowGuiTests`](../tests/test_report_workflow.py#L162) (`unittest.TestCase`)

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
- [tabs.yolo_test_aeb](#tabs-yolo-test-aeb): `LazyImageCache`
- [ui.annotation_review_view](#ui-annotation-review-view): `(модуль)`
- [ui.image_view](#ui-image-view): `(модуль)`
- [ui.settings_service](#ui-settings-service): `(модуль)`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `LazyImageCache as canonical`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`
- [utils_new.presets_store](#utils-new-presets-store): `read_presets`, `write_presets`
- [utils_new.segmentation.export](#utils-new-segmentation-export): `write_annotations_atomic`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `image_session`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `postprocessing`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `segment_contours`, `segment_min_area_rect`, `segment_projections`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`SegmentationCommonTests`](../tests/test_segmentation_shared.py#L21) (`unittest.TestCase`)
- Класс [`SegmentationSharedGuiTests`](../tests/test_segmentation_shared.py#L147) (`unittest.TestCase`)

<a id="tests-test-segmentation-tasks"></a>

## tests.test_segmentation_tasks

Исходник: [tests/test_segmentation_tasks.py](../tests/test_segmentation_tasks.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.segmentation.operations](#utils-new-segmentation-operations): `operations`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `binary_mask`
- [utils_ops.interactive_ops](#utils-ops-interactive-ops): `interactive_ops`

**Внешние/stdlib импорты:** `concurrent.futures.CancelledError`, `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `threading.Event`, `types.SimpleNamespace`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`SegmentationTaskGuiTests`](../tests/test_segmentation_tasks.py#L35) (`unittest.TestCase`)
- Класс [`NumericalOperationTests`](../tests/test_segmentation_tasks.py#L213) (`unittest.TestCase`)

<a id="tests-test-segmentation-unification"></a>

## tests.test_segmentation_unification

Исходник: [tests/test_segmentation_unification.py](../tests/test_segmentation_unification.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `from_legacy`
- [utils_new.annotations.modes](#utils-new-annotations-modes): `MODE_LABELS`, `supports_hull`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `binary_mask`, `masked_preview`

**Внешние/stdlib импорты:** `numpy`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `unittest`.

**Объявления:**

- Класс [`UnifiedCoreTests`](../tests/test_segmentation_unification.py#L19) (`unittest.TestCase`)
- Класс [`UnifiedGuiTests`](../tests/test_segmentation_unification.py#L47) (`unittest.TestCase`)

<a id="tests-test-source-unification"></a>

## tests.test_source_unification

Исходник: [tests/test_source_unification.py](../tests/test_source_unification.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `TAB_DESCRIPTIONS`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `collect_image_paths`

**Внешние/stdlib импорты:** `ast`, `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `textwrap`, `unittest`.

**Объявления:**

- Класс [`SourceContracts`](../tests/test_source_unification.py#L17) (`unittest.TestCase`)
- Класс [`SourceGuiTests`](../tests/test_source_unification.py#L64) (`unittest.TestCase`)

<a id="tests-test-threshold-autoresearch"></a>

## tests.test_threshold_autoresearch

Исходник: [tests/test_threshold_autoresearch.py](../tests/test_threshold_autoresearch.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation): `evaluate_dataset`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `parse_reports`, `preset_from_row`, `read_report`
- [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search): `evaluate`, `run_search`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`, `default_settings`, `iter_presets`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`

**Внешние/stdlib импорты:** `cv2`, `json`, `numpy`, `os`, `pandas`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `threading.Event`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`SearchTests`](../tests/test_threshold_autoresearch.py#L25) (`unittest.TestCase`)
- Класс [`SearchGuiTests`](../tests/test_threshold_autoresearch.py#L177) (`unittest.TestCase`)

<a id="tests-test-torch-runtime"></a>

## tests.test_torch_runtime

Исходник: [tests/test_torch_runtime.py](../tests/test_torch_runtime.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `(модуль)`

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `subprocess`, `sys`, `tempfile`, `types.SimpleNamespace`, `unittest`, `unittest.mock.Mock`, `unittest.mock.patch`.

**Объявления:**

- Класс [`TorchRuntimeTests`](../tests/test_torch_runtime.py#L16) (`unittest.TestCase`)

<a id="tests-test-yolo-worker"></a>

## tests.test_yolo_worker

Исходник: [tests/test_yolo_worker.py](../tests/test_yolo_worker.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_training.train_yolo_worker](#utils-training-train-yolo-worker): `train_yolo_worker as worker`

**Внешние/stdlib импорты:** `contextlib`, `io`, `json`, `os`, `pathlib.Path`, `signal`, `subprocess`, `sys`, `tempfile`, `types.ModuleType`, `types.SimpleNamespace`, `unittest`, `unittest.mock.patch`.

**Объявления:**

- Класс [`YoloWorkerTests`](../tests/test_yolo_worker.py#L21) (`unittest.TestCase`)
- Класс [`YoloWorkerGuiTests`](../tests/test_yolo_worker.py#L316) (`unittest.TestCase`)

<a id="ui-DL-predict-settings"></a>

## ui.DL_predict_settings

Исходник: [ui/DL_predict_settings.py](../ui/DL_predict_settings.py#L1).

**Прямые потребители:** [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`BaseInferenceSettings`](../ui/DL_predict_settings.py#L19) (`QGroupBox`)
- Класс [`YOLOInferenceSettings`](../ui/DL_predict_settings.py#L43) (`BaseInferenceSettings`)
- Класс [`UNetInferenceSettings`](../ui/DL_predict_settings.py#L130) (`BaseInferenceSettings`)
- Класс [`DeepLabV3InferenceSettings`](../ui/DL_predict_settings.py#L475) (`BaseInferenceSettings`)
- Класс [`SegFormerInferenceSettings`](../ui/DL_predict_settings.py#L862) (`BaseInferenceSettings`)
- Класс [`SAMInferenceSettings`](../ui/DL_predict_settings.py#L899) (`BaseInferenceSettings`)
- Класс [`CustomONNXInferenceSettings`](../ui/DL_predict_settings.py#L940) (`BaseInferenceSettings`)

<a id="ui-DL-train-settings-base"></a>

## ui.DL_train_settings_base

Исходник: [ui/DL_train_settings_base.py](../ui/DL_train_settings_base.py#L1).

**Прямые потребители:** [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `os`, `sys`, `yaml`.

**Объявления:**

- Класс [`BaseTrainWidget`](../ui/DL_train_settings_base.py#L28) (`QWidget`)
- Класс [`YOLOTrainWidget`](../ui/DL_train_settings_base.py#L173) (`BaseTrainWidget`)
- Класс [`UNetTrainWidget`](../ui/DL_train_settings_base.py#L420) (`BaseTrainWidget`)

<a id="ui-DL-train-settings-new-model"></a>

## ui.DL_train_settings_new_model

Исходник: [ui/DL_train_settings_new_model.py](../ui/DL_train_settings_new_model.py#L1).

**Прямые потребители:** [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui).

**Прямые локальные импорты:**
- [ui.DL_train_settings_base](#ui-DL-train-settings-base): `BaseTrainWidget`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `ast`, `os`.

**Объявления:**

- Класс [`DeepLabV3TrainWidget`](../ui/DL_train_settings_new_model.py#L30) (`BaseTrainWidget`)
- Класс [`SegFormerTrainWidget`](../ui/DL_train_settings_new_model.py#L1027) (`BaseTrainWidget`)
- Класс [`SAMTrainWidget`](../ui/DL_train_settings_new_model.py#L1166) (`BaseTrainWidget`)

<a id="ui-annotation-objects"></a>

## ui.annotation_objects

Исходник: [ui/annotation_objects.py](../ui/annotation_objects.py#L1).

**Прямые потребители:** [ui.segmentation.panels](#ui-segmentation-panels).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `ImageSize`
- [utils_new.annotations.modes](#utils-new-annotations-modes): `MODE_LABELS`
- [utils_new.annotations.object_info](#utils-new-annotations-object-info): `ObjectInfo`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `serialize_yolo_labels`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QAbstractTableModel`, `PyQt5.QtCore.QModelIndex`, `PyQt5.QtCore.QSortFilterProxyModel`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QTableView`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `numpy`.

**Объявления:**

- Класс [`ObjectTableModel`](../ui/annotation_objects.py#L35) (`QAbstractTableModel`)
- Класс [`ObjectFilter`](../ui/annotation_objects.py#L150) (`QSortFilterProxyModel`)
- Класс [`VertexTableModel`](../ui/annotation_objects.py#L165) (`QAbstractTableModel`)
- Класс [`AnnotationObjectsWidget`](../ui/annotation_objects.py#L195) (`QWidget`)

<a id="ui-annotation-review"></a>

## ui.annotation_review

Исходник: [ui/annotation_review.py](../ui/annotation_review.py#L1).

**Прямые потребители:** [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [ui.dataset_audit.workers](#ui-dataset-audit-workers): `AuditWorker`
- [utils_new.annotation_processing](#utils-new-annotation-processing): `find_strong_overlaps`
- [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review): `AnnotationReviewJob`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSettings`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QIcon`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QStyle`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `pathlib.Path`.

**Объявления:**

- Класс [`ReviewWorker`](../ui/annotation_review.py#L26) (`AuditWorker`)
- Класс [`AnnotationReviewWidget`](../ui/annotation_review.py#L30) (`QWidget`)

<a id="ui-annotation-review-view"></a>

## ui.annotation_review_view

Исходник: [ui/annotation_review_view.py](../ui/annotation_review_view.py#L1).

**Прямые потребители:** [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.panels](#ui-segmentation-panels), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

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

<a id="ui-autoreseach-ui---init--"></a>

## ui.autoreseach_ui.__init__

Исходник: [ui/autoreseach_ui/__init__.py](../ui/autoreseach_ui/__init__.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="ui-autoreseach-ui-controls"></a>

## ui.autoreseach_ui.controls

Исходник: [ui/autoreseach_ui/controls.py](../ui/autoreseach_ui/controls.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search): `run_search`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `concurrent.futures.ThreadPoolExecutor`, `copy.deepcopy`, `threading.Event`, `time.monotonic`.

**Объявления:**

- Класс [`AutoResearchPanel`](../ui/autoreseach_ui/controls.py#L21) (`QWidget`)
- Класс [`SearchController`](../ui/autoreseach_ui/controls.py#L36) (`QObject`)

<a id="ui-autoreseach-ui-evaluation-controller"></a>

## ui.autoreseach_ui.evaluation_controller

Исходник: [ui/autoreseach_ui/evaluation_controller.py](../ui/autoreseach_ui/evaluation_controller.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation): `evaluate_dataset`
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow): `load_collection`, `load_folder_collection`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `concurrent.futures.ThreadPoolExecutor`, `threading.Event`, `time.monotonic`.

**Объявления:**

- Класс [`EvaluationController`](../ui/autoreseach_ui/evaluation_controller.py#L11) (`QObject`)

<a id="ui-autoreseach-ui-evaluation-controls"></a>

## ui.autoreseach_ui.evaluation_controls

Исходник: [ui/autoreseach_ui/evaluation_controls.py](../ui/autoreseach_ui/evaluation_controls.py#L1).

**Прямые потребители:** [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`, `annotation_format_combo`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`hint(text)`](../ui/autoreseach_ui/evaluation_controls.py#L21)
  Явно импортируют: [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).
- [`setup_evaluation_controls(owner, layout)`](../ui/autoreseach_ui/evaluation_controls.py#L28)
  Явно импортируют: [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).

<a id="ui-autoreseach-ui-model-research-ui"></a>

## ui.autoreseach_ui.model_research_ui

Исходник: [ui/autoreseach_ui/model_research_ui.py](../ui/autoreseach_ui/model_research_ui.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research).

**Прямые локальные импорты:**
- [ui.autoreseach_ui.evaluation_controls](#ui-autoreseach-ui-evaluation-controls): `hint`, `setup_evaluation_controls`
- [ui.autoreseach_ui.result_models](#ui-autoreseach-ui-result-models): `ComparisonTable`, `IMAGE_COLUMNS`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export): `preset_settings_text`
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow): `result_guidance`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSignalBlocker`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTextBrowser`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `pandas`.

**Объявления:**

- Класс [`ComparisonResultsWidget`](../ui/autoreseach_ui/model_research_ui.py#L26) (`QWidget`)
- [`setup_model_research_ui(owner)`](../ui/autoreseach_ui/model_research_ui.py#L255)
  Явно импортируют: [tabs.model_research](#tabs-model-research).

<a id="ui-autoreseach-ui-result-models"></a>

## ui.autoreseach_ui.result_models

Исходник: [ui/autoreseach_ui/result_models.py](../ui/autoreseach_ui/result_models.py#L1).

**Прямые потребители:** [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).

**Прямые локальные импорты:**
- [utils_new.annotations.modes](#utils-new-annotations-modes): `MODE_LABELS`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QAbstractTableModel`, `PyQt5.QtCore.QModelIndex`, `PyQt5.QtCore.QSortFilterProxyModel`, `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QTableView`, `math`, `pandas`.

**Объявления:**

- Класс [`ComparisonTableModel`](../ui/autoreseach_ui/result_models.py#L90) (`QAbstractTableModel`)
- Класс [`ComparisonTable`](../ui/autoreseach_ui/result_models.py#L174) (`QTableView`)

<a id="ui-autoreseach-ui-settings-dialog"></a>

## ui.autoreseach_ui.settings_dialog

Исходник: [ui/autoreseach_ui/settings_dialog.py](../ui/autoreseach_ui/settings_dialog.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `POST_SPECS`, `combination_count`, `default_settings`, `search_kind`, `specs_for`
- [utils_new.morphology](#utils-new-morphology): `SUPPORTED_OPERATIONS`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QDialogButtonBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `copy.deepcopy`.

**Объявления:**

- Класс [`AutoSettingsDialog`](../ui/autoreseach_ui/settings_dialog.py#L30) (`QDialog`)

<a id="ui-control-buttons"></a>

## ui.control_buttons

Исходник: [ui/control_buttons.py](../ui/control_buttons.py#L1).

**Прямые потребители:** [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`ControlButtons`](../ui/control_buttons.py#L6) (`QWidget`)

<a id="ui-dataset-audit---init--"></a>

## ui.dataset_audit.__init__

Исходник: [ui/dataset_audit/__init__.py](../ui/dataset_audit/__init__.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="ui-dataset-audit-duplicates"></a>

## ui.dataset_audit.duplicates

Исходник: [ui/dataset_audit/duplicates.py](../ui/dataset_audit/duplicates.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `TrainingImageLabel`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `load_pair_annotations`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_image`, `read_index_mask`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QProgressDialog`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `cv2`, `numpy`, `os`.

**Объявления:**

- Класс [`DatasetDuplicatesController`](../ui/dataset_audit/duplicates.py#L29)

<a id="ui-dataset-audit-embeddings"></a>

## ui.dataset_audit.embeddings

Исходник: [ui/dataset_audit/embeddings.py](../ui/dataset_audit/embeddings.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle): `stale`
- [ui.dataset_audit.workers](#ui-dataset-audit-workers): `EmbeddingClusterWorker`
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks): `CLIP_AVAILABLE`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QMessageBox`, `collections.Counter`, `matplotlib.pyplot`, `numpy`, `os`.

**Объявления:**

- Класс [`DatasetEmbeddingController`](../ui/dataset_audit/embeddings.py#L12)

<a id="ui-dataset-audit-lifecycle"></a>

## ui.dataset_audit.lifecycle

Исходник: [ui/dataset_audit/lifecycle.py](../ui/dataset_audit/lifecycle.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.reporting](#ui-dataset-audit-reporting), [ui.dataset_audit.selection](#ui-dataset-audit-selection).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QAbstractButton`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QAbstractSpinBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QSlider`, `functools.wraps`, `inspect.signature`.

**Объявления:**

- [`serialized_scan(function)`](../ui/dataset_audit/lifecycle.py#L17)
  Явно импортируют: [ui.dataset_audit.selection](#ui-dataset-audit-selection).
- [`stale(owner, result=None)`](../ui/dataset_audit/lifecycle.py#L74)
  Явно импортируют: [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.reporting](#ui-dataset-audit-reporting).
- [`close_workers(owner)`](../ui/dataset_audit/lifecycle.py#L89)
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis).

<a id="ui-dataset-audit-models"></a>

## ui.dataset_audit.models

Исходник: [ui/dataset_audit/models.py](../ui/dataset_audit/models.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `require_torch`
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle): `stale`
- [ui.dataset_audit.workers](#ui-dataset-audit-workers): `LossInspectionWorker`, `YoloValidationWorker`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `load_pair_annotations`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_image`, `read_index_mask`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QTableWidgetItem`, `PyQt5.QtWidgets.QVBoxLayout`, `cv2`, `numpy`, `os`, `subprocess`, `sys`, `ultralytics.YOLO`.

**Объявления:**

- Класс [`DatasetModelController`](../ui/dataset_audit/models.py#L28)

<a id="ui-dataset-audit-page-duplicates"></a>

## ui.dataset_audit.page_duplicates

Исходник: [ui/dataset_audit/page_duplicates.py](../ui/dataset_audit/page_duplicates.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`build_duplicates(parent)`](../ui/dataset_audit/page_duplicates.py#L18)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-embeddings"></a>

## ui.dataset_audit.page_embeddings

Исходник: [ui/dataset_audit/page_embeddings.py](../ui/dataset_audit/page_embeddings.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.backends.backend_qt5agg.NavigationToolbar2QT`, `matplotlib.figure.Figure`.

**Объявления:**

- [`build_embeddings(parent)`](../ui/dataset_audit/page_embeddings.py#L18)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-helpers"></a>

## ui.dataset_audit.page_helpers

Исходник: [ui/dataset_audit/page_helpers.py](../ui/dataset_audit/page_helpers.py#L1).

**Прямые потребители:** [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.page_duplicates](#ui-dataset-audit-page-duplicates), [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings), [ui.dataset_audit.page_models](#ui-dataset-audit-page-models), [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality), [ui.dataset_audit.page_report](#ui-dataset-audit-page-report), [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial), [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics), [ui.dataset_audit.page_training](#ui-dataset-audit-page-training), [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSize`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `html.escape`, `re`.

**Объявления:**

- Класс [`TrainingImageLabel`](../ui/dataset_audit/page_helpers.py#L21) (`QLabel`)
- [`_make_page(toolbox, title, description, content, buttons, min_height=240, content_stretch=1)`](../ui/dataset_audit/page_helpers.py#L60)
  Явно импортируют: [ui.dataset_audit.page_duplicates](#ui-dataset-audit-page-duplicates), [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings), [ui.dataset_audit.page_models](#ui-dataset-audit-page-models), [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality), [ui.dataset_audit.page_report](#ui-dataset-audit-page-report), [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial), [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics), [ui.dataset_audit.page_training](#ui-dataset-audit-page-training).
- [`_mk_min_spin(rng, suffix='', special='Нет', tooltip='')`](../ui/dataset_audit/page_helpers.py#L99)
  Явно импортируют: [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).
- [`_mk_max_spin(rng, suffix='', special='Нет', tooltip='')`](../ui/dataset_audit/page_helpers.py#L112)
  Явно импортируют: [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).
- [`_mk_min_dspin(rng, decimals=2, suffix='', special='Нет', tooltip='')`](../ui/dataset_audit/page_helpers.py#L125)
  Явно импортируют: [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).
- [`_mk_max_dspin(rng, decimals=2, suffix='', special='Нет', tooltip='')`](../ui/dataset_audit/page_helpers.py#L139)
  Явно импортируют: [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).

<a id="ui-dataset-audit-page-models"></a>

## ui.dataset_audit.page_models

Исходник: [ui/dataset_audit/page_models.py](../ui/dataset_audit/page_models.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`, `make_source_button`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`build_models(parent)`](../ui/dataset_audit/page_models.py#L23)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-quality"></a>

## ui.dataset_audit.page_quality

Исходник: [ui/dataset_audit/page_quality.py](../ui/dataset_audit/page_quality.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`build_quality(parent)`](../ui/dataset_audit/page_quality.py#L21)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-report"></a>

## ui.dataset_audit.page_report

Исходник: [ui/dataset_audit/page_report.py](../ui/dataset_audit/page_report.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`build_report(parent)`](../ui/dataset_audit/page_report.py#L8)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-spatial"></a>

## ui.dataset_audit.page_spatial

Исходник: [ui/dataset_audit/page_spatial.py](../ui/dataset_audit/page_spatial.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`build_spatial(parent)`](../ui/dataset_audit/page_spatial.py#L17)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-statistics"></a>

## ui.dataset_audit.page_statistics

Исходник: [ui/dataset_audit/page_statistics.py](../ui/dataset_audit/page_statistics.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_make_page`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`build_statistics(parent)`](../ui/dataset_audit/page_statistics.py#L17)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-training"></a>

## ui.dataset_audit.page_training

Исходник: [ui/dataset_audit/page_training.py](../ui/dataset_audit/page_training.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `TrainingImageLabel`, `_make_page`
- [ui.dataset_source](#ui-dataset-source): `make_source_button`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`build_training(parent)`](../ui/dataset_audit/page_training.py#L17)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-page-wiring"></a>

## ui.dataset_audit.page_wiring

Исходник: [ui/dataset_audit/page_wiring.py](../ui/dataset_audit/page_wiring.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`connect_source(parent)`](../ui/dataset_audit/page_wiring.py#L6)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-presentation"></a>

## ui.dataset_audit.presentation

Исходник: [ui/dataset_audit/presentation.py](../ui/dataset_audit/presentation.py#L1).

**Прямые потребители:** [ui.dataset_audit.reporting](#ui-dataset-audit-reporting), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [ui.dataset_audit.session](#ui-dataset-audit-session).

**Прямые локальные импорты:**
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `histogram_edges`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QTableWidgetItem`, `matplotlib.pyplot`.

**Объявления:**

- [`get_included_class_ids(class_table)`](../ui/dataset_audit/presentation.py#L9)
- [`get_class_names_by_orig_id(class_table)`](../ui/dataset_audit/presentation.py#L21)
- [`get_class_names_included(class_table)`](../ui/dataset_audit/presentation.py#L33)
- [`render_metrics(metrics_summary_label, class_stats_table, ax_all, ax_classes, canvas, filtered_pairs, all_pairs, class_counts, total_objects, object_areas, class_areas, name_by_orig, ax_pie=None, ax_bar=None, ax_area_pct=None, object_areas_pct=None)`](../ui/dataset_audit/presentation.py#L49)

<a id="ui-dataset-audit-quality-controller"></a>

## ui.dataset_audit.quality_controller

Исходник: [ui/dataset_audit/quality_controller.py](../ui/dataset_audit/quality_controller.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `histogram_edges`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QTableWidgetItem`, `pathlib.Path`.

**Объявления:**

- Класс [`DatasetQualityController`](../ui/dataset_audit/quality_controller.py#L11)

<a id="ui-dataset-audit-reporting"></a>

## ui.dataset_audit.reporting

Исходник: [ui/dataset_audit/reporting.py](../ui/dataset_audit/reporting.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle): `stale`
- [ui.dataset_audit.presentation](#ui-dataset-audit-presentation): `presentation as audit_presentation`
- [ui.dataset_audit.workers](#ui-dataset-audit-workers): `ReportWorker`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMessageBox`, `numpy`, `os`, `shutil`.

**Объявления:**

- Класс [`DatasetReportController`](../ui/dataset_audit/reporting.py#L13)

<a id="ui-dataset-audit-selection"></a>

## ui.dataset_audit.selection

Исходник: [ui/dataset_audit/selection.py](../ui/dataset_audit/selection.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.lifecycle](#ui-dataset-audit-lifecycle): `serialized_scan`
- [ui.dataset_audit.presentation](#ui-dataset-audit-presentation): `presentation as audit_presentation`
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `ClassImbalanceAnalyzer`, `DatasetAnalyzer`, `FilterParams`, `clear_brightness_cache`
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks): `LeakageChecker`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `read_objects`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QMenu`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QTableWidgetItem`, `collections.Counter`, `os`, `subprocess`, `sys`.

**Объявления:**

- Класс [`DatasetSelectionController`](../ui/dataset_audit/selection.py#L32)

<a id="ui-dataset-audit-session"></a>

## ui.dataset_audit.session

Исходник: [ui/dataset_audit/session.py](../ui/dataset_audit/session.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.presentation](#ui-dataset-audit-presentation): `presentation as audit_presentation`
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.dataset_audit.splits](#utils-new-dataset-audit-splits): `split_memberships`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QWidget`, `traceback`.

**Объявления:**

- Класс [`DatasetSessionController`](../ui/dataset_audit/session.py#L14)

<a id="ui-dataset-audit-source-page"></a>

## ui.dataset_audit.source_page

Исходник: [ui/dataset_audit/source_page.py](../ui/dataset_audit/source_page.py#L1).

**Прямые потребители:** [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

**Прямые локальные импорты:**
- [ui.dataset_audit.page_helpers](#ui-dataset-audit-page-helpers): `_mk_max_dspin`, `_mk_max_spin`, `_mk_min_dspin`, `_mk_min_spin`
- [ui.dataset_audit.workspace](#ui-dataset-audit-workspace): `AnalysisPages`, `build_overview`
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`build_source(parent)`](../ui/dataset_audit/source_page.py#L32)
  Явно импортируют: [ui_tabs.dataset_analysis_ui](#ui-tabs-dataset-analysis-ui).

<a id="ui-dataset-audit-spatial"></a>

## ui.dataset_audit.spatial

Исходник: [ui/dataset_audit/spatial.py](../ui/dataset_audit/spatial.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`DatasetSpatialController`](../ui/dataset_audit/spatial.py#L4)

<a id="ui-dataset-audit-tasks"></a>

## ui.dataset_audit.tasks

Исходник: [ui/dataset_audit/tasks.py](../ui/dataset_audit/tasks.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_audit.workers](#ui-dataset-audit-workers): `AuditWorker`
- [utils_new.dataset_audit.scans](#utils-new-dataset-audit-scans): `ScanJob`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QMessageBox`.

**Объявления:**

- Класс [`ScanWorker`](../ui/dataset_audit/tasks.py#L10) (`AuditWorker`)
- Класс [`ScanController`](../ui/dataset_audit/tasks.py#L14) (`QObject`)

<a id="ui-dataset-audit-training-results"></a>

## ui.dataset_audit.training_results

Исходник: [ui/dataset_audit/training_results.py](../ui/dataset_audit/training_results.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis): `TrainingResultsLoader`

**Внешние/stdlib импорты:** `PyQt5.QtGui.QPixmap`, `os`.

**Объявления:**

- Класс [`DatasetTrainingResultsController`](../ui/dataset_audit/training_results.py#L12)

<a id="ui-dataset-audit-workers"></a>

## ui.dataset_audit.workers

Исходник: [ui/dataset_audit/workers.py](../ui/dataset_audit/workers.py#L1).

**Прямые потребители:** [ui.annotation_review](#ui-annotation-review), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.reporting](#ui-dataset-audit-reporting), [ui.dataset_audit.tasks](#ui-dataset-audit-tasks).

**Прямые локальные импорты:**
- [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs): `EmbeddingClusterJob`, `LossInspectionJob`, `ReportJob`, `YoloValidationJob`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `concurrent.futures.ThreadPoolExecutor`, `copy.deepcopy`.

**Объявления:**

- Класс [`AuditWorker`](../ui/dataset_audit/workers.py#L14) (`QObject`)
- Класс [`LossInspectionWorker`](../ui/dataset_audit/workers.py#L92) (`AuditWorker`)
- Класс [`EmbeddingClusterWorker`](../ui/dataset_audit/workers.py#L96) (`AuditWorker`)
- Класс [`ReportWorker`](../ui/dataset_audit/workers.py#L100) (`AuditWorker`)
- Класс [`YoloValidationWorker`](../ui/dataset_audit/workers.py#L104) (`AuditWorker`)

<a id="ui-dataset-audit-workspace"></a>

## ui.dataset_audit.workspace

Исходник: [ui/dataset_audit/workspace.py](../ui/dataset_audit/workspace.py#L1).

**Прямые потребители:** [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QStackedWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`AnalysisPages`](../ui/dataset_audit/workspace.py#L18) (`QWidget`)
- [`build_overview(owner)`](../ui/dataset_audit/workspace.py#L58)
  Явно импортируют: [ui.dataset_audit.source_page](#ui-dataset-audit-source-page).

<a id="ui-dataset-source"></a>

## ui.dataset_source

Исходник: [ui/dataset_source.py](../ui/dataset_source.py#L1).

**Прямые потребители:** [main](#main), [main_ui](#main-ui), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.model_research](#tabs-model-research), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_source_unification](#tests-test-source-unification), [ui.DL_predict_settings](#ui-DL-predict-settings), [ui.DL_train_settings_base](#ui-DL-train-settings-base), [ui.DL_train_settings_new_model](#ui-DL-train-settings-new-model), [ui.autoreseach_ui.evaluation_controls](#ui-autoreseach-ui-evaluation-controls), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.page_models](#ui-dataset-audit-page-models), [ui.dataset_audit.page_training](#ui-dataset-audit-page-training), [ui.dataset_audit.source_page](#ui-dataset-audit-source-page), [ui.dataset_audit.training_results](#ui-dataset-audit-training-results), [ui.image_navigation](#ui-image-navigation), [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.slice_dataset_sahi_ui](#ui-tabs-slice-dataset-sahi-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

**Прямые локальные импорты:**
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `discover_coco_json`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QSettings`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFileDialog`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`, `pathlib.Path`.

**Объявления:**

- [`make_source_button(key, parent=None)`](../ui/dataset_source.py#L215)
  Явно импортируют: [ui.dataset_audit.page_models](#ui-dataset-audit-page-models), [ui.dataset_audit.page_training](#ui-dataset-audit-page-training), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).
- [`annotation_format_combo(parent=None, *, mixed=True)`](../ui/dataset_source.py#L223)
  Явно импортируют: [ui.autoreseach_ui.evaluation_controls](#ui-autoreseach-ui-evaluation-controls).
- [`get_open_file(owner, key, *, multiple=False, directory='')`](../ui/dataset_source.py#L234)
- [`get_directory(owner, key, *, directory='')`](../ui/dataset_source.py#L250)
- Класс [`DatasetSourceWidget`](../ui/dataset_source.py#L262) (`QGroupBox`)
- [`choose_coco_sources(owner)`](../ui/dataset_source.py#L343) — One COCO file/folder chooser used by editors, preparation, audit and SAHI.

<a id="ui-image-conversion"></a>

## ui.image_conversion

Исходник: [ui/image_conversion.py](../ui/image_conversion.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.image_view](#ui-image-view), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.presentation](#ui-segmentation-presentation), [ui.segmentation.results](#ui-segmentation-results).

**Прямые локальные импорты:**
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`

**Внешние/stdlib импорты:** `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QPixmap`, `cv2`, `numpy`.

**Объявления:**

- [`numpy_to_qpixmap(img_bgr, max_side=8192)`](../ui/image_conversion.py#L7) — Display-only conversion. Geometry stays in source-image coordinates.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.image_view](#ui-image-view), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.presentation](#ui-segmentation-presentation), [ui.segmentation.results](#ui-segmentation-results).

<a id="ui-image-navigation"></a>

## ui.image_navigation

Исходник: [ui/image_navigation.py](../ui/image_navigation.py#L1).

**Прямые потребители:** [main](#main), [ui.segmentation.panels](#ui-segmentation-panels), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`
- [ui.settings_service](#ui-settings-service): `settings`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QEvent`, `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QAbstractSpinBox`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`ImageNavigationWidget`](../ui/image_navigation.py#L20) (`QWidget`)
- [`parse_key_spec(spec)`](../ui/image_navigation.py#L131) — Парсит строку вида:
- Класс [`NavigationShortcutInstaller`](../ui/image_navigation.py#L169) (`QObject`)

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

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `importlib.util.find_spec`.

**Объявления:**

- Класс [`InteractiveParametersUI`](../ui/interactive_params_ui.py#L19) (`QWidget`)

<a id="ui-log-widget"></a>

## ui.log_widget

Исходник: [ui/log_widget.py](../ui/log_widget.py#L1).

**Прямые потребители:** [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui), [ui.dataset_audit.page_wiring](#ui-dataset-audit-page-wiring), [ui.segmentation.panels](#ui-segmentation-panels), [ui_tabs.deep_learning_training_ui](#ui-tabs-deep-learning-training-ui), [ui_tabs.demo_ensemble_methods_ui](#ui-tabs-demo-ensemble-methods-ui), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui), [ui_tabs.preparing_dataset_yaml_ui](#ui-tabs-preparing-dataset-yaml-ui), [ui_tabs.viewing_dataset_ui](#ui-tabs-viewing-dataset-ui), [ui_tabs.yolo_find_img_ui](#ui-tabs-yolo-find-img-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PyQt5.QtCore.pyqtSignal`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`LogWidget`](../ui/log_widget.py#L13) (`QWidget`)

<a id="ui-morphology-ui"></a>

## ui.morphology_ui

Исходник: [ui/morphology_ui.py](../ui/morphology_ui.py#L1).

**Прямые потребители:** [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui), [ui_tabs.yolo_test_aeb_ui](#ui-tabs-yolo-test-aeb-ui).

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
- [ui.annotation_review](#ui-annotation-review): `AnnotationReviewWidget`
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.dataset_io_new](#utils-new-dataset-io-new): `_read_with_fallback`
- [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new): `blend_index_mask`
- [utils_new.dataset_validation_new](#utils-new-dataset-validation-new): `ImageStats`, `load_annotations`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QImage`, `PyQt5.QtGui.QKeySequence`, `PyQt5.QtGui.QPixmap`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QDialog`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QShortcut`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `cv2`, `numpy`, `os`.

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

<a id="ui-segmentation-computation"></a>

## ui.segmentation.computation

Исходник: [ui/segmentation/computation.py](../ui/segmentation/computation.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.segmentation.errors](#ui-segmentation-errors): `errors`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation`
- [ui.segmentation.session](#ui-segmentation-session): `session`
- [utils_new.segmentation.operations](#utils-new-segmentation-operations): `operations`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`, `SegmentationResult`, `binary_mask`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QMessageBox`, `functools.wraps`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`gui_action(function)`](../ui/segmentation/computation.py#L15)
- [`frame_for(owner)`](../ui/segmentation/computation.py#L26)
- [`accept_mask(owner, attribute, mask, shape)`](../ui/segmentation/computation.py#L37)
- [`run_interactive(owner)`](../ui/segmentation/computation.py#L44)
- [`run_dl(owner)`](../ui/segmentation/computation.py#L94)
- [`cluster_result(owner, result)`](../ui/segmentation/computation.py#L138)
- [`predict_ml(owner, cluster=False)`](../ui/segmentation/computation.py#L169)
- [`train_ml(owner)`](../ui/segmentation/computation.py#L199)
- [`find_k(owner)`](../ui/segmentation/computation.py#L264)
- [`load_dl(owner, name)`](../ui/segmentation/computation.py#L290)
- [`load_ml(owner, path=None)`](../ui/segmentation/computation.py#L347)

<a id="ui-segmentation-controller"></a>

## ui.segmentation.controller

Исходник: [ui/segmentation/controller.py](../ui/segmentation/controller.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.segmentation.panels](#ui-segmentation-panels).

**Прямые локальные импорты:**
- [ui.segmentation.jobs](#ui-segmentation-jobs): `TaskController`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation`
- [ui.segmentation.results](#ui-segmentation-results): `results`
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`, `SegmentationResult`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QTimer`, `PyQt5.QtWidgets.QMessageBox`.

**Объявления:**

- [`initialize(owner)`](../ui/segmentation/controller.py#L12)
  Явно импортируют: [ui.segmentation.panels](#ui-segmentation-panels).
- [`schedule(owner, stage='prediction')`](../ui/segmentation/controller.py#L55)
- [`invalidate_prediction(owner)`](../ui/segmentation/controller.py#L59)
- [`resize_changed(owner, enabled)`](../ui/segmentation/controller.py#L70)
- [`close(owner)`](../ui/segmentation/controller.py#L87)

<a id="ui-segmentation-errors"></a>

## ui.segmentation.errors

Исходник: [ui/segmentation/errors.py](../ui/segmentation/errors.py#L1).

**Прямые потребители:** [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.jobs](#ui-segmentation-jobs), [ui.segmentation.presentation](#ui-segmentation-presentation).

**Прямые локальные импорты:**
- [ui.segmentation.histogram](#ui-segmentation-histogram): `update_histogram`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`, `SegmentationResult`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QMessageBox`.

**Объявления:**

- [`require(owner, condition, message)`](../ui/segmentation/errors.py#L8)
- [`report(owner, context, error, *, modal=True)`](../ui/segmentation/errors.py#L15)

<a id="ui-segmentation-export"></a>

## ui.segmentation.export

Исходник: [ui/segmentation/export.py](../ui/segmentation/export.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `image_size_from_file`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `build_label_save_plan`, `format_label_save_confirmation`
- [utils_new.segmentation.export](#utils-new-segmentation-export): `write_annotations_atomic`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QMessageBox`, `pathlib.Path`.

**Объявления:**

- [`save_selected(owner)`](../ui/segmentation/export.py#L14)

<a id="ui-segmentation-histogram"></a>

## ui.segmentation.histogram

Исходник: [ui/segmentation/histogram.py](../ui/segmentation/histogram.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.segmentation.errors](#ui-segmentation-errors), [ui.segmentation.panels](#ui-segmentation-panels), [ui.segmentation.presentation](#ui-segmentation-presentation).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- Класс [`HistogramCanvas`](../ui/segmentation/histogram.py#L6) (`FigureCanvasQTAgg`)
- [`update_histogram(owner, gray, threshold=None)`](../ui/segmentation/histogram.py#L12)
  Явно импортируют: [ui.segmentation.errors](#ui-segmentation-errors).
- [`update_current(owner)`](../ui/segmentation/histogram.py#L25)

<a id="ui-segmentation-image-input"></a>

## ui.segmentation.image_input

Исходник: [ui/segmentation/image_input.py](../ui/segmentation/image_input.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `no_frame`
- [ui.segmentation.results](#ui-segmentation-results): `clear_views`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `create_cache`, `image_paths`, `load_one_image_item`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QProgressDialog`, `pathlib.Path`.

**Объявления:**

- [`choose_images(owner, load)`](../ui/segmentation/image_input.py#L17)
- [`choose_folder(owner, load)`](../ui/segmentation/image_input.py#L24)
- [`load_cached(owner, source, view_names, *, max_side=1024, reset=None)`](../ui/segmentation/image_input.py#L31)
- [`load_images_universal(source, require_annotations=False, resize_enabled=True, max_side=1024, progress_callback=None, parent=None, format_hint='auto', record_callback=None, keep_invalid=False)`](../ui/segmentation/image_input.py#L66) — Temporary eager adapter for ML training; no Qt in the actual reader.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).
- [`load_eager(owner, source, *, max_side=640)`](../ui/segmentation/image_input.py#L120)
- [`reload_images(owner, load)`](../ui/segmentation/image_input.py#L141)
- [`clear_session(owner, *, reset=None, empty_fields=(), none_fields=())`](../ui/segmentation/image_input.py#L149)

<a id="ui-segmentation-interactive-view"></a>

## ui.segmentation.interactive_view

Исходник: [ui/segmentation/interactive_view.py](../ui/segmentation/interactive_view.py#L1).

**Прямые потребители:** [ui.segmentation.panels](#ui-segmentation-panels).

**Прямые локальные импорты:**
- [ui.image_view](#ui-image-view): `SmartGraphicsViewNew`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`.

**Объявления:**

- Класс [`SegmentationInputView`](../ui/segmentation/interactive_view.py#L11) (`SmartGraphicsViewNew`)

<a id="ui-segmentation-jobs"></a>

## ui.segmentation.jobs

Исходник: [ui/segmentation/jobs.py](../ui/segmentation/jobs.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.segmentation.controller](#ui-segmentation-controller).

**Прямые локальные импорты:**
- [ui.segmentation.errors](#ui-segmentation-errors): `errors`
- [ui.segmentation.presentation](#ui-segmentation-presentation): `presentation`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QObject`, `PyQt5.QtCore.Qt`, `PyQt5.QtCore.pyqtSignal`, `concurrent.futures.CancelledError`, `concurrent.futures.ThreadPoolExecutor`, `concurrent.futures.TimeoutError`, `threading.Event`.

**Объявления:**

- Класс [`TaskController`](../ui/segmentation/jobs.py#L12) (`QObject`)
- Класс [`BackgroundTask`](../ui/segmentation/jobs.py#L141) (`QObject`)

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

<a id="ui-segmentation-panels"></a>

## ui.segmentation.panels

Исходник: [ui/segmentation/panels.py](../ui/segmentation/panels.py#L1).

**Прямые потребители:** [ui.segmentation.presentation](#ui-segmentation-presentation), [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui).

**Прямые локальные импорты:**
- [ui.annotation_objects](#ui-annotation-objects): `AnnotationObjectsWidget`
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.segmentation.controller](#ui-segmentation-controller): `initialize`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `HistogramCanvas`
- [ui.segmentation.interactive_view](#ui-segmentation-interactive-view): `SegmentationInputView`
- [ui.segmentation.results](#ui-segmentation-results): `results`
- [utils_new.annotations.modes](#utils-new-annotations-modes): `MODE_LABELS`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`PresetPanel`](../ui/segmentation/panels.py#L63) (`QWidget`)
- Класс [`ResultPanel`](../ui/segmentation/panels.py#L82) (`QWidget`)
- [`begin_layout(owner, kind)`](../ui/segmentation/panels.py#L103)
  Явно импортируют: [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui).
- [`finish_layout(owner)`](../ui/segmentation/panels.py#L163)
  Явно импортируют: [ui_tabs.deep_learning_methods_ui](#ui-tabs-deep-learning-methods-ui), [ui_tabs.gradient_methods_ui](#ui-tabs-gradient-methods-ui), [ui_tabs.interactive_methods_ui](#ui-tabs-interactive-methods-ui), [ui_tabs.threshold_methods_ui](#ui-tabs-threshold-methods-ui), [ui_tabs.traditional_ml_methods_ui](#ui-tabs-traditional-ml-methods-ui).

<a id="ui-segmentation-presentation"></a>

## ui.segmentation.presentation

Исходник: [ui/segmentation/presentation.py](../ui/segmentation/presentation.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.controller](#ui-segmentation-controller), [ui.segmentation.errors](#ui-segmentation-errors), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.jobs](#ui-segmentation-jobs).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [ui.segmentation.errors](#ui-segmentation-errors): `errors`
- [ui.segmentation.histogram](#ui-segmentation-histogram): `histogram`
- [ui.segmentation.panels](#ui-segmentation-panels): `STAGES`
- [ui.segmentation.results](#ui-segmentation-results): `results`
- [ui.segmentation.session](#ui-segmentation-session): `session`
- [utils_new.annotations.modes](#utils-new-annotations-modes): `supports_hull`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ResultStatus`, `SegmentationResult`, `binary_mask`, `masked_preview`

**Внешние/stdlib импорты:** `cv2`, `pathlib.Path`.

**Объявления:**

- [`show_image(view, image, annotations=())`](../ui/segmentation/presentation.py#L29)
- [`set_status(owner, status, error=None)`](../ui/segmentation/presentation.py#L41)
- [`show_unavailable(owner, status, error=None)`](../ui/segmentation/presentation.py#L48)
- [`no_frame(owner, status=ResultStatus.NO_IMAGE, error=None)`](../ui/segmentation/presentation.py#L67)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).
- [`geometry_changed(owner, *_)`](../ui/segmentation/presentation.py#L82)
- [`postprocess(owner, frame)`](../ui/segmentation/presentation.py#L96)
- [`display(owner)`](../ui/segmentation/presentation.py#L131)

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
- Класс [`ThresholdPresetManager`](../ui/segmentation/preset_bindings.py#L245) (`MethodPresetManager`)
- Класс [`GradientPresetManager`](../ui/segmentation/preset_bindings.py#L298) (`MethodPresetManager`)
- Класс [`InteractivePresetManager`](../ui/segmentation/preset_bindings.py#L327) (`MethodPresetManager`)
- Класс [`TraditionalMLPresetManager`](../ui/segmentation/preset_bindings.py#L360) (`MethodPresetManager`)
- Класс [`DeepLearningPresetManager`](../ui/segmentation/preset_bindings.py#L390) (`MethodPresetManager`)

<a id="ui-segmentation-results"></a>

## ui.segmentation.results

Исходник: [ui/segmentation/results.py](../ui/segmentation/results.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [ui.segmentation.controller](#ui-segmentation-controller), [ui.segmentation.image_input](#ui-segmentation-image-input), [ui.segmentation.panels](#ui-segmentation-panels), [ui.segmentation.presentation](#ui-segmentation-presentation).

**Прямые локальные импорты:**
- [ui.image_conversion](#ui-image-conversion): `numpy_to_qpixmap`
- [utils_new.annotations.object_info](#utils-new-annotations-object-info): `ObjectInfo`

**Внешние/stdlib импорты:** `PyQt5.QtCore.QRectF`, `PyQt5.QtCore.Qt`.

**Объявления:**

- [`current_image(owner)`](../ui/segmentation/results.py#L9)
- [`update_object_list(owner)`](../ui/segmentation/results.py#L16)
- [`update_overlay(owner)`](../ui/segmentation/results.py#L31)
- [`on_checks_changed(owner, indices)`](../ui/segmentation/results.py#L69)
- [`prepare_frame(owner, image)`](../ui/segmentation/results.py#L76) — Scene geometry always uses working pixels, even if QPixmap is capped.
- [`clear_views(owner)`](../ui/segmentation/results.py#L83)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).
- [`refresh_styles(owner)`](../ui/segmentation/results.py#L91) — Settings only affect Qt paint; preserve pixels, masks, selection and text.
- [`focus_row(owner, row)`](../ui/segmentation/results.py#L97)
- [`focus_annotation(owner, visible_index)`](../ui/segmentation/results.py#L104)
- [`zoom_object(owner, index)`](../ui/segmentation/results.py#L117)

<a id="ui-segmentation-session"></a>

## ui.segmentation.session

Исходник: [ui/segmentation/session.py](../ui/segmentation/session.py#L1).

**Прямые потребители:** [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.presentation](#ui-segmentation-presentation).

**Прямые локальные импорты:**
- [utils_new.segmentation.result](#utils-new-segmentation-result): `ImageFrame`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`current_frame(owner)`](../ui/segmentation/session.py#L6)
- [`prefetch(owner)`](../ui/segmentation/session.py#L42)

<a id="ui-settings-service"></a>

## ui.settings_service

Исходник: [ui/settings_service.py](../ui/settings_service.py#L1).

**Прямые потребители:** [main](#main), [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [ui.dataset_audit.session](#ui-dataset-audit-session), [ui.image_navigation](#ui-image-navigation), [ui.image_view](#ui-image-view), [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new), [ui.segmentation.controller](#ui-segmentation-controller), [ui.settings_ui](#ui-settings-ui), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).

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
- [ui.dataset_audit.page_duplicates](#ui-dataset-audit-page-duplicates): `build_duplicates`
- [ui.dataset_audit.page_embeddings](#ui-dataset-audit-page-embeddings): `build_embeddings`
- [ui.dataset_audit.page_models](#ui-dataset-audit-page-models): `build_models`
- [ui.dataset_audit.page_quality](#ui-dataset-audit-page-quality): `build_quality`
- [ui.dataset_audit.page_report](#ui-dataset-audit-page-report): `build_report`
- [ui.dataset_audit.page_spatial](#ui-dataset-audit-page-spatial): `build_spatial`
- [ui.dataset_audit.page_statistics](#ui-dataset-audit-page-statistics): `build_statistics`
- [ui.dataset_audit.page_training](#ui-dataset-audit-page-training): `build_training`
- [ui.dataset_audit.page_wiring](#ui-dataset-audit-page-wiring): `connect_source`
- [ui.dataset_audit.source_page](#ui-dataset-audit-source-page): `build_source`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`setup_dataset_analysis_ui(parent)`](../ui_tabs/dataset_analysis_ui.py#L15)
  Явно импортируют: [tabs.dataset_analysis](#tabs-dataset-analysis).

<a id="ui-tabs-deep-learning-methods-ui"></a>

## ui_tabs.deep_learning_methods_ui

Исходник: [ui_tabs/deep_learning_methods_ui.py](../ui_tabs/deep_learning_methods_ui.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods).

**Прямые локальные импорты:**
- [ui.DL_predict_settings](#ui-DL-predict-settings): `CustomONNXInferenceSettings`, `DeepLabV3InferenceSettings`, `SAMInferenceSettings`, `SegFormerInferenceSettings`, `UNetInferenceSettings`, `YOLOInferenceSettings`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.panels](#ui-segmentation-panels): `ResultPanel`, `begin_layout`, `finish_layout`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_deep_learning_methods_ui(parent)`](../ui_tabs/deep_learning_methods_ui.py#L25)
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
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QToolButton`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_demo_ensemble_methods_ui(parent)`](../ui_tabs/demo_ensemble_methods_ui.py#L27)
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods).

<a id="ui-tabs-gradient-methods-ui"></a>

## ui_tabs.gradient_methods_ui

Исходник: [ui_tabs/gradient_methods_ui.py](../ui_tabs/gradient_methods_ui.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods).

**Прямые локальные импорты:**
- [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls): `AutoResearchPanel`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.panels](#ui-segmentation-panels): `ResultPanel`, `begin_layout`, `finish_layout`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_gradient_ui(parent)`](../ui_tabs/gradient_methods_ui.py#L18)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods).

<a id="ui-tabs-interactive-methods-ui"></a>

## ui_tabs.interactive_methods_ui

Исходник: [ui_tabs/interactive_methods_ui.py](../ui_tabs/interactive_methods_ui.py#L1).

**Прямые потребители:** [tabs.interactive_methods](#tabs-interactive-methods).

**Прямые локальные импорты:**
- [ui.interactive_params_ui](#ui-interactive-params-ui): `InteractiveParametersUI`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.panels](#ui-segmentation-panels): `ResultPanel`, `begin_layout`, `finish_layout`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QVBoxLayout`.

**Объявления:**

- [`setup_interactive_ui(parent)`](../ui_tabs/interactive_methods_ui.py#L18) — Создаёт UI для вкладки интерактивных методов сегментации.
  Явно импортируют: [tabs.interactive_methods](#tabs-interactive-methods).

<a id="ui-tabs-layout-dataset-ui"></a>

## ui_tabs.layout_dataset_ui

Исходник: [ui_tabs/layout_dataset_ui.py](../ui_tabs/layout_dataset_ui.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset).

**Прямые локальные импорты:**
- [ui.annotation_review](#ui-annotation-review): `AnnotationReviewWidget`
- [ui.dataset_source](#ui-dataset-source): `make_source_button`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.image_view](#ui-image-view): `SmartGraphicsViewNew`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [utils_ops.segmentation_ops](#utils-ops-segmentation-ops): `get_class_color`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QButtonGroup`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`setup_layout_dataset_ui(parent)`](../ui_tabs/layout_dataset_ui.py#L33)
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset).

<a id="ui-tabs-preparing-dataset-yaml-ui"></a>

## ui_tabs.preparing_dataset_yaml_ui

Исходник: [ui_tabs/preparing_dataset_yaml_ui.py](../ui_tabs/preparing_dataset_yaml_ui.py#L1).

**Прямые потребители:** [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

**Прямые локальные импорты:**
- [ui.control_buttons](#ui-control-buttons): `ControlButtons`
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QRadioButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTabWidget`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`, `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg`, `matplotlib.figure.Figure`.

**Объявления:**

- [`_section(title, description=None)`](../ui_tabs/preparing_dataset_yaml_ui.py#L37)
- [`_page()`](../ui_tabs/preparing_dataset_yaml_ui.py#L51)
- [`_wrap_row(*widgets)`](../ui_tabs/preparing_dataset_yaml_ui.py#L64)
- [`setup_preparing_dataset_yaml_ui(parent)`](../ui_tabs/preparing_dataset_yaml_ui.py#L73)
  Явно импортируют: [tabs.preparing_dataset_yaml](#tabs-preparing-dataset-yaml).

<a id="ui-tabs-slice-dataset-sahi-ui"></a>

## ui_tabs.slice_dataset_sahi_ui

Исходник: [ui_tabs/slice_dataset_sahi_ui.py](../ui_tabs/slice_dataset_sahi_ui.py#L1).

**Прямые потребители:** [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QAbstractItemView`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFormLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QHeaderView`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QTableWidget`, `PyQt5.QtWidgets.QTextEdit`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_sahi_slicer_ui(parent)`](../ui_tabs/slice_dataset_sahi_ui.py#L29) — SAHI slicer UI: слева — пары/валидация, справа — SAHI.
  Явно импортируют: [tabs.slice_dataset_sahi](#tabs-slice-dataset-sahi).

<a id="ui-tabs-threshold-methods-ui"></a>

## ui_tabs.threshold_methods_ui

Исходник: [ui_tabs/threshold_methods_ui.py](../ui_tabs/threshold_methods_ui.py#L1).

**Прямые потребители:** [tabs.threshold_methods](#tabs-threshold-methods).

**Прямые локальные импорты:**
- [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls): `AutoResearchPanel`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.panels](#ui-segmentation-panels): `ResultPanel`, `begin_layout`, `finish_layout`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_threshold_ui(parent)`](../ui_tabs/threshold_methods_ui.py#L20) — Создаёт UI для пороговых методов внутри QMainWindow.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods).

<a id="ui-tabs-traditional-ml-methods-ui"></a>

## ui_tabs.traditional_ml_methods_ui

Исходник: [ui_tabs/traditional_ml_methods_ui.py](../ui_tabs/traditional_ml_methods_ui.py#L1).

**Прямые потребители:** [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

**Прямые локальные импорты:**
- [ui.dataset_source](#ui-dataset-source): `make_source_button`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`
- [ui.segmentation.panels](#ui-segmentation-panels): `ResultPanel`, `begin_layout`, `finish_layout`

**Внешние/stdlib импорты:** `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_traditional_ml_ui(parent)`](../ui_tabs/traditional_ml_methods_ui.py#L20)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).

<a id="ui-tabs-viewing-dataset-ui"></a>

## ui_tabs.viewing_dataset_ui

Исходник: [ui_tabs/viewing_dataset_ui.py](../ui_tabs/viewing_dataset_ui.py#L1).

**Прямые потребители:** [tabs.viewing_dataset](#tabs-viewing-dataset).

**Прямые локальные импорты:**
- [ui.annotation_review](#ui-annotation-review): `AnnotationReviewWidget`
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
- [ui.dataset_source](#ui-dataset-source): `DatasetSourceWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QProgressBar`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QStackedWidget`, `PyQt5.QtWidgets.QTreeView`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- Класс [`setup_yolo_find_img_ui`](../ui_tabs/yolo_find_img_ui.py#L29) (`object`)

<a id="ui-tabs-yolo-test-aeb-ui"></a>

## ui_tabs.yolo_test_aeb_ui

Исходник: [ui_tabs/yolo_test_aeb_ui.py](../ui_tabs/yolo_test_aeb_ui.py#L1).

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

**Прямые локальные импорты:**
- [ui.annotation_review_view](#ui-annotation-review-view): `DatasetReviewGraphicsView`
- [ui.dataset_source](#ui-dataset-source): `make_source_button`
- [ui.image_navigation](#ui-image-navigation): `ImageNavigationWidget`
- [ui.log_widget](#ui-log-widget): `LogWidget`
- [ui.morphology_ui](#ui-morphology-ui): `MorphologyWidget`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QComboBox`, `PyQt5.QtWidgets.QDoubleSpinBox`, `PyQt5.QtWidgets.QFrame`, `PyQt5.QtWidgets.QGridLayout`, `PyQt5.QtWidgets.QGroupBox`, `PyQt5.QtWidgets.QHBoxLayout`, `PyQt5.QtWidgets.QLabel`, `PyQt5.QtWidgets.QLineEdit`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QPushButton`, `PyQt5.QtWidgets.QScrollArea`, `PyQt5.QtWidgets.QSizePolicy`, `PyQt5.QtWidgets.QSlider`, `PyQt5.QtWidgets.QSpinBox`, `PyQt5.QtWidgets.QSplitter`, `PyQt5.QtWidgets.QVBoxLayout`, `PyQt5.QtWidgets.QWidget`.

**Объявления:**

- [`setup_yolo_ae_vae_ui(parent)`](../ui_tabs/yolo_test_aeb_ui.py#L45)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).

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

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.annotation_review](#ui-annotation-review), [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review), [utils_new.dataset_view_new](#utils-new-dataset-view-new), [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing).

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
- [`annotation_in_crop(annotation, original_w: int, original_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L139) — Пересчитывает аннотацию в координаты кропа для векторного Qt-слоя.
- [`mask_roi_at_image_scale(mask: np.ndarray, image_w: int, image_h: int, bounds: Tuple[int, int, int, int])`](../utils_new/annotation_processing.py#L169) — Извлекает маску точно для ROI без ресайза всей маски к размеру снимка.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`apply_class_mask_overlay(image: np.ndarray, mask: Optional[np.ndarray], opacity: float, color_for_class, *, pixel_offset: int=1, background_value: int=0, annotations=None, restrict_to_annotations=False)`](../utils_new/annotation_processing.py#L189) — Показывает индексную маску цветами палитры классов (BGR).
- [`_shapely_polygon(points)`](../utils_new/annotation_processing.py#L285)
- [`_raster_intersection_area(a: np.ndarray, b: np.ndarray, rect: Tuple[float, float, float, float], max_pixels=2000000)`](../utils_new/annotation_processing.py#L297) — Fallback для невыпуклых полигонов без shapely.
- [`overlap_ratio_of_smaller(annotation_a, annotation_b, width: int, height: int)`](../utils_new/annotation_processing.py#L321) — Пересечение / площадь МЕНЬШЕГО объекта, в интервале [0,1].
- [`find_strong_overlaps(annotations, width: int, height: int, threshold: float=0.7, *, metric='smaller', same_class=False)`](../utils_new/annotation_processing.py#L366) — Сопоставляет индексы аннотаций с перекрывающими их соседями.
  Явно импортируют: [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.annotation_review](#ui-annotation-review), [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review).
- [`detection_to_unified(detection, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L417) — Return one normalized annotation for a YOLO+AE/VAE detection dict.
- [`detections_to_unified(detections, image_w: int, image_h: int)`](../utils_new/annotation_processing.py#L462) — Convert all detections, keeping list positions stable for Qt selection.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`_name_for_class(names, class_id)`](../utils_new/annotation_processing.py#L478)
- [`_as_numpy(value)`](../utils_new/annotation_processing.py#L487) — Accept Ultralytics tensors and plain numpy arrays for standalone tests.
- [`yolo_result_to_detections(result, names=None, *, angle_fn=None)`](../utils_new/annotation_processing.py#L498) — Decode YOLO detect/OBB/segment into the existing AE/VAE dict contract.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`prepare_annotation_preview(image, annotation, padding=10, *, mask=None, opacity=0.5, color_for_class=None, pixel_offset=1, background_value=0)`](../utils_new/annotation_processing.py#L581) — Single crop pipeline used by the dataset viewer and YOLO-side previews.
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`render_unified_annotations(image, annotations, color_for_class, *, selected_index=-1, selected_color=(0, 255, 255), text_color=(255, 255, 255), thickness=1, font_scale=0.5)`](../utils_new/annotation_processing.py#L606) — Optional OpenCV renderer for exporting detection imagery.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`render_ae_error_map(error, base, *, valid_region=None, overlay=True, alpha=0.5, show_mask=False, threshold=float('nan'))`](../utils_new/annotation_processing.py#L638) — Render an AE/VAE error heatmap with the original percentile stretch.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`longest_edge_angle_deg(corners)`](../utils_new/annotation_processing.py#L684)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`build_detection_mask(det, W, H)`](../utils_new/annotation_processing.py#L699) — Бинарная маска (uint8, 0/255) для детекции.
- [`morph_image_in_mask(img, mask, erosion_px, dilation_px, shape_name='Rectangle', contour_only=True)`](../utils_new/annotation_processing.py#L721) — Применяет эрозию и/или дилатацию к изображению, сохраняя результат
- [`apply_object_morphology(img, det, erosion_px=1, dilation_px=1, shape_name='Rectangle', contour_only=True, *, operations=None, contour_width_px=None)`](../utils_new/annotation_processing.py#L774) — Морфология изображения внутри выбранной YOLO-детекции.
- [`fill_outside_object(img, det, fill_mode, rotation_matrix=None, crop_offset=(0, 0))`](../utils_new/annotation_processing.py#L828) — Заливает пиксели вне маски объекта (fill_mode: 'none' / 'black' / 'white').
- [`crop_to_detection(img, det, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L871) — Обрезает изображение по xyxy объекта с учётом padding и смещений.
- [`rotate_with_optional_crop(img, det, auto_crop, padding, off_x, off_y, fill_mode)`](../utils_new/annotation_processing.py#L888) — Доворачивает изображение так, чтобы длинная сторона объекта стала
- [`cls_suffix_from_detection(det, index=None)`](../utils_new/annotation_processing.py#L930) — Безопасный суффикс имени файла по классу объекта и его номеру в списке.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`limit_image_size(img, max_side)`](../utils_new/annotation_processing.py#L954) — Пропорционально уменьшает изображение, если длинная сторона > max_side.
- [`aeb_transform_image(image, detection, *, erode=1, dilate=1, morphology_steps=None, contour_width_px=None, shape_name='Rectangle', contour_only=True, auto_crop=False, auto_rotate=False, padding=0, off_x=0, off_y=0, fine_angle=0.0, fill_mode='none', manual_ops=(), invert=False)`](../utils_new/annotation_processing.py#L968) — Common sequential image pipeline for live preview and save-all.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`annotation_to_detection(annotation, image_w, image_h, *, class_name=None)`](../utils_new/annotation_processing.py#L1031) — Build an AEB-compatible detection dict from normalized YOLO annotation.
- [`transform_annotation_image(image, annotation, **kwargs)`](../utils_new/annotation_processing.py#L1075) — Run the same AEB pipeline for a label from a saved dataset.
- [`_mask_extraction_dimensions(binary: np.ndarray, image_shape=None, min_area=None)`](../utils_new/annotation_processing.py#L1092) — Площадь отсечения — та же, что в старом get_display_params().
- [`extract_segments_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1110) — Segment: внешние контуры CHAIN_APPROX_SIMPLE, нормализованные вершины.
- [`extract_boxes_from_mask(binary: np.ndarray, *, image_shape=None, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1137) — Detect: connectedComponentsWithStats, 8-связность, как в старом коде.
- [`extract_obb_from_mask(binary: np.ndarray, *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1160) — OBB: minAreaRect + boxPoints, с исходным усечением до int32.
- [`extract_annotations_from_mask(binary: np.ndarray, mode='detect', *, image_shape=None, use_hull=False, min_area=None, class_id=0)`](../utils_new/annotation_processing.py#L1182) — Единая точка получения Detect/OBB/Segment из бинарной маски.
  Явно импортируют: [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing).
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_new/annotation_processing.py#L1204) — Форматирует аннотацию для отображения в QListWidget.
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_new/annotation_processing.py#L1233) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).

<a id="utils-new-annotations---init--"></a>

## utils_new.annotations.__init__

Исходник: [utils_new/annotations/__init__.py](../utils_new/annotations/__init__.py#L1).

**Прямые потребители:** [tests.test_annotation_objects](#tests-test-annotation-objects), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_coco_new](#utils-new-dataset-coco-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels).

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

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.object_info](#utils-new-annotations-object-info).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `bounds`, `convert_annotation`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`from_legacy(value, size: ImageSize, *, allow_classless=False)`](../utils_new/annotations/adapters.py#L9) — Legacy tuple -> pixel Annotation; strict by default for dataset export.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.object_info](#utils-new-annotations-object-info).
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

<a id="utils-new-annotations-boxes"></a>

## utils_new.annotations.boxes

Исходник: [utils_new/annotations/boxes.py](../utils_new/annotations/boxes.py#L1).

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_report_workflow](#tests-test-report-workflow).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `numpy`.

**Объявления:**

- [`yolo_to_xyxy(cx, cy, w, h)`](../utils_new/annotations/boxes.py#L6) — Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow).
- [`xyxy_to_yolo(x1, y1, x2, y2)`](../utils_new/annotations/boxes.py#L18) — Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow).
- [`iou_xyxy(box1, box2)`](../utils_new/annotations/boxes.py#L29) — Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow).
- [`_thresholds(iou_threshold, conf_threshold)`](../utils_new/annotations/boxes.py#L47)
- [`non_max_suppression(boxes, scores, iou_threshold=0.5, conf_threshold=0.1)`](../utils_new/annotations/boxes.py#L52) — Stable single-class NMS over xyxy boxes; callers separate class IDs.
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow).
- [`_contains_box(a, b)`](../utils_new/annotations/boxes.py#L72)
- [`ensemble_detections(detections_list, weights, iou_threshold=0.5, conf_threshold=0.1, use_containment=False, method='wbf', verbose=False)`](../utils_new/annotations/boxes.py#L76) — Fuse normalized Detect tuples per class, with explicit source weights.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration).

<a id="utils-new-annotations-geometry"></a>

## utils_new.annotations.geometry

Исходник: [utils_new/annotations/geometry.py](../utils_new/annotations/geometry.py#L1).

**Прямые потребители:** [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.object_info](#utils-new-annotations-object-info), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).

**Прямые локальные импорты:**
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`, `shapely.geometry.Polygon`, `shapely.geometry.box`, `shapely.validation.make_valid`.

**Объявления:**

- Класс [`GeometryError`](../utils_new/annotations/geometry.py#L20) (`ValueError`)
- Класс [`LossyConversionError`](../utils_new/annotations/geometry.py#L24) (`GeometryError`)
- [`polygon_area(points)`](../utils_new/annotations/geometry.py#L28)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.object_info](#utils-new-annotations-object-info), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects).
- [`bounds(points)`](../utils_new/annotations/geometry.py#L36)
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.object_info](#utils-new-annotations-object-info), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).
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

**Прямые потребители:** [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels).

**Прямые локальные импорты:**
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `GeometryError`, `bounds`, `canonical_obb_points`, `is_axis_aligned`, `rectangle_points`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`, `MaskEncoding`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`_dtype_for_values(values)`](../utils_new/annotations/masks.py#L17)
- [`annotation_binary_mask(ann: Annotation, size: ImageSize)`](../utils_new/annotations/masks.py#L22) — Raster footprint with half-open pixel-center semantics for rectangles.
  Явно импортируют: [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.dataset_mask_preview_new](#utils-new-dataset-mask-preview-new), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels).
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

**Прямые потребители:** [ui.annotation_objects](#ui-annotation-objects), [utils_new.annotations.__init__](#utils-new-annotations---init--), [utils_new.annotations.adapters](#utils-new-annotations-adapters), [utils_new.annotations.augmentation](#utils-new-annotations-augmentation), [utils_new.annotations.geometry](#utils-new-annotations-geometry), [utils_new.annotations.masks](#utils-new-annotations-masks), [utils_new.annotations.object_info](#utils-new-annotations-object-info), [utils_new.annotations.rendering](#utils-new-annotations-rendering), [utils_new.annotations.transforms](#utils-new-annotations-transforms).

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

<a id="utils-new-annotations-modes"></a>

## utils_new.annotations.modes

Исходник: [utils_new/annotations/modes.py](../utils_new/annotations/modes.py#L1).

**Прямые потребители:** [tests.test_segmentation_unification](#tests-test-segmentation-unification), [ui.annotation_objects](#ui-annotation-objects), [ui.autoreseach_ui.result_models](#ui-autoreseach-ui-result-models), [ui.segmentation.panels](#ui-segmentation-panels), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export), [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing), [utils_new.segmentation.presets](#utils-new-segmentation-presets).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `typing.Literal`.

**Объявления:**

- [`validate_mode(mode)`](../utils_new/annotations/modes.py#L8)
  Явно импортируют: [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing), [utils_new.segmentation.presets](#utils-new-segmentation-presets).
- [`supports_hull(mode)`](../utils_new/annotations/modes.py#L14)
  Явно импортируют: [tests.test_segmentation_unification](#tests-test-segmentation-unification), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing).

<a id="utils-new-annotations-object-info"></a>

## utils_new.annotations.object_info

Исходник: [utils_new/annotations/object_info.py](../utils_new/annotations/object_info.py#L1).

**Прямые потребители:** [tests.test_annotation_objects](#tests-test-annotation-objects), [ui.annotation_objects](#ui-annotation-objects), [ui.segmentation.results](#ui-segmentation-results).

**Прямые локальные импорты:**
- [utils_new.annotations.adapters](#utils-new-annotations-adapters): `from_legacy`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `bounds`, `polygon_area`
- [utils_new.annotations.model](#utils-new-annotations-model): `Annotation`, `ImageSize`

**Внешние/stdlib импорты:** `dataclasses.dataclass`, `numpy`.

**Объявления:**

- Класс [`ObjectGeometry`](../utils_new/annotations/object_info.py#L13)
  - [`ObjectGeometry.width(self)`](../utils_new/annotations/object_info.py#L20)
  - [`ObjectGeometry.height(self)`](../utils_new/annotations/object_info.py#L24)
- Класс [`ObjectInfo`](../utils_new/annotations/object_info.py#L29)
  - [`ObjectInfo.from_tuple(cls, index, value)`](../utils_new/annotations/object_info.py#L36)
  - [`ObjectInfo.geometry(self, size)`](../utils_new/annotations/object_info.py#L39)
  - [`ObjectInfo.area_fraction(self)`](../utils_new/annotations/object_info.py#L50)
  - [`ObjectInfo.source_annotation(self, source_size)`](../utils_new/annotations/object_info.py#L53)

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

<a id="utils-new-autoreseach-utils---init--"></a>

## utils_new.autoreseach_utils.__init__

Исходник: [utils_new/autoreseach_utils/__init__.py](../utils_new/autoreseach_utils/__init__.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-autoreseach-utils-evaluation"></a>

## utils_new.autoreseach_utils.evaluation

Исходник: [utils_new/autoreseach_utils/evaluation.py](../utils_new/autoreseach_utils/evaluation.py#L1).

**Прямые потребители:** [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics): `count_metrics`, `evaluate_image`, `normalize_annotations`
- [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth): `digest_file`, `load_ground_truth`, `match_sample`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `describe_preset`, `preset_from_row`

**Внешние/stdlib импорты:** `dataclasses.dataclass`, `hashlib`, `json`, `math`, `pandas`, `pathlib.Path`.

**Объявления:**

- Класс [`EvaluationResult`](../utils_new/autoreseach_utils/evaluation.py#L15)
- [`evaluate_dataset(reports, reference, *, format_hint='auto', threshold=0.5, class_aware=False, cancel_check=lambda: False, progress=lambda *args: None)`](../utils_new/autoreseach_utils/evaluation.py#L27)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller).

<a id="utils-new-autoreseach-utils-evaluation-export"></a>

## utils_new.autoreseach_utils.evaluation_export

Исходник: [utils_new/autoreseach_utils/evaluation_export.py](../utils_new/autoreseach_utils/evaluation_export.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research), [tests.test_report_evaluation](#tests-test-report-evaluation), [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).

**Прямые локальные импорты:**
- [utils_new.annotations.modes](#utils-new-annotations-modes): `MODE_LABELS`
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `describe_preset`, `preset_from_row`
- [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow): `result_guidance`

**Внешние/stdlib импорты:** `json`, `math`, `os`, `pathlib.Path`, `shutil`, `tempfile`.

**Объявления:**

- [`preset_settings_text(row)`](../utils_new/autoreseach_utils/evaluation_export.py#L30) — Complete reproducible settings, for both UI selection and saved summary.
  Явно импортируют: [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui).
- [`recommendation_text(result)`](../utils_new/autoreseach_utils/evaluation_export.py#L104)
- [`export_evaluation(result, destination)`](../utils_new/autoreseach_utils/evaluation_export.py#L136)
  Явно импортируют: [tabs.model_research](#tabs-model-research), [tests.test_report_evaluation](#tests-test-report-evaluation).

<a id="utils-new-autoreseach-utils-geometry-metrics"></a>

## utils_new.autoreseach_utils.geometry_metrics

Исходник: [utils_new/autoreseach_utils/geometry_metrics.py](../utils_new/autoreseach_utils/geometry_metrics.py#L1).

**Прямые потребители:** [tests.test_report_evaluation](#tests-test-report-evaluation), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `from_legacy`

**Внешние/stdlib импорты:** `json`, `numpy`, `scipy.optimize.linear_sum_assignment`, `shapely.geometry.Polygon`.

**Объявления:**

- [`parse_sequence(value)`](../utils_new/autoreseach_utils/geometry_metrics.py#L12)
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation).
- [`normalize_annotations(values, size)`](../utils_new/autoreseach_utils/geometry_metrics.py#L25) — Return class + normalized polygon, preserving physical source geometry.
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports).
- [`overlap_matrix(predictions, truth, *, class_aware=False)`](../utils_new/autoreseach_utils/geometry_metrics.py#L39)
- [`match_overlaps(overlaps, threshold=0.5)`](../utils_new/autoreseach_utils/geometry_metrics.py#L51)
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation).
- [`count_metrics(tp, fp, fn)`](../utils_new/autoreseach_utils/geometry_metrics.py#L77)
  Явно импортируют: [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation).
- [`evaluate_image(predictions, truth, *, threshold=0.5, class_aware=False)`](../utils_new/autoreseach_utils/geometry_metrics.py#L87)
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation).

<a id="utils-new-autoreseach-utils-gradient-processing"></a>

## utils_new.autoreseach_utils.gradient_processing

Исходник: [utils_new/autoreseach_utils/gradient_processing.py](../utils_new/autoreseach_utils/gradient_processing.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).

**Прямые локальные импорты:**
- [utils_ops.gradient_ops](#utils-ops-gradient-ops): `apply_gradient_method`

**Внешние/stdlib импорты:** `cv2`, `numpy`.

**Объявления:**

- [`gradient_binary(gray, method, params, threshold)`](../utils_new/autoreseach_utils/gradient_processing.py#L8)
  Явно импортируют: [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`finalize_gradient_mask(processed, min_area_percent, fill_contours)`](../utils_new/autoreseach_utils/gradient_processing.py#L14)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).

<a id="utils-new-autoreseach-utils-ground-truth"></a>

## utils_new.autoreseach_utils.ground_truth

Исходник: [utils_new/autoreseach_utils/ground_truth.py](../utils_new/autoreseach_utils/ground_truth.py#L1).

**Прямые потребители:** [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `to_legacy`
- [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics): `normalize_annotations`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `read_yolo_labels`

**Внешние/stdlib импорты:** `PIL.Image`, `dataclasses.dataclass`, `hashlib`, `json`, `pathlib.Path`, `yaml`.

**Объявления:**

- [`path_parts(value)`](../utils_new/autoreseach_utils/ground_truth.py#L16)
- [`digest_file(path)`](../utils_new/autoreseach_utils/ground_truth.py#L24)
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- Класс [`TruthSample`](../utils_new/autoreseach_utils/ground_truth.py#L33)
- Класс [`GroundTruth`](../utils_new/autoreseach_utils/ground_truth.py#L42)
- [`_read_sample(key, image, label, size, format_hint)`](../utils_new/autoreseach_utils/ground_truth.py#L48)
- [`_images(entry, root, visited=None)`](../utils_new/autoreseach_utils/ground_truth.py#L64)
- [`_label_for(image, root)`](../utils_new/autoreseach_utils/ground_truth.py#L95)
- [`_labels_only(root, reports, format_hint, cancel_check)`](../utils_new/autoreseach_utils/ground_truth.py#L116)
- [`load_ground_truth(source, *, reports=None, format_hint='auto', cancel_check=lambda: False)`](../utils_new/autoreseach_utils/ground_truth.py#L167)
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation).
- [`match_sample(image_path, truth)`](../utils_new/autoreseach_utils/ground_truth.py#L224) — Exact path, then longest unique portable suffix; never guess duplicate basenames.
  Явно импортируют: [tests.test_report_evaluation](#tests-test-report-evaluation), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation).

<a id="utils-new-autoreseach-utils-reports"></a>

## utils_new.autoreseach_utils.reports

Исходник: [utils_new/autoreseach_utils/reports.py](../utils_new/autoreseach_utils/reports.py#L1).

**Прямые потребители:** [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export), [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.geometry_metrics](#utils-new-autoreseach-utils-geometry-metrics): `normalize_annotations`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `method_params`, `validate_search_preset`

**Внешние/stdlib импорты:** `json`, `pandas`, `pathlib.Path`.

**Объявления:**

- [`preset_from_row(row)`](../utils_new/autoreseach_utils/reports.py#L10)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export), [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow).
- [`describe_preset(preset)`](../utils_new/autoreseach_utils/reports.py#L17) — Derive display/export parameters from the actual recipe, never a sample row.
  Явно импортируют: [utils_new.autoreseach_utils.evaluation](#utils-new-autoreseach-utils-evaluation), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export).
- [`_size(value)`](../utils_new/autoreseach_utils/reports.py#L43)
- [`read_report(path, *, allow_partial=False, cancel_check=lambda: False, log=lambda message: None)`](../utils_new/autoreseach_utils/reports.py#L53)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_report_workflow](#tests-test-report-workflow), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch).
- [`parse_reports(paths, log=lambda message: None, *, cancel_check=lambda: False, progress=lambda *args: None)`](../utils_new/autoreseach_utils/reports.py#L163)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.workflow](#utils-new-autoreseach-utils-workflow).

<a id="utils-new-autoreseach-utils-search"></a>

## utils_new.autoreseach_utils.search

Исходник: [utils_new/autoreseach_utils/search.py](../utils_new/autoreseach_utils/search.py#L1).

**Прямые потребители:** [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing): `finalize_gradient_mask`, `gradient_binary`
- [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth): `digest_file`
- [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space): `combination_count`, `iter_presets`, `method_params`, `search_kind`
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.image_session](#utils-new-segmentation-image-session): `load_one_image_item`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method`, `apply_threshold_preset`

**Внешние/stdlib импорты:** `cv2`, `json`, `os`, `pathlib.Path`, `tempfile`.

**Объявления:**

- [`evaluate(gray, image, preset, *, raw_mask=None)`](../utils_new/autoreseach_utils/search.py#L19)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch).
- Класс [`RawMaskCache`](../utils_new/autoreseach_utils/search.py#L46)
  - [`RawMaskCache.__init__(self, gray)`](../utils_new/autoreseach_utils/search.py#L49)
  - [`RawMaskCache.get(self, preset)`](../utils_new/autoreseach_utils/search.py#L54)
- [`run_search(paths, settings, cancel, progress=lambda *args: None, *, resize_enabled=True, max_side=1024)`](../utils_new/autoreseach_utils/search.py#L73)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls).

<a id="utils-new-autoreseach-utils-search-space"></a>

## utils_new.autoreseach_utils.search_space

Исходник: [utils_new/autoreseach_utils/search_space.py](../utils_new/autoreseach_utils/search_space.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).

**Прямые локальные импорты:**
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `default_morphology`, `make_preset`, `validate_morphology`, `validate_preset`

**Внешние/stdlib импорты:** `bisect.bisect_left`, `copy.deepcopy`, `decimal.Decimal`, `itertools.product`, `math.isfinite`, `math.prod`.

**Объявления:**

- [`search_kind(settings)`](../utils_new/autoreseach_utils/search_space.py#L70)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`specs_for(kind)`](../utils_new/autoreseach_utils/search_space.py#L77)
  Явно импортируют: [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog).
- [`default_settings(kind='threshold')`](../utils_new/autoreseach_utils/search_space.py#L83)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog).
- [`range_values(values, spec)`](../utils_new/autoreseach_utils/search_space.py#L108)
- [`dimensions(settings)`](../utils_new/autoreseach_utils/search_space.py#L138)
- [`gradient_axes(settings)`](../utils_new/autoreseach_utils/search_space.py#L186)
- [`_parameter_count(method, keys, values)`](../utils_new/autoreseach_utils/search_space.py#L204)
- [`combination_count(settings)`](../utils_new/autoreseach_utils/search_space.py#L213)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.controls](#ui-autoreseach-ui-controls), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`gradient_preset(method, params, **extras)`](../utils_new/autoreseach_utils/search_space.py#L223)
- [`method_params(preset)`](../utils_new/autoreseach_utils/search_space.py#L248)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`iter_presets(settings)`](../utils_new/autoreseach_utils/search_space.py#L268)
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_report_evaluation](#tests-test-report-evaluation), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`validate_params(method, params, kind='threshold')`](../utils_new/autoreseach_utils/search_space.py#L305)
- [`validate_search_preset(preset)`](../utils_new/autoreseach_utils/search_space.py#L315)
  Явно импортируют: [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports).

<a id="utils-new-autoreseach-utils-workflow"></a>

## utils_new.autoreseach_utils.workflow

Исходник: [utils_new/autoreseach_utils/workflow.py](../utils_new/autoreseach_utils/workflow.py#L1).

**Прямые потребители:** [tabs.model_research](#tabs-model-research), [tests.test_report_workflow](#tests-test-report-workflow), [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller), [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export).

**Прямые локальные импорты:**
- [utils_new.autoreseach_utils.reports](#utils-new-autoreseach-utils-reports): `parse_reports`, `preset_from_row`

**Внешние/stdlib импорты:** `dataclasses.dataclass`, `pathlib.Path`.

**Объявления:**

- Класс [`ReportCollection`](../utils_new/autoreseach_utils/workflow.py#L9)
- [`discover_reports(folder)`](../utils_new/autoreseach_utils/workflow.py#L18)
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow).
- [`load_collection(paths, *, cancel_check=lambda: False, progress=lambda *args: None)`](../utils_new/autoreseach_utils/workflow.py#L34)
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow), [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller).
- [`input_guidance(reference, collection)`](../utils_new/autoreseach_utils/workflow.py#L54) — Basic readiness; geometry and correspondence are checked by comparison.
  Явно импортируют: [tabs.model_research](#tabs-model-research), [tests.test_report_workflow](#tests-test-report-workflow).
- [`result_guidance(result)`](../utils_new/autoreseach_utils/workflow.py#L77) — Explain the conclusion and the next useful action without hiding failures.
  Явно импортируют: [tests.test_report_workflow](#tests-test-report-workflow), [ui.autoreseach_ui.model_research_ui](#ui-autoreseach-ui-model-research-ui), [utils_new.autoreseach_utils.evaluation_export](#utils-new-autoreseach-utils-evaluation-export).
- [`load_folder_collection(folder, *, existing=(), cancel_check=lambda: False, progress=lambda *args: None)`](../utils_new/autoreseach_utils/workflow.py#L107)
  Явно импортируют: [ui.autoreseach_ui.evaluation_controller](#ui-autoreseach-ui-evaluation-controller).
- [`selected_presets(frame)`](../utils_new/autoreseach_utils/workflow.py#L117)
  Явно импортируют: [tabs.model_research](#tabs-model-research), [tests.test_report_workflow](#tests-test-report-workflow).

<a id="utils-new-dataset-audit---init--"></a>

## utils_new.dataset_audit.__init__

Исходник: [utils_new/dataset_audit/__init__.py](../utils_new/dataset_audit/__init__.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Нет собственных верхнеуровневых функций/классов; см. импорты и исполняемый код.

<a id="utils-new-dataset-audit-analysis"></a>

## utils_new.dataset_audit.analysis

Исходник: [utils_new/dataset_audit/analysis.py](../utils_new/dataset_audit/analysis.py#L1).

**Прямые потребители:** [tabs.dataset_analysis](#tabs-dataset-analysis), [tests.test_dataset_audit](#tests-test-dataset-audit), [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.dataset_audit.presentation](#ui-dataset-audit-presentation), [ui.dataset_audit.quality_controller](#ui-dataset-audit-quality-controller), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [ui.dataset_audit.training_results](#ui-dataset-audit-training-results).

**Прямые локальные импорты:**
- [utils_new.dataset_audit.export](#utils-new-dataset-audit-export): `export_subset`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `read_objects`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_image_with_fallback`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`

**Внешние/stdlib импорты:** `PIL.Image`, `cv2`, `numpy`, `os`.

**Объявления:**

- [`histogram_edges(values, max_bins=50)`](../utils_new/dataset_audit/analysis.py#L31) — Finite, distinct edges even for areas differing only by rounding noise.
  Явно импортируют: [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.dataset_audit.presentation](#ui-dataset-audit-presentation), [ui.dataset_audit.quality_controller](#ui-dataset-audit-quality-controller).
- [`get_image_brightness(path)`](../utils_new/dataset_audit/analysis.py#L44)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit).
- [`clear_brightness_cache()`](../utils_new/dataset_audit/analysis.py#L65) — Сброс кэша яркости (например, после удаления файлов).
  Явно импортируют: [ui.dataset_audit.selection](#ui-dataset-audit-selection).
- [`_safe_progress(progress_cb, i, total, step=25)`](../utils_new/dataset_audit/analysis.py#L70)
- [`_check_cancel(cancel_check)`](../utils_new/dataset_audit/analysis.py#L75)
- Класс [`FilterParams`](../utils_new/dataset_audit/analysis.py#L82)
  - [`FilterParams.__init__(self, min_w=0, max_w=0, min_h=0, max_h=0, min_brightness=-1, max_brightness=-1, min_objs=0, max_objs=0, min_area=0, max_area=0, min_area_pct=0.0, max_area_pct=0.0, min_aspect=0.0, max_aspect=0.0)`](../utils_new/dataset_audit/analysis.py#L111)
- Класс [`DatasetAnalyzer`](../utils_new/dataset_audit/analysis.py#L144)
  - [`DatasetAnalyzer.__init__(self, dataset_type=0, input_format='auto')`](../utils_new/dataset_audit/analysis.py#L145)
  - [`DatasetAnalyzer.set_dataset_type(self, dt)`](../utils_new/dataset_audit/analysis.py#L149)
  - [`DatasetAnalyzer.set_input_format(self, fmt)`](../utils_new/dataset_audit/analysis.py#L152)
  - [`DatasetAnalyzer.read_objects(self, pair)`](../utils_new/dataset_audit/analysis.py#L155)
  - [`DatasetAnalyzer.filter_pairs(self, pairs, included_class_ids, total_classes, params, progress_cb=None, include_empty=False, cancel_check=None)`](../utils_new/dataset_audit/analysis.py#L158)
- Класс [`ClassImbalanceAnalyzer`](../utils_new/dataset_audit/analysis.py#L274)
  - [`ClassImbalanceAnalyzer.analyze(class_counts)`](../utils_new/dataset_audit/analysis.py#L276)
- Класс [`ResolutionAnalyzer`](../utils_new/dataset_audit/analysis.py#L328)
  - [`ResolutionAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils_new/dataset_audit/analysis.py#L330)
- Класс [`BrightnessAnalyzer`](../utils_new/dataset_audit/analysis.py#L360)
  - [`BrightnessAnalyzer.analyze(pairs, progress_cb=None, cancel_check=None)`](../utils_new/dataset_audit/analysis.py#L365)
- Класс [`TrainingResultsLoader`](../utils_new/dataset_audit/analysis.py#L389)
  - [`TrainingResultsLoader.scan(folder)`](../utils_new/dataset_audit/analysis.py#L413)
  - [`TrainingResultsLoader.summary_text(data)`](../utils_new/dataset_audit/analysis.py#L464)

<a id="utils-new-dataset-audit-annotation-review"></a>

## utils_new.dataset_audit.annotation_review

Исходник: [utils_new/dataset_audit/annotation_review.py](../utils_new/dataset_audit/annotation_review.py#L1).

**Прямые потребители:** [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.annotation_review](#ui-annotation-review).

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `find_strong_overlaps`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `image_size`, `load_pair_annotations`, `read_objects`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`AnnotationReviewJob`](../utils_new/dataset_audit/annotation_review.py#L8)
  - [`AnnotationReviewJob.__init__(self, pairs, dataset_type=0, input_format='auto', options=None, memory=None)`](../utils_new/dataset_audit/annotation_review.py#L9)
  - [`AnnotationReviewJob.cancel(self)`](../utils_new/dataset_audit/annotation_review.py#L21)
  - [`AnnotationReviewJob.run(self)`](../utils_new/dataset_audit/annotation_review.py#L24)

<a id="utils-new-dataset-audit-atomic"></a>

## utils_new.dataset_audit.atomic

Исходник: [utils_new/dataset_audit/atomic.py](../utils_new/dataset_audit/atomic.py#L1).

**Прямые потребители:** [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `os`, `pathlib.Path`, `tempfile`.

**Объявления:**

- [`write_text_atomic(path, text)`](../utils_new/dataset_audit/atomic.py#L8)
  Явно импортируют: [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

<a id="utils-new-dataset-audit-checks"></a>

## utils_new.dataset_audit.checks

Исходник: [utils_new/dataset_audit/checks.py](../utils_new/dataset_audit/checks.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.embeddings](#ui-dataset-audit-embeddings), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs), [utils_new.dataset_audit.scans](#utils-new-dataset-audit-scans).

**Прямые локальные импорты:**
- [runtime_torch](#runtime-torch): `get_torch_runtime`
- [utils_new.dataset_audit.export](#utils-new-dataset-audit-export): `infer_split`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `read_objects`

**Внешние/stdlib импорты:** `PIL.Image`, `collections.Counter`, `collections.defaultdict`, `cv2`, `hashlib`, `imagehash`, `importlib.util.find_spec`, `numpy`, `os`.

**Объявления:**

- [`_safe_progress(progress_cb, i, total, step=25)`](../utils_new/dataset_audit/checks.py#L28)
  Явно импортируют: [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).
- [`_check_cancel(cancel_check)`](../utils_new/dataset_audit/checks.py#L33)
- [`detect_device()`](../utils_new/dataset_audit/checks.py#L40) — Автоопределение устройства: 'cuda:0', если доступно, иначе 'cpu'.
  Явно импортируют: [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).
- Класс [`DuplicateDetector`](../utils_new/dataset_audit/checks.py#L47)
  - [`DuplicateDetector._file_md5(path, chunk_size=1 << 20)`](../utils_new/dataset_audit/checks.py#L49)
  - [`DuplicateDetector._phash(path, hash_size=8)`](../utils_new/dataset_audit/checks.py#L60)
  - [`DuplicateDetector.find_duplicates(pairs, phash_threshold=8, progress_cb=None, cancel_check=None)`](../utils_new/dataset_audit/checks.py#L72)
- Класс [`LeakageChecker`](../utils_new/dataset_audit/checks.py#L130)
  - [`LeakageChecker.check(duplicate_groups, pairs, split_by_image=None)`](../utils_new/dataset_audit/checks.py#L136)
- Класс [`ObjectDistributionHeatmap`](../utils_new/dataset_audit/checks.py#L165)
  - [`ObjectDistributionHeatmap.compute(pairs, dataset_type=0, grid_size=16, progress_cb=None, cancel_check=None, input_format='auto')`](../utils_new/dataset_audit/checks.py#L167)

<a id="utils-new-dataset-audit-export"></a>

## utils_new.dataset_audit.export

Исходник: [utils_new/dataset_audit/export.py](../utils_new/dataset_audit/export.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `ImageSize`, `from_legacy`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `image_size`, `load_pair_annotations`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`, `write_image`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `write_yolo_labels`

**Внешние/stdlib импорты:** `concurrent.futures.CancelledError`, `numpy`, `os`, `pathlib.Path`, `shutil`, `tempfile`, `yaml`.

**Объявления:**

- [`infer_split(path)`](../utils_new/dataset_audit/export.py#L15)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks).
- [`export_subset(filtered_pairs, out_dir, class_names, progress_cb=None, log_cb=None, included_class_ids=None, *, dataset_type=0, input_format='auto', cancel_check=None, split_by_image=None)`](../utils_new/dataset_audit/export.py#L27)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis).

<a id="utils-new-dataset-audit-jobs"></a>

## utils_new.dataset_audit.jobs

Исходник: [utils_new/dataset_audit/jobs.py](../utils_new/dataset_audit/jobs.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.workers](#ui-dataset-audit-workers).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [runtime_torch](#runtime-torch): `require_torch`
- [utils_new.dataset_audit.atomic](#utils-new-dataset-audit-atomic): `write_text_atomic`
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks): `CLIP_AVAILABLE`, `HDBSCAN_AVAILABLE`, `UMAP_AVAILABLE`, `_CLIP_BACKEND`, `_safe_progress`, `detect_device`
- [utils_new.dataset_audit.matching](#utils-new-dataset-audit-matching): `match_detections`, `project_embeddings`
- [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects): `read_objects`

**Внешние/stdlib импорты:** `PIL.Image`, `datetime.datetime`, `hdbscan`, `logging`, `numpy`, `open_clip`, `os`, `pathlib.Path`, `sklearn.cluster.DBSCAN`, `traceback`, `transformers.CLIPModel`, `transformers.CLIPProcessor`, `ultralytics.YOLO`, `ultralytics.utils.LOGGER`.

**Объявления:**

- Класс [`LossInspectionJob`](../utils_new/dataset_audit/jobs.py#L21)
  - [`LossInspectionJob.__init__(self, pairs, model_path, dataset_type=0, iou_threshold=0.3, conf_threshold=0.25, imgsz=640, device=None, input_format='auto', mask_class_map=None)`](../utils_new/dataset_audit/jobs.py#L22)
  - [`LossInspectionJob.cancel(self)`](../utils_new/dataset_audit/jobs.py#L47)
  - [`LossInspectionJob.run(self)`](../utils_new/dataset_audit/jobs.py#L50)
- Класс [`EmbeddingClusterJob`](../utils_new/dataset_audit/jobs.py#L237)
  - [`EmbeddingClusterJob.__init__(self, pairs, batch_size=16, device=None)`](../utils_new/dataset_audit/jobs.py#L238)
  - [`EmbeddingClusterJob.cancel(self)`](../utils_new/dataset_audit/jobs.py#L246)
  - [`EmbeddingClusterJob.run(self)`](../utils_new/dataset_audit/jobs.py#L249)
- Класс [`ReportJob`](../utils_new/dataset_audit/jobs.py#L377)
  - [`ReportJob.__init__(self, data, save_path, sections)`](../utils_new/dataset_audit/jobs.py#L378)
  - [`ReportJob.cancel(self)`](../utils_new/dataset_audit/jobs.py#L386)
  - [`ReportJob.run(self)`](../utils_new/dataset_audit/jobs.py#L389)
- Класс [`YoloValidationJob`](../utils_new/dataset_audit/jobs.py#L731)
  - [`YoloValidationJob.__init__(self, model_path, data_path, split='val', imgsz=640, conf=0.001, iou=0.6, device=None)`](../utils_new/dataset_audit/jobs.py#L732)
  - [`YoloValidationJob.request_stop(self)`](../utils_new/dataset_audit/jobs.py#L753)
  - [`YoloValidationJob._extract_metrics(self, metrics)`](../utils_new/dataset_audit/jobs.py#L756)
  - [`YoloValidationJob.run(self)`](../utils_new/dataset_audit/jobs.py#L830)

<a id="utils-new-dataset-audit-matching"></a>

## utils_new.dataset_audit.matching

Исходник: [utils_new/dataset_audit/matching.py](../utils_new/dataset_audit/matching.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `numpy`, `sklearn.decomposition.PCA`, `umap`.

**Объявления:**

- [`match_detections(gt_boxes, gt_classes, pred_boxes, pred_classes, threshold=0.3)`](../utils_new/dataset_audit/matching.py#L6)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).
- [`project_embeddings(values, prefer_umap=False, log=lambda text: None)`](../utils_new/dataset_audit/matching.py#L47)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

<a id="utils-new-dataset-audit-objects"></a>

## utils_new.dataset_audit.objects

Исходник: [utils_new/dataset_audit/objects.py](../utils_new/dataset_audit/objects.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`, `to_legacy`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `polygon_area`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_index_mask`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `read_yolo_labels`

**Внешние/stdlib импорты:** `PIL.Image`, `cv2`, `dataclasses.dataclass`, `numpy`, `pathlib.Path`.

**Объявления:**

- Класс [`DatasetObject`](../utils_new/dataset_audit/objects.py#L15)
- [`image_size(path)`](../utils_new/dataset_audit/objects.py#L23)
  Явно импортируют: [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export).
- [`load_pair_annotations(image_path, label_path, format_hint='auto')`](../utils_new/dataset_audit/objects.py#L28)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export).
- [`read_objects(pair, dataset_type=0, input_format='auto')`](../utils_new/dataset_audit/objects.py#L41)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.selection](#ui-dataset-audit-selection), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.annotation_review](#utils-new-dataset-audit-annotation-review), [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks), [utils_new.dataset_audit.jobs](#utils-new-dataset-audit-jobs).

<a id="utils-new-dataset-audit-quality"></a>

## utils_new.dataset_audit.quality

Исходник: [utils_new/dataset_audit/quality.py](../utils_new/dataset_audit/quality.py#L1).

**Прямые потребители:** [tests.test_dataset_workspace](#tests-test-dataset-workspace), [utils_new.dataset_audit.scans](#utils-new-dataset-audit-scans).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `read_image_with_fallback`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`

**Внешние/stdlib импорты:** `cv2`, `dataclasses.dataclass`, `math`, `numpy`.

**Объявления:**

- Класс [`QualityOptions`](../utils_new/dataset_audit/quality.py#L14)
  - [`QualityOptions.__post_init__(self)`](../utils_new/dataset_audit/quality.py#L20)
- Класс [`QualityJob`](../utils_new/dataset_audit/quality.py#L35)
  - [`QualityJob.__init__(self, pairs, options=None)`](../utils_new/dataset_audit/quality.py#L36)
  - [`QualityJob.cancel(self)`](../utils_new/dataset_audit/quality.py#L43)
  - [`QualityJob.run(self)`](../utils_new/dataset_audit/quality.py#L46)

<a id="utils-new-dataset-audit-scans"></a>

## utils_new.dataset_audit.scans

Исходник: [utils_new/dataset_audit/scans.py](../utils_new/dataset_audit/scans.py#L1).

**Прямые потребители:** [tests.test_dataset_workspace](#tests-test-dataset-workspace), [ui.dataset_audit.tasks](#ui-dataset-audit-tasks).

**Прямые локальные импорты:**
- [utils_new.dataset_audit.checks](#utils-new-dataset-audit-checks): `DuplicateDetector`, `LeakageChecker`, `ObjectDistributionHeatmap`
- [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality): `QualityJob`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- Класс [`ScanJob`](../utils_new/dataset_audit/scans.py#L7)
  - [`ScanJob.__init__(self, kind, pairs, options=None)`](../utils_new/dataset_audit/scans.py#L8)
  - [`ScanJob.cancel(self)`](../utils_new/dataset_audit/scans.py#L17)
  - [`ScanJob.run(self)`](../utils_new/dataset_audit/scans.py#L22)

<a id="utils-new-dataset-audit-splits"></a>

## utils_new.dataset_audit.splits

Исходник: [utils_new/dataset_audit/splits.py](../utils_new/dataset_audit/splits.py#L1).

**Прямые потребители:** [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.session](#ui-dataset-audit-session).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `pathlib.Path`, `yaml`.

**Объявления:**

- [`split_memberships(yaml_path, pairs)`](../utils_new/dataset_audit/splits.py#L7)
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_audit.session](#ui-dataset-audit-session).

<a id="utils-new-dataset-catalog-new"></a>

## utils_new.dataset_catalog_new

Исходник: [utils_new/dataset_catalog_new.py](../utils_new/dataset_catalog_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_dataset_audit](#tests-test-dataset-audit), [ui.dataset_source](#ui-dataset-source), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).

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
  Явно импортируют: [tests.test_dataset_audit](#tests-test-dataset-audit), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new).
- [`discover_coco_json(folder)`](../utils_new/dataset_catalog_new.py#L76) — Find only JSON files containing COCO images and annotations fields.
  Явно импортируют: [ui.dataset_source](#ui-dataset-source).
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

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [tests.test_source_unification](#tests-test-source-unification), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [ui.segmentation.export](#ui-segmentation-export), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).

**Прямые локальные импорты:**
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`

**Внешние/stdlib импорты:** `PIL.Image`, `collections.OrderedDict`, `concurrent.futures.ThreadPoolExecutor`, `cv2`, `numpy`, `os`, `pathlib.Path`, `threading`.

**Объявления:**

- [`_read_with_fallback(path, flags)`](../utils_new/dataset_files_new.py#L10) — Read image/mask with Unicode-path fallback (notably Windows Cyrillic).
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image(path)`](../utils_new/dataset_files_new.py#L23) — Read BGR image, keeping existing generator's 3-channel convention.
  Явно импортируют: [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`normalize_index_mask_shape(mask, *, path=None)`](../utils_new/dataset_files_new.py#L31) — Normalize equivalent OpenCV single-channel shapes to (H, W).
- [`read_index_mask(path)`](../utils_new/dataset_files_new.py#L51) — Read indexed PNG as 2D integer array; accept decoder's (H,W,1).
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset), [ui.dataset_audit.duplicates](#ui-dataset-audit-duplicates), [ui.dataset_audit.models](#ui-dataset-audit-models), [utils_new.DatasetGenerator_new](#utils-new-DatasetGenerator-new), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`write_image(path, image)`](../utils_new/dataset_files_new.py#L59) — Write image safely also when cv2.imwrite cannot handle the path.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_mask_diagnostics_new](#utils-new-dataset-mask-diagnostics-new), [utils_new.dataset_output_new](#utils-new-dataset-output-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`read_image_with_fallback(path)`](../utils_new/dataset_files_new.py#L80) — Read image with original depth, TIFF and Unicode-path support.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).
- [`read_mask_with_fallback(path)`](../utils_new/dataset_files_new.py#L119) — Nullable indexed-mask reader; never squeeze a real RGB mask.
- [`image_size_from_file(path)`](../utils_new/dataset_files_new.py#L127) — Full-resolution (width,height), without decoding image pixels if possible.
  Явно импортируют: [ui.segmentation.export](#ui-segmentation-export), [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new).
- [`resize_to_max_side(img, max_side=1024)`](../utils_new/dataset_files_new.py#L141)
  Явно импортируют: [tabs.yolo_find_img](#tabs-yolo-find-img).
- [`normalize_to_uint8(img)`](../utils_new/dataset_files_new.py#L153) — Use the already-shared ImageStats conversion, not a second algorithm.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img).
- [`convert_to_grayscale(img)`](../utils_new/dataset_files_new.py#L160)
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint='auto', hard_max_side=HARD_MAX_SIDE)`](../utils_new/dataset_files_new.py#L166) — Qt-free lazy cache record used by both dataset tabs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- Класс [`LazyImageCache`](../utils_new/dataset_files_new.py#L217)
  - [`LazyImageCache.__init__(self, paths, loader, window=8, prefetch_ahead=4, max_resident=16, max_workers=2)`](../utils_new/dataset_files_new.py#L218)
  - [`LazyImageCache.__len__(self)`](../utils_new/dataset_files_new.py#L236)
  - [`LazyImageCache._trim_locked(self)`](../utils_new/dataset_files_new.py#L239)
  - [`LazyImageCache._load_sync(self, idx)`](../utils_new/dataset_files_new.py#L243)
  - [`LazyImageCache.is_loaded(self, idx)`](../utils_new/dataset_files_new.py#L260)
  - [`LazyImageCache.get_error(self, idx)`](../utils_new/dataset_files_new.py#L264)
  - [`LazyImageCache.get(self, idx)`](../utils_new/dataset_files_new.py#L268)
  - [`LazyImageCache._load_bg(self, idx)`](../utils_new/dataset_files_new.py#L286)
  - [`LazyImageCache.prefetch(self, indices)`](../utils_new/dataset_files_new.py#L313)
  - [`LazyImageCache.trim_around(self, center, keep_behind=3, keep_ahead=None)`](../utils_new/dataset_files_new.py#L328)
  - [`LazyImageCache.shutdown(self)`](../utils_new/dataset_files_new.py#L341)
- [`collect_image_paths(source, *, recursive=False)`](../utils_new/dataset_files_new.py#L361) — Ordered, deduplicated image inputs; recursion is explicit, never implicit.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tabs.yolo_find_img](#tabs-yolo-find-img), [tests.test_source_unification](#tests-test-source-unification), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).

<a id="utils-new-dataset-image-stats-new"></a>

## utils_new.dataset_image_stats_new

Исходник: [utils_new/dataset_image_stats_new.py](../utils_new/dataset_image_stats_new.py#L1).

**Прямые потребители:** [ui.image_conversion](#ui-image-conversion), [utils_new.dataset_audit.analysis](#utils-new-dataset-audit-analysis), [utils_new.dataset_audit.quality](#utils-new-dataset-audit-quality), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

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
- [ui.annotation_review](#ui-annotation-review): `AnnotationReviewWidget`
- [ui.dataset_source](#ui-dataset-source): `dataset_source as source_dialogs`
- [ui.pair_validation_dialog_new](#ui-pair-validation-dialog-new): `PairValidationDialog`
- [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new): `CatalogSourceError`, `collect_label_classes`, `collect_mask_classes`, `dense_class_remap`, `import_coco_catalog`, `load_unet_catalog`, `load_yolo_catalog`, `resolve_class_name`
- [utils_new.dataset_coco_new](#utils-new-dataset-coco-new): `write_yolo_txt_for_coco_items`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new): `build_pair_display`, `scan_detection_pairs`, `scan_image_only_pairs`, `scan_segmentation_pairs`
- [utils_new.dataset_unet_new](#utils-new-dataset-unet-new): `is_unet_yaml`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtGui.QColor`, `PyQt5.QtWidgets.QApplication`, `PyQt5.QtWidgets.QCheckBox`, `PyQt5.QtWidgets.QListWidgetItem`, `PyQt5.QtWidgets.QMessageBox`, `PyQt5.QtWidgets.QTableWidgetItem`, `os`, `tempfile`, `traceback`.

**Объявления:**

- Класс [`DatasetLoaderMixin`](../utils_new/dataset_loader_new.py#L45)
  - [`DatasetLoaderMixin._after_load_hook(self)`](../utils_new/dataset_loader_new.py#L67)
  - [`DatasetLoaderMixin._after_type_change_hook(self, new_type)`](../utils_new/dataset_loader_new.py#L74)
  - [`DatasetLoaderMixin._sync_all_states(self, *_)`](../utils_new/dataset_loader_new.py#L79)
  - [`DatasetLoaderMixin._refresh_mask_pixel_column(self)`](../utils_new/dataset_loader_new.py#L88)
  - [`DatasetLoaderMixin.log(self, message)`](../utils_new/dataset_loader_new.py#L98)
  - [`DatasetLoaderMixin._log_debug(self, msg)`](../utils_new/dataset_loader_new.py#L107)
  - [`DatasetLoaderMixin._log_exception(self, context, exc=None)`](../utils_new/dataset_loader_new.py#L114)
  - [`DatasetLoaderMixin._copy_selected_rows_to_clipboard(self)`](../utils_new/dataset_loader_new.py#L126)
  - [`DatasetLoaderMixin._on_file_list_double_clicked(self, item)`](../utils_new/dataset_loader_new.py#L134)
  - [`DatasetLoaderMixin.select_images_folder(self)`](../utils_new/dataset_loader_new.py#L146)
  - [`DatasetLoaderMixin.select_labels_folder(self)`](../utils_new/dataset_loader_new.py#L156)
  - [`DatasetLoaderMixin.load_from_yaml(self)`](../utils_new/dataset_loader_new.py#L176)
  - [`DatasetLoaderMixin._load_yolo_yaml_from_path(self, file_path)`](../utils_new/dataset_loader_new.py#L201)
  - [`DatasetLoaderMixin.load_from_coco(self)`](../utils_new/dataset_loader_new.py#L254)
  - [`DatasetLoaderMixin.add_from_coco(self)`](../utils_new/dataset_loader_new.py#L257)
  - [`DatasetLoaderMixin._handle_load_coco(self, replace)`](../utils_new/dataset_loader_new.py#L260)
  - [`DatasetLoaderMixin._get_coco_tmp_labels_dir(self, first_json_path=None)`](../utils_new/dataset_loader_new.py#L265)
  - [`DatasetLoaderMixin._load_coco_from_paths(self, json_paths, replace=True)`](../utils_new/dataset_loader_new.py#L273)
  - [`DatasetLoaderMixin.load_from_unet_yaml(self)`](../utils_new/dataset_loader_new.py#L356)
  - [`DatasetLoaderMixin._load_unet_from_path(self, yaml_path)`](../utils_new/dataset_loader_new.py#L362)
  - [`DatasetLoaderMixin._apply_loaded_pairs(self, pairs, input_type, input_format, class_names_from_yaml=None, class_names_override=None, source_label='', source_yaml=None)`](../utils_new/dataset_loader_new.py#L410)
  - [`DatasetLoaderMixin.scan_pairs(self)`](../utils_new/dataset_loader_new.py#L473)
  - [`DatasetLoaderMixin._validate_all_pairs(self)`](../utils_new/dataset_loader_new.py#L546)
  - [`DatasetLoaderMixin._recompute_color_stats(self)`](../utils_new/dataset_loader_new.py#L570)
  - [`DatasetLoaderMixin._update_color_stats_label(self)`](../utils_new/dataset_loader_new.py#L578)
  - [`DatasetLoaderMixin._open_pair_validation(self, pair)`](../utils_new/dataset_loader_new.py#L589)
  - [`DatasetLoaderMixin._make_list_item(self, text, tag=None, error_msg='')`](../utils_new/dataset_loader_new.py#L620)
  - [`DatasetLoaderMixin._rebuild_file_list_display(self, pairs=None)`](../utils_new/dataset_loader_new.py#L644)
  - [`DatasetLoaderMixin.closeEvent(self, event)`](../utils_new/dataset_loader_new.py#L744)
  - [`DatasetLoaderMixin.shutdown_annotation_review(self)`](../utils_new/dataset_loader_new.py#L748)
  - [`DatasetLoaderMixin._is_include_empty(self)`](../utils_new/dataset_loader_new.py#L752)
  - [`DatasetLoaderMixin._on_include_empty_toggled(self, *_)`](../utils_new/dataset_loader_new.py#L755)
  - [`DatasetLoaderMixin._on_dataset_type_changed(self, idx)`](../utils_new/dataset_loader_new.py#L761)
  - [`DatasetLoaderMixin._on_input_format_changed(self, idx)`](../utils_new/dataset_loader_new.py#L786)
  - [`DatasetLoaderMixin._fill_class_table_from_dict(self, class_names_dict, preserve_include=True)`](../utils_new/dataset_loader_new.py#L798)
  - [`DatasetLoaderMixin._get_yaml_name_for_id(self, cid)`](../utils_new/dataset_loader_new.py#L842)
  - [`DatasetLoaderMixin.collect_classes(self)`](../utils_new/dataset_loader_new.py#L845)
  - [`DatasetLoaderMixin.update_included_classes(self)`](../utils_new/dataset_loader_new.py#L935)
  - [`DatasetLoaderMixin.on_class_table_item_changed(self, item)`](../utils_new/dataset_loader_new.py#L1020)
  - [`DatasetLoaderMixin._update_pair_count_label(self, extra_text, error=False)`](../utils_new/dataset_loader_new.py#L1026)

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
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `_read_with_fallback`, `collect_image_paths`, `write_image`
- [utils_new.dataset_image_stats_new](#utils-new-dataset-image-stats-new): `ImageStats`
- [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new): `IMG_EXTENSIONS`, `INPUT_FORMAT_AUTO`, `PairValidator`, `read_mask_with_fallback`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `os`.

**Объявления:**

- [`_list_images(folder)`](../utils_new/dataset_pair_tools_new.py#L18)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_detection_pairs(images_folder, labels_folder, log_cb=None, format_hint=INPUT_FORMAT_AUTO)`](../utils_new/dataset_pair_tools_new.py#L23)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_segmentation_pairs(images_folder, masks_folder, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L43)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`scan_image_only_pairs(images_folder)`](../utils_new/dataset_pair_tools_new.py#L78)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_conversion(pairs, validator, target)`](../utils_new/dataset_pair_tools_new.py#L83)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_color(pairs, target, progress_cb=None, log_cb=None)`](../utils_new/dataset_pair_tools_new.py#L98)
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`select_pairs_for_8bit_png(pairs, validator)`](../utils_new/dataset_pair_tools_new.py#L126) — Возвращает (all_pairs, need_convert).
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`convert_pairs_to_8bit_png(pairs, progress_cb=None, log_cb=None, skip_masks=False)`](../utils_new/dataset_pair_tools_new.py#L141) — skip_masks: если True — pairs трактуются как pairs из (img, mask),
  Явно импортируют: [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).
- [`build_pair_display(pair, validator, dataset_type, include_empty)`](../utils_new/dataset_pair_tools_new.py#L184)
  Явно импортируют: [utils_new.dataset_loader_new](#utils-new-dataset-loader-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

<a id="utils-new-dataset-pair-validator-new"></a>

## utils_new.dataset_pair_validator_new

Исходник: [utils_new/dataset_pair_validator_new.py](../utils_new/dataset_pair_validator_new.py#L1).

**Прямые потребители:** [utils_new.dataset_pair_tools_new](#utils-new-dataset-pair-tools-new), [utils_new.dataset_validation_new](#utils-new-dataset-validation-new).

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
  Явно импортируют: [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_yolo_yaml(yaml_path, log_cb=None)`](../utils_new/dataset_yaml_new.py#L151) — Парсит YOLO data.yaml.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new).
- [`build_yolo_save_target_map(yaml_path, image_paths, *, root=None)`](../utils_new/dataset_yaml_new.py#L233) — Resolve canonical labels/<split>/<image>.txt destinations for YOLO YAML.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge).

<a id="utils-new-dataset-yolo-labels-new"></a>

## utils_new.dataset_yolo_labels_new

Исходник: [utils_new/dataset_yolo_labels_new.py](../utils_new/dataset_yolo_labels_new.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_annotation_objects](#tests-test-annotation-objects), [tests.test_label_save_merge](#tests-test-label-save-merge), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.annotation_objects](#ui-annotation-objects), [ui.segmentation.export](#ui-segmentation-export), [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new), [utils_new.segmentation.export](#utils-new-segmentation-export), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `Annotation`, `ImageSize`, `from_legacy`, `to_legacy`
- [utils_new.annotations.geometry](#utils-new-annotations-geometry): `is_rectangle`
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `image_size_from_file`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `dataclasses.dataclass`, `math`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`_xywhr_points(vals, size=None)`](../utils_new/dataset_yolo_labels_new.py#L16) — Decode normalized ``cx,cy,w,h,angle`` into rectangle corners.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`read_yolo_labels(path, size: ImageSize, *, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L43) — Read YOLO txt variants into canonical pixel-space annotations.
  Явно импортируют: [utils_new.autoreseach_utils.ground_truth](#utils-new-autoreseach-utils-ground-truth), [utils_new.dataset_audit.objects](#utils-new-dataset-audit-objects), [utils_new.dataset_catalog_new](#utils-new-dataset-catalog-new), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_pair_validator_new](#utils-new-dataset-pair-validator-new), [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels).
- [`_normalized_xywhr(ann, size)`](../utils_new/dataset_yolo_labels_new.py#L115) — OBB as ``cx/W, cy/H, width_px/W, height_px/H, radians_px``.
  Явно импортируют: [utils_new.dataset_io_new](#utils-new-dataset-io-new).
- [`serialize_yolo_labels(annotations, size: ImageSize, *, obb_format='corners')`](../utils_new/dataset_yolo_labels_new.py#L138) — Return exactly the TXT contents used by the writer, without filesystem I/O.
  Явно импортируют: [tests.test_annotation_objects](#tests-test-annotation-objects), [ui.annotation_objects](#ui-annotation-objects).
- [`write_yolo_labels(path, annotations, size: ImageSize, *, obb_format='corners')`](../utils_new/dataset_yolo_labels_new.py#L166)
  Явно импортируют: [tests.test_annotation_objects](#tests-test-annotation-objects), [utils_new.dataset_audit.export](#utils-new-dataset-audit-export), [utils_new.dataset_io_new](#utils-new-dataset-io-new), [utils_new.dataset_unet_new](#utils-new-dataset-unet-new).
- [`canonicalize_obb_corners(norm_points, width=1, height=1)`](../utils_new/dataset_yolo_labels_new.py#L184) — Order the original four vertices, without altering their coordinates.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`load_annotations_for_image(label_path, image_path, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L207) — Read normalized tuples using the associated image's *real* dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`save_annotations_for_image(annotations, label_path, image_path, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L215) — Prevent Save All from using the current (different) image dimensions.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset).
- [`_tolerant_editor_rows(path, size, format_hint='auto')`](../utils_new/dataset_yolo_labels_new.py#L224) — Recover syntactically valid, geometrically invalid rows for manual repair.
- [`load_annotations(txt_path, img_w, img_h, format_hint='auto', log=None)`](../utils_new/dataset_yolo_labels_new.py#L266)
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared), [utils_new.dataset_files_new](#utils-new-dataset-files-new), [utils_new.segmentation.image_session](#utils-new-segmentation-image-session).
- [`save_annotations(annotations, txt_path, img_w, img_h, obb_as_xywhr=False)`](../utils_new/dataset_yolo_labels_new.py#L286)
  Явно импортируют: [utils_new.segmentation.export](#utils-new-segmentation-export).
- Класс [`LabelSavePlan`](../utils_new/dataset_yolo_labels_new.py#L333)
- [`build_label_save_plan(entries)`](../utils_new/dataset_yolo_labels_new.py#L341) — Refuse collisions; count only existing *destination* .txt files.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [ui.segmentation.export](#ui-segmentation-export).
- [`format_label_save_confirmation(plan, *, operation='Save', obb_format='corners', source_kind='images', unloaded_count=0)`](../utils_new/dataset_yolo_labels_new.py#L362) — Human-readable summary for the shared Qt Yes/No dialogs.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset), [tabs.viewing_dataset](#tabs-viewing-dataset), [tests.test_label_save_merge](#tests-test-label-save-merge), [ui.segmentation.export](#ui-segmentation-export).

<a id="utils-new-json-store"></a>

## utils_new.json_store

Исходник: [utils_new/json_store.py](../utils_new/json_store.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `json`, `os`, `pathlib.Path`, `tempfile`.

**Объявления:**

- [`read_json_object(path)`](../utils_new/json_store.py#L9)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).
- [`write_json_atomic(path, value)`](../utils_new/json_store.py#L17)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tabs.threshold_methods](#tabs-threshold-methods), [utils_new.presets_store](#utils-new-presets-store), [utils_new.settings_store](#utils-new-settings-store).

<a id="utils-new-morphology"></a>

## utils_new.morphology

Исходник: [utils_new/morphology.py](../utils_new/morphology.py#L1).

**Прямые потребители:** [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.autoreseach_ui.settings_dialog](#ui-autoreseach-ui-settings-dialog), [ui.morphology_ui](#ui-morphology-ui), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_new.segmentation.presets](#utils-new-segmentation-presets), [utils_new.threshold_processing](#utils-new-threshold-processing).

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
  Явно импортируют: [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.annotation_processing](#utils-new-annotation-processing), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_new.threshold_processing](#utils-new-threshold-processing).

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

<a id="utils-new-segmentation-autoencoder"></a>

## utils_new.segmentation.autoencoder

Исходник: [utils_new/segmentation/autoencoder.py](../utils_new/segmentation/autoencoder.py#L1).

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `concurrent.futures.CancelledError`, `cv2`, `numpy`, `torch`, `torch.nn`, `torch.nn.functional`.

**Объявления:**

- Класс [`ConvEncoder`](../utils_new/segmentation/autoencoder.py#L39) (`nn.Module`)
  - [`ConvEncoder.__init__(self, in_channels)`](../utils_new/segmentation/autoencoder.py#L40)
  - [`ConvEncoder.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L61)
- Класс [`ConvDecoder`](../utils_new/segmentation/autoencoder.py#L65) (`nn.Module`)
  - [`ConvDecoder.__init__(self, in_channels)`](../utils_new/segmentation/autoencoder.py#L66)
  - [`ConvDecoder.forward(self, z)`](../utils_new/segmentation/autoencoder.py#L82)
- Класс [`ConvAutoencoder`](../utils_new/segmentation/autoencoder.py#L86) (`nn.Module`)
  - [`ConvAutoencoder.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_new/segmentation/autoencoder.py#L87)
  - [`ConvAutoencoder.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L97)
- Класс [`ConvVAE`](../utils_new/segmentation/autoencoder.py#L105) (`nn.Module`)
  - [`ConvVAE.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_new/segmentation/autoencoder.py#L106)
  - [`ConvVAE.reparameterize(self, mu, logvar)`](../utils_new/segmentation/autoencoder.py#L118)
  - [`ConvVAE.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L126)
- [`_gn(channels, max_groups=8)`](../utils_new/segmentation/autoencoder.py#L139) — GroupNorm с автоматическим подбором числа групп — как в обучении.
- Класс [`ConvEncoderGN`](../utils_new/segmentation/autoencoder.py#L147) (`nn.Module`)
  - [`ConvEncoderGN.__init__(self, in_channels)`](../utils_new/segmentation/autoencoder.py#L148)
  - [`ConvEncoderGN.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L169)
- Класс [`ConvDecoderGN`](../utils_new/segmentation/autoencoder.py#L173) (`nn.Module`)
  - [`ConvDecoderGN.__init__(self, in_channels)`](../utils_new/segmentation/autoencoder.py#L174)
  - [`ConvDecoderGN.forward(self, z)`](../utils_new/segmentation/autoencoder.py#L190)
- Класс [`ConvAutoencoderGN`](../utils_new/segmentation/autoencoder.py#L194) (`nn.Module`)
  - [`ConvAutoencoderGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_new/segmentation/autoencoder.py#L195)
  - [`ConvAutoencoderGN.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L205)
- Класс [`ConvVAEGN`](../utils_new/segmentation/autoencoder.py#L213) (`nn.Module`)
  - [`ConvVAEGN.__init__(self, in_channels, latent_dim, img_h, img_w)`](../utils_new/segmentation/autoencoder.py#L214)
  - [`ConvVAEGN.reparameterize(self, mu, logvar)`](../utils_new/segmentation/autoencoder.py#L226)
  - [`ConvVAEGN.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L234)
- Класс [`UNetEncoder`](../utils_new/segmentation/autoencoder.py#L247) (`nn.Module`)
  - [`UNetEncoder.__init__(self, in_channels, base=32)`](../utils_new/segmentation/autoencoder.py#L248)
  - [`UNetEncoder.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L285)
- Класс [`UNetDecoder`](../utils_new/segmentation/autoencoder.py#L293) (`nn.Module`)
  - [`UNetDecoder.__init__(self, out_channels, base=32)`](../utils_new/segmentation/autoencoder.py#L294)
  - [`UNetDecoder.forward(self, h, f0, f1, f2)`](../utils_new/segmentation/autoencoder.py#L336)
- Класс [`UNetAE`](../utils_new/segmentation/autoencoder.py#L349) (`nn.Module`)
  - [`UNetAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_new/segmentation/autoencoder.py#L350)
  - [`UNetAE.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L362)
- Класс [`UNetVAE`](../utils_new/segmentation/autoencoder.py#L370) (`nn.Module`)
  - [`UNetVAE.__init__(self, in_channels, latent_channels, img_h, img_w, base=32)`](../utils_new/segmentation/autoencoder.py#L371)
  - [`UNetVAE.reparameterize(self, mu, logvar)`](../utils_new/segmentation/autoencoder.py#L384)
  - [`UNetVAE.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L392)
- [`_norm_v2(c, kind='gn')`](../utils_new/segmentation/autoencoder.py#L407) — GroupNorm с тем же подбором групп, что и в тренировочном скрипте.
- [`_act_v2()`](../utils_new/segmentation/autoencoder.py#L417)
- Класс [`ConvBlockV2`](../utils_new/segmentation/autoencoder.py#L421) (`nn.Module`)
  - [`ConvBlockV2.__init__(self, in_c, out_c, stride=1, n_convs=2, norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L422)
  - [`ConvBlockV2.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L434)
- Класс [`DilatedResidualV2`](../utils_new/segmentation/autoencoder.py#L438) (`nn.Module`)
  - [`DilatedResidualV2.__init__(self, c, dilations=(2, 4), norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L439)
  - [`DilatedResidualV2.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L455)
- Класс [`UNetEncoderV2`](../utils_new/segmentation/autoencoder.py#L461) (`nn.Module`)
  - [`UNetEncoderV2.__init__(self, in_channels, base=32, dilations=(2, 4), norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L462)
  - [`UNetEncoderV2.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L471)
- Класс [`DecoderBlockV2`](../utils_new/segmentation/autoencoder.py#L480) (`nn.Module`)
  - [`DecoderBlockV2.__init__(self, in_c, skip_c, out_c, norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L481)
  - [`DecoderBlockV2.forward(self, x, skip)`](../utils_new/segmentation/autoencoder.py#L491)
- Класс [`UNetDecoderV2`](../utils_new/segmentation/autoencoder.py#L503) (`nn.Module`)
  - [`UNetDecoderV2.__init__(self, out_channels, base=32, norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L504)
  - [`UNetDecoderV2.forward(self, h, f0, f1, f2)`](../utils_new/segmentation/autoencoder.py#L515)
- Класс [`UNetAEV2`](../utils_new/segmentation/autoencoder.py#L522) (`nn.Module`)
  - [`UNetAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn')`](../utils_new/segmentation/autoencoder.py#L523)
  - [`UNetAEV2.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L547)
- Класс [`UNetVAEV2`](../utils_new/segmentation/autoencoder.py#L555) (`nn.Module`)
  - [`UNetVAEV2.__init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32, dilations=(2, 4), norm_type='gn', logvar_min=-10.0, logvar_max=10.0)`](../utils_new/segmentation/autoencoder.py#L556)
  - [`UNetVAEV2.reparameterize(self, mu, logvar)`](../utils_new/segmentation/autoencoder.py#L585)
  - [`UNetVAEV2.forward(self, x)`](../utils_new/segmentation/autoencoder.py#L593)
- [`_median_blur_torch(x, k=3)`](../utils_new/segmentation/autoencoder.py#L614) — x: [B,C,H,W]. Медианный фильтр через unfold (совместим с обучением).
- [`compute_edge_map_torch(img_tensor, edge_kernel='scharr', median_kernel=3, edge_lo=0.1, edge_hi=0.35)`](../utils_new/segmentation/autoencoder.py#L625) — img_tensor: [C,H,W] или [B,C,H,W], float32 в [0,1].
- [`detect_arch_from_state_dict(state_dict)`](../utils_new/segmentation/autoencoder.py#L669) — Определяет архитектуру по ключам state_dict.
- [`detect_arch(cfg, state_dict=None)`](../utils_new/segmentation/autoencoder.py#L686) — Определяет архитектуру AE/VAE.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb).
- [`build_ae_model_from_cfg(cfg, arch=None)`](../utils_new/segmentation/autoencoder.py#L733) — Собирает модель AE/VAE по config чекпойнта.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration).
- [`preprocess_image_for_ae(img_bgr, cfg)`](../utils_new/segmentation/autoencoder.py#L825) — Приводит произвольный BGR-снимок к тензору [C, H, W] float32 [0..1],
- [`infer_autoencoder(image, model, cfg, device, cancel=None)`](../utils_new/segmentation/autoencoder.py#L936) — Validated reconstruction/error on the model grid; no widgets or Qt.
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration).

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

**Прямые потребители:** [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui.segmentation.image_input](#ui-segmentation-image-input), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).

**Прямые локальные импорты:**
- [utils_new.dataset_files_new](#utils-new-dataset-files-new): `HARD_MAX_SIDE`, `LazyImageCache`, `collect_image_paths`, `read_image_with_fallback`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `load_annotations`

**Внешние/stdlib импорты:** `cv2`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`image_paths(source)`](../utils_new/segmentation/image_session.py#L21)
  Явно импортируют: [tabs.yolo_test_aeb](#tabs-yolo-test-aeb), [ui.segmentation.image_input](#ui-segmentation-image-input).
- [`legacy_uint8(image)`](../utils_new/segmentation/image_session.py#L25)
- [`resize_legacy(image, max_side)`](../utils_new/segmentation/image_session.py#L35)
- [`load_one_image_item(image_path, resize_enabled=True, max_side=1024, safe_max_side=None, load_annotations_flag=True, format_hint='auto')`](../utils_new/segmentation/image_session.py#L47)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`create_cache(source, **loader_options)`](../utils_new/segmentation/image_session.py#L92)
  Явно импортируют: [ui.segmentation.image_input](#ui-segmentation-image-input).

<a id="utils-new-segmentation-model-export"></a>

## utils_new.segmentation.model_export

Исходник: [utils_new/segmentation/model_export.py](../utils_new/segmentation/model_export.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `torch`.

**Объявления:**

- [`stride_padded_model(model)`](../utils_new/segmentation/model_export.py#L4)
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker).

<a id="utils-new-segmentation-model-metrics"></a>

## utils_new.segmentation.model_metrics

Исходник: [utils_new/segmentation/model_metrics.py](../utils_new/segmentation/model_metrics.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `torch`.

**Объявления:**

- [`prediction_labels(logits, num_classes, threshold=0.5)`](../utils_new/segmentation/model_metrics.py#L4)
  Явно импортируют: [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).
- [`segmentation_metrics(logits, target, num_classes, *, threshold=0.5, ignore_index=None)`](../utils_new/segmentation/model_metrics.py#L15)
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).
- [`confusion_matrix(logits, target, num_classes, *, threshold=0.5, ignore_index=None)`](../utils_new/segmentation/model_metrics.py#L76)
  Явно импортируют: [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).

<a id="utils-new-segmentation-model-output"></a>

## utils_new.segmentation.model_output

Исходник: [utils_new/segmentation/model_output.py](../utils_new/segmentation/model_output.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `numpy`.

**Объявления:**

- [`decode_output(output, *, output_kind='logits', threshold=0.5, target_shape=None)`](../utils_new/segmentation/model_output.py#L7)
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops).

<a id="utils-new-segmentation-operations"></a>

## utils_new.segmentation.operations

Исходник: [utils_new/segmentation/operations.py](../utils_new/segmentation/operations.py#L1).

**Прямые потребители:** [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_segmentation_tasks](#tests-test-segmentation-tasks), [ui.segmentation.computation](#ui-segmentation-computation).

**Прямые локальные импорты:**
- [utils_new.segmentation.training_labels](#utils-new-segmentation-training-labels): `manual_target`
- [utils_ops.deep_learning_ops](#utils-ops-deep-learning-ops): `create_segmentor`
- [utils_ops.interactive_ops](#utils-ops-interactive-ops): `interactive_ops as ops`
- [utils_ops.traditional_ml_ops](#utils-ops-traditional-ml-ops): `traditional_ml_ops as ops`

**Внешние/stdlib импорты:** `concurrent.futures.CancelledError`, `cv2`, `joblib`, `numpy`, `sklearn.preprocessing.StandardScaler`.

**Объявления:**

- [`checkpoint(cancel)`](../utils_new/segmentation/operations.py#L9)
- [`interactive(method, image, gray, rect, fg, bg, params, cancel)`](../utils_new/segmentation/operations.py#L14)
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms).
- [`features(image, spec)`](../utils_new/segmentation/operations.py#L62)
- [`prepare_training(images, grays, spec, cancel, paths=None)`](../utils_new/segmentation/operations.py#L72)
- [`train_prepared(x, y, spec, cancel)`](../utils_new/segmentation/operations.py#L105)
- [`predict_ml(image, gray, model, metadata, spec, cluster, cancel)`](../utils_new/segmentation/operations.py#L137)
- [`optimal_k(images, spec, cancel)`](../utils_new/segmentation/operations.py#L164)
- [`predict_dl(model, image, parameters, overrides, cancel)`](../utils_new/segmentation/operations.py#L179)
- [`load_dl(name, path, device, cancel)`](../utils_new/segmentation/operations.py#L189)
- [`load_ml(path, cancel)`](../utils_new/segmentation/operations.py#L202)

<a id="utils-new-segmentation-postprocessing"></a>

## utils_new.segmentation.postprocessing

Исходник: [utils_new/segmentation/postprocessing.py](../utils_new/segmentation/postprocessing.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_new.threshold_processing](#utils-new-threshold-processing).

**Прямые локальные импорты:**
- [utils_new.annotation_processing](#utils-new-annotation-processing): `extract_annotations_from_mask`
- [utils_new.annotations.modes](#utils-new-annotations-modes): `supports_hull`, `validate_mode`

**Внешние/stdlib импорты:** нет.

**Объявления:**

- [`extract_objects(image, binary, mode, use_hull=False)`](../utils_new/segmentation/postprocessing.py#L6)
  Явно импортируют: [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tabs.gradient_methods](#tabs-gradient-methods), [tabs.interactive_methods](#tabs-interactive-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [ui.segmentation.presentation](#ui-segmentation-presentation), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_new.threshold_processing](#utils-new-threshold-processing).

<a id="utils-new-segmentation-presets"></a>

## utils_new.segmentation.presets

Исходник: [utils_new/segmentation/presets.py](../utils_new/segmentation/presets.py#L1).

**Прямые потребители:** [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space), [utils_new.threshold_processing](#utils-new-threshold-processing).

**Прямые локальные импорты:**
- [utils_new.annotations.modes](#utils-new-annotations-modes): `validate_mode`
- [utils_new.morphology](#utils-new-morphology): `SUPPORTED_OPERATIONS`, `SUPPORTED_SHAPES`

**Внешние/stdlib импорты:** `copy.deepcopy`.

**Объявления:**

- [`default_morphology()`](../utils_new/segmentation/presets.py#L12)
  Явно импортируют: [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space).
- [`validate_morphology(state)`](../utils_new/segmentation/presets.py#L25)
  Явно импортируют: [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space).
- [`validate_preset(state, kind=None)`](../utils_new/segmentation/presets.py#L60)
  Явно импортируют: [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space), [utils_new.threshold_processing](#utils-new-threshold-processing).
- [`make_preset(kind, **parameters)`](../utils_new/segmentation/presets.py#L98)
  Явно импортируют: [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [ui.segmentation.preset_bindings](#ui-segmentation-preset-bindings), [utils_new.autoreseach_utils.search_space](#utils-new-autoreseach-utils-search-space).

<a id="utils-new-segmentation-result"></a>

## utils_new.segmentation.result

Исходник: [utils_new/segmentation/result.py](../utils_new/segmentation/result.py#L1).

**Прямые потребители:** [tabs.deep_learning_methods](#tabs-deep-learning-methods), [tests.test_segmentation_tasks](#tests-test-segmentation-tasks), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.controller](#ui-segmentation-controller), [ui.segmentation.errors](#ui-segmentation-errors), [ui.segmentation.export](#ui-segmentation-export), [ui.segmentation.jobs](#ui-segmentation-jobs), [ui.segmentation.presentation](#ui-segmentation-presentation), [ui.segmentation.session](#ui-segmentation-session).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `dataclasses.dataclass`, `enum.Enum`, `numpy`.

**Объявления:**

- Класс [`ResultStatus`](../utils_new/segmentation/result.py#L10) (`str, Enum`)
- Класс [`SegmentationResult`](../utils_new/segmentation/result.py#L19)
- Класс [`ImageFrame`](../utils_new/segmentation/result.py#L26)
- [`binary_mask(mask, image_shape)`](../utils_new/segmentation/result.py#L36) — Convert binary/probability/label-union output to uint8 0/255.
  Явно импортируют: [tests.test_segmentation_tasks](#tests-test-segmentation-tasks), [tests.test_segmentation_unification](#tests-test-segmentation-unification), [ui.segmentation.computation](#ui-segmentation-computation), [ui.segmentation.presentation](#ui-segmentation-presentation).
- [`masked_preview(image, mask)`](../utils_new/segmentation/result.py#L70)
  Явно импортируют: [tests.test_segmentation_unification](#tests-test-segmentation-unification), [ui.segmentation.presentation](#ui-segmentation-presentation).

<a id="utils-new-segmentation-training-contracts"></a>

## utils_new.segmentation.training_contracts

Исходник: [utils_new/segmentation/training_contracts.py](../utils_new/segmentation/training_contracts.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `PIL.Image`, `numpy`, `torch.nn.functional`.

**Объявления:**

- [`validate_mask_files(paths, num_classes, ignore_index=None)`](../utils_new/segmentation/training_contracts.py#L7) — Read EVERY source mask before training; never silently relabel bad IDs.
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_DeepLabV3_worker](#utils-training-train-DeepLabV3-worker), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).
- [`apply_decoder_dropout(model, probability)`](../utils_new/segmentation/training_contracts.py#L24) — SMP U-Net has no decoder_dropout argument. Apply it to decoder features.
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_unet_worker](#utils-training-train-unet-worker), [utils_training.train_unet_worker_рабочий](#utils-training-train-unet-worker-рабочий).

<a id="utils-new-segmentation-training-labels"></a>

## utils_new.segmentation.training_labels

Исходник: [utils_new/segmentation/training_labels.py](../utils_new/segmentation/training_labels.py#L1).

**Прямые потребители:** [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_new.segmentation.operations](#utils-new-segmentation-operations).

**Прямые локальные импорты:**
- [utils_new.annotations.__init__](#utils-new-annotations---init--): `ImageSize`
- [utils_new.annotations.masks](#utils-new-annotations-masks): `annotation_binary_mask`
- [utils_new.dataset_yolo_labels_new](#utils-new-dataset-yolo-labels-new): `read_yolo_labels`

**Внешние/stdlib импорты:** `PIL.Image`, `numpy`, `pathlib.Path`.

**Объявления:**

- [`manual_target(image_path, shape)`](../utils_new/segmentation/training_labels.py#L13) — YOLO TXT next to image, or images/... -> labels/...; union of all objects.
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_new.segmentation.operations](#utils-new-segmentation-operations).

<a id="utils-new-segmentation-yolo-metrics"></a>

## utils_new.segmentation.yolo_metrics

Исходник: [utils_new/segmentation/yolo_metrics.py](../utils_new/segmentation/yolo_metrics.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `math`.

**Объявления:**

- [`validation_metrics(result)`](../utils_new/segmentation/yolo_metrics.py#L6)
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts), [utils_training.train_yolo_worker](#utils-training-train-yolo-worker).

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

**Прямые потребители:** [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tabs.threshold_methods](#tabs-threshold-methods), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_pixel_morphology](#tests-test-pixel-morphology), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).

**Прямые локальные импорты:**
- [utils_new.morphology](#utils-new-morphology): `apply_morphology`
- [utils_new.segmentation.postprocessing](#utils-new-segmentation-postprocessing): `extract_objects`
- [utils_new.segmentation.presets](#utils-new-segmentation-presets): `validate_preset`

**Внешние/stdlib импорты:** `__future__.annotations`, `cv2`, `numpy`.

**Объявления:**

- [`simple_threshold(img, thresh)`](../utils_new/threshold_processing.py#L30)
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`otsu_threshold(img)`](../utils_new/threshold_processing.py#L35)
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`triangle_threshold(img)`](../utils_new/threshold_processing.py#L40)
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`niblack_threshold(img, window_size=15, k=-0.2)`](../utils_new/threshold_processing.py#L45)
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`sauvola_threshold(img, window_size=25, k=0.2, R=128)`](../utils_new/threshold_processing.py#L58)
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`isodata_threshold(img, initial_threshold=128)`](../utils_new/threshold_processing.py#L73)
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`background_symmetry_threshold(img, excess_threshold=0.2)`](../utils_new/threshold_processing.py#L90) — Project histogram heuristic; symmetric background peak, excess tail.
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`row_adaptive_threshold(img, window_size=50, k=0.5)`](../utils_new/threshold_processing.py#L116)
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`normalize_method_name(method_name)`](../utils_new/threshold_processing.py#L136) — Приводит имя метода к каноническому виду для UI.
  Явно импортируют: [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`apply_threshold_method(gray_img, method_name, params)`](../utils_new/threshold_processing.py#L162) — Применяет метод бинаризации по имени и параметрам.
  Явно импортируют: [tabs.threshold_methods](#tabs-threshold-methods), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_threshold_autoresearch](#tests-test-threshold-autoresearch), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search), [utils_ops.segmentation_ops](#utils-ops-segmentation-ops).
- [`apply_threshold_preset(image, gray, preset)`](../utils_new/threshold_processing.py#L227) — Canonical preset pipeline shared by preview, ensemble and research.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [utils_new.autoreseach_utils.search](#utils-new-autoreseach-utils-search).
- [`get_detections_from_preset(image, gray, preset, confidence=0.7, *, class_id=0)`](../utils_new/threshold_processing.py#L249) — Return enclosing Detect boxes with a user-assigned, uncalibrated score.
  Явно импортируют: [tabs.demo_ensemble_methods](#tabs-demo-ensemble-methods), [tests.test_legacy_tabs_migration](#tests-test-legacy-tabs-migration), [tests.test_pixel_morphology](#tests-test-pixel-morphology).

<a id="utils-ops-deep-learning-ops"></a>

## utils_ops.deep_learning_ops

Исходник: [utils_ops/deep_learning_ops.py](../utils_ops/deep_learning_ops.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts), [utils_new.segmentation.operations](#utils-new-segmentation-operations).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [utils_new.segmentation.model_output](#utils-new-segmentation-model-output): `decode_output`

**Внешние/stdlib импорты:** `cv2`, `importlib.util.find_spec`, `json`, `numpy`, `onnxruntime`, `os`, `segmentation_models_pytorch`, `sys`, `time`, `torch`, `torchvision.transforms`, `ultralytics.YOLO`, `ultralytics.utils.ops.scale_masks`.

**Объявления:**

- [`_log(msg)`](../utils_ops/deep_learning_ops.py#L14)
- Класс [`BaseSegmentor`](../utils_ops/deep_learning_ops.py#L31)
  - [`BaseSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L33)
  - [`BaseSegmentor.load(self, **kwargs)`](../utils_ops/deep_learning_ops.py#L39)
  - [`BaseSegmentor.predict(self, image, **kwargs)`](../utils_ops/deep_learning_ops.py#L42)
- Класс [`TorchSegmentor`](../utils_ops/deep_learning_ops.py#L46) (`BaseSegmentor`)
  - [`TorchSegmentor.__init__(self, model_path=None, device='cpu')`](../utils_ops/deep_learning_ops.py#L48)
  - [`TorchSegmentor._get_device(self)`](../utils_ops/deep_learning_ops.py#L53)
  - [`TorchSegmentor._resolve_resize_mode(self)`](../utils_ops/deep_learning_ops.py#L72)
  - [`TorchSegmentor._preprocess(self, image, input_size)`](../utils_ops/deep_learning_ops.py#L107)
  - [`TorchSegmentor._postprocess_mask(self, mask_np, orig_size, pad_info, resize_mode)`](../utils_ops/deep_learning_ops.py#L177)
- Класс [`UNetSegmentor`](../utils_ops/deep_learning_ops.py#L198) (`TorchSegmentor`)
  - [`UNetSegmentor.load(self, weights_path=None, encoder=None, input_size=None)`](../utils_ops/deep_learning_ops.py#L216)
  - [`UNetSegmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L359)
- Класс [`DeepLabV3Segmentor`](../utils_ops/deep_learning_ops.py#L413) (`TorchSegmentor`)
  - [`DeepLabV3Segmentor.load(self, weights_path=None, backbone=None, output_stride=None)`](../utils_ops/deep_learning_ops.py#L436)
  - [`DeepLabV3Segmentor.predict(self, image, threshold=0.5)`](../utils_ops/deep_learning_ops.py#L606)
- Класс [`YOLOSegmentor`](../utils_ops/deep_learning_ops.py#L678) (`BaseSegmentor`)
  - [`YOLOSegmentor.load(self, model_path=None)`](../utils_ops/deep_learning_ops.py#L693)
  - [`YOLOSegmentor._otsu_inside_roi(image, x1, y1, x2, y2)`](../utils_ops/deep_learning_ops.py#L716)
  - [`YOLOSegmentor.predict(self, image, conf=0.25, iou=0.45, imgsz=640, save=False, allow_box_otsu=False)`](../utils_ops/deep_learning_ops.py#L731)
- Класс [`SAMSegmentor`](../utils_ops/deep_learning_ops.py#L869) (`BaseSegmentor`)
  - [`SAMSegmentor.load(self, model_type='vit_h', model_path=None)`](../utils_ops/deep_learning_ops.py#L871)
  - [`SAMSegmentor.predict(self, image, prompt_type='everything')`](../utils_ops/deep_learning_ops.py#L880)
- Класс [`ONNXSegmentor`](../utils_ops/deep_learning_ops.py#L887) (`BaseSegmentor`)
  - [`ONNXSegmentor.load(self, model_path)`](../utils_ops/deep_learning_ops.py#L893)
  - [`ONNXSegmentor.predict(self, image, threshold=None)`](../utils_ops/deep_learning_ops.py#L922)
- [`create_segmentor(model_type, **kwargs)`](../utils_ops/deep_learning_ops.py#L964)
  Явно импортируют: [utils_new.segmentation.operations](#utils-new-segmentation-operations).

<a id="utils-ops-gradient-ops"></a>

## utils_ops.gradient_ops

Исходник: [utils_ops/gradient_ops.py](../utils_ops/gradient_ops.py#L1).

**Прямые потребители:** [tabs.gradient_methods](#tabs-gradient-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `numpy`.

**Объявления:**

- [`apply_gradient_method(gray, method, params)`](../utils_ops/gradient_ops.py#L5)
  Явно импортируют: [tabs.gradient_methods](#tabs-gradient-methods), [tests.test_gradient_autoresearch](#tests-test-gradient-autoresearch), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_processing_smoke](#tests-test-processing-smoke), [utils_new.autoreseach_utils.gradient_processing](#utils-new-autoreseach-utils-gradient-processing).

<a id="utils-ops-interactive-ops"></a>

## utils_ops.interactive_ops

Исходник: [utils_ops/interactive_ops.py](../utils_ops/interactive_ops.py#L1).

**Прямые потребители:** [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_segmentation_tasks](#tests-test-segmentation-tasks), [utils_new.segmentation.operations](#utils-new-segmentation-operations).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `importlib.util.find_spec`, `networkx`, `numpy`, `skimage.feature.peak_local_max`, `skimage.filters.sobel`, `skimage.segmentation.active_contour`, `skimage.segmentation.morphological_chan_vese`, `skimage.segmentation.random_walker`, `skimage.segmentation.slic`, `skimage.segmentation.watershed`, `sklearn.mixture.GaussianMixture`.

**Объявления:**

- [`_image(image)`](../utils_ops/interactive_ops.py#L12)
- [`_rect(rect, shape)`](../utils_ops/interactive_ops.py#L19)
- [`_seeds(fg, bg, shape, required=True)`](../utils_ops/interactive_ops.py#L31)
- [`_positive(name, value, zero=False)`](../utils_ops/interactive_ops.py#L48)
- [`grabcut_segmentation(img, rect=None, mask=None, fg_scribbles=None, bg_scribbles=None, iterations=5, mode='Both')`](../utils_ops/interactive_ops.py#L55) — Initialize once: rectangle, seeds, or rectangle with hard seed constraints.
- [`watershed_segmentation(gray, distance_threshold=50, min_distance=20, fg_scribbles=None, bg_scribbles=None, mode='automatic')`](../utils_ops/interactive_ops.py#L110) — Marker watershed; automatic mode assumes dark objects on a light background.
- [`random_walker_segmentation(img, markers, beta=130, mode='cg_j', use_color=True)`](../utils_ops/interactive_ops.py#L153)
- [`active_contour_segmentation(gray, rect, alpha=0.01, beta=0.1, gamma=0.01, max_iter=250, convergence=0.001, smooth=True, init_type='Ellipse')`](../utils_ops/interactive_ops.py#L192)
- [`morphological_chan_vese_segmentation(gray, rect, iterations=100, smoothing=2, lambda_inside=1.0, lambda_outside=1.0)`](../utils_ops/interactive_ops.py#L249) — Region-based evolution, useful when an object has a weak edge.
- [`_superpixels(img, superpixel_size, compactness, sigma)`](../utils_ops/interactive_ops.py#L276)
- [`_appearance_costs(colors, fg_pixels, bg_pixels)`](../utils_ops/interactive_ops.py#L314)
- [`_cut(segments, colors, pairs, costs, color_sigma, spatial_weight=1.0, fg=None, bg=None)`](../utils_ops/interactive_ops.py#L336)
- [`supercut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, lambda_val=0.5, sigma_color=10.0)`](../utils_ops/interactive_ops.py#L374) — Project variant: ROI-occupancy prior and color-sensitive SLIC graph.
- [`onecut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, spatial_weight=1.0, data_weight=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L397) — Project variant: GMM learned from ROI/exterior and a SLIC graph cut.
- [`lazy_snapping(img, fg_scribbles, bg_scribbles, superpixel_size=50, compactness=10, sigma=1.0, color_sigma=10.0)`](../utils_ops/interactive_ops.py#L417) — Project variant: seed-trained GMM unary costs with hard superpixel seeds.

<a id="utils-ops-segmentation-ops"></a>

## utils_ops.segmentation_ops

Исходник: [utils_ops/segmentation_ops.py](../utils_ops/segmentation_ops.py#L1).

**Прямые потребители:** [tabs.layout_dataset](#tabs-layout-dataset), [tests.test_method_algorithms](#tests-test-method-algorithms), [tests.test_processing_smoke](#tests-test-processing-smoke), [tests.test_segmentation_shared](#tests-test-segmentation-shared), [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui).

**Прямые локальные импорты:**
- [ui.settings_service](#ui-settings-service): `settings`
- [utils_new.threshold_processing](#utils-new-threshold-processing): `apply_threshold_method as apply_threshold_method`, `background_symmetry_threshold as background_symmetry_threshold`, `isodata_threshold as isodata_threshold`, `niblack_threshold as niblack_threshold`, `normalize_method_name as normalize_method_name`, `otsu_threshold as otsu_threshold`, `row_adaptive_threshold as row_adaptive_threshold`, `sauvola_threshold as sauvola_threshold`, `simple_threshold as simple_threshold`, `triangle_threshold as triangle_threshold`

**Внешние/stdlib импорты:** `PyQt5.QtCore.Qt`, `PyQt5.QtWidgets.QListWidget`, `PyQt5.QtWidgets.QListWidgetItem`, `cv2`, `numpy`.

**Объявления:**

- [`get_display_params(img_shape)`](../utils_ops/segmentation_ops.py#L23)
- [`draw_label(img_color, label, x, y, box_w, box_h, font_scale, font_thickness, thickness, color=None)`](../utils_ops/segmentation_ops.py#L37)
- [`draw_label_rotated(img_color, label, center, font_scale, font_thickness, color=None)`](../utils_ops/segmentation_ops.py#L60)
- [`get_class_color(cls_id)`](../utils_ops/segmentation_ops.py#L97) — Цвет (BGR) для класса. Для id >= 10 цвета циклически повторяются.
  Явно импортируют: [ui_tabs.layout_dataset_ui](#ui-tabs-layout-dataset-ui).
- [`draw_class_label(img, text, x, y, box_w, box_h, font_scale, font_thickness, bg_color, text_color=(255, 255, 255), padding=2, margin=5)`](../utils_ops/segmentation_ops.py#L108) — Рисует текстовую подпись поверх изображения на цветной плашке.
- [`segment_contours(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L165) — Возвращает (img_color, objects) где objects = список ('segment', 0, points) в нормализованных координатах.
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared).
- [`segment_projections(img, binary, draw=True)`](../utils_ops/segmentation_ops.py#L205) — Возвращает (img_color, objects) где objects = список ('detect', 0, cx, cy, w, h)
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared).
- [`segment_min_area_rect(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L243) — Возвращает (img_color, objects) где objects = список ('obb', 0, points) где points = 8 нормализованных координат углов.
  Явно импортируют: [tests.test_segmentation_shared](#tests-test-segmentation-shared).
- [`segment_polygons(img, binary, use_hull, draw=True)`](../utils_ops/segmentation_ops.py#L283) — Алиас для segment_contours (уже возвращает segment).
- [`draw_yolo_annotations(img, annotations, color=(0, 255, 0))`](../utils_ops/segmentation_ops.py#L289) — Рисует аннотации любого типа (detect, obb, segment) на изображении.
- [`format_annotation_display(index, ann, img_w, img_h)`](../utils_ops/segmentation_ops.py#L340) — Форматирует аннотацию для отображения в QListWidget.
- [`update_annotation_list(list_widget: QListWidget, annotations, img_w: int, img_h: int)`](../utils_ops/segmentation_ops.py#L370) — Очищает и заполняет QListWidget строками аннотаций. Каждому элементу в UserRole записывается индекс.
  Явно импортируют: [tabs.layout_dataset](#tabs-layout-dataset).
- [`delete_annotation_by_index(current_annotations, idx, all_annotations=None, current_index=None)`](../utils_ops/segmentation_ops.py#L380) — Удаляет аннотацию с индексом idx. Возвращает (new_annotations, new_selected_index).
- [`draw_selected_objects(img, objects, selected_indices, draw_mode, use_hull, color_rect, color_label, thickness, font_scale, font_thickness)`](../utils_ops/segmentation_ops.py#L393) — Рисует выбранные объекты на изображении. Поддерживает ('detect',...), ('obb',...), ('segment',...).
- [`format_object_for_list(index, obj, img_w, img_h)`](../utils_ops/segmentation_ops.py#L442) — Универсальное форматирование для старого стиля (используется в threshold_methods.py).
- [`get_morph_kernel(size, shape_name)`](../utils_ops/segmentation_ops.py#L463)
- [`apply_morphology(binary, close_factor, open_factor, kernel_shape, img_shape=None)`](../utils_ops/segmentation_ops.py#L473)

<a id="utils-ops-traditional-ml-ops"></a>

## utils_ops.traditional_ml_ops

Исходник: [utils_ops/traditional_ml_ops.py](../utils_ops/traditional_ml_ops.py#L1).

**Прямые потребители:** [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tests.test_method_algorithms](#tests-test-method-algorithms), [utils_new.segmentation.operations](#utils-new-segmentation-operations).

**Прямые локальные импорты:**
- нет

**Внешние/stdlib импорты:** `cv2`, `joblib`, `numpy`, `skimage.feature`, `skimage.segmentation`, `sklearn.cluster.KMeans`, `sklearn.cluster.MeanShift`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.svm.SVC`, `sklearn.tree.DecisionTreeClassifier`, `warnings`, `xgboost`.

**Объявления:**

- [`extract_pixel_features(image, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L20)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`extract_superpixel_features(image, segments, use_intensity=True, use_texture=False, use_spatial=False)`](../utils_ops/traditional_ml_ops.py#L47)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods).
- [`train_kmeans(X, n_clusters=3)`](../utils_ops/traditional_ml_ops.py#L85)
- [`train_meanshift(X)`](../utils_ops/traditional_ml_ops.py#L92)
- [`train_svm(X, y, kernel='rbf', C=10)`](../utils_ops/traditional_ml_ops.py#L97)
- [`train_random_forest(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L102)
- [`train_xgboost(X, y, n_estimators=100)`](../utils_ops/traditional_ml_ops.py#L107)
- [`train_decision_tree(X, y)`](../utils_ops/traditional_ml_ops.py#L114)
- [`predict_pixelwise(model, X, original_shape)`](../utils_ops/traditional_ml_ops.py#L119)
- [`predict_superpixel(model, segments, X_superpixel)`](../utils_ops/traditional_ml_ops.py#L123)
- [`create_superpixels(image, superpixel_size=50)`](../utils_ops/traditional_ml_ops.py#L130)
  Явно импортируют: [tabs.traditional_ml_methods](#tabs-traditional-ml-methods), [tests.test_method_algorithms](#tests-test-method-algorithms).
- [`compute_elbow_kmeans(X, max_k=10)`](../utils_ops/traditional_ml_ops.py#L140) — Вычисляет инерцию (сумму квадратов расстояний) для K от 1 до max_k.
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms).
- [`find_optimal_k_elbow(inertias)`](../utils_ops/traditional_ml_ops.py#L155) — Находит оптимальное K как точку максимального изгиба (метод локтя).
  Явно импортируют: [tests.test_method_algorithms](#tests-test-method-algorithms).

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

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts).

**Прямые локальные импорты:**
- [utils_new.segmentation.model_export](#utils-new-segmentation-model-export): `stride_padded_model`
- [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics): `confusion_matrix`, `prediction_labels`, `segmentation_metrics`
- [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts): `validate_mask_files`

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `copy`, `csv`, `datetime.datetime`, `glob.glob`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `onnx`, `onnxruntime`, `os`, `pathlib.Path`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `traceback`, `yaml`.

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
  - [`VOCSegmentationPairs.__len__(self)`](../utils_training/train_DeepLabV3_worker.py#L404)
  - [`VOCSegmentationPairs.__getitem__(self, idx)`](../utils_training/train_DeepLabV3_worker.py#L407)
- [`get_loss_function(loss_name, num_classes, ignore_index=None, class_weights=None, loss_weights=None, device='cpu')`](../utils_training/train_DeepLabV3_worker.py#L475) — Патч 1: weights_tensor создаётся на нужном устройстве (device).
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts).
- [`get_optimizer(model, opt_name, lr, momentum=0.9, weight_decay=0.0001)`](../utils_training/train_DeepLabV3_worker.py#L530)
- [`get_layerwise_optimizer(model, opt_name, base_lr, momentum=0.9, weight_decay=0.0001, decay_factor=0.1)`](../utils_training/train_DeepLabV3_worker.py#L543) — Layer-wise LR. Устойчив к тому, что get_stages() может возвращать
- [`get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-06)`](../utils_training/train_DeepLabV3_worker.py#L602)
- Класс [`PolyLRSchedulerIter`](../utils_training/train_DeepLabV3_worker.py#L615)
  - [`PolyLRSchedulerIter.__init__(self, optimizer, max_iter, power=0.9, min_lr=0.0)`](../utils_training/train_DeepLabV3_worker.py#L616)
  - [`PolyLRSchedulerIter.step(self)`](../utils_training/train_DeepLabV3_worker.py#L624)
  - [`PolyLRSchedulerIter.get_last_lr(self)`](../utils_training/train_DeepLabV3_worker.py#L630)
- Класс [`ModelEMA`](../utils_training/train_DeepLabV3_worker.py#L634)
  - [`ModelEMA.__init__(self, model, decay=0.999, device=None)`](../utils_training/train_DeepLabV3_worker.py#L635)
  - [`ModelEMA.update(self, model)`](../utils_training/train_DeepLabV3_worker.py#L646)
  - [`ModelEMA.state_dict(self)`](../utils_training/train_DeepLabV3_worker.py#L657)
- [`set_bn_eval(module)`](../utils_training/train_DeepLabV3_worker.py#L661)
- [`_pad_to_multiple(x, multiple=16)`](../utils_training/train_DeepLabV3_worker.py#L667)
- [`_crop_to_hw(x, hw)`](../utils_training/train_DeepLabV3_worker.py#L677)
- [`_forward_safe(model, x, use_amp, device, output_stride=None)`](../utils_training/train_DeepLabV3_worker.py#L682) — Патч 3: убран nan_to_num(nan=0.0) — он маскировал проблему,
- [`predict_batch_multiscale(model, images, device, scales, use_amp)`](../utils_training/train_DeepLabV3_worker.py#L712)
- [`compute_metrics(pred_logits, true_masks, num_classes, ignore_index=None, output_activation='argmax', threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L737)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_DeepLabV3_worker.py#L746)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L775)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_DeepLabV3_worker.py#L787)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_DeepLabV3_worker.py#L794)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_DeepLabV3_worker.py#L815)
- [`save_pr_curve(preds, masks, save_dir, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L823)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes, ignore_index=None, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L842)
- [`save_train_batch_examples(loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_DeepLabV3_worker.py#L863)
- [`save_val_examples(loader, model, device, save_dir, num_samples=8, num_classes=1, output_activation='argmax', normalize=False, use_amp=False, multi_scale=False, scales=None, threshold=0.5)`](../utils_training/train_DeepLabV3_worker.py#L895)
- [`main()`](../utils_training/train_DeepLabV3_worker.py#L948)

<a id="utils-training-train-unet-worker"></a>

## utils_training.train_unet_worker

Исходник: [utils_training/train_unet_worker.py](../utils_training/train_unet_worker.py#L1).

**Прямые потребители:** [tests.test_model_contracts](#tests-test-model-contracts).

**Прямые локальные импорты:**
- [utils_new.segmentation.model_export](#utils-new-segmentation-model-export): `stride_padded_model`
- [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics): `confusion_matrix`, `prediction_labels`, `segmentation_metrics`
- [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts): `apply_decoder_dropout`, `validate_mask_files`

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `onnx`, `onnxruntime`, `os`, `pathlib.Path`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `traceback`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker.py#L134)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker.py#L139)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker.py#L161) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker.py#L162)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker.py#L206)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker.py#L209)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker.py#L265)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker.py#L332)
  Явно импортируют: [tests.test_model_contracts](#tests-test-model-contracts).
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker.py#L370)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker.py#L385)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L398)
- [`auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker.py#L404)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker.py#L447)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker.py#L459)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker.py#L467)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker.py#L501)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker.py#L510)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker.py#L569)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, normalize=False)`](../utils_training/train_unet_worker.py#L615)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1, normalize=False, threshold=0.5)`](../utils_training/train_unet_worker.py#L658)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker.py#L727)
- [`main()`](../utils_training/train_unet_worker.py#L777)

<a id="utils-training-train-unet-worker-рабочий"></a>

## utils_training.train_unet_worker_рабочий

Исходник: [utils_training/train_unet_worker_рабочий.py](../utils_training/train_unet_worker_рабочий.py#L1).

**Прямые потребители:** не найдены в import-графе; это не доказательство неиспользования.

**Прямые локальные импорты:**
- [utils_new.segmentation.model_metrics](#utils-new-segmentation-model-metrics): `confusion_matrix`, `prediction_labels`, `segmentation_metrics`
- [utils_new.segmentation.training_contracts](#utils-new-segmentation-training-contracts): `apply_decoder_dropout`, `validate_mask_files`

**Внешние/stdlib импорты:** `PIL.Image`, `argparse`, `csv`, `datetime.datetime`, `io`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib.Path`, `psutil`, `seaborn`, `segmentation_models_pytorch`, `signal`, `sys`, `torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`, `torch.utils.data.Dataset`, `torchvision.transforms`, `torchvision.transforms.functional`, `yaml`.

**Объявления:**

- [`log(msg)`](../utils_training/train_unet_worker_рабочий.py#L129)
- [`signal_handler(signum, frame)`](../utils_training/train_unet_worker_рабочий.py#L133)
- Класс [`SegmentationDataset`](../utils_training/train_unet_worker_рабочий.py#L154) (`Dataset`)
  - [`SegmentationDataset.__init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None, normalize=False, resize_mode='hard', is_train=True)`](../utils_training/train_unet_worker_рабочий.py#L155)
  - [`SegmentationDataset.__len__(self)`](../utils_training/train_unet_worker_рабочий.py#L199)
  - [`SegmentationDataset._resize_pair(self, image, mask)`](../utils_training/train_unet_worker_рабочий.py#L202)
  - [`SegmentationDataset.__getitem__(self, idx)`](../utils_training/train_unet_worker_рабочий.py#L267)
- [`get_loss_function(loss_name, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L337)
- [`get_optimizer(model, opt_name, lr)`](../utils_training/train_unet_worker_рабочий.py#L375)
- [`get_scheduler(optimizer, scheduler_name, epochs)`](../utils_training/train_unet_worker_рабочий.py#L390)
- [`compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L403)
- [`auto_batch_size(device, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX)`](../utils_training/train_unet_worker_рабочий.py#L409)
- [`save_args_yaml(args_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L433)
- [`save_model_meta(meta_dict, model_path)`](../utils_training/train_unet_worker_рабочий.py#L444)
- [`save_training_plots(history_dict, save_dir)`](../utils_training/train_unet_worker_рабочий.py#L452)
- [`save_csv_results(history_dict, save_path)`](../utils_training/train_unet_worker_рабочий.py#L479)
- [`save_pr_curve(preds, masks, save_dir, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L487)
- [`save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L530)
- [`save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1)`](../utils_training/train_unet_worker_рабочий.py#L571)
- [`save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1, threshold=0.5)`](../utils_training/train_unet_worker_рабочий.py#L602)
- [`check_dataset_masks_consistency(dataset, num_classes, max_check=20)`](../utils_training/train_unet_worker_рабочий.py#L658) — Проверяет первые max_check масок: собирает уникальные значения пикселей,
- [`main()`](../utils_training/train_unet_worker_рабочий.py#L713)

<a id="utils-training-train-yolo-worker"></a>

## utils_training.train_yolo_worker

Исходник: [utils_training/train_yolo_worker.py](../utils_training/train_yolo_worker.py#L1).

**Прямые потребители:** [tests.test_yolo_worker](#tests-test-yolo-worker).

**Прямые локальные импорты:**
- [runtime_policy](#runtime-policy): `configure_ultralytics`
- [utils_new.segmentation.yolo_metrics](#utils-new-segmentation-yolo-metrics): `validation_metrics`

**Внешние/stdlib импорты:** `argparse`, `datetime.datetime`, `json`, `math`, `onnx`, `pathlib.Path`, `re`, `shutil`, `signal`, `sys`, `traceback`, `ultralytics`, `ultralytics.YOLO`.

**Объявления:**

- [`log(message)`](../utils_training/train_yolo_worker.py#L26)
- [`is_standard_yolo_model(model_path)`](../utils_training/train_yolo_worker.py#L30) — Allow downloads only for a known bare model name, never a missing user path.
- [`signal_handler(signum, frame)`](../utils_training/train_yolo_worker.py#L41)
- [`save_model_meta(metadata, model_path)`](../utils_training/train_yolo_worker.py#L50)
- [`build_train_args(params)`](../utils_training/train_yolo_worker.py#L60) — Validate before loading CUDA. Do not estimate or replace the user's batch.
- [`resolved_settings(trainer, args)`](../utils_training/train_yolo_worker.py#L152)
- [`export_onnx(model, metadata, imgsz, run_dir=None)`](../utils_training/train_yolo_worker.py#L168) — Optional export. Its failure must not hide successful training/validation.
- [`main(argv=None)`](../utils_training/train_yolo_worker.py#L208)

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
