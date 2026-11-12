# Итоговый аудит импортов и зависимостей

[План](Import_Architecture_Plan.md) · [Архитектура](Architecture.md) · [Установка](Dependencies.md)

Исторический отчёт миграции импортов. Текущий перенос и приёмка:
[унификация пяти вкладок](Segmentation_Unification_Status.md), [удаление утилит](Utils_Removal_Audit.md).

При синхронизации с master проверено 137 Python-файлов: 131 прикладной модуль, пять файлов тестов и генератор документации. Все прикладные модули импортированы первыми в отдельных процессах. Переименование файлов, обучение и загрузка весов при аудите не запускались.

## Результаты

- Все 21 тест прежней полной suite и четыре регрессии слияния (25 суммарно) прошли в Python 3.12 / Linux / Torch CPU.
- AST/компиляция и Ruff F821/F822/F823/F403/F405 прошли для всех Python-файлов.
- В eager-графе локальных импортов нет циклов; wildcard и внутреннее использование import_libs_* отсутствуют.
- Все 15 вкладок импортируются по одной и в обоих порядках без main; 30 запусков direct/module прошли.
- Все 16 пунктов ALL_TABS (15 вкладок и настройки) одновременно созданы, показаны и закрыты с общим Qt lifecycle.
- Четыре training workers и yolo2voc выдают --help без обучения; независимые импорты utils_training не загружают PyQt5.
- Геометрия, YOLO TXT, индексные маски, YAML, COCO, пять пороговых и два градиентных метода, Albumentations и Torch CPU прошли smoke-проверки.
- Повторный импорт cv2/torch сохраняет идентичность модулей; Ultralytics использует тот же cv2.
- AutoInstall Ultralytics выключен; проверка отсутствующего backend не вызывает pip/subprocess.
- Полный install.py --torch cpu прошёл в рабочем venv. Единственный поставщик cv2: opencv-python-headless 4.13.0.92.
- dependency_health проверяет прямые pins и pip check. Требование Ultralytics к opencv-python выделено как явное исключение метаданных; прочие ошибки не допускаются.

## Воспроизведение

```bash
python install.py --torch cpu
python dependency_health.py --load
python -m unittest discover -s tests -v
NEURO_RUN_GUI_TESTS=1 python -m unittest discover -s tests -v
python Docs/generate_reference.py
```

## Покрытие по файлам

Файлы, которые уже имели правильные явные импорты, сохранены без искусственной переработки. Старые совместимые фасады и алгоритмы не удалены. В таблице отражено, как проверен каждый файл.

| Python-файл | Проверка |
| --- | --- |
| [Docs/generate_reference.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/Docs/generate_reference.py) | AST и выполнение генератора справочников |
| [cluster/canonicalize_crops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/canonicalize_crops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [cluster/canonicalize_crops_method2.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/canonicalize_crops_method2.py) | Независимый импорт — OK; AST/компиляция — OK |
| [cluster/cluster_embeddings.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/cluster_embeddings.py) | Независимый импорт — OK; AST/компиляция — OK |
| [cluster/extract_embeddings.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/extract_embeddings.py) | Независимый импорт — OK; AST/компиляция — OK |
| [cluster/save_yolo_object_to_path.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/save_yolo_object_to_path.py) | Независимый импорт — OK; AST/компиляция — OK |
| [cluster/test_new_image.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/cluster/test_new_image.py) | Независимый импорт — OK; AST/компиляция — OK |
| [dependency_health.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/dependency_health.py) | Независимый импорт — OK; AST/компиляция — OK |
| [generate.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/generate.py) | Независимый импорт — OK; AST/компиляция — OK |
| [import_libs_external.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/import_libs_external.py) | Независимый импорт — OK; AST/компиляция — OK |
| [import_libs_internal.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/import_libs_internal.py) | Независимый импорт — OK; AST/компиляция — OK |
| [import_libs_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/import_libs_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [install.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/install.py) | Независимый импорт — OK; AST/компиляция — OK |
| [main.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/main.py) | Независимый импорт — OK; AST/компиляция — OK |
| [main_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/main_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [path_setup.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/path_setup.py) | Независимый импорт — OK; AST/компиляция — OK |
| [rename_data.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/rename_data.py) | Независимый импорт — OK; AST/компиляция — OK |
| [runtime_policy.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/runtime_policy.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/dataset_analysis.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/dataset_analysis.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/deep_learning_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/deep_learning_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/deep_learning_training.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/deep_learning_training.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/demo_ensemble_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/demo_ensemble_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/gradient_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/gradient_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/interactive_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/interactive_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/layout_dataset.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/layout_dataset.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/model_research.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/model_research.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/preparing_dataset_yaml.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/preparing_dataset_yaml.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/slice_dataset_sahi.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/slice_dataset_sahi.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/threshold_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/threshold_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/traditional_ml_methods.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/traditional_ml_methods.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/viewing_dataset.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/viewing_dataset.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/yolo_find_img.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/yolo_find_img.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tabs/yolo_test_aeb.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tabs/yolo_test_aeb.py) | Независимый импорт — OK; AST/компиляция — OK |
| [tests/test_dependency_installation.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_dependency_installation.py) | Импорт unittest и выполнение полной suite |
| [tests/test_import_architecture.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_import_architecture.py) | Импорт unittest и выполнение полной suite |
| [tests/test_processing_smoke.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_processing_smoke.py) | Импорт unittest и выполнение полной suite |
| [tests/test_label_save_merge.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_label_save_merge.py) | Четыре проверки сохранения после слияния master |
| [tests/test_runtime_imports.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_runtime_imports.py) | Импорт unittest и выполнение полной suite |
| [ui/DL_predict_settings.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/DL_predict_settings.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/DL_train_settings_base.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/DL_train_settings_base.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/DL_train_settings_new_model.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/DL_train_settings_new_model.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/application.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/application.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/control_buttons.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/control_buttons.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/image_navigation.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/image_navigation.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/interactive_params_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/interactive_params_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/log_widget.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/log_widget.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/morphology_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/morphology_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/pair_validation_dialog_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/pair_validation_dialog_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui/settings_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui/settings_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/dataset_analysis_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/dataset_analysis_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/deep_learning_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/deep_learning_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/deep_learning_training_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/deep_learning_training_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/demo_ensemble_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/demo_ensemble_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/gradient_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/gradient_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/interactive_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/interactive_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/layout_dataset_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/layout_dataset_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/model_research_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/model_research_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/preparing_dataset_yaml_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/preparing_dataset_yaml_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/slice_dataset_sahi_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/slice_dataset_sahi_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/threshold_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/threshold_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/traditional_ml_methods_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/traditional_ml_methods_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/viewing_dataset_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/viewing_dataset_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/yolo_find_img_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/yolo_find_img_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [ui_tabs/yolo_test_aeb_ui.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/ui_tabs/yolo_test_aeb_ui.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/DatasetGenerator.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/DatasetGenerator.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/analysis.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/analysis.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/annotation_converter.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/annotation_converter.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/aug_utils.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/aug_utils.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/augmentation_base.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/augmentation_base.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/augmentation_random.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/augmentation_random.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/coco_export.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/coco_export.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/coco_io.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/coco_io.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/dataset_analysis_logic.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/dataset_analysis_logic.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/dataset_audit_logic.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/dataset_audit_logic.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/dataset_validation.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/dataset_validation.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/ensemble_utils.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/ensemble_utils.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/image_io.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/image_io.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/metrics_core.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/metrics_core.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/prepare.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/prepare.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/presets.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/presets.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/settings.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/settings.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/smart_view.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/smart_view.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/unet_export.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/unet_export.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/unet_io.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/unet_io.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils/yaml_io.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils/yaml_io.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/auto-aeb_def.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/auto-aeb_def.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/auto-aeb_def_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/auto-aeb_def_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/auto-aeb_old.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/auto-aeb_old.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/auto-aeb_unet.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/auto-aeb_unet.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/auto-aeb_unet_with_grad.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/auto-aeb_unet_with_grad.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_aeb/yolo_aeb_utils.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_aeb/yolo_aeb_utils.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_auto_research/gradient_auto_research.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_auto_research/gradient_auto_research.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_auto_research/gradient_auto_settings_dialog.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_auto_research/gradient_auto_settings_dialog.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_auto_research/threshold_auto_research.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_auto_research/threshold_auto_research.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_auto_research/threshold_auto_settings_dialog.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_auto_research/threshold_auto_settings_dialog.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/DatasetGenerator_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/DatasetGenerator_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotation_processing.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotation_processing.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/__init__.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/__init__.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/adapters.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/adapters.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/augmentation.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/augmentation.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/geometry.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/geometry.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/masks.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/masks.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/model.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/model.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/rendering.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/rendering.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/annotations/transforms.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/annotations/transforms.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_catalog_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_catalog_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_coco_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_coco_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_files_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_files_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_image_stats_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_image_stats_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_io_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_io_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_loader_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_loader_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_mask_diagnostics_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_mask_diagnostics_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_mask_preview_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_mask_preview_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_output_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_output_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_pair_tools_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_pair_tools_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_pair_validator_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_pair_validator_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_preparation_bridge.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_preparation_bridge.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_review_view.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_review_view.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_unet_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_unet_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_validation_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_validation_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_view_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_view_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_yaml_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_yaml_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/dataset_yolo_labels_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/dataset_yolo_labels_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/morphology.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/morphology.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/smart_view_new.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/smart_view_new.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_new/threshold_processing.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_new/threshold_processing.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_ops/deep_learning_ops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_ops/deep_learning_ops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_ops/gradient_ops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_ops/gradient_ops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_ops/interactive_ops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_ops/interactive_ops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_ops/segmentation_ops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_ops/segmentation_ops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_ops/traditional_ml_ops.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_ops/traditional_ml_ops.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/test_train_worker.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/test_train_worker.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/train_DeepLabV3_worker.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/train_DeepLabV3_worker.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/train_unet_worker.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/train_unet_worker.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/train_unet_worker_рабочий.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/train_unet_worker_рабочий.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/train_yolo_worker.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/train_yolo_worker.py) | Независимый импорт — OK; AST/компиляция — OK |
| [utils_training/yolo2voc.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/utils_training/yolo2voc.py) | Независимый импорт — OK; AST/компиляция — OK |

## Границы проверки

CUDA/MPS, полное обучение, инференс с реальными весами, качество моделей и все необязательные backend не проверены. Возможности SAM/SegFormer, ранее являвшиеся заглушками, остаются такими. Qt проверен через offscreen; это не проверка каждого действия мышью и кнопки. Обычный pip check не является чистым из-за описанного исключения Ultralytics; оно не скрывается диагностикой. Внешний пользователь старого from import_libs_* import * по-прежнему загружает все экспорты; приложение использует только явные импорты владельцев.

Повторная проверка OpenCV 11 ноября 2026 года: выбран headless, проверены
встречные требования Ultralytics и Albumentations/albucore, добавлены две
регрессии ремонта единственного wheel (повреждённые файлы/неверная версия).
Сравнение и источники: [OpenCV_Decision.md](OpenCV_Decision.md).

Синхронизация с master `5043de7`: конфликты явных импортов разрешены без
возврата агрегаторов. Проверены канонические YAML-пути и вложенность, коллизии
TXT, подсчёт перезаписей и предупреждение U-Net; обе вкладки проверены при
отказе/согласии и Save All с непрочитанной записью. Новые проверки находятся
в [tests/test_label_save_merge.py](https://github.com/VorotilovRoman/Neuro_threshold/blob/89c9c11/tests/test_label_save_merge.py).

Исправление Windows DLL: добавлены runtime_torch.py и tests/test_torch_runtime.py
(всего 139 Python-файлов, 132 прикладных модуля и шесть файлов тестов). Новые
пять проверок покрывают имитацию DLL-сбоя, кеширование успеха/ошибки, CUDA-probe,
Windows-only подготовку, обычный анализ и возврат ошибки model worker.
Это не проверка настоящих Windows DLL; см. [Windows_Torch_DLL](Windows_Torch_DLL.md).

После исправления загрузчика Torch полная suite с NEURO_RUN_GUI_TESTS=1
прошла: 30 тестов, Linux CPU, включая имитацию WinError 1114. Компиляция всех
Python-файлов и Ruff F821/F822/F823/F403/F405 также прошли.

Профиль cu126: в официальном индексе подтверждены Windows/Python 3.12 wheels
Torch 2.11.0+cu126 и torchvision 0.26.0+cu126. Добавлены профиль установщика
и тест передачи параметра --torch cu126. Корневые переносимые pins сохранены.
Дополнительная GUI-проверка выполняет нажатия Load Images и скрытия рамок
в редакторе и просмотре, встроенных в ImagePlayer. Остальные неуточнённые
сообщения пользователя о кнопках пока не воспроизведены.

Полная suite после добавления cu126 и проверки встроенных кнопок прошла:
32 теста, Linux CPU; CUDA-wheel проверен по официальному индексу, без его
установки вместо CPU-сборки в среде разработки.

Исправление недоступных кнопок методов: воспроизведено перекрытие навигации
пустым DeepLearningWindow (геометрия 640×480), созданным прежней схемой
takeCentralWidget/setParent. Попадание мыши возвращало пустое окно вместо
Load Images. Теперь пять окон методов встроены целиком с Qt.Widget; новая
регрессия проверяет реальную маршрутизацию клика через QWindow, владельца
диалога, загрузку снимка и реакцию порогового результата на ползунок.
Прежние проверки прямой отправки события кнопке не обнаруживали перекрытие.

После исправления перекрытия полный запуск NEURO_RUN_GUI_TESTS=1 прошёл:
33 теста, включая совместное создание/закрытие всех вкладок и автономные
точки входа. Проверки выполнены в Linux offscreen; реальный Windows после
этого изменения ещё должен быть проверен пользователем.
