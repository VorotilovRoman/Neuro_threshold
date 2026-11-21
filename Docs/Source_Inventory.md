# Источники данных: воспроизводимый аудит

[Как пользоваться](Source_Unification_Status.md) · [План](Source_Unification_Plan.md)

Сгенерировано `python Docs/audit_sources.py > Docs/Source_Inventory.md`. Скрипт читает AST; не импортирует Qt/Torch и не открывает диалоги. Перечень вызовов включает скрытые формы и условные ветки; наличие вызова не доказывает доступность backend. Фактические сценарии и ограничения — в описании результата.

## Назначение всех вкладок

| ID | Вкладка | Описание из общего реестра UI |
| --- | --- | --- |
| `threshold` | Пороговые методы | Пороговая сегментация снимков и папок, пиксельная морфология, BBox/OBB/Segment и автоподбор по эталону. |
| `gradient` | Градиентные методы | Градиентная сегментация снимков и папок, морфология, разметка объектов и автоподбор. |
| `interactive` | Интерактивные методы | Интерактивная сегментация снимков: ROI, штрихи переднего/заднего плана и выбранный алгоритм. |
| `traditional_ml` | Традиционные методы ML | Классический ML: обучение на загруженных снимках по соседним YOLO TXT либо явным псевдометкам Otsu; кластеризация без меток; импорт модели. |
| `deep_learning` | Deep Learning | Применение моделей сегментации к снимкам и папкам. Загрузите совместимые веса; обучение находится в отдельной вкладке. |
| `reports` | Анализ отчётов | Сравнение пороговых/градиентных отчётов с правильной ручной разметкой: YOLO YAML или папка TXT; рейтинг методов и пресеты. |
| `labeler` | Добавить аннотации | Создание и исправление BBox/OBB/Segment: снимки, TXT, YOLO YAML, COCO, импорт масочного YAML в полигоны. |
| `test` | Просмотр аннотаций | Просмотр и исправление аннотаций: снимки, TXT, YOLO YAML, COCO, индексные маски и их наложение; поиск перекрывающихся объектов. |
| `dataset` | Подготовка датасета | Подготовка и экспорт датасета: снимки + TXT, индексные маски или снимки без разметки; YOLO YAML, COCO и YAML с масками. |
| `dataset_analysis` | Анализ датасета | Проверка качества, разметки, повторов и split: TXT/индексные маски/снимки без разметки; общий отчёт и экспорт. |
| `sahi` | SAHI Нарезка датасета | Тайловая нарезка датасета из снимков с TXT, YOLO YAML или COCO. Индексные маски как вход здесь не поддерживаются. |
| `yolo_train` | Обучение модели | Обучение YOLO, U-Net и DeepLab: папка либо YAML соответствующего формата. SAM/SegFormer остаются заготовками интерфейса. |
| `yolo_demo` | Демонстрация результата | Сравнение YOLO и пороговых пресетов и объединение BBox по классам через NMS/WBF; снимки и папки. |
| `yolo_sort` | Сортировка с YOLO | Рекурсивный поиск снимков по классу YOLO и геометрии объекта; список, миниатюры и перенос только изображений. |
| `yolo_aeb` | YOLO + AE/VAE | YOLO и AE/VAE: снимки/папки, выбор объекта, кропы, преобразования и карта ошибки реконструкции по checkpoint. |
| `settings` | Настройки | Глобальные настройки интерфейса, цветов, графики и навигации. Источников датасета на этой вкладке нет. |

## Общие действия

| Ключ | Кнопка | Источник / фильтр | Подсказка |
| --- | --- | --- | --- |
| `images` | Снимки… | Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp) | Выбрать один или несколько снимков. Текущий набор заменяется. |
| `folder` | Папка снимков… | Папка | Загрузить снимки из выбранной папки. Вложенные папки не включаются, если вкладка явно не указывает рекурсивный поиск. |
| `dataset_root` | Папка датасета… | Папка | Выбрать корень датасета; ожидаемая структура зависит от выбранной модели или операции. |
| `reference_folder` | Папка эталона… | Папка | Изображения с TXT либо папка только с TXT. Для варианта без снимков нужны размеры из отчётов. |
| `labels` | Папка разметки TXT… | Папка | Одноимённые TXT: BBox, OBB или Segment. Это папка меток, а не снимков. |
| `masks` | Папка масок… | Папка | Индексные маски: целые значения пикселей обозначают классы; RGB-картинка не равна индексной маске. |
| `yolo_yaml` | YOLO YAML… | YAML (*.yaml *.yml) | Загрузить снимки, TXT, классы и split из YOLO data.yaml. Текущий набор заменяется. |
| `coco` | COCO JSON… | COCO JSON (*.json) | Импортировать COCO JSON с категориями и аннотациями; текущий набор заменяется. |
| `add_coco` | Добавить COCO… | COCO JSON (*.json) | Добавить COCO к текущему набору; классы объединяются по имени. |
| `unet_yaml` | YAML с масками… | YAML (*.yaml *.yml) | Загрузить пары снимок–индексная маска и классы из masks_data.yaml. В редакторе маски преобразуются в полигоны с потерей деталей. |
| `reports` | Отчёты автоподбора… | Отчёты (*_full_report.txt *_gradient_report.txt);;Текст (*.txt) | Добавить готовые пороговые и градиентные отчёты; это результаты поиска, а не входной датасет. |
| `report_folder` | Папка отчётов… | Папка | Добавить отчёты из папки и её вложенных каталогов. |
| `model_yolo` | Модель YOLO… | YOLO (*.pt *.onnx *.engine) | Веса YOLO для инференса. Возможность ONNX/engine зависит от установленного backend. |
| `model_yolo_train` | Веса YOLO… | PyTorch (*.pt) | Исходные PT-веса YOLO; экспортированные ONNX/engine не являются обучаемым checkpoint. |
| `model_torch` | Веса PyTorch… | PyTorch (*.pth *.pt) | Checkpoint, совместимый с архитектурой и предобработкой выбранной модели. |
| `model_onnx` | Модель ONNX… | ONNX (*.onnx) | ONNX-модель; проверьте входы, выходы и метаданные предобработки. |
| `model_ml` | Модель ML… | Модели ML (*.pkl *.joblib) | Сохранённая модель, совместимая с выбранными признаками и классами. |
| `preset` | Импорт пресета… | Пресеты обучения (*.txt) | Текстовый пресет обучения DeepLab: строки «ключ: значение». Пресет не содержит входной датасет. |
| `class_weights` | Веса классов… | JSON (*.json) | JSON-массив численных весов классов для функции потерь. |
| `training_run` | Результаты обучения… | Папка | Папка run с графиками, метриками и checkpoint; это не обучающий датасет. |

## Конфигурации UI и кнопок

| Файл | Область | Вызов |
| --- | --- | --- |
| [ui/DL_predict_settings.py:59](../ui/DL_predict_settings.py#L59) | `YOLOInferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_yolo', self)` |
| [ui/DL_predict_settings.py:156](../ui/DL_predict_settings.py#L156) | `UNetInferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_predict_settings.py:502](../ui/DL_predict_settings.py#L502) | `DeepLabV3InferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_predict_settings.py:875](../ui/DL_predict_settings.py#L875) | `SegFormerInferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_predict_settings.py:913](../ui/DL_predict_settings.py#L913) | `SAMInferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_predict_settings.py:952](../ui/DL_predict_settings.py#L952) | `CustomONNXInferenceSettings._setup_ui` | `source_dialogs.make_source_button('model_onnx', self)` |
| [ui/DL_train_settings_base.py:191](../ui/DL_train_settings_base.py#L191) | `YOLOTrainWidget.__init__` | `source_dialogs.DatasetSourceWidget(('dataset_root', 'yolo_yaml'), profile='yolo_train')` |
| [ui/DL_train_settings_base.py:253](../ui/DL_train_settings_base.py#L253) | `YOLOTrainWidget._add_model_selection` | `source_dialogs.make_source_button('model_yolo_train', self)` |
| [ui/DL_train_settings_base.py:443](../ui/DL_train_settings_base.py#L443) | `UNetTrainWidget.__init__` | `source_dialogs.DatasetSourceWidget(('dataset_root', 'unet_yaml'), profile='yolo_train')` |
| [ui/DL_train_settings_base.py:545](../ui/DL_train_settings_base.py#L545) | `UNetTrainWidget._add_model_selection` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_train_settings_new_model.py:101](../ui/DL_train_settings_new_model.py#L101) | `DeepLabV3TrainWidget.__init__` | `source_dialogs.DatasetSourceWidget(('dataset_root', 'unet_yaml'), profile='yolo_train')` |
| [ui/DL_train_settings_new_model.py:171](../ui/DL_train_settings_new_model.py#L171) | `DeepLabV3TrainWidget._add_preset_controls` | `source_dialogs.make_source_button('preset', self)` |
| [ui/DL_train_settings_new_model.py:231](../ui/DL_train_settings_new_model.py#L231) | `DeepLabV3TrainWidget._add_model_selection` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_train_settings_new_model.py:319](../ui/DL_train_settings_new_model.py#L319) | `DeepLabV3TrainWidget._add_class_weights_fields` | `source_dialogs.make_source_button('class_weights', self)` |
| [ui/DL_train_settings_new_model.py:1043](../ui/DL_train_settings_new_model.py#L1043) | `SegFormerTrainWidget.__init__` | `source_dialogs.DatasetSourceWidget(('dataset_root',), profile='yolo_train')` |
| [ui/DL_train_settings_new_model.py:1087](../ui/DL_train_settings_new_model.py#L1087) | `SegFormerTrainWidget._add_model_selection` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/DL_train_settings_new_model.py:1182](../ui/DL_train_settings_new_model.py#L1182) | `SAMTrainWidget.__init__` | `source_dialogs.DatasetSourceWidget(('dataset_root',), profile='yolo_train')` |
| [ui/DL_train_settings_new_model.py:1224](../ui/DL_train_settings_new_model.py#L1224) | `SAMTrainWidget._add_model_selection` | `source_dialogs.make_source_button('model_torch', self)` |
| [ui/autoreseach_ui/evaluation_controls.py:37](../ui/autoreseach_ui/evaluation_controls.py#L37) | `setup_evaluation_controls` | `DatasetSourceWidget(('reference_folder', 'yolo_yaml'))` |
| [ui/autoreseach_ui/evaluation_controls.py:62](../ui/autoreseach_ui/evaluation_controls.py#L62) | `setup_evaluation_controls` | `DatasetSourceWidget(('reports', 'report_folder'))` |
| [ui/dataset_audit/page_models.py:34](../ui/dataset_audit/page_models.py#L34) | `build_models` | `make_source_button('model_yolo')` |
| [ui/dataset_audit/page_models.py:109](../ui/dataset_audit/page_models.py#L109) | `build_models` | `DatasetSourceWidget(('dataset_root', 'yolo_yaml'), title='Датасет валидации YOLO')` |
| [ui/dataset_audit/page_training.py:31](../ui/dataset_audit/page_training.py#L31) | `build_training` | `make_source_button('training_run')` |
| [ui/dataset_audit/source_page.py:41](../ui/dataset_audit/source_page.py#L41) | `build_source` | `DatasetSourceWidget(('folder', 'labels', 'yolo_yaml', 'coco', 'add_coco', 'unet_yaml'), types=(0, 1, 2), formats=True, profile='dataset_analysis')` |
| [ui/dataset_source.py:286](../ui/dataset_source.py#L286) | `DatasetSourceWidget.__init__` | `make_source_button(key, self)` |
| [ui/image_navigation.py:42](../ui/image_navigation.py#L42) | `ImageNavigationWidget.__init__` | `DatasetSourceWidget(source_actions, profile=profile, parent=self)` |
| [ui/segmentation/panels.py:109](../ui/segmentation/panels.py#L109) | `begin_layout` | `ImageNavigationWidget(profile=kind)` |
| [ui_tabs/demo_ensemble_methods_ui.py:286](../ui_tabs/demo_ensemble_methods_ui.py#L286) | `setup_demo_ensemble_methods_ui` | `ImageNavigationWidget(profile='yolo_demo')` |
| [ui_tabs/layout_dataset_ui.py:42](../ui_tabs/layout_dataset_ui.py#L42) | `setup_layout_dataset_ui` | `ImageNavigationWidget(source_actions=('images', 'folder', 'labels', 'yolo_yaml', 'coco', 'unet_yaml'), profile='labeler')` |
| [ui_tabs/layout_dataset_ui.py:160](../ui_tabs/layout_dataset_ui.py#L160) | `setup_layout_dataset_ui` | `make_source_button('model_yolo')` |
| [ui_tabs/preparing_dataset_yaml_ui.py:82](../ui_tabs/preparing_dataset_yaml_ui.py#L82) | `setup_preparing_dataset_yaml_ui` | `DatasetSourceWidget(('folder', 'labels', 'yolo_yaml', 'coco', 'add_coco', 'unet_yaml'), types=(0, 1, 2), formats=True, profile='dataset')` |
| [ui_tabs/slice_dataset_sahi_ui.py:40](../ui_tabs/slice_dataset_sahi_ui.py#L40) | `setup_sahi_slicer_ui` | `DatasetSourceWidget(('folder', 'labels', 'yolo_yaml', 'coco', 'add_coco'), types=(0,), formats=True, profile='sahi')` |
| [ui_tabs/traditional_ml_methods_ui.py:144](../ui_tabs/traditional_ml_methods_ui.py#L144) | `setup_traditional_ml_ui` | `make_source_button('model_ml')` |
| [ui_tabs/viewing_dataset_ui.py:35](../ui_tabs/viewing_dataset_ui.py#L35) | `setup_viewing_dataset_ui` | `ImageNavigationWidget(source_actions=('images', 'folder', 'labels', 'yolo_yaml', 'coco', 'unet_yaml', 'masks'), profile='test')` |
| [ui_tabs/yolo_find_img_ui.py:60](../ui_tabs/yolo_find_img_ui.py#L60) | `setup_yolo_find_img_ui.setupUi` | `DatasetSourceWidget(('folder',), profile='yolo_sort')` |
| [ui_tabs/yolo_test_aeb_ui.py:52](../ui_tabs/yolo_test_aeb_ui.py#L52) | `setup_yolo_ae_vae_ui` | `ImageNavigationWidget(profile='yolo_aeb')` |
| [ui_tabs/yolo_test_aeb_ui.py:181](../ui_tabs/yolo_test_aeb_ui.py#L181) | `setup_yolo_ae_vae_ui` | `make_source_button('model_yolo')` |
| [ui_tabs/yolo_test_aeb_ui.py:231](../ui_tabs/yolo_test_aeb_ui.py#L231) | `setup_yolo_ae_vae_ui` | `make_source_button('model_torch')` |

## Обработчики общих диалогов

| Файл | Область | Вызов |
| --- | --- | --- |
| [tabs/demo_ensemble_methods.py:728](../tabs/demo_ensemble_methods.py#L728) | `YoloDemoWindow.load_images` | `source_dialogs.get_open_file(self, 'images', multiple=True)` |
| [tabs/demo_ensemble_methods.py:751](../tabs/demo_ensemble_methods.py#L751) | `YoloDemoWindow.load_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [tabs/layout_dataset.py:485](../tabs/layout_dataset.py#L485) | `Labeler.load_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [tabs/layout_dataset.py:492](../tabs/layout_dataset.py#L492) | `Labeler.load_images` | `source_dialogs.get_open_file(self, 'images', multiple=True)` |
| [tabs/layout_dataset.py:581](../tabs/layout_dataset.py#L581) | `Labeler.load_labels_folder` | `source_dialogs.get_directory(self, 'labels')` |
| [tabs/layout_dataset.py:606](../tabs/layout_dataset.py#L606) | `Labeler.load_yaml` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [tabs/layout_dataset.py:653](../tabs/layout_dataset.py#L653) | `Labeler.load_coco_json` | `source_dialogs.choose_coco_sources(self)` |
| [tabs/layout_dataset.py:701](../tabs/layout_dataset.py#L701) | `Labeler.load_unet_yaml` | `source_dialogs.get_open_file(self, 'unet_yaml')` |
| [tabs/layout_dataset.py:1120](../tabs/layout_dataset.py#L1120) | `Labeler.load_auto_model` | `source_dialogs.get_open_file(self, 'model_yolo')` |
| [tabs/model_research.py:134](../tabs/model_research.py#L134) | `ModelResearchApp.select_truth_yaml` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [tabs/model_research.py:139](../tabs/model_research.py#L139) | `ModelResearchApp.select_truth_folder` | `source_dialogs.get_directory(self, 'reference_folder')` |
| [tabs/model_research.py:144](../tabs/model_research.py#L144) | `ModelResearchApp.select_report_files` | `source_dialogs.get_open_file(self, 'reports', multiple=True)` |
| [tabs/model_research.py:160](../tabs/model_research.py#L160) | `ModelResearchApp.select_folder` | `source_dialogs.get_directory(self, 'report_folder')` |
| [tabs/viewing_dataset.py:618](../tabs/viewing_dataset.py#L618) | `ViewingDataset.load_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [tabs/viewing_dataset.py:626](../tabs/viewing_dataset.py#L626) | `ViewingDataset.load_images` | `source_dialogs.get_open_file(self, 'images', multiple=True)` |
| [tabs/viewing_dataset.py:734](../tabs/viewing_dataset.py#L734) | `ViewingDataset.load_labels_folder` | `source_dialogs.get_directory(self, 'labels')` |
| [tabs/viewing_dataset.py:762](../tabs/viewing_dataset.py#L762) | `ViewingDataset.load_masks_folder` | `source_dialogs.get_directory(self, 'masks')` |
| [tabs/viewing_dataset.py:1584](../tabs/viewing_dataset.py#L1584) | `ViewingDataset.load_yaml` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [tabs/viewing_dataset.py:1640](../tabs/viewing_dataset.py#L1640) | `ViewingDataset.load_coco_json` | `source_dialogs.choose_coco_sources(self)` |
| [tabs/viewing_dataset.py:1690](../tabs/viewing_dataset.py#L1690) | `ViewingDataset.load_unet_yaml` | `source_dialogs.get_open_file(self, 'unet_yaml')` |
| [tabs/yolo_find_img.py:457](../tabs/yolo_find_img.py#L457) | `FindImagesWindow.select_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [tabs/yolo_test_aeb.py:405](../tabs/yolo_test_aeb.py#L405) | `YoloInspectWindow.load_images_from_dialog` | `source_dialogs.get_open_file(self, 'images', multiple=True)` |
| [tabs/yolo_test_aeb.py:411](../tabs/yolo_test_aeb.py#L411) | `YoloInspectWindow.load_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [tabs/yolo_test_aeb.py:546](../tabs/yolo_test_aeb.py#L546) | `YoloInspectWindow.load_yolo_model` | `source_dialogs.get_open_file(self, 'model_yolo')` |
| [tabs/yolo_test_aeb.py:1056](../tabs/yolo_test_aeb.py#L1056) | `YoloInspectWindow.load_ae_model` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_predict_settings.py:116](../ui/DL_predict_settings.py#L116) | `YOLOInferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_yolo')` |
| [ui/DL_predict_settings.py:303](../ui/DL_predict_settings.py#L303) | `UNetInferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_predict_settings.py:644](../ui/DL_predict_settings.py#L644) | `DeepLabV3InferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_predict_settings.py:891](../ui/DL_predict_settings.py#L891) | `SegFormerInferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_predict_settings.py:932](../ui/DL_predict_settings.py#L932) | `SAMInferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_predict_settings.py:975](../ui/DL_predict_settings.py#L975) | `CustomONNXInferenceSettings._browse_file` | `source_dialogs.get_open_file(self, 'model_onnx')` |
| [ui/DL_train_settings_base.py:223](../ui/DL_train_settings_base.py#L223) | `YOLOTrainWidget._browse_dataset_folder` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/DL_train_settings_base.py:229](../ui/DL_train_settings_base.py#L229) | `YOLOTrainWidget._browse_dataset_yaml` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [ui/DL_train_settings_base.py:275](../ui/DL_train_settings_base.py#L275) | `YOLOTrainWidget._browse_custom_model` | `source_dialogs.get_open_file(self, 'model_yolo_train')` |
| [ui/DL_train_settings_base.py:480](../ui/DL_train_settings_base.py#L480) | `UNetTrainWidget._browse_dataset` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/DL_train_settings_base.py:486](../ui/DL_train_settings_base.py#L486) | `UNetTrainWidget._browse_unet_yaml` | `source_dialogs.get_open_file(self, 'unet_yaml')` |
| [ui/DL_train_settings_base.py:565](../ui/DL_train_settings_base.py#L565) | `UNetTrainWidget._browse_custom_model` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_train_settings_new_model.py:190](../ui/DL_train_settings_new_model.py#L190) | `DeepLabV3TrainWidget._browse_preset_file` | `source_dialogs.get_open_file(self, 'preset')` |
| [ui/DL_train_settings_new_model.py:198](../ui/DL_train_settings_new_model.py#L198) | `DeepLabV3TrainWidget._browse_dataset` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/DL_train_settings_new_model.py:204](../ui/DL_train_settings_new_model.py#L204) | `DeepLabV3TrainWidget._browse_deeplab_yaml` | `source_dialogs.get_open_file(self, 'unet_yaml')` |
| [ui/DL_train_settings_new_model.py:249](../ui/DL_train_settings_new_model.py#L249) | `DeepLabV3TrainWidget._browse_custom_model` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_train_settings_new_model.py:338](../ui/DL_train_settings_new_model.py#L338) | `DeepLabV3TrainWidget._browse_weights_file` | `source_dialogs.get_open_file(self, 'class_weights')` |
| [ui/DL_train_settings_new_model.py:1062](../ui/DL_train_settings_new_model.py#L1062) | `SegFormerTrainWidget._browse_dataset` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/DL_train_settings_new_model.py:1107](../ui/DL_train_settings_new_model.py#L1107) | `SegFormerTrainWidget._browse_custom_model` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/DL_train_settings_new_model.py:1201](../ui/DL_train_settings_new_model.py#L1201) | `SAMTrainWidget._browse_dataset` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/DL_train_settings_new_model.py:1244](../ui/DL_train_settings_new_model.py#L1244) | `SAMTrainWidget._browse_custom_model` | `source_dialogs.get_open_file(self, 'model_torch')` |
| [ui/dataset_audit/models.py:30](../ui/dataset_audit/models.py#L30) | `DatasetModelController.load_audit_model` | `source_dialogs.get_open_file(self, 'model_yolo')` |
| [ui/dataset_audit/models.py:488](../ui/dataset_audit/models.py#L488) | `DatasetModelController.browse_val_data` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [ui/dataset_audit/models.py:493](../ui/dataset_audit/models.py#L493) | `DatasetModelController.browse_val_folder` | `source_dialogs.get_directory(self, 'dataset_root')` |
| [ui/dataset_audit/training_results.py:14](../ui/dataset_audit/training_results.py#L14) | `DatasetTrainingResultsController.browse_training_results` | `source_dialogs.get_directory(self, 'training_run')` |
| [ui/dataset_audit/training_results.py:21](../ui/dataset_audit/training_results.py#L21) | `DatasetTrainingResultsController.load_training_results` | `source_dialogs.get_directory(self, 'training_run')` |
| [ui/dataset_source.py:358](../ui/dataset_source.py#L358) | `choose_coco_sources` | `get_open_file(owner, 'coco', multiple=True)` |
| [ui/segmentation/computation.py:351](../ui/segmentation/computation.py#L351) | `load_ml` | `source_dialogs.get_open_file(owner, 'model_ml')` |
| [ui/segmentation/image_input.py:18](../ui/segmentation/image_input.py#L18) | `choose_images` | `source_dialogs.get_open_file(owner, 'images', multiple=True)` |
| [ui/segmentation/image_input.py:25](../ui/segmentation/image_input.py#L25) | `choose_folder` | `source_dialogs.get_directory(owner, 'folder')` |
| [utils_new/dataset_loader_new.py:147](../utils_new/dataset_loader_new.py#L147) | `DatasetLoaderMixin.select_images_folder` | `source_dialogs.get_directory(self, 'folder')` |
| [utils_new/dataset_loader_new.py:160](../utils_new/dataset_loader_new.py#L160) | `DatasetLoaderMixin.select_labels_folder` | `source_dialogs.get_directory(self, kind)` |
| [utils_new/dataset_loader_new.py:177](../utils_new/dataset_loader_new.py#L177) | `DatasetLoaderMixin.load_from_yaml` | `source_dialogs.get_open_file(self, 'yolo_yaml')` |
| [utils_new/dataset_loader_new.py:261](../utils_new/dataset_loader_new.py#L261) | `DatasetLoaderMixin._handle_load_coco` | `source_dialogs.choose_coco_sources(self)` |
| [utils_new/dataset_loader_new.py:357](../utils_new/dataset_loader_new.py#L357) | `DatasetLoaderMixin.load_from_unet_yaml` | `source_dialogs.get_open_file(self, 'unet_yaml')` |

## Нативные диалоги: владелец и отдельные пути экспорта

| Файл | Область | Вызов |
| --- | --- | --- |
| [tabs/model_research.py:271](../tabs/model_research.py#L271) | `ModelResearchApp.save_evaluation` | `QFileDialog.getExistingDirectory(self, 'Выберите отдельную пустую папку для сводки, настроек и результатов по снимкам')` |
| [tabs/preparing_dataset_yaml.py:667](../tabs/preparing_dataset_yaml.py#L667) | `DatasetPreparationWindow.select_output_folder` | `QFileDialog.getExistingDirectory(self, 'Select folder to save YAML and dataset')` |
| [tabs/slice_dataset_sahi.py:948](../tabs/slice_dataset_sahi.py#L948) | `SahiSlicerWindow.browse_output` | `QFileDialog.getExistingDirectory(self, 'Выберите папку для нового датасета')` |
| [tabs/yolo_find_img.py:1024](../tabs/yolo_find_img.py#L1024) | `FindImagesWindow.browse_target_folder` | `QFileDialog.getExistingDirectory(self, 'Выберите целевую папку')` |
| [tabs/yolo_test_aeb.py:980](../tabs/yolo_test_aeb.py#L980) | `YoloInspectWindow.save_all_crops` | `QFileDialog.getExistingDirectory(self, 'Папка для сохранения кропов', start_dir)` |
| [ui/DL_train_settings_base.py:67](../ui/DL_train_settings_base.py#L67) | `BaseTrainWidget._browse_project` | `QFileDialog.getExistingDirectory(self, 'Выберите папку для сохранения результатов')` |
| [ui/dataset_audit/selection.py:520](../ui/dataset_audit/selection.py#L520) | `DatasetSelectionController.export_filtered_dataset` | `QFileDialog.getExistingDirectory(self, 'Папка для датасета')` |
| [ui/dataset_source.py:254](../ui/dataset_source.py#L254) | `get_directory` | `QFileDialog.getExistingDirectory(owner, ACTIONS[key][1], directory or str(settings.value(key, '')))` |
| [ui/dataset_source.py:361](../ui/dataset_source.py#L361) | `choose_coco_sources` | `QFileDialog.getExistingDirectory(owner, 'Выберите папку COCO JSON', str(settings.value('coco_folder', '')))` |

Последняя таблица содержит также выбор **выходных** каталогов. Эти операции не являются импортом и не переносятся в панель источника. Все getOpenFileName(s) принадлежат ui/dataset_source.py. Смена типа данных меняет назначение кнопки labels на masks; переменная kind в обработчике отражает это состояние.
