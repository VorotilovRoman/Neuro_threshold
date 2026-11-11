# Аудит первых пяти вкладок сегментации

[План миграции](Segmentation_Migration_Plan.md) · [Документация](README.md)

Исходная точка: `89c9c11`, ветка `work`, 11 ноября 2026 года. Это исследование
и планирование: код вкладок, алгоритмы и пользовательские настройки не изменялись.
Проверены пять контроллеров, пять сборщиков UI, прямые и существенные транзитивные
зависимости, новые файловые/геометрические службы, морфология, пресеты и автоподбор.

## 1. Фактическое состояние

Пять файлов вкладок содержат 5 444 строки и 248 методов основных окон, включая
обработчики Qt. Это размер изученной области, а не оценка числа лишних строк.

| Вкладка | Основной файл | Строк / методов окна | Изображения | Представление |
| --- | --- | --- | --- | --- |
| Пороговые | `tabs/threshold_methods.py` | 1052 / 52 | Ленивый кэш, рабочая сторона до 1024 | 4 новых DatasetReviewGraphicsView |
| Градиентные | `tabs/gradient_methods.py` | 847 / 44 | Ленивый кэш, до 1024 | 6 старых SmartGraphicsView |
| Интерактивные | `tabs/interactive_methods.py` | 1069 / 48 | Ленивый кэш, до 1024 | 4 старых SmartGraphicsView; ROI/штрихи FG/BG |
| Традиционное ML | `tabs/traditional_ml_methods.py` | 1278 / 58 | Eager-списки images/gray/annotations, до 640 | 4 старых SmartGraphicsView |
| Deep Learning | `tabs/deep_learning_methods.py` | 1198 / 46 | Ленивый кэш, до 1024 | 4 старых SmartGraphicsView |

Итого 22 видовых окна: четыре уже используют новый read-only просмотрщик,
18 — старый. Число стадий различается: градиенты дополнительно показывают
градиентную карту и залитую маску. Нельзя навязать всем одинаковые четыре стадии.

## 2. Какие старые утилиты реально используются

| Старый модуль / имена | Прямые потребители из пяти вкладок | Назначение | Готовая замена |
| --- | --- | --- | --- |
| `utils.image_io.LazyImageCache`, `load_one_image_item` | threshold, gradient, interactive, DL | Файлы, grayscale, resize, LRU/prefetch | `utils_new.dataset_files_new`; API близок, поведение I/O проверять отдельно |
| `utils.image_io.load_images_universal` | ML | Полная загрузка изображений и обучающих аннотаций, Qt progress | Прямого полного аналога нет: нужен адаптер сессии, использующий новые leaf-модули; Qt progress — в ui |
| `utils.image_io.numpy_to_qpixmap` | Все пять | NumPy → Qt | Новая реализация есть в `utils_new.smart_view_new`; каноническое место предлагается в `ui/image_conversion.py` |
| `utils.image_io.save_annotations` | Все пять | Запись выбранных объектов в TXT рядом со снимком | `utils_new.dataset_yolo_labels_new`, включая план сохранения и адаптеры форматов |
| `utils.presets.*PresetManager` | Все пять | JSON, чтение полей окна и запись значений в виджеты | Готовой отделённой службы нет; разделить store/schema и Qt binding |
| `utils.settings.settings` | Все пять | Общая тема/цвета/параметры отображения | Сохранить один QObject-singleton, затем разделить Qt-service и чистое хранилище |
| `utils.smart_view.SmartGraphicsView` | Через UI gradient, interactive, ML, DL | Отображение, инструменты и рисование | Новый SmartGraphicsViewNew / DatasetReviewGraphicsView; переносить по возможностям конкретного вида |

Отдельно: `utils_ops` и `utils_auto_research` — другие каталоги. Их нельзя
автоматически считать той же папкой `/utils`, но они входят в граф миграции:

- `utils_ops.segmentation_ops` смешивает алгоритмы, рисование OpenCV,
  форматирование списков, Qt и `utils.settings`. Четыре вкладки используют его
  извлечение контуров/боксов/OBB, морфологию и растровое рисование выбранных объектов.
- threshold уже использует `utils_new.threshold_processing`,
  `utils_new.annotation_processing` и `utils_new.morphology`.
- gradient использует `utils_ops.gradient_ops.apply_gradient_method`.
- interactive использует семь функций из `utils_ops.interactive_ops`:
  GrabCut, Watershed, Random Walker, Active Contour, SuperCut, OneCut, Lazy Snapping.
- ML использует признаки pixel/superpixel, кластеризацию, supervised train/predict
  и elbow из `utils_ops.traditional_ml_ops`; это самостоятельная алгоритмическая часть.
- DL использует `utils_ops.deep_learning_ops.create_segmentor` и классы моделей.
- Пороговый и градиентный автоподбор находятся в `utils_auto_research`; они
  управляют полями окна и форматируют отчёты, а не являются независимым backend.

### Транзитивные зависимости, которые легко пропустить

`utils_new.smart_view_new` всё ещё импортирует `utils.settings`. То же делает
`ui.image_navigation`, а `utils.presets` импортирует `normalize_method_name` из
старого `utils_ops.segmentation_ops`. Поэтому замена только импортов пяти
контроллеров не устранит все старые зависимости.

`utils_new.dataset_view_new` — compatibility alias для `annotation_processing`,
а не новый графический просмотрщик. Для UI это неверная цель миграции.
`dataset_io_new` и `dataset_validation_new` — фасады; новому коду предпочтительнее
импортировать владельцев функций напрямую.

## 3. Уже общие компоненты UI

| Компонент | Текущее применение | Что делать |
| --- | --- | --- |
| `ui.image_navigation.ImageNavigationWidget` | Все пять: кнопки загрузки, навигация, номер кадра, resize | Оставить; вынести из вкладок одинаковые file dialogs и управление сессией в Qt-controller |
| `ui.image_navigation.NavigationShortcutInstaller` | Встраивание через main, общая обработка клавиш | Сохранить; тестировать реальную активную вкладку и фокус ввода |
| `ui.log_widget.LogWidget` | Все пять UI | Сохранить, унифицировать вызов log/progress вместо разных проверок атрибутов |
| `ui.morphology_ui.MorphologyWidget` | Пороговая вкладка | Использовать повторно для остальных; не создавать второй виджет морфологии |
| `ui.interactive_params_ui.InteractiveParametersUI` | Интерактивная вкладка | Оставить специализированным; убрать ненужное `self.parent`, затеняющее QObject.parent |
| `ui.DL_predict_settings` | DL-параметры разных моделей | Оставить специализированными: metadata/override/resize специфичны для моделей |
| `ui.application.exec_window` | main и автономные окна | Сохранить жизненный цикл Qt, раннюю подготовку Torch на Windows и позднее уничтожение QApplication |
| `ui.control_buttons.ControlButtons` | Не импортируется этими пятью UI | Не подключать формально ради повторного использования: Generate/Cancel/Save не заменяют панели методов |

Гистограмма, список результатов, координаты, экспорт, пресеты и сетка стадий
повторяются внутри пяти UI и контроллеров. Их общего законченного компонента
пока нет. Распределение алгоритмов по методам при этом отличается закономерно.

## 4. Повторяющиеся функции контроллеров

У всех пяти есть `load_images_from_dialog`, `load_folder`, `reload_current_images`,
`prev_image`, `next_image`, `goto_image`, `clear_images`, `reset_all_zooms`,
`update_navigation_state`, `schedule_update`, обработка resize и настроек.

Также повторяются `update_histogram`, `toggle_histogram`, `toggle_log`,
`update_object_list`, `on_object_selection_changed`, `update_coordinates_display`,
`save_current_annotations`, `save_as_preset`, `delete_current_preset`,
`update_preset_combo`, `on_preset_activated`. Пять функций сохранения занимают
по 47–48 строк и реализуют почти один пользовательский сценарий.

`display_current_image` и `draw_objects_on_image` нельзя целиком объединять
по совпадению имени: внутри них находятся разные конвейеры. Общими являются
управление входом, доставка результата, постобработка и представление; сам
алгоритм и специфичные стадии должны остаться раздельными.

## 5. Подтверждённые дефекты и различия

### A. Автоподбор уже несовместим с ленивой загрузкой — P0

`AutoResearch.run()` обращается к `self.main.display_images`, а
`GradientAutoResearch.run()` — к `self.parent.display_images`. У актуальных
ThresholdWindow и GradientMethodsWindow такого атрибута нет.

В отдельном offscreen-процессе создано каждое окно и вызван обработчик:

```text
ThresholdWindow display_images present: False
AttributeError: 'ThresholdWindow' object has no attribute 'display_images'
GradientMethodsWindow display_images present: False
AttributeError: 'GradientMethodsWindow' object has no attribute 'display_images'
```

Существующий запуск через кнопку может не дойти до обработчика без картинки,
но после загрузки картинок отсутствующий атрибут не появляется. Нужна работа
через image_paths/session.get(index), а не восстановление полной eager-копии.

### B. Состояние морфологии теряет точность при старом автоподборе — P0/P1

Threshold уже хранит ядра в пикселях. `_sync_legacy_morph_sliders` переводит их
в проценты, округляет и ограничивает значением 10. AutoResearch сохраняет и
восстанавливает только значения этих legacy sliders, не полный get_settings.

Следствие по коду: при min(H,W)=64 ядро 13 px превращается в slider=10, а обратный
перевод даёт 6 px. После исправления A автоподбор не должен получать такой
способ восстановления. Этот сценарий выявлен по коду; полный перебор с таким
состоянием не запускался. Нужно сохранять полный снимок схемы морфологии.

### C. Новая конвертация числовых изображений не эквивалентна старой — миграционный риск

Read-only проба float32 `[0, 2, 5, 10]`:

| Реализация | uint8 |
| --- | --- |
| `utils.image_io.normalize_to_uint8` | `[0, 2, 5, 10]` |
| `utils_new.dataset_files_new.normalize_to_uint8` / ImageStats | `[0, 51, 127, 255]` |

Это другой диапазон интенсивностей и другой результат сегментации при том же
пороге. Нужна явно выбранная политика преобразования; нельзя молча менять её
вместе с импортом. Для signed/float и исходников выше 8 бит нужны отдельные fixtures.

### D. Resize отличается, а старый падает на тонком изображении — подтверждено

При уменьшении изображения 301×1000 до стороны 640 старая функция даёт
192×640, новая — 193×640 (floor против round). Для 1×10000 → 640 старый код
вычисляет нулевую высоту и получает ошибку OpenCV; новый возвращает 1×640.
Нужно исправить нулевой размер, зафиксировать правило округления и считать
реальные масштабы sx/sy, не один предполагаемый коэффициент.

### E. Четырёхточечный Segment при старом сохранении меняет тип — подтверждено

Старый writer сохраняет class+8 координат, то есть 9 токенов. Его же auto-reader
читает такую строку как OBB. Новый writer добавляет коллинеарную точку: 11 токенов,
геометрия та же, auto-reader сохраняет Segment. Это важно и для результатов
пороговых/градиентных методов, где прямоугольный контур имеет четыре вершины.

### F. Legacy OBB xywhr ошибочен на непропорциональном кадре — подтверждено

Для кадра 1000×200 и физического прямоугольника 300×60 под углом 15°:

| Writer | cx cy w h angle |
| --- | --- |
| Старый | `0.5 0.5 0.484451 0.290193 0.929597` |
| Новый | `0.5 0.5 0.3 0.3 0.261799` |

Оба вызова вернули успех. Старый алгоритм вычисляет прямоугольник в искажённых
нормализованных осях. Пять текущих Save по умолчанию пишут corners, поэтому этот
дефект нельзя выдавать за ошибку каждого их обычного сохранения. Но при унификации
экспорта и добавлении xywhr нужно использовать физическую геометрию нового ядра.

### G. Морфология градиентов имеет особые правила — подтверждённое отличие

Основные close/open sliders у gradient делятся на 200; у interactive/ML/DL — на
100. Дополнительно gradient после заливки контуров повторно делает closing:
Rectangle, размер не меньше 3 и округление до нечётного. Это отдельная стадия,
не эквивалентная простой замене на MorphologyWidget. Порядок инверсии тоже важен.

### H. Классовые выходы моделей сведены к бинарной логике — ограничение

UNet/DeepLab multiclass-ветки масштабируют class IDs в uint8; дальше морфология и
контуры работают с ненулевыми пикселями, обычно создавая class 0. В ML supervised
предсказание также может превращаться в uint8*255. Это не полноценный class-aware
экспорт. В миграции надо сохранить исходную карту классов и отдельно обозначить
текущий режим «объект/фон»; class-aware режим вводить отдельным этапом.

### I. Длительные вычисления идут в GUI-потоке — риск отзывчивости

В DL `run_segmentation` непосредственно вызывает predict, в ML train/apply —
fit/predict. processEvents перед вызовом не делает само вычисление асинхронным.
Необходимо разделить запрос/результат и позднее ввести worker с защитой от
устаревшего результата, без одновременной переработки математики моделей.

### J. Выбор объектов и экспорт требуют отдельного контракта

Сейчас все Save пишут выбранные галочками объекты рядом со снимком; пустой TXT
является валидным отрицательным примером. Пересчёт часто заново создаёт список
и отмечает все объекты. Нельзя перепутать checked-for-export, одиночный фокус
просмотрщика и индекс в отфильтрованном векторном списке.

## 6. Что уже можно переиспользовать с высокой уверенностью

- Два класса LazyImageCache структурно совпали по AST без номеров строк.
  Это хороший первый перенос, но loader и жизненный цикл всё равно тестируются.
- На простой прямоугольной маске старые segment_contours/projections/min_area_rect
  и новые extract_annotations_from_mask дали одинаковые Segment/Detect/OBB.
  Это полезная начальная проба, не доказательство для дыр, касаний края и вырождений.
- Новые view уже имеют set_callbacks для ROI/FG/BG и set_image_data, поэтому не
  нужен второй редактор с нуля. Но проверка штрихов при zoom/resize обязательна:
  имя метода ещё не доказывает одинаковую семантику.
- MorphologyWidget поддерживает close/open/erode/dilate, порядок, форму и JSON;
  set_legacy_threshold подходит как основа адаптера коэффициентов, но не заменяет
  специфическую градиентную стадию и снимок состояния автоподбора.

## 7. Границы исследования

Синтетические пробы запускались в Linux CPU/offscreen, без обучения, весов и
изменения датасетов. Код миграции не выполнялся; дефекты A–I здесь не объявляются
исправленными. Windows/GPU, все параметры интерактивных методов и реальные
многоклассовые модели требуют отдельной приёмки. Для реализации нужен план ниже.

## Приложение: прямые локальные импорты пяти вкладок и UI

Имена ниже действительно используются в AST соответствующего файла; это статическая зависимость, а не подтверждение выполнения каждой ветки.

### tabs/threshold_methods.py

- [runtime_torch](../runtime_torch.py): `prepare_windows_torch`.
- [path_setup](../path_setup.py): `get_project_root`.
- [ui_tabs.threshold_methods_ui](../ui_tabs/threshold_methods_ui.py): `setup_threshold_ui`.
- [utils.image_io](../utils/image_io.py): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`, `save_annotations`.
- [utils.presets](../utils/presets.py): `PresetManager`.
- [utils.settings](../utils/settings.py): `settings`.
- [utils_auto_research.threshold_auto_research](../utils_auto_research/threshold_auto_research.py): `AutoResearch`.
- [utils_auto_research.threshold_auto_settings_dialog](../utils_auto_research/threshold_auto_settings_dialog.py): `AutoSettingsDialog`.
- [utils_new.annotation_processing](../utils_new/annotation_processing.py): `extract_annotations_from_mask`, `format_object_for_list`.
- [utils_new.morphology](../utils_new/morphology.py): `apply_morphology`.
- [utils_new.threshold_processing](../utils_new/threshold_processing.py): `apply_threshold_method`, `normalize_method_name`.
- [ui.application](../ui/application.py): `exec_window`.

### tabs/gradient_methods.py

- [runtime_torch](../runtime_torch.py): `prepare_windows_torch`.
- [path_setup](../path_setup.py): `get_project_root`.
- [ui_tabs.gradient_methods_ui](../ui_tabs/gradient_methods_ui.py): `setup_gradient_ui`.
- [utils.image_io](../utils/image_io.py): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`, `save_annotations`.
- [utils.presets](../utils/presets.py): `GradientPresetManager`.
- [utils.settings](../utils/settings.py): `settings`.
- [utils_auto_research.gradient_auto_research](../utils_auto_research/gradient_auto_research.py): `GradientAutoResearch`.
- [utils_auto_research.gradient_auto_settings_dialog](../utils_auto_research/gradient_auto_settings_dialog.py): `GradientAutoSettingsDialog`.
- [utils_ops.gradient_ops](../utils_ops/gradient_ops.py): `apply_gradient_method`.
- [utils_ops.segmentation_ops](../utils_ops/segmentation_ops.py): `apply_morphology`, `draw_selected_objects`, `format_object_for_list`, `get_display_params`, `segment_contours`, `segment_min_area_rect`, `segment_projections`.
- [ui.application](../ui/application.py): `exec_window`.

### tabs/interactive_methods.py

- [runtime_torch](../runtime_torch.py): `prepare_windows_torch`.
- [ui_tabs.interactive_methods_ui](../ui_tabs/interactive_methods_ui.py): `setup_interactive_ui`.
- [utils.image_io](../utils/image_io.py): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`, `save_annotations`.
- [utils.presets](../utils/presets.py): `InteractivePresetManager`.
- [utils.settings](../utils/settings.py): `settings`.
- [utils_ops.interactive_ops](../utils_ops/interactive_ops.py): `active_contour_segmentation`, `grabcut_segmentation`, `lazy_snapping`, `onecut_segmentation`, `random_walker_segmentation`, `supercut_segmentation`, `watershed_segmentation`.
- [utils_ops.segmentation_ops](../utils_ops/segmentation_ops.py): `apply_morphology`, `draw_selected_objects`, `format_object_for_list`, `get_display_params`, `segment_contours`, `segment_min_area_rect`, `segment_projections`.
- [ui.application](../ui/application.py): `exec_window`.

### tabs/traditional_ml_methods.py

- [runtime_torch](../runtime_torch.py): `prepare_windows_torch`.
- [ui_tabs.traditional_ml_methods_ui](../ui_tabs/traditional_ml_methods_ui.py): `setup_traditional_ml_ui`.
- [utils.image_io](../utils/image_io.py): `load_images_universal`, `numpy_to_qpixmap`, `save_annotations`.
- [utils.presets](../utils/presets.py): `TraditionalMLPresetManager`.
- [utils.settings](../utils/settings.py): `settings`.
- [utils_ops.segmentation_ops](../utils_ops/segmentation_ops.py): `apply_morphology`, `draw_selected_objects`, `format_object_for_list`, `get_display_params`, `segment_contours`, `segment_min_area_rect`, `segment_projections`.
- [utils_ops.traditional_ml_ops](../utils_ops/traditional_ml_ops.py): `compute_elbow_kmeans`, `create_superpixels`, `extract_pixel_features`, `extract_superpixel_features`, `find_optimal_k_elbow`, `predict_superpixel`, `train_decision_tree`, `train_kmeans`, `train_meanshift`, `train_random_forest`, `train_svm`, `train_xgboost`.
- [ui.application](../ui/application.py): `exec_window`.

### tabs/deep_learning_methods.py

- [runtime_torch](../runtime_torch.py): `prepare_windows_torch`.
- [ui_tabs.deep_learning_methods_ui](../ui_tabs/deep_learning_methods_ui.py): `setup_deep_learning_methods_ui`.
- [utils.image_io](../utils/image_io.py): `LazyImageCache`, `load_one_image_item`, `numpy_to_qpixmap`, `save_annotations`.
- [utils.presets](../utils/presets.py): `DeepLearningPresetManager`.
- [utils.settings](../utils/settings.py): `settings`.
- [utils_ops.deep_learning_ops](../utils_ops/deep_learning_ops.py): `create_segmentor`.
- [utils_ops.segmentation_ops](../utils_ops/segmentation_ops.py): `apply_morphology`, `draw_selected_objects`, `format_object_for_list`, `get_display_params`, `segment_contours`, `segment_min_area_rect`, `segment_projections`.
- [ui.application](../ui/application.py): `exec_window`.

### ui_tabs/threshold_methods_ui.py

- [ui.image_navigation](../ui/image_navigation.py): `ImageNavigationWidget`.
- [ui.log_widget](../ui/log_widget.py): `LogWidget`.
- [ui.morphology_ui](../ui/morphology_ui.py): `MorphologyWidget`.
- [utils_new.dataset_review_view](../utils_new/dataset_review_view.py): `DatasetReviewGraphicsView`.

### ui_tabs/gradient_methods_ui.py

- [ui.image_navigation](../ui/image_navigation.py): `ImageNavigationWidget`.
- [ui.log_widget](../ui/log_widget.py): `LogWidget`.
- [utils.smart_view](../utils/smart_view.py): `SmartGraphicsView`.

### ui_tabs/interactive_methods_ui.py

- [ui.image_navigation](../ui/image_navigation.py): `ImageNavigationWidget`.
- [ui.log_widget](../ui/log_widget.py): `LogWidget`.
- [utils.smart_view](../utils/smart_view.py): `SmartGraphicsView`.
- [ui.interactive_params_ui](../ui/interactive_params_ui.py): `InteractiveParametersUI`.

### ui_tabs/traditional_ml_methods_ui.py

- [ui.image_navigation](../ui/image_navigation.py): `ImageNavigationWidget`.
- [ui.log_widget](../ui/log_widget.py): `LogWidget`.
- [utils.smart_view](../utils/smart_view.py): `SmartGraphicsView`.

### ui_tabs/deep_learning_methods_ui.py

- [ui.DL_predict_settings](../ui/DL_predict_settings.py): `CustomONNXInferenceSettings`, `DeepLabV3InferenceSettings`, `SAMInferenceSettings`, `SegFormerInferenceSettings`, `UNetInferenceSettings`, `YOLOInferenceSettings`.
- [ui.image_navigation](../ui/image_navigation.py): `ImageNavigationWidget`.
- [ui.log_widget](../ui/log_widget.py): `LogWidget`.
- [utils.smart_view](../utils/smart_view.py): `SmartGraphicsView`.
