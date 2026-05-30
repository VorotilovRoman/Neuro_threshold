
# tools.md

## Общие инструменты, используемые во всех вкладках

Ниже описаны компоненты и функции, которые не привязаны к конкретной вкладке и переиспользуются по всему приложению. Они обеспечивают загрузку изображений, навигацию, логирование, работу с аннотациями и графическими видами.

### 1. ImageNavigationWidget (модуль `ui.image_navigation`)

Стандартный виджет для управления набором изображений.

**Внешний вид:**  
- Кнопка «Load Images» – загрузить отдельные файлы.  
- Кнопка «Load Folder» – загрузить все изображения из папки.  
- Кнопки «◀ Previous» и «Next ▶» – листание.  
- Поле ввода номера страницы (QSpinBox) с отображением общего количества.  
- Чекбокс «Resize to max 1024px» – управляет ресайзом при загрузке.

**Сигналы (pyqtSignal):**
- `load_images()` – запросить загрузку файлов.  
- `load_folder()` – запросить загрузку папки.  
- `prev()` – предыдущее изображение.  
- `next()` – следующее изображение.  
- `goto_page(int)` – переход к изображению с номером (начиная с 1).  
- `resize_toggled(bool)` – изменён режим ресайза.

**Публичные методы:**
- `set_navigation_enabled(enabled)`, `set_prev_enabled(enabled)`, `set_next_enabled(enabled)` – управление доступностью кнопок.  
- `set_resize_checked(checked)` – программно установить чекбокс.  
- `is_resize_enabled()` – текущее состояние ресайза.  
- `set_current_index(idx, total)` – обновить отображаемый номер (idx с 0) и общее количество.

---

### 2. LogWidget (модуль `ui.log_widget`)

Виджет для вывода текстовых сообщений (лог).

**Параметры конструктора:**
- `show_clear_btn` – показывать кнопку «Очистить лог».  
- `show_progress` – показывать QProgressBar под логом.

**Методы:**
- `log(message)` – добавить строку в лог.  
- `clear()` – очистить весь лог.  
- `set_progress(value, max_val=None, fmt=None)` – обновить прогресс‑бар (если он есть).  

**Сигнал:** `cleared()` – испускается при очистке лога.

---

### 3. SmartGraphicsView (модуль `utils.smart_view`)

Расширенный `QGraphicsView` для отображения изображений с интерактивными возможностями: зум, панорамирование, рисование прямоугольников/штрихов, редактирование аннотаций.

**Основные возможности:**
- Зум: колёсико мыши, точка привязки – под курсором.  
- Панорамирование: перетаскивание левой кнопкой (режим по умолчанию).  
- Сброс зума: метод `reset_view()`.  
- Сохранение текущего изображения в PNG (контекстное меню правой кнопкой).  
- Режимы рисования (`set_drawing_tool(tool)`):  
  - `"rect"` – прямоугольник (выделение области).  
  - `"fg"` / `"bg"` – штрихи переднего/фонового плана (для интерактивной сегментации).  
- Режим редактирования аннотаций (`set_edit_mode(enabled)`) – поддержка перемещения/изменения углов/точек для типов `detect`, `obb`, `segment`.  

**Основные методы:**
- `set_pixmap(pixmap, preserve_view=True)` – отобразить изображение.  
- `set_annotations(annotations, img_width, img_height)` – загрузить аннотации для отображения и редактирования.  
- `set_selected_index(idx)` – выделить аннотацию с индексом `idx`.  
- `set_callbacks(...)` – установить коллбэки для событий рисования и модификации аннотаций.  
- `save_current_image()` – сохранить видимую область в файл.

**Коллбэки (через `set_callbacks`):**
- `on_rect_drawn(rect)` – пользователь нарисовал прямоугольник (координаты в пикселях).  
- `on_scribble_added(x, y, tool)` – добавлена точка штриха (инструмент `fg`/`bg`).  
- `on_annotation_modified(index, new_ann)` – аннотация изменена.  
- `on_display_update()` – запрос на обновление интерфейса.  
- `on_reset_tool()` – сбросить текущий инструмент.  
- `on_selection_changed(index)` – изменилась выделенная аннотация.

---

### 4. Загрузка изображений и аннотаций (модуль `utils.image_io`)

#### `load_images_universal(source, require_annotations, resize_enabled, max_side, progress_callback, parent)`
Универсальная функция загрузки.  
- `source`: список путей к файлам или строка‑путь к папке.  
- `require_annotations`: если `True`, пропускаются изображения без `.txt`‑файла.  
- `resize_enabled`, `max_side`: уменьшать ли большие изображения.  
- Возвращает кортеж:  
  `(image_paths, images, gray_images, annotations_list)`.  
  `images` – список цветных изображений (BGR, uint8),  
  `gray_images` – список оттенков серого,  
  `annotations_list` – список списков аннотаций (каждая аннотация – кортеж, см. ниже).

#### `load_dataset_from_yaml(yaml_path, ...)`
Загружает датасет по YAML‑файлу (формат YOLO). Поддерживает секции `train`, `val`, `test`. Автоматически ищет файлы аннотаций в папках `labels/`.

#### `load_annotations(txt_path, img_w, img_h)`
Загружает аннотации из `.txt` в YOLO‑формате. Возвращает список кортежей, где каждый кортеж имеет вид:
- `('detect', class_id, cx, cy, w, h)` – прямоугольник (нормализованные координаты).  
- `('obb', class_id, [x1,y1,x2,y2,x3,y3,x4,y4])` – oriented bounding box (8 нормализованных координат).  
- `('segment', class_id, [x1,y1,x2,y2,...])` – полигон (чётное количество нормализованных координат).

#### `save_annotations(annotations, txt_path, img_w, img_h)`
Сохраняет аннотации в `.txt` (нормализованные координаты). Поддерживаются все три типа.

**Вспомогательные функции:**
- `read_image_with_fallback` – загрузка изображения (OpenCV → PIL → tifffile).  
- `resize_to_max_side`, `normalize_to_uint8`, `convert_to_grayscale`.  
- `numpy_to_qpixmap` – конвертация numpy (BGR) в QPixmap.

---

### 5. Операции сегментации и отрисовки (модуль `utils_ops.segmentation_ops`)

Функции, используемые на разных вкладках для бинаризации, морфологии и выделения объектов.

#### Бинаризация
- `apply_threshold_method(gray_img, method_name, params)` – применяет один из методов (Simple, Otsu, Triangle, Adaptive, Niblack, Sauvola, ISODATA, Background Symmetry, Row Adaptive). Возвращает `(binary, threshold)`.

#### Морфология
- `apply_morphology(binary, close_factor, open_factor, kernel_shape, img_shape)` – выполняет закрытие и/или открытие. Факторы – относительный размер ядра (доля от минимальной стороны изображения).

#### Выделение объектов
- `segment_contours(img, binary, use_hull, draw)` – возвращает `(img_color, objects)`, где объекты – полигоны (`'segment'`).  
- `segment_projections(img, binary, draw)` – возвращает ограничивающие прямоугольники (`'detect'`).  
- `segment_min_area_rect(img, binary, use_hull, draw)` – возвращает ориентированные прямоугольники (`'obb'`).  
Параметр `draw=False` подавляет рисование на изображении (но объекты всё равно вычисляются).

#### Отрисовка аннотаций
- `draw_selected_objects(img, objects, selected_indices, draw_mode, use_hull, ...)` – рисует выбранные объекты на изображении.  
- `draw_yolo_annotations(img, annotations, color)` – рисует все аннотации (любого типа).  
- `get_display_params(img_shape)` – вычисляет толщину линий и размер шрифта в зависимости от размера изображения и глобальных настроек.  
- `draw_label`, `draw_label_rotated` – утилиты для подписей.

#### Работа со списками объектов
- `format_annotation_display(index, ann, img_w, img_h)` – формирует текстовое описание для QListWidget.  
- `update_annotation_list(list_widget, annotations, img_w, img_h)` – заполняет виджет списком аннотаций.  
- `delete_annotation_by_index(...)` – удаляет аннотацию из списка.

---

### 6. PresetManager (модуль `utils.presets`)

Класс для управления пресетами (сохранёнными настройками вкладки). Каждая вкладка имеет свой экземпляр, но интерфейс унифицирован.

**Основные методы:**
- `get_preset_names()` – возвращает список имён.  
- `apply_preset(name)` – загружает параметры в интерфейс.  
- `add_preset(name)` – сохраняет текущие настройки.  
- `delete_preset(name)` – удаляет пресет.  
- `get_current_settings()` – собирает текущие настройки из UI.

Пресеты хранятся в JSON‑файле в папке `settings/` (имя файла зависит от вкладки).

---

### 7. Глобальные настройки (модуль `utils.settings`)

Объект `settings` (singleton) обеспечивает хранение пользовательских настроек (цвета, толщина линий, масштаб шрифта) и их применение.  
**Сигнал:** `settings_changed(new_settings)` – испускается при изменении настроек, все вкладки подписываются и обновляют отображение.

**Ключевые методы:**
- `get_color(key)` – возвращает цвет в формате BGR (tuple).  
- `get_line_thickness_factor()`, `get_font_scale_factor()` – множители для размеров.

---

### 8. Прочие общие утилиты

- **`utils.ensemble_utils`** – функции для ансамблирования детекций (используются на вкладке «Демонстрация результата»).  
- **`utils.aug_utils`** – аугментация изображений и аннотаций (поворот, масштабирование и т.д.).  
- **`utils.DatasetGenerator`** – поток для генерации датасетов (используется на вкладке «Подготовка датасета»).  
- **`ui.DL_train_settings_*`** – виджеты для настройки обучения нейронных сетей (YOLO, UNet, DeepLabV3, SegFormer, SAM).  
- **`utils_ops.traditional_ml_ops`**, **`utils_ops.deep_learning_ops`** – операции для соответствующих вкладок.

Все эти модули подключаются через `import_libs_internal.py`, который уже импортирует большинство необходимых компонентов.