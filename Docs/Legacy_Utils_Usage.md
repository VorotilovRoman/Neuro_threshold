# Оставшиеся потребители старой папки utils

Проверка по AST импортов и строковым записям реестра. Дата: 21 ноября 2026.

В utils осталось два Python-файла. Оба используются; удалять их сейчас нельзя.

## Прямые импорты

| Файл | Строка | Старый модуль |
| --- | --- | --- |
| `tabs/demo_ensemble_methods.py` | 34 | `utils.ensemble_utils` |
| `tabs/demo_ensemble_methods.py` | 38 | `utils.image_io` |
| `tabs/yolo_find_img.py` | 40 | `utils.image_io` |
| `tabs/yolo_test_aeb.py` | 53 | `utils.image_io` |

## Вкладки и зависимости

| Вкладка | Путь зависимости |
| --- | --- |
| `tabs.demo_ensemble_methods` | `tabs.demo_ensemble_methods` → `utils.ensemble_utils` |
| `tabs.demo_ensemble_methods` | `tabs.demo_ensemble_methods` → `utils.image_io` |
| `tabs.yolo_find_img` | `tabs.yolo_find_img` → `utils.image_io` |
| `tabs.yolo_test_aeb` | `tabs.yolo_test_aeb` → `utils.image_io` |

Другие вкладки не имеют пути к utils по статическим Python-импортам. Строковые записи ленивого реестра перечислены отдельно: запись ещё не означает загрузку модуля.

## Ленивый реестр

| Файл | Строка | Модуль |
| --- | --- | --- |
| `import_libs_internal.py` | 24 | `utils.image_io` |
| `import_libs_internal.py` | 25 | `utils.image_io` |
| `import_libs_internal.py` | 26 | `utils.image_io` |
| `import_libs_internal.py` | 27 | `utils.image_io` |
| `import_libs_internal.py` | 28 | `utils.image_io` |
| `import_libs_internal.py` | 29 | `utils.image_io` |
| `import_libs_internal.py` | 141 | `utils.image_io` |
| `import_libs_internal.py` | 148 | `utils.image_io` |
| `import_libs_internal.py` | 149 | `utils.image_io` |
| `import_libs_internal.py` | 161 | `utils.ensemble_utils` |
| `import_libs_internal.py` | 164 | `utils.ensemble_utils` |
| `import_libs_internal.py` | 169 | `utils.ensemble_utils` |
| `import_libs_internal.py` | 179 | `utils.image_io` |
| `import_libs_internal.py` | 180 | `utils.image_io` |
| `import_libs_internal.py` | 181 | `utils.image_io` |
| `import_libs_internal.py` | 182 | `utils.image_io` |
| `import_libs_internal.py` | 183 | `utils.image_io` |
| `import_libs_internal.py` | 191 | `utils.image_io` |
| `import_libs_internal.py` | 208 | `utils.image_io` |
| `import_libs_internal.py` | 209 | `utils.image_io` |
| `import_libs_internal.py` | 210 | `utils.image_io` |
| `import_libs_internal.py` | 211 | `utils.image_io` |
| `import_libs_internal.py` | 212 | `utils.ensemble_utils` |
| `import_libs_internal.py` | 213 | `utils.image_io` |
| `import_libs_internal.py` | 214 | `utils.image_io` |

## Следующие переносы

- `yolo_find_img`: нормализация и чтение снимков → dataset_files_new; отдельно проверить специализированный fallback поиска.
- `yolo_test_aeb`: LazyImageCache/read и Qt-конвертация → dataset_files_new + ui/image_conversion.
- `demo_ensemble_methods`: loader Qt-контракта и ensemble_utils; разделить вычисления ансамбля и GUI, затем проверить веса/координаты/NMS/WBF.
- После переноса актуализировать import_libs_internal и повторно запустить этот скрипт. Импорты utils_ops и utils_aeb относятся к другим папкам и в эту статистику не включены.
