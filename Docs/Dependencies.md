# Зависимости, установка и диагностика

[Навигация](README.md) · [Архитектура](Architecture.md) · [План миграции](Import_Architecture_Plan.md)

## Окружение и установка

Проверенная версия Python — 3.12, Linux x86_64, Torch CPU. Закреплённый набор
современных пакетов не рассчитан на Python 3.8. Используйте отдельный venv;
установщик отказывается изменять глобальный Python.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python install.py --torch cpu
python dependency_health.py --load
python main.py
```

В Windows активация: `.venv\Scripts\activate`. Для совместимого GPU выберите
`python install.py --torch cu130`. Профиль задаётся явно; скрытого fallback на
CPU или неприкреплённые torch/torchvision нет. CUDA/MPS в этой миграции не
проверялись. Наличие пункта UI не доказывает доступность ускорителя.

`install.py` ставит согласованную пару одним вызовом pip, затем общий набор и
выбранные extras одним resolver-вызовом, нормализует поставщика cv2 и проверяет
итог в новом процессе. Ошибка pip или диагностики даёт ненулевой код выхода.
Скрипт работает независимо от текущего каталога; используется текущий Python.

```bash
python install.py --dry-run --torch cpu --extra training
python install.py --torch cpu --extra training --extra export
python install.py --torch cpu --extra audit --extra sahi
```

## Один поставщик cv2 и отдельный Qt

[Сравнение вариантов и обоснование решения](OpenCV_Decision.md).

Все четыре wheels `opencv-python`, `opencv-python-headless`,
`opencv-contrib-python`, `opencv-contrib-python-headless` предоставляют общий
модуль `cv2` и пересекающиеся файлы. Их совместная установка недопустима даже
при совпадающих версиях: удаление одного может удалить файлы другого.

Выбран **opencv-python-headless 4.13.0.92**. Он поддерживает используемые
вычисления OpenCV, а окна приложения создаёт **PyQt5**. Вызовов HighGUI
(`cv2.imshow`, `cv2.waitKey`) в проекте нет; отдельная сборка Qt из desktop
OpenCV не нужна и может конфликтовать с plugins PyQt5.

Ultralytics 8.4.36 требует именно дистрибутив `opencv-python` в метаданных,
хотя использует доступный в headless API `cv2`. Общий resolver поэтому может
временно установить оба wheel. Установщик после этого удаляет конкурентов и
обязательно переустанавливает выбранный wheel с `--no-deps --force-reinstall`,
восстанавливая общие файлы. Он не импортирует нативные модули до завершения
установки. Не запускайте приложение и не устанавливайте пакеты одновременно.

Команда ремонта всегда принудительно восстанавливает закреплённый wheel,
даже если его метаданные присутствуют. Обычная установка также исправляет
неверную версию единственного поставщика.

После любых ручных изменений зависимостей:

```bash
python install.py --repair-opencv
python dependency_health.py
python dependency_health.py --load
```

Обычный `python -m pip check` сообщает:

```text
ultralytics 8.4.36 requires opencv-python, which is not installed.
```

Это **явное исключение метаданных**, а не чистый результат pip check.
`dependency_health.py` допускает только эту конкретную недостающую зависимость
Ultralytics при наличии единственного headless-поставщика. Любые прочие
несовместимости, отсутствие других библиотек или несколько поставщиков — ошибка.
Полной согласованности с метаданными сторонних wheels такая политика не обещает;
не следует устанавливать второй wheel лишь ради устранения сообщения pip check.

Ultralytics AutoInstall отключён через `runtime_policy.configure_ultralytics()`
во всех путях приложения, workers и CLI. Проверки backend не должны менять
venv во время инференса или возвращать второй wheel OpenCV. Если дополнительный
backend нужен, установите его явно и повторите нормализацию/диагностику.

## Прямые версии и профили

`requirements.txt` фиксирует прямые версии, но не является полным lockfile
транзитивных зависимостей. `requirements/constraints.txt` удерживает пару Torch
при добавлении extras и версию временного OpenCV wheel. CPU/CUDA wheels указаны
отдельно в `requirements/torch-cpu.txt` и `requirements/torch-cu130.txt`.

| Пакет | Версия | Назначение |
| --- | --- | --- |
| numpy | 2.4.3 | Массивы |
| opencv-python-headless | 4.13.0.92 | Изображения и геометрия без собственного Qt |
| scipy | 1.17.1 | Численные операции |
| scikit-image | 0.26.0 | Сегментация |
| scikit-learn | 1.8.0 | Классическое ML |
| joblib | 1.5.3 | Сохранение моделей |
| networkx | 3.6.1 | Графовые методы |
| pandas | 3.0.2 | Табличные отчёты |
| tqdm | 4.67.3 | Прогресс CLI |
| requests | 2.33.1 | HTTP |
| PyYAML | 6.0.3 | YAML |
| Pillow | 12.1.1 | Чтение изображений |
| torch / torchvision | 2.11.0 / 0.26.0 | Пара с суффиксом +cpu или +cu130 выбранного профиля |
| ultralytics | 8.4.36 | YOLO |
| onnxruntime | 1.25.1 | ONNX CPU-инференс; CUDA provider не обещается |
| albumentations | 2.0.8 | Аугментации изображений/масок |
| PyQt5 | 5.15.11 | GUI |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | Графики |

Обычный `pip install -r requirements.txt` не выполняет нормализацию cv2 и не
выбирает явный Torch backend. Предпочтителен `install.py`; после ручного pip
нужен `--repair-opencv` и диагностика.

## Дополнительные возможности

| Extra | Пакеты | Возможности |
| --- | --- | --- |
| `training` | segmentation-models-pytorch, psutil | U-Net/DeepLab, мониторинг обучения |
| `audit` | ImageHash, open-clip-torch, transformers, umap-learn, hdbscan, xgboost | Аудит, эмбеддинги, исследовательское ML |
| `sahi` | sahi | Нарезка датасета |
| `export` | onnx, tifffile, shapely, pascal-voc-writer | Экспорт моделей/разметки, TIFF, геометрия |

Extras декларированы в отдельных файлах, их транзитивные версии не заперты.
Полная совместимость всех optional backend и весов не проверена. Доступность
через `find_spec` означает наличие пакета, не успешную загрузку модели. Отсутствие
optional пакета не мешает базовому запуску. UNet/DeepLab workers обрабатывают
`--help` до загрузки SMP; обучение требует SMP. SAM/SegFormer в GUI по-прежнему
имеют незавершённые ветки; установка дополнительных пакетов их не реализует.

## Сеть и переменные

- Для установки нужны PyPI/wheel storage и `download.pytorch.org` с возможными redirects.
- Ultralytics, SMP/torchvision и CLIP могут скачивать веса с GitHub, PyTorch
  storage и Hugging Face. Успешный импорт не подтверждает доступность весов.
- `COLLECT_LIBS_INFO=1` сохраняет версии через metadata, не загружая ML при
  закрытии главного окна; CUDA опрашивается только если Torch уже загружен.
- `QT_QPA_PLATFORM=offscreen` — только для автоматических GUI-проверок.
- `NO_ALBUMENTATIONS_UPDATE=1` отключает проверку обновлений Albumentations.
- `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `YOLO_CONFIG_DIR` позволяют
  выбрать доступные каталоги кешей. JSON в `settings/` проекта остаются отдельно.

## Проверки

```bash
python -m unittest discover -s tests -v
NEURO_RUN_GUI_TESTS=1 python -m unittest discover -s tests -v
python dependency_health.py --load
```

Проверяются границы импортов, отсутствие циклов верхнего уровня, отдельные и
совместные вкладки, прямой/модульный запуск и CLI workers без обучения. Подробнее
о результатах и ограничениях — в [плане](Import_Architecture_Plan.md).
GPU/MPS, качество обучения и инференс с реальными весами эти тесты не покрывают.
