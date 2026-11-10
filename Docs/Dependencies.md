# Зависимости и установка

[Навигация](README.md) · [Все места импортов](Module_Reference.md#внешние-импортируемые-модули)

## Базовая среда

Проверенная среда: Linux x86_64, Python **3.12.14**, PyTorch **2.11.0+cpu**, torchvision **0.26.0+cpu**. Для обычного запуска нужен графический сеанс Qt. Для автоматической проверки можно использовать `QT_QPA_PLATFORM=offscreen`; этот режим не показывает окно пользователю. GPU не обязателен. Минимальная версия Python не закреплена метаданными проекта: прежнее заявление «3.8+» не следует считать проверенным для текущего набора пакетов. Используйте Python 3.12.

`requirements.txt` закрепляет прямые зависимости, но не является полным lockfile транзитивных пакетов и не включает все дополнительные возможности. Qt, NumPy, OpenCV, PyTorch, Ultralytics, ONNX Runtime и другие зависимости импортируются общим агрегатором при запуске, даже если соответствующая DL-вкладка не выбрана.

| Пакет в requirements.txt | Версия | Роль |
| --- | --- | --- |
| numpy | 2.4.3 | Массивы, геометрия, маски |
| opencv-python | 4.13.0.92 | Чтение, обработка, контуры, морфология; import `cv2` |
| scipy | 1.17.1 | Численные алгоритмы и обработка изображений |
| scikit-image | 0.26.0 | Пороговые/интерактивные операции, суперпиксели; import `skimage` |
| scikit-learn | 1.8.0 | ML, кластеризация, метрики; import `sklearn` |
| joblib | 1.5.3 | Сохранение ML-моделей и кэшей |
| networkx | 3.6.1 | Графовые алгоритмы сегментации |
| pandas | 3.0.2 | Таблицы отчётов и статистики |
| tqdm | 4.67.3 | Прогресс операций, в том числе зависимость ML-инструментов |
| requests | 2.33.1 | HTTP-клиент в общем пространстве импортов |
| PyYAML | 6.0.3 | YAML датасетов; import `yaml` |
| Pillow | 12.1.1 | Изображения; import `PIL` |
| torch | 2.11.0+cu130 | Нейросети; в проверенной CPU-среде заменён на 2.11.0+cpu |
| torchvision | 0.26.0+cu130 | Модели и преобразования; CPU-вариант 0.26.0+cpu |
| ultralytics | 8.4.36 | YOLO: обучение, инференс, сортировка, авторазметка |
| onnxruntime | 1.25.1 | ONNX-инференс, базовый пакет без обещания CUDA execution provider |
| albumentations | 2.0.8 | Аугментации |
| PyQt5 | 5.15.11 | Десктопный интерфейс |
| matplotlib | 3.10.8 | Графики, гистограммы, Qt canvas |
| seaborn | 0.13.2 | Визуализация статистики |

`install.py` сначала пытается установить CUDA-сборки из индекса PyTorch, при неудаче вызывает неприкреплённые `torch` и `torchvision`, затем по одному ставит остальные пакеты. Он перехватывает ошибки pip и не гарантирует ненулевой код выхода при неполной установке. Надпись «Готово» не заменяет `pip check` и запуск приложения. Для воспроизводимой CPU-установки используйте следующие команды, не меняющие requirements.txt.

## Установка CPU

Выполняйте из корня репозитория. Пример для Bash; в Windows активируйте `.venv\Scripts\activate` и замените команды подготовки файлов эквивалентным Python-скриптом.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.11.0+cpu' 'torchvision==0.26.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
python - <<'PY'
from pathlib import Path
lines = Path('requirements.txt').read_text().splitlines()
Path('.venv/requirements-cpu.txt').write_text('\n'.join(
    line for line in lines if not line.startswith(('torch==', 'torchvision=='))
) + '\n')
Path('.venv/constraints-cpu.txt').write_text('torch==2.11.0+cpu\ntorchvision==0.26.0+cpu\n')
PY
python -m pip install -r .venv/requirements-cpu.txt -c .venv/constraints-cpu.txt
python -m pip check
python main.py
```

Для GPU выбирайте согласованную пару torch/torchvision под драйвер и платформу по официальным инструкциям PyTorch. Значение `+cu130` требует соответствующего индекса сборок; простой `pip install -r requirements.txt` с PyPI не является универсальной CUDA-установкой. Код поддерживает CPU/CUDA/MPS не во всех компонентах одинаково; наличие пункта в UI не доказывает доступность устройства. Проверьте `torch.cuda.is_available()` или MPS перед длительным обучением.

## Дополнительные зависимости

Они найдены в исходниках, но не закреплены в requirements.txt. Версии для них проект не задаёт, полная совместимость всех сочетаний не проверена. Устанавливайте только нужный набор в тот же venv, затем повторяйте `python -m pip check`.

| Пакет / Python import | Возможность и потребитель | Что происходит без пакета |
| --- | --- | --- |
| `xgboost` | `utils_ops/traditional_ml_ops.py`, классификатор XGBoost | Метод недоступен, остальные ML-модели остаются |
| `segmentation-models-pytorch` / `segmentation_models_pytorch` | U‑Net и DeepLab: `utils_ops/deep_learning_ops.py`, workers обучения | Нельзя загрузить соответствующий segmentor; UNet worker проверяет доступность отдельно |
| `sahi` | `tabs/slice_dataset_sahi.py` | Вкладка открывается, slicing требует пакет |
| `ImageHash` / `imagehash` | `utils/dataset_audit_logic.py`, поиск похожих изображений | Нет перцептивного поиска дубликатов; точное хеширование — отдельная операция |
| `open-clip-torch` / `open_clip` | CLIP-эмбеддинги в аудите и `cluster/extract_embeddings.py` | В аудите предусмотрена попытка fallback на transformers |
| `transformers` | CLIP fallback и отдельные операции извлечения/классификации эмбеддингов в cluster | Нужен для выбранного backend; установка не реализует SegFormer в GUI |
| `umap-learn` / `umap` | Проекция эмбеддингов: аудит и cluster | В соответствующих ветках есть fallback PCA |
| `hdbscan` | Кластеризация эмбеддингов | В аудите fallback DBSCAN; другие CLI проверять отдельно |
| `shapely` | `annotation_processing`, `annotations.geometry`, геометрия генератора | Часть операций использует OpenCV fallback; невыпуклые случаи и точность могут различаться |
| `tifffile` | Чтение сложных TIFF в `dataset_files_new` | Альтернативные читатели доступны, но поддержка конкретного TIFF не гарантирована |
| `segment-anything` / `segment_anything` | `SAMSegmentor` в deep_learning_ops | Импорт проверяется, но загрузка и predict SAM — заглушки; установка пакета их не реализует |
| `onnx` | ONNX export/check в UNet/DeepLab worker | Не требуется для самого ONNX-инференса через onnxruntime; нужен для экспортной ветки |
| `pascal-voc-writer` / `pascal_voc_writer` | `utils_training/yolo2voc.py`, XML-конвертация | CLI выводит инструкцию по установке |
| `psutil` | Мониторинг памяти, выбор batch в workers | Может поступать транзитивно; не следует полагаться на это как на прямую декларацию |

`pip` нужен установщику. `gradio` и `imutils` встречаются в строковом сборе версий `main._save_libs_info`, но не являются основанием для обязательной установки. `dataset_analysis_ui` в fallback-импорте — ошибочный локальный путь, а не пакет PyPI. `utils_new` — локальный namespace package.

Пример установки выбранных возможностей:

```bash
python -m pip install xgboost sahi ImageHash segmentation-models-pytorch
python -m pip check
```

Для CLIP/SAM устанавливайте backend из доверенного источника и учитывайте его собственные требования. Автоматическая установка всех исследовательских backend не нужна для работы базовых вкладок.

## Файлы, сеть и переменные

- Сервер БД, web-сервер и обязательные API-ключи не требуются. Датасеты и веса пользователь выбирает с диска.
- Для установки нужны PyPI и хранилища wheel (`pypi.org`, `files.pythonhosted.org`), CPU/CUDA PyTorch (`download.pytorch.org`, возможный redirect `download-r2.pytorch.org`).
- Ultralytics, SMP/torchvision и CLIP могут скачивать веса: возможны GitHub release assets, PyTorch model storage и Hugging Face. Точный адрес зависит от модели/backend. Успешная установка пакетов не проверяет доступ к этим весам. В ограниченной сети сначала выберите модель и проверьте конкретное скачивание; можно предоставить совместимые файлы локально.
- `COLLECT_LIBS_INFO=1`: записать `libs_info.txt` при закрытии главного окна.
- `QT_QPA_PLATFORM=offscreen`: только для headless-проверок.
- `NO_ALBUMENTATIONS_UPDATE=1`: отключает проверку обновлений Albumentations.
- `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `YOLO_CONFIG_DIR`: задавайте доступные для записи каталоги в ограниченной среде. Они не перенаправляют JSON из `settings/` проекта.

## Что проверено

В текущей Linux CPU-среде установлены базовые зависимости с CPU-парой torch/torchvision. `pip check` прошёл. Все **16** вкладок по отдельности созданы и показаны в Qt offscreen без загрузки весов; это проверка инициализации, не всех кнопок. Проверены загрузка/отображение `Docs/test_image.png`, пять пороговых и два градиентных метода и CPU-операция torch. Проверки roundtrip геометрии, кодирования масок, resize и генерации YOLO/mask/YAML/COCO на искусственном примере также прошли и описаны в [руководстве](User_Guide.md#проверка-работоспособности).

Полное обучение, качество нейросетевого инференса, экспорт всех архитектур, GPU/MPS и интерактивная работа мышью не проверены. Ошибка сохранения облачного черновика относится к платформе среды, а не к зависимости приложения; инструкции установки выше не требуют этого API.
