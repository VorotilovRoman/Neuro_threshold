# Скрипты кластеризации объектов

[README проекта](../README.md)

Экспериментальный CLI-конвейер: **YOLO → кропы одного класса → выравнивание →
DINOv2 → кластеризация**. Эти скрипты не являются вкладкой приложения и частично
дублируют чтение снимков, работу моделей и выравнивание из `utils_new`.
Имена кластеров назначайте после ручной проверки; номер группы не является оценкой дефекта.

## Скрипты

| Файл | Назначение | Основной результат |
| --- | --- | --- |
| [save_yolo_object_to_path.py](save_yolo_object_to_path.py) | Рекурсивный YOLO-поиск выбранного класса, conf/размер/отступ; ручные TXT не читаются | Кропы объектов |
| [canonicalize_crops.py](canonicalize_crops.py) | Повороты/отражения и корреляция с эталоном | Выровненные копии, `alignment_log.csv`, коллаж |
| [canonicalize_crops_method2.py](canonicalize_crops_method2.py) | Альтернатива: SIFT/AKAZE/ORB + RANSAC либо корреляция | Копии, журнал метода/угла и коллаж |
| [extract_embeddings.py](extract_embeddings.py) | DINOv2, при необходимости TTA | `*_embeding.txt` рядом с кропами; написание суффикса соответствует коду |
| [cluster_embeddings.py](cluster_embeddings.py) | PCA/UMAP, HDBSCAN/K-Means, назначение новых векторов по центроидам | CSV, график, коллажи, `centroids.npy`, `model.json`, `reducer.joblib` |
| [test_new_image.py](test_new_image.py) | YOLO → базовое выравнивание → DINOv2 → группа для новых полных снимков | `results.csv`, коллажи, опционально кропы/рамки |

Метка `-1` означает шум/непринятый объект. ID групп локальны для состояния
и класса YOLO. Для каждого класса используйте отдельные эталон и папку результатов.

## Пример из корня проекта

Замените пути и ID класса своими. Нужны рабочие веса YOLO и доступ к DINOv2
при первом запуске либо заранее подготовленный кеш `torch.hub`.

```powershell
python install.py --extra audit
python cluster/save_yolo_object_to_path.py -i "D:/dataset/images/train" -o "D:/cluster/class0/crops" -m "D:/models/best.pt" --class-id 0 --conf 0.7 --imgsz 1280 --padding -10
python cluster/canonicalize_crops.py -i "D:/cluster/class0/crops" -o "D:/cluster/class0/aligned" -r "D:/cluster/class0/reference.png" --angles 0,90,180,270
python cluster/extract_embeddings.py -i "D:/cluster/class0/aligned" --model dinov2_vits14 --rotation-mode none --batch-size 8 --device auto
python cluster/cluster_embeddings.py -i "D:/cluster/class0/aligned" -o "D:/cluster/class0/state" --reduce-method umap --reduce-dim 20 --method auto
```

Вместо базового выравнивания можно исследовать `canonicalize_crops_method2.py`
с `--match-method auto --feature-detector sift`. Перед дальнейшим применением
проверьте совместимость этой предобработки с `test_new_image.py`.
Положительный padding расширяет кроп, отрицательный сужает.
Существующие выровненные копии пропускаются без `--overwrite`.

Просмотрите `labels.csv`, `summary.csv` и коллажи. При недостаточной выборке
уменьшите число компонент/кластеров; для явного K-Means используйте `--method kmeans -k 4`.
Вложенные папки кластеризатор включает с `--recursive`. Файлы по папкам групп
не перекладываются: результат представлен CSV и коллажами.

Для новых **эмбеддингов**, рассчитанных тем же конвейером:

```powershell
python cluster/cluster_embeddings.py --predict -i "D:/cluster/new_aligned" -m "D:/cluster/class0/state" -o "D:/cluster/predictions" --make-montages
```

Для новых **полных снимков** задайте `target_classes` и `class_configs` в
`test_new_image.py`: ID YOLO, эталон, папку состояния, DINOv2, padding и выравнивание.

```powershell
python cluster/test_new_image.py -i "D:/new_images" -m "D:/models/best.pt" -o "D:/cluster/results" --conf 0.7 --imgsz 1280
```

UMAP/HDBSCAN и остальные дополнительные зависимости устанавливаются через extra audit.
При отсутствии UMAP возможен PCA; `auto` пробует HDBSCAN, затем K-Means.
TTA увеличивает нагрузку на память; уменьшайте batch при OOM.

## Ограничения текущего кода

- `test_new_image.py` не повторяет features-выравнивание `method2` и TTA.
  Настройки подготовки и применения нужно согласовать вручную.
- После PCA/UMAP векторы не нормализуются перед произведением с центроидами:
  `cosine_sim`/`similarity` нельзя считать калиброванной cosine-оценкой.
- Сохраняются центроиды, не обученные K-Means/HDBSCAN. Назначение нового объекта
  может отличаться от исходного алгоритма; у `--predict` нет порога отклонения.
- В состоянии не зафиксирован весь конвейер: версии/предобработка DINOv2, TTA,
  эталон и выравнивание. После смены reducer используйте новую папку состояния,
  чтобы не подхватить оставшийся `reducer.joblib`.
- Корреляционные режимы/TTA рассчитаны на углы, кратные 90°; произвольный угол
  поддерживает features-ветка. Настройка 45° не делает базовый поворот на 45°.
- В `CONFIG` встречаются повторяющиеся ключи и локальные пути; действует последнее
  значение. CLI не переопределяет все настройки `test_new_image.py`.
- Отладочные коллажи удерживают кропы в памяти; одинаковые имена в подпапках
  могут конфликтовать. Валидация малых выборок, размерностей и NaN неполная.

Эти ограничения документированы, но в рамках обновления документации не исправлялись.
Для проверок используйте отдельные входные/выходные папки.
