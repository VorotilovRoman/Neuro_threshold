# Threshold‑Researcher

Приложение PyQt5 для сегментации изображений, разметки и подготовки датасетов,
исследования классических методов и работы с нейросетевыми моделями.

## Быстрый запуск

Проверено с Python 3.12. Создайте и активируйте отдельное окружение:

```bash
python -m venv .venv
source .venv/bin/activate
python install.py --torch cpu
python dependency_health.py --load
python main.py
```

В Windows активация: `.venv\Scripts\activate`. Для совместимого GPU явно
выберите `--torch cu126` (проверено пользователем на GTX 1650) или `--torch cu130`
для совместимой системы; автоматического fallback на CPU нет.
Дополнительные возможности: `--extra training`, `--extra audit`, `--extra sahi`,
`--extra export` — флаги можно сочетать.

## OpenCV

Используется один `opencv-python-headless==4.13.0.92`; окна создаёт PyQt5.
Замена на desktop OpenCV не устраняет встречные требования Albumentations.
Сообщение pip check о требовании Ultralytics к `opencv-python` — явное исключение
метаданных; остальные ошибки диагностика не допускает. После ручных изменений
пакетов: `python install.py --repair-opencv`.

[Обоснование выбора](Docs/OpenCV_Decision.md) · [Установка и диагностика](Docs/Dependencies.md)

## Документация и проверки

- [Навигация по Docs](Docs/README.md)
- [Windows: ошибка c10.dll / WinError 1114](Docs/Windows_Torch_DLL.md)
- [Руководство: вкладки, форматы, сохранение](Docs/User_Guide.md)
- [Архитектура](Docs/Architecture.md) · [Аудит импортов](Docs/Import_Audit.md)
- [Общий автоподбор и анализ отчётов](Docs/Unified_Autoresearch_Status.md): пороговые и градиентные методы, px-диапазоны, отмена и сохранение пресетов обеих вкладок.
- [Новый инспектор объектов](Docs/Object_Inspector.md): таблица, поиск, свойства и копирование YOLO; отдельное поле координат удалено.
- [Пять вкладок сегментации](Docs/Segmentation_Unification_Status.md): единые BBox/OBB/Segment, объекты, состояния масок и фоновые задачи.
- [Выполненный план](Docs/Segmentation_Unification_Plan.md): общие UI/утилиты, пресеты v3; оба автоподбора используют общее ядро.
- [Новый просмотрщик демонстрации ансамблей](Docs/Ensemble_View_Migration.md)
- [Удаление неиспользуемых utils](Docs/Utils_Removal_Audit.md)
- [Этапы миграции и отложенный автоподбор](Docs/Segmentation_Migration_Plan.md)

```bash
python -m unittest discover -s tests -v
NEURO_RUN_GUI_TESTS=1 python -m unittest discover -s tests -v
```

Вторая команда включает GUI offscreen, автономные запуски и небольшие проверки
классического ML без скачивания весов. Текущий результат — 85 тестов успешно. В PowerShell сначала задайте `$env:NEURO_RUN_GUI_TESTS="1"`.
Отдельная вкладка: `python -m tabs.threshold_methods` из корня проекта.
