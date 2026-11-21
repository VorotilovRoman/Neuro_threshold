"""Shared source controls and import dialogs. Parsing stays in utils_new catalogs."""

from pathlib import Path

from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

# Stable action keys; one caption/filter/meaning across every consumer.
ACTIONS = {
    "model_ml": (
        "Модель ML…",
        "Выберите модель классического ML",
        "Модели ML (*.pkl *.joblib)",
        False,
        "Сохранённая модель, совместимая с выбранными признаками и классами.",
    ),
    "images": (
        "Снимки…",
        "Выберите изображения",
        "Изображения (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        False,
        "Выбрать один или несколько снимков. Текущий набор заменяется.",
    ),
    "folder": (
        "Папка снимков…",
        "Выберите папку со снимками",
        "",
        True,
        "Загрузить снимки из выбранной папки. Вложенные папки не включаются, если вкладка явно не указывает рекурсивный поиск.",
    ),
    "labels": (
        "Папка разметки TXT…",
        "Выберите папку разметки TXT",
        "",
        True,
        "Одноимённые TXT: BBox, OBB или Segment. Это папка меток, а не снимков.",
    ),
    "masks": (
        "Папка масок…",
        "Выберите папку индексных масок",
        "",
        True,
        "Индексные маски: целые значения пикселей обозначают классы; RGB-картинка не равна индексной маске.",
    ),
    "yolo_yaml": (
        "YOLO YAML…",
        "Выберите YOLO YAML",
        "YAML (*.yaml *.yml)",
        False,
        "Загрузить снимки, TXT, классы и split из YOLO data.yaml. Текущий набор заменяется.",
    ),
    "coco": (
        "COCO JSON…",
        "Выберите COCO JSON",
        "COCO JSON (*.json)",
        False,
        "Импортировать COCO JSON с категориями и аннотациями; текущий набор заменяется.",
    ),
    "add_coco": (
        "Добавить COCO…",
        "Выберите COCO JSON для добавления",
        "COCO JSON (*.json)",
        False,
        "Добавить COCO к текущему набору; классы объединяются по имени.",
    ),
    "unet_yaml": (
        "YAML с масками…",
        "Выберите YAML датасета с масками",
        "YAML (*.yaml *.yml)",
        False,
        "Загрузить пары снимок–индексная маска и классы из masks_data.yaml. В редакторе маски преобразуются в полигоны с потерей деталей.",
    ),
    "dataset_root": (
        "Папка датасета…",
        "Выберите корневую папку датасета",
        "",
        True,
        "Выбрать корень датасета; ожидаемая структура зависит от выбранной модели или операции.",
    ),
    "reference_folder": (
        "Папка эталона…",
        "Выберите папку правильной ручной разметки",
        "",
        True,
        "Изображения с TXT либо папка только с TXT. Для варианта без снимков нужны размеры из отчётов.",
    ),
    "reports": (
        "Отчёты автоподбора…",
        "Выберите готовые отчёты автоподбора",
        "Отчёты (*_full_report.txt *_gradient_report.txt);;Текст (*.txt)",
        False,
        "Добавить готовые пороговые и градиентные отчёты; это результаты поиска, а не входной датасет.",
    ),
    "report_folder": (
        "Папка отчётов…",
        "Выберите папку готовых отчётов",
        "",
        True,
        "Добавить отчёты из папки и её вложенных каталогов.",
    ),
    "model_yolo": (
        "Модель YOLO…",
        "Выберите модель YOLO",
        "YOLO (*.pt *.onnx *.engine)",
        False,
        "Веса YOLO для инференса. Возможность ONNX/engine зависит от установленного backend.",
    ),
    "model_yolo_train": (
        "Веса YOLO…",
        "Выберите веса YOLO для обучения",
        "PyTorch (*.pt)",
        False,
        "Исходные PT-веса YOLO; экспортированные ONNX/engine не являются обучаемым checkpoint.",
    ),
    "model_torch": (
        "Веса PyTorch…",
        "Выберите веса PyTorch",
        "PyTorch (*.pth *.pt)",
        False,
        "Checkpoint, совместимый с архитектурой и предобработкой выбранной модели.",
    ),
    "model_onnx": (
        "Модель ONNX…",
        "Выберите модель ONNX",
        "ONNX (*.onnx)",
        False,
        "ONNX-модель; проверьте входы, выходы и метаданные предобработки.",
    ),
    "preset": (
        "Импорт пресета…",
        "Выберите пресет",
        "Пресеты обучения (*.txt)",
        False,
        "Текстовый пресет обучения DeepLab: строки «ключ: значение». Пресет не содержит входной датасет.",
    ),
    "class_weights": (
        "Веса классов…",
        "Выберите веса классов",
        "JSON (*.json)",
        False,
        "JSON-массив численных весов классов для функции потерь.",
    ),
    "training_run": (
        "Результаты обучения…",
        "Выберите папку результатов обучения",
        "",
        True,
        "Папка run с графиками, метриками и checkpoint; это не обучающий датасет.",
    ),
}
ORDER = (
    "images",
    "folder",
    "dataset_root",
    "reference_folder",
    "labels",
    "masks",
    "yolo_yaml",
    "coco",
    "add_coco",
    "unet_yaml",
    "reports",
    "report_folder",
    "model_yolo",
    "model_yolo_train",
    "model_torch",
    "model_onnx",
    "model_ml",
    "preset",
    "class_weights",
    "training_run",
)
FORMATS = (
    ("Автоматически", "auto"),
    ("BBox — осевые рамки", "detect"),
    ("OBB — четыре угла", "obb_corners"),
    ("OBB — XYWHR", "obb_xywhr"),
    ("Segment — полигоны", "segment"),
    ("Смешанный формат", "mixed"),
)
DATA_TYPES = (
    ("Разметка TXT: BBox / OBB / Segment", 0),
    ("Индексные маски", 1),
    ("Только снимки", 2),
)
TAB_DESCRIPTIONS = {
    "threshold": "Пороговая сегментация снимков и папок, пиксельная морфология, BBox/OBB/Segment и автоподбор по эталону.",
    "gradient": "Градиентная сегментация снимков и папок, морфология, разметка объектов и автоподбор.",
    "interactive": "Интерактивная сегментация снимков: ROI, штрихи переднего/заднего плана и выбранный алгоритм.",
    "traditional_ml": "Классический ML: обучение на загруженных снимках по соседним YOLO TXT либо явным псевдометкам Otsu; кластеризация без меток; импорт модели.",
    "deep_learning": "Применение моделей сегментации к снимкам и папкам. Загрузите совместимые веса; обучение находится в отдельной вкладке.",
    "reports": "Сравнение пороговых/градиентных отчётов с правильной ручной разметкой: YOLO YAML или папка TXT; рейтинг методов и пресеты.",
    "labeler": "Создание и исправление BBox/OBB/Segment: снимки, TXT, YOLO YAML, COCO, импорт масочного YAML в полигоны.",
    "test": "Просмотр и исправление аннотаций: снимки, TXT, YOLO YAML, COCO, индексные маски и их наложение; поиск перекрывающихся объектов.",
    "dataset": "Подготовка и экспорт датасета: снимки + TXT, индексные маски или снимки без разметки; YOLO YAML, COCO и YAML с масками.",
    "dataset_analysis": "Проверка качества, разметки, повторов и split: TXT/индексные маски/снимки без разметки; общий отчёт и экспорт.",
    "sahi": "Тайловая нарезка датасета из снимков с TXT, YOLO YAML или COCO. Индексные маски как вход здесь не поддерживаются.",
    "yolo_train": "Обучение YOLO, U-Net и DeepLab: папка либо YAML соответствующего формата. SAM/SegFormer остаются заготовками интерфейса.",
    "yolo_demo": "Сравнение YOLO и пороговых пресетов и объединение BBox по классам через NMS/WBF; снимки и папки.",
    "yolo_sort": "Рекурсивный поиск снимков по классу YOLO и геометрии объекта; список, миниатюры и перенос только изображений.",
    "yolo_aeb": "YOLO и AE/VAE: снимки/папки, выбор объекта, кропы, преобразования и карта ошибки реконструкции по checkpoint.",
    "settings": "Глобальные настройки интерфейса, цветов, графики и навигации. Источников датасета на этой вкладке нет.",
}


def make_source_button(key, parent=None):
    caption, _, _, _, hint = ACTIONS[key]
    button = QPushButton(caption, parent)
    button.setObjectName("source_" + key)
    button.setToolTip(hint)
    return button


def annotation_format_combo(parent=None, *, mixed=True):
    combo = QComboBox(parent)
    for caption, value in FORMATS:
        if mixed or value != "mixed":
            combo.addItem(caption, value)
    combo.setToolTip(
        "Auto может быть неоднозначен для четырёхточечного Segment. Выберите формат явно. У Segment чётное число координат, а вся строка YOLO имеет 7, 9, 11… токенов."
    )
    return combo


def get_open_file(owner, key, *, multiple=False, directory=""):
    caption, title, filters, is_folder, _ = ACTIONS[key]
    if is_folder:
        raise ValueError("Используйте get_directory для папки.")
    settings = QSettings("NeuroThreshold", "SourceDialogs")
    initial = directory or str(settings.value(key, ""))
    result = (
        QFileDialog.getOpenFileNames if multiple else QFileDialog.getOpenFileName
    )(owner, title, initial, filters)
    selected = result[0]
    first = selected[0] if multiple and selected else selected if not multiple else ""
    if first:
        settings.setValue(key, str(Path(first).parent))
    return result


def get_directory(owner, key, *, directory=""):
    if not ACTIONS[key][3]:
        raise ValueError("Ожидается источник-папка.")
    settings = QSettings("NeuroThreshold", "SourceDialogs")
    selected = QFileDialog.getExistingDirectory(
        owner, ACTIONS[key][1], directory or str(settings.value(key, ""))
    )
    if selected:
        settings.setValue(key, selected)
    return selected


class DatasetSourceWidget(QGroupBox):
    requested = pyqtSignal(str)

    def __init__(
        self,
        actions=("images", "folder"),
        *,
        types=None,
        formats=False,
        profile=None,
        title="Источник данных",
        parent=None,
    ):
        super().__init__(title, parent)
        self.profile = profile
        self.buttons = {}
        self.actions = tuple(key for key in ORDER if key in actions)
        if set(actions) - set(ACTIONS):
            raise ValueError("Неизвестное действие источника.")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 5, 8, 5)
        grid = QGridLayout()
        self._layout.addLayout(grid)
        for index, key in enumerate(self.actions):
            button = make_source_button(key, self)
            self.buttons[key] = button
            grid.addWidget(button, index // 4, index % 4)
            button.clicked.connect(
                lambda checked=False, key=key: self.requested.emit(key)
            )
        self.type_combo = None
        self.format_combo = None
        if types is not None:
            row = QHBoxLayout()
            row.addWidget(QLabel("Тип данных:"))
            self.type_combo = QComboBox()
            for caption, value in DATA_TYPES:
                if value in types:
                    self.type_combo.addItem(caption, value)
            row.addWidget(self.type_combo, 1)
            self._layout.addLayout(row)
            self.type_combo.currentIndexChanged.connect(
                lambda: self.set_dataset_type(self.type_combo.currentData())
            )
        if formats:
            row = QHBoxLayout()
            row.addWidget(QLabel("Формат TXT:"))
            self.format_combo = annotation_format_combo()
            row.addWidget(self.format_combo, 1)
            self._layout.addLayout(row)
        self.hint = QLabel(TAB_DESCRIPTIONS.get(profile, ""))
        self.hint.setWordWrap(True)
        self.hint.setVisible(bool(self.hint.text()))
        self._layout.addWidget(self.hint)
        if types is not None:
            self.set_dataset_type(self.type_combo.currentData())

    def button(self, key):
        return self.buttons[key]

    def set_dataset_type(self, value):
        if "labels" in self.buttons:
            button = self.buttons["labels"]
            kind = "masks" if value == 1 else "labels"
            button.setText(ACTIONS[kind][0])
            button.setToolTip(ACTIONS[kind][4])
            button.setEnabled(value != 2)
        if self.format_combo is not None:
            self.format_combo.setEnabled(value == 0)

    def add_path_field(self, caption, field):
        row = QHBoxLayout()
        row.addWidget(QLabel(caption))
        row.addWidget(field, 1)
        self._layout.addLayout(row)

    def set_description(self, text):
        self.hint.setText(text)
        self.hint.setVisible(bool(text))


def choose_coco_sources(owner):
    """One COCO file/folder chooser used by editors, preparation, audit and SAHI."""
    from PyQt5.QtWidgets import QMessageBox

    from utils_new.dataset_catalog_new import discover_coco_json

    box = QMessageBox(owner)
    box.setWindowTitle("Источник COCO")
    box.setText("Выберите источник COCO JSON")
    files = box.addButton("Файлы JSON…", QMessageBox.AcceptRole)
    folder = box.addButton("Папка JSON…", QMessageBox.AcceptRole)
    box.addButton("Отмена", QMessageBox.RejectRole)
    box.setDefaultButton(files)
    box.exec_()
    if box.clickedButton() is files:
        return get_open_file(owner, "coco", multiple=True)[0]
    if box.clickedButton() is folder:
        settings = QSettings("NeuroThreshold", "SourceDialogs")
        directory = QFileDialog.getExistingDirectory(
            owner, "Выберите папку COCO JSON", str(settings.value("coco_folder", ""))
        )
        if not directory:
            return []
        settings.setValue("coco_folder", directory)
        try:
            paths = discover_coco_json(directory)
            if not paths:
                QMessageBox.warning(owner, "COCO", "В папке не найдено COCO JSON.")
            return paths
        except (OSError, ValueError) as exc:
            QMessageBox.warning(owner, "COCO", str(exc))
    return []
