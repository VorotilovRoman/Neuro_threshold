"""Shared source controls and import dialogs. Parsing stays in utils_new catalogs."""

from pathlib import Path

from PyQt5.QtCore import QEvent, QSettings, QSize, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QFileDialog, QHBoxLayout, QLabel, QLayout, QMenu, QPushButton, QSizePolicy, QToolButton, QToolTip, QVBoxLayout, QWidget

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
        "Папка…",
        "Выберите папку со снимками",
        "",
        True,
        "Загрузить снимки из выбранной папки. Вложенные папки не включаются, если вкладка явно не указывает рекурсивный поиск.",
    ),
    "labels": (
        "TXT…",
        "Выберите папку разметки TXT",
        "",
        True,
        "Одноимённые TXT: BBox, OBB или Segment. Это папка меток, а не снимков.",
    ),
    "masks": (
        "Маски…",
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
        "Проверить path, источник TXT и количество файлов перед загрузкой YOLO data.yaml. Снимки без меток сохраняются.",
    ),
    "coco": (
        "COCO…",
        "Выберите COCO JSON",
        "COCO JSON (*.json)",
        False,
        "Проверить пути снимков, используемые COCO JSON и количество файлов перед импортом; текущий набор заменяется.",
    ),
    "add_coco": (
        "+ COCO…",
        "Выберите COCO JSON для добавления",
        "COCO JSON (*.json)",
        False,
        "Добавить COCO к текущему набору; классы объединяются по имени.",
    ),
    "unet_yaml": (
        "Маски YAML…",
        "Выберите YAML датасета с масками",
        "YAML (*.yaml *.yml)",
        False,
        "Проверить пути, источник индексных масок и количество файлов. Снимки без масок сохраняются; в редакторе маски преобразуются в полигоны.",
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
    "sam_tokenizer": (
        "Токенизатор SAM3…",
        "Выберите локальный токенизатор SAM3",
        "BPE GZIP (*.gz)",
        False,
        "Файл bpe_simple_vocab_16e6.txt.gz для текстовых подсказок SAM3; веса выбираются отдельно.",
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
    "sam_tokenizer",
    "model_onnx",
    "model_ml",
    "preset",
    "class_weights",
    "training_run",
)
FORMATS = (
    ("Auto", "auto"),
    ("BBox", "detect"),
    ("OBB: углы", "obb_corners"),
    ("OBB: XYWHR", "obb_xywhr"),
    ("Segment", "segment"),
    ("Mixed", "mixed"),
)
DATA_TYPES = (
    ("TXT", 0),
    ("Маски", 1),
    ("Без разметки", 2),
)
TAB_DESCRIPTIONS = {
    "threshold": "Пороговая сегментация снимков и папок, пиксельная морфология, BBox/OBB/Segment и автоподбор по эталону.",
    "gradient": "Градиентная сегментация снимков и папок, морфология, разметка объектов и автоподбор.",
    "interactive": "Интерактивная сегментация снимков: ROI, штрихи переднего/заднего плана и выбранный алгоритм.",
    "traditional_ml": "Классический ML: обучение на загруженных снимках по соседним YOLO TXT либо явным псевдометкам Otsu; кластеризация без меток; импорт модели.",
    "deep_learning": "Применение моделей сегментации к снимкам и папкам. Загрузите совместимые веса; обучение находится в отдельной вкладке.",
    "reports": "Сравнение пороговых/градиентных отчётов с правильной ручной разметкой: YOLO YAML или папка TXT; рейтинг методов и пресеты.",
    "labeler": "Создание и исправление BBox/OBB/Segment: снимки, TXT, YOLO YAML, COCO и масочный YAML. Авторазметка YOLO или локальные SAM/SAM2/SAM3 по точкам, рамке и тексту SAM3.",
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
    button.setAccessibleName(ACTIONS[key][1])
    button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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


def get_open_file(owner, key, *, multiple=False, directory="", confirm=True):
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
        if confirm and key in ('yolo_yaml', 'unet_yaml', 'coco', 'add_coco'):
            try:
                approved = confirm_dataset_import(owner, selected if multiple else [selected],
                    'coco' if key in ('coco', 'add_coco') else ('unet' if key == 'unet_yaml' else 'yolo'))
            except (OSError, ValueError, TypeError) as exc:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(owner, 'Источник датасета', str(exc))
                approved = []
            return (approved if multiple else (approved[0] if approved else ''), result[1])
    return result


def confirm_dataset_import(owner, paths, kind):
    from ui.dataset_import import confirm_dataset_import as confirm
    return confirm(owner, paths, kind)


def set_dataset_path(edit, path):
    from utils_new.import_export.dataset_sources import original_source
    display = original_source(path)
    edit.setProperty('dataset_effective_path', str(path))
    edit.setProperty('dataset_display_path', display)
    same = edit.text() == display
    edit.setText(display)
    if same:
        edit.textChanged.emit(display)


def dataset_path(edit):
    if edit.text().strip() == edit.property('dataset_display_path'):
        return edit.property('dataset_effective_path') or edit.text().strip()
    return edit.text().strip()


def display_dataset_source(path):
    from utils_new.import_export.dataset_sources import original_source
    return original_source(path)


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


class DatasetSourceWidget(QWidget):
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
        super().__init__(parent)
        self.profile = profile
        self.setAccessibleName(title)
        self.buttons = {}
        self.actions = tuple(key for key in ORDER if key in actions)
        if set(actions) - set(ACTIONS):
            raise ValueError("Неизвестное действие источника.")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._layout.setSizeConstraint(QLayout.SetNoConstraint)
        self._bar = QWidget(self)
        self._row = QHBoxLayout(self._bar)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(4)
        self._row.setSizeConstraint(QLayout.SetNoConstraint)
        self._layout.addWidget(self._bar)
        for key in self.actions:
            button = make_source_button(key, self._bar)
            self.buttons[key] = button
            self._row.addWidget(button)
            button.clicked.connect(
                lambda checked=False, key=key: self.requested.emit(key)
            )
        self.overflow = QToolButton(self._bar)
        self.overflow.setText("Ещё…")
        self.overflow.setToolTip(
            "Остальные источники: кнопки, не поместившиеся в строке."
        )
        self.overflow.setPopupMode(QToolButton.InstantPopup)
        self.overflow.setMenu(QMenu(self.overflow))
        self.overflow.menu().aboutToShow.connect(self._refresh_menu)
        self.overflow.hide()
        self._row.addWidget(self.overflow)
        self._controls = []
        self.type_combo = None
        self.format_combo = None
        if types is not None:
            label = QLabel("Тип:")
            self.type_combo = QComboBox()
            for caption, value in DATA_TYPES:
                if value in types:
                    self.type_combo.addItem(caption, value)
            self.type_combo.setFixedWidth(124)
            self.type_combo.setToolTip(
                "Тип входа: разметка TXT, индексные маски классов или снимки без разметки."
            )
            self._controls.extend((label, self.type_combo))
            self.type_combo.currentIndexChanged.connect(
                lambda: self.set_dataset_type(self.type_combo.currentData())
            )
        if formats:
            label = QLabel("TXT:")
            self.format_combo = annotation_format_combo()
            self.format_combo.setFixedWidth(112)
            self._controls.extend((label, self.format_combo))
        self.help_button = QToolButton(self._bar)
        self.help_button.setText("?")
        self.help_button.setAutoRaise(True)
        self.help_button.setAccessibleName("Назначение вкладки и источники данных")
        self.help_button.clicked.connect(
            lambda: QToolTip.showText(
                self.help_button.mapToGlobal(self.help_button.rect().bottomLeft()),
                self.help_button.toolTip(),
                self.help_button,
            )
        )
        self.set_description(TAB_DESCRIPTIONS.get(profile, ""))
        self._controls.append(self.help_button)
        for control in self._controls:
            self._row.addWidget(control)
        self._row.addStretch()
        self._paths = None
        self._bar.installEventFilter(self)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
        self._fit_buttons(self._bar.width())

    def add_path_field(self, caption, field):
        if self._paths is None:
            self._paths = QHBoxLayout()
            self._paths.setSpacing(4)
            self._layout.addLayout(self._paths)
        self._paths.addWidget(QLabel(caption))
        self._paths.addWidget(field, 1)

    def set_description(self, text):
        self.setToolTip(text)
        self.help_button.setToolTip(text)
        self.help_button.setVisible(bool(text))

    def minimumSizeHint(self):
        controls = [c for c in self._controls if not c.isHidden()]
        width = sum(
            c.width() if c.minimumWidth() == c.maximumWidth() else c.sizeHint().width()
            for c in controls
        )
        width += self.overflow.sizeHint().width() + self._row.spacing() * len(controls)
        if self._paths is not None:
            width = max(width, self._paths.minimumSize().width())
        return QSize(width, self.sizeHint().height())

    def eventFilter(self, watched, event):
        if watched is self._bar and event.type() == QEvent.Resize:
            self._fit_buttons(event.size().width())
        return super().eventFilter(watched, event)

    def _fit_buttons(self, width):
        """Keep a single row; hidden buttons retain their real signals/state."""
        buttons = list(self.buttons.values())
        controls = [c for c in self._controls if not c.isHidden()]
        reserved = sum(
            c.width() if c.minimumWidth() == c.maximumWidth() else c.sizeHint().width()
            for c in controls
        )
        count = len(buttons)
        while count >= 0:
            overflow = count < len(buttons)
            total = reserved + sum(b.sizeHint().width() for b in buttons[:count])
            total += self._row.spacing() * max(
                0, count + len(controls) + int(overflow) - 1
            )
            if overflow:
                total += self.overflow.sizeHint().width()
            if total <= width or count == 0:
                break
            count -= 1
        for index, button in enumerate(buttons):
            button.setVisible(index < count)
        self.overflow.setVisible(count < len(buttons))

    def _refresh_menu(self):
        menu = self.overflow.menu()
        menu.clear()
        for button in self.buttons.values():
            if button.isHidden():
                action = menu.addAction(button.text())
                action.setToolTip(button.toolTip())
                action.setEnabled(button.isEnabled())
                action.triggered.connect(button.click)


def choose_coco_sources(owner):
    """One COCO file/folder chooser used by editors, preparation, audit and SAHI."""
    from PyQt5.QtWidgets import QMessageBox

    from utils_new.import_export.catalog import discover_coco_json

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
            return confirm_dataset_import(owner, paths, 'coco') if paths else []
        except (OSError, ValueError) as exc:
            QMessageBox.warning(owner, "COCO", str(exc))
    return []
