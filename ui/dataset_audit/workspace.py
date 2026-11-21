"""Task-oriented dataset analysis navigation and overview."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class AnalysisPages(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.navigation = QListWidget()
        self.navigation.setWordWrap(True)
        self.navigation.setMaximumWidth(190)
        self.stack = QStackedWidget()
        layout.addWidget(self.navigation)
        layout.addWidget(self.stack, 1)
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)

    def addItem(self, widget, title):
        labels = {
            "1. Классы, объекты и площади": "Классы и объекты",
            "2. Качество изображений": "Качество снимков",
            "3. Дубликаты и утечка": "Повторы и утечки",
            "4. Тепловая карта объектов": "Геометрия объектов",
            "5. Аудит модели (Loss Inspection) и валидация YOLO": "Проверка моделью",
            "6. Кластеризация эмбеддингов": "Сходство изображений",
            "7. Результаты обучения модели": "Результаты обучения",
            "8. Общий отчёт": "Отчёт",
        }
        self.stack.addWidget(widget)
        self.navigation.addItem(labels.get(title, title))
        if self.navigation.currentRow() < 0:
            self.navigation.setCurrentRow(0)
        return self.stack.count() - 1

    def setCurrentIndex(self, index):
        self.navigation.setCurrentRow(index)

    def currentIndex(self):
        return self.stack.currentIndex()

    def count(self):
        return self.stack.count()


def build_overview(owner):
    page = QWidget()
    layout = QVBoxLayout(page)
    title = QLabel("Проверка датасета перед обучением")
    title.setWordWrap(True)
    layout.addWidget(title)
    intro = QLabel(
        "1. Загрузите источник.  2. Проверьте набор и найденные кандидаты.  3. Сохраните отчёт или экспортируйте выбранные данные."
    )
    intro.setWordWrap(True)
    layout.addWidget(intro)
    owner.workspace_summary = QLabel()
    owner.workspace_summary.setWordWrap(True)
    owner.workspace_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(owner.workspace_summary)
    row = QHBoxLayout()
    row.addWidget(QLabel("Область проверок:"))
    owner.scope_combo = QComboBox()
    owner.scope_combo.addItem("Весь загруженный набор", "all")
    owner.scope_combo.addItem("После фильтров и выбора классов", "filtered")
    owner.scope_combo.setCurrentIndex(1)
    row.addWidget(owner.scope_combo, 1)
    layout.addLayout(row)
    note = QLabel(
        "Сводка классов относится к фильтрам слева. Проверки качества, повторов и геометрии используют выбранную выше область. Пустой результат фильтра не заменяется всем набором."
    )
    note.setWordWrap(True)
    layout.addWidget(note)
    actions = QHBoxLayout()
    owner.btn_run_basic = QPushButton("Запустить базовую проверку")
    owner.btn_stop_basic = QPushButton("Остановить")
    owner.btn_stop_basic.setEnabled(False)
    actions.addWidget(owner.btn_run_basic)
    actions.addWidget(owner.btn_stop_basic)
    layout.addLayout(actions)
    owner.btn_run_basic.setToolTip(
        "Качество снимков, точные/похожие повторы, утечки между split и распределение объектов. Модель и GPU не требуются."
    )
    owner.analysis_status = QLabel("Загрузите набор, затем запустите проверку.")
    owner.analysis_status.setWordWrap(True)
    layout.addWidget(owner.analysis_status)
    owner.analysis_progress = QProgressBar()
    owner.analysis_progress.setVisible(False)
    layout.addWidget(owner.analysis_progress)
    box = QGroupBox("Что проверить и что делать дальше")
    content = QVBoxLayout(box)
    owner.workspace_guidance = QLabel()
    owner.workspace_guidance.setWordWrap(True)
    owner.workspace_guidance.setTextInteractionFlags(Qt.TextSelectableByMouse)
    content.addWidget(owner.workspace_guidance)
    layout.addWidget(box)
    for title, index in (
        ("Открыть качество снимков", 2),
        ("Открыть повторы и утечки", 3),
        ("Открыть итоговый отчёт", 8),
    ):
        button = QPushButton(title)
        button.clicked.connect(
            lambda checked=False, index=index: owner.toolbox.setCurrentIndex(index)
        )
        layout.addWidget(button)
    layout.addStretch()
    owner.toolbox.addItem(page, "Начать и проверить")
    owner.btn_run_basic.clicked.connect(owner.run_basic_analysis)
    owner.btn_stop_basic.clicked.connect(lambda: owner.scans.cancel())
    owner.scope_combo.currentIndexChanged.connect(owner._scope_changed)
