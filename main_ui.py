from import_libs_external import *

# Список всех возможных вкладок (внутреннее имя, отображаемое название)
ALL_TABS = [
    ("threshold", "Пороговые методы"),
    ("gradient", "Градиентные методы"),
    ("interactive", "Интерактивные методы"),
    ("traditional_ml", "Традиционные методы ML"),
    ("deep_learning", "Deep Learning"),
    ("reports", "Анализ отчётов"),
    ("labeler", "Добавить аннотации"),
    ("test", "Просмотр аннотаций"),
    ("dataset", "Подготовка датасета"),
    ("yolo_train", "Обучение модели"),
    ("yolo_demo", "Демонстрация результата"),
    ("yolo_sort", "Сортировка с YOLO"),
    ("settings", "Настройки")
]


class StartupDialog(QDialog):
    """Диалог выбора вкладок, которые будут отображаться в главном окне."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор вкладок Threshold‑Researcher")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        layout = QVBoxLayout(self)

        # Группа с чекбоксами
        group = QGroupBox("Выберите вкладки для отображения:")
        group_layout = QVBoxLayout(group)

        self.checkboxes = {}
        for tab_id, tab_name in ALL_TABS:
            cb = QCheckBox(tab_name)
            cb.setObjectName(tab_id)
            self.checkboxes[tab_id] = cb
            group_layout.addWidget(cb)

        # Загружаем сохранённые настройки
        self.load_settings()

        layout.addWidget(group)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Запустить")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.reject)
        self.select_all_btn = QPushButton("Выбрать все")
        self.select_all_btn.clicked.connect(self.select_all)
        self.clear_all_btn = QPushButton("Снять все")
        self.clear_all_btn.clicked.connect(self.clear_all)

        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.clear_all_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def clear_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def load_settings(self):
        """Загружает предыдущий выбор из QSettings."""
        settings = QSettings("ThresholdResearcher", "TabSelector")
        for tab_id, cb in self.checkboxes.items():
            checked = settings.value(f"tab_{tab_id}", True, type=bool)
            cb.setChecked(checked)

    def save_settings(self):
        """Сохраняет текущий выбор в QSettings."""
        settings = QSettings("ThresholdResearcher", "TabSelector")
        for tab_id, cb in self.checkboxes.items():
            settings.setValue(f"tab_{tab_id}", cb.isChecked())

    def get_selected_tabs(self):
        """Возвращает список кортежей (id, название) выбранных вкладок."""
        selected = []
        for tab_id, cb in self.checkboxes.items():
            if cb.isChecked():
                name = next((n for tid, n in ALL_TABS if tid == tab_id), tab_id)
                selected.append((tab_id, name))
        return selected

    def accept(self):
        self.save_settings()
        super().accept()