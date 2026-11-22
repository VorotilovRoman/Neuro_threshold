"""Guided comparison page: reference → search reports → comparison → settings."""

import pandas as pd
from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget
from ui.log_widget import LogWidget
from ui.autoreseach_ui.evaluation_controls import hint, setup_evaluation_controls
from ui.autoreseach_ui.result_models import ComparisonTable, IMAGE_COLUMNS
from utils_new.import_export.evaluation_export import preset_settings_text
from utils_new.autoreseach_utils.workflow import result_guidance


class ComparisonResultsWidget(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.result = None
        self._updating = False
        self._recipe_key = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = hint(
            "Здесь появится ответ: какой метод и какие настройки лучше подходят для вашего датасета."
        )
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        self.summary = hint(
            "Сначала выберите эталон и отчёты, затем нажмите «Сравнить методы»."
        )
        layout.addWidget(self.summary)
        self.next_action = hint(
            "Выбор настройки и просмотр снимков помогут понять причины результата."
        )
        layout.addWidget(self.next_action)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Показать:"))
        self.kind_filter = QComboBox()
        for text, value in [
            ("Все методы", "all"),
            ("Пороговые", "threshold"),
            ("Градиентные", "gradient"),
        ]:
            self.kind_filter.addItem(text, value)
        self.kind_filter.setToolTip(
            "Фильтрует только видимые строки; оценка датасета остаётся прежней."
        )
        toolbar.addWidget(self.kind_filter)
        toolbar.addStretch()
        self.selection_hint = hint(
            "Нажмите строку метода или настройки, чтобы увидеть параметры справа."
        )
        toolbar.addWidget(self.selection_hint, 1)
        layout.addLayout(toolbar)
        split = QSplitter(Qt.Horizontal)
        self.tabs = QTabWidget()
        self.methods_table = ComparisonTable()
        self.rank_table = ComparisonTable()
        self.image_table = ComparisonTable(IMAGE_COLUMNS)
        self.tabs.addTab(self.methods_table, "Лучшие настройки методов")
        self.tabs.addTab(self.rank_table, "Все настройки")
        images = QWidget()
        box = QVBoxLayout(images)
        box.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Снимки выбранной настройки:"))
        self.problem_filter = QComboBox()
        for text, value in [
            ("Все снимки", "all"),
            ("С проблемами", "problems"),
            ("Есть лишние объекты", "fp"),
            ("Есть пропуски объектов", "fn"),
            ("Нет вычисления или ошибка", "missing"),
        ]:
            self.problem_filter.addItem(text, value)
        self.problem_filter.setToolTip(
            "Фильтрует только просмотр результатов по снимкам. Не удаляет данные и не меняет рейтинг."
        )
        row.addWidget(self.problem_filter)
        row.addStretch()
        box.addLayout(row)
        box.addWidget(self.image_table)
        self.tabs.addTab(images, "Снимки выбранной настройки")
        split.addWidget(self.tabs)
        self.properties = QTextBrowser()
        self.properties.setPlaceholderText(
            "Полные параметры выбранной настройки и условия её применения."
        )
        self.properties.setMinimumWidth(230)
        split.addWidget(self.properties)
        split.setSizes([850, 300])
        layout.addWidget(split, 1)
        self.metric_hint = hint(
            "F1 — баланс точности и полноты. Лишние — объекты без пары в эталоне; пропущено — объекты эталона, которые метод не нашёл. Наведите указатель на заголовок столбца для пояснения."
        )
        layout.addWidget(self.metric_hint)
        actions = QHBoxLayout()
        self.add_btn = QPushButton("Добавить выбранные настройки в пресеты")
        self.add_btn.setToolTip(
            "Сохранить выбранные полные настройки во вкладках «Пороговые методы» или «Градиентные методы». Несколько строк выбираются с Ctrl/Shift."
        )
        self.owner.btn_export_evaluation = QPushButton("Сохранить сравнение…")
        self.owner.btn_export_evaluation.setToolTip(
            "Сохранить понятную сводку, рейтинг, полные пресеты и результаты по каждому снимку в отдельную пустую папку."
        )
        actions.addWidget(self.add_btn)
        actions.addWidget(self.owner.btn_export_evaluation)
        layout.addLayout(actions)
        self.add_btn.clicked.connect(self._add_selected)
        self.kind_filter.currentIndexChanged.connect(self._refresh_settings)
        self.problem_filter.currentIndexChanged.connect(self._refresh_images)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.methods_table.selectionModel().selectionChanged.connect(
            lambda *args: self._selection_changed(self.methods_table)
        )
        self.rank_table.selectionModel().selectionChanged.connect(
            lambda *args: self._selection_changed(self.rank_table)
        )
        self._last_settings_table = self.methods_table
        self.clear()

    def clear(self):
        self._updating = True
        self.result = None
        self._recipe_key = None
        for table in (self.methods_table, self.rank_table, self.image_table):
            table.set_frame(pd.DataFrame())
        self.properties.clear()
        self.title.setText("Результатов пока нет")
        self.summary.setText(
            "Загрузите ручной эталон и результаты автоподбора, затем сравните методы."
        )
        self.next_action.setText(
            "После сравнения здесь будут лучший метод, полные параметры и ошибки по снимкам."
        )
        self.add_btn.setEnabled(False)
        self._updating = False

    def display_evaluation(self, result):
        self.result = result
        title, body, action = result_guidance(result)
        self.title.setText(title)
        self.summary.setText(body)
        self.next_action.setText(action)
        with QSignalBlocker(self.kind_filter):
            self.kind_filter.setCurrentIndex(0)
        self.tabs.setCurrentIndex(0 if not result.methods.empty else 1)
        self._refresh_settings()

    def _refresh_settings(self, *args):
        if self.result is None or self._updating:
            return
        self._updating = True
        family = self.kind_filter.currentData()
        methods = self.result.methods.copy()
        methods["rank"] = range(1, len(methods) + 1)
        rankings = self.result.rankings
        if family != "all":
            methods = methods[methods.kind == family]
            rankings = rankings[rankings.kind == family]
        self.methods_table.set_frame(methods)
        self.rank_table.set_frame(rankings)
        self.methods_table.sortByColumn(0, Qt.AscendingOrder)
        self.rank_table.sortByColumn(0, Qt.AscendingOrder)
        self._updating = False
        table = self.rank_table if self.tabs.currentIndex() == 1 else self.methods_table
        if table.source.frame.empty and not self.rank_table.source.frame.empty:
            self.tabs.setCurrentIndex(1)
            table = self.rank_table
        self._last_settings_table = table
        if not table.source.frame.empty:
            if not table.select_recipe(self._recipe_key):
                table.selectRow(0)
        else:
            self._recipe_key = None
            self.properties.setPlainText(
                "Для выбранного вида методов результатов нет. Добавьте его отчёты или выберите «Все методы»."
            )
            self._refresh_images()
            self.add_btn.setEnabled(False)

    def _selection_changed(self, table):
        if self._updating or self.result is None:
            return
        selected = table.selected_frame()
        self._last_settings_table = table
        if selected.empty:
            self.add_btn.setEnabled(False)
            return
        row = selected.iloc[0]
        self._recipe_key = row["recipe_key"]
        self.properties.setPlainText(preset_settings_text(row))
        self.selection_hint.setText(
            f"Выбрано настроек: {len(selected)}. Для снимков используется первая выбранная."
        )
        allowed = "recommendable" if "recommendable" in selected else "eligible"
        self.add_btn.setEnabled(
            bool(self.result.reference_objects and selected[allowed].all())
        )
        self._refresh_images()

    def _tab_changed(self, index):
        if self._updating or self.result is None:
            return
        if index in (0, 1):
            table = self.methods_table if index == 0 else self.rank_table
            if table.selected_frame().empty and not table.source.frame.empty:
                table.selectRow(0)
            else:
                self._selection_changed(table)

    def _refresh_images(self, *args):
        if self.result is None or self._recipe_key is None:
            self.image_table.set_frame(pd.DataFrame())
            return
        frame = self.result.details[self.result.details.recipe_key == self._recipe_key]
        mode = self.problem_filter.currentData()
        fp = frame.get("fp", pd.Series(0, index=frame.index)).fillna(0) > 0
        fn = frame.get("fn", pd.Series(0, index=frame.index)).fillna(0) > 0
        if mode == "problems":
            frame = frame[(frame.status != "ok") | fp | fn]
        elif mode == "missing":
            frame = frame[frame.status != "ok"]
        elif mode == "fp":
            frame = frame[fp]
        elif mode == "fn":
            frame = frame[fn]
        self.image_table.set_frame(frame)

    def _add_selected(self):
        if self.result is None:
            return
        selected = self._last_settings_table.selected_frame()
        if selected.empty or not self.result.reference_objects:
            return
        allowed = "recommendable" if "recommendable" in selected else "eligible"
        if selected[allowed].all():
            self.owner.add_to_presets(selected)


def setup_model_research_ui(owner):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    scroll.setWidget(content)
    owner.setCentralWidget(scroll)
    header = QHBoxLayout()
    title = hint("Какие методы и настройки лучше подходят для разметки моего датасета?")
    font = title.font()
    font.setBold(True)
    title.setFont(font)
    header.addWidget(title, 1)
    owner.btn_help = QPushButton("Как пользоваться")
    owner.btn_help.setCheckable(True)
    header.addWidget(owner.btn_help)
    layout.addLayout(header)
    help_box = QGroupBox("Краткая инструкция")
    help_layout = QVBoxLayout(help_box)
    help_layout.addWidget(
        hint(
            "1. Выберите правильную ручную разметку — она будет эталоном.\n2. На вкладках пороговых/градиентных методов выполните автоподбор для этих же изображений, затем добавьте готовые отчёты сюда.\n3. Нажмите «Сравнить методы». F1 оценивает пропущенные и лишние объекты; для победы нужен успешный результат на всех снимках.\n4. Выберите строку метода, изучите настройки и ошибки по снимкам. Добавьте нужный пресет в соответствующую вкладку методов или сохраните сравнение.\nОтчёты partial не подходят. Для папки только с TXT нужны отчёты с исходными размерами. Если методы запускались с разными диапазонами, дополните автоподбор, чтобы сравнить один и тот же пресет на всём эталоне."
        )
    )
    help_box.setVisible(False)
    owner.btn_help.toggled.connect(help_box.setVisible)
    layout.addWidget(help_box)
    setup_evaluation_controls(owner, layout)
    result_box = QGroupBox("4. Вывод, настройки и проверка по снимкам")
    result_layout = QVBoxLayout(result_box)
    owner.results_widget = ComparisonResultsWidget(owner)
    owner.results_widget.setMinimumHeight(350)
    result_layout.addWidget(owner.results_widget)
    layout.addWidget(result_box, 1)
    owner.log_toggle = QCheckBox("Подробности выполнения")
    owner.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    owner.log_widget.setVisible(False)
    owner.log_widget.text.setMaximumHeight(140)
    owner.log_toggle.toggled.connect(owner.log_widget.setVisible)
    layout.addWidget(owner.log_toggle)
    layout.addWidget(owner.log_widget)
