from import_libs_internal import *


class ModelResearchWidget(QWidget):
    """Виджет для отображения результатов анализа модели."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.top5_df = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier New", 10))
        layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Добавить в пресеты")
        self.add_btn.setToolTip("Добавить выбранные топ комбинаций в список пресетов")
        self.add_btn.clicked.connect(self.add_to_presets)
        btn_layout.addWidget(self.add_btn)

        self.save_btn = QPushButton("Сохранить топ-пресеты")
        self.save_btn.setToolTip("Сохранить текущий список топ-пресетов в файл (CSV или Pickle)")
        self.save_btn.clicked.connect(self.save_top_presets)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

    def save_top_presets(self):
        if self.top5_df is None or self.top5_df.empty:
            QMessageBox.warning(self, "Нет данных", "Нет данных для сохранения.")
            return
        if self.parent_app:
            self.parent_app.save_dataframe(self.top5_df, "Сохранить топ-пресеты")
        else:
            print("No parent app")

    def display_results(self, df):
        self.top5_df = df
        if df is None or df.empty:
            self.text_edit.clear()
            self.text_edit.append("Нет данных для отображения.")
            return
        text = "=== ТОП пресетов ===\n\n"
        if 'rank' in df.columns:
            df = df.sort_values('rank')
        for idx, row in df.iterrows():
            text += f"Ранг: {row.get('rank', idx + 1)}\n"
            text += f"Метод: {row['method']}\n"
            text += f"Параметры: {row['params']}\n"
            text += f"Инверсия: {'Да' if row['invert'] else 'Нет'}\n"
            text += f"Close factor: {row['close_factor']:.2f}\n"
            text += f"Open factor: {row['open_factor']:.2f}\n"
            if 'actual_f1' in row and not pd.isna(row['actual_f1']):
                text += f"Реальный F1: {row['actual_f1']:.6f}\n"
            if 'pred_f1' in row and not pd.isna(row['pred_f1']):
                text += f"Предсказанный F1: {row['pred_f1']:.6f}\n"
            if 'mean_f1' in row and not pd.isna(row['mean_f1']):
                text += f"Mean F1: {row['mean_f1']:.6f}\n"
            text += "-" * 50 + "\n"
        self.text_edit.setText(text)

    def add_to_presets(self):
        """Вызывает метод добавления пресетов у родительского окна"""
        if self.top5_df is None or self.top5_df.empty:
            QMessageBox.warning(self, "Нет данных", "Нет данных для добавления в пресеты.")
            return
        if self.parent_app and hasattr(self.parent_app, 'add_to_presets'):
            self.parent_app.add_to_presets(self.top5_df)
        else:
            print("No parent app or missing add_to_presets method")


def setup_model_research_ui(parent):
    """Настройка UI для ModelResearchApp."""
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)

    # ========== ПЕРВЫЙ ГОРИЗОНТАЛЬНЫЙ БЛОК (загрузка/сохранение и отображение выборок) ==========
    top_row_layout = QHBoxLayout()

    # ---- Левый блок: загрузка и сохранение датасета ----
    left_load_group = QGroupBox("Загрузка / сохранение датасета")
    left_load_layout = QHBoxLayout()
    parent.btn_select = QPushButton("Выбрать папку")
    parent.btn_select.setToolTip("Выбрать папку с отчетами (*_full_report.txt) и аннотациями (*.txt)")
    parent.btn_load = QPushButton("Загрузить датасет")
    parent.btn_load.setToolTip("Загрузить ранее сохраненный датасет (CSV или Pickle)")
    parent.btn_save = QPushButton("Сохранить датасет")
    parent.btn_save.setToolTip("Сохранить текущий датасет в файл (CSV или Pickle)")
    parent.btn_save.setEnabled(False)

    left_load_layout.addWidget(parent.btn_select)
    left_load_layout.addWidget(parent.btn_load)
    left_load_layout.addWidget(parent.btn_save)
    left_load_group.setLayout(left_load_layout)

    # ---- Правый блок: отображение выборок ----
    right_view_group = QGroupBox("Отображение выборок")
    right_view_layout = QHBoxLayout()
    parent.btn_first = QPushButton("Первые 100")
    parent.btn_first.setToolTip("Показать первые 100 строк датасета")
    parent.btn_last = QPushButton("Последние 100")
    parent.btn_last.setToolTip("Показать последние 100 строк датасета")
    parent.btn_random = QPushButton("Случайные 100")
    parent.btn_random.setToolTip("Показать 100 случайных строк датасета")
    parent.btn_save_sample = QPushButton("Сохранить отображаемые")
    parent.btn_save_sample.setToolTip("Сохранить текущий отображаемый датасет (например, отфильтрованный) в файл")
    parent.btn_all = QPushButton("Показать всё")
    parent.btn_all.setToolTip("Показать весь датасет (осторожно, если много строк)")

    parent.btn_first.setEnabled(False)
    parent.btn_last.setEnabled(False)
    parent.btn_random.setEnabled(False)
    parent.btn_save_sample.setEnabled(False)
    parent.btn_all.setEnabled(False)

    right_view_layout.addWidget(parent.btn_first)
    right_view_layout.addWidget(parent.btn_last)
    right_view_layout.addWidget(parent.btn_random)
    right_view_layout.addWidget(parent.btn_save_sample)
    right_view_layout.addWidget(parent.btn_all)
    right_view_group.setLayout(right_view_layout)

    top_row_layout.addWidget(left_load_group)
    top_row_layout.addWidget(right_view_group)
    main_layout.addLayout(top_row_layout)

    # ========== ВТОРОЙ ГОРИЗОНТАЛЬНЫЙ БЛОК (фильтрация и подготовка) ==========
    second_row_layout = QHBoxLayout()

    # ---- Левый блок: фильтрация датасета (кнопки в 2 столбца) ----
    left_filter_group = QGroupBox("Фильтрация датасета")
    left_filter_layout = QVBoxLayout()

    # Сетка 2x3 для кнопок фильтрации
    filter_grid = QGridLayout()
    filter_grid.setSpacing(10)

    parent.btn_filter_zero_true = QPushButton("Удалить: num_objects=0 и true>=1")
    parent.btn_filter_zero_true.setToolTip("Удалить строки, где нет предсказанных объектов, но есть 1 и более истинных")
    parent.btn_filter_overseg = QPushButton("Удалить: pred > 2*true")
    parent.btn_filter_overseg.setToolTip(
        "Удалить строки с сильной oversegmentation (предсказаний больше чем в 3 раза больше истинных)")
    parent.btn_filter_duplicates = QPushButton("Удалить дубликаты строк")
    parent.btn_filter_duplicates.setToolTip("Удалить полностью дублирующиеся строки")
    parent.btn_filter_invalid_params = QPushButton("Удалить некорректные параметры")
    parent.btn_filter_invalid_params.setToolTip("Удалить строки с некорректными (нечисловыми) параметрами бинаризации")
    parent.btn_filter_anomaly_count = QPushButton("Удалить: num>15 (объектов или истинных)")
    parent.btn_filter_anomaly_count.setToolTip(
        "Удалить строки, где количество объектов или истинных объектов превышает 15")
    parent.btn_filter_full_image = QPushButton("Удалить: объект на весь снимок")
    parent.btn_filter_full_image.setToolTip(
        "Удалить строки, где среди предсказанных объектов есть объект, покрывающий почти всё изображение (центр 0.5,0.5 и размеры 1,1)")

    # Устанавливаем enabled состояние (по умолчанию выключены)
    parent.btn_filter_zero_true.setEnabled(False)
    parent.btn_filter_overseg.setEnabled(False)
    parent.btn_filter_duplicates.setEnabled(False)
    parent.btn_filter_invalid_params.setEnabled(False)
    parent.btn_filter_anomaly_count.setEnabled(False)
    parent.btn_filter_full_image.setEnabled(False)

    # Размещаем кнопки в сетке 2x3
    filter_grid.addWidget(parent.btn_filter_zero_true, 0, 0)
    filter_grid.addWidget(parent.btn_filter_overseg, 0, 1)
    filter_grid.addWidget(parent.btn_filter_duplicates, 1, 0)
    filter_grid.addWidget(parent.btn_filter_invalid_params, 1, 1)
    filter_grid.addWidget(parent.btn_filter_anomaly_count, 2, 0)
    filter_grid.addWidget(parent.btn_filter_full_image, 2, 1)

    left_filter_layout.addLayout(filter_grid)

    # Кнопка удаления строк без истинных объектов (num_true_objects == 0)
    parent.btn_remove_no_true = QPushButton("Удалить строки без истинных объектов (num_true_objects == 0)")
    parent.btn_remove_no_true.setToolTip("Удалить все строки, где нет ни одного истинного объекта (пустые изображения)")
    parent.btn_remove_no_true.setEnabled(False)
    left_filter_layout.addWidget(parent.btn_remove_no_true)

    # Строка со случайной выборкой
    sample_layout = QHBoxLayout()
    parent.sample_label = QLabel("Случайная выборка:")
    parent.sample_spin = QSpinBox()
    parent.sample_spin.setRange(1, 1000000)
    parent.sample_spin.setValue(10000)
    parent.sample_spin.setToolTip("Количество строк для случайной выборки")
    parent.btn_sample_n = QPushButton("Выбрать N строк")
    parent.btn_sample_n.setToolTip("Оставить в датасете только N случайных строк")
    parent.btn_sample_n.setEnabled(False)
    sample_layout.addWidget(parent.sample_label)
    sample_layout.addWidget(parent.sample_spin)
    sample_layout.addWidget(parent.btn_sample_n)
    sample_layout.addStretch()
    left_filter_layout.addLayout(sample_layout)

    # Счётчик строк
    parent.lbl_row_count = QLabel("Строк в датасете: 0")
    left_filter_layout.addWidget(parent.lbl_row_count)

    left_filter_group.setLayout(left_filter_layout)

    # ---- Правый блок: подготовка датасета и анализ ----
    right_prepare_group = QGroupBox("Подготовка датасета и анализ")
    right_prepare_layout = QVBoxLayout()

    # Первая строка: сохранение уникальных параметров и кластеризация
    prepare_row1 = QHBoxLayout()
    parent.btn_save_unique_params = QPushButton("Сохранить уникальные params")
    parent.btn_save_unique_params.setToolTip(
        "Сохранить все уникальные строки params для каждого метода в текстовый файл")
    parent.btn_cluster = QPushButton("Кластеризация параметров")
    parent.btn_cluster.setToolTip(
        "Добавить колонку 'cluster' с меткой кластера для каждой строки (на основе параметров)")
    parent.btn_cluster_co = QPushButton("Кластеризация open/close")
    parent.btn_cluster_co.setToolTip(
        "Добавить колонку 'cluster_co' – кластеризация open_factor и close_factor для каждого cluster каждого метода")
    parent.btn_save_unique_params.setEnabled(False)
    parent.btn_cluster.setEnabled(False)
    parent.btn_cluster_co.setEnabled(False)

    prepare_row1.addWidget(parent.btn_save_unique_params)
    prepare_row1.addWidget(parent.btn_cluster)
    prepare_row1.addWidget(parent.btn_cluster_co)
    prepare_row1.addStretch()
    right_prepare_layout.addLayout(prepare_row1)

    # Вторая строка: расчёт meanAP и фильтр по mean_f1
    prepare_row2 = QHBoxLayout()
    parent.iou_label = QLabel("IoU порог (для f1_50):")
    parent.iou_spin = QDoubleSpinBox()
    parent.iou_spin.setRange(0.0, 1.0)
    parent.iou_spin.setSingleStep(0.05)
    parent.iou_spin.setValue(0.5)
    parent.iou_spin.setToolTip("Порог пересечения для определения TP (используется только для колонки f1_50)")
    parent.btn_compute_f1 = QPushButton("Расчет meanAP")
    parent.btn_compute_f1.setToolTip("Добавить колонки: mean_f1 (среднее по порогам 0.1-0.9), avg_iou_tp, f1_50 и др.")
    parent.btn_compute_f1.setEnabled(False)

    parent.f1_threshold_label = QLabel("Порог mean_f1:")
    parent.f1_threshold_spin = QDoubleSpinBox()
    parent.f1_threshold_spin.setRange(0.0, 1.0)
    parent.f1_threshold_spin.setSingleStep(0.05)
    parent.f1_threshold_spin.setValue(0.1)
    parent.f1_threshold_spin.setToolTip("Удалить строки с mean_f1 ниже этого порога")
    parent.btn_filter_f1 = QPushButton("Удалить строки с mean_f1 < порога")
    parent.btn_filter_f1.setToolTip("Требуется наличие колонки mean_f1")
    parent.btn_filter_f1.setEnabled(False)

    prepare_row2.addWidget(parent.iou_label)
    prepare_row2.addWidget(parent.iou_spin)
    prepare_row2.addWidget(parent.btn_compute_f1)
    prepare_row2.addSpacing(20)
    prepare_row2.addWidget(parent.f1_threshold_label)
    prepare_row2.addWidget(parent.f1_threshold_spin)
    prepare_row2.addWidget(parent.btn_filter_f1)
    prepare_row2.addStretch()
    right_prepare_layout.addLayout(prepare_row2)

    # Третья строка: агрегация и настройки анализа
    prepare_row3 = QHBoxLayout()
    parent.label_top_k = QLabel("Топ-K для агрегации:")
    parent.top_k_spin = QSpinBox()
    parent.top_k_spin.setRange(1, 20)
    parent.top_k_spin.setValue(1)
    parent.top_k_spin.setToolTip("Сколько лучших строк (по mean_f1) оставлять в каждой группе при агрегации")
    parent.btn_aggregate = QPushButton("Агрегировать по параметрам")
    parent.btn_aggregate.setToolTip("Сгруппировать строки с одинаковыми параметрами, усреднив метрики")
    parent.btn_aggregate.setEnabled(False)

    prepare_row3.addWidget(parent.label_top_k)
    prepare_row3.addWidget(parent.top_k_spin)
    prepare_row3.addWidget(parent.btn_aggregate)
    prepare_row3.addStretch()
    right_prepare_layout.addLayout(prepare_row3)

    # Четвёртая строка: оптимизация инверсии
    prepare_row_invert = QHBoxLayout()
    parent.btn_optimize_invert = QPushButton("Оптимизировать инверсию")
    parent.btn_optimize_invert.setToolTip(
        "Для каждого метода оставить только строки с лучшим значением invert (по среднему mean_f1)")
    parent.btn_optimize_invert.setEnabled(False)
    prepare_row_invert.addWidget(parent.btn_optimize_invert)
    prepare_row_invert.addStretch()
    right_prepare_layout.addLayout(prepare_row_invert)

    # Пятая строка: галочки и кнопка анализа
    prepare_row5 = QHBoxLayout()
    parent.chk_use_diversity = QCheckBox("Использовать разнообразие при выборе топ‑5")
    parent.chk_use_diversity.setChecked(True)
    parent.chk_use_diversity.setToolTip(
        "Включает жадный алгоритм для выбора максимально разных пресетов из всего пула уникальных параметров")
    parent.chk_iou_penalty = QCheckBox("Штраф за низкое качество TP (mean_f1 * avg_iou_tp^0.5)")
    parent.chk_iou_penalty.setToolTip("При анализе использовать взвешенный F1 = mean_f1 * sqrt(avg_iou_tp)")
    parent.btn_analyze = QPushButton("Запустить анализ модели")
    parent.btn_analyze.setToolTip(
        "Обучить регрессию и выдать топ-5 комбинаций параметров для каждого метода (с учётом diversity и штрафа)")
    parent.btn_analyze.setEnabled(False)

    prepare_row5.addWidget(parent.chk_use_diversity)
    prepare_row5.addWidget(parent.chk_iou_penalty)
    prepare_row5.addStretch()
    prepare_row5.addWidget(parent.btn_analyze)
    right_prepare_layout.addLayout(prepare_row5)

    # Шестая строка: загрузка топ-пресетов по методам и глобальный топ-5
    prepare_row6 = QHBoxLayout()
    parent.btn_load_top_methods = QPushButton("Загрузить топ по методам")
    parent.btn_load_top_methods.setToolTip(
        "Загрузить предварительно сохранённый датасет с топ пресетами для каждого метода")
    parent.btn_load_top_methods.setEnabled(True)
    parent.btn_global_top5 = QPushButton("Глобальный топ методов")
    parent.btn_global_top5.setToolTip(
        "Выбрать глобальный топ разнообразных пресетов из загруженного датасета (по реальному F1)")
    parent.btn_global_top5.setEnabled(False)
    prepare_row6.addWidget(parent.btn_load_top_methods)
    prepare_row6.addWidget(parent.btn_global_top5)
    prepare_row6.addStretch()
    right_prepare_layout.addLayout(prepare_row6)

    # Седьмая строка: удаление идеальных строк (actual_f1 >= порога)
    prepare_row7 = QHBoxLayout()
    parent.label_perfect_threshold = QLabel("Удалить строки с actual_f1 ≥")
    parent.perfect_threshold_spin = QDoubleSpinBox()
    parent.perfect_threshold_spin.setRange(0.0, 1.0)
    parent.perfect_threshold_spin.setSingleStep(0.05)
    parent.perfect_threshold_spin.setValue(1.0)
    parent.perfect_threshold_spin.setToolTip("Удалить строки, где actual_f1 больше или равно этому порогу")
    parent.btn_remove_perfect = QPushButton("Удалить идеальные строки")
    parent.btn_remove_perfect.setToolTip(
        "Удалить из датасета строки с actual_f1 >= порога (полезно для фильтрации переобученных пресетов)")
    parent.btn_remove_perfect.setEnabled(False)
    prepare_row7.addWidget(parent.label_perfect_threshold)
    prepare_row7.addWidget(parent.perfect_threshold_spin)
    prepare_row7.addWidget(parent.btn_remove_perfect)
    prepare_row7.addStretch()
    right_prepare_layout.addLayout(prepare_row7)

    right_prepare_group.setLayout(right_prepare_layout)

    second_row_layout.addWidget(left_filter_group, 40)  # 40% ширины
    second_row_layout.addWidget(right_prepare_group, 60)  # 60% ширины
    main_layout.addLayout(second_row_layout)

    # ========== Таблица и результаты ==========
    main_splitter = QSplitter(Qt.Vertical)
    parent.table = QTableWidget()
    parent.table.setAlternatingRowColors(True)
    parent.table.setSortingEnabled(True)
    parent.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    main_splitter.addWidget(parent.table)

    bottom_splitter = QSplitter(Qt.Horizontal)
    parent.results_widget = ModelResearchWidget(parent)
    bottom_splitter.addWidget(parent.results_widget)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_widget.text.setMaximumHeight(150)
    bottom_splitter.addWidget(parent.log_widget)

    main_splitter.addWidget(bottom_splitter)
    main_splitter.setSizes([500, 200])
    bottom_splitter.setSizes([600, 400])

    main_layout.addWidget(main_splitter)