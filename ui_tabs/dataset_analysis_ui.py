# dataset_analysis_ui.py
from import_libs_internal import *
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class TrainingImageLabel(QLabel):
    """QLabel, который масштабирует QPixmap под свой размер
    с сохранением пропорций и перерисовывает при resize."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("background-color: white;")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pixmap = None
        self.setText("(нет данных)")

    def set_image(self, pixmap):
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.setText("(нет данных)")
            self.setPixmap(QPixmap())
        else:
            self.setText("")
            self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def sizeHint(self):
        return QSize(800, 400)


def _make_page(toolbox, title, description, content, buttons,
               min_height=240, content_stretch=1):
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(6, 6, 6, 6)

    if description:
        d = QLabel(description)
        d.setWordWrap(True)
        d.setStyleSheet("color: #555; font-size: 13px;")
        d.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(d)

    if content is not None:
        content.setMinimumHeight(min_height)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(content, content_stretch)

    if buttons:
        br = QHBoxLayout()
        for b in buttons:
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            br.addWidget(b)
        br.addStretch()
        layout.addLayout(br)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    scroll.setFrameShape(QFrame.NoFrame)
    toolbox.addItem(scroll, title)
    return scroll


def _mk_min_spin(rng, suffix="", special="Не задано", tooltip=""):
    w = QSpinBox()
    w.setRange(*rng)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_max_spin(rng, suffix="", special="Без ограничений", tooltip=""):
    w = QSpinBox()
    w.setRange(*rng)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_min_dspin(rng, decimals=2, suffix="", special="Не задано",
                  tooltip=""):
    w = QDoubleSpinBox()
    w.setRange(*rng)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_max_dspin(rng, decimals=2, suffix="", special="Без ограничений",
                  tooltip=""):
    w = QDoubleSpinBox()
    w.setRange(*rng)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def setup_dataset_analysis_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # ============================================================
    # ВЕРХНЯЯ ПАНЕЛЬ
    # ============================================================
    top_layout = QHBoxLayout()
    parent.btn_folder = QPushButton("Load Images")
    parent.btn_labels = QPushButton("Load Labels")
    parent.btn_load_yaml = QPushButton("Load YOLO YAML")
    parent.btn_load_coco = QPushButton("Load COCO")
    parent.btn_add_coco = QPushButton("+ Add COCO")
    parent.btn_load_unet = QPushButton("Load U-Net YAML")
    for b in (parent.btn_folder, parent.btn_labels, parent.btn_load_yaml,
              parent.btn_load_coco, parent.btn_add_coco, parent.btn_load_unet):
        top_layout.addWidget(b)

    parent.dataset_type_combo = QComboBox()
    parent.dataset_type_combo.addItem("Метки (bbox, obb, seg)", 0)
    parent.dataset_type_combo.addItem("Маски (masks)", 1)
    parent.dataset_type_combo.addItem("Только изображения", 2)
    top_layout.addWidget(QLabel("Input type:"))
    top_layout.addWidget(parent.dataset_type_combo)

    parent.input_format_combo = QComboBox()
    parent.input_format_combo.addItem("Auto (по содержимому)", "auto")
    parent.input_format_combo.addItem("YOLO detect (5 токенов)", "detect")
    parent.input_format_combo.addItem("YOLO OBB — 4 угла (9 токенов)", "obb_corners")
    parent.input_format_combo.addItem("YOLO OBB — xywhr (6 токенов)", "obb_xywhr")
    parent.input_format_combo.addItem("YOLO segment (≥7 токенов, чётные)", "segment")
    parent.input_format_combo.addItem("Mixed (per-line эвристика)", "mixed")
    top_layout.addWidget(QLabel("Input fmt:"))
    top_layout.addWidget(parent.input_format_combo)
    top_layout.addStretch()

    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    top_layout.addWidget(parent.toggle_log_btn)
    main_layout.addLayout(top_layout)

    # ============================================================
    # СПЛИТТЕРЫ
    # ============================================================
    parent.main_v_splitter = QSplitter(Qt.Vertical)
    main_layout.addWidget(parent.main_v_splitter, 1)
    h_splitter = QSplitter(Qt.Horizontal)
    parent.main_v_splitter.addWidget(h_splitter)

    # ============================================================
    # ЛЕВАЯ ПАНЕЛЬ
    # ============================================================
    left_splitter = QSplitter(Qt.Vertical)

    # ---------- ВЕРХ: список пар + служебные метки + легенда ----------
    top_left = QWidget()
    tl = QVBoxLayout(top_left)
    tl.setContentsMargins(0, 0, 5, 0)
    tl.addWidget(QLabel("Найденные пары (двойной клик — снимок валидации):"))

    parent.file_list = QListWidget()
    parent.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    parent.file_list.setToolTip(
        "Ctrl+C — скопировать выделенные строки.\n"
        "Двойной клик — открыть снимок валидации.\n"
        "Правый клик — контекстное меню (показать в проводнике / удалить пару).\n"
        "Серый — отброшенные фильтром; цвет класса — прошедшие.\n"
        "Розовый/оранжевый фон — дубликаты (pHash/MD5), "
        "красный — утечка между сплитами.")
    tl.addWidget(parent.file_list, 1)

    parent.pair_count_label = QLabel("Всего пар: 0")
    parent.pair_count_label.setWordWrap(True)
    parent.pair_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    tl.addWidget(parent.pair_count_label)

    parent.validation_progress_label = QLabel("")
    parent.validation_progress_label.setWordWrap(True)
    parent.validation_progress_label.setStyleSheet(
        "color: #0055aa; font-weight: bold;")
    parent.validation_progress_label.setVisible(False)
    tl.addWidget(parent.validation_progress_label)

    parent.validation_warning_label = QLabel("")
    parent.validation_warning_label.setWordWrap(True)
    parent.validation_warning_label.setStyleSheet("color: #b00000;")
    parent.validation_warning_label.setVisible(False)
    tl.addWidget(parent.validation_warning_label)

    parent.color_stats_label = QLabel("")
    parent.color_stats_label.setWordWrap(True)
    parent.color_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.color_stats_label.setStyleSheet("font-weight: bold; color: #333;")
    tl.addWidget(parent.color_stats_label)

    parent.class_legend_label = QLabel("<b>Легенда классов:</b>")
    parent.class_legend_label.setVisible(False)
    tl.addWidget(parent.class_legend_label)

    parent.class_legend_scroll = QScrollArea()
    parent.class_legend_scroll.setWidgetResizable(True)
    parent.class_legend_scroll.setMaximumHeight(180)
    parent.class_legend_scroll.setFrameShape(QFrame.NoFrame)
    parent.class_legend_scroll.setVisible(False)

    parent.class_legend_container = QWidget()
    parent.class_legend_layout = QVBoxLayout(parent.class_legend_container)
    parent.class_legend_layout.setContentsMargins(0, 0, 0, 0)
    parent.class_legend_layout.setSpacing(2)
    parent.class_legend_scroll.setWidget(parent.class_legend_container)
    tl.addWidget(parent.class_legend_scroll)

    top_left.setMinimumHeight(300)

    # ---------- НИЗ: фильтры + кнопки экспорта ----------
    bottom_left = QWidget()
    bl = QVBoxLayout(bottom_left)
    bl.setContentsMargins(0, 5, 5, 0)
    bl.setSpacing(6)

    filter_header = QLabel(
        "<b>Фильтры</b> "
        "<span style='color:#888;font-size:11px'>"
        "(влияют на все виды анализа и на экспорт нового датасета)"
        "</span>")
    filter_header.setWordWrap(True)
    bl.addWidget(filter_header)

    filter_scroll = QScrollArea()
    filter_scroll.setWidgetResizable(True)
    filter_scroll.setFrameShape(QFrame.NoFrame)

    filter_container = QWidget()
    fl = QVBoxLayout(filter_container)
    fl.setContentsMargins(0, 0, 0, 0)

    # =========================================================
    # ФОРМА ФИЛЬТРОВ: 3 колонки — Параметр | Мин | Макс
    # =========================================================
    form = QGridLayout()
    form.setHorizontalSpacing(8)
    form.setVerticalSpacing(4)
    form.setColumnStretch(0, 3)
    form.setColumnStretch(1, 1)
    form.setColumnStretch(2, 1)

    form.addWidget(QLabel("<b>Параметр</b>"), 0, 0)
    form.addWidget(QLabel("<b>Мин</b>"), 0, 1)
    form.addWidget(QLabel("<b>Макс</b>"), 0, 2)

    # 1. Ширина изображения (px)
    parent.filter_min_w = _mk_min_spin(
        (0, 100000),
        tooltip="Минимальная ширина изображения (px).\n"
                "0 = без ограничения.")
    parent.filter_max_w = _mk_max_spin(
        (0, 100000),
        tooltip="Максимальная ширина изображения (px).\n"
                "0 = без ограничения сверху.")
    form.addWidget(QLabel("Ширина изобр. (px):"), 1, 0)
    form.addWidget(parent.filter_min_w, 1, 1)
    form.addWidget(parent.filter_max_w, 1, 2)

    # 2. Высота изображения (px)
    parent.filter_min_h = _mk_min_spin(
        (0, 100000),
        tooltip="Минимальная высота изображения (px).\n"
                "0 = без ограничения.")
    parent.filter_max_h = _mk_max_spin(
        (0, 100000),
        tooltip="Максимальная высота изображения (px).\n"
                "0 = без ограничения сверху.")
    form.addWidget(QLabel("Высота изобр. (px):"), 2, 0)
    form.addWidget(parent.filter_min_h, 2, 1)
    form.addWidget(parent.filter_max_h, 2, 2)

    # 3. Средняя яркость изображения (0..255)
    parent.filter_min_brightness = _mk_min_spin(
        (-1, 255),
        tooltip="Минимальная средняя яркость изображения (0..255).\n"
                "Считается так же, как на странице «Качество изображений».\n"
                "-1 = без ограничения (значение 0 — валидное: очень тёмные).")
    parent.filter_min_brightness.setValue(-1)
    parent.filter_max_brightness = _mk_max_spin(
        (-1, 255),
        tooltip="Максимальная средняя яркость изображения (0..255).\n"
                "-1 = без ограничения (значение 0 — валидное).")
    parent.filter_max_brightness.setValue(-1)
    form.addWidget(QLabel("Яркость изобр. (0–255):"), 3, 0)
    form.addWidget(parent.filter_min_brightness, 3, 1)
    form.addWidget(parent.filter_max_brightness, 3, 2)

    # 4. Количество объектов на снимке
    parent.filter_min_objs = _mk_min_spin(
        (0, 10000),
        tooltip="Минимальное число объектов в изображении "
                "(после учёта включённых классов).\n"
                "0 = допускаются пустые (если включена галочка ниже).")
    parent.filter_max_objs = _mk_max_spin(
        (0, 10000),
        tooltip="Максимальное число объектов в изображении.\n"
                "0 = без ограничения сверху.")
    form.addWidget(QLabel("Кол-во объектов:"), 4, 0)
    form.addWidget(parent.filter_min_objs, 4, 1)
    form.addWidget(parent.filter_max_objs, 4, 2)

    # 5. Площадь объекта (px²)
    parent.filter_min_area = _mk_min_spin(
        (0, 1000000),
        suffix=" px²",
        tooltip="Минимальная площадь объекта (px²).\n"
                "bbox = w·h; segment/OBB = площадь по формуле шнурков.")
    parent.filter_max_area = _mk_max_spin(
        (0, 10000000),
        suffix=" px²",
        tooltip="Максимальная площадь объекта (px²).\n"
                "0 = без ограничения сверху.")
    form.addWidget(QLabel("Площадь объекта (px²):"), 5, 0)
    form.addWidget(parent.filter_min_area, 5, 1)
    form.addWidget(parent.filter_max_area, 5, 2)

    # 6. Площадь объекта в % от площади изображения
    parent.filter_min_area_pct = _mk_min_dspin(
        (0.0, 100.0), decimals=2, suffix=" %",
        tooltip="Минимальная площадь объекта в % от площади изображения.\n"
                "0.00 = без ограничения.")
    parent.filter_max_area_pct = _mk_max_dspin(
        (0.0, 100.0), decimals=2, suffix=" %",
        tooltip="Максимальная площадь объекта в % от площади изображения.\n"
                "0.00 = без ограничения сверху.")
    form.addWidget(QLabel("Площадь объекта (%):"), 6, 0)
    form.addWidget(parent.filter_min_area_pct, 6, 1)
    form.addWidget(parent.filter_max_area_pct, 6, 2)

    # 7. Соотношение сторон W/H
    parent.filter_min_aspect = _mk_min_dspin(
        (0.0, 100.0), decimals=2,
        tooltip="Минимальное соотношение сторон W/H.\n"
                "0.00 = без ограничения.")
    parent.filter_max_aspect = _mk_max_dspin(
        (0.0, 100.0), decimals=2,
        tooltip="Максимальное соотношение сторон W/H.\n"
                "0.00 = без ограничения сверху.")
    form.addWidget(QLabel("Соотношение сторон W/H:"), 7, 0)
    form.addWidget(parent.filter_min_aspect, 7, 1)
    form.addWidget(parent.filter_max_aspect, 7, 2)

    fl.addLayout(form)

    # --- Таблица классов ---
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(5)
    parent.class_table.setHorizontalHeaderLabels(
        ["New ID", "Original ID", "Name", "Mask pixel", "Include"])
    parent.class_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.class_table.setColumnHidden(3, True)
    parent.class_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.class_table.setMinimumHeight(150)
    fl.addWidget(parent.class_table, 1)

    parent.include_empty_annotations = QCheckBox(
        "Include images with empty annotations")
    parent.include_empty_annotations.setToolTip(
        "Включать в результат изображения, у которых после фильтрации "
        "по классам не осталось ни одного объекта.\n"
        "Работает только если «Мин. кол-во объектов» = 0.")
    fl.addWidget(parent.include_empty_annotations)

    parent.chk_exclude_classes_from_files = QCheckBox(
        "Исключать аннотации исключённых классов из .txt при экспорте")
    parent.chk_exclude_classes_from_files.setChecked(False)
    parent.chk_exclude_classes_from_files.setToolTip(
        "При экспорте в .txt оставить только строки с классами, "
        "включёнными в таблице (Include).")
    fl.addWidget(parent.chk_exclude_classes_from_files)

    parent.progress_bar = QProgressBar()
    parent.progress_bar.setVisible(False)
    fl.addWidget(parent.progress_bar)

    filter_scroll.setWidget(filter_container)
    bl.addWidget(filter_scroll, 1)

    # --- Кнопки (фиксированы внизу левой панели) ---
    parent.btn_apply_filters = QPushButton(
        "Обновить метрики (с учётом фильтров)")
    parent.btn_apply_filters.setStyleSheet(
        "background-color: #4CAF50; color: white; font-weight: bold;")

    parent.btn_export_filtered = QPushButton("Создать новый датасет")
    parent.btn_export_filtered.setStyleSheet(
        "background-color: #2196F3; color: white; font-weight: bold;")
    parent.btn_export_filtered.setToolTip(
        "Экспортирует все пары, прошедшие фильтрацию "
        "(включая пустые снимки, если включена соответствующая галочка).")

    parent.btn_open_last_export = QPushButton("📂 Открыть папку датасета")
    parent.btn_open_last_export.setToolTip(
        "Открыть папку последнего экспортированного датасета")
    parent.btn_open_last_export.setEnabled(False)

    btn_row = QHBoxLayout()
    btn_row.addWidget(parent.btn_apply_filters)
    btn_row.addWidget(parent.btn_export_filtered)
    btn_row.addWidget(parent.btn_open_last_export)
    bl.addLayout(btn_row)

    bottom_left.setMinimumHeight(280)

    left_splitter.addWidget(top_left)
    left_splitter.addWidget(bottom_left)
    left_splitter.setSizes([460, 360])
    h_splitter.addWidget(left_splitter)

    # ============================================================
    # ПРАВАЯ ПАНЕЛЬ — TOOLBOX
    # ============================================================
    parent.toolbox = QToolBox()
    h_splitter.addWidget(parent.toolbox)
    h_splitter.setSizes([520, 680])

    # ============================================================
    # 1. Классы, объекты и площади
    # ============================================================
    metrics_content = QWidget()
    mc = QVBoxLayout(metrics_content)
    mc.setContentsMargins(0, 0, 0, 0)

    parent.metrics_summary_label = QLabel(
        "Загрузите датасет для отображения метрик.")
    parent.metrics_summary_label.setWordWrap(True)
    mc.addWidget(parent.metrics_summary_label)

    parent.class_stats_table = QTableWidget()
    parent.class_stats_table.setColumnCount(3)
    parent.class_stats_table.setHorizontalHeaderLabels(
        ["Класс", "Кол-во объектов", "Доля (%)"])
    parent.class_stats_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.class_stats_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.class_stats_table.setSelectionBehavior(
        QAbstractItemView.SelectRows)
    parent.class_stats_table.setMinimumHeight(120)
    mc.addWidget(parent.class_stats_table, 1)

    # --- Фигура площадей: 3 подграфика (px², %, классы) ---
    parent.figure = Figure(figsize=(8, 3.5))
    parent.canvas = FigureCanvas(parent.figure)
    parent.ax = parent.figure.add_subplot(131)          # площадь px²
    parent.ax_area_pct = parent.figure.add_subplot(132)  # площадь %
    parent.ax_classes = parent.figure.add_subplot(133)   # по классам
    parent.canvas.setMinimumHeight(280)
    mc.addWidget(parent.canvas, 2)

    parent.class_chart_figure = Figure(figsize=(6, 2.8))
    parent.class_chart_canvas = FigureCanvas(parent.class_chart_figure)
    parent.ax_pie = parent.class_chart_figure.add_subplot(121)
    parent.ax_bar = parent.class_chart_figure.add_subplot(122)
    parent.class_chart_canvas.setMinimumHeight(220)
    mc.addWidget(parent.class_chart_canvas, 1)

    imb_header = QLabel("<b>Дисбаланс классов</b>")
    mc.addWidget(imb_header)

    parent.imbalance_summary_label = QLabel(
        "Рассчитывается автоматически после фильтрации.")
    parent.imbalance_summary_label.setWordWrap(True)
    mc.addWidget(parent.imbalance_summary_label)

    parent.imbalance_table = QTableWidget()
    parent.imbalance_table.setColumnCount(2)
    parent.imbalance_table.setHorizontalHeaderLabels(
        ["Метрика", "Значение"])
    parent.imbalance_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.imbalance_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.imbalance_table.setMinimumHeight(140)
    parent.imbalance_table.setToolTip(
        "Наведите курсор на строку — всплывёт описание метрики.")
    mc.addWidget(parent.imbalance_table, 1)

    _make_page(
        parent.toolbox, "1. Классы, объекты и площади",
        "**Таблица классов** — число объектов каждого класса и доля. "
        "**Площади** — bbox: w·h, segment/OBB: shoelace. "
        "Слева — гистограмма площадей в пикселях, в центре — "
        "в % от площади изображения, справа — распределение по классам. "
        "**Круговая/столбчатая диаграммы** — распределение по классам. "
        "**Дисбаланс:** ratio = max/min; Gini (0 — равномерно); "
        "Entropy (выше — равномернее). Тултипы на строках таблицы.",
        metrics_content, buttons=[],
        min_height=650, content_stretch=1,
    )

    # ============================================================
    # 2. Качество изображений
    # ============================================================
    quality_content = QWidget()
    qc = QVBoxLayout(quality_content)
    qc.setContentsMargins(0, 0, 0, 0)

    parent.quality_summary_label = QLabel("Запустите анализ качества.")
    parent.quality_summary_label.setWordWrap(True)
    qc.addWidget(parent.quality_summary_label)

    parent.quality_figure = Figure(figsize=(6, 4))
    parent.quality_canvas = FigureCanvas(parent.quality_figure)
    parent.quality_ax1 = parent.quality_figure.add_subplot(221)
    parent.quality_ax2 = parent.quality_figure.add_subplot(222)
    parent.quality_ax3 = parent.quality_figure.add_subplot(223)
    parent.quality_ax4 = parent.quality_figure.add_subplot(224)
    parent.quality_canvas.setMinimumHeight(340)
    qc.addWidget(parent.quality_canvas, 1)

    parent.btn_run_quality = QPushButton("Запустить анализ")
    parent.btn_stop_quality = QPushButton("⏹ Остановить")
    parent.btn_stop_quality.setEnabled(False)

    _make_page(
        parent.toolbox, "2. Качество изображений",
        "**imgsz** — рекомендуемый размер для обучения. "
        "**Яркость:** тёмные < 40, пересвеченные > 215. "
        "**Aspect** W/H — соотношение сторон. "
        "**Работает по filtered_pairs** (то есть с учётом фильтров "
        "из левой панели).",
        quality_content,
        buttons=[parent.btn_run_quality, parent.btn_stop_quality],
        min_height=400, content_stretch=1,
    )

    # ============================================================
    # 3. Дубликаты и утечка
    # ============================================================
    dup_content = QWidget()
    dc = QVBoxLayout(dup_content)
    dc.setContentsMargins(0, 0, 0, 0)

    parent.dup_summary_label = QLabel("Нажмите «Найти дубликаты».")
    parent.dup_summary_label.setWordWrap(True)
    dc.addWidget(parent.dup_summary_label)

    parent.dup_table = QTableWidget()
    parent.dup_table.setColumnCount(5)
    parent.dup_table.setHorizontalHeaderLabels(
        ["Группа", "N", "Сплиты", "Файлы", "Утечка"])
    parent.dup_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.dup_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.dup_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    parent.dup_table.setSelectionMode(QAbstractItemView.SingleSelection)
    parent.dup_table.setMinimumHeight(140)
    parent.dup_table.setToolTip(
        "Каждая строка — группа дубликатов.\n"
        "Одиночный клик по строке — выделить все пары этой группы в "
        "списке слева.\n"
        "Двойной клик — открыть валидацию всех пар группы "
        "(последовательно).")
    dc.addWidget(parent.dup_table, 1)

    parent.btn_run_duplicates = QPushButton("Найти дубликаты и утечку")
    parent.btn_stop_duplicates = QPushButton("⏹ Остановить")
    parent.btn_stop_duplicates.setEnabled(False)

    _make_page(
        parent.toolbox, "3. Дубликаты и утечка",
        "**Точные** — MD5. **Почти** — pHash Hamming ≤ 8. "
        "**Утечка** — группа в ≥2 сплитах. "
        "Подсветка в списке: 🔴 точный, 🟠 похожее, 🌕 утечка. "
        "**Выбор строки** — выделяет пары группы в списке слева. "
        "**Двойной клик** — открывает валидацию всех пар группы "
        "последовательно. **Работает по filtered_pairs.**",
        dup_content,
        buttons=[parent.btn_run_duplicates, parent.btn_stop_duplicates],
        min_height=320, content_stretch=1,
    )

    # ============================================================
    # 4. Тепловая карта объектов
    # ============================================================
    heat_content = QWidget()
    hc = QVBoxLayout(heat_content)
    hc.setContentsMargins(0, 0, 0, 0)

    parent.heatmap_summary_label = QLabel("Постройте тепловую карту.")
    parent.heatmap_summary_label.setWordWrap(True)
    hc.addWidget(parent.heatmap_summary_label)

    grid_row = QHBoxLayout()
    grid_row.addWidget(QLabel("Размер сетки:"))
    parent.heatmap_grid_spin = QSpinBox()
    parent.heatmap_grid_spin.setRange(8, 128)
    parent.heatmap_grid_spin.setValue(16)
    parent.heatmap_grid_spin.setSingleStep(8)
    parent.heatmap_grid_spin.setToolTip(
        "16×16 — обзорно; 64–128 — детальный анализ.")
    grid_row.addWidget(parent.heatmap_grid_spin)
    grid_row.addStretch()
    hc.addLayout(grid_row)

    parent.heatmap_figure = Figure(figsize=(5, 4))
    parent.heatmap_canvas = FigureCanvas(parent.heatmap_figure)
    parent.heatmap_ax = parent.heatmap_figure.add_subplot(111)
    parent.heatmap_canvas.setMinimumHeight(320)
    hc.addWidget(parent.heatmap_canvas, 1)

    parent.btn_run_heatmap = QPushButton("Построить тепловую карту")
    parent.btn_stop_heatmap = QPushButton("⏹ Остановить")
    parent.btn_stop_heatmap.setEnabled(False)

    _make_page(
        parent.toolbox, "4. Тепловая карта объектов",
        "Центры bbox/полигонов на сетке N×N.\n"
        "**Оси X/Y** — % от ширины/высоты. **Цвет/колорбар** — "
        "число центров объектов в ячейке.\n"
        "**Размер сетки** 8…128. **Работает по filtered_pairs.**",
        heat_content,
        buttons=[parent.btn_run_heatmap, parent.btn_stop_heatmap],
        min_height=420, content_stretch=1,
    )

    # ============================================================
    # 5. Аудит модели (Loss Inspection) + валидация YOLO
    # ============================================================
    audit_content = QWidget()
    ac = QVBoxLayout(audit_content)
    ac.setContentsMargins(0, 0, 0, 0)

    model_row = QHBoxLayout()
    parent.audit_model_edit = QLineEdit()
    parent.audit_model_edit.setReadOnly(True)
    parent.audit_model_edit.setPlaceholderText("Модель не выбрана")
    parent.btn_load_audit_model = QPushButton("Загрузить модель")
    model_row.addWidget(QLabel("YOLO:"))
    model_row.addWidget(parent.audit_model_edit, 1)
    model_row.addWidget(parent.btn_load_audit_model)
    ac.addLayout(model_row)

    params_row = QHBoxLayout()
    parent.audit_iou_spin = QDoubleSpinBox()
    parent.audit_iou_spin.setRange(0.0, 1.0)
    parent.audit_iou_spin.setValue(0.3)
    parent.audit_iou_spin.setSingleStep(0.05)
    parent.audit_iou_spin.setDecimals(2)
    parent.audit_iou_spin.setToolTip(
        "Порог среднего IoU между GT и предсказанием.\n"
        "Пары с mean IoU < порога попадают в «подозрительные».")
    parent.audit_conf_spin = QDoubleSpinBox()
    parent.audit_conf_spin.setRange(0.0, 1.0)
    parent.audit_conf_spin.setValue(0.25)
    parent.audit_conf_spin.setSingleStep(0.05)
    parent.audit_conf_spin.setDecimals(2)
    parent.audit_conf_spin.setToolTip(
        "Confidence-порог модели при аудите.")
    parent.audit_imgsz_spin = QSpinBox()
    parent.audit_imgsz_spin.setRange(64, 4096)
    parent.audit_imgsz_spin.setValue(640)
    parent.audit_imgsz_spin.setSingleStep(32)
    parent.audit_imgsz_spin.setToolTip(
        "Размер входного изображения для модели при аудите "
        "(imgsz, по умолчанию 640).")

    params_row.addWidget(QLabel("Порог IoU:"))
    params_row.addWidget(parent.audit_iou_spin)
    params_row.addWidget(QLabel("Confidence:"))
    params_row.addWidget(parent.audit_conf_spin)
    params_row.addWidget(QLabel("imgsz:"))
    params_row.addWidget(parent.audit_imgsz_spin)
    params_row.addStretch()
    ac.addLayout(params_row)

    parent.audit_summary_label = QLabel(
        "Загрузите модель и запустите аудит.")
    parent.audit_summary_label.setWordWrap(True)
    ac.addWidget(parent.audit_summary_label)

    audit_filter_row = QHBoxLayout()
    audit_filter_row.addWidget(QLabel("Показывать:"))
    parent.audit_filter_combo = QComboBox()
    parent.audit_filter_combo.addItem(
        "Только подозрительные", "suspicious")
    parent.audit_filter_combo.addItem(
        "Все аудированные пары", "all")
    parent.audit_filter_combo.setToolTip(
        "«Только подозрительные» — пары с низким IoU, "
        "пропущенными объектами (GT без Pred) "
        "или ложными срабатываниями (Pred без GT).\n"
        "«Все аудированные» — каждая пара, для которой "
        "удалось сравнить GT и Pred.")
    audit_filter_row.addWidget(parent.audit_filter_combo)
    audit_filter_row.addStretch()
    ac.addLayout(audit_filter_row)

    parent.audit_table = QTableWidget()
    parent.audit_table.setColumnCount(4)
    parent.audit_table.setHorizontalHeaderLabels(
        ["Изображение", "GT", "Pred", "Ср. IoU"])
    parent.audit_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.audit_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.audit_table.setMinimumHeight(200)
    parent.audit_table.setToolTip(
        "Двойной клик по строке — открыть двойное окно сравнения.")
    ac.addWidget(parent.audit_table, 1)

    val_sep = QLabel(
        "<hr><b>Валидация модели на тестовом датасете (YOLO val)</b>")
    ac.addWidget(val_sep)

    val_row = QHBoxLayout()
    parent.val_data_edit = QLineEdit()
    parent.val_data_edit.setPlaceholderText(
        "Путь к data.yaml или к папке датасета")
    parent.btn_browse_val_data = QPushButton("Обзор…")
    val_row.addWidget(QLabel("Test data:"))
    val_row.addWidget(parent.val_data_edit, 1)
    val_row.addWidget(parent.btn_browse_val_data)
    ac.addLayout(val_row)

    val_params_row = QHBoxLayout()
    parent.val_split_combo = QComboBox()
    parent.val_split_combo.addItems(['val', 'test', 'train'])
    parent.val_split_combo.setToolTip(
        "Сплит из YAML для валидации.")

    parent.val_imgsz_spin = QSpinBox()
    parent.val_imgsz_spin.setRange(64, 4096)
    parent.val_imgsz_spin.setValue(640)
    parent.val_imgsz_spin.setSingleStep(32)
    parent.val_imgsz_spin.setToolTip(
        "Размер изображений для валидации.")

    parent.val_conf_spin = QDoubleSpinBox()
    parent.val_conf_spin.setRange(0.001, 1.0)
    parent.val_conf_spin.setValue(0.001)
    parent.val_conf_spin.setDecimals(3)
    parent.val_conf_spin.setSingleStep(0.01)
    parent.val_conf_spin.setToolTip(
        "Confidence для валидации. Обычно 0.001.")

    parent.val_iou_spin = QDoubleSpinBox()
    parent.val_iou_spin.setRange(0.1, 0.99)
    parent.val_iou_spin.setValue(0.6)
    parent.val_iou_spin.setDecimals(2)
    parent.val_iou_spin.setSingleStep(0.05)
    parent.val_iou_spin.setToolTip(
        "IoU-порог для NMS при валидации.")

    val_params_row.addWidget(QLabel("Split:"))
    val_params_row.addWidget(parent.val_split_combo)
    val_params_row.addWidget(QLabel("imgsz:"))
    val_params_row.addWidget(parent.val_imgsz_spin)
    val_params_row.addWidget(QLabel("conf:"))
    val_params_row.addWidget(parent.val_conf_spin)
    val_params_row.addWidget(QLabel("iou:"))
    val_params_row.addWidget(parent.val_iou_spin)
    val_params_row.addStretch()
    ac.addLayout(val_params_row)

    parent.val_summary_label = QLabel(
        "Укажите путь к тестовому датасету и запустите валидацию.")
    parent.val_summary_label.setWordWrap(True)
    ac.addWidget(parent.val_summary_label)

    parent.val_table = QTableWidget()
    parent.val_table.setColumnCount(6)
    parent.val_table.setHorizontalHeaderLabels(
        ["Класс", "Precision", "Recall", "F1", "mAP50", "mAP50-95"])
    parent.val_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.val_table.setEditTriggers(
        QAbstractItemView.NoEditTriggers)
    parent.val_table.setMinimumHeight(150)
    ac.addWidget(parent.val_table, 1)

    val_btn_row = QHBoxLayout()
    parent.btn_run_validation = QPushButton("▶ Запустить валидацию")
    parent.btn_run_validation.setStyleSheet(
        "background-color: #00897B; color: white; font-weight: bold;")
    parent.btn_stop_validation = QPushButton("⏹ Остановить валидацию")
    parent.btn_stop_validation.setEnabled(False)
    val_btn_row.addWidget(parent.btn_run_validation)
    val_btn_row.addWidget(parent.btn_stop_validation)
    val_btn_row.addStretch()
    ac.addLayout(val_btn_row)

    parent.btn_run_audit = QPushButton("Запустить аудит модели")
    parent.btn_stop_audit = QPushButton("⏹ Остановить")
    parent.btn_stop_audit.setEnabled(False)

    _make_page(
        parent.toolbox, "5. Аудит модели (Loss Inspection) и валидация YOLO",
        "**Loss Inspection:** модель прогоняется по каждой GT-разметке. "
        "**Двойной клик по строке** — двойное окно сравнения (GT ↔ Pred). "
        "**Валидация (YOLO val):** укажите путь к `data.yaml` тестового "
        "датасета (или папке), выберите сплит, запустите. "
        "Метрики P/R/F1/mAP50/mAP50-95 по каждому классу выдаёт "
        "Ultralytics. **Работает по filtered_pairs.**",
        audit_content,
        buttons=[parent.btn_run_audit, parent.btn_stop_audit],
        min_height=700, content_stretch=1,
    )

    # ============================================================
    # 6. Кластеризация эмбеддингов
    # ============================================================
    cluster_content = QWidget()
    cc = QVBoxLayout(cluster_content)
    cc.setContentsMargins(0, 0, 0, 0)

    parent.cluster_summary_label = QLabel(
        "Требуется CLIP (open_clip_torch или transformers). "
        "UMAP и HDBSCAN опциональны.")
    parent.cluster_summary_label.setWordWrap(True)
    cc.addWidget(parent.cluster_summary_label)

    cluster_radius_row = QHBoxLayout()
    cluster_radius_row.addWidget(QLabel("Радиус клика по точке:"))
    parent.cluster_click_radius_spin = QDoubleSpinBox()
    parent.cluster_click_radius_spin.setRange(0.1, 5.0)
    parent.cluster_click_radius_spin.setValue(0.5)
    parent.cluster_click_radius_spin.setSingleStep(0.1)
    parent.cluster_click_radius_spin.setDecimals(2)
    parent.cluster_click_radius_spin.setToolTip(
        "Радиус захвата клика в UMAP-пространстве.")
    cluster_radius_row.addWidget(parent.cluster_click_radius_spin)
    cluster_radius_row.addStretch()
    cc.addLayout(cluster_radius_row)

    parent.cluster_figure = Figure(figsize=(5, 4))
    parent.cluster_canvas = FigureCanvas(parent.cluster_figure)
    parent.cluster_ax = parent.cluster_figure.add_subplot(111)
    parent.cluster_canvas.setMinimumHeight(320)
    cc.addWidget(parent.cluster_canvas, 1)

    parent.cluster_toolbar = NavigationToolbar(
        parent.cluster_canvas, parent)
    cc.addWidget(parent.cluster_toolbar)

    parent.btn_run_cluster = QPushButton("Кластеризовать эмбеддинги")
    parent.btn_stop_cluster = QPushButton("⏹ Остановить")
    parent.btn_stop_cluster.setEnabled(False)

    _make_page(
        parent.toolbox, "6. Кластеризация эмбеддингов",
        "CLIP → UMAP (2D) → HDBSCAN.\n"
        "**Панель инструментов над графиком:** 🔍 зум, ✋ pan, "
        "🏠 сброс, 💾 сохранить PNG.\n"
        "**Клик по точке** (не в зуме) — открыть пару. "
        "«Шум» — выбросы. **Работает по filtered_pairs.**",
        cluster_content,
        buttons=[parent.btn_run_cluster, parent.btn_stop_cluster],
        min_height=420, content_stretch=1,
    )

    # ============================================================
    # 7. Результаты обучения модели
    # ============================================================
    tr_content = QWidget()
    tcl = QVBoxLayout(tr_content)
    tcl.setContentsMargins(0, 0, 0, 0)
    tcl.setSpacing(8)

    tr_row = QHBoxLayout()
    parent.training_results_edit = QLineEdit()
    parent.training_results_edit.setReadOnly(True)
    parent.training_results_edit.setPlaceholderText(
        "Папка run/ с результатами обучения не выбрана")
    parent.btn_browse_training_results = QPushButton("Обзор…")
    parent.btn_load_training_results = QPushButton("Загрузить")
    tr_row.addWidget(QLabel("Папка run/:"))
    tr_row.addWidget(parent.training_results_edit, 1)
    tr_row.addWidget(parent.btn_browse_training_results)
    tr_row.addWidget(parent.btn_load_training_results)
    tcl.addLayout(tr_row)

    parent.training_results_summary_label = QLabel(
        "Укажите папку, где лежат results.png, labels.jpg, "
        "confusion_matrix_normalized.png, weights/.")
    parent.training_results_summary_label.setWordWrap(True)
    tcl.addWidget(parent.training_results_summary_label)

    for key, title in (
        ('results', "results.png — кривые обучения"),
        ('labels', "labels.jpg — распределение bbox"),
        ('confusion', "confusion_matrix_normalized.png"),
    ):
        lbl_title = QLabel(f"<b>{title}</b>")
        tcl.addWidget(lbl_title)
        img_lbl = TrainingImageLabel()
        setattr(parent, f"training_img_{key}", img_lbl)
        tcl.addWidget(img_lbl)

    _make_page(
        parent.toolbox, "7. Результаты обучения модели",
        "Укажите папку run/ (где лежат результаты обучения YOLO). "
        "**В окне** отображаются: `results.png`, `labels.jpg`, "
        "`confusion_matrix_normalized.png`.\n"
        "**Остальные файлы** (`Box*/Mask*_curve.png`, "
        "`confusion_matrix.png`, `results.csv`, `args.yaml`, "
        "`train_batch*`, `val_batch*`, `weights/*`) попадают в "
        "Markdown-отчёт.\n"
        "**При сохранении отчёта** в `dataset_report_assets/` "
        "копируются `best.pt` и `best_meta.json` (если есть).",
        tr_content,
        buttons=[],
        min_height=650, content_stretch=1,
    )

    # ============================================================
    # 8. Общий отчёт
    # ============================================================
    report_content = QWidget()
    rc = QVBoxLayout(report_content)
    rc.setContentsMargins(0, 0, 0, 0)

    parent.report_info_label = QLabel(
        "Собирает все результаты текущей сессии в один Markdown-файл "
        "(в фоновом потоке — UI не блокируется).\n"
        "Рядом с .md создаётся папка <name>_assets/ с PNG "
        "(areas, classes, quality, heatmap, cluster) и, если загружены "
        "результаты обучения — `best.pt`, `best_meta.json`.")
    parent.report_info_label.setWordWrap(True)
    rc.addWidget(parent.report_info_label)

    parent.chk_report_include_quality = QCheckBox(
        "Включить качество изображений")
    parent.chk_report_include_dup = QCheckBox(
        "Включить дубликаты и утечку")
    parent.chk_report_include_heatmap = QCheckBox(
        "Включить тепловую карту")
    parent.chk_report_include_audit = QCheckBox(
        "Включить Loss Inspection")
    parent.chk_report_include_validation = QCheckBox(
        "Включить валидацию YOLO")
    parent.chk_report_include_cluster = QCheckBox(
        "Включить кластеризацию")
    parent.chk_report_include_training_results = QCheckBox(
        "Включить результаты обучения модели")
    for cb in (parent.chk_report_include_quality,
               parent.chk_report_include_dup,
               parent.chk_report_include_heatmap,
               parent.chk_report_include_audit,
               parent.chk_report_include_validation,
               parent.chk_report_include_cluster,
               parent.chk_report_include_training_results):
        cb.setChecked(True)
        rc.addWidget(cb)

    rc.addStretch()
    parent.btn_generate_report = QPushButton("💾 Сгенерировать .md отчёт")
    parent.btn_generate_report.setStyleSheet(
        "background-color: #673AB7; color: white; font-weight: bold;")
    parent.btn_open_report_folder = QPushButton("📂 Открыть папку отчёта")
    parent.btn_open_report_folder.setToolTip(
        "Открыть папку с последним отчётом (.md) и assets")
    parent.btn_open_report_folder.setEnabled(False)

    _make_page(
        parent.toolbox, "8. Общий отчёт",
        "Собирает результаты всех запущенных анализов в один файл.",
        report_content, buttons=[parent.btn_generate_report,
                                 parent.btn_open_report_folder],
        min_height=340, content_stretch=0,
    )

    # ============================================================
    # Лог
    # ============================================================
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.main_v_splitter.addWidget(parent.log_widget)
    parent.main_v_splitter.setSizes([600, 150])
    parent.log_widget.setVisible(False)

    # ============================================================
    # Подключения верхней панели
    # ============================================================
    parent.btn_folder.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_load_yaml.clicked.connect(parent.load_from_yaml)
    parent.btn_load_coco.clicked.connect(parent.load_from_coco)
    parent.btn_add_coco.clicked.connect(parent.add_from_coco)
    parent.btn_load_unet.clicked.connect(parent.load_from_unet_yaml)