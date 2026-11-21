from ui.dataset_source import DatasetSourceWidget

"""Dataset analysis source, filters and export controls."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import (
    _mk_max_dspin,
    _mk_max_spin,
    _mk_min_dspin,
    _mk_min_spin,
)


def build_source(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # ============================================================
    # ВЕРХНЯЯ ПАНЕЛЬ
    # ============================================================
    parent.dataset_source = DatasetSourceWidget(
        ("folder", "labels", "yolo_yaml", "coco", "add_coco", "unet_yaml"),
        types=(0, 1, 2),
        formats=True,
        profile="dataset_analysis",
    )
    parent.btn_folder = parent.dataset_source.button("folder")
    parent.btn_labels = parent.dataset_source.button("labels")
    parent.btn_load_yaml = parent.dataset_source.button("yolo_yaml")
    parent.btn_load_coco = parent.dataset_source.button("coco")
    parent.btn_add_coco = parent.dataset_source.button("add_coco")
    parent.dataset_type_combo = parent.dataset_source.type_combo
    parent.input_format_combo = parent.dataset_source.format_combo
    main_layout.addWidget(parent.dataset_source)
    parent.btn_load_unet = parent.dataset_source.button("unet_yaml")
    top_layout = QHBoxLayout()
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
    pairs_hint = QLabel("Снимки и разметка (двойной клик — просмотр)")
    pairs_hint.setWordWrap(True)
    tl.addWidget(pairs_hint)

    parent.file_list = QListWidget()
    parent.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    parent.file_list.setToolTip(
        "Ctrl+C — скопировать выделенные строки.\n"
        "Двойной клик — открыть снимок валидации.\n"
        "Правый клик — контекстное меню (показать в проводнике / удалить пару).\n"
        "Серый — отброшенные фильтром; цвет класса — прошедшие.\n"
        "Розовый/оранжевый фон — дубликаты (pHash/MD5), "
        "красный — утечка между сплитами."
    )
    tl.addWidget(parent.file_list, 1)

    parent.pair_count_label = QLabel("Всего пар: 0")
    parent.pair_count_label.setWordWrap(True)
    parent.pair_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    tl.addWidget(parent.pair_count_label)

    parent.validation_progress_label = QLabel("")
    parent.validation_progress_label.setWordWrap(True)
    parent.validation_progress_label.setStyleSheet("color: #0055aa; font-weight: bold;")
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

    top_left.setMinimumHeight(220)

    # ---------- НИЗ: фильтры + кнопки экспорта ----------
    bottom_left = QWidget()
    bl = QVBoxLayout(bottom_left)
    bl.setContentsMargins(0, 5, 5, 0)
    bl.setSpacing(6)

    filter_header = QLabel(
        "<b>Фильтры</b> "
        "<span style='color:#888;font-size:11px'>"
        "(для сводки классов, экспорта и области «После фильтров»)"
        "</span>"
    )
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
        tooltip="Минимальная ширина изображения (px).\n0 = без ограничения.",
    )
    parent.filter_max_w = _mk_max_spin(
        (0, 100000),
        tooltip="Максимальная ширина изображения (px).\n0 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Ширина изобр. (px):"), 1, 0)
    form.addWidget(parent.filter_min_w, 1, 1)
    form.addWidget(parent.filter_max_w, 1, 2)

    # 2. Высота изображения (px)
    parent.filter_min_h = _mk_min_spin(
        (0, 100000),
        tooltip="Минимальная высота изображения (px).\n0 = без ограничения.",
    )
    parent.filter_max_h = _mk_max_spin(
        (0, 100000),
        tooltip="Максимальная высота изображения (px).\n0 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Высота изобр. (px):"), 2, 0)
    form.addWidget(parent.filter_min_h, 2, 1)
    form.addWidget(parent.filter_max_h, 2, 2)

    # 3. Средняя яркость изображения (0..255)
    parent.filter_min_brightness = _mk_min_spin(
        (-1, 255),
        tooltip="Минимальная средняя яркость изображения (0..255).\n"
        "Считается так же, как на странице «Качество изображений».\n"
        "-1 = без ограничения (значение 0 — валидное: очень тёмные).",
    )
    parent.filter_min_brightness.setValue(-1)
    parent.filter_max_brightness = _mk_max_spin(
        (-1, 255),
        tooltip="Максимальная средняя яркость изображения (0..255).\n"
        "-1 = без ограничения (значение 0 — валидное).",
    )
    parent.filter_max_brightness.setValue(-1)
    form.addWidget(QLabel("Яркость изобр. (0–255):"), 3, 0)
    form.addWidget(parent.filter_min_brightness, 3, 1)
    form.addWidget(parent.filter_max_brightness, 3, 2)

    # 4. Количество объектов на снимке
    parent.filter_min_objs = _mk_min_spin(
        (0, 10000),
        tooltip="Минимальное число объектов в изображении "
        "(после учёта включённых классов).\n"
        "0 = допускаются пустые (если включена галочка ниже).",
    )
    parent.filter_max_objs = _mk_max_spin(
        (0, 10000),
        tooltip="Максимальное число объектов в изображении.\n"
        "0 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Кол-во объектов:"), 4, 0)
    form.addWidget(parent.filter_min_objs, 4, 1)
    form.addWidget(parent.filter_max_objs, 4, 2)

    # 5. Площадь объекта (px²)
    parent.filter_min_area = _mk_min_spin(
        (0, 1000000),
        suffix=" px²",
        tooltip="Минимальная площадь объекта (px²).\n"
        "bbox = w·h; segment/OBB = площадь по формуле шнурков.",
    )
    parent.filter_max_area = _mk_max_spin(
        (0, 10000000),
        suffix=" px²",
        tooltip="Максимальная площадь объекта (px²).\n0 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Площадь объекта (px²):"), 5, 0)
    form.addWidget(parent.filter_min_area, 5, 1)
    form.addWidget(parent.filter_max_area, 5, 2)

    # 6. Площадь объекта в % от площади изображения
    parent.filter_min_area_pct = _mk_min_dspin(
        (0.0, 100.0),
        decimals=2,
        suffix=" %",
        tooltip="Минимальная площадь объекта в % от площади изображения.\n"
        "0.00 = без ограничения.",
    )
    parent.filter_max_area_pct = _mk_max_dspin(
        (0.0, 100.0),
        decimals=2,
        suffix=" %",
        tooltip="Максимальная площадь объекта в % от площади изображения.\n"
        "0.00 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Площадь объекта (%):"), 6, 0)
    form.addWidget(parent.filter_min_area_pct, 6, 1)
    form.addWidget(parent.filter_max_area_pct, 6, 2)

    # 7. Соотношение сторон W/H
    parent.filter_min_aspect = _mk_min_dspin(
        (0.0, 100.0),
        decimals=2,
        tooltip="Минимальное соотношение сторон W/H.\n0.00 = без ограничения.",
    )
    parent.filter_max_aspect = _mk_max_dspin(
        (0.0, 100.0),
        decimals=2,
        tooltip="Максимальное соотношение сторон W/H.\n0.00 = без ограничения сверху.",
    )
    form.addWidget(QLabel("Соотношение сторон W/H:"), 7, 0)
    form.addWidget(parent.filter_min_aspect, 7, 1)
    form.addWidget(parent.filter_max_aspect, 7, 2)

    fl.addLayout(form)

    for index in range(form.count()):
        widget = form.itemAt(index).widget()
        if isinstance(widget, QLabel):
            widget.setWordWrap(True)

    # --- Таблица классов ---
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(5)
    parent.class_table.setHorizontalHeaderLabels(
        ["Новый ID", "Исходный ID", "Класс", "Пиксель", "Включён"]
    )
    parent.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.class_table.setColumnHidden(3, True)
    parent.class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.class_table.setMinimumHeight(150)
    fl.addWidget(parent.class_table, 1)

    parent.include_empty_annotations = QCheckBox("Включать снимки с пустой разметкой")
    parent.include_empty_annotations.setToolTip(
        "Включать в результат изображения, у которых после фильтрации "
        "по классам не осталось ни одного объекта.\n"
        "Работает только если «Мин. кол-во объектов» = 0."
    )
    parent.include_empty_annotations.setChecked(True)
    fl.addWidget(parent.include_empty_annotations)

    parent.chk_exclude_classes_from_files = QCheckBox(
        "При экспорте удалять аннотации\nисключённых классов"
    )
    parent.chk_exclude_classes_from_files.setChecked(False)
    parent.chk_exclude_classes_from_files.setToolTip(
        "При экспорте в .txt оставить только строки с классами, "
        "включёнными в таблице (Include)."
    )
    fl.addWidget(parent.chk_exclude_classes_from_files)

    parent.progress_bar = QProgressBar()
    parent.progress_bar.setVisible(False)
    fl.addWidget(parent.progress_bar)

    filter_scroll.setWidget(filter_container)
    bl.addWidget(filter_scroll, 1)

    # --- Кнопки (фиксированы внизу левой панели) ---
    parent.btn_apply_filters = QPushButton("Применить фильтры")
    parent.btn_apply_filters.setStyleSheet(
        "background-color: #4CAF50; color: white; font-weight: bold;"
    )

    parent.btn_export_filtered = QPushButton("Экспорт набора")
    parent.btn_export_filtered.setStyleSheet(
        "background-color: #2196F3; color: white; font-weight: bold;"
    )
    parent.btn_export_filtered.setToolTip(
        "Экспортирует все пары, прошедшие фильтрацию "
        "(включая пустые снимки, если включена соответствующая галочка)."
    )

    parent.btn_open_last_export = QPushButton("Папка экспорта")
    parent.btn_open_last_export.setToolTip(
        "Открыть папку последнего экспортированного датасета"
    )
    parent.btn_open_last_export.setEnabled(False)

    btn_row = QGridLayout()
    btn_row.addWidget(parent.btn_apply_filters, 0, 0, 1, 2)
    btn_row.addWidget(parent.btn_export_filtered, 1, 0)
    btn_row.addWidget(parent.btn_open_last_export, 1, 1)
    bl.addLayout(btn_row)

    bottom_left.setMinimumHeight(200)

    left_splitter.addWidget(top_left)
    left_splitter.addWidget(bottom_left)
    left_splitter.setSizes([460, 360])
    h_splitter.addWidget(left_splitter)

    # ============================================================
    # ПРАВАЯ ПАНЕЛЬ — TOOLBOX
    # ============================================================
    from ui.dataset_audit.workspace import AnalysisPages, build_overview

    parent.toolbox = AnalysisPages()
    build_overview(parent)
    h_splitter.addWidget(parent.toolbox)
    h_splitter.setStretchFactor(0, 0)
    h_splitter.setStretchFactor(1, 1)
    h_splitter.setSizes([460, 980])

    # ============================================================
