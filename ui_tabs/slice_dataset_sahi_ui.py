# slice_dataset_sahi_ui.py
from import_libs_internal import *


def setup_sahi_slicer_ui(parent):
    """SAHI slicer UI: слева — пары/валидация, справа — SAHI."""
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(8, 8, 8, 8)
    main_layout.setSpacing(8)

    # ============================================================
    # TOP TOOLBAR — COCO + YOLO + прогресс
    # ============================================================
    top_layout = QHBoxLayout()

    parent.btn_images = QPushButton("Load Images")
    parent.btn_labels = QPushButton("Load Labels")
    parent.btn_load_yaml = QPushButton("Load YOLO YAML")
    parent.btn_load_coco = QPushButton("Load COCO")
    parent.btn_add_coco = QPushButton("+ Add COCO")

    parent.btn_images.setToolTip("Выбрать папку с изображениями")
    parent.btn_labels.setToolTip("Выбрать папку с YOLO .txt")
    parent.btn_load_yaml.setToolTip(
        "Загрузить YOLO-датасет из data.yaml")
    parent.btn_load_coco.setToolTip("Загрузить COCO JSON")
    parent.btn_add_coco.setToolTip("Добавить COCO JSON к текущему")

    top_layout.addWidget(parent.btn_images)
    top_layout.addWidget(parent.btn_labels)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addWidget(parent.btn_load_coco)
    top_layout.addWidget(parent.btn_add_coco)

    parent.dataset_type_combo = QComboBox()
    parent.dataset_type_combo.addItem("Метки (bbox, obb, seg)", 0)
    parent.dataset_type_combo.setCurrentIndex(0)
    parent.dataset_type_combo.setToolTip(
        "SAHI поддерживает только метки (COCO / YOLO).")
    top_layout.addWidget(QLabel("Input type:"))
    top_layout.addWidget(parent.dataset_type_combo)

    parent.input_format_combo = QComboBox()
    parent.input_format_combo.addItem("YOLO fmt: Auto", "auto")
    parent.input_format_combo.addItem("detect (5 токенов)", "detect")
    parent.input_format_combo.addItem("OBB — 4 угла (9 токенов)", "obb_corners")
    parent.input_format_combo.addItem("OBB — xywhr (6 токенов)", "obb_xywhr")
    parent.input_format_combo.addItem("segment (чётные)", "segment")
    parent.input_format_combo.addItem("mixed (per-line)", "mixed")
    parent.input_format_combo.setCurrentIndex(0)
    parent.input_format_combo.setToolTip(
        "Формат .txt при загрузке YOLO:\n"
        "• Auto — по числу токенов в строке;\n"
        "• явный выбор убирает неоднозначность.")
    top_layout.addWidget(QLabel("Input fmt:"))
    top_layout.addWidget(parent.input_format_combo)

    top_layout.addStretch()

    parent.progress_bar = QProgressBar()
    parent.progress_bar.setMinimumWidth(220)
    parent.progress_bar.setMaximumWidth(360)
    parent.progress_bar.setMinimum(0)
    parent.progress_bar.setMaximum(100)
    parent.progress_bar.setValue(0)
    parent.progress_bar.setFormat("%p%")
    parent.progress_bar.setVisible(True)
    top_layout.addWidget(parent.progress_bar)

    parent.btn_toggle_log = QPushButton("Показать лог")
    parent.btn_toggle_log.setCheckable(True)
    parent.btn_toggle_log.setChecked(False)
    top_layout.addWidget(parent.btn_toggle_log)

    main_layout.addLayout(top_layout)

    # ============================================================
    # MAIN HORIZONTAL SPLITTER
    # ============================================================
    h_splitter = QSplitter(Qt.Horizontal)
    main_layout.addWidget(h_splitter, 1)

    # ------------------------------------------------------------
    # LEFT PANEL
    # ------------------------------------------------------------
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 5, 0)

    left_layout.addWidget(QLabel("Найденные пары (двойной клик — валидация):"))

    parent.file_list = QListWidget()
    parent.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    parent.file_list.setToolTip(
        "Ctrl+C — скопировать выделенные строки.\n"
        "Двойной клик — открыть снимок валидации.")
    left_layout.addWidget(parent.file_list, 1)

    parent.pair_count_label = QLabel("Всего пар: 0")
    parent.pair_count_label.setWordWrap(True)
    parent.pair_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    left_layout.addWidget(parent.pair_count_label)

    parent.validation_progress_label = QLabel("")
    parent.validation_progress_label.setWordWrap(True)
    parent.validation_progress_label.setStyleSheet(
        "color: #0055aa; font-weight: bold; font-size: 11pt;")
    parent.validation_progress_label.setVisible(False)
    left_layout.addWidget(parent.validation_progress_label)

    parent.validation_warning_label = QLabel("")
    parent.validation_warning_label.setWordWrap(True)
    parent.validation_warning_label.setTextInteractionFlags(
        Qt.TextSelectableByMouse)
    parent.validation_warning_label.setStyleSheet("color: #b00000;")
    parent.validation_warning_label.setVisible(False)
    left_layout.addWidget(parent.validation_warning_label)

    parent.color_stats_label = QLabel("")
    parent.color_stats_label.setWordWrap(True)
    parent.color_stats_label.setTextInteractionFlags(
        Qt.TextSelectableByMouse)
    parent.color_stats_label.setStyleSheet(
        "font-weight: bold; color: #333;")
    left_layout.addWidget(parent.color_stats_label)

    labels_group = QGroupBox("Class Labels")
    labels_vbox = QVBoxLayout(labels_group)
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(5)
    parent.class_table.setHorizontalHeaderLabels(
        ["New ID", "Original ID", "Name", "Mask pixel", "Include"])
    parent.class_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch)
    parent.class_table.setColumnHidden(3, True)
    parent.class_table.setMaximumHeight(160)
    labels_vbox.addWidget(parent.class_table)

    parent.include_empty_annotations = QCheckBox(
        "Include images with empty annotations")
    parent.include_empty_annotations.setChecked(False)
    labels_vbox.addWidget(parent.include_empty_annotations)

    left_layout.addWidget(labels_group)

    h_splitter.addWidget(left_widget)

    # ------------------------------------------------------------
    # RIGHT PANEL
    # ------------------------------------------------------------
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 0, 0, 0)

    ds_group = QGroupBox("Исходный датасет (SAHI)")
    ds_form = QFormLayout(ds_group)

    parent.coco_label = QLabel("—")
    parent.coco_label.setWordWrap(True)
    parent.coco_label.setStyleSheet("color: #888;")
    parent.coco_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    ds_form.addRow("COCO JSON:", parent.coco_label)

    parent.images_label = QLabel("—")
    parent.images_label.setWordWrap(True)
    parent.images_label.setStyleSheet("color: #888;")
    parent.images_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    ds_form.addRow("Изображения:", parent.images_label)

    parent.validation_summary_label = QLabel("")
    parent.validation_summary_label.setWordWrap(True)
    parent.validation_summary_label.setTextInteractionFlags(
        Qt.TextSelectableByMouse)
    parent.validation_summary_label.setStyleSheet(
        "font-weight: bold; color: #333;")
    ds_form.addRow(parent.validation_summary_label)

    right_layout.addWidget(ds_group)

    out_group = QGroupBox("Новый датасет (Результат)")
    out_form = QFormLayout(out_group)

    parent.output_edit = QLineEdit()
    parent.output_edit.setReadOnly(True)
    parent.output_edit.setPlaceholderText("Папка не выбрана")
    parent.btn_browse_output = QPushButton("Выбрать...")
    parent.btn_open_output = QPushButton("Открыть")
    parent.btn_open_output.setMaximumWidth(90)
    parent.btn_open_output.setEnabled(False)

    out_row = QHBoxLayout()
    out_row.addWidget(parent.output_edit, 1)
    out_row.addWidget(parent.btn_browse_output)
    out_row.addWidget(parent.btn_open_output)
    out_form.addRow("Папка:", out_row)

    right_layout.addWidget(out_group)

    par_group = QGroupBox("Параметры нарезки")
    par_form = QFormLayout(par_group)

    parent.slice_size = QSpinBox()
    parent.slice_size.setRange(64, 4096)
    parent.slice_size.setSingleStep(32)
    parent.slice_size.setValue(640)
    par_form.addRow("Размер тайла:", parent.slice_size)

    parent.overlap = QDoubleSpinBox()
    parent.overlap.setRange(0.0, 0.9)
    parent.overlap.setSingleStep(0.05)
    parent.overlap.setDecimals(2)
    parent.overlap.setValue(0.20)
    par_form.addRow("Перекрытие (0..1):", parent.overlap)

    parent.min_area_ratio = QDoubleSpinBox()
    parent.min_area_ratio.setRange(0.01, 1.0)
    parent.min_area_ratio.setSingleStep(0.05)
    parent.min_area_ratio.setDecimals(2)
    parent.min_area_ratio.setValue(0.30)
    par_form.addRow("Мин. доля площади:", parent.min_area_ratio)

    parent.keep_negative = QDoubleSpinBox()
    parent.keep_negative.setRange(0.0, 1.0)
    parent.keep_negative.setSingleStep(0.05)
    parent.keep_negative.setDecimals(2)
    parent.keep_negative.setValue(0.10)
    par_form.addRow("Доля фоновых:", parent.keep_negative)

    parent.split_ratio = QDoubleSpinBox()
    parent.split_ratio.setRange(0.0, 1.0)
    parent.split_ratio.setSingleStep(0.05)
    parent.split_ratio.setDecimals(2)
    parent.split_ratio.setValue(0.80)
    par_form.addRow("Доля train:", parent.split_ratio)

    parent.format_combo = QComboBox()
    parent.format_combo.addItem("both (COCO + YOLO)", "both")
    parent.format_combo.addItem("coco", "coco")
    parent.format_combo.addItem("yolo", "yolo")
    par_form.addRow("Формат выхода:", parent.format_combo)

    parent.annotation_format_combo = QComboBox()
    parent.annotation_format_combo.addItem("auto (по содержимому)", "auto")
    parent.annotation_format_combo.addItem("bbox (HBB)", "bbox")
    parent.annotation_format_combo.addItem(
        "segmentation (полигон)", "segmentation")
    parent.annotation_format_combo.setCurrentIndex(0)
    parent.annotation_format_combo.setToolTip(
        "Что писать в YOLO .txt и как чистить sliced_coco.json:\n\n"
        "  • auto — по содержимому:\n"
        "      – segmentation, если он есть в исходнике → пишется как есть;\n"
        "      – иначе bbox → сегментация (если была) удаляется из COCO.\n\n"
        "  • bbox (HBB) — class cx cy w h. Также:\n"
        "      – из sliced_coco.json удаляется поле segmentation;\n"
        "      – остаются только bbox-координаты.\n\n"
        "  • segmentation — class x1 y1 ... xn yn:\n"
        "      – sliced_coco.json сохраняется как есть (segmentation остаётся).")
    par_form.addRow("Формат YOLO:", parent.annotation_format_combo)

    parent.channels_combo = QComboBox()
    parent.channels_combo.addItem("1 (Grayscale)", 1)
    parent.channels_combo.addItem("3 (RGB)", 3)
    parent.channels_combo.setCurrentIndex(1)
    par_form.addRow("channels:", parent.channels_combo)

    parent.seed_spin = QSpinBox()
    parent.seed_spin.setRange(0, 99999)
    parent.seed_spin.setValue(42)
    par_form.addRow("Seed:", parent.seed_spin)

    parent.visualize_spin = QSpinBox()
    parent.visualize_spin.setRange(0, 50)
    parent.visualize_spin.setValue(0)
    par_form.addRow("Визуализировать:", parent.visualize_spin)

    right_layout.addWidget(par_group)

    parent.btn_generate = QPushButton("Сгенерировать новый датасет")
    parent.btn_generate.setStyleSheet(
        "background-color: #4CAF50; color: white; "
        "font-weight: bold; padding: 10px; font-size: 12pt;")
    parent.btn_generate.setEnabled(False)
    right_layout.addWidget(parent.btn_generate)

    parent.log_view = QTextEdit()
    parent.log_view.setReadOnly(True)
    parent.log_view.setMaximumHeight(180)
    parent.log_view.setStyleSheet(
        "font-family: Consolas, monospace; font-size: 10pt;")
    parent.log_view.setPlaceholderText(
        "Лог операций появится здесь при генерации.")
    parent.log_view.setVisible(False)
    right_layout.addWidget(parent.log_view)

    parent.status_label = QLabel("Готов к работе")
    parent.status_label.setStyleSheet("color: #333; font-weight: bold;")
    right_layout.addWidget(parent.status_label)

    h_splitter.addWidget(right_widget)
    h_splitter.setSizes([420, 780])

    # ============================================================
    # ПОДКЛЮЧЕНИЯ
    # ============================================================
    parent.btn_images.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_load_yaml.clicked.connect(parent.load_from_yaml)
    parent.btn_load_coco.clicked.connect(parent.load_from_coco)
    parent.btn_add_coco.clicked.connect(parent.add_from_coco)

    parent.btn_browse_output.clicked.connect(parent.browse_output)
    parent.btn_open_output.clicked.connect(parent.open_output_folder)
    parent.btn_generate.clicked.connect(parent.on_generate)