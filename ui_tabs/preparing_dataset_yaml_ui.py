# preparing_dataset_yaml_ui.py
from import_libs_internal import *


def setup_preparing_dataset_yaml_ui(parent):
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
    parent.btn_load_yaml.setToolTip(
        "Загрузить YOLO-датасет из data.yaml.\n"
        "Поддерживаются images/... + labels/... и txt-списки."
    )
    parent.btn_load_coco = QPushButton("Load COCO")
    parent.btn_load_coco.setToolTip(
        "Загрузить COCO JSON (annotations_*.json).\n"
        "Заменит текущий датасет.\n"
        "Можно выбрать несколько файлов или папку с JSON."
    )
    parent.btn_add_coco = QPushButton("+ Add COCO")
    parent.btn_add_coco.setToolTip(
        "Добавить COCO JSON к уже загруженному датасету.\n"
        "Категории объединяются по имени: если 'cat' уже есть,\n"
        "новый 'cat' получит тот же класс.\n"
        "Можно выбрать несколько файлов или папку с JSON."
    )
    parent.btn_load_unet = QPushButton("Load U-Net YAML")
    parent.btn_load_unet.setToolTip(
        "Загрузить U-Net-датасет (masks_data.yaml).\n"
        "Пары (image, mask) + class_names + mask_class_offset."
    )

    top_layout.addWidget(parent.btn_folder)
    top_layout.addWidget(parent.btn_labels)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addWidget(parent.btn_load_coco)
    top_layout.addWidget(parent.btn_add_coco)
    top_layout.addWidget(parent.btn_load_unet)

    # --- Input type ---
    parent.dataset_type_combo = QComboBox()
    parent.dataset_type_combo.addItem("Метки (bbox, obb, seg)", 0)
    parent.dataset_type_combo.addItem("Маски (masks)", 1)
    parent.dataset_type_combo.addItem("Только изображения", 2)
    parent.dataset_type_combo.setCurrentIndex(0)

    top_layout.addWidget(QLabel("Input type:"))
    top_layout.addWidget(parent.dataset_type_combo)

    # --- Input annotation format ---
    parent.input_format_combo = QComboBox()
    parent.input_format_combo.addItem("Auto (по содержимому)", "auto")
    parent.input_format_combo.addItem("YOLO detect (5 токенов)", "detect")
    parent.input_format_combo.addItem("YOLO OBB — 4 угла (9 токенов)", "obb_corners")
    parent.input_format_combo.addItem("YOLO OBB — xywhr (6 токенов)", "obb_xywhr")
    parent.input_format_combo.addItem("YOLO segment (≥7 токенов, чётные)", "segment")
    parent.input_format_combo.addItem("Mixed (per-line эвристика)", "mixed")
    parent.input_format_combo.setCurrentIndex(0)
    parent.input_format_combo.setToolTip(
        "Формат .txt при загрузке аннотаций.\n"
        "• Auto — эвристика по числу токенов в строке (текущее поведение).\n"
        "• Явное указание формата убирает неоднозначность:\n"
        "   4-точечный полигон (9 токенов) можно читать как segment."
    )
    top_layout.addWidget(QLabel("Input fmt:"))
    top_layout.addWidget(parent.input_format_combo)

    top_layout.addStretch()

    parent.btn_output = QPushButton("Save YAML")
    top_layout.addWidget(parent.btn_output)

    parent.output_path_display = QLineEdit()
    parent.output_path_display.setReadOnly(True)
    parent.output_path_display.setPlaceholderText("Not selected")
    parent.output_path_display.setMinimumWidth(400)
    top_layout.addWidget(parent.output_path_display)

    parent.btn_open_output = QPushButton("Open Folder")
    parent.btn_open_output.setEnabled(False)
    top_layout.addWidget(parent.btn_open_output)
    main_layout.addLayout(top_layout)

    # ============================================================
    # СПЛИТТЕРЫ
    # ============================================================
    main_v_splitter = QSplitter(Qt.Vertical)
    main_layout.addWidget(main_v_splitter, 1)

    h_splitter = QSplitter(Qt.Horizontal)
    main_v_splitter.addWidget(h_splitter)

    # ---- Левая панель: список пар ----
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 5, 0)
    left_layout.addWidget(QLabel("Найденные пары (двойной клик — снимок валидации):"))
    parent.file_list = QListWidget()
    parent.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    parent.file_list.setToolTip(
        "Ctrl+C — скопировать выделенные строки.\n"
        "Двойной клик — открыть снимок валидации этой пары."
    )
    left_layout.addWidget(parent.file_list)

    parent.pair_count_label = QLabel("Всего пар: 0")
    parent.pair_count_label.setWordWrap(True)
    parent.pair_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    left_layout.addWidget(parent.pair_count_label)

    parent.validation_progress_label = QLabel("")
    parent.validation_progress_label.setWordWrap(True)
    parent.validation_progress_label.setStyleSheet("color: #0055aa; font-weight: bold;")
    parent.validation_progress_label.setVisible(False)
    left_layout.addWidget(parent.validation_progress_label)

    parent.validation_warning_label = QLabel("")
    parent.validation_warning_label.setWordWrap(True)
    parent.validation_warning_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.validation_warning_label.setStyleSheet("color: #b00000;")
    parent.validation_warning_label.setVisible(False)
    left_layout.addWidget(parent.validation_warning_label)

    # Статистика
    parent.color_stats_label = QLabel("")
    parent.color_stats_label.setWordWrap(True)
    parent.color_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.color_stats_label.setStyleSheet("font-weight: bold; color: #333;")
    left_layout.addWidget(parent.color_stats_label)

    # Кнопки конвертации
    parent.color_convert_buttons_layout = QHBoxLayout()
    parent.btn_convert_gray = QPushButton("Конвертировать всё в ЧБ")
    parent.btn_convert_rgb = QPushButton("Конвертировать всё в RGB")
    parent.btn_convert_8bit_png = QPushButton("Перевести в 8 бит PNG")
    parent.btn_convert_gray.setEnabled(False)
    parent.btn_convert_rgb.setEnabled(False)
    parent.btn_convert_8bit_png.setEnabled(False)
    parent.btn_convert_gray.setToolTip(
        "Перезаписать ЦВЕТНЫЕ и ВИЗУАЛЬНО-СЕРЫЕ изображения в реально-ЧБ.\n"
        "Файлы, уже соответствующие целевому формату, не трогаются."
    )
    parent.btn_convert_rgb.setToolTip(
        "Перезаписать РЕАЛЬНО-ЧБ изображения в 3-канальные (BGR).\n"
        "Цветные и визуально-серые не трогаются."
    )
    parent.btn_convert_8bit_png.setToolTip(
        "Привести изображения к 8-битному PNG.\n"
        "Решает ошибку 'Unsupported depth of input image: CV_16U'.\n"
        "Маски не конвертируются — они хранятся в своём dtype."
    )
    parent.color_convert_buttons_layout.addWidget(parent.btn_convert_gray)
    parent.color_convert_buttons_layout.addWidget(parent.btn_convert_rgb)
    parent.color_convert_buttons_layout.addWidget(parent.btn_convert_8bit_png)
    left_layout.addLayout(parent.color_convert_buttons_layout)

    h_splitter.addWidget(left_widget)

    # ---- Правая панель ----
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 0, 0, 0)

    # Блок 1: Split
    split_group = QGroupBox("Train / Validation / Test Split")
    split_layout = QFormLayout()
    parent.train_slider = QSlider(Qt.Horizontal)
    parent.train_slider.setRange(0, 100)
    parent.train_slider.setValue(70)
    parent.train_slider.setTickInterval(10)
    parent.val_slider = QSlider(Qt.Horizontal)
    parent.val_slider.setRange(0, 100)
    parent.val_slider.setValue(20)
    parent.test_slider = QSlider(Qt.Horizontal)
    parent.test_slider.setRange(0, 100)
    parent.test_slider.setValue(10)

    parent.train_label = QLabel("70%")
    parent.val_label = QLabel("20%")
    parent.test_label = QLabel("10%")
    parent.train_count_label = QLabel("0")
    parent.val_count_label = QLabel("0")
    parent.test_count_label = QLabel("0")
    parent.total_count_label = QLabel("Общее кол-во: 0")

    parent.train_slider.valueChanged.connect(lambda v: parent.update_split(v, 'train'))
    parent.val_slider.valueChanged.connect(lambda v: parent.update_split(v, 'val'))
    parent.test_slider.valueChanged.connect(lambda v: parent.update_split(v, 'test'))

    for name, slider, label, count in (
            ("Train:", parent.train_slider, parent.train_label, parent.train_count_label),
            ("Validation:", parent.val_slider, parent.val_label, parent.val_count_label),
            ("Test:", parent.test_slider, parent.test_label, parent.test_count_label)):
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        row.addWidget(QLabel("("))
        row.addWidget(count)
        row.addWidget(QLabel(" снимков)"))
        split_layout.addRow(name, row)

    split_group.setLayout(split_layout)
    right_layout.addWidget(split_group)

    # Блок 2: Preprocessing
    preproc_group = QGroupBox("Preprocessing")
    preproc_layout = QFormLayout()

    parent.resize_keep = QRadioButton("Keep original size")
    parent.resize_fixed = QRadioButton("Resize to square (keep ratio, pad)")
    parent.resize_stretch = QRadioButton("Resize to square (stretch)")
    parent.resize_fixed.setChecked(True)

    parent.resize_size = QSpinBox()
    parent.resize_size.setRange(32, 4096)
    parent.resize_size.setValue(640)

    resize_layout = QHBoxLayout()
    resize_layout.addWidget(parent.resize_keep)
    resize_layout.addWidget(parent.resize_fixed)
    resize_layout.addWidget(parent.resize_stretch)
    resize_layout.addWidget(parent.resize_size)
    preproc_layout.addRow("Image size:", resize_layout)

    parent.bg_color_combo = QComboBox()
    parent.bg_color_combo.addItems(["Black", "White"])
    parent.bg_color_combo.setCurrentIndex(0)
    preproc_layout.addRow("Background fill color:", parent.bg_color_combo)

    parent.channels_combo = QComboBox()
    parent.channels_combo.addItems([
        "3 (RGB — читать всё как цветное)",
        "1 (Grayscale — читать всё как Ч/Б)",
    ])
    parent.channels_combo.setCurrentIndex(0)
    parent.channels_combo.setToolTip(
        "Значение поля channels в data.yaml.\n"
        "• 3 — любое изображение читается как 3-канальное.\n"
        "• 1 — всё читается как одноканальное (grayscale)."
    )
    preproc_layout.addRow("Channels:", parent.channels_combo)

    parent.include_empty_annotations = QCheckBox("Include images with empty annotations")
    parent.include_empty_annotations.setChecked(False)
    parent.include_empty_annotations.setToolTip(
        "Если включено, снимки с пустыми .txt (или маской только из фона) "
        "попадут в датасет как негативные примеры."
    )
    preproc_layout.addRow("", parent.include_empty_annotations)

    # NEW: минимальная площадь аннотации в пикселях²
    parent.min_ann_area_spin = QSpinBox()
    parent.min_ann_area_spin.setRange(0, 1_000_000)
    parent.min_ann_area_spin.setValue(0)
    parent.min_ann_area_spin.setSingleStep(10)
    parent.min_ann_area_spin.setSuffix(" px²")
    parent.min_ann_area_spin.setToolTip(
        "Минимальная площадь аннотации в пикселях².\n"
        "Аннотации меньше этого порога отбрасываются.\n"
        "• 0 — фильтр выключен (по умолчанию);\n"
        "• применяется как к базовым парам, так и ко всем\n"
        "  аугментированным копиям.\n"
        "• для detect площадь = ширина × высота bbox;\n"
        "• для OBB/segment — площадь полигона."
    )
    min_area_row = QHBoxLayout()
    min_area_row.addWidget(QLabel("Min annotation area:"))
    min_area_row.addWidget(parent.min_ann_area_spin)
    min_area_row.addWidget(QLabel("(0 = off)"))
    min_area_row.addStretch()
    preproc_layout.addRow("", min_area_row)

    parent.delete_source_after_success = QCheckBox(
        "Delete source files after successful generation. Исходные снимки удалятся!"
    )
    parent.delete_source_after_success.setChecked(False)
    parent.delete_source_after_success.setToolTip(
        "После успешной генерации будут удалены ТОЛЬКО те исходные "
        "изображения и аннотации/маски, которые реально попали в датасет."
    )
    preproc_layout.addRow("", parent.delete_source_after_success)

    preproc_group.setLayout(preproc_layout)
    right_layout.addWidget(preproc_group)

    # ============================================================
    # Блок 3: OUTPUT FORMAT
    # ============================================================
    output_group = QGroupBox("Output format")
    output_layout = QFormLayout()

    # YOLO txt
    parent.out_yolo_txt = QCheckBox("YOLO labels (.txt)")
    parent.out_yolo_txt.setChecked(True)
    parent.out_yolo_txt.setToolTip(
        "Сохранить .txt с аннотациями в labels/{split}.\n"
        "Нужен для YOLO-det / YOLO-obb / YOLO-seg."
    )
    output_layout.addRow("", parent.out_yolo_txt)

    parent.out_yolo_subtype = QComboBox()
    parent.out_yolo_subtype.addItem("Keep original type", "keep")
    parent.out_yolo_subtype.addItem("Convert to detect (bbox)", "detect")
    parent.out_yolo_subtype.addItem("Convert to OBB", "obb")
    parent.out_yolo_subtype.addItem("Convert to segment (polygon)", "segment")
    parent.out_yolo_subtype.setToolTip(
        "Конвертировать все аннотации в единый subtype.\n"
        "«Keep» — сохранить типы как есть (detect / obb / segment)."
    )
    parent.out_yolo_subtype_row = QHBoxLayout()
    parent.out_yolo_subtype_row.addWidget(QLabel("YOLO subtype:"))
    parent.out_yolo_subtype_row.addWidget(parent.out_yolo_subtype)
    parent.out_yolo_subtype_row.addStretch()
    output_layout.addRow("", parent.out_yolo_subtype_row)

    parent.out_yolo_obb_format = QComboBox()
    parent.out_yolo_obb_format.addItem("4 corners (x1..y4)", "corners")
    parent.out_yolo_obb_format.addItem("xywhr (cx,cy,w,h,angle)", "xywhr")
    parent.out_yolo_obb_format.setToolTip(
        "Формат записи OBB в .txt (только если subtype = keep или obb)."
    )
    parent.out_yolo_obb_row = QHBoxLayout()
    parent.out_yolo_obb_row.addWidget(QLabel("OBB format:"))
    parent.out_yolo_obb_row.addWidget(parent.out_yolo_obb_format)
    parent.out_yolo_obb_row.addStretch()
    output_layout.addRow("", parent.out_yolo_obb_row)

    # U-Net masks
    parent.out_masks = QCheckBox("U-Net masks (.png)")
    parent.out_masks.setChecked(False)
    parent.out_masks.setToolTip(
        "Сохранить одноканальные PNG-маски в masks/{split}.\n"
        "• input=labels — маски генерируются из .txt (fillPoly).\n"
        "• input=masks — маски сохраняются как есть (lossless)."
    )
    output_layout.addRow("", parent.out_masks)

    parent.out_masks_offset = QSpinBox()
    parent.out_masks_offset.setRange(0, 1)
    parent.out_masks_offset.setValue(1)
    parent.out_masks_offset.setToolTip(
        "Смещение классов в маске:\n"
        "• 1 — фон=0, класс i → pixel i+1 (U-Net-стиль, рекомендуется);\n"
        "• 0 — pixel = class_id напрямую (класс 0 = фон)."
    )
    parent.out_masks_offset_row = QHBoxLayout()
    parent.out_masks_offset_row.addWidget(QLabel("Mask class offset:"))
    parent.out_masks_offset_row.addWidget(parent.out_masks_offset)
    parent.out_masks_offset_row.addStretch()
    output_layout.addRow("", parent.out_masks_offset_row)

    parent.out_unet_yaml = QCheckBox("U-Net YAML (masks_data.yaml)")
    parent.out_unet_yaml.setChecked(False)
    parent.out_unet_yaml.setToolTip(
        "Записать masks_data.yaml со структурой U-Net-датасета.\n"
        "Требует включённого U-Net masks."
    )
    output_layout.addRow("", parent.out_unet_yaml)

    # COCO
    parent.out_coco = QCheckBox("COCO annotations_{split}.json")
    parent.out_coco.setChecked(False)
    parent.out_coco.setToolTip(
        "Записать COCO JSON в корень датасета.\n"
        "• input=labels — из .txt (lossless).\n"
        "• input=masks — через findContours (⚠️ lossy)."
    )
    output_layout.addRow("", parent.out_coco)

    # Лейбл-предупреждение о lossy
    parent.lossy_warning_label = QLabel("")
    parent.lossy_warning_label.setWordWrap(True)
    parent.lossy_warning_label.setStyleSheet("color: #b06000; font-weight: bold;")
    parent.lossy_warning_label.setVisible(False)
    output_layout.addRow("", parent.lossy_warning_label)

    output_group.setLayout(output_layout)
    right_layout.addWidget(output_group)

    # ============================================================
    # Блок 4: AUGMENTATION
    # ============================================================
    aug_group = QGroupBox("Augmentation")
    aug_layout = QGridLayout()
    parent.flip_horizontal = QCheckBox("Horizontal Flip")
    parent.flip_vertical = QCheckBox("Vertical Flip")
    aug_layout.addWidget(parent.flip_horizontal, 0, 0)
    aug_layout.addWidget(parent.flip_vertical, 0, 1)
    parent.rotate_90 = QCheckBox("Rotate 90/180/270°")
    aug_layout.addWidget(parent.rotate_90, 1, 0, 1, 2)

    parent.random_rotate = QCheckBox("Random Rotation (±)")
    parent.random_rotate_angle = QDoubleSpinBox()
    parent.random_rotate_angle.setRange(0, 15)
    parent.random_rotate_angle.setValue(5)
    parent.random_rotate_angle.setSuffix("°")
    random_rotate_layout = QHBoxLayout()
    random_rotate_layout.addWidget(parent.random_rotate)
    random_rotate_layout.addWidget(QLabel("Max angle:"))
    random_rotate_layout.addWidget(parent.random_rotate_angle)
    aug_layout.addLayout(random_rotate_layout, 2, 0, 1, 2)

    parent.shear = QCheckBox("Random Shear (±)")
    parent.shear_angle = QDoubleSpinBox()
    parent.shear_angle.setRange(0, 15)
    parent.shear_angle.setValue(5)
    parent.shear_angle.setSuffix("°")
    shear_layout = QHBoxLayout()
    shear_layout.addWidget(parent.shear)
    shear_layout.addWidget(QLabel("Max Angle:"))
    shear_layout.addWidget(parent.shear_angle)
    aug_layout.addLayout(shear_layout, 3, 0, 1, 2)

    parent.random_crop = QCheckBox("Random Crop")
    parent.random_crop.setToolTip(
        "Случайный кроп: вырезает прямоугольную область\n"
        "(от 70% до 100% кадра) из изображения и аннотаций.\n"
        "При 'Keep original size' итоговый размер снимка уменьшится.\n"
        "⚠️ Не применяется при input=masks (только flip и rot90)."
    )
    aug_layout.addWidget(parent.random_crop, 4, 0)

    parent.aug_mask_note = QLabel(
        "⚠️ При input=masks доступны только flip и rotate 90° "
        "(lossless). Остальные аугментации игнорируются."
    )
    parent.aug_mask_note.setWordWrap(True)
    parent.aug_mask_note.setStyleSheet("color: #666; font-size: 11px;")
    parent.aug_mask_note.setVisible(False)
    aug_layout.addWidget(parent.aug_mask_note, 5, 0, 1, 2)

    aug_group.setLayout(aug_layout)
    right_layout.addWidget(aug_group)

    # ============================================================
    # Блок 5: MULTIPLIER
    # ============================================================
    multiplier_group = QGroupBox("Augmentation multiplier")
    multiplier_layout = QFormLayout()
    parent.aug_multiplier_slider = QSlider(Qt.Horizontal)
    parent.aug_multiplier_slider.setRange(2, 10)
    parent.aug_multiplier_slider.setValue(2)
    parent.aug_multiplier_slider.setTickInterval(1)
    parent.aug_multiplier_slider.setEnabled(False)
    parent.aug_multiplier_label = QLabel("2x")
    multiplier_slider_layout = QHBoxLayout()
    multiplier_slider_layout.addWidget(parent.aug_multiplier_slider)
    multiplier_slider_layout.addWidget(parent.aug_multiplier_label)
    multiplier_layout.addRow("Multiply dataset by:", multiplier_slider_layout)
    parent.total_count_with_multiplier = QLabel("Итоговое кол-во снимков: 0")
    multiplier_layout.addRow(parent.total_count_with_multiplier)
    multiplier_group.setLayout(multiplier_layout)
    right_layout.addWidget(multiplier_group)

    # ============================================================
    # Блок 6: CLASS LABELS
    # ============================================================
    labels_group = QGroupBox("Class Labels")
    labels_layout = QVBoxLayout()
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(5)
    parent.class_table.setHorizontalHeaderLabels(
        ["New ID", "Original ID", "Name", "Mask pixel", "Include"])
    parent.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    # Колонка Mask pixel (индекс 3) скрыта по умолчанию,
    # включается автоматически при input=masks или out_masks=True.
    parent.class_table.setColumnHidden(3, True)
    parent.class_table.setToolTip(
        "New ID — плотная нумерация 0..N-1 (пересчитывается автоматически).\n"
        "Original ID — как в исходном датасете (только чтение).\n"
        "Name — произвольное имя класса.\n"
        "Include — включить класс в выходной датасет."
    )
    labels_layout.addWidget(parent.class_table)
    labels_group.setLayout(labels_layout)
    right_layout.addWidget(labels_group)

    # ============================================================
    # Кнопки управления + лог
    # ============================================================
    parent.control_buttons = ControlButtons(
        show_generate=True,
        show_cancel=True,
        show_save=False
    )
    parent.generate_btn = parent.control_buttons._generate_btn
    parent.cancel_btn = parent.control_buttons._cancel_btn
    parent.generate_btn.setText("Generate Dataset")
    parent.cancel_btn.setText("Cancel")
    parent.generate_btn.setEnabled(False)
    parent.cancel_btn.setEnabled(False)

    parent.control_buttons.generate.connect(parent.on_generate)
    parent.control_buttons.cancel.connect(parent.on_cancel_generation)

    log_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(False)
    log_layout.addWidget(parent.control_buttons)
    log_layout.addStretch()
    log_layout.addWidget(parent.toggle_log_btn)
    right_layout.addLayout(log_layout)

    h_splitter.addWidget(right_widget)
    h_splitter.setSizes([500, 500])

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=True)
    parent.log_text = parent.log_widget.text
    parent.progress_bar = parent.log_widget._progress
    parent.log_widget._progress.setFormat("Готово: %p%")
    parent.log_widget.setVisible(False)

    main_v_splitter.addWidget(parent.log_widget)
    main_v_splitter.setSizes([600, 150])

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    parent.hist_canvas = FigureCanvas(Figure(figsize=(5, 2)))
    parent.hist_ax = parent.hist_canvas.figure.add_subplot(111)
    parent.hist_container = QWidget()
    hist_layout = QVBoxLayout(parent.hist_container)
    hist_layout.setContentsMargins(0, 0, 0, 0)
    hist_layout.addWidget(parent.hist_canvas)
    parent.hist_container.setVisible(False)
    main_layout.addWidget(parent.hist_container)

    # ============================================================
    # ПОДКЛЮЧЕНИЯ
    # ============================================================
    parent.btn_folder.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_load_yaml.clicked.connect(parent.load_from_yaml)
    parent.btn_load_coco.clicked.connect(parent.load_from_coco)
    parent.btn_add_coco.clicked.connect(parent.add_from_coco)
    parent.btn_load_unet.clicked.connect(parent.load_from_unet_yaml)
    parent.btn_output.clicked.connect(parent.select_output_folder)

    # ============================================================
    # СИНХРОНИЗАЦИЯ СОСТОЯНИЙ
    # ============================================================
    def _sync_all_states(*_):
        """
        Единая точка управления видимостью/активностью виджетов
        в зависимости от input_type и выбранных output-флагов.
        """
        input_type = parent.dataset_type_combo.currentData()  # 0/1/2

        # --- Input format combo: активен только при input=labels ---
        parent.input_format_combo.setEnabled(input_type == 0)

        # --- Кнопка Load Labels: показывает "Mask" при input=masks ---
        if input_type == 0:
            parent.btn_labels.setText("Load Labels")
            parent.btn_labels.setEnabled(True)
        elif input_type == 1:
            parent.btn_labels.setText("Load Masks")
            parent.btn_labels.setEnabled(True)
        else:
            parent.btn_labels.setText("No labels (images only)")
            parent.btn_labels.setEnabled(False)

        # --- Output group ---
        is_images_only = (input_type == 2)

        parent.out_yolo_txt.setEnabled(not is_images_only)
        parent.out_masks.setEnabled(not is_images_only)
        parent.out_coco.setEnabled(not is_images_only)
        parent.out_unet_yaml.setEnabled(
            not is_images_only and parent.out_masks.isChecked()
        )

        # subtype combo активен только при out_yolo_txt
        subtype_active = (not is_images_only and parent.out_yolo_txt.isChecked())
        parent.out_yolo_subtype.setEnabled(subtype_active)

        # obb format активен при out_yolo_txt и subtype in ('keep', 'obb')
        subtype = parent.out_yolo_subtype.currentData()
        obb_fmt_active = (subtype_active and subtype in ('keep', 'obb'))
        parent.out_yolo_obb_format.setEnabled(obb_fmt_active)

        # mask offset активен при out_masks
        parent.out_masks_offset.setEnabled(
            not is_images_only and parent.out_masks.isChecked()
        )

        # --- Lossy warning при input=masks + (yolo_txt или coco) ---
        lossy = (input_type == 1 and
                 (parent.out_yolo_txt.isChecked() or parent.out_coco.isChecked()))
        if lossy:
            parts = []
            if parent.out_yolo_txt.isChecked():
                parts.append("YOLO labels (.txt)")
            if parent.out_coco.isChecked():
                parts.append("COCO JSON")
            parent.lossy_warning_label.setText(
                f"⚠️ Выходные форматы ({', '.join(parts)}) будут получены "
                f"через findContours из масок. Это lossy-преобразование: "
                f"теряются дырки и мелкие детали. Маски останутся "
                f"в masks/{{split}} в исходном виде."
            )
            parent.lossy_warning_label.setVisible(True)
        else:
            parent.lossy_warning_label.setVisible(False)

        # --- Augmentation: при input=masks доступны только простые ---
        is_masks_input = (input_type == 1)
        if is_masks_input:
            # Снимаем complex-чекбоксы (не удаляем — пользователь вернётся
            # к input=labels, и они снова станут доступны).
            for cb in (parent.random_rotate, parent.shear, parent.random_crop):
                cb.setEnabled(False)
            parent.random_rotate_angle.setEnabled(False)
            parent.shear_angle.setEnabled(False)
            parent.aug_mask_note.setVisible(True)
        else:
            for cb in (parent.random_rotate, parent.shear, parent.random_crop):
                cb.setEnabled(True)
            parent.random_rotate_angle.setEnabled(True)
            parent.shear_angle.setEnabled(True)
            parent.aug_mask_note.setVisible(False)

        # Обновим состояние множителя — зависит от чекбоксов аугментации.
        if hasattr(parent, 'update_multiplier_slider_state'):
            parent.update_multiplier_slider_state()
        # --- Колонка "Mask pixel": показываем, когда работаем с масками ---
        show_mask_pixel = (input_type == 1) or parent.out_masks.isChecked()
        try:
            parent.class_table.setColumnHidden(3, not show_mask_pixel)
        except Exception:
            pass
        if hasattr(parent, '_refresh_mask_pixel_column'):
            parent._refresh_mask_pixel_column()

    parent._sync_all_states = _sync_all_states

    # Подписка на изменения.
    # Первичный _sync_all_states() вызывается из DatasetPreparationWindow.__init__
    # после того, как все атрибуты окна (dataset_type и т.д.) установлены.
    parent.dataset_type_combo.currentIndexChanged.connect(_sync_all_states)
    parent.out_yolo_txt.toggled.connect(_sync_all_states)
    parent.out_masks.toggled.connect(_sync_all_states)
    parent.out_coco.toggled.connect(_sync_all_states)
    parent.out_yolo_subtype.currentIndexChanged.connect(_sync_all_states)
    parent.out_masks_offset.valueChanged.connect(_sync_all_states)