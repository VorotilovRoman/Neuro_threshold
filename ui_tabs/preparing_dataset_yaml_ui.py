from import_libs_internal import *

def setup_preparing_dataset_yaml_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # --- Верхняя панель с кнопками и типом датасета ---
    top_layout = QHBoxLayout()
    parent.btn_folder = QPushButton("Load Images")
    parent.btn_labels = QPushButton("Load Labels")
    top_layout.addWidget(parent.btn_folder)
    top_layout.addWidget(parent.btn_labels)

    parent.dataset_type_combo = QComboBox()
    parent.dataset_type_combo.addItems(["Метки (bbox, obb, seg)", "Маски (masks)", "Только изображения"])
    parent.dataset_type_combo.setCurrentIndex(0)
    def on_dataset_type_changed(idx):
        if idx == 0:
            parent.btn_labels.setText("Load Labels")
            parent.btn_labels.setEnabled(True)
        elif idx == 1:
            parent.btn_labels.setText("Load Masks")
            parent.btn_labels.setEnabled(True)
        else:  # idx == 2: только изображения
            parent.btn_labels.setText("No labels (images only)")
            parent.btn_labels.setEnabled(False)  # кнопка неактивна
    parent.dataset_type_combo.currentIndexChanged.connect(on_dataset_type_changed)
    top_layout.addWidget(QLabel("Dataset type:"))
    top_layout.addWidget(parent.dataset_type_combo)
    top_layout.addStretch()

    parent.btn_output = QPushButton("Save YAML")
    top_layout.addWidget(parent.btn_output)
    parent.output_path_display = QLineEdit()
    parent.output_path_display.setReadOnly(True)
    parent.output_path_display.setPlaceholderText("Not selected")
    parent.output_path_display.setMinimumWidth(400)
    top_layout.addWidget(parent.output_path_display)
    main_layout.addLayout(top_layout)

    # Добавить:
    parent.btn_open_output = QPushButton("Open Folder")
    parent.btn_open_output.setEnabled(False)  # станет активным после выбора папки

    # Разместить в top_layout, например, после output_path_display:
    top_layout.addWidget(parent.output_path_display)
    top_layout.addWidget(parent.btn_open_output)

    # --- Вертикальный сплиттер: основной контент (вверху) и лог (внизу) ---
    main_v_splitter = QSplitter(Qt.Vertical)
    main_layout.addWidget(main_v_splitter, 1)

    # ---- Верхняя часть: горизонтальный сплиттер (список слева, параметры справа) ----
    h_splitter = QSplitter(Qt.Horizontal)
    main_v_splitter.addWidget(h_splitter)

    # ---- Левая панель: список файлов и счётчик ----
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 5, 0)
    left_layout.addWidget(QLabel("Найденные пары:"))
    parent.file_list = QListWidget()
    left_layout.addWidget(parent.file_list)
    parent.pair_count_label = QLabel("Всего пар: 0")
    left_layout.addWidget(parent.pair_count_label)
    h_splitter.addWidget(left_widget)

    # ---- Правая панель: параметры ----
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 0, 0, 0)

    # Блок 1: Train/Val/Test Split
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

    train_layout = QHBoxLayout()
    train_layout.addWidget(parent.train_slider)
    train_layout.addWidget(parent.train_label)
    train_layout.addWidget(QLabel("("))
    train_layout.addWidget(parent.train_count_label)
    train_layout.addWidget(QLabel(" снимков)"))
    split_layout.addRow("Train:", train_layout)

    val_layout = QHBoxLayout()
    val_layout.addWidget(parent.val_slider)
    val_layout.addWidget(parent.val_label)
    val_layout.addWidget(QLabel("("))
    val_layout.addWidget(parent.val_count_label)
    val_layout.addWidget(QLabel(" снимков)"))
    split_layout.addRow("Validation:", val_layout)

    test_layout = QHBoxLayout()
    test_layout.addWidget(parent.test_slider)
    test_layout.addWidget(parent.test_label)
    test_layout.addWidget(QLabel("("))
    test_layout.addWidget(parent.test_count_label)
    test_layout.addWidget(QLabel(" снимков)"))
    split_layout.addRow("Test:", test_layout)

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
    preproc_group.setLayout(preproc_layout)
    right_layout.addWidget(preproc_group)

    # Блок 3: Augmentation
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
    parent.random_crop.setEnabled(False)   # отключаем, т.к. функционал не реализован
    parent.random_crop.setToolTip("Функционал временно недоступен")
    aug_layout.addWidget(parent.random_crop, 4, 0)
    aug_group.setLayout(aug_layout)
    right_layout.addWidget(aug_group)

    # Блок 4: Augmentation multiplier
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

    # Блок 5: Class Labels
    labels_group = QGroupBox("Class Labels")
    labels_layout = QVBoxLayout()
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(4)
    parent.class_table.setHorizontalHeaderLabels(["New ID", "Original ID", "Name", "Include"])
    parent.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    labels_layout.addWidget(parent.class_table)
    labels_group.setLayout(labels_layout)
    right_layout.addWidget(labels_group)

    # --- Кнопки управления ---
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
    parent.toggle_log_btn.setCheckable(True)
    log_layout.addWidget(parent.control_buttons)
    log_layout.addStretch()
    log_layout.addWidget(parent.toggle_log_btn)
    right_layout.addLayout(log_layout)

    h_splitter.addWidget(right_widget)
    h_splitter.setSizes([400, 500])

    # --- Нижняя панель: лог и прогресс ---
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=True)
    parent.log_text = parent.log_widget.text
    parent.progress_bar = parent.log_widget._progress
    parent.log_widget._progress.setFormat("Готово: %p%")
    parent.log_widget.setVisible(False)

    main_v_splitter.addWidget(parent.log_widget)
    main_v_splitter.setSizes([600, 150])

    # ----- Контейнер гистограммы (изначально скрыт) -----
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

    # Подключение кнопок выбора папок
    parent.btn_folder.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_output.clicked.connect(parent.select_output_folder)