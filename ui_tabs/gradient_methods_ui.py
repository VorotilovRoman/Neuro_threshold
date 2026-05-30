from import_libs_internal import *

def setup_gradient_ui(parent):
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # Навигация
    parent.nav_widget = ImageNavigationWidget()
    parent.reset_zoom_button = QPushButton("Reset zoom in all views")
    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.reset_zoom_button)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    main_splitter = QSplitter(Qt.Horizontal)

    # Левая часть: 6 видов в сетке 3x2
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    grid = QGridLayout()
    grid.setSpacing(10)

    # Ряд 1
    parent.original_view = SmartGraphicsView()
    parent.original_view.setMinimumSize(400, 300)
    parent.original_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.original_view, 0, 0)

    parent.gradient_view = SmartGraphicsView()
    parent.gradient_view.setMinimumSize(400, 300)
    parent.gradient_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.gradient_view, 0, 1)

    parent.binary_view = SmartGraphicsView()
    parent.binary_view.setMinimumSize(400, 300)
    parent.binary_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.binary_view, 0, 2)

    # Ряд 2
    parent.morph_view = SmartGraphicsView()
    parent.morph_view.setMinimumSize(400, 300)
    parent.morph_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.morph_view, 1, 0)

    parent.filled_view = SmartGraphicsView()
    parent.filled_view.setMinimumSize(400, 300)
    parent.filled_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.filled_view, 1, 1)

    parent.annotated_view = SmartGraphicsView()
    parent.annotated_view.setMinimumSize(400, 300)
    parent.annotated_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.annotated_view, 1, 2)

    grid.setRowStretch(0, 1)
    grid.setRowStretch(1, 1)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(2, 1)

    left_layout.addLayout(grid)

    parent.info_label = QLabel("No images")
    parent.info_label.setAlignment(Qt.AlignCenter)
    left_layout.addWidget(parent.info_label)

    # ----- ПРАВАЯ ПАНЕЛЬ С ПРОКРУТКОЙ -----
    right_scroll_area = QScrollArea()
    right_scroll_area.setWidgetResizable(True)
    right_scroll_area.setFrameShape(QFrame.NoFrame)

    right_content = QWidget()
    right_layout = QVBoxLayout(right_content)
    right_layout.setContentsMargins(0, 0, 0, 0)

    controls_group = QGroupBox("Gradient Methods Controls")
    controls_layout = QVBoxLayout()

    # Пресеты
    presets_layout = QHBoxLayout()
    presets_layout.addWidget(QLabel("Пресеты:"))
    parent.preset_combo = QComboBox()
    parent.preset_combo.setEditable(False)
    parent.preset_combo.setMinimumWidth(150)
    presets_layout.addWidget(parent.preset_combo)
    parent.save_preset_btn = QPushButton("Сохранить как пресет")
    presets_layout.addWidget(parent.save_preset_btn)
    parent.delete_preset_btn = QPushButton("Удалить пресет")
    presets_layout.addWidget(parent.delete_preset_btn)
    controls_layout.addLayout(presets_layout)

    # Выбор метода
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Gradient method:"))
    parent.method_combo = QComboBox()
    methods = ["Sobel", "Scharr", "Laplacian", "Canny", "Prewitt", "Roberts", "Kirsch"]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # Контейнеры параметров
    parent.sobel_container = QWidget()
    sobel_layout = QHBoxLayout(parent.sobel_container)
    sobel_layout.addWidget(QLabel("Kernel size (odd):"))
    parent.sobel_kernel = QSpinBox()
    parent.sobel_kernel.setRange(1, 31)
    parent.sobel_kernel.setSingleStep(2)
    parent.sobel_kernel.setValue(3)
    sobel_layout.addWidget(parent.sobel_kernel)
    sobel_layout.addWidget(QLabel("Scale:"))
    parent.sobel_scale = QSpinBox()
    parent.sobel_scale.setRange(1, 10)
    parent.sobel_scale.setValue(1)
    sobel_layout.addWidget(parent.sobel_scale)
    controls_layout.addWidget(parent.sobel_container)

    parent.laplacian_container = QWidget()
    lap_layout = QHBoxLayout(parent.laplacian_container)
    lap_layout.addWidget(QLabel("Kernel size (odd):"))
    parent.lap_kernel = QSpinBox()
    parent.lap_kernel.setRange(1, 31)
    parent.lap_kernel.setSingleStep(2)
    parent.lap_kernel.setValue(3)
    lap_layout.addWidget(parent.lap_kernel)
    lap_layout.addWidget(QLabel("Scale:"))
    parent.lap_scale = QSpinBox()
    parent.lap_scale.setRange(1, 10)
    parent.lap_scale.setValue(1)
    lap_layout.addWidget(parent.lap_scale)
    controls_layout.addWidget(parent.laplacian_container)

    parent.canny_container = QWidget()
    canny_layout = QGridLayout(parent.canny_container)
    canny_layout.addWidget(QLabel("Threshold1:"), 0, 0)
    parent.canny_thresh1 = QSlider(Qt.Horizontal)
    parent.canny_thresh1.setRange(0, 255)
    parent.canny_thresh1.setValue(50)
    parent.canny_label1 = QLabel("50")
    canny_layout.addWidget(parent.canny_thresh1, 0, 1)
    canny_layout.addWidget(parent.canny_label1, 0, 2)
    canny_layout.addWidget(QLabel("Threshold2:"), 1, 0)
    parent.canny_thresh2 = QSlider(Qt.Horizontal)
    parent.canny_thresh2.setRange(0, 255)
    parent.canny_thresh2.setValue(150)
    parent.canny_label2 = QLabel("150")
    canny_layout.addWidget(parent.canny_thresh2, 1, 1)
    canny_layout.addWidget(parent.canny_label2, 1, 2)
    canny_layout.addWidget(QLabel("Aperture size (odd):"), 2, 0)
    parent.canny_aperture = QSpinBox()
    parent.canny_aperture.setRange(3, 7)
    parent.canny_aperture.setSingleStep(2)
    parent.canny_aperture.setValue(3)
    canny_layout.addWidget(parent.canny_aperture, 2, 1)
    controls_layout.addWidget(parent.canny_container)

    parent.prewitt_container = QWidget()
    prewitt_layout = QHBoxLayout(parent.prewitt_container)
    prewitt_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.prewitt_container)

    parent.roberts_container = QWidget()
    roberts_layout = QHBoxLayout(parent.roberts_container)
    roberts_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.roberts_container)

    parent.kirsch_container = QWidget()
    kirsch_layout = QHBoxLayout(parent.kirsch_container)
    kirsch_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.kirsch_container)

    # Скрыть контейнеры
    parent.sobel_container.setVisible(True)
    parent.laplacian_container.setVisible(False)
    parent.canny_container.setVisible(False)
    parent.prewitt_container.setVisible(False)
    parent.roberts_container.setVisible(False)
    parent.kirsch_container.setVisible(False)

    # Порог бинаризации
    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(QLabel("Gradient threshold (0-255):"))
    parent.threshold_slider = QSlider(Qt.Horizontal)
    parent.threshold_slider.setRange(0, 255)
    parent.threshold_slider.setValue(127)
    parent.threshold_label = QLabel("127")
    threshold_layout.addWidget(parent.threshold_slider)
    threshold_layout.addWidget(parent.threshold_label)
    controls_layout.addLayout(threshold_layout)

    # Морфологические операции
    morph_label = QLabel("Morphological operations (applied after binarization):")
    controls_layout.addWidget(morph_label)
    kernel_shape_layout = QHBoxLayout()
    kernel_shape_layout.addWidget(QLabel("Kernel shape:"))
    parent.kernel_shape_combo = QComboBox()
    parent.kernel_shape_combo.addItems(["Rectangle", "Ellipse", "Cross"])
    kernel_shape_layout.addWidget(parent.kernel_shape_combo)
    controls_layout.addLayout(kernel_shape_layout)

    # Closing (0-0.05, шаг 0.005)
    close_layout = QHBoxLayout()
    close_layout.addWidget(QLabel("Closing factor:"))
    parent.close_kernel_slider = QSlider(Qt.Horizontal)
    parent.close_kernel_slider.setRange(0, 10)
    parent.close_kernel_slider.setSingleStep(1)
    parent.close_kernel_slider.setValue(0)
    parent.close_kernel_label = QLabel("0.00")
    close_layout.addWidget(parent.close_kernel_slider)
    close_layout.addWidget(parent.close_kernel_label)
    controls_layout.addLayout(close_layout)

    # Opening (0-0.05, шаг 0.005)
    open_layout = QHBoxLayout()
    open_layout.addWidget(QLabel("Opening factor:"))
    parent.open_kernel_slider = QSlider(Qt.Horizontal)
    parent.open_kernel_slider.setRange(0, 10)
    parent.open_kernel_slider.setSingleStep(1)
    parent.open_kernel_slider.setValue(0)
    parent.open_kernel_label = QLabel("0.00")
    open_layout.addWidget(parent.open_kernel_slider)
    open_layout.addWidget(parent.open_kernel_label)
    controls_layout.addLayout(open_layout)

    # Слайдер минимальной площади (0.0-0.1%, шаг 0.001)
    min_area_layout = QHBoxLayout()
    min_area_layout.addWidget(QLabel("Min contour area (% of image, 0 = off, 0.0-0.1%):"))
    parent.min_area_spinbox = QDoubleSpinBox()
    parent.min_area_spinbox.setRange(0.0, 0.1)
    parent.min_area_spinbox.setSingleStep(0.002)
    parent.min_area_spinbox.setDecimals(3)
    parent.min_area_spinbox.setValue(0.0)
    parent.min_area_spinbox.setSuffix("%")
    min_area_layout.addWidget(parent.min_area_spinbox)
    controls_layout.addLayout(min_area_layout)

    # Чекбокс заливки контуров
    parent.fill_contours_checkbox = QCheckBox("Fill closed contours")
    parent.fill_contours_checkbox.setChecked(True)
    controls_layout.addWidget(parent.fill_contours_checkbox)

    # Инверсия маски
    parent.invert_checkbox = QCheckBox("Invert mask")
    parent.invert_checkbox.setChecked(False)
    controls_layout.addWidget(parent.invert_checkbox)

    # Режим отрисовки объектов
    draw_layout = QHBoxLayout()
    draw_layout.addWidget(QLabel("Draw objects:"))
    parent.draw_combo = QComboBox()
    parent.draw_combo.addItems(["None", "Segmentation (Polygon)", "Bounding Box (Detect)", "OBB (Oriented Box)"])
    draw_layout.addWidget(parent.draw_combo)
    controls_layout.addLayout(draw_layout)
    parent.hull_checkbox = QCheckBox("Use convex hull (for contours/rect)")
    parent.hull_checkbox.setChecked(False)
    controls_layout.addWidget(parent.hull_checkbox)

    # Список объектов и координаты
    coord_layout = QVBoxLayout()
    coord_layout.addWidget(QLabel("Objects:"))
    parent.object_list = QListWidget()
    parent.object_list.setSelectionMode(QAbstractItemView.NoSelection)
    coord_layout.addWidget(parent.object_list)
    coord_layout.addWidget(QLabel("Selected object coordinates:"))
    parent.coord_text = QTextEdit()
    parent.coord_text.setReadOnly(True)
    parent.coord_text.setMaximumHeight(150)
    coord_layout.addWidget(parent.coord_text)
    controls_layout.addLayout(coord_layout)

    # Кнопки
    buttons_layout = QHBoxLayout()
    parent.save_button = QPushButton("Save Labels")
    parent.auto_button = QPushButton("Запустить автоподбор")
    buttons_layout.addWidget(parent.save_button)
    buttons_layout.addWidget(parent.auto_button)
    controls_layout.addLayout(buttons_layout)
    auto_config_layout = QHBoxLayout()
    parent.auto_enable_checkbox = QCheckBox("Enable auto research")
    parent.configure_auto_button = QPushButton("Configure auto research")
    auto_config_layout.addWidget(parent.auto_enable_checkbox)
    auto_config_layout.addWidget(parent.configure_auto_button)
    controls_layout.addLayout(auto_config_layout)
    parent.search_all_checkbox = QCheckBox("Search all images")
    controls_layout.addWidget(parent.search_all_checkbox)

    controls_group.setLayout(controls_layout)
    right_layout.addWidget(controls_group)
    right_layout.addStretch()   # добавляем растяжение, чтобы содержимое не прижималось вниз

    right_scroll_area.setWidget(right_content)

    main_splitter.addWidget(left_widget)
    main_splitter.addWidget(right_scroll_area)
    main_splitter.setSizes([400, 500])   # начальные пропорции левой и правой части
    main_layout.addWidget(main_splitter, 1)

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

    # ----- Кнопки управления логом и гистограммой -----
    log_hist_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    parent.toggle_hist_btn = QPushButton("Показать гистограмму")
    parent.toggle_hist_btn.setCheckable(True)
    log_hist_layout.addWidget(parent.toggle_log_btn)
    log_hist_layout.addWidget(parent.toggle_hist_btn)
    log_hist_layout.addStretch()
    main_layout.addLayout(log_hist_layout)

    # Лог (скрыт по умолчанию)
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.log_widget.setVisible(False)
    main_layout.addWidget(parent.log_widget)

    parent.hull_checkbox.setVisible(False)