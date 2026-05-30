# threshold_methods_ui.py
from import_libs_internal import *

def setup_threshold_ui(parent):
    """
    Создаёт UI для пороговых методов внутри QMainWindow.
    parent – экземпляр QMainWindow (например, ThresholdWindow).
    """
    # Создаём центральный виджет
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    central_layout = QVBoxLayout(central_widget)
    central_layout.setContentsMargins(0, 0, 0, 0)

    # ---------- Верхняя панель: навигация по изображениям ----------
    parent.nav_widget = ImageNavigationWidget()
    parent.reset_zoom_button = QPushButton("Reset zoom in all views")

    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.reset_zoom_button)
    top_layout.addStretch()
    central_layout.addLayout(top_layout)

    # ---------- Основной сплиттер: слева изображения, справа управление ----------
    main_splitter = QSplitter(Qt.Horizontal)

    # Левая часть: 4 SmartGraphicsView в сетке 2x2
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    grid = QGridLayout()
    grid.setSpacing(10)

    parent.original_view = SmartGraphicsView()
    parent.original_view.setMinimumSize(400, 400)
    parent.original_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.original_view, 0, 0)

    parent.binary_view = SmartGraphicsView()
    parent.binary_view.setMinimumSize(400, 400)
    parent.binary_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.binary_view, 0, 1)

    parent.morph_view = SmartGraphicsView()
    parent.morph_view.setMinimumSize(400, 400)
    parent.morph_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.morph_view, 1, 0)

    parent.annotated_view = SmartGraphicsView()
    parent.annotated_view.setMinimumSize(400, 400)
    parent.annotated_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.annotated_view, 1, 1)

    grid.setRowStretch(0, 1)
    grid.setRowStretch(1, 1)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)

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

    # Группа управления
    controls_group = QGroupBox("Controls")
    controls_layout = QVBoxLayout()

    # ----- Блок пресетов -----
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

    # ----- Выбор метода бинаризации -----
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Binarization method:"))
    parent.method_combo = QComboBox()
    methods = [
        "Simple Threshold",
        "Otsu",
        "Triangle",
        "Adaptive Mean",
        "Adaptive Gauss",
        "Niblack",
        "Sauvola",
        "ISODATA",
        "Background Symmetry",
        "Row Adaptive"
    ]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # ----- Контейнеры параметров для каждого метода -----
    # 1. Simple Threshold
    parent.simple_threshold_container = QWidget()
    simple_th_layout = QHBoxLayout(parent.simple_threshold_container)
    simple_th_layout.addWidget(QLabel("Threshold value:"))
    parent.threshold_slider = QSlider(Qt.Horizontal)
    parent.threshold_slider.setRange(0, 255)
    parent.threshold_slider.setValue(127)
    parent.threshold_value_label = QLabel("127")
    simple_th_layout.addWidget(parent.threshold_slider)
    simple_th_layout.addWidget(parent.threshold_value_label)
    controls_layout.addWidget(parent.simple_threshold_container)

    # 2. Adaptive Mean / Gauss
    parent.adaptive_container = QWidget()
    adaptive_layout = QHBoxLayout(parent.adaptive_container)
    adaptive_layout.addWidget(QLabel("Window size (odd):"))
    parent.adaptive_win = QSpinBox()
    parent.adaptive_win.setRange(3, 201)
    parent.adaptive_win.setSingleStep(2)
    parent.adaptive_win.setValue(25)
    adaptive_layout.addWidget(parent.adaptive_win)
    adaptive_layout.addWidget(QLabel("C:"))
    parent.adaptive_c = QSpinBox()
    parent.adaptive_c.setRange(-20, 20)
    parent.adaptive_c.setValue(3)
    adaptive_layout.addWidget(parent.adaptive_c)
    controls_layout.addWidget(parent.adaptive_container)

    # 3. Niblack
    parent.niblack_container = QWidget()
    niblack_layout = QVBoxLayout(parent.niblack_container)
    win_layout = QHBoxLayout()
    win_layout.addWidget(QLabel("Window size (odd):"))
    parent.niblack_win = QSpinBox()
    parent.niblack_win.setRange(3, 201)
    parent.niblack_win.setSingleStep(2)
    parent.niblack_win.setValue(25)
    win_layout.addWidget(parent.niblack_win)
    niblack_layout.addLayout(win_layout)

    k_layout = QHBoxLayout()
    k_layout.addWidget(QLabel("k:"))
    parent.niblack_k = QSlider(Qt.Horizontal)
    parent.niblack_k.setRange(-100, 100)
    parent.niblack_k.setValue(20)
    parent.niblack_k_label = QLabel("0.20")
    k_layout.addWidget(parent.niblack_k)
    k_layout.addWidget(parent.niblack_k_label)
    niblack_layout.addLayout(k_layout)
    controls_layout.addWidget(parent.niblack_container)

    # 4. Sauvola
    parent.sauvola_container = QWidget()
    sauvola_layout = QVBoxLayout(parent.sauvola_container)
    win_s_layout = QHBoxLayout()
    win_s_layout.addWidget(QLabel("Window size (odd):"))
    parent.sauvola_win = QSpinBox()
    parent.sauvola_win.setRange(3, 201)
    parent.sauvola_win.setSingleStep(2)
    parent.sauvola_win.setValue(25)
    win_s_layout.addWidget(parent.sauvola_win)
    sauvola_layout.addLayout(win_s_layout)

    k_s_layout = QHBoxLayout()
    k_s_layout.addWidget(QLabel("k:"))
    parent.sauvola_k = QSlider(Qt.Horizontal)
    parent.sauvola_k.setRange(-100, 100)
    parent.sauvola_k.setValue(20)
    parent.sauvola_k_label = QLabel("0.20")
    k_s_layout.addWidget(parent.sauvola_k)
    k_s_layout.addWidget(parent.sauvola_k_label)
    sauvola_layout.addLayout(k_s_layout)

    r_layout = QHBoxLayout()
    r_layout.addWidget(QLabel("R:"))
    parent.sauvola_r = QSlider(Qt.Horizontal)
    parent.sauvola_r.setRange(1, 255)
    parent.sauvola_r.setValue(128)
    parent.sauvola_r_label = QLabel("128")
    r_layout.addWidget(parent.sauvola_r)
    r_layout.addWidget(parent.sauvola_r_label)
    sauvola_layout.addLayout(r_layout)
    controls_layout.addWidget(parent.sauvola_container)

    # 5. ISODATA
    parent.isodata_container = QWidget()
    iso_layout = QHBoxLayout(parent.isodata_container)
    iso_layout.addWidget(QLabel("Initial threshold:"))
    parent.isodata_init = QSpinBox()
    parent.isodata_init.setRange(0, 255)
    parent.isodata_init.setValue(128)
    iso_layout.addWidget(parent.isodata_init)
    controls_layout.addWidget(parent.isodata_container)

    # 6. Background Symmetry
    parent.background_container = QWidget()
    bg_layout = QVBoxLayout(parent.background_container)
    bg_thresh_layout = QHBoxLayout()
    bg_thresh_layout.addWidget(QLabel("Excess threshold (sensitivity):"))
    parent.bg_excess = QSlider(Qt.Horizontal)
    parent.bg_excess.setRange(1, 100)
    parent.bg_excess.setValue(20)
    parent.bg_excess_label = QLabel("0.20")
    bg_thresh_layout.addWidget(parent.bg_excess)
    bg_thresh_layout.addWidget(parent.bg_excess_label)
    bg_layout.addLayout(bg_thresh_layout)
    controls_layout.addWidget(parent.background_container)

    # 7. Row Adaptive
    parent.row_adaptive_container = QWidget()
    row_layout = QVBoxLayout(parent.row_adaptive_container)
    win_row_layout = QHBoxLayout()
    win_row_layout.addWidget(QLabel("Window size:"))
    parent.row_win = QSpinBox()
    parent.row_win.setRange(10, 500)
    parent.row_win.setValue(50)
    win_row_layout.addWidget(parent.row_win)
    row_layout.addLayout(win_row_layout)

    k_row_layout = QHBoxLayout()
    k_row_layout.addWidget(QLabel("k:"))
    parent.row_k = QSlider(Qt.Horizontal)
    parent.row_k.setRange(0, 100)
    parent.row_k.setValue(50)
    parent.row_k_label = QLabel("0.50")
    k_row_layout.addWidget(parent.row_k)
    k_row_layout.addWidget(parent.row_k_label)
    row_layout.addLayout(k_row_layout)
    controls_layout.addWidget(parent.row_adaptive_container)

    # Скрываем все контейнеры изначально
    parent.simple_threshold_container.setVisible(False)
    parent.adaptive_container.setVisible(False)
    parent.niblack_container.setVisible(False)
    parent.sauvola_container.setVisible(False)
    parent.isodata_container.setVisible(False)
    parent.background_container.setVisible(False)
    parent.row_adaptive_container.setVisible(False)

    # Invert mask checkbox
    parent.invert_checkbox = QCheckBox("Invert mask")
    parent.invert_checkbox.setChecked(True)
    controls_layout.addWidget(parent.invert_checkbox)

    # ----- Морфологические операции -----
    morph_label = QLabel("Morphological operations (applied after threshold/invert):")
    controls_layout.addWidget(morph_label)

    kernel_shape_layout = QHBoxLayout()
    kernel_shape_layout.addWidget(QLabel("Kernel shape:"))
    parent.kernel_shape_combo = QComboBox()
    parent.kernel_shape_combo.addItems(["Rectangle", "Ellipse", "Cross"])
    kernel_shape_layout.addWidget(parent.kernel_shape_combo)
    controls_layout.addLayout(kernel_shape_layout)

    close_layout = QHBoxLayout()
    close_layout.addWidget(QLabel("Closing factor:"))
    parent.close_kernel_slider = QSlider(Qt.Horizontal)
    parent.close_kernel_slider.setRange(0, 10)
    parent.close_kernel_slider.setValue(0)
    parent.close_kernel_label = QLabel("0.00")
    close_layout.addWidget(parent.close_kernel_slider)
    close_layout.addWidget(parent.close_kernel_label)
    controls_layout.addLayout(close_layout)

    open_layout = QHBoxLayout()
    open_layout.addWidget(QLabel("Opening factor:"))
    parent.open_kernel_slider = QSlider(Qt.Horizontal)
    parent.open_kernel_slider.setRange(0, 10)
    parent.open_kernel_slider.setValue(0)
    parent.open_kernel_label = QLabel("0.00")
    open_layout.addWidget(parent.open_kernel_slider)
    open_layout.addWidget(parent.open_kernel_label)
    controls_layout.addLayout(open_layout)

    # ----- Режим отрисовки объектов -----
    draw_layout = QHBoxLayout()
    draw_layout.addWidget(QLabel("Draw objects:"))
    parent.draw_combo = QComboBox()
    parent.draw_combo.addItems([
        "None",
        "Segmentation (Polygon)",
        "Bounding Box (Detect)",
        "OBB (Oriented Box)"
    ])
    draw_layout.addWidget(parent.draw_combo)
    controls_layout.addLayout(draw_layout)

    parent.hull_checkbox = QCheckBox("Use convex hull (for contours/rect)")
    parent.hull_checkbox.setChecked(False)
    controls_layout.addWidget(parent.hull_checkbox)

    # ----- Список объектов и координаты -----
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

    # ----- Кнопки управления -----
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
    central_layout.addWidget(main_splitter, 1)

    # ----- Контейнер гистограммы -----
    parent.hist_canvas = FigureCanvas(Figure(figsize=(5, 2)))
    parent.hist_ax = parent.hist_canvas.figure.add_subplot(111)
    parent.hist_container = QWidget()
    hist_layout = QVBoxLayout(parent.hist_container)
    hist_layout.setContentsMargins(0, 0, 0, 0)
    hist_layout.addWidget(parent.hist_canvas)
    parent.hist_container.setVisible(False)
    central_layout.addWidget(parent.hist_container)

    # ----- Кнопки управления логом и гистограммой -----
    log_hist_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    parent.toggle_hist_btn = QPushButton("Показать гистограмму")
    parent.toggle_hist_btn.setCheckable(True)
    log_hist_layout.addWidget(parent.toggle_log_btn)
    log_hist_layout.addWidget(parent.toggle_hist_btn)
    log_hist_layout.addStretch()
    central_layout.addLayout(log_hist_layout)

    # ---------- Лог выполнения ----------
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.log_widget.setVisible(False)
    central_layout.addWidget(parent.log_widget)

    # Изначально скрываем чекбокс convex hull
    parent.hull_checkbox.setVisible(False)