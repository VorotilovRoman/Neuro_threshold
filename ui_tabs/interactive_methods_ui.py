# interactive_methods_ui.py
from import_libs_internal import *

def setup_interactive_ui(parent):
    """Создаёт UI для вкладки интерактивных методов сегментации."""
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # Верхняя панель: навигация
    parent.nav_widget = ImageNavigationWidget()
    parent.reset_zoom_button = QPushButton("Reset zoom in all views")
    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.reset_zoom_button)
    parent.pan_zoom_btn = QPushButton("Pan/Zoom")
    parent.pan_zoom_btn.setCheckable(True)
    top_layout.addWidget(parent.pan_zoom_btn)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    # Основной сплиттер: слева 4 вида, справа управление (с прокруткой)
    main_splitter = QSplitter(Qt.Horizontal)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    grid = QGridLayout()
    grid.setSpacing(10)

    parent.original_view = SmartGraphicsView()
    parent.original_view.setMinimumSize(400, 400)
    parent.original_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.original_view, 0, 0)

    parent.segmentation_view = SmartGraphicsView()
    parent.segmentation_view.setMinimumSize(400, 400)
    parent.segmentation_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.segmentation_view, 0, 1)

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

    # Группа управления (без гистограммы)
    controls_group = QGroupBox("Interactive Segmentation Controls")
    controls_layout = QVBoxLayout()

    # ---------- Пресеты ----------
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
    method_layout.addWidget(QLabel("Method:"))
    parent.method_combo = QComboBox()
    methods = ["GrabCut", "Lazy Snapping", "SuperCut", "OneCut", "Random Walker", "Watershed (marker)", "Active Contours"]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # Группа для отображения требований
    parent.requirements_frame = QFrame()
    parent.requirements_frame.setFrameShape(QFrame.StyledPanel)
    req_layout = QHBoxLayout(parent.requirements_frame)
    parent.requirements_label = QLabel("Требования:")
    parent.requirements_label.setStyleSheet("font-weight: bold;")
    req_layout.addWidget(parent.requirements_label)
    parent.requirements_text = QLabel("")
    parent.requirements_text.setWordWrap(True)
    req_layout.addWidget(parent.requirements_text)
    controls_layout.addWidget(parent.requirements_frame)

    # Параметры методов
    from ui.interactive_params_ui import InteractiveParametersUI
    parent.params_widget = InteractiveParametersUI(parent)
    controls_layout.addWidget(parent.params_widget)

    # ---------- Инструменты интерактивного ввода ----------
    input_group = QGroupBox("Interactive Input")
    input_layout = QVBoxLayout()

    row1 = QHBoxLayout()
    parent.rect_mode_btn = QPushButton("Draw Bounding Box")
    parent.rect_mode_btn.setCheckable(True)
    parent.fg_scribble_btn = QPushButton("Scribble FG")
    parent.fg_scribble_btn.setCheckable(True)
    parent.bg_scribble_btn = QPushButton("Scribble BG")
    parent.bg_scribble_btn.setCheckable(True)
    row1.addWidget(parent.rect_mode_btn)
    row1.addWidget(parent.fg_scribble_btn)
    row1.addWidget(parent.bg_scribble_btn)
    input_layout.addLayout(row1)

    row2 = QHBoxLayout()
    parent.reset_mask_btn = QPushButton("Reset Input")
    parent.run_seg_btn = QPushButton("Run Segmentation")
    row2.addWidget(parent.reset_mask_btn)
    row2.addWidget(parent.run_seg_btn)
    input_layout.addLayout(row2)

    input_group.setLayout(input_layout)
    controls_layout.addWidget(input_group)

    # Invert mask
    parent.invert_checkbox = QCheckBox("Invert mask")
    controls_layout.addWidget(parent.invert_checkbox)

    # Морфологические операции
    morph_label = QLabel("Morphological operations (applied after segmentation):")
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

    # Режим отрисовки объектов
    draw_layout = QHBoxLayout()
    draw_layout.addWidget(QLabel("Draw objects:"))
    parent.draw_combo = QComboBox()
    parent.draw_combo.addItems([
        "None",
        "Contours (simple)",
        "Projections (connected components)",
        "Min area rectangle"
    ])
    draw_layout.addWidget(parent.draw_combo)
    controls_layout.addLayout(draw_layout)

    parent.hull_checkbox = QCheckBox("Use convex hull (for contours/rect)")
    parent.hull_checkbox.setChecked(False)
    controls_layout.addWidget(parent.hull_checkbox)

    # Список объектов и координаты
    coord_layout = QVBoxLayout()
    coord_layout.addWidget(QLabel("Segmented objects:"))
    parent.object_list = QListWidget()
    parent.object_list.setSelectionMode(QAbstractItemView.NoSelection)
    coord_layout.addWidget(parent.object_list)

    coord_layout.addWidget(QLabel("Selected object coordinates:"))
    parent.coord_text = QTextEdit()
    parent.coord_text.setReadOnly(True)
    parent.coord_text.setMaximumHeight(150)
    coord_layout.addWidget(parent.coord_text)
    controls_layout.addLayout(coord_layout)

    # Кнопка сохранения аннотаций
    parent.save_button = QPushButton("Save Labels")
    controls_layout.addWidget(parent.save_button)

    controls_group.setLayout(controls_layout)
    right_layout.addWidget(controls_group)
    right_layout.addStretch()   # добавляем растяжение, чтобы всё не прижималось вниз

    right_scroll_area.setWidget(right_content)

    main_splitter.addWidget(left_widget)
    main_splitter.addWidget(right_scroll_area)
    main_splitter.setSizes([400, 500])   # начальные пропорции (левая больше правой)
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