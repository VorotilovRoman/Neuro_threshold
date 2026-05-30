# deep_learning_methods_ui.py
from import_libs_internal import *


def setup_deep_learning_methods_ui(parent):
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # Навигация
    parent.nav_widget = ImageNavigationWidget()
    parent.reset_zoom_button = QPushButton("Reset zoom")
    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.reset_zoom_button)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    # Сплиттер: слева 4 вида, справа управление с прокруткой
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

    parent.dl_view = SmartGraphicsView()
    parent.dl_view.setMinimumSize(400, 400)
    parent.dl_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.dl_view, 0, 1)

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

    controls_group = QGroupBox("Deep Learning Segmentation")
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

    # Выбор модели
    model_layout = QHBoxLayout()
    model_layout.addWidget(QLabel("Model:"))
    parent.model_combo = QComboBox()
    models = ["U-Net", "DeepLabV3+", "SegFormer", "SAM", "YOLO-seg", "Custom ONNX"]
    parent.model_combo.addItems(models)
    model_layout.addWidget(parent.model_combo)
    controls_layout.addLayout(model_layout)

    # Виджеты настроек
    parent.unet_settings = UNetInferenceSettings()
    parent.unet_settings.setVisible(False)
    parent.unet_settings._encoder.setEnabled(False)
    parent.unet_settings._input_size.setEnabled(False)
    controls_layout.addWidget(parent.unet_settings)

    parent.deeplab_settings = DeepLabV3InferenceSettings()
    parent.deeplab_settings.setVisible(False)
    parent.deeplab_settings._backbone.setEnabled(False)
    parent.deeplab_settings._output_stride.setEnabled(False)
    controls_layout.addWidget(parent.deeplab_settings)

    parent.segformer_settings = SegFormerInferenceSettings()
    parent.segformer_settings.setVisible(False)
    parent.segformer_settings._variant.setEnabled(False)
    controls_layout.addWidget(parent.segformer_settings)

    parent.sam_settings = SAMInferenceSettings()
    parent.sam_settings.setVisible(False)
    parent.sam_settings._model_type.setEnabled(False)
    parent.sam_settings._prompt.setEnabled(False)
    controls_layout.addWidget(parent.sam_settings)

    parent.yolo_settings = YOLOInferenceSettings()
    parent.yolo_settings.setVisible(False)
    parent.yolo_settings._conf.setEnabled(True)
    parent.yolo_settings._iou.setEnabled(True)
    controls_layout.addWidget(parent.yolo_settings)

    parent.custom_settings = CustomONNXInferenceSettings()
    parent.custom_settings.setVisible(False)
    controls_layout.addWidget(parent.custom_settings)

    # Общий выбор устройства
    device_container = QWidget()
    device_layout = QHBoxLayout(device_container)
    device_layout.addWidget(QLabel("Device:"))
    parent.device_combo = QComboBox()
    parent.device_combo.addItems(["Auto (best available)", "CPU", "CUDA (GPU)", "MPS (Apple Silicon)"])
    device_layout.addWidget(parent.device_combo)
    device_layout.addStretch()
    controls_layout.addWidget(device_container)

    # Кнопка сегментации и прогресс
    parent.segment_button = QPushButton("Segment Current Image")
    controls_layout.addWidget(parent.segment_button)
    parent.progress_bar = QProgressBar()
    controls_layout.addWidget(parent.progress_bar)

    # Invert mask
    parent.invert_checkbox = QCheckBox("Invert mask")
    controls_layout.addWidget(parent.invert_checkbox)

    # Морфология
    morph_label = QLabel("Post-processing morphology:")
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

    # Отрисовка объектов
    draw_layout = QHBoxLayout()
    draw_layout.addWidget(QLabel("Draw objects:"))
    parent.draw_combo = QComboBox()
    parent.draw_combo.addItems(["None", "Segmentation (Polygon)", "Bounding Box (Detect)", "OBB (Oriented Box)"])
    draw_layout.addWidget(parent.draw_combo)
    controls_layout.addLayout(draw_layout)

    parent.hull_checkbox = QCheckBox("Use convex hull")
    parent.hull_checkbox.setChecked(False)
    controls_layout.addWidget(parent.hull_checkbox)

    # Список объектов и координаты
    coord_layout = QVBoxLayout()
    coord_layout.addWidget(QLabel("Detected objects:"))
    parent.object_list = QListWidget()
    coord_layout.addWidget(parent.object_list)
    coord_layout.addWidget(QLabel("Details:"))
    parent.coord_text = QTextEdit()
    parent.coord_text.setReadOnly(True)
    parent.coord_text.setMaximumHeight(150)
    coord_layout.addWidget(parent.coord_text)
    controls_layout.addLayout(coord_layout)

    parent.save_button = QPushButton("Save Labels")
    controls_layout.addWidget(parent.save_button)

    controls_group.setLayout(controls_layout)
    right_layout.addWidget(controls_group)
    right_layout.addStretch()   # равномерное распределение

    right_scroll_area.setWidget(right_content)

    main_splitter.addWidget(left_widget)
    main_splitter.addWidget(right_scroll_area)
    main_splitter.setSizes([400, 500])   # начальные пропорции левой и правой части
    main_layout.addWidget(main_splitter, 1)

    # Гистограмма
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

    # Кнопки лога/гистограммы
    log_hist_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    parent.toggle_hist_btn = QPushButton("Показать гистограмму")
    parent.toggle_hist_btn.setCheckable(True)
    log_hist_layout.addWidget(parent.toggle_log_btn)
    log_hist_layout.addWidget(parent.toggle_hist_btn)
    log_hist_layout.addStretch()
    main_layout.addLayout(log_hist_layout)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.log_widget.setVisible(False)
    main_layout.addWidget(parent.log_widget)

    parent.on_model_changed(0)   # вызов для первоначальной инициализации видимости