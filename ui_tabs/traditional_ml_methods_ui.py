from import_libs_internal import *


def setup_traditional_ml_ui(parent):
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

    parent.ml_view = SmartGraphicsView()          # результат ML
    parent.ml_view.setMinimumSize(400, 400)
    parent.ml_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.ml_view, 0, 1)

    parent.morph_view = SmartGraphicsView()       # после морфологии
    parent.morph_view.setMinimumSize(400, 400)
    parent.morph_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.morph_view, 1, 0)

    parent.annotated_view = SmartGraphicsView()   # с аннотациями
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

    # Группа управления ML (гистограмма убрана)
    controls_group = QGroupBox("Traditional ML Segmentation")
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

    # Выбор модели
    model_layout = QHBoxLayout()
    model_layout.addWidget(QLabel("Model:"))
    parent.model_combo = QComboBox()
    models = ["K-Means Clustering", "MeanShift", "SVM (pixel-wise)", "Random Forest", "XGBoost", "Decision Tree"]
    parent.model_combo.addItems(models)
    model_layout.addWidget(parent.model_combo)
    controls_layout.addLayout(model_layout)

    # Параметры K-Means
    parent.kmeans_container = QWidget()
    kmeans_layout = QGridLayout(parent.kmeans_container)
    kmeans_layout.addWidget(QLabel("Number of clusters K:"), 0, 0)
    parent.kmeans_clusters = QSpinBox()
    parent.kmeans_clusters.setRange(2, 10)
    parent.kmeans_clusters.setValue(3)
    kmeans_layout.addWidget(parent.kmeans_clusters, 0, 1)
    controls_layout.addWidget(parent.kmeans_container)

    # Параметры SVM / Random Forest (общие)
    parent.svm_container = QWidget()
    svm_layout = QGridLayout(parent.svm_container)
    svm_layout.addWidget(QLabel("Kernel:"), 0, 0)
    parent.svm_kernel_combo = QComboBox()
    parent.svm_kernel_combo.addItems(["linear", "rbf", "poly"])
    svm_layout.addWidget(parent.svm_kernel_combo, 0, 1)
    svm_layout.addWidget(QLabel("C:"), 1, 0)
    parent.svm_c = QSpinBox()
    parent.svm_c.setRange(1, 100)
    parent.svm_c.setValue(10)
    svm_layout.addWidget(parent.svm_c, 1, 1)
    controls_layout.addWidget(parent.svm_container)

    # Параметры Random Forest
    parent.rf_container = QWidget()
    rf_layout = QGridLayout(parent.rf_container)
    rf_layout.addWidget(QLabel("Number of trees:"), 0, 0)
    parent.rf_trees = QSpinBox()
    parent.rf_trees.setRange(10, 200)
    parent.rf_trees.setValue(50)
    rf_layout.addWidget(parent.rf_trees, 0, 1)
    controls_layout.addWidget(parent.rf_container)

    # Параметры XGBoost
    parent.xgb_container = QWidget()
    xgb_layout = QGridLayout(parent.xgb_container)
    xgb_layout.addWidget(QLabel("Number of trees:"), 0, 0)
    parent.xgb_trees = QSpinBox()
    parent.xgb_trees.setRange(10, 200)
    parent.xgb_trees.setValue(50)
    xgb_layout.addWidget(parent.xgb_trees, 0, 1)
    controls_layout.addWidget(parent.xgb_container)

    # Скрыть контейнеры по умолчанию
    parent.kmeans_container.setVisible(True)
    parent.svm_container.setVisible(False)
    parent.rf_container.setVisible(False)
    parent.xgb_container.setVisible(False)

    # ---------- Кластеры для K-Means / MeanShift ----------
    parent.cluster_group = QGroupBox("Cluster selection (K‑Means / MeanShift)")
    cluster_layout = QVBoxLayout()
    parent.cluster_label = QLabel("Target cluster: (not trained yet)")
    parent.cluster_combo = QComboBox()
    parent.cluster_combo.setEnabled(False)

    # Горизонтальный ряд: комбобокс + кнопка поиска K
    cluster_combo_layout = QHBoxLayout()
    cluster_combo_layout.addWidget(parent.cluster_combo, 1)
    parent.find_best_k_btn = QPushButton("Find best K (elbow)")
    cluster_combo_layout.addWidget(parent.find_best_k_btn)

    cluster_layout.addWidget(parent.cluster_label)
    cluster_layout.addLayout(cluster_combo_layout)
    parent.cluster_group.setLayout(cluster_layout)
    controls_layout.addWidget(parent.cluster_group)

    # ---------- Выбор признаков (в одну строку) ----------
    features_group = QGroupBox("Feature extraction")
    features_layout = QHBoxLayout()
    parent.color_feature = QCheckBox("Intensity (gray)")
    parent.color_feature.setChecked(True)
    parent.texture_feature = QCheckBox("Texture (LBP)")
    parent.texture_feature.setChecked(False)
    parent.spatial_feature = QCheckBox("Spatial coordinates (x,y)")
    parent.spatial_feature.setChecked(False)
    features_layout.addWidget(parent.color_feature)
    features_layout.addWidget(parent.texture_feature)
    features_layout.addWidget(parent.spatial_feature)
    parent.normalize_features = QCheckBox("Normalize features (StandardScaler)")
    parent.normalize_features.setToolTip("Apply StandardScaler to features (recommended for SVM)")
    features_layout.addWidget(parent.normalize_features)
    features_layout.addStretch()
    features_group.setLayout(features_layout)
    controls_layout.addWidget(features_group)

    # ---------- Суперпиксели (в одну строку) ----------
    superpixel_group = QGroupBox("Superpixel processing")
    superpixel_layout = QHBoxLayout()
    parent.use_superpixels = QCheckBox("Use superpixels (SLIC)")
    parent.use_superpixels.setChecked(False)
    superpixel_layout.addWidget(parent.use_superpixels)
    superpixel_layout.addWidget(QLabel("Superpixel size:"))
    parent.superpixel_size = QSpinBox()
    parent.superpixel_size.setRange(10, 200)
    parent.superpixel_size.setValue(50)
    parent.superpixel_size.setEnabled(False)
    superpixel_layout.addWidget(parent.superpixel_size)
    superpixel_layout.addStretch()
    superpixel_group.setLayout(superpixel_layout)
    controls_layout.addWidget(superpixel_group)

    # Кнопки обучения, загрузки, сохранения
    train_layout = QHBoxLayout()
    parent.train_button = QPushButton("Train Model")
    parent.load_model_button = QPushButton("Load Model")
    parent.save_model_btn = QPushButton("Save Model")
    train_layout.addWidget(parent.train_button)
    train_layout.addWidget(parent.load_model_button)
    train_layout.addWidget(parent.save_model_btn)
    controls_layout.addLayout(train_layout)

    # ----- Отображение пути к модели -----
    model_path_layout = QHBoxLayout()
    model_path_layout.addWidget(QLabel("Model file:"))
    parent.model_path_edit = QLineEdit()
    parent.model_path_edit.setReadOnly(True)
    parent.model_path_edit.setPlaceholderText("No model loaded")
    model_path_layout.addWidget(parent.model_path_edit, 1)
    controls_layout.addLayout(model_path_layout)

    # Индикатор соответствия настроек и кнопка сброса
    model_status_layout = QHBoxLayout()
    parent.reset_to_model_btn = QPushButton("↻ Сбросить к настройкам модели")
    parent.reset_to_model_btn.setEnabled(False)
    model_status_layout.addWidget(parent.reset_to_model_btn)
    parent.model_mismatch_label = QLabel("⚙️ Модель не обучена")
    parent.model_mismatch_label.setStyleSheet("color: gray;")
    model_status_layout.addWidget(parent.model_mismatch_label)
    model_status_layout.addStretch()
    controls_layout.addLayout(model_status_layout)

    parent.apply_button = QPushButton("Segment Current Image")
    controls_layout.addWidget(parent.apply_button)

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
    parent.draw_combo.addItems(["None", "Segmentation (Polygon)", "Bounding Box (Detect)", "OBB (Oriented Box)"])
    draw_layout.addWidget(parent.draw_combo)
    controls_layout.addLayout(draw_layout)
    parent.hull_checkbox = QCheckBox("Use convex hull")
    parent.hull_checkbox.setChecked(False)
    controls_layout.addWidget(parent.hull_checkbox)

    # Список объектов и координаты
    coord_layout = QVBoxLayout()
    coord_layout.addWidget(QLabel("Segmented objects:"))
    parent.object_list = QListWidget()
    coord_layout.addWidget(parent.object_list)
    coord_layout.addWidget(QLabel("Coordinates:"))
    parent.coord_text = QTextEdit()
    parent.coord_text.setReadOnly(True)
    parent.coord_text.setMaximumHeight(150)
    coord_layout.addWidget(parent.coord_text)
    controls_layout.addLayout(coord_layout)

    parent.save_button = QPushButton("Save Labels")
    controls_layout.addWidget(parent.save_button)

    controls_group.setLayout(controls_layout)
    right_layout.addWidget(controls_group)
    right_layout.addStretch()   # добавляем растяжение, чтобы всё не прижималось вниз

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