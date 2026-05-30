from import_libs_internal import *

def setup_demo_ensemble_methods_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)

    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(6)

    # Горизонтальный сплиттер (левая и правая панели)
    h_splitter = QSplitter(Qt.Horizontal)
    main_layout.addWidget(h_splitter, 1)

    # -------------------------------------------------------------
    # LEFT PANEL – только изображения (без лога и прогресса)
    # -------------------------------------------------------------
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(0)

    # Сетка изображений
    grid_widget = QWidget()
    grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid_layout = QVBoxLayout(grid_widget)
    grid_layout.setContentsMargins(4, 4, 4, 4)
    grid_layout.setSpacing(10)

    grid = QGridLayout()
    grid.setSpacing(10)

    views_config = [
        ("Исходное + истинные координаты", "preview_view", "groundtruth_analysis_text"),
        ("YOLO инференс + координаты из модели", "model_result_view", "model_analysis_text"),
        ("Пресет + координаты по пресету", "preset_result_view", "preset_analysis_text"),
        ("Ансамбль + координаты по ансамблю", "ensemble_result_view", "ensemble_analysis_text")
    ]

    parent.view_widgets = {}
    parent.text_widgets = {}

    for idx, (title, view_attr, text_attr) in enumerate(views_config):
        row, col = divmod(idx, 2)
        group = QGroupBox()
        group.setTitle(title)
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)

        # Кнопка скрытия текстового поля (сброс масштаба вынесен в общую кнопку)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        toggle_btn = QToolButton()
        toggle_btn.setText("▼")
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(False)
        toggle_btn.setToolTip("Показать/скрыть текстовые аннотации")
        btn_layout.addWidget(toggle_btn)

        group_layout.addLayout(btn_layout)

        view = SmartGraphicsView()
        view.setMinimumSize(400, 400)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setStyleSheet("background-color: #2b2b2b;")
        group_layout.addWidget(view, stretch=1)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier New")
        text_edit.setMaximumHeight(150)
        text_edit.setVisible(False)
        group_layout.addWidget(text_edit)

        setattr(parent, view_attr, view)
        setattr(parent, text_attr, text_edit)
        parent.view_widgets[view_attr] = view
        parent.text_widgets[text_attr] = text_edit

        def make_toggle(text_widget, btn):
            def toggle():
                visible = btn.isChecked()
                text_widget.setVisible(visible)
                btn.setText("▲" if visible else "▼")
            return toggle
        toggle_btn.clicked.connect(make_toggle(text_edit, toggle_btn))

        grid.addWidget(group, row, col)

    grid.setRowStretch(0, 1)
    grid.setRowStretch(1, 1)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    grid_layout.addLayout(grid)
    left_layout.addWidget(grid_widget, stretch=1)

    h_splitter.addWidget(left_widget)

    # -------------------------------------------------------------
    # RIGHT PANEL (с прокруткой) – все настройки + лог и прогресс
    # -------------------------------------------------------------
    right_scroll_area = QScrollArea()
    right_scroll_area.setWidgetResizable(True)
    right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    right_content = QWidget()
    right_layout = QVBoxLayout(right_content)
    right_layout.setContentsMargins(4, 4, 4, 4)
    right_layout.setSpacing(8)

    # ----- Информация о режиме -----
    info_label = QLabel("<b>⚠️ Внимание:</b> Данная вкладка предназначена для <b>детекции объектов</b> (bounding boxes), "
                        "а не сегментации. Используются модели YOLO Detection.")
    info_label.setWordWrap(True)
    right_layout.addWidget(info_label)

    # MODEL SETTINGS (YOLO Inference)
    parent.yolo_settings = YOLOInferenceSettings()
    right_layout.addWidget(parent.yolo_settings)

    # Общий выбор устройства
    device_container = QWidget()
    device_layout = QHBoxLayout(device_container)
    device_layout.addWidget(QLabel("Device:"))
    parent.device_combo = QComboBox()
    parent.device_combo.addItems(["Auto (best available)", "CPU", "CUDA (GPU)", "MPS (Apple Silicon)"])
    device_layout.addWidget(parent.device_combo)
    device_layout.addStretch()
    right_layout.addWidget(device_container)

    # PRESETS
    preset_group = QGroupBox("Пресеты обработки")
    preset_layout = QVBoxLayout(preset_group)
    preset_select_layout = QHBoxLayout()
    parent.preset_combo = QComboBox()
    parent.preset_combo.setToolTip("Выбор набора параметров обработки.")
    preset_select_layout.addWidget(QLabel("Активный пресет"))
    preset_select_layout.addWidget(parent.preset_combo)
    preset_select_layout.addStretch()
    preset_layout.addLayout(preset_select_layout)

    parent.preset_info_label = QLabel("Описание пресета")
    parent.preset_info_label.setStyleSheet("QLabel { color: #888; font-style: italic; }")
    parent.preset_info_label.setToolTip("Краткое описание выбранной конфигурации.")
    preset_layout.addWidget(parent.preset_info_label)
    right_layout.addWidget(preset_group)

    # ENSEMBLE SETTINGS
    ensemble_group = QGroupBox("Ансамбль детекций")
    ensemble_layout = QVBoxLayout(ensemble_group)

    parent.weights_container = QVBoxLayout()
    ensemble_layout.addLayout(parent.weights_container)

    # YOLO weight row
    parent.yolo_weight_layout = QHBoxLayout()
    parent.yolo_weight_label = QLabel("Вес  YOLO")
    parent.yolo_weight_spin = QDoubleSpinBox()
    parent.yolo_weight_spin.setRange(0.0, 2.0)
    parent.yolo_weight_spin.setSingleStep(0.05)
    parent.yolo_weight_spin.setValue(0.5)
    parent.yolo_weight_spin.setToolTip("Коэффициент вклада модели YOLO.")
    parent.yolo_weight_layout.addWidget(parent.yolo_weight_label)
    parent.yolo_weight_layout.addWidget(parent.yolo_weight_spin)
    parent.yolo_weight_layout.addStretch()
    parent.weights_container.addLayout(parent.yolo_weight_layout)

    # Заголовок таблицы весов пресетов
    header_layout = QHBoxLayout()
    header_layout.addWidget(QLabel("Пресет"))
    header_layout.addWidget(QLabel("            Вес"))
    header_layout.addWidget(QLabel("        Уверенность"))
    header_layout.addStretch()
    parent.weights_container.addLayout(header_layout)

    parent.preset_weights_container = QVBoxLayout()
    parent.weights_container.addLayout(parent.preset_weights_container)

    parent.total_weight_label = QLabel("Сумма весов: 0.50")
    parent.total_weight_label.setAlignment(Qt.AlignCenter)
    #parent.total_weight_label.setStyleSheet("QLabel { font-weight: 600; color: #27ae60; }")
    ensemble_layout.addWidget(parent.total_weight_label)

    parent.normalize_btn = QPushButton("Нормализовать")
    ensemble_layout.addWidget(parent.normalize_btn)
    parent.normalize_btn.clicked.connect(parent.normalize_weights)

    ensemble_layout.addWidget(QLabel(""))

    # Пороги в одной строке: уверенность, затем IoU
    thresholds_row = QHBoxLayout()
    thresholds_row.addWidget(QLabel("Порог уверенности:"))
    parent.ensemble_conf = QDoubleSpinBox()
    parent.ensemble_conf.setRange(0.0, 1.0)
    parent.ensemble_conf.setValue(0.1)
    parent.ensemble_conf.setSingleStep(0.05)
    thresholds_row.addWidget(parent.ensemble_conf)

    thresholds_row.addSpacing(20)
    thresholds_row.addWidget(QLabel("Порог IoU (NMS):"))
    parent.ensemble_iou = QDoubleSpinBox()
    parent.ensemble_iou.setRange(0.01, 1.0)
    parent.ensemble_iou.setValue(0.5)
    parent.ensemble_iou.setSingleStep(0.05)
    thresholds_row.addWidget(parent.ensemble_iou)
    thresholds_row.addStretch()
    ensemble_layout.addLayout(thresholds_row)

    ensemble_containment_row = QHBoxLayout()
    parent.ensemble_use_containment = QCheckBox("Учитывать вложенность (центр одного бокса внутри другого)")
    parent.ensemble_use_containment.setChecked(False)
    ensemble_containment_row.addWidget(parent.ensemble_use_containment)
    ensemble_containment_row.addStretch()
    ensemble_layout.addLayout(ensemble_containment_row)

    ensemble_method_row = QHBoxLayout()
    ensemble_method_row.addWidget(QLabel("Метод объединения:"))
    parent.ensemble_method = QComboBox()
    parent.ensemble_method.addItems(["WBF (Weighted Boxes Fusion)", "NMS (Non-Maximum Suppression)"])
    parent.ensemble_method.currentTextChanged.connect(parent.on_ensemble_method_changed)
    ensemble_method_row.addWidget(parent.ensemble_method)
    ensemble_method_row.addStretch()
    ensemble_layout.addLayout(ensemble_method_row)

    right_layout.addWidget(ensemble_group)

    # SOURCE (ImageNavigationWidget)
    source_group = QGroupBox("Источник данных")
    source_layout = QVBoxLayout(source_group)
    parent.nav_widget = ImageNavigationWidget()
    source_layout.addWidget(parent.nav_widget)
    parent.current_file_label = QLabel("")
    source_layout.addWidget(parent.current_file_label)
    right_layout.addWidget(source_group)

    # CONTROL BUTTONS
    btn_layout = QHBoxLayout()
    parent.run_yolo_btn = QPushButton("Запустить YOLO")
    parent.run_yolo_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: 600; }")
    parent.run_ensemble_btn = QPushButton("Запустить ансамбль")
    parent.reset_zoom_all_btn = QPushButton("Reset zoom in all views")
    parent.reset_zoom_all_btn.setToolTip("Сбросить масштаб во всех четырёх окнах")
    btn_layout.addWidget(parent.run_yolo_btn)
    btn_layout.addWidget(parent.run_ensemble_btn)
    btn_layout.addWidget(parent.reset_zoom_all_btn)
    btn_layout.addStretch()
    right_layout.addLayout(btn_layout)

    # -------------------------------------------------------------
    # Log and Progress Bar (перенесены в правую панель)
    # -------------------------------------------------------------
    log_progress_widget = QWidget()
    log_progress_layout = QVBoxLayout(log_progress_widget)
    log_progress_layout.setContentsMargins(0, 4, 0, 4)
    log_progress_layout.setSpacing(4)

    parent.infer_progress = QProgressBar()
    parent.infer_progress.setRange(0, 100)
    parent.infer_progress.setFormat("Обработка: 0%")
    parent.infer_progress.setToolTip("Прогресс выполнения инференса.")
    log_progress_layout.addWidget(parent.infer_progress)

    log_toggle_layout = QHBoxLayout()
    parent.log_toggle_btn = QToolButton()
    parent.log_toggle_btn.setText("▼ Лог выполнения")
    parent.log_toggle_btn.setCheckable(True)
    parent.log_toggle_btn.setChecked(True)
    parent.log_toggle_btn.setToolTip("Показать / скрыть лог выполнения.")
    log_toggle_layout.addWidget(parent.log_toggle_btn)
    log_toggle_layout.addStretch()
    log_progress_layout.addLayout(log_toggle_layout)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.log_text.setToolTip("Технический лог выполнения операций.")
    log_progress_layout.addWidget(parent.log_widget, 1)

    def toggle_log():
        visible = parent.log_toggle_btn.isChecked()
        parent.log_widget.setVisible(visible)
        parent.log_toggle_btn.setText("▼ Лог выполнения" if visible else "► Лог выполнения")
    parent.log_toggle_btn.clicked.connect(toggle_log)

    right_layout.addWidget(log_progress_widget)
    right_layout.addStretch()

    right_scroll_area.setWidget(right_content)
    h_splitter.addWidget(right_scroll_area)
    h_splitter.setSizes([350, 550])