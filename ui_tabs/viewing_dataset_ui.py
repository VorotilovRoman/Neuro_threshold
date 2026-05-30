from import_libs_internal import *

def setup_viewing_dataset_ui(parent):
    """Создаёт UI для просмотра датасета с поддержкой аннотаций, масок и YAML."""
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # ----- Верхняя панель с навигацией, загрузкой и настройками -----
    top_layout = QHBoxLayout()

    # Виджет навигации (загрузка папки/изображений, Prev/Next, resize)
    parent.nav_widget = ImageNavigationWidget()
    top_layout.addWidget(parent.nav_widget)

    # Кнопки загрузки
    parent.btn_load_labels = QPushButton("Load Labels")
    parent.btn_load_masks = QPushButton("Load Masks")
    parent.btn_load_yaml = QPushButton("Load YAML")
    parent.btn_save = QPushButton("Save Labels")

    top_layout.addWidget(parent.btn_load_labels)
    top_layout.addWidget(parent.btn_load_masks)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addWidget(parent.btn_save)

    top_layout.addStretch()

    # Слайдер прозрачности маски
    top_layout.addWidget(QLabel("Mask opacity:"))
    parent.opacity_slider = QSlider(Qt.Horizontal)
    parent.opacity_slider.setRange(0, 100)
    parent.opacity_slider.setValue(50)
    parent.opacity_slider.setFixedWidth(100)
    top_layout.addWidget(parent.opacity_slider)
    parent.opacity_label = QLabel("50%")
    parent.opacity_label.setFixedWidth(40)
    top_layout.addWidget(parent.opacity_label)

    # Информация о текущем изображении
    parent.info_label = QLabel("No images")
    parent.info_label.setAlignment(Qt.AlignCenter)
    parent.info_label.setMinimumWidth(150)
    top_layout.addWidget(parent.info_label)

    main_layout.addLayout(top_layout)

    # ----- Основной сплиттер (изображение слева, список и лог справа) -----
    main_splitter = QSplitter(Qt.Horizontal)

    # Левая часть: изображение
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)

    parent.viewer = SmartGraphicsView()
    parent.viewer.setMinimumSize(600, 500)
    parent.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    left_layout.addWidget(parent.viewer)

    main_splitter.addWidget(left_widget)

    # Правая часть: список объектов + лог
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)

    # Группа "Bounding boxes"
    objects_group = QGroupBox("Bounding boxes")
    objects_layout = QVBoxLayout()
    parent.coord_list = QListWidget()
    parent.coord_list.setSelectionMode(QAbstractItemView.SingleSelection)
    objects_layout.addWidget(parent.coord_list)

    # Горизонтальный ряд кнопок
    button_row = QHBoxLayout()
    parent.btn_delete = QPushButton("Delete Selected")
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    button_row.addWidget(parent.btn_delete)
    button_row.addWidget(parent.toggle_log_btn)
    button_row.addStretch()          # прижимает кнопки к левому краю (опционально)
    objects_layout.addLayout(button_row)
    objects_group.setLayout(objects_layout)
    right_layout.addWidget(objects_group, 2)  # растягивается

    # Лог-виджет
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_widget.text.setMaximumHeight(150)
    right_layout.addWidget(parent.log_widget, 1)

    main_splitter.addWidget(right_widget)
    main_splitter.setSizes([800, 400])  # начальные размеры
    main_layout.addWidget(main_splitter, 1)  # растягивается

    # ----- Гистограмма (скрыта по умолчанию) -----
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    parent.hist_canvas = FigureCanvas(Figure(figsize=(5, 2)))
    parent.hist_ax = parent.hist_canvas.figure.add_subplot(111)

    parent.hist_container = QWidget()
    hist_layout = QVBoxLayout(parent.hist_container)
    hist_layout.setContentsMargins(0, 0, 0, 0)
    hist_layout.addWidget(parent.hist_canvas)
    parent.hist_container.setVisible(False)
    main_layout.addWidget(parent.hist_container)
