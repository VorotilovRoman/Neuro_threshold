from import_libs_internal import *

def setup_layout_dataset_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # Навигация
    parent.nav_widget = ImageNavigationWidget()




    # Кнопки: сохранение и режим рисования
    parent.btn_load_labels = QPushButton("Load Labels")
    parent.btn_load_yaml = QPushButton("Load YAML")
    parent.save_button = QPushButton("Save Labels")
    parent.add_rect_button = QPushButton("Drawing Mode")
    parent.add_rect_button.setCheckable(True)
    parent.add_rect_button.setChecked(True)

    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.btn_load_labels)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addStretch()
    top_layout.addWidget(parent.add_rect_button)
    top_layout.addWidget(parent.save_button)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    # Основной сплиттер
    main_splitter = QSplitter(Qt.Horizontal)

    # Левая часть: изображение (SmartGraphicsView)
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)

    parent.image_view = SmartGraphicsView()
    parent.image_view.setMinimumSize(600, 500)
    parent.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    left_layout.addWidget(parent.image_view)

    parent.info_label = QLabel("No image")
    parent.info_label.setAlignment(Qt.AlignCenter)
    left_layout.addWidget(parent.info_label)

    main_splitter.addWidget(left_widget)

    # Правая часть: список объектов и лог
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)

    objects_group = QGroupBox("Objects")
    objects_layout = QVBoxLayout()
    parent.object_list = QListWidget()
    parent.object_list.setSelectionMode(QAbstractItemView.SingleSelection)
    objects_layout.addWidget(parent.object_list)

    parent.delete_button = QPushButton("Delete Selected")
    objects_layout.addWidget(parent.delete_button)
    objects_group.setLayout(objects_layout)
    right_layout.addWidget(objects_group, 2)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    right_layout.addWidget(parent.log_widget, 1)

    main_splitter.addWidget(right_widget)
    main_splitter.setSizes([800, 400])
    main_layout.addWidget(main_splitter, 1)

    # Гистограмма (скрыта по умолчанию)
    parent.hist_canvas = FigureCanvas(Figure(figsize=(5, 2)))
    parent.hist_ax = parent.hist_canvas.figure.add_subplot(111)

    parent.hist_container = QWidget()
    hist_layout = QVBoxLayout(parent.hist_container)
    hist_layout.setContentsMargins(0, 0, 0, 0)
    hist_layout.addWidget(parent.hist_canvas)
    parent.hist_container.setVisible(False)
    main_layout.addWidget(parent.hist_container)

    # Кнопки управления логом и гистограммой
    log_hist_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    parent.toggle_hist_btn = QPushButton("Показать гистограмму")
    parent.toggle_hist_btn.setCheckable(True)
    log_hist_layout.addWidget(parent.toggle_log_btn)
    log_hist_layout.addWidget(parent.toggle_hist_btn)
    log_hist_layout.addStretch()
    main_layout.addLayout(log_hist_layout)