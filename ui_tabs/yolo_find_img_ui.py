from import_libs_internal import *

class setup_yolo_find_img_ui(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("FindImagesWindow")
        MainWindow.setWindowTitle("YOLO Find Images by Label")
        MainWindow.setMinimumSize(1000, 700)

        self.centralwidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        main_layout = QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # Горизонтальный сплиттер
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1)

        # ========== ЛЕВАЯ ПАНЕЛЬ: дерево файлов ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.select_folder_btn = QPushButton("Выбрать папку")
        left_layout.addWidget(self.select_folder_btn)

        self.file_tree = QTreeView()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(14)
        left_layout.addWidget(self.file_tree)

        h_splitter.addWidget(left_widget)

        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        # Группа настроек YOLO
        self.yolo_group = QGroupBox("Параметры YOLO")
        yolo_layout = QVBoxLayout(self.yolo_group)
        self.yolo_settings = YOLOInferenceSettings()
        yolo_layout.addWidget(self.yolo_settings)
        right_layout.addWidget(self.yolo_group)

        # Группа цели поиска
        self.target_group = QGroupBox("Целевая метка")
        target_layout = QVBoxLayout(self.target_group)

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Class ID:"))
        self.target_class_spin = QSpinBox()
        self.target_class_spin.setRange(0, 999)
        self.target_class_spin.setValue(0)
        self.target_class_spin.setToolTip("ID класса, который нужно найти (0, 1, 2, ...)")
        id_layout.addWidget(self.target_class_spin)
        id_layout.addStretch()
        target_layout.addLayout(id_layout)

        self.class_list_label = QLabel("Доступные классы модели:")
        self.class_list_widget = QListWidget()
        self.class_list_widget.setMaximumHeight(150)
        self.class_list_widget.setToolTip("Список классов, которые умеет распознавать модель")
        target_layout.addWidget(self.class_list_label)
        target_layout.addWidget(self.class_list_widget)

        right_layout.addWidget(self.target_group)

        # Прогресс-бар и кнопки управления
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Готов к работе")
        progress_layout.addWidget(self.progress_bar)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; }")
        progress_layout.addWidget(self.stop_btn)
        right_layout.addLayout(progress_layout)

        # Кнопка запуска сканирования
        self.scan_btn = QPushButton("Начать поиск")
        self.scan_btn.setStyleSheet("QPushButton { background-color: #3498db; color: white; font-weight: 600; }")
        right_layout.addWidget(self.scan_btn)

        # Список результатов (найденные файлы) с чекбоксами и переключением режима
        results_group = QGroupBox("Найденные файлы (содержат метку)")
        results_layout = QVBoxLayout(results_group)

        # Верхняя панель с чекбоксом "All", кнопкой сортировки и кнопкой переключения режима
        results_top_layout = QHBoxLayout()
        self.select_all_checkbox = QCheckBox("All")
        self.select_all_checkbox.setToolTip("Выделить/снять все файлы")
        results_top_layout.addWidget(self.select_all_checkbox)

        self.sort_btn = QPushButton("Сортировать по убыванию уверенности")
        self.sort_btn.setToolTip("Пересортировать список файлов по максимальной уверенности предсказания")
        results_top_layout.addWidget(self.sort_btn)

        results_top_layout.addStretch()
        self.toggle_view_btn = QPushButton("Режим: миниатюры")
        self.toggle_view_btn.setCheckable(True)
        self.toggle_view_btn.setToolTip("Переключить отображение между списком и миниатюрами")
        results_top_layout.addWidget(self.toggle_view_btn)
        results_layout.addLayout(results_top_layout)

        # QStackedWidget для двух режимов
        self.stacked_view = QStackedWidget()
        results_layout.addWidget(self.stacked_view, 1)

        # Страница 0: список
        self.results_list = QListWidget()
        self.results_list.setToolTip("Файлы, в которых обнаружен объект с указанным class ID")
        self.results_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.stacked_view.addWidget(self.results_list)

        # Страница 1: миниатюры
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_container = QWidget()
        self.thumbnail_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.thumbnail_grid = QGridLayout(self.thumbnail_container)
        self.thumbnail_grid.setSpacing(10)
        self.thumbnail_grid.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.thumbnail_container)
        self.stacked_view.addWidget(self.scroll_area)

        # Панель копирования/перемещения
        copy_group = QGroupBox("Копирование/перемещение выбранных файлов")
        copy_layout = QVBoxLayout(copy_group)

        target_folder_layout = QHBoxLayout()
        target_folder_layout.addWidget(QLabel("Целевая папка:"))
        self.target_folder_edit = QLineEdit()
        self.target_folder_edit.setReadOnly(True)
        target_folder_layout.addWidget(self.target_folder_edit)
        self.browse_target_btn = QPushButton("Обзор")
        target_folder_layout.addWidget(self.browse_target_btn)
        copy_layout.addLayout(target_folder_layout)

        copy_buttons_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Копировать")
        self.move_btn = QPushButton("Переместить")
        copy_buttons_layout.addWidget(self.copy_btn)
        copy_buttons_layout.addWidget(self.move_btn)
        copy_buttons_layout.addStretch()
        copy_layout.addLayout(copy_buttons_layout)

        results_layout.addWidget(copy_group)
        right_layout.addWidget(results_group, 1)

        # Лог выполнения
        self.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
        right_layout.addWidget(self.log_widget)

        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([300, 700])

        self.statusbar = MainWindow.statusBar()
        self.statusbar.showMessage("Готов")

        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        pass