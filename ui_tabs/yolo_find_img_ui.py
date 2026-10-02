from import_libs_internal import *


class setup_yolo_find_img_ui(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("FindImagesWindow")
        MainWindow.setWindowTitle("YOLO Find Images by Label")
        MainWindow.setMinimumSize(1150, 780)

        self.centralwidget = QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        main_layout = QVBoxLayout(self.centralwidget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ==================== ГОРИЗОНТАЛЬНЫЙ СПЛИТТЕР ====================
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, 1)

        # ========== ЛЕВАЯ ПАНЕЛЬ: дерево файлов ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        left_header = QHBoxLayout()
        left_header.addWidget(QLabel("Файловая система"))
        left_header.addStretch()
        self.select_folder_btn = QPushButton("Выбрать папку…")
        left_header.addWidget(self.select_folder_btn)
        left_layout.addLayout(left_header)

        self.current_folder_label = QLabel("Папка не выбрана")
        self.current_folder_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        self.current_folder_label.setWordWrap(True)
        left_layout.addWidget(self.current_folder_label)

        self.file_tree = QTreeView()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setIndentation(14)
        left_layout.addWidget(self.file_tree, 1)

        h_splitter.addWidget(left_widget)

        # ========== ПРАВАЯ ПАНЕЛЬ ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # Вертикальный сплиттер: сверху — настройки, снизу — результаты
        right_v_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_v_splitter, 1)

        # ==================== ВЕРХ: НАСТРОЙКИ ====================
        top_settings_widget = QWidget()
        top_settings_layout = QVBoxLayout(top_settings_widget)
        top_settings_layout.setContentsMargins(0, 0, 0, 0)
        top_settings_layout.setSpacing(6)

        # --- YOLO + цель поиска в одной горизонтальной строке ---
        settings_h_row = QHBoxLayout()
        settings_h_row.setSpacing(6)

        self.yolo_group = QGroupBox("Параметры YOLO")
        yolo_layout = QVBoxLayout(self.yolo_group)
        self.yolo_settings = YOLOInferenceSettings()
        yolo_layout.addWidget(self.yolo_settings)
        settings_h_row.addWidget(self.yolo_group, 3)

        self.target_group = QGroupBox("Целевая метка")
        target_layout = QVBoxLayout(self.target_group)
        target_layout.setSpacing(4)

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Class ID:"))
        self.target_class_spin = QSpinBox()
        self.target_class_spin.setRange(0, 999)
        self.target_class_spin.setValue(0)
        self.target_class_spin.setToolTip("ID класса, который нужно найти в изображениях")
        id_layout.addWidget(self.target_class_spin)
        id_layout.addStretch()
        target_layout.addLayout(id_layout)

        target_layout.addWidget(QLabel("Доступные классы модели:"))
        self.class_list_widget = QListWidget()
        self.class_list_widget.setMaximumHeight(110)
        self.class_list_widget.setToolTip("Список классов, которые умеет распознавать модель")
        target_layout.addWidget(self.class_list_widget, 1)

        settings_h_row.addWidget(self.target_group, 2)
        top_settings_layout.addLayout(settings_h_row)

        # --- Дополнительные критерии (все в одной строке) ---
        self.criteria_group = QGroupBox("Дополнительные критерии")
        criteria_layout = QHBoxLayout(self.criteria_group)
        criteria_layout.setSpacing(8)

        criteria_layout.addWidget(QLabel("Отступ от краёв (px):"))
        self.edge_margin_spin = QSpinBox()
        self.edge_margin_spin.setRange(0, 2000)
        self.edge_margin_spin.setValue(0)
        self.edge_margin_spin.setFixedWidth(90)
        self.edge_margin_spin.setToolTip(
            "0 — критерий отключён.\n"
            "N > 0 — бокс объекта должен быть не ближе N px "
            "к каждому краю кадра (сверху, снизу, слева, справа)."
        )
        criteria_layout.addWidget(self.edge_margin_spin)

        criteria_layout.addWidget(QLabel("Мин. размер (%):"))
        self.min_object_size_spin = QDoubleSpinBox()
        self.min_object_size_spin.setRange(0.0, 100.0)
        self.min_object_size_spin.setDecimals(2)
        self.min_object_size_spin.setSingleStep(0.1)
        self.min_object_size_spin.setValue(0.0)
        self.min_object_size_spin.setFixedWidth(90)
        self.min_object_size_spin.setToolTip(
            "0 — критерий отключён.\n"
            "N > 0 — площадь бокса объекта должна быть ≥ N% от площади кадра.\n"
            "Например, 0.5 — объект занимает не меньше 0.5% площади снимка."
        )
        criteria_layout.addWidget(self.min_object_size_spin)

        criteria_layout.addWidget(QLabel("Макс. размер (%):"))
        self.max_object_size_spin = QDoubleSpinBox()
        self.max_object_size_spin.setRange(0.0, 100.0)
        self.max_object_size_spin.setDecimals(2)
        self.max_object_size_spin.setSingleStep(0.1)
        self.max_object_size_spin.setValue(0.0)
        self.max_object_size_spin.setFixedWidth(90)
        self.max_object_size_spin.setToolTip(
            "0 — критерий отключён.\n"
            "N > 0 — площадь бокса объекта должна быть ≤ N% от площади кадра.\n"
            "Например, 90 — объект занимает не больше 90% площади снимка.\n"
            "Полезно, чтобы исключить кадры, где объект «занимает весь снимок»."
        )
        criteria_layout.addWidget(self.max_object_size_spin)

        criteria_layout.addStretch()
        self.criteria_group.setToolTip(
            "Все три критерия применяются одновременно. Снимок попадает в результаты, "
            "если хотя бы один объект целевого класса удовлетворяет всем активным условиям."
        )
        top_settings_layout.addWidget(self.criteria_group)

        # --- Прогресс + кнопки ---
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(6)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Готов к работе")
        progress_layout.addWidget(self.progress_bar, 1)

        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; font-weight: 600; padding: 6px 14px; }"
        )
        progress_layout.addWidget(self.stop_btn)

        self.scan_btn = QPushButton("Начать поиск")
        self.scan_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; font-weight: 600; padding: 6px 18px; }"
        )
        progress_layout.addWidget(self.scan_btn)

        top_settings_layout.addLayout(progress_layout)

        right_v_splitter.addWidget(top_settings_widget)

        # ==================== НИЗ: РЕЗУЛЬТАТЫ ====================
        bottom_results_widget = QWidget()
        bottom_results_layout = QVBoxLayout(bottom_results_widget)
        bottom_results_layout.setContentsMargins(0, 0, 0, 0)
        bottom_results_layout.setSpacing(6)

        results_group = QGroupBox("Найденные файлы")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(6)

        # --- Панель управления результатами ---
        results_top_layout = QHBoxLayout()
        results_top_layout.setSpacing(6)

        self.select_all_checkbox = QCheckBox("Выделить всё")
        self.select_all_checkbox.setToolTip("Отметить/снять все файлы")
        results_top_layout.addWidget(self.select_all_checkbox)

        self.sort_btn = QPushButton("↓ Сортировать по уверенности")
        self.sort_btn.setToolTip("Отсортировать по убыванию максимальной уверенности")
        results_top_layout.addWidget(self.sort_btn)

        results_top_layout.addStretch()

        results_top_layout.addWidget(QLabel("Размер миниатюр:"))
        self.thumb_size_combo = QComboBox()
        self.thumb_size_combo.addItems(["Мелкие", "Средние", "Крупные"])
        self.thumb_size_combo.setCurrentIndex(1)
        self.thumb_size_combo.setFixedWidth(110)
        self.thumb_size_combo.setToolTip(
            "Мелкие — ~120px, Средние — ~180px, Крупные — ~260px"
        )
        results_top_layout.addWidget(self.thumb_size_combo)

        self.toggle_view_btn = QPushButton("Режим: миниатюры")
        self.toggle_view_btn.setCheckable(True)
        self.toggle_view_btn.setToolTip("Переключить между списком и сеткой миниатюр")
        results_top_layout.addWidget(self.toggle_view_btn)

        results_layout.addLayout(results_top_layout)

        # --- Переключаемое представление ---
        self.stacked_view = QStackedWidget()
        results_layout.addWidget(self.stacked_view, 1)

        # Список
        self.results_list = QListWidget()
        self.results_list.setToolTip(
            "Файлы, в которых обнаружен объект с указанным class ID"
        )
        self.results_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.stacked_view.addWidget(self.results_list)

        # Миниатюры
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_container = QWidget()
        self.thumbnail_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.thumbnail_grid = QGridLayout(self.thumbnail_container)
        self.thumbnail_grid.setSpacing(10)
        self.thumbnail_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.thumbnail_container)
        self.stacked_view.addWidget(self.scroll_area)

        bottom_results_layout.addWidget(results_group, 1)

        # --- Панель копирования/перемещения ---
        copy_group = QGroupBox("Копирование / перемещение выбранных")
        copy_layout = QHBoxLayout(copy_group)
        copy_layout.setSpacing(6)
        copy_layout.addWidget(QLabel("Целевая папка:"))
        self.target_folder_edit = QLineEdit()
        self.target_folder_edit.setReadOnly(True)
        copy_layout.addWidget(self.target_folder_edit, 1)
        self.browse_target_btn = QPushButton("Обзор…")
        copy_layout.addWidget(self.browse_target_btn)
        self.copy_btn = QPushButton("Копировать")
        copy_layout.addWidget(self.copy_btn)
        self.move_btn = QPushButton("Переместить")
        copy_layout.addWidget(self.move_btn)
        bottom_results_layout.addWidget(copy_group)

        # --- Кнопка показать/скрыть лог ---
        self.toggle_log_btn = QPushButton("Показать лог")
        self.toggle_log_btn.setCheckable(True)
        self.toggle_log_btn.setToolTip("Показать или скрыть панель лога")
        bottom_results_layout.addWidget(self.toggle_log_btn)

        # --- Лог (скрыт по умолчанию) ---
        self.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
        self.log_widget.setVisible(False)
        self.log_widget.setMinimumHeight(140)
        bottom_results_layout.addWidget(self.log_widget)

        right_v_splitter.addWidget(bottom_results_widget)
        right_v_splitter.setStretchFactor(0, 0)
        right_v_splitter.setStretchFactor(1, 1)
        right_v_splitter.setSizes([340, 520])

        h_splitter.addWidget(right_widget)
        h_splitter.setSizes([260, 890])

        self.statusbar = MainWindow.statusBar()
        self.statusbar.showMessage("Готов")

        self.retranslateUi(MainWindow)

    def retranslateUi(self, MainWindow):
        pass