# layout_dataset_ui.py
from import_libs_internal import *


def setup_layout_dataset_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # ============================================================
    # Верхняя панель: навигация + загрузка + инструменты + сохранение
    # ============================================================
    parent.nav_widget = ImageNavigationWidget()

    parent.btn_load_labels = QPushButton("Load Labels")
    parent.btn_load_yaml = QPushButton("Load YAML")
    parent.save_button = QPushButton("Save Labels")
    parent.btn_save_all = QPushButton("Save All")
    parent.btn_save_all.setToolTip(
        "Сохранить аннотации всех загруженных (и изменённых) изображений.\n"
        "Файлы .txt перезаписываются в текущем формате OBB."
    )

    # --- Группа эксклюзивных инструментов рисования ---
    parent.tool_group = QButtonGroup(parent)
    parent.tool_group.setExclusive(True)

    parent.btn_tool_rect = QPushButton("📦 Box")
    parent.btn_tool_rect.setToolTip("Рисовать ограничивающий прямоугольник (detect)")

    parent.btn_tool_segment = QPushButton("✏️ Segment")
    parent.btn_tool_segment.setToolTip("Рисовать полигон сегментации (segment)")

    parent.btn_tool_obb = QPushButton("🔷 OBB")
    parent.btn_tool_obb.setToolTip("Рисовать ориентированный прямоугольник (OBB)")

    for b in (parent.btn_tool_rect, parent.btn_tool_segment, parent.btn_tool_obb):
        b.setCheckable(True)
        parent.tool_group.addButton(b)

    parent.btn_tool_rect.setChecked(True)

    # Обратная совместимость: старый код, который обращается к add_rect_button,
    # будет работать и дальше, поскольку мы просто алиасим имя.
    parent.add_rect_button = parent.btn_tool_rect

    # --- Чекбокс формата сохранения OBB ---
    parent.chk_obb_xywhr = QCheckBox("OBB → xywhr")
    parent.chk_obb_xywhr.setChecked(False)   # по умолчанию — 4 угла (текущий формат)
    parent.chk_obb_xywhr.setToolTip(
        "Формат записи OBB при сохранении в .txt:\n"
        "☐  class_index x1 y1 x2 y2 x3 y3 x4 y4   (по умолчанию)\n"
        "☑  class_index cx cy w h angle_rad        (xywhr)"
    )

    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.btn_load_labels)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addStretch()
    top_layout.addWidget(parent.btn_tool_rect)
    top_layout.addWidget(parent.btn_tool_segment)
    top_layout.addWidget(parent.btn_tool_obb)
    top_layout.addWidget(parent.chk_obb_xywhr)
    top_layout.addWidget(parent.save_button)
    top_layout.addWidget(parent.btn_save_all)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    # ============================================================
    # Основной сплиттер
    # ============================================================
    main_splitter = QSplitter(Qt.Horizontal)

    # ----- Левая часть: изображение (SmartGraphicsView) -----
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

    # ----- Правая часть: авторазметка, объекты, лог -----
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)

    # --- Группа автразметки YOLO (сверху справа) ---
    parent.auto_group = QGroupBox("Auto-label (YOLO)")
    auto_layout = QVBoxLayout(parent.auto_group)

    # Путь к модели
    auto_model_row = QHBoxLayout()
    parent.auto_model_edit = QLineEdit()
    parent.auto_model_edit.setReadOnly(True)
    parent.auto_model_edit.setPlaceholderText("Путь к .pt / .onnx…")
    parent.btn_load_auto_model = QPushButton("Загрузить")
    auto_model_row.addWidget(parent.auto_model_edit, 1)
    auto_model_row.addWidget(parent.btn_load_auto_model)
    auto_layout.addLayout(auto_model_row)

    # Параметры инференса
    auto_params = QGridLayout()

    auto_params.addWidget(QLabel("conf:"), 0, 0)
    parent.auto_conf_spin = QDoubleSpinBox()
    parent.auto_conf_spin.setRange(0.0, 1.0)
    parent.auto_conf_spin.setSingleStep(0.05)
    parent.auto_conf_spin.setDecimals(2)
    parent.auto_conf_spin.setValue(0.25)
    auto_params.addWidget(parent.auto_conf_spin, 0, 1)

    auto_params.addWidget(QLabel("iou:"), 1, 0)
    parent.auto_iou_spin = QDoubleSpinBox()
    parent.auto_iou_spin.setRange(0.0, 1.0)
    parent.auto_iou_spin.setSingleStep(0.05)
    parent.auto_iou_spin.setDecimals(2)
    parent.auto_iou_spin.setValue(0.45)
    auto_params.addWidget(parent.auto_iou_spin, 1, 1)

    auto_params.addWidget(QLabel("imgsz:"), 2, 0)
    parent.auto_imgsz_spin = QSpinBox()
    parent.auto_imgsz_spin.setRange(64, 4096)
    parent.auto_imgsz_spin.setSingleStep(32)
    parent.auto_imgsz_spin.setValue(640)
    auto_params.addWidget(parent.auto_imgsz_spin, 2, 1)

    auto_params.addWidget(QLabel("device:"), 3, 0)
    parent.auto_device_combo = QComboBox()
    parent.auto_device_combo.addItems(["auto", "cpu", "0", "mps"])
    auto_params.addWidget(parent.auto_device_combo, 3, 1)

    auto_layout.addLayout(auto_params)

    # Поведение при авторазметке
    parent.chk_auto_replace = QCheckBox("Заменить существующие аннотации")
    parent.chk_auto_replace.setChecked(False)
    parent.chk_auto_replace.setToolTip(
        "☐  Добавить найденные объекты к уже существующим.\n"
        "☑  Очистить текущие аннотации и записать только найденные."
    )
    auto_layout.addWidget(parent.chk_auto_replace)

    # Индикатор типа вывода модели (obb / seg / detect)
    parent.auto_kind_label = QLabel("Тип вывода: —")
    parent.auto_kind_label.setStyleSheet("color: #666; font-size: 11px;")
    auto_layout.addWidget(parent.auto_kind_label)

    # Кнопки запуска
    auto_btn_row = QHBoxLayout()
    parent.btn_auto_current = QPushButton("Разметить текущий")
    parent.btn_auto_all = QPushButton("Разметить все")
    auto_btn_row.addWidget(parent.btn_auto_current)
    auto_btn_row.addWidget(parent.btn_auto_all)
    auto_layout.addLayout(auto_btn_row)

    right_layout.addWidget(parent.auto_group)

    # ============================================================
    # Строка предупреждения о некорректной разметке (перенесена сюда)
    # ============================================================
    parent.validation_status_label = QLabel("")
    parent.validation_status_label.setWordWrap(True)
    parent.validation_status_label.setStyleSheet(
        "color: #b00000; font-weight: bold; padding: 4px;"
    )
    parent.validation_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.validation_status_label.setVisible(False)
    # Максимальная высота, чтобы длинные списки не «раздували» правую панель
    parent.validation_status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    right_layout.addWidget(parent.validation_status_label)

    # --- Список объектов ---
    objects_group = QGroupBox("Objects")
    objects_layout = QVBoxLayout()
    parent.object_list = QListWidget()
    parent.object_list.setSelectionMode(QAbstractItemView.SingleSelection)
    objects_layout.addWidget(parent.object_list)

    parent.delete_button = QPushButton("Delete Selected")
    objects_layout.addWidget(parent.delete_button)

    parent.delete_all_button = QPushButton("Delete All on Image")
    parent.delete_all_button.setToolTip(
        "Удалить ВСЕ аннотации на текущем снимке.\n"
        "Действие не затрагивает другие изображения."
    )
    objects_layout.addWidget(parent.delete_all_button)

    # Подсказка по горячим клавишам
    hint_label = QLabel("Ctrl+0…9 — сменить класс выделенного объекта")
    hint_label.setStyleSheet("color: #666; font-size: 10px;")
    hint_label.setWordWrap(True)
    objects_layout.addWidget(hint_label)

    objects_group.setLayout(objects_layout)
    right_layout.addWidget(objects_group, 2)

    # --- Лог ---
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    right_layout.addWidget(parent.log_widget, 1)

    main_splitter.addWidget(right_widget)
    main_splitter.setSizes([800, 400])
    main_layout.addWidget(main_splitter, 1)

    # ============================================================
    # Гистограмма (скрыта по умолчанию)
    # ============================================================
    parent.hist_canvas = FigureCanvas(Figure(figsize=(5, 2)))
    parent.hist_ax = parent.hist_canvas.figure.add_subplot(111)

    parent.hist_container = QWidget()
    hist_layout = QVBoxLayout(parent.hist_container)
    hist_layout.setContentsMargins(0, 0, 0, 0)
    hist_layout.addWidget(parent.hist_canvas)
    parent.hist_container.setVisible(False)
    main_layout.addWidget(parent.hist_container)

    # ============================================================
    # Кнопки управления логом и гистограммой
    # ============================================================
    log_hist_layout = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    parent.toggle_hist_btn = QPushButton("Показать гистограмму")
    parent.toggle_hist_btn.setCheckable(True)
    log_hist_layout.addWidget(parent.toggle_log_btn)
    log_hist_layout.addWidget(parent.toggle_hist_btn)
    log_hist_layout.addStretch()
    main_layout.addLayout(log_hist_layout)