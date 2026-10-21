# slice_dataset_sahi_ui.py
from import_libs_internal import *


def setup_sahi_slicer_ui(parent):
    """Создаёт все виджеты для SahiSlicerWindow."""
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(8, 8, 8, 8)
    main_layout.setSpacing(8)

    # ============================================================
    # ВЕРХНЯЯ ПАНЕЛЬ — загрузка и toggle лога
    # ============================================================
    top_layout = QHBoxLayout()

    # Кнопки загрузки, как на вкладке анализа
    parent.btn_load_coco = QPushButton("Load COCO JSON")
    parent.btn_load_coco.setStyleSheet("font-weight: bold;")
    parent.btn_load_images = QPushButton("Load Images")
    parent.btn_load_images.setStyleSheet("font-weight: bold;")

    top_layout.addWidget(parent.btn_load_coco)
    top_layout.addWidget(parent.btn_load_images)
    top_layout.addStretch()

    parent.btn_toggle_log = QPushButton("Показать лог")
    parent.btn_toggle_log.setCheckable(True)
    parent.btn_toggle_log.setChecked(False)
    parent.btn_toggle_log.setToolTip("Показать/скрыть лог операций")
    top_layout.addWidget(parent.btn_toggle_log)

    main_layout.addLayout(top_layout)

    # ============================================================
    # ИСХОДНЫЙ ДАТАСЕТ
    # ============================================================
    ds_group = QGroupBox("Исходный датасет")
    ds_layout = QFormLayout(ds_group)

    parent.coco_label = QLabel("—")
    parent.coco_label.setWordWrap(True)
    parent.coco_label.setStyleSheet("color: #888;")
    parent.coco_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    ds_layout.addRow("COCO JSON:", parent.coco_label)

    parent.images_label = QLabel("—")
    parent.images_label.setWordWrap(True)
    parent.images_label.setStyleSheet("color: #888;")
    parent.images_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    ds_layout.addRow("Изображения:", parent.images_label)

    main_layout.addWidget(ds_group)

    # ============================================================
    # НОВЫЙ ДАТАСЕТ
    # ============================================================
    out_group = QGroupBox("Новый датасет")
    out_layout = QFormLayout(out_group)

    parent.output_edit = QLineEdit()
    parent.output_edit.setReadOnly(True)
    parent.output_edit.setPlaceholderText("Папка не выбрана")
    parent.btn_browse_output = QPushButton("Выбрать...")
    parent.btn_open_output = QPushButton("Открыть")
    parent.btn_open_output.setMaximumWidth(90)
    parent.btn_open_output.setEnabled(False)
    parent.btn_open_output.setToolTip("Открыть папку последнего созданного датасета")

    out_row = QHBoxLayout()
    out_row.addWidget(parent.output_edit, 1)
    out_row.addWidget(parent.btn_browse_output)
    out_row.addWidget(parent.btn_open_output)
    out_layout.addRow("Папка:", out_row)

    main_layout.addWidget(out_group)

    # ============================================================
    # ПАРАМЕТРЫ НАРЕЗКИ
    # ============================================================
    par_group = QGroupBox("Параметры нарезки")
    par_layout = QFormLayout(par_group)

    parent.slice_size = QSpinBox()
    parent.slice_size.setRange(64, 4096)
    parent.slice_size.setSingleStep(32)
    parent.slice_size.setValue(640)
    parent.slice_size.setToolTip(
        "Размер тайла (высота=ширина), в пикселях.\n"
        "Рекомендуется 512–640 для мелких объектов.")
    par_layout.addRow("Размер тайла:", parent.slice_size)

    parent.overlap = QDoubleSpinBox()
    parent.overlap.setRange(0.0, 0.9)
    parent.overlap.setSingleStep(0.05)
    parent.overlap.setDecimals(2)
    parent.overlap.setValue(0.20)
    parent.overlap.setToolTip(
        "Доля перекрытия соседних тайлов, 0..0.9.\n"
        "Обычно 0.2–0.3 — чтобы не разрезать объекты на стыках.")
    par_layout.addRow("Перекрытие (0..1):", parent.overlap)

    parent.min_area_ratio = QDoubleSpinBox()
    parent.min_area_ratio.setRange(0.01, 1.0)
    parent.min_area_ratio.setSingleStep(0.05)
    parent.min_area_ratio.setDecimals(2)
    parent.min_area_ratio.setValue(0.30)
    parent.min_area_ratio.setToolTip(
        "Минимальная доля площади объекта после обрезки.\n"
        "0.30 = объекты, обрезанные более чем на 70%, отбрасываются.\n"
        "0.50 = обрезка не более 50%.")
    par_layout.addRow("Мин. доля площади объекта:", parent.min_area_ratio)

    parent.keep_negative = QDoubleSpinBox()
    parent.keep_negative.setRange(0.0, 1.0)
    parent.keep_negative.setSingleStep(0.05)
    parent.keep_negative.setDecimals(2)
    parent.keep_negative.setValue(0.10)
    parent.keep_negative.setToolTip(
        "Доля фоновых (пустых) тайлов, которую оставить.\n"
        "0.10 = оставить 10%, 1.0 = оставить все, 0 = удалить все.")
    par_layout.addRow("Доля фоновых тайлов:", parent.keep_negative)

    parent.split_ratio = QDoubleSpinBox()
    parent.split_ratio.setRange(0.0, 1.0)
    parent.split_ratio.setSingleStep(0.05)
    parent.split_ratio.setDecimals(2)
    parent.split_ratio.setValue(0.80)
    parent.split_ratio.setToolTip(
        "Доля train. 0.8 = 80/20; 0 = без разделения (всё в одну папку).")
    par_layout.addRow("Доля train:", parent.split_ratio)

    parent.format_combo = QComboBox()
    parent.format_combo.addItem("both (COCO + YOLO)", "both")
    parent.format_combo.addItem("coco", "coco")
    parent.format_combo.addItem("yolo", "yolo")
    par_layout.addRow("Формат выхода:", parent.format_combo)

    parent.channels_combo = QComboBox()
    parent.channels_combo.addItem("1 (Grayscale)", 1)
    parent.channels_combo.addItem("3 (RGB)", 3)
    parent.channels_combo.setCurrentIndex(1)
    parent.channels_combo.setToolTip("Поле channels в data.yaml (для YOLO).")
    par_layout.addRow("channels:", parent.channels_combo)

    parent.seed_spin = QSpinBox()
    parent.seed_spin.setRange(0, 99999)
    parent.seed_spin.setValue(42)
    parent.seed_spin.setToolTip("Seed для воспроизводимости (split + субдискретизация фона).")
    par_layout.addRow("Seed:", parent.seed_spin)

    parent.visualize_spin = QSpinBox()
    parent.visualize_spin.setRange(0, 50)
    parent.visualize_spin.setValue(0)
    parent.visualize_spin.setToolTip(
        "Сколько случайных тайлов визуализировать с bbox для проверки.\n"
        "0 = не визуализировать.")
    par_layout.addRow("Визуализировать тайлов:", parent.visualize_spin)

    main_layout.addWidget(par_group)

    # ============================================================
    # КНОПКА ГЕНЕРАЦИИ
    # ============================================================
    parent.btn_generate = QPushButton("Сгенерировать новый датасет")
    parent.btn_generate.setStyleSheet(
        "background-color: #4CAF50; color: white; "
        "font-weight: bold; padding: 10px; font-size: 12pt;")
    parent.btn_generate.setEnabled(False)
    main_layout.addWidget(parent.btn_generate)

    # ============================================================
    # ЛОГ (скрыт по умолчанию)
    # ============================================================
    parent.log_view = QTextEdit()
    parent.log_view.setReadOnly(True)
    parent.log_view.setMaximumHeight(220)
    parent.log_view.setStyleSheet(
        "font-family: Consolas, monospace; font-size: 10pt;")
    parent.log_view.setPlaceholderText(
        "Лог операций появится здесь при генерации.")
    parent.log_view.setVisible(False)
    main_layout.addWidget(parent.log_view)

    # ============================================================
    # СТАТУС-БАР: статус-текст + прогресс
    # ============================================================
    status_row = QHBoxLayout()
    parent.status_label = QLabel("Готов к работе")
    parent.status_label.setStyleSheet("color: #555;")
    status_row.addWidget(parent.status_label, 1)

    parent.progress_bar = QProgressBar()
    parent.progress_bar.setMaximumWidth(320)
    parent.progress_bar.setMinimum(0)
    parent.progress_bar.setMaximum(100)
    parent.progress_bar.setVisible(False)
    status_row.addWidget(parent.progress_bar)

    main_layout.addLayout(status_row)

    # ============================================================
    # ПОДКЛЮЧЕНИЯ
    # ============================================================
    parent.btn_load_coco.clicked.connect(parent.load_coco_json)
    parent.btn_load_images.clicked.connect(parent.load_images_folder)
    parent.btn_browse_output.clicked.connect(parent.browse_output)
    parent.btn_open_output.clicked.connect(parent.open_output_folder)
    parent.btn_generate.clicked.connect(parent.on_generate)
    parent.btn_toggle_log.clicked.connect(parent.toggle_log)