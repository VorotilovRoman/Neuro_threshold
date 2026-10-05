# yolo_test_aeb_ui.py
"""
UI-слой вкладки YOLO + AE/VAE.

Структура:
  • ImageNavigationWidget — загрузка/навигация;
  • в верхней панели сразу после блока загрузки — кнопки
    «Сохранить снимок» и «Сохранить карту дефектов»;
  • 3 × SmartGraphicsView: сверху — YOLO-результат на всю ширину,
    снизу слева — обработанный снимок, снизу справа — карта ошибок;
  • правая панель управления в QScrollArea;
  • лог с кнопкой переключения — в правой половине, под настройками.
"""
from import_libs_internal import *


def setup_yolo_ae_vae_ui(parent):
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)

    # ---------- Верхняя панель: навигация + сохранение ----------
    parent.nav_widget = ImageNavigationWidget()
    parent.reset_zoom_button = QPushButton("Reset zoom")

    # Кнопки сохранения — сразу после блока загрузки изображений.
    parent.btn_save_image = QPushButton("💾 Сохранить снимок")
    parent.btn_save_image.setToolTip(
        "Сохранить итоговое изображение из третьего окна:\n"
        "после авто-обрезки, авто-доворота, ручных поворотов,\n"
        "морфологии по контуру объекта и инверсии."
    )
    parent.btn_save_image.setStyleSheet(
        "QPushButton { background-color: #27ae60; color: white; font-weight: 600; }"
    )

    parent.btn_save_errmap = QPushButton("🗺 Сохранить карту дефектов")
    parent.btn_save_errmap.setToolTip(
        "Сохранить текущую карту ошибок AE/VAE (то, что в 4-м окне)."
    )
    parent.btn_save_errmap.setStyleSheet(
        "QPushButton { background-color: #16a085; color: white; font-weight: 600; }"
    )

    top_layout = QHBoxLayout()
    top_layout.addWidget(parent.nav_widget)
    top_layout.addWidget(parent.btn_save_image)
    top_layout.addWidget(parent.btn_save_errmap)
    top_layout.addWidget(parent.reset_zoom_button)
    top_layout.addStretch()
    main_layout.addLayout(top_layout)

    # ---------- Основной сплиттер ----------
    main_splitter = QSplitter(Qt.Horizontal)

    # ----- Левая часть: 3 SmartGraphicsView (1 сверху, 2 снизу) -----
    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 0, 0)
    grid = QGridLayout()
    grid.setSpacing(10)

    # Верх: YOLO-результат — на всю ширину.
    parent.yolo_view = SmartGraphicsView()
    parent.yolo_view.setMinimumSize(400, 300)
    parent.yolo_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.yolo_view, 0, 0, 1, 2)

    # Низ слева: обработанный снимок.
    parent.transformed_view = SmartGraphicsView()
    parent.transformed_view.setMinimumSize(400, 300)
    parent.transformed_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.transformed_view, 1, 0)

    # Низ справа: карта ошибок AE/VAE.
    parent.error_view = SmartGraphicsView()
    parent.error_view.setMinimumSize(400, 300)
    parent.error_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    grid.addWidget(parent.error_view, 1, 1)

    grid.setRowStretch(0, 1)
    grid.setRowStretch(1, 1)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    left_layout.addLayout(grid)

    parent.info_label = QLabel("No images")
    parent.info_label.setAlignment(Qt.AlignCenter)
    left_layout.addWidget(parent.info_label)

    # ----- Правая панель с прокруткой -----
    right_scroll_area = QScrollArea()
    right_scroll_area.setWidgetResizable(True)
    right_scroll_area.setFrameShape(QFrame.NoFrame)

    right_content = QWidget()
    right_layout = QVBoxLayout(right_content)
    right_layout.setContentsMargins(0, 0, 0, 0)

    controls_group = QGroupBox("YOLO + AE/VAE")
    controls_layout = QVBoxLayout()

    # --- Пресеты ---
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

    # --- Модель YOLO ---
    controls_layout.addWidget(QLabel("Модель YOLO:"))
    model_row = QHBoxLayout()
    parent.model_edit = QLineEdit()
    parent.model_edit.setReadOnly(True)
    parent.model_edit.setPlaceholderText("Путь к .pt-файлу YOLO…")
    parent.btn_load_model = QPushButton("Загрузить")
    model_row.addWidget(parent.model_edit, 1)
    model_row.addWidget(parent.btn_load_model)
    controls_layout.addLayout(model_row)

    # --- Параметры инференса YOLO ---
    yolo_params = QGridLayout()
    yolo_params.addWidget(QLabel("conf:"), 0, 0)
    parent.conf_spin = QDoubleSpinBox()
    parent.conf_spin.setRange(0.0, 1.0)
    parent.conf_spin.setSingleStep(0.05)
    parent.conf_spin.setDecimals(2)
    parent.conf_spin.setValue(0.75)
    yolo_params.addWidget(parent.conf_spin, 0, 1)

    yolo_params.addWidget(QLabel("iou:"), 1, 0)
    parent.iou_spin = QDoubleSpinBox()
    parent.iou_spin.setRange(0.0, 1.0)
    parent.iou_spin.setSingleStep(0.05)
    parent.iou_spin.setDecimals(2)
    parent.iou_spin.setValue(0.45)
    yolo_params.addWidget(parent.iou_spin, 1, 1)

    yolo_params.addWidget(QLabel("imgsz:"), 2, 0)
    parent.imgsz_spin = QSpinBox()
    parent.imgsz_spin.setRange(64, 4096)
    parent.imgsz_spin.setSingleStep(32)
    parent.imgsz_spin.setValue(640)
    yolo_params.addWidget(parent.imgsz_spin, 2, 1)

    yolo_params.addWidget(QLabel("device:"), 3, 0)
    parent.device_combo = QComboBox()
    parent.device_combo.addItems(["auto", "cpu", "0", "mps"])
    yolo_params.addWidget(parent.device_combo, 3, 1)
    controls_layout.addLayout(yolo_params)

    parent.chk_show_boxes = QCheckBox("Показывать боксы YOLO на втором снимке")
    parent.chk_show_boxes.setChecked(True)
    parent.chk_show_boxes.setToolTip(
        "Влияет только на второй снимок (результат YOLO).\n"
        "На третьем и четвёртом снимке боксы не рисуются."
    )
    controls_layout.addWidget(parent.chk_show_boxes)

    # --- Модель AE/VAE ---
    controls_layout.addWidget(QLabel("Модель AE/VAE:"))
    ae_row = QHBoxLayout()
    parent.ae_model_edit = QLineEdit()
    parent.ae_model_edit.setReadOnly(True)
    parent.ae_model_edit.setPlaceholderText("Путь к чекпойнту AE/VAE…")
    parent.btn_load_ae = QPushButton("Загрузить AE/VAE")
    ae_row.addWidget(parent.ae_model_edit, 1)
    ae_row.addWidget(parent.btn_load_ae)
    controls_layout.addLayout(ae_row)

    ae_btns = QHBoxLayout()
    parent.btn_clear_ae = QPushButton("Очистить AE/VAE")
    parent.btn_apply_ae = QPushButton("Применить к снимку")
    ae_btns.addWidget(parent.btn_clear_ae)
    ae_btns.addWidget(parent.btn_apply_ae)
    controls_layout.addLayout(ae_btns)

    # --- Параметры AE/VAE ---
    ae_row1 = QHBoxLayout()
    ae_row1.addWidget(QLabel("Порог, перцентиль:"))
    parent.ae_perc_spin = QDoubleSpinBox()
    parent.ae_perc_spin.setRange(50.0, 100.0)
    parent.ae_perc_spin.setSingleStep(0.5)
    parent.ae_perc_spin.setDecimals(1)
    parent.ae_perc_spin.setValue(99.0)
    ae_row1.addWidget(parent.ae_perc_spin)

    parent.chk_ae_use_ckpt_thr = QCheckBox("Порог из чекпойнта")
    parent.chk_ae_use_ckpt_thr.setChecked(True)
    parent.chk_ae_use_ckpt_thr.setToolTip(
        "Если в чекпойнте есть готовый порог (threshold_value),\n"
        "использовать его. Иначе — перцентиль слева."
    )
    ae_row1.addWidget(parent.chk_ae_use_ckpt_thr)

    parent.chk_ae_overlay = QCheckBox("Наложить на снимок")
    parent.chk_ae_overlay.setChecked(True)
    ae_row1.addWidget(parent.chk_ae_overlay)

    parent.chk_ae_show_mask = QCheckBox("Подсветить > порога")
    ae_row1.addWidget(parent.chk_ae_show_mask)
    ae_row1.addStretch()
    controls_layout.addLayout(ae_row1)

    ae_row2 = QHBoxLayout()
    ae_row2.addWidget(QLabel("Прозрачность:"))
    parent.ae_alpha_slider = QSlider(Qt.Horizontal)
    parent.ae_alpha_slider.setRange(0, 100)
    parent.ae_alpha_slider.setValue(50)
    ae_row2.addWidget(parent.ae_alpha_slider, 1)
    parent.ae_alpha_label = QLabel("50")
    parent.ae_alpha_label.setFixedWidth(28)
    ae_row2.addWidget(parent.ae_alpha_label)
    controls_layout.addLayout(ae_row2)

    # --- Найденные объекты ---
    controls_layout.addWidget(QLabel("Найденные объекты:"))
    parent.objects_list = QListWidget()
    parent.objects_list.setSelectionMode(QListWidget.SingleSelection)
    parent.objects_list.setMinimumHeight(120)
    controls_layout.addWidget(parent.objects_list)

    pad_row = QHBoxLayout()
    pad_row.addWidget(QLabel("Отступ (px):"))
    parent.crop_padding_spin = QSpinBox()
    parent.crop_padding_spin.setRange(-500, 500)
    parent.crop_padding_spin.setValue(0)
    pad_row.addWidget(parent.crop_padding_spin)

    pad_row.addWidget(QLabel("Смещ. X:"))
    parent.crop_offset_x_spin = QSpinBox()
    parent.crop_offset_x_spin.setRange(-2000, 2000)
    parent.crop_offset_x_spin.setValue(0)
    parent.crop_offset_x_spin.setToolTip(
        "Смещение области обрезки по горизонтали в пикселях.\n"
        "Может быть отрицательным (влево) или положительным (вправо)."
    )
    pad_row.addWidget(parent.crop_offset_x_spin)

    pad_row.addWidget(QLabel("Смещ. Y:"))
    parent.crop_offset_y_spin = QSpinBox()
    parent.crop_offset_y_spin.setRange(-2000, 2000)
    parent.crop_offset_y_spin.setValue(0)
    parent.crop_offset_y_spin.setToolTip(
        "Смещение области обрезки по вертикали в пикселях.\n"
        "Может быть отрицательным (вверх) или положительным (вниз)."
    )
    pad_row.addWidget(parent.crop_offset_y_spin)

    # NEW: точный доворот кадра вокруг его центра
    pad_row.addWidget(QLabel("Поворот (°):"))
    parent.crop_rotation_spin = QDoubleSpinBox()
    parent.crop_rotation_spin.setRange(-90.0, 90.0)
    parent.crop_rotation_spin.setSingleStep(1.0)
    parent.crop_rotation_spin.setDecimals(1)
    parent.crop_rotation_spin.setValue(0.0)
    parent.crop_rotation_spin.setToolTip(
        "Точный доворот уже обрезанного кадра вокруг его центра.\n"
        "Диапазон: -90°..+90°. Положительное значение — по часовой\n"
        "стрелке, отрицательное — против часовой."
    )
    pad_row.addWidget(parent.crop_rotation_spin)

    # NEW: сброс отступа и смещений
    parent.btn_reset_offsets = QPushButton("Сбросить смещения")
    parent.btn_reset_offsets.setToolTip(
        "Обнулить отступ и смещения по X и Y."
    )
    pad_row.addWidget(parent.btn_reset_offsets)

    pad_row.addStretch()
    controls_layout.addLayout(pad_row)

    # --- Авто-действия ---
    parent.chk_auto_crop = QCheckBox("Авто-обрезка по объекту")
    controls_layout.addWidget(parent.chk_auto_crop)

    parent.chk_auto_rotate = QCheckBox("Авто-доворот объекта")
    controls_layout.addWidget(parent.chk_auto_rotate)

    fill_row = QHBoxLayout()
    fill_row.addWidget(QLabel("Заливка фона:"))
    parent.fill_combo = QComboBox()
    parent.fill_combo.addItem("Нет", "none")
    parent.fill_combo.addItem("Чёрный", "black")
    parent.fill_combo.addItem("Белый", "white")
    parent.fill_combo.setEnabled(False)
    fill_row.addWidget(parent.fill_combo)
    fill_row.addStretch()
    controls_layout.addLayout(fill_row)

    # --- Ручные трансформации ---
    controls_layout.addWidget(QLabel("Ручные трансформации:"))
    m_row1 = QHBoxLayout()
    parent.btn_rot_cw = QPushButton("↻ 90° по часовой")
    parent.btn_rot_ccw = QPushButton("↺ 90° против часовой")
    m_row1.addWidget(parent.btn_rot_cw)
    m_row1.addWidget(parent.btn_rot_ccw)
    controls_layout.addLayout(m_row1)

    m_row2 = QHBoxLayout()
    parent.btn_flip_v = QPushButton("⇅ Отразить по вертикали")
    parent.btn_flip_h = QPushButton("⇆ Отразить по горизонтали")
    m_row2.addWidget(parent.btn_flip_v)
    m_row2.addWidget(parent.btn_flip_h)
    controls_layout.addLayout(m_row2)

    parent.btn_reset = QPushButton("Сбросить (показать оригинал)")
    controls_layout.addWidget(parent.btn_reset)

    # --- Морфологические операции (по контуру выбранного объекта) ---
    morph_label = QLabel("Морфология по контуру выбранного объекта (пикс.):")
    morph_label.setStyleSheet("QLabel { color: #666; margin-top: 4px; }")
    controls_layout.addWidget(morph_label)

    kernel_shape_row = QHBoxLayout()
    kernel_shape_row.addWidget(QLabel("Форма ядра:"))
    parent.kernel_shape_combo = QComboBox()
    parent.kernel_shape_combo.addItems(["Rectangle", "Ellipse", "Cross"])
    kernel_shape_row.addWidget(parent.kernel_shape_combo)
    kernel_shape_row.addStretch()
    controls_layout.addLayout(kernel_shape_row)

    erosion_row = QHBoxLayout()
    erosion_row.addWidget(QLabel("Эрозия:"))
    parent.erosion_slider = QSlider(Qt.Horizontal)
    parent.erosion_slider.setRange(1, 100)
    parent.erosion_slider.setValue(1)
    parent.erosion_slider.setToolTip(
        "1 — операция отключена.\n"
        "2..100 — размер ядра в пикселях (нечётное значение)."
    )
    parent.erosion_label = QLabel("1")
    parent.erosion_label.setFixedWidth(36)
    erosion_row.addWidget(parent.erosion_slider, 1)
    erosion_row.addWidget(parent.erosion_label)
    controls_layout.addLayout(erosion_row)

    dilation_row = QHBoxLayout()
    dilation_row.addWidget(QLabel("Расширение:"))
    parent.dilation_slider = QSlider(Qt.Horizontal)
    parent.dilation_slider.setRange(1, 100)
    parent.dilation_slider.setValue(1)
    parent.dilation_slider.setToolTip(
        "1 — операция отключена.\n"
        "2..100 — размер ядра в пикселях (нечётное значение)."
    )
    parent.dilation_label = QLabel("1")
    parent.dilation_label.setFixedWidth(36)
    dilation_row.addWidget(parent.dilation_slider, 1)
    dilation_row.addWidget(parent.dilation_label)
    controls_layout.addLayout(dilation_row)

    parent.invert_image_checkbox = QCheckBox("Инвертировать изображение")
    controls_layout.addWidget(parent.invert_image_checkbox)

    # NEW: режим «только по контуру»
    parent.chk_contour_only = QCheckBox("Только по контуру объекта")
    parent.chk_contour_only.setChecked(True)
    parent.chk_contour_only.setToolTip(
        "Если включено — эрозия/дилатация применяются только по контуру\n"
        "выбранного объекта (граница маски сегментации или четырёхугольника).\n"
        "Если выключено — по всей площади внутри объекта."
    )
    controls_layout.addWidget(parent.chk_contour_only)

    # --- Кнопки сохранения теперь в верхней панели, здесь их нет ---

    controls_group.setLayout(controls_layout)
    right_layout.addWidget(controls_group)

    # --- Лог (кнопка переключения + сам лог) ---
    log_toggle_row = QHBoxLayout()
    parent.toggle_log_btn = QPushButton("Показать лог")
    parent.toggle_log_btn.setCheckable(True)
    log_toggle_row.addWidget(parent.toggle_log_btn)
    log_toggle_row.addStretch()
    right_layout.addLayout(log_toggle_row)

    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=False)
    parent.log_text = parent.log_widget.text
    parent.log_widget.setVisible(False)
    parent.log_widget.setMinimumHeight(200)
    right_layout.addWidget(parent.log_widget, 1)

    right_scroll_area.setWidget(right_content)

    main_splitter.addWidget(left_widget)
    main_splitter.addWidget(right_scroll_area)
    main_splitter.setSizes([400, 500])
    main_layout.addWidget(main_splitter, 1)