# -*- coding: utf-8 -*-
"""Dataset Preparation V18: compact task-oriented UI, same data-loading strip/log."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QPushButton, QRadioButton, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTabWidget, QScrollArea, QFrame, QVBoxLayout, QWidget,
)
from ui.control_buttons import ControlButtons
from ui.log_widget import LogWidget


def _section(title, description=None):
    group = QGroupBox(title)
    group.setObjectName('preparationSection')
    layout = QVBoxLayout(group)
    layout.setContentsMargins(16, 17, 16, 15)
    layout.setSpacing(12)
    if description:
        note = QLabel(description)
        note.setObjectName('sectionHint')
        note.setWordWrap(True)
        layout.addWidget(note)
    return group, layout


def _page():
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    host = QWidget()
    host.setObjectName('preparationPage')
    layout = QVBoxLayout(host)
    layout.setContentsMargins(10, 12, 10, 12)
    layout.setSpacing(13)
    scroll.setWidget(host)
    return scroll, layout


def _wrap_row(*widgets):
    row = QHBoxLayout()
    row.setSpacing(9)
    for widget in widgets:
        row.addWidget(widget)
    row.addStretch()
    return row


def setup_preparing_dataset_yaml_ui(parent):
    central = QWidget()
    parent.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(9, 9, 9, 8)
    main_layout.setSpacing(8)
    # ============================================================
    # ВЕРХНЯЯ ПАНЕЛЬ
    # ============================================================
    top_layout = QHBoxLayout()
    parent.btn_folder = QPushButton("Load Images")
    parent.btn_labels = QPushButton("Load Labels")

    parent.btn_load_yaml = QPushButton("Load YOLO YAML")
    parent.btn_load_yaml.setToolTip(
        "Загрузить YOLO-датасет из data.yaml.\n"
        "Поддерживаются images/... + labels/... и txt-списки."
    )
    parent.btn_load_coco = QPushButton("Load COCO")
    parent.btn_load_coco.setToolTip(
        "Загрузить COCO JSON (annotations_*.json).\n"
        "Заменит текущий датасет.\n"
        "Можно выбрать несколько файлов или папку с JSON."
    )
    parent.btn_add_coco = QPushButton("+ Add COCO")
    parent.btn_add_coco.setToolTip(
        "Добавить COCO JSON к уже загруженному датасету.\n"
        "Категории объединяются по имени: если 'cat' уже есть,\n"
        "новый 'cat' получит тот же класс.\n"
        "Можно выбрать несколько файлов или папку с JSON."
    )
    parent.btn_load_unet = QPushButton("Load U-Net YAML")
    parent.btn_load_unet.setToolTip(
        "Загрузить U-Net-датасет (masks_data.yaml).\n"
        "Пары (image, mask) + class_names + mask_class_offset."
    )

    top_layout.addWidget(parent.btn_folder)
    top_layout.addWidget(parent.btn_labels)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addWidget(parent.btn_load_coco)
    top_layout.addWidget(parent.btn_add_coco)
    top_layout.addWidget(parent.btn_load_unet)

    # --- Input type ---
    parent.dataset_type_combo = QComboBox()
    parent.dataset_type_combo.addItem("Метки (bbox, obb, seg)", 0)
    parent.dataset_type_combo.addItem("Маски (masks)", 1)
    parent.dataset_type_combo.addItem("Только изображения", 2)
    parent.dataset_type_combo.setCurrentIndex(0)

    top_layout.addWidget(QLabel("Input type:"))
    top_layout.addWidget(parent.dataset_type_combo)

    # --- Input annotation format ---
    parent.input_format_combo = QComboBox()
    parent.input_format_combo.addItem("Auto (по содержимому)", "auto")
    parent.input_format_combo.addItem("YOLO detect (5 токенов)", "detect")
    parent.input_format_combo.addItem("YOLO OBB — 4 угла (9 токенов)", "obb_corners")
    parent.input_format_combo.addItem("YOLO OBB — xywhr (6 токенов)", "obb_xywhr")
    parent.input_format_combo.addItem("YOLO segment (≥7 токенов, чётные)", "segment")
    parent.input_format_combo.addItem("Mixed (per-line эвристика)", "mixed")
    parent.input_format_combo.setCurrentIndex(0)
    parent.input_format_combo.setToolTip(
        "Формат .txt при загрузке аннотаций.\n"
        "• Auto — эвристика по числу токенов в строке (текущее поведение).\n"
        "• Явное указание формата убирает неоднозначность:\n"
        "   4-точечный полигон (9 токенов) можно читать как segment."
    )
    top_layout.addWidget(QLabel("Input fmt:"))
    top_layout.addWidget(parent.input_format_combo)

    top_layout.addStretch()

    parent.btn_output = QPushButton("Save YAML")
    top_layout.addWidget(parent.btn_output)

    parent.output_path_display = QLineEdit()
    parent.output_path_display.setReadOnly(True)
    parent.output_path_display.setPlaceholderText("Not selected")
    parent.output_path_display.setMinimumWidth(400)
    top_layout.addWidget(parent.output_path_display)

    parent.btn_open_output = QPushButton("Open Folder")
    parent.btn_open_output.setEnabled(False)
    top_layout.addWidget(parent.btn_open_output)
    main_layout.addLayout(top_layout)

    # Main workspace: file validation at left, configuration tabs at right.
    main_v_splitter = QSplitter(Qt.Vertical)
    main_v_splitter.setChildrenCollapsible(False)
    main_layout.addWidget(main_v_splitter, 1)
    h_splitter = QSplitter(Qt.Horizontal)
    h_splitter.setChildrenCollapsible(False)
    main_v_splitter.addWidget(h_splitter)

    left_widget = QWidget()
    left_layout = QVBoxLayout(left_widget)
    left_layout.setContentsMargins(2, 2, 5, 2)
    left_layout.setSpacing(8)
    pair_head = QHBoxLayout()
    pair_title = QLabel('Исходные данные')
    pair_title.setObjectName('workspaceTitle')
    parent.pair_count_label = QLabel('Всего пар: 0')
    parent.pair_count_label.setObjectName('countChip')
    parent.pair_count_label.setWordWrap(True)
    parent.pair_count_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    pair_head.addWidget(pair_title)
    pair_head.addStretch()
    pair_head.addWidget(parent.pair_count_label)
    left_layout.addLayout(pair_head)
    instruction = QLabel('Двойной щелчок — просмотр и проверка пары. Ctrl+C — копировать.')
    instruction.setObjectName('sectionHint')
    instruction.setWordWrap(True)
    left_layout.addWidget(instruction)
    parent.file_list = QListWidget()
    parent.file_list.setAlternatingRowColors(True)
    parent.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    parent.file_list.setToolTip('Ctrl+C — скопировать выделенные строки; двойной клик — проверить пару.')
    left_layout.addWidget(parent.file_list, 1)

    parent.validation_progress_label = QLabel('')
    parent.validation_progress_label.setWordWrap(True)
    parent.validation_progress_label.setObjectName('infoLabel')
    parent.validation_progress_label.setVisible(False)
    left_layout.addWidget(parent.validation_progress_label)
    parent.validation_warning_label = QLabel('')
    parent.validation_warning_label.setWordWrap(True)
    parent.validation_warning_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.validation_warning_label.setObjectName('warningLabel')
    parent.validation_warning_label.setVisible(False)
    left_layout.addWidget(parent.validation_warning_label)
    parent.color_stats_label = QLabel('')
    parent.color_stats_label.setWordWrap(True)
    parent.color_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.color_stats_label.setObjectName('statsLabel')
    left_layout.addWidget(parent.color_stats_label)

    convert_group, convert_layout = _section('Конвертация изображений')
    parent.color_convert_buttons_layout = QHBoxLayout()
    parent.btn_convert_gray = QPushButton('В Ч/Б')
    parent.btn_convert_rgb = QPushButton('В RGB')
    parent.btn_convert_8bit_png = QPushButton('В 8-бит PNG')
    for widget in (parent.btn_convert_gray, parent.btn_convert_rgb, parent.btn_convert_8bit_png):
        widget.setEnabled(False)
        parent.color_convert_buttons_layout.addWidget(widget)
    parent.btn_convert_gray.setToolTip('Перезаписывает цветные и визуально-серые изображения в одноканальные.')
    parent.btn_convert_rgb.setToolTip('Перезаписывает одноканальные изображения в BGR.')
    parent.btn_convert_8bit_png.setToolTip('Приводит изображения к 8-бит PNG; маски не меняются.')
    convert_layout.addLayout(parent.color_convert_buttons_layout)
    left_layout.addWidget(convert_group)
    h_splitter.addWidget(left_widget)

    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)
    right_layout.setContentsMargins(5, 0, 0, 0)
    right_layout.setSpacing(8)
    header_row = QHBoxLayout()
    heading = QLabel('Настройки датасета')
    heading.setObjectName('workspaceTitle')
    header_row.addWidget(heading)
    header_row.addStretch()
    parent.total_count_with_multiplier = QLabel('Итоговое количество: 0')
    parent.total_count_with_multiplier.setObjectName('countChip')
    header_row.addWidget(parent.total_count_with_multiplier)
    right_layout.addLayout(header_row)
    parent.preparation_tabs = QTabWidget()
    parent.preparation_tabs.setDocumentMode(True)
    right_layout.addWidget(parent.preparation_tabs, 1)

    # TAB 1 — preprocessing
    page, page_layout = _page()
    group, group_layout = _section('Размер и каналы', 'Настройка изображения перед записью выходного датасета.')
    form = QFormLayout()
    form.setSpacing(12)
    parent.resize_keep = QRadioButton('Исходный размер')
    parent.resize_fixed = QRadioButton('Вписать с полями')
    parent.resize_stretch = QRadioButton('Растянуть')
    parent.resize_fixed.setChecked(True)
    parent.resize_size = QSpinBox()
    parent.resize_size.setRange(32, 4096)
    parent.resize_size.setSingleStep(32)
    parent.resize_size.setValue(640)
    parent.resize_size.setSuffix(' px')
    form.addRow('Режим:', _wrap_row(parent.resize_keep, parent.resize_fixed, parent.resize_stretch))
    form.addRow('Размер квадрата:', parent.resize_size)
    parent.bg_color_combo = QComboBox()
    parent.bg_color_combo.addItems(['Black', 'White'])
    form.addRow('Цвет полей:', parent.bg_color_combo)
    parent.channels_combo = QComboBox()
    parent.channels_combo.addItems(['3 (RGB — читать всё как цветное)', '1 (Grayscale — читать всё как Ч/Б)'])
    form.addRow('Каналы YOLO YAML:', parent.channels_combo)
    group_layout.addLayout(form)
    page_layout.addWidget(group)

    group, group_layout = _section('Проверка и фильтрация')
    parent.include_empty_annotations = QCheckBox('Включать снимки с пустой разметкой (негативные примеры)')
    parent.include_empty_annotations.setChecked(False)
    parent.include_empty_annotations.setToolTip('Применяется к пустым YOLO TXT и маскам только с фоном.')
    group_layout.addWidget(parent.include_empty_annotations)
    form = QFormLayout()
    parent.min_ann_area_spin = QSpinBox()
    parent.min_ann_area_spin.setRange(0, 1_000_000)
    parent.min_ann_area_spin.setValue(0)
    parent.min_ann_area_spin.setSingleStep(10)
    parent.min_ann_area_spin.setSuffix(' px²')
    parent.min_ann_area_spin.setToolTip('0 — отключено. Для Detect: площадь bbox; для OBB/Segment: площадь полигона.')
    form.addRow('Минимальная площадь:', parent.min_ann_area_spin)
    group_layout.addLayout(form)
    page_layout.addWidget(group)
    group, group_layout = _section('Исходные файлы')
    parent.delete_source_after_success = QCheckBox('Удалить исходные файлы после успешной генерации')
    parent.delete_source_after_success.setChecked(False)
    parent.delete_source_after_success.setToolTip('НЕОБРАТИМО. Удаляются только обработанные исходные снимки/метки/маски после успешной генерации.')
    group_layout.addWidget(parent.delete_source_after_success)
    warn = QLabel('Внимание: удаление исходников необратимо. Для первой проверки не включайте этот параметр.')
    warn.setObjectName('warningLabel')
    warn.setWordWrap(True)
    group_layout.addWidget(warn)
    page_layout.addWidget(group)
    page_layout.addStretch()
    parent.preparation_tabs.addTab(page, 'Обработка')

    # TAB 2 — Augmentation and multiplier
    page, page_layout = _page()
    group, group_layout = _section('Геометрические аугментации', 'При множителе 1× изменяются существующие снимки без создания дополнительных копий.')
    grid = QGridLayout()
    grid.setHorizontalSpacing(25)
    grid.setVerticalSpacing(12)
    parent.flip_horizontal = QCheckBox('Горизонтальное отражение')
    parent.flip_vertical = QCheckBox('Вертикальное отражение')
    parent.rotate_90 = QCheckBox('Поворот на 90° / 180° / 270°')
    parent.random_crop = QCheckBox('Случайная обрезка')
    parent.random_rotate = QCheckBox('Произвольный поворот ±')
    parent.random_rotate_angle = QDoubleSpinBox()
    parent.random_rotate_angle.setRange(0, 15)
    parent.random_rotate_angle.setValue(5)
    parent.random_rotate_angle.setSuffix('°')
    parent.shear = QCheckBox('Сдвиг (shear) ±')
    parent.shear_angle = QDoubleSpinBox()
    parent.shear_angle.setRange(0, 15)
    parent.shear_angle.setValue(5)
    parent.shear_angle.setSuffix('°')
    grid.addWidget(parent.flip_horizontal, 0, 0)
    grid.addWidget(parent.flip_vertical, 0, 1)
    grid.addWidget(parent.rotate_90, 1, 0, 1, 2)
    grid.addWidget(parent.random_crop, 2, 0, 1, 2)
    grid.addLayout(_wrap_row(parent.random_rotate, parent.random_rotate_angle), 3, 0)
    grid.addLayout(_wrap_row(parent.shear, parent.shear_angle), 3, 1)
    group_layout.addLayout(grid)
    parent.aug_mask_note = QLabel('Для входных индексных масок используются только Flip и 90° (без потери классов).')
    parent.aug_mask_note.setObjectName('infoLabel')
    parent.aug_mask_note.setWordWrap(True)
    parent.aug_mask_note.setVisible(False)
    group_layout.addWidget(parent.aug_mask_note)
    page_layout.addWidget(group)

    group, group_layout = _section('Количество выходных изображений')
    parent.aug_multiplier_slider = QSlider(Qt.Horizontal)
    parent.aug_multiplier_slider.setRange(1, 10)
    parent.aug_multiplier_slider.setValue(2)
    parent.aug_multiplier_slider.setTickInterval(1)
    parent.aug_multiplier_slider.setTickPosition(QSlider.TicksBelow)
    parent.aug_multiplier_slider.setEnabled(False)
    parent.aug_multiplier_label = QLabel('2×')
    parent.aug_multiplier_label.setObjectName('multiplierBadge')
    multiplier_row = QHBoxLayout()
    multiplier_row.addWidget(QLabel('Множитель'))
    multiplier_row.addWidget(parent.aug_multiplier_slider, 1)
    multiplier_row.addWidget(parent.aug_multiplier_label)
    group_layout.addLayout(multiplier_row)
    parent.augmentation_mode_hint = QLabel('Включите операции аугментации, чтобы настроить множитель.')
    parent.augmentation_mode_hint.setObjectName('sectionHint')
    parent.augmentation_mode_hint.setWordWrap(True)
    group_layout.addWidget(parent.augmentation_mode_hint)
    page_layout.addWidget(group)

    group, group_layout = _section('Воспроизводимость')
    parent.use_random_seed = QCheckBox('Фиксировать seed')
    parent.random_seed_spin = QSpinBox()
    parent.random_seed_spin.setRange(0, 2147483647)
    parent.random_seed_spin.setValue(2026)
    parent.random_seed_spin.setEnabled(False)
    parent.use_random_seed.toggled.connect(parent.random_seed_spin.setEnabled)
    parent.use_random_seed.setToolTip('Воспроизводимые геометрические преобразования и разделение Train/Val/Test.')
    group_layout.addLayout(_wrap_row(parent.use_random_seed, parent.random_seed_spin))
    page_layout.addWidget(group)
    page_layout.addStretch()
    parent.preparation_tabs.addTab(page, 'Аугментации')

    # TAB 3 — split
    page, page_layout = _page()
    group, group_layout = _section('Train / Validation / Test', 'Суммарная доля всегда составляет 100%.')
    parent.train_slider = QSlider(Qt.Horizontal)
    parent.train_slider.setRange(0, 100)
    parent.train_slider.setValue(70)
    parent.train_slider.setTickInterval(10)
    parent.val_slider = QSlider(Qt.Horizontal)
    parent.val_slider.setRange(0, 100)
    parent.val_slider.setValue(20)
    parent.test_slider = QSlider(Qt.Horizontal)
    parent.test_slider.setRange(0, 100)
    parent.test_slider.setValue(10)
    parent.train_label = QLabel('70%')
    parent.val_label = QLabel('20%')
    parent.test_label = QLabel('10%')
    parent.train_count_label = QLabel('0')
    parent.val_count_label = QLabel('0')
    parent.test_count_label = QLabel('0')
    parent.total_count_label = QLabel('Общее кол-во: 0')
    form = QFormLayout()
    form.setSpacing(17)
    for name, slider, label, count in (
        ('Train', parent.train_slider, parent.train_label, parent.train_count_label),
        ('Validation', parent.val_slider, parent.val_label, parent.val_count_label),
        ('Test', parent.test_slider, parent.test_label, parent.test_count_label),
    ):
        row = QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(label)
        row.addWidget(QLabel('('))
        row.addWidget(count)
        row.addWidget(QLabel('снимков)'))
        form.addRow(name, row)
    group_layout.addLayout(form)
    parent.train_slider.valueChanged.connect(lambda v: parent.update_split(v, 'train'))
    parent.val_slider.valueChanged.connect(lambda v: parent.update_split(v, 'val'))
    parent.test_slider.valueChanged.connect(lambda v: parent.update_split(v, 'test'))
    page_layout.addWidget(group)
    group, group_layout = _section('Разделение по исходным изображениям')
    parent.group_augmentations_by_source = QCheckBox('Копии одного снимка — только в одном split')
    parent.group_augmentations_by_source.setChecked(True)
    parent.group_augmentations_by_source.setToolTip('Предотвращает утечку аугментированных копий между Train/Validation/Test.')
    group_layout.addWidget(parent.group_augmentations_by_source)
    page_layout.addWidget(group)
    page_layout.addStretch()
    parent.preparation_tabs.addTab(page, 'Разделение')

    # TAB 4 — output formats
    page, page_layout = _page()
    group, group_layout = _section('YOLO', 'Экспорт TXT-разметки и приведение типов аннотаций.')
    parent.out_yolo_txt = QCheckBox('Создать YOLO labels (.txt)')
    parent.out_yolo_txt.setChecked(True)
    parent.out_yolo_txt.setToolTip('labels/{split} для YOLO detect/OBB/Segment')
    group_layout.addWidget(parent.out_yolo_txt)
    form = QFormLayout()
    parent.out_yolo_subtype = QComboBox()
    parent.out_yolo_subtype.addItem('Сохранить тип', 'keep')
    parent.out_yolo_subtype.addItem('Detect (bbox)', 'detect')
    parent.out_yolo_subtype.addItem('OBB', 'obb')
    parent.out_yolo_subtype.addItem('Segment (полигон)', 'segment')
    parent.out_yolo_subtype_row = QHBoxLayout()
    parent.out_yolo_subtype_row.addWidget(parent.out_yolo_subtype)
    form.addRow('Выходной тип:', parent.out_yolo_subtype_row)
    parent.out_yolo_obb_format = QComboBox()
    parent.out_yolo_obb_format.addItem('4 угла (x1…y4)', 'corners')
    parent.out_yolo_obb_format.addItem('xywhr (cx, cy, w, h, angle)', 'xywhr')
    parent.out_yolo_obb_row = QHBoxLayout()
    parent.out_yolo_obb_row.addWidget(parent.out_yolo_obb_format)
    form.addRow('Запись OBB:', parent.out_yolo_obb_row)
    group_layout.addLayout(form)
    page_layout.addWidget(group)

    group, group_layout = _section('Индексные маски U-Net')
    parent.out_masks = QCheckBox('Создать PNG-маски')
    parent.out_masks.setChecked(False)
    group_layout.addWidget(parent.out_masks)
    parent.out_masks_offset = QSpinBox()
    parent.out_masks_offset.setRange(1, 1)
    parent.out_masks_offset.setValue(1)
    parent.out_masks_offset.setToolTip('Генератор использует 1: фон=0, класс 0 → пиксель 1.')
    parent.out_masks_offset_row = QHBoxLayout()
    parent.out_masks_offset_row.addWidget(parent.out_masks_offset)
    label = QLabel('Смещение классов маски:')
    row = QHBoxLayout()
    row.addWidget(label)
    row.addLayout(parent.out_masks_offset_row)
    row.addStretch()
    group_layout.addLayout(row)
    parent.out_unet_yaml = QCheckBox('Создать U-Net YAML (masks_data.yaml)')
    parent.out_unet_yaml.setChecked(False)
    group_layout.addWidget(parent.out_unet_yaml)
    page_layout.addWidget(group)

    group, group_layout = _section('COCO')
    parent.out_coco = QCheckBox('Создать COCO annotations_{split}.json')
    parent.out_coco.setChecked(False)
    group_layout.addWidget(parent.out_coco)
    parent.lossy_warning_label = QLabel('')
    parent.lossy_warning_label.setObjectName('warningLabel')
    parent.lossy_warning_label.setWordWrap(True)
    parent.lossy_warning_label.setVisible(False)
    group_layout.addWidget(parent.lossy_warning_label)
    page_layout.addWidget(group)
    page_layout.addStretch()
    parent.preparation_tabs.addTab(page, 'Экспорт')

    # TAB 5 — classes; uses all remaining vertical space.
    class_page = QWidget()
    class_layout = QVBoxLayout(class_page)
    class_layout.setContentsMargins(12, 12, 12, 12)
    class_layout.setSpacing(8)
    title = QLabel('Классы выходного датасета')
    title.setObjectName('workspaceTitle')
    class_layout.addWidget(title)
    note = QLabel('New ID — новая сквозная нумерация · Original ID — исходный класс · Mask pixel — номер пикселя U-Net.')
    note.setObjectName('sectionHint')
    note.setWordWrap(True)
    class_layout.addWidget(note)
    labels_group = QGroupBox('Таблица классов')
    labels_layout = QVBoxLayout(labels_group)
    parent.class_table = QTableWidget()
    parent.class_table.setColumnCount(5)
    parent.class_table.setHorizontalHeaderLabels(['New ID', 'Original ID', 'Name', 'Mask pixel', 'Include'])
    parent.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.class_table.setAlternatingRowColors(True)
    parent.class_table.setColumnHidden(3, True)
    parent.class_table.setToolTip('New ID: плотная нумерация 0…N−1; Original ID: исходный; Name: имя класса; Include: выбор класса.')
    labels_layout.addWidget(parent.class_table)
    class_layout.addWidget(labels_group, 1)
    parent.preparation_tabs.addTab(class_page, 'Классы')

    parent.control_buttons = ControlButtons(show_generate=True, show_cancel=True, show_save=False)
    parent.generate_btn = parent.control_buttons._generate_btn
    parent.cancel_btn = parent.control_buttons._cancel_btn
    parent.generate_btn.setText('Создать датасет')
    parent.cancel_btn.setText('Отмена')
    parent.generate_btn.setEnabled(False)
    parent.cancel_btn.setEnabled(False)
    parent.control_buttons.generate.connect(parent.on_generate)
    parent.control_buttons.cancel.connect(parent.on_cancel_generation)
    control_row = QHBoxLayout()
    control_row.addWidget(parent.control_buttons)
    control_row.addStretch()
    parent.toggle_log_btn = QPushButton('Показать лог')
    parent.toggle_log_btn.setCheckable(True)
    control_row.addWidget(parent.toggle_log_btn)
    right_layout.addLayout(control_row)
    h_splitter.addWidget(right_widget)
    h_splitter.setSizes([480, 820])

    # Existing log and histogram — unchanged infrastructure.
    parent.log_widget = LogWidget(show_clear_btn=True, show_progress=True)
    parent.log_text = parent.log_widget.text
    parent.progress_bar = parent.log_widget._progress
    parent.log_widget._progress.setFormat('Готово: %p%')
    parent.log_widget.setVisible(False)
    main_v_splitter.addWidget(parent.log_widget)
    main_v_splitter.setSizes([750, 140])
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

    parent.btn_folder.clicked.connect(parent.select_images_folder)
    parent.btn_labels.clicked.connect(parent.select_labels_folder)
    parent.btn_load_yaml.clicked.connect(parent.load_from_yaml)
    parent.btn_load_coco.clicked.connect(parent.load_from_coco)
    parent.btn_add_coco.clicked.connect(parent.add_from_coco)
    parent.btn_load_unet.clicked.connect(parent.load_from_unet_yaml)
    parent.btn_output.clicked.connect(parent.select_output_folder)

    # Style only the redesigned workspace. The input strip and LogWidget keep
    # their existing Qt controls and behavior.
    h_splitter.setStyleSheet("""
    QWidget#preparationPage { background: transparent; }
    QGroupBox#preparationSection, QGroupBox {
        border: 1px solid palette(mid); border-radius: 8px;
        margin-top: 10px; font-weight: 600;
    }
    QGroupBox#preparationSection::title, QGroupBox::title {
        subcontrol-origin: margin; left: 12px; padding: 0 5px;
    }
    QLabel#workspaceTitle { font-size: 16px; font-weight: 650; }
    QLabel#sectionHint { color: palette(mid); font-size: 11px; }
    QLabel#infoLabel { color: #2276aa; font-size: 11px; }
    QLabel#warningLabel { color: #aa6500; font-size: 11px; }
    QLabel#countChip { padding: 5px 9px; border: 1px solid palette(mid); border-radius: 7px; font-weight: 600; }
    QLabel#multiplierBadge { font-size: 18px; font-weight: 700; min-width: 45px; }
    QTabWidget::pane { border: 1px solid palette(mid); border-radius: 8px; }
    QTabBar::tab { padding: 8px 12px; }
    QPushButton { min-height: 24px; }
    QScrollArea { border: none; }
    """)

    # ============================================================
    # СИНХРОНИЗАЦИЯ СОСТОЯНИЙ
    # ============================================================
    def _sync_all_states(*_):
        """
        Единая точка управления видимостью/активностью виджетов
        в зависимости от input_type и выбранных output-флагов.
        """
        input_type = parent.dataset_type_combo.currentData()  # 0/1/2

        # --- Input format combo: активен только при input=labels ---
        parent.input_format_combo.setEnabled(input_type == 0)

        # --- Кнопка Load Labels: показывает "Mask" при input=masks ---
        if input_type == 0:
            parent.btn_labels.setText("Load Labels")
            parent.btn_labels.setEnabled(True)
        elif input_type == 1:
            parent.btn_labels.setText("Load Masks")
            parent.btn_labels.setEnabled(True)
        else:
            parent.btn_labels.setText("No labels (images only)")
            parent.btn_labels.setEnabled(False)

        # --- Output group ---
        is_images_only = (input_type == 2)

        parent.out_yolo_txt.setEnabled(not is_images_only)
        parent.out_masks.setEnabled(not is_images_only)
        parent.out_coco.setEnabled(not is_images_only)
        parent.out_unet_yaml.setEnabled(
            not is_images_only and parent.out_masks.isChecked()
        )

        # subtype combo активен только при out_yolo_txt
        subtype_active = (not is_images_only and parent.out_yolo_txt.isChecked())
        parent.out_yolo_subtype.setEnabled(subtype_active)

        # obb format активен при out_yolo_txt и subtype in ('keep', 'obb')
        subtype = parent.out_yolo_subtype.currentData()
        obb_fmt_active = (subtype_active and subtype in ('keep', 'obb'))
        parent.out_yolo_obb_format.setEnabled(obb_fmt_active)

        # mask offset активен при out_masks
        parent.out_masks_offset.setEnabled(
            not is_images_only and parent.out_masks.isChecked()
        )

        # --- Lossy warning при input=masks + (yolo_txt или coco) ---
        lossy = (input_type == 1 and
                 (parent.out_yolo_txt.isChecked() or parent.out_coco.isChecked()))
        if lossy:
            parts = []
            if parent.out_yolo_txt.isChecked():
                parts.append("YOLO labels (.txt)")
            if parent.out_coco.isChecked():
                parts.append("COCO JSON")
            parent.lossy_warning_label.setText(
                f"⚠️ Выходные форматы ({', '.join(parts)}) будут получены "
                f"через findContours из масок. Это lossy-преобразование: "
                f"теряются дырки и мелкие детали. Маски останутся "
                f"в masks/{{split}} в исходном виде."
            )
            parent.lossy_warning_label.setVisible(True)
        else:
            parent.lossy_warning_label.setVisible(False)

        # --- Augmentation: при input=masks доступны только простые ---
        is_masks_input = (input_type == 1)
        if is_masks_input:
            # Снимаем complex-чекбоксы (не удаляем — пользователь вернётся
            # к input=labels, и они снова станут доступны).
            for cb in (parent.random_rotate, parent.shear, parent.random_crop):
                cb.setEnabled(False)
            parent.random_rotate_angle.setEnabled(False)
            parent.shear_angle.setEnabled(False)
            parent.aug_mask_note.setVisible(True)
        else:
            for cb in (parent.random_rotate, parent.shear, parent.random_crop):
                cb.setEnabled(True)
            parent.random_rotate_angle.setEnabled(True)
            parent.shear_angle.setEnabled(True)
            parent.aug_mask_note.setVisible(False)

        # Обновим состояние множителя — зависит от чекбоксов аугментации.
        if hasattr(parent, 'update_multiplier_slider_state'):
            parent.update_multiplier_slider_state()
        # --- Колонка "Mask pixel": показываем, когда работаем с масками ---
        show_mask_pixel = (input_type == 1) or parent.out_masks.isChecked()
        try:
            parent.class_table.setColumnHidden(3, not show_mask_pixel)
        except Exception:
            pass
        if hasattr(parent, '_refresh_mask_pixel_column'):
            parent._refresh_mask_pixel_column()

    parent._sync_all_states = _sync_all_states

    # Подписка на изменения.
    # Первичный _sync_all_states() вызывается из DatasetPreparationWindow.__init__
    # после того, как все атрибуты окна (dataset_type и т.д.) установлены.
    parent.dataset_type_combo.currentIndexChanged.connect(_sync_all_states)
    parent.out_yolo_txt.toggled.connect(_sync_all_states)
    parent.out_masks.toggled.connect(_sync_all_states)
    parent.out_coco.toggled.connect(_sync_all_states)
    parent.out_yolo_subtype.currentIndexChanged.connect(_sync_all_states)
    parent.out_masks_offset.valueChanged.connect(_sync_all_states)