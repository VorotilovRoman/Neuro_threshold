from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.image_navigation import ImageNavigationWidget
from ui.log_widget import LogWidget
from utils_new.dataset_review_view import DatasetReviewGraphicsView


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
    parent.btn_load_yaml.setToolTip(
        "Загрузить YOLO-датасет из data.yaml.\n"
        "Поддерживаются images/... + labels/..., txt-списки, valid/validation.\n"
        "Маски (если есть) подхватятся автоматически."
    )
    parent.btn_load_coco = QPushButton("Load COCO")
    parent.btn_load_coco.setToolTip(
        "Загрузить COCO JSON. Можно несколько файлов сразу (train/val/test).\n"
        "bbox → detect; segmentation → segment или obb.\n"
        "Категории объединяются по имени."
    )
    parent.btn_load_unet = QPushButton("Load U-Net YAML")
    parent.btn_load_unet.setToolTip(
        "Загрузить U-Net-датасет (masks_data.yaml).\n"
        "Маски конвертируются в polygon-аннотации (findContours).\n"
        "Оригинальные маски остаются для отображения."
    )
    parent.btn_save = QPushButton("Save Labels")
    parent.btn_save_all = QPushButton("Save All")
    parent.btn_save_all.setToolTip(
        "Сохранить аннотации всех уже просмотренных или изменённых изображений. "
        "Неоткрытые изображения не перезаписываются.")
    parent.chk_obb_xywhr = QCheckBox("OBB → xywhr")
    parent.chk_obb_xywhr.setChecked(False)
    parent.chk_obb_xywhr.setToolTip(
        "Формат сохранения OBB: 4 угла (выключено) или cx cy w h angle (включено).")

    parent.btn_convert_all_to_obb = QPushButton("All Box → OBB")
    parent.btn_convert_all_to_obb.setToolTip(
        "Перевести все обычные bounding box (detect)\n"
        "во всех уже загруженных в память снимках в OBB-аннотации.\n"
        "Изменения затрагивают только память — не забудьте Save Labels."
    )

    top_layout.addWidget(parent.btn_load_labels)
    top_layout.addWidget(parent.btn_load_masks)
    top_layout.addWidget(parent.btn_load_yaml)
    top_layout.addWidget(parent.btn_load_coco)
    top_layout.addWidget(parent.btn_load_unet)
    top_layout.addWidget(parent.chk_obb_xywhr)
    top_layout.addWidget(parent.btn_save)
    top_layout.addWidget(parent.btn_save_all)
    top_layout.addWidget(parent.btn_convert_all_to_obb)

    top_layout.addStretch()

    # Слайдер прозрачности маски
    top_layout.addWidget(QLabel("Mask opacity:"))
    parent.opacity_slider = QSlider(Qt.Horizontal)
    parent.opacity_slider.setRange(0, 100)
    parent.opacity_slider.setValue(50)
    parent.opacity_slider.setFixedWidth(200)
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

    # >>> NEW: строка предупреждения о некорректной разметке
    parent.validation_status_label = QLabel("")
    parent.validation_status_label.setWordWrap(True)
    parent.validation_status_label.setStyleSheet(
        "color: #b00000; font-weight: bold; padding: 2px;"
    )
    parent.validation_status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    parent.validation_status_label.setVisible(False)
    main_layout.addWidget(parent.validation_status_label)

    # ----- Основной сплиттер (изображение слева, список и лог справа) -----
    main_splitter = QSplitter(Qt.Horizontal)

    # Левая часть: горизонтальный splitter с двумя независимыми view.
    # QSplitter сохраняет пользовательское соотношение размеров окон.
    parent.image_panels_splitter = QSplitter(Qt.Horizontal)

    parent.full_image_panel = QWidget()
    full_layout = QVBoxLayout(parent.full_image_panel)
    full_layout.setContentsMargins(0, 0, 0, 0)
    full_layout.addWidget(QLabel("Исходный снимок"))
    parent.viewer = DatasetReviewGraphicsView()
    parent.viewer.setMinimumSize(180, 260)
    parent.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    full_layout.addWidget(parent.viewer, 1)
    parent.image_panels_splitter.addWidget(parent.full_image_panel)

    parent.crop_panel = QWidget()
    crop_layout = QVBoxLayout(parent.crop_panel)
    crop_layout.setContentsMargins(0, 0, 0, 0)
    crop_layout.addWidget(QLabel("Выбранная аннотация · исходное разрешение"))
    parent.crop_view = DatasetReviewGraphicsView()
    parent.crop_view.selection_enabled = False
    parent.crop_view.setMinimumSize(180, 260)
    parent.crop_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    crop_layout.addWidget(parent.crop_view, 1)
    parent.image_panels_splitter.addWidget(parent.crop_panel)
    parent.image_panels_splitter.setSizes([600, 600])
    parent.crop_panel.setVisible(False)  # режим 1 при запуске

    main_splitter.addWidget(parent.image_panels_splitter)

    # Правая часть: список объектов + лог
    right_widget = QWidget()
    right_layout = QVBoxLayout(right_widget)

    # Группа "Bounding boxes"
    objects_group = QGroupBox("Bounding boxes")
    objects_layout = QVBoxLayout()
    # Три режима показа (одна общая панель со списком справа).
    mode_row = QHBoxLayout()
    objects_layout.addWidget(QLabel("Режим просмотра:"))
    parent.view_mode_combo = QComboBox()
    parent.view_mode_combo.addItem("1 · Снимок и список объектов", 1)
    parent.view_mode_combo.addItem("2 · Снимок + выбранный объект + список", 2)
    parent.view_mode_combo.addItem("3 · Выбранный объект и список", 3)
    parent.view_mode_combo.setToolTip(
        "Выбранный в списке объект показывается во втором окне "
        "с отступом, без поворота и без изменения исходных пикселей.")
    objects_layout.addWidget(parent.view_mode_combo)
    # Кнопка выбора режима и отступ в одном компактном блоке справа.
    mode_row.addWidget(QLabel("Отступ (px):"))
    parent.crop_padding_spin = QSpinBox()
    parent.crop_padding_spin.setRange(0, 1000)
    parent.crop_padding_spin.setValue(10)
    parent.crop_padding_spin.setToolTip("Отступ от границ аннотации в пикселях исходного снимка.")
    mode_row.addWidget(parent.crop_padding_spin)
    parent.crop_detail_label = QLabel("Выберите объект в списке")
    parent.crop_detail_label.setWordWrap(True)
    # Статус выбранного объекта отдельной строкой, чтобы правая панель не расширялась.
    mode_row.addStretch()
    objects_layout.addLayout(mode_row)
    objects_layout.addWidget(parent.crop_detail_label)

    # Те же независимые визуальные переключатели, что на вкладке разметки.
    visibility_row = QHBoxLayout()
    parent.btn_hide_frames = QPushButton("Скрыть рамки")
    parent.btn_hide_labels = QPushButton("Скрыть метки")
    parent.btn_fill_objects = QPushButton("Закрасить объекты")
    for btn in (parent.btn_hide_frames, parent.btn_hide_labels,
                parent.btn_fill_objects):
        btn.setCheckable(True)
    visibility_row.addWidget(parent.btn_hide_frames)
    visibility_row.addWidget(parent.btn_hide_labels)
    parent.btn_hide_frames.setToolTip("Скрыть постоянные контуры, оставив подписи и выделение.")
    parent.btn_hide_labels.setToolTip("Скрыть только подписи классов.")
    parent.btn_fill_objects.setToolTip("Залить области объектов выбранным цветом. Исходное изображение не изменяется.")
    parent.btn_fill_color = QPushButton("Цвет: #FFFFFF")
    parent.btn_fill_color.setToolTip("Выбрать цвет заливки объектов (по умолчанию белый).")
    visibility_row.setSpacing(4)
    visibility_row.addWidget(parent.btn_fill_objects)
    visibility_row.addWidget(parent.btn_fill_color)
    visibility_row.addStretch()
    objects_layout.addLayout(visibility_row)

    objects_layout.addSpacing(8)
    parent.coord_list = QListWidget()
    parent.coord_list.setSelectionMode(QAbstractItemView.SingleSelection)
    objects_layout.addWidget(parent.coord_list)
    hint = QLabel("Ctrl+0…9 — присвоить класс выбранному объекту.  "
                  "Красный объект в списке — некорректная разметка "
                  "или пересечение ≥70% площади меньшего объекта.")
    hint.setWordWrap(True)
    objects_layout.addWidget(hint)

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
