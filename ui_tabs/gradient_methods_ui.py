from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from ui.morphology_ui import MorphologyWidget
from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout


def setup_gradient_ui(parent):
    controls_layout = begin_layout(parent, 'gradient')

    # Выбор метода
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Gradient method:"))
    parent.method_combo = QComboBox()
    methods = ["Sobel", "Scharr", "Laplacian", "Canny", "Prewitt", "Roberts", "Kirsch"]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # Контейнеры параметров
    parent.sobel_container = QWidget()
    sobel_layout = QHBoxLayout(parent.sobel_container)
    sobel_layout.addWidget(QLabel("Kernel size (odd):"))
    parent.sobel_kernel = QSpinBox()
    parent.sobel_kernel.setRange(1, 31)
    parent.sobel_kernel.setSingleStep(2)
    parent.sobel_kernel.setValue(3)
    sobel_layout.addWidget(parent.sobel_kernel)
    sobel_layout.addWidget(QLabel("Scale:"))
    parent.sobel_scale = QSpinBox()
    parent.sobel_scale.setRange(1, 10)
    parent.sobel_scale.setValue(1)
    sobel_layout.addWidget(parent.sobel_scale)
    controls_layout.addWidget(parent.sobel_container)

    parent.laplacian_container = QWidget()
    lap_layout = QHBoxLayout(parent.laplacian_container)
    lap_layout.addWidget(QLabel("Kernel size (odd):"))
    parent.lap_kernel = QSpinBox()
    parent.lap_kernel.setRange(1, 31)
    parent.lap_kernel.setSingleStep(2)
    parent.lap_kernel.setValue(3)
    lap_layout.addWidget(parent.lap_kernel)
    lap_layout.addWidget(QLabel("Scale:"))
    parent.lap_scale = QSpinBox()
    parent.lap_scale.setRange(1, 10)
    parent.lap_scale.setValue(1)
    lap_layout.addWidget(parent.lap_scale)
    controls_layout.addWidget(parent.laplacian_container)

    parent.canny_container = QWidget()
    canny_layout = QGridLayout(parent.canny_container)
    canny_layout.addWidget(QLabel("Threshold1:"), 0, 0)
    parent.canny_thresh1 = QSlider(Qt.Horizontal)
    parent.canny_thresh1.setRange(0, 255)
    parent.canny_thresh1.setValue(50)
    parent.canny_label1 = QLabel("50")
    canny_layout.addWidget(parent.canny_thresh1, 0, 1)
    canny_layout.addWidget(parent.canny_label1, 0, 2)
    canny_layout.addWidget(QLabel("Threshold2:"), 1, 0)
    parent.canny_thresh2 = QSlider(Qt.Horizontal)
    parent.canny_thresh2.setRange(0, 255)
    parent.canny_thresh2.setValue(150)
    parent.canny_label2 = QLabel("150")
    canny_layout.addWidget(parent.canny_thresh2, 1, 1)
    canny_layout.addWidget(parent.canny_label2, 1, 2)
    canny_layout.addWidget(QLabel("Aperture size (odd):"), 2, 0)
    parent.canny_aperture = QSpinBox()
    parent.canny_aperture.setRange(3, 7)
    parent.canny_aperture.setSingleStep(2)
    parent.canny_aperture.setValue(3)
    canny_layout.addWidget(parent.canny_aperture, 2, 1)
    controls_layout.addWidget(parent.canny_container)

    parent.prewitt_container = QWidget()
    prewitt_layout = QHBoxLayout(parent.prewitt_container)
    prewitt_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.prewitt_container)

    parent.roberts_container = QWidget()
    roberts_layout = QHBoxLayout(parent.roberts_container)
    roberts_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.roberts_container)

    parent.kirsch_container = QWidget()
    kirsch_layout = QHBoxLayout(parent.kirsch_container)
    kirsch_layout.addWidget(QLabel("(нет параметров)"))
    controls_layout.addWidget(parent.kirsch_container)

    # Скрыть контейнеры
    parent.sobel_container.setVisible(True)
    parent.laplacian_container.setVisible(False)
    parent.canny_container.setVisible(False)
    parent.prewitt_container.setVisible(False)
    parent.roberts_container.setVisible(False)
    parent.kirsch_container.setVisible(False)

    # Порог бинаризации
    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(QLabel("Gradient threshold (0-255):"))
    parent.threshold_slider = QSlider(Qt.Horizontal)
    parent.threshold_slider.setRange(0, 255)
    parent.threshold_slider.setValue(127)
    parent.threshold_label = QLabel("127")
    threshold_layout.addWidget(parent.threshold_slider)
    threshold_layout.addWidget(parent.threshold_label)
    controls_layout.addLayout(threshold_layout)

    # Морфологические операции
    morph_label = QLabel("Morphological operations (applied after binarization):")
    controls_layout.addWidget(morph_label)
    parent.morphology_widget = MorphologyWidget(contour_available=False)
    controls_layout.addWidget(parent.morphology_widget)

    # Слайдер минимальной площади (0.0-0.1%, шаг 0.001)
    min_area_layout = QHBoxLayout()
    min_area_layout.addWidget(QLabel("Min contour area (% of image, 0 = off, 0.0-0.1%):"))
    parent.min_area_spinbox = QDoubleSpinBox()
    parent.min_area_spinbox.setRange(0.0, 0.1)
    parent.min_area_spinbox.setSingleStep(0.002)
    parent.min_area_spinbox.setDecimals(3)
    parent.min_area_spinbox.setValue(0.0)
    parent.min_area_spinbox.setSuffix("%")
    min_area_layout.addWidget(parent.min_area_spinbox)
    controls_layout.addLayout(min_area_layout)

    # Чекбокс заливки контуров
    parent.fill_contours_checkbox = QCheckBox("Fill closed contours")
    parent.fill_contours_checkbox.setChecked(True)
    controls_layout.addWidget(parent.fill_contours_checkbox)

    # Инверсия маски
    parent.invert_checkbox = QCheckBox("Invert mask")
    parent.invert_checkbox.setChecked(False)
    controls_layout.addWidget(parent.invert_checkbox)

    # Режим отрисовки объектов
    controls_layout.addWidget(ResultPanel(parent))

    buttons_layout = QHBoxLayout()
    parent.auto_button = QPushButton("Запустить автоподбор")
    buttons_layout.addWidget(parent.auto_button)
    controls_layout.addLayout(buttons_layout)
    auto_config_layout = QHBoxLayout()
    parent.auto_enable_checkbox = QCheckBox("Enable auto research")
    parent.configure_auto_button = QPushButton("Configure auto research")
    for control in (parent.auto_button, parent.auto_enable_checkbox, parent.configure_auto_button):
        control.setEnabled(False)
        control.setToolTip("Автоподбор будет переведён на пиксельную морфологию отдельным последним этапом.")
    auto_config_layout.addWidget(parent.auto_enable_checkbox)
    auto_config_layout.addWidget(parent.configure_auto_button)
    controls_layout.addLayout(auto_config_layout)
    parent.search_all_checkbox = QCheckBox("Search all images")
    controls_layout.addWidget(parent.search_all_checkbox)

    finish_layout(parent)
