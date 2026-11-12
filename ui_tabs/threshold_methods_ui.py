from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout

# -*- coding: utf-8 -*-
"""Qt интерфейс пороговых методов с четырьмя read-only окнами utils_new."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.morphology_ui import MorphologyWidget


def setup_threshold_ui(parent):
    """
    Создаёт UI для пороговых методов внутри QMainWindow.
    parent – экземпляр QMainWindow (например, ThresholdWindow).
    """
    # Создаём центральный виджет
    controls_layout = begin_layout(parent, 'threshold')

    # ----- Выбор метода бинаризации -----
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Binarization method:"))
    parent.method_combo = QComboBox()
    methods = [
        "Simple Threshold",
        "Otsu",
        "Triangle",
        "Adaptive Mean",
        "Adaptive Gauss",
        "Niblack",
        "Sauvola",
        "ISODATA",
        "Background Symmetry",
        "Row Adaptive"
    ]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # ----- Контейнеры параметров для каждого метода -----
    # 1. Simple Threshold
    parent.simple_threshold_container = QWidget()
    simple_th_layout = QHBoxLayout(parent.simple_threshold_container)
    simple_th_layout.addWidget(QLabel("Threshold value:"))
    parent.threshold_slider = QSlider(Qt.Horizontal)
    parent.threshold_slider.setRange(0, 255)
    parent.threshold_slider.setValue(127)
    parent.threshold_value_label = QLabel("127")
    simple_th_layout.addWidget(parent.threshold_slider)
    simple_th_layout.addWidget(parent.threshold_value_label)
    controls_layout.addWidget(parent.simple_threshold_container)

    # 2. Adaptive Mean / Gauss
    parent.adaptive_container = QWidget()
    adaptive_layout = QHBoxLayout(parent.adaptive_container)
    adaptive_layout.addWidget(QLabel("Window size (odd):"))
    parent.adaptive_win = QSpinBox()
    parent.adaptive_win.setRange(3, 201)
    parent.adaptive_win.setSingleStep(2)
    parent.adaptive_win.setValue(25)
    adaptive_layout.addWidget(parent.adaptive_win)
    adaptive_layout.addWidget(QLabel("C:"))
    parent.adaptive_c = QSpinBox()
    parent.adaptive_c.setRange(-20, 20)
    parent.adaptive_c.setValue(3)
    adaptive_layout.addWidget(parent.adaptive_c)
    controls_layout.addWidget(parent.adaptive_container)

    # 3. Niblack
    parent.niblack_container = QWidget()
    niblack_layout = QVBoxLayout(parent.niblack_container)
    win_layout = QHBoxLayout()
    win_layout.addWidget(QLabel("Window size (odd):"))
    parent.niblack_win = QSpinBox()
    parent.niblack_win.setRange(3, 201)
    parent.niblack_win.setSingleStep(2)
    parent.niblack_win.setValue(25)
    win_layout.addWidget(parent.niblack_win)
    niblack_layout.addLayout(win_layout)

    k_layout = QHBoxLayout()
    k_layout.addWidget(QLabel("k:"))
    parent.niblack_k = QSlider(Qt.Horizontal)
    parent.niblack_k.setRange(-100, 100)
    parent.niblack_k.setValue(20)
    parent.niblack_k_label = QLabel("0.20")
    k_layout.addWidget(parent.niblack_k)
    k_layout.addWidget(parent.niblack_k_label)
    niblack_layout.addLayout(k_layout)
    controls_layout.addWidget(parent.niblack_container)

    # 4. Sauvola
    parent.sauvola_container = QWidget()
    sauvola_layout = QVBoxLayout(parent.sauvola_container)
    win_s_layout = QHBoxLayout()
    win_s_layout.addWidget(QLabel("Window size (odd):"))
    parent.sauvola_win = QSpinBox()
    parent.sauvola_win.setRange(3, 201)
    parent.sauvola_win.setSingleStep(2)
    parent.sauvola_win.setValue(25)
    win_s_layout.addWidget(parent.sauvola_win)
    sauvola_layout.addLayout(win_s_layout)

    k_s_layout = QHBoxLayout()
    k_s_layout.addWidget(QLabel("k:"))
    parent.sauvola_k = QSlider(Qt.Horizontal)
    parent.sauvola_k.setRange(-100, 100)
    parent.sauvola_k.setValue(20)
    parent.sauvola_k_label = QLabel("0.20")
    k_s_layout.addWidget(parent.sauvola_k)
    k_s_layout.addWidget(parent.sauvola_k_label)
    sauvola_layout.addLayout(k_s_layout)

    r_layout = QHBoxLayout()
    r_layout.addWidget(QLabel("R:"))
    parent.sauvola_r = QSlider(Qt.Horizontal)
    parent.sauvola_r.setRange(1, 255)
    parent.sauvola_r.setValue(128)
    parent.sauvola_r_label = QLabel("128")
    r_layout.addWidget(parent.sauvola_r)
    r_layout.addWidget(parent.sauvola_r_label)
    sauvola_layout.addLayout(r_layout)
    controls_layout.addWidget(parent.sauvola_container)

    # 5. ISODATA
    parent.isodata_container = QWidget()
    iso_layout = QHBoxLayout(parent.isodata_container)
    iso_layout.addWidget(QLabel("Initial threshold:"))
    parent.isodata_init = QSpinBox()
    parent.isodata_init.setRange(0, 255)
    parent.isodata_init.setValue(128)
    iso_layout.addWidget(parent.isodata_init)
    controls_layout.addWidget(parent.isodata_container)

    # 6. Background Symmetry
    parent.background_container = QWidget()
    bg_layout = QVBoxLayout(parent.background_container)
    bg_thresh_layout = QHBoxLayout()
    bg_thresh_layout.addWidget(QLabel("Excess threshold (sensitivity):"))
    parent.bg_excess = QSlider(Qt.Horizontal)
    parent.bg_excess.setRange(1, 100)
    parent.bg_excess.setValue(20)
    parent.bg_excess_label = QLabel("0.20")
    bg_thresh_layout.addWidget(parent.bg_excess)
    bg_thresh_layout.addWidget(parent.bg_excess_label)
    bg_layout.addLayout(bg_thresh_layout)
    controls_layout.addWidget(parent.background_container)

    # 7. Row Adaptive
    parent.row_adaptive_container = QWidget()
    row_layout = QVBoxLayout(parent.row_adaptive_container)
    win_row_layout = QHBoxLayout()
    win_row_layout.addWidget(QLabel("Window size:"))
    parent.row_win = QSpinBox()
    parent.row_win.setRange(10, 500)
    parent.row_win.setValue(50)
    win_row_layout.addWidget(parent.row_win)
    row_layout.addLayout(win_row_layout)

    k_row_layout = QHBoxLayout()
    k_row_layout.addWidget(QLabel("k:"))
    parent.row_k = QSlider(Qt.Horizontal)
    parent.row_k.setRange(0, 100)
    parent.row_k.setValue(50)
    parent.row_k_label = QLabel("0.50")
    k_row_layout.addWidget(parent.row_k)
    k_row_layout.addWidget(parent.row_k_label)
    row_layout.addLayout(k_row_layout)
    controls_layout.addWidget(parent.row_adaptive_container)

    # Скрываем все контейнеры изначально
    parent.simple_threshold_container.setVisible(False)
    parent.adaptive_container.setVisible(False)
    parent.niblack_container.setVisible(False)
    parent.sauvola_container.setVisible(False)
    parent.isodata_container.setVisible(False)
    parent.background_container.setVisible(False)
    parent.row_adaptive_container.setVisible(False)

    # Invert mask checkbox
    parent.invert_checkbox = QCheckBox("Invert mask")
    parent.invert_checkbox.setChecked(True)
    controls_layout.addWidget(parent.invert_checkbox)

    # Общий UI морфологии, применяемой ко всей бинарной маске.
    # Все четыре операции доступны; отключённый шаг = 0 или 1 px.
    parent.morphology_widget = MorphologyWidget(
        parent, contour_available=False,
        initial_order=("close", "open", "erode", "dilate"),
        title="Морфология бинарной маски",
    )
    controls_layout.addWidget(parent.morphology_widget)

    # ----- Режим отрисовки объектов -----
    controls_layout.addWidget(ResultPanel(parent))

    from ui.autoreseach_ui.controls import AutoResearchPanel
    controls_layout.addWidget(AutoResearchPanel(parent))

    finish_layout(parent)
