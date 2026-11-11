from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ui.morphology_ui import MorphologyWidget

# interactive_methods_ui.py
from ui.segmentation.panels import ResultPanel, begin_layout, finish_layout


def setup_interactive_ui(parent):
    """Создаёт UI для вкладки интерактивных методов сегментации."""
    controls_layout = begin_layout(parent, 'interactive')

    # Выбор метода
    method_layout = QHBoxLayout()
    method_layout.addWidget(QLabel("Method:"))
    parent.method_combo = QComboBox()
    methods = ["GrabCut", "Lazy Snapping", "SuperCut", "OneCut", "Random Walker", "Watershed (marker)", "Active Contours"]
    parent.method_combo.addItems(methods)
    method_layout.addWidget(parent.method_combo)
    controls_layout.addLayout(method_layout)

    # Группа для отображения требований
    parent.requirements_frame = QFrame()
    parent.requirements_frame.setFrameShape(QFrame.StyledPanel)
    req_layout = QHBoxLayout(parent.requirements_frame)
    parent.requirements_label = QLabel("Требования:")
    parent.requirements_label.setStyleSheet("font-weight: bold;")
    req_layout.addWidget(parent.requirements_label)
    parent.requirements_text = QLabel("")
    parent.requirements_text.setWordWrap(True)
    req_layout.addWidget(parent.requirements_text)
    controls_layout.addWidget(parent.requirements_frame)

    # Параметры методов
    from ui.interactive_params_ui import InteractiveParametersUI
    parent.params_widget = InteractiveParametersUI(parent)
    controls_layout.addWidget(parent.params_widget)

    # ---------- Инструменты интерактивного ввода ----------
    input_group = QGroupBox("Interactive Input")
    input_layout = QVBoxLayout()

    row1 = QHBoxLayout()
    parent.rect_mode_btn = QPushButton("Draw Bounding Box")
    parent.rect_mode_btn.setCheckable(True)
    parent.fg_scribble_btn = QPushButton("Scribble FG")
    parent.fg_scribble_btn.setCheckable(True)
    parent.bg_scribble_btn = QPushButton("Scribble BG")
    parent.bg_scribble_btn.setCheckable(True)
    row1.addWidget(parent.rect_mode_btn)
    row1.addWidget(parent.fg_scribble_btn)
    row1.addWidget(parent.bg_scribble_btn)
    input_layout.addLayout(row1)

    row2 = QHBoxLayout()
    parent.reset_mask_btn = QPushButton("Reset Input")
    parent.run_seg_btn = QPushButton("Run Segmentation")
    row2.addWidget(parent.reset_mask_btn)
    row2.addWidget(parent.run_seg_btn)
    input_layout.addLayout(row2)

    input_group.setLayout(input_layout)
    controls_layout.addWidget(input_group)

    # Invert mask
    parent.invert_checkbox = QCheckBox("Invert mask")
    controls_layout.addWidget(parent.invert_checkbox)

    # Морфологические операции
    morph_label = QLabel("Morphological operations (applied after segmentation):")
    controls_layout.addWidget(morph_label)

    parent.morphology_widget = MorphologyWidget(contour_available=False)
    controls_layout.addWidget(parent.morphology_widget)

    # Режим отрисовки объектов
    controls_layout.addWidget(ResultPanel(parent))

    finish_layout(parent)
