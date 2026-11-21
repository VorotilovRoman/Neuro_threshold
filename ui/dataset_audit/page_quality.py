"""Dataset analysis UI page construction."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_quality(parent):
    # 2. Качество изображений
    # ============================================================
    quality_content = QWidget()
    qc = QVBoxLayout(quality_content)
    qc.setContentsMargins(0, 0, 0, 0)

    parent.quality_summary_label = QLabel("Запустите анализ качества.")
    parent.quality_summary_label.setWordWrap(True)
    qc.addWidget(parent.quality_summary_label)
    options = QGridLayout()
    for row, (name, title, minimum, maximum, value) in enumerate(
        [
            ("quality_dark", "Тёмный: средняя яркость ниже", 0, 254, 40),
            ("quality_bright", "Светлый: средняя яркость выше", 1, 255, 215),
            ("quality_contrast", "Низкий контраст: std ниже (0 = выкл.)", 0, 255, 10),
            (
                "quality_sharpness",
                "Резкость: var(Laplacian) ниже (0 = выкл.)",
                0,
                1000000,
                20,
            ),
        ]
    ):
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        setattr(parent, name, control)
        label = QLabel(title)
        label.setWordWrap(True)
        options.addWidget(label, row, 0)
        options.addWidget(control, row, 1)
        control.valueChanged.connect(parent.quality_parameters_changed)
    qc.addLayout(options)
    note = QLabel(
        "Яркость и контраст — в шкале 0..255 после преобразования просмотра, резкость — условная величина на исходной сетке. Пороги подберите для своего типа снимков."
    )
    note.setWordWrap(True)
    qc.addWidget(note)
    parent.quality_only_issues = QCheckBox("Показывать только кандидатов")
    parent.quality_only_issues.setChecked(True)
    qc.addWidget(parent.quality_only_issues)
    parent.quality_table = QTableWidget(0, 6)
    parent.quality_table.setHorizontalHeaderLabels(
        ["Снимок", "Размер", "Яркость", "Контраст", "Резкость", "Проверить"]
    )
    parent.quality_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeToContents
    )
    parent.quality_table.horizontalHeader().setStretchLastSection(True)
    parent.quality_table.setToolTip(
        "Двойной клик — открыть исходный снимок и его разметку."
    )
    parent.quality_table.setMinimumHeight(160)
    parent.quality_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    qc.addWidget(parent.quality_table)
    parent.quality_only_issues.toggled.connect(parent._populate_quality_table)
    parent.quality_table.cellDoubleClicked.connect(parent.open_quality_row)

    parent.quality_figure = Figure(figsize=(6, 4))
    parent.quality_canvas = FigureCanvas(parent.quality_figure)
    parent.quality_ax1 = parent.quality_figure.add_subplot(221)
    parent.quality_ax2 = parent.quality_figure.add_subplot(222)
    parent.quality_ax3 = parent.quality_figure.add_subplot(223)
    parent.quality_ax4 = parent.quality_figure.add_subplot(224)
    parent.quality_canvas.setMinimumHeight(340)
    qc.addWidget(parent.quality_canvas, 1)

    parent.btn_run_quality = QPushButton("Запустить анализ")
    parent.btn_stop_quality = QPushButton("⏹ Остановить")
    parent.btn_stop_quality.setEnabled(False)

    _make_page(
        parent.toolbox,
        "2. Качество изображений",
        "Проверьте читаемость и яркость, контраст и резкость по заданным порогам. "
        "Двойной клик по строке открывает снимок с разметкой. "
        "Область проверки выбирается в разделе «Начать и проверить».",
        quality_content,
        buttons=[parent.btn_run_quality, parent.btn_stop_quality],
        min_height=400,
        content_stretch=1,
    )

    # ============================================================
