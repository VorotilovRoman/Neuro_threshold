"""Dataset analysis UI page construction."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_spatial(parent):
    # 4. Тепловая карта объектов
    # ============================================================
    heat_content = QWidget()
    hc = QVBoxLayout(heat_content)
    hc.setContentsMargins(0, 0, 0, 0)

    parent.heatmap_summary_label = QLabel("Постройте тепловую карту.")
    parent.heatmap_summary_label.setWordWrap(True)
    hc.addWidget(parent.heatmap_summary_label)

    grid_row = QHBoxLayout()
    grid_row.addWidget(QLabel("Размер сетки:"))
    parent.heatmap_grid_spin = QSpinBox()
    parent.heatmap_grid_spin.setRange(8, 128)
    parent.heatmap_grid_spin.setValue(16)
    parent.heatmap_grid_spin.setSingleStep(8)
    parent.heatmap_grid_spin.setToolTip("16×16 — обзорно; 64–128 — детальный анализ.")
    grid_row.addWidget(parent.heatmap_grid_spin)
    grid_row.addStretch()
    hc.addLayout(grid_row)

    parent.heatmap_figure = Figure(figsize=(5, 4))
    parent.heatmap_canvas = FigureCanvas(parent.heatmap_figure)
    parent.heatmap_ax = parent.heatmap_figure.add_subplot(111)
    parent.heatmap_canvas.setMinimumHeight(320)
    hc.addWidget(parent.heatmap_canvas, 1)

    parent.btn_run_heatmap = QPushButton("Построить тепловую карту")
    parent.btn_stop_heatmap = QPushButton("⏹ Остановить")
    parent.btn_stop_heatmap.setEnabled(False)

    _make_page(
        parent.toolbox,
        "4. Тепловая карта объектов",
        "Центры bbox/полигонов на сетке N×N.\n"
        "Оси X/Y — % от ширины/высоты. Цвет — "
        "число центров объектов в ячейке.\n"
        "Размер сетки 8…128. Область — из раздела «Начать и проверить».",
        heat_content,
        buttons=[parent.btn_run_heatmap, parent.btn_stop_heatmap],
        min_height=420,
        content_stretch=1,
    )

    # ============================================================
