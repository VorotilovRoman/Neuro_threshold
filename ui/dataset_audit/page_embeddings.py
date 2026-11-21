"""Dataset analysis UI page construction."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_embeddings(parent):
    # 6. Кластеризация эмбеддингов
    # ============================================================
    cluster_content = QWidget()
    cc = QVBoxLayout(cluster_content)
    cc.setContentsMargins(0, 0, 0, 0)

    parent.cluster_summary_label = QLabel(
        "Требуется CLIP (open_clip_torch или transformers). UMAP и HDBSCAN опциональны."
    )
    parent.cluster_summary_label.setWordWrap(True)
    cc.addWidget(parent.cluster_summary_label)

    cluster_radius_row = QHBoxLayout()
    cluster_radius_row.addWidget(QLabel("Радиус клика по точке:"))
    parent.cluster_click_radius_spin = QDoubleSpinBox()
    parent.cluster_click_radius_spin.setRange(0.1, 5.0)
    parent.cluster_click_radius_spin.setValue(0.5)
    parent.cluster_click_radius_spin.setSingleStep(0.1)
    parent.cluster_click_radius_spin.setDecimals(2)
    parent.cluster_click_radius_spin.setToolTip(
        "Радиус захвата клика в UMAP-пространстве."
    )
    cluster_radius_row.addWidget(parent.cluster_click_radius_spin)
    cluster_radius_row.addStretch()
    cc.addLayout(cluster_radius_row)

    parent.cluster_figure = Figure(figsize=(5, 4))
    parent.cluster_canvas = FigureCanvas(parent.cluster_figure)
    parent.cluster_ax = parent.cluster_figure.add_subplot(111)
    parent.cluster_canvas.setMinimumHeight(320)
    cc.addWidget(parent.cluster_canvas, 1)

    parent.cluster_toolbar = NavigationToolbar(parent.cluster_canvas, parent)
    cc.addWidget(parent.cluster_toolbar)

    parent.btn_run_cluster = QPushButton("Кластеризовать эмбеддинги")
    parent.btn_stop_cluster = QPushButton("⏹ Остановить")
    parent.btn_stop_cluster.setEnabled(False)

    _make_page(
        parent.toolbox,
        "6. Кластеризация эмбеддингов",
        "CLIP → UMAP (2D) → HDBSCAN.\n"
        "**Панель инструментов над графиком:** 🔍 зум, ✋ pan, "
        "🏠 сброс, 💾 сохранить PNG.\n"
        "**Клик по точке** (не в зуме) — открыть пару. "
        "«Шум» — выбросы. **Работает по filtered_pairs.**",
        cluster_content,
        buttons=[parent.btn_run_cluster, parent.btn_stop_cluster],
        min_height=420,
        content_stretch=1,
    )

    # ============================================================
