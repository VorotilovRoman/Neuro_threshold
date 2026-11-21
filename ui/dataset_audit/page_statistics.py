"""Dataset analysis UI page construction."""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_statistics(parent):
    # 1. Классы, объекты и площади
    # ============================================================
    metrics_content = QWidget()
    mc = QVBoxLayout(metrics_content)
    mc.setContentsMargins(0, 0, 0, 0)

    parent.metrics_summary_label = QLabel("Загрузите датасет для отображения метрик.")
    parent.metrics_summary_label.setWordWrap(True)
    mc.addWidget(parent.metrics_summary_label)

    parent.class_stats_table = QTableWidget()
    parent.class_stats_table.setColumnCount(3)
    parent.class_stats_table.setHorizontalHeaderLabels(
        ["Класс", "Кол-во объектов", "Доля (%)"]
    )
    parent.class_stats_table.horizontalHeader().setSectionResizeMode(
        QHeaderView.Stretch
    )
    parent.class_stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.class_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    parent.class_stats_table.setMinimumHeight(120)
    mc.addWidget(parent.class_stats_table, 1)

    # --- Фигура площадей: 3 подграфика (px², %, классы) ---
    parent.figure = Figure(figsize=(8, 3.5))
    parent.canvas = FigureCanvas(parent.figure)
    parent.ax = parent.figure.add_subplot(131)  # площадь px²
    parent.ax_area_pct = parent.figure.add_subplot(132)  # площадь %
    parent.ax_classes = parent.figure.add_subplot(133)  # по классам
    parent.canvas.setMinimumHeight(280)
    mc.addWidget(parent.canvas, 2)

    parent.class_chart_figure = Figure(figsize=(6, 2.8))
    parent.class_chart_canvas = FigureCanvas(parent.class_chart_figure)
    parent.ax_pie = parent.class_chart_figure.add_subplot(121)
    parent.ax_bar = parent.class_chart_figure.add_subplot(122)
    parent.class_chart_canvas.setMinimumHeight(220)
    mc.addWidget(parent.class_chart_canvas, 1)

    imb_header = QLabel("<b>Дисбаланс классов</b>")
    mc.addWidget(imb_header)

    parent.imbalance_summary_label = QLabel(
        "Рассчитывается автоматически после фильтрации."
    )
    parent.imbalance_summary_label.setWordWrap(True)
    mc.addWidget(parent.imbalance_summary_label)

    parent.imbalance_table = QTableWidget()
    parent.imbalance_table.setColumnCount(2)
    parent.imbalance_table.setHorizontalHeaderLabels(["Метрика", "Значение"])
    parent.imbalance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.imbalance_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.imbalance_table.setMinimumHeight(140)
    parent.imbalance_table.setToolTip(
        "Наведите курсор на строку — всплывёт описание метрики."
    )
    mc.addWidget(parent.imbalance_table, 1)

    _make_page(
        parent.toolbox,
        "1. Классы, объекты и площади",
        "**Таблица классов** — число объектов каждого класса и доля. "
        "**Площади** — bbox: w·h, segment/OBB: shoelace. "
        "Слева — гистограмма площадей в пикселях, в центре — "
        "в % от площади изображения, справа — распределение по классам. "
        "**Круговая/столбчатая диаграммы** — распределение по классам. "
        "**Дисбаланс:** ratio = max/min; Gini (0 — равномерно); "
        "Entropy (выше — равномернее). Тултипы на строках таблицы.",
        metrics_content,
        buttons=[],
        min_height=650,
        content_stretch=1,
    )

    # ============================================================
