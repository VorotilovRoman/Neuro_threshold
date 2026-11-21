"""Dataset analysis UI page construction."""

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_duplicates(parent):
    # 3. Дубликаты и утечка
    # ============================================================
    dup_content = QWidget()
    dc = QVBoxLayout(dup_content)
    threshold_row = QHBoxLayout()
    threshold_row.addWidget(QLabel("Похожесть pHash: максимальное расстояние битов"))
    parent.phash_threshold = QSpinBox()
    parent.phash_threshold.setRange(0, 64)
    parent.phash_threshold.setValue(8)
    parent.phash_threshold.valueChanged.connect(parent.duplicate_parameters_changed)
    threshold_row.addWidget(parent.phash_threshold)
    dc.addLayout(threshold_row)

    dc.setContentsMargins(0, 0, 0, 0)

    parent.dup_summary_label = QLabel("Нажмите «Найти дубликаты».")
    parent.dup_summary_label.setWordWrap(True)
    dc.addWidget(parent.dup_summary_label)

    parent.dup_table = QTableWidget()
    parent.dup_table.setColumnCount(5)
    parent.dup_table.setHorizontalHeaderLabels(
        ["Группа", "N", "Сплиты", "Файлы", "Утечка"]
    )
    parent.dup_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.dup_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.dup_table.setSelectionBehavior(QAbstractItemView.SelectRows)
    parent.dup_table.setSelectionMode(QAbstractItemView.SingleSelection)
    parent.dup_table.setMinimumHeight(140)
    parent.dup_table.setToolTip(
        "Каждая строка — группа дубликатов.\n"
        "Одиночный клик по строке — выделить все пары этой группы в "
        "списке слева.\n"
        "Двойной клик — открыть валидацию всех пар группы "
        "(последовательно)."
    )
    dc.addWidget(parent.dup_table, 1)

    parent.btn_run_duplicates = QPushButton("Найти дубликаты и утечку")
    parent.btn_stop_duplicates = QPushButton("⏹ Остановить")
    parent.btn_stop_duplicates.setEnabled(False)

    _make_page(
        parent.toolbox,
        "3. Дубликаты и утечка",
        "Точные повторы — одинаковые байты (MD5). Похожие — pHash с расстоянием "
        "не выше выбранного порога: меньше означает строже. Утечка — группа "
        "в нескольких split. Строка выделяет файлы слева; двойной клик открывает "
        "просмотр группы. Область выбирается в разделе «Начать и проверить».",
        dup_content,
        buttons=[parent.btn_run_duplicates, parent.btn_stop_duplicates],
        min_height=320,
        content_stretch=1,
    )

    # ============================================================
