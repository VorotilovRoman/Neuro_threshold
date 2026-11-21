"""Dataset analysis UI page construction."""

from PyQt5.QtWidgets import QCheckBox, QLabel, QPushButton, QVBoxLayout, QWidget

from ui.dataset_audit.page_helpers import _make_page


def build_report(parent):
    # 8. Общий отчёт
    # ============================================================
    report_content = QWidget()
    rc = QVBoxLayout(report_content)
    rc.setContentsMargins(0, 0, 0, 0)

    parent.report_info_label = QLabel(
        "Собирает все результаты текущей сессии в один Markdown-файл "
        "(в фоновом потоке — UI не блокируется).\n"
        "Рядом с .md создаётся папка <name>_assets/ с PNG "
        "(areas, classes, quality, heatmap, cluster) и, если загружены "
        "результаты обучения — `best.pt`, `best_meta.json`."
    )
    parent.report_info_label.setWordWrap(True)
    rc.addWidget(parent.report_info_label)

    parent.chk_report_include_quality = QCheckBox("Включить качество изображений")
    parent.chk_report_include_dup = QCheckBox("Включить дубликаты и утечку")
    parent.chk_report_include_heatmap = QCheckBox("Включить тепловую карту")
    parent.chk_report_include_audit = QCheckBox("Включить Loss Inspection")
    parent.chk_report_include_validation = QCheckBox("Включить валидацию YOLO")
    parent.chk_report_include_cluster = QCheckBox("Включить кластеризацию")
    parent.chk_report_include_training_results = QCheckBox(
        "Включить результаты обучения модели"
    )
    for cb in (
        parent.chk_report_include_quality,
        parent.chk_report_include_dup,
        parent.chk_report_include_heatmap,
        parent.chk_report_include_audit,
        parent.chk_report_include_validation,
        parent.chk_report_include_cluster,
        parent.chk_report_include_training_results,
    ):
        cb.setChecked(True)
        rc.addWidget(cb)

    rc.addStretch()
    parent.btn_generate_report = QPushButton("💾 Сгенерировать .md отчёт")
    parent.btn_generate_report.setStyleSheet(
        "background-color: #673AB7; color: white; font-weight: bold;"
    )
    parent.btn_open_report_folder = QPushButton("📂 Открыть папку отчёта")
    parent.btn_open_report_folder.setToolTip(
        "Открыть папку с последним отчётом (.md) и assets"
    )
    parent.btn_open_report_folder.setEnabled(False)

    _make_page(
        parent.toolbox,
        "8. Общий отчёт",
        "Собирает результаты всех запущенных анализов в один файл.",
        report_content,
        buttons=[parent.btn_generate_report, parent.btn_open_report_folder],
        min_height=340,
        content_stretch=0,
    )

    # ============================================================
