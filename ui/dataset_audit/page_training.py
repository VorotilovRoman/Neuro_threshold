"""Dataset analysis UI page construction."""

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import TrainingImageLabel, _make_page


def build_training(parent):
    # 7. Результаты обучения модели
    # ============================================================
    tr_content = QWidget()
    tcl = QVBoxLayout(tr_content)
    tcl.setContentsMargins(0, 0, 0, 0)
    tcl.setSpacing(8)

    tr_row = QHBoxLayout()
    parent.training_results_edit = QLineEdit()
    parent.training_results_edit.setReadOnly(True)
    parent.training_results_edit.setPlaceholderText(
        "Папка run/ с результатами обучения не выбрана"
    )
    parent.btn_browse_training_results = QPushButton("Обзор…")
    parent.btn_load_training_results = QPushButton("Загрузить")
    tr_row.addWidget(QLabel("Папка run/:"))
    tr_row.addWidget(parent.training_results_edit, 1)
    tr_row.addWidget(parent.btn_browse_training_results)
    tr_row.addWidget(parent.btn_load_training_results)
    tcl.addLayout(tr_row)

    parent.training_results_summary_label = QLabel(
        "Укажите папку, где лежат results.png, labels.jpg, "
        "confusion_matrix_normalized.png, weights/."
    )
    parent.training_results_summary_label.setWordWrap(True)
    tcl.addWidget(parent.training_results_summary_label)

    for key, title in (
        ("results", "results.png — кривые обучения"),
        ("labels", "labels.jpg — распределение bbox"),
        ("confusion", "confusion_matrix_normalized.png"),
    ):
        lbl_title = QLabel(f"<b>{title}</b>")
        tcl.addWidget(lbl_title)
        img_lbl = TrainingImageLabel()
        setattr(parent, f"training_img_{key}", img_lbl)
        tcl.addWidget(img_lbl)

    _make_page(
        parent.toolbox,
        "7. Результаты обучения модели",
        "Укажите папку run/ (где лежат результаты обучения YOLO). "
        "**В окне** отображаются: `results.png`, `labels.jpg`, "
        "`confusion_matrix_normalized.png`.\n"
        "**Остальные файлы** (`Box*/Mask*_curve.png`, "
        "`confusion_matrix.png`, `results.csv`, `args.yaml`, "
        "`train_batch*`, `val_batch*`, `weights/*`) попадают в "
        "Markdown-отчёт.\n"
        "**При сохранении отчёта** в `dataset_report_assets/` "
        "копируются `best.pt` и `best_meta.json` (если есть).",
        tr_content,
        buttons=[],
        min_height=650,
        content_stretch=1,
    )

    # ============================================================
