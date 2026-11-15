"""Primary reference → reports → measured recommendation workflow."""

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
)


def setup_evaluation_controls(owner, layout):
    group = QGroupBox("Выбор методов по ручной разметке")
    grid = QGridLayout(group)
    owner.evaluation_grid = grid
    owner.btn_truth_yaml = QPushButton("1. Эталон: YOLO YAML")
    owner.btn_truth_folder = QPushButton("1. Эталон: папка")
    owner.btn_report_files = QPushButton("2. Добавить отчёты")
    owner.btn_stop_evaluation = QPushButton("Остановить оценку")
    owner.btn_stop_evaluation.setEnabled(False)
    owner.btn_export_evaluation = QPushButton("Сохранить оценку")
    owner.btn_export_evaluation.setEnabled(False)
    for i, widget in enumerate(
        (
            owner.btn_truth_yaml,
            owner.btn_truth_folder,
            owner.btn_report_files,
            owner.btn_stop_evaluation,
            owner.btn_export_evaluation,
        )
    ):
        grid.addWidget(widget, 0, i)
    owner.btn_clear_reports = QPushButton("Очистить отчёты")
    grid.addWidget(owner.btn_clear_reports, 4, 4)
    owner.lbl_truth = QLabel(
        "Эталон не выбран. Папка: images + labels либо снимки с TXT; папку только с TXT загружайте вместе с отчётами."
    )
    owner.lbl_truth.setWordWrap(True)
    grid.addWidget(owner.lbl_truth, 1, 0, 1, 5)
    owner.truth_format = QComboBox()
    for title, key in [
        ("YOLO: автоматически", "auto"),
        ("BBox", "detect"),
        ("OBB: углы", "obb_corners"),
        ("OBB: XYWHR", "obb_xywhr"),
        ("Segment", "segment"),
    ]:
        owner.truth_format.addItem(title, key)
    owner.evaluation_iou = QDoubleSpinBox()
    owner.evaluation_iou.setRange(0.05, 1.0)
    owner.evaluation_iou.setSingleStep(0.05)
    owner.evaluation_iou.setValue(0.5)
    owner.class_aware = QCheckBox("Учитывать классы объектов")
    owner.class_aware.setChecked(False)
    owner.class_aware.setToolTip(
        "Классические методы выделяют объекты без распознавания семантики. Выключено: сравнивать геометрию всех классов. Включено: совпадение требует одинакового class ID."
    )
    grid.addWidget(owner.truth_format, 2, 0)
    grid.addWidget(QLabel("IoU для Precision / Recall / F1:"), 2, 1)
    grid.addWidget(owner.evaluation_iou, 2, 2)
    grid.addWidget(owner.class_aware, 2, 3, 1, 2)
    owner.lbl_evaluation_status = QLabel(
        "3. Запустите оценку всего эталона. Ошибки и отсутствующие вычисления исключают пресет из выбора победителя."
    )
    owner.lbl_evaluation_status.setWordWrap(True)
    grid.addWidget(owner.lbl_evaluation_status, 3, 0, 1, 5)
    layout.addWidget(group)
