from ui.dataset_source import DatasetSourceWidget, annotation_format_combo

"""Guided inputs and comparison controls with explicit next-action hints."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def hint(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.PlainText)
    return label


def setup_evaluation_controls(owner, layout):
    sources = QHBoxLayout()
    reference = QGroupBox("1. Правильная ручная разметка")
    box = QVBoxLayout(reference)
    box.addWidget(
        hint(
            "Выберите эталон: изображения с правильными рамками или контурами. Программа сравнит методы с этой разметкой."
        )
    )
    owner.reference_source = DatasetSourceWidget(("reference_folder", "yolo_yaml"))
    owner.btn_truth_yaml = owner.reference_source.button("yolo_yaml")
    owner.btn_truth_folder = owner.reference_source.button("reference_folder")
    box.addWidget(owner.reference_source)
    owner.truth_path = QLineEdit()
    owner.truth_path.setReadOnly(True)
    owner.truth_path.setPlaceholderText("Эталон ещё не выбран")
    box.addWidget(owner.truth_path)
    format_row = QHBoxLayout()
    format_row.addWidget(QLabel("Формат разметки:"))
    owner.truth_format = annotation_format_combo(mixed=False)
    format_row.addWidget(owner.truth_format, 1)
    box.addLayout(format_row)
    owner.lbl_truth = hint(
        "Можно выбрать YAML, папку изображений с TXT или папку только с TXT. В последнем случае нужны отчёты с размерами снимков."
    )
    box.addWidget(owner.lbl_truth)
    sources.addWidget(reference, 1)
    reports = QGroupBox("2. Результаты автоподбора")
    box = QVBoxLayout(reports)
    box.addWidget(
        hint(
            "Добавьте готовые отчёты для тех же изображений. Можно сравнить пороговые и градиентные методы вместе."
        )
    )
    owner.report_source = DatasetSourceWidget(("reports", "report_folder"))
    owner.btn_report_files = owner.report_source.button("reports")
    owner.btn_select = owner.report_source.button("report_folder")
    owner.btn_clear_reports = QPushButton("Очистить отчёты")
    box.addWidget(owner.report_source)
    box.addWidget(owner.btn_clear_reports)

    owner.lbl_reports = hint("Отчёты ещё не загружены.")
    box.addWidget(owner.lbl_reports)
    box.addWidget(
        hint(
            "Где взять отчёты: вкладка методов → «Диапазоны автоподбора» → «Все загруженные изображения» → «Запустить автоподбор». Файлы появятся рядом со снимками."
        )
    )
    box.addStretch()
    sources.addWidget(reports, 1)
    layout.addLayout(sources)
    comparison = QGroupBox("3. Сравнить методы с эталоном")
    box = QVBoxLayout(comparison)
    owner.lbl_evaluation_status = hint("Выберите ручной эталон и добавьте отчёты.")
    box.addWidget(owner.lbl_evaluation_status)
    row = QHBoxLayout()
    owner.btn_analyze = QPushButton("Сравнить методы")
    owner.btn_analyze.setToolTip(
        "Проверить соответствие данных и сравнить каждый полный пресет на всём эталоне. Результат учитывает пропущенные и лишние объекты."
    )
    owner.btn_stop_evaluation = QPushButton("Остановить")
    owner.btn_stop_evaluation.setToolTip(
        "Остановить чтение отчётов или сравнение. Незавершённая оценка не становится результатом."
    )
    owner.progress_bar = QProgressBar()
    owner.progress_bar.setVisible(False)
    row.addWidget(owner.btn_analyze)
    row.addWidget(owner.btn_stop_evaluation)
    row.addWidget(owner.progress_bar, 1)
    owner.settings_toggle = QCheckBox("Условия сравнения")
    row.addWidget(owner.settings_toggle)
    box.addLayout(row)
    settings = QWidget()
    settings.setVisible(False)
    grid = QGridLayout(settings)
    owner.evaluation_iou = QDoubleSpinBox()
    owner.evaluation_iou.setRange(0.05, 1.0)
    owner.evaluation_iou.setSingleStep(0.05)
    owner.evaluation_iou.setValue(0.5)
    owner.evaluation_iou.setToolTip(
        "IoU — площадь пересечения рамок/контуров, делённая на площадь их объединения. При 0.50 совпадение требует IoU не ниже 50%; более высокий порог строже."
    )
    owner.class_aware = QCheckBox("Требовать совпадения классов объектов")
    owner.class_aware.setToolTip(
        "Включайте, если предсказания должны иметь те же class ID, что и ручная разметка. Классические методы обычно выделяют геометрию и не распознают семантические классы."
    )
    grid.addWidget(QLabel("Минимальное совпадение рамок/контуров (IoU):"), 0, 0)
    grid.addWidget(owner.evaluation_iou, 0, 1)
    grid.addWidget(owner.class_aware, 0, 2)
    grid.addWidget(
        hint(
            "Рекомендуемое начало: IoU = 0.50, без учёта классов. Для выбора метода нужны положительные примеры. Пресет может стать победителем только после успешной проверки на всех снимках эталона."
        ),
        1,
        0,
        1,
        3,
    )
    owner.settings_toggle.toggled.connect(settings.setVisible)
    box.addWidget(settings)
    layout.addWidget(comparison)
