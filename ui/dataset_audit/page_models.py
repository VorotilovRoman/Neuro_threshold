"""Dataset analysis UI page construction."""

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from ui.dataset_audit.page_helpers import _make_page


def build_models(parent):
    # 5. Аудит модели (Loss Inspection) + валидация YOLO
    # ============================================================
    audit_content = QWidget()
    ac = QVBoxLayout(audit_content)
    ac.setContentsMargins(0, 0, 0, 0)

    model_row = QHBoxLayout()
    parent.audit_model_edit = QLineEdit()
    parent.audit_model_edit.setReadOnly(True)
    parent.audit_model_edit.setPlaceholderText("Модель не выбрана")
    parent.btn_load_audit_model = QPushButton("Загрузить модель")
    model_row.addWidget(QLabel("YOLO:"))
    model_row.addWidget(parent.audit_model_edit, 1)
    model_row.addWidget(parent.btn_load_audit_model)
    ac.addLayout(model_row)

    params_row = QHBoxLayout()
    parent.audit_iou_spin = QDoubleSpinBox()
    parent.audit_iou_spin.setRange(0.0, 1.0)
    parent.audit_iou_spin.setValue(0.3)
    parent.audit_iou_spin.setSingleStep(0.05)
    parent.audit_iou_spin.setDecimals(2)
    parent.audit_iou_spin.setToolTip(
        "Порог среднего IoU между GT и предсказанием.\n"
        "Пары с mean IoU < порога попадают в «подозрительные»."
    )
    parent.audit_conf_spin = QDoubleSpinBox()
    parent.audit_conf_spin.setRange(0.0, 1.0)
    parent.audit_conf_spin.setValue(0.25)
    parent.audit_conf_spin.setSingleStep(0.05)
    parent.audit_conf_spin.setDecimals(2)
    parent.audit_conf_spin.setToolTip("Confidence-порог модели при аудите.")
    parent.audit_imgsz_spin = QSpinBox()
    parent.audit_imgsz_spin.setRange(64, 4096)
    parent.audit_imgsz_spin.setValue(640)
    parent.audit_imgsz_spin.setSingleStep(32)
    parent.audit_imgsz_spin.setToolTip(
        "Размер входного изображения для модели при аудите (imgsz, по умолчанию 640)."
    )

    params_row.addWidget(QLabel("Порог IoU:"))
    params_row.addWidget(parent.audit_iou_spin)
    params_row.addWidget(QLabel("Confidence:"))
    params_row.addWidget(parent.audit_conf_spin)
    params_row.addWidget(QLabel("imgsz:"))
    params_row.addWidget(parent.audit_imgsz_spin)
    params_row.addStretch()
    ac.addLayout(params_row)

    parent.audit_summary_label = QLabel("Загрузите модель и запустите аудит.")
    parent.audit_summary_label.setWordWrap(True)
    ac.addWidget(parent.audit_summary_label)

    audit_filter_row = QHBoxLayout()
    audit_filter_row.addWidget(QLabel("Показывать:"))
    parent.audit_filter_combo = QComboBox()
    parent.audit_filter_combo.addItem("Только подозрительные", "suspicious")
    parent.audit_filter_combo.addItem("Все аудированные пары", "all")
    parent.audit_filter_combo.setToolTip(
        "«Только подозрительные» — пары с низким IoU, "
        "пропущенными объектами (GT без Pred) "
        "или ложными срабатываниями (Pred без GT).\n"
        "«Все аудированные» — каждая пара, для которой "
        "удалось сравнить GT и Pred."
    )
    audit_filter_row.addWidget(parent.audit_filter_combo)
    audit_filter_row.addStretch()
    ac.addLayout(audit_filter_row)

    parent.audit_table = QTableWidget()
    parent.audit_table.setColumnCount(4)
    parent.audit_table.setHorizontalHeaderLabels(
        ["Изображение", "GT", "Pred", "Ср. IoU"]
    )
    parent.audit_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.audit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.audit_table.setMinimumHeight(200)
    parent.audit_table.setToolTip(
        "Двойной клик по строке — открыть двойное окно сравнения."
    )
    ac.addWidget(parent.audit_table, 1)

    val_sep = QLabel("<hr><b>Валидация модели на тестовом датасете (YOLO val)</b>")
    ac.addWidget(val_sep)

    val_row = QHBoxLayout()
    parent.val_data_edit = QLineEdit()
    parent.val_data_edit.setPlaceholderText("Путь к data.yaml или к папке датасета")
    parent.btn_browse_val_data = QPushButton("Обзор…")
    val_row.addWidget(QLabel("Test data:"))
    val_row.addWidget(parent.val_data_edit, 1)
    val_row.addWidget(parent.btn_browse_val_data)
    ac.addLayout(val_row)

    val_params_row = QHBoxLayout()
    parent.val_split_combo = QComboBox()
    parent.val_split_combo.addItems(["val", "test", "train"])
    parent.val_split_combo.setToolTip("Сплит из YAML для валидации.")

    parent.val_imgsz_spin = QSpinBox()
    parent.val_imgsz_spin.setRange(64, 4096)
    parent.val_imgsz_spin.setValue(640)
    parent.val_imgsz_spin.setSingleStep(32)
    parent.val_imgsz_spin.setToolTip("Размер изображений для валидации.")

    parent.val_conf_spin = QDoubleSpinBox()
    parent.val_conf_spin.setRange(0.001, 1.0)
    parent.val_conf_spin.setValue(0.001)
    parent.val_conf_spin.setDecimals(3)
    parent.val_conf_spin.setSingleStep(0.01)
    parent.val_conf_spin.setToolTip("Confidence для валидации. Обычно 0.001.")

    parent.val_iou_spin = QDoubleSpinBox()
    parent.val_iou_spin.setRange(0.1, 0.99)
    parent.val_iou_spin.setValue(0.6)
    parent.val_iou_spin.setDecimals(2)
    parent.val_iou_spin.setSingleStep(0.05)
    parent.val_iou_spin.setToolTip("IoU-порог для NMS при валидации.")

    val_params_row.addWidget(QLabel("Split:"))
    val_params_row.addWidget(parent.val_split_combo)
    val_params_row.addWidget(QLabel("imgsz:"))
    val_params_row.addWidget(parent.val_imgsz_spin)
    val_params_row.addWidget(QLabel("conf:"))
    val_params_row.addWidget(parent.val_conf_spin)
    val_params_row.addWidget(QLabel("iou:"))
    val_params_row.addWidget(parent.val_iou_spin)
    val_params_row.addStretch()
    ac.addLayout(val_params_row)

    parent.val_summary_label = QLabel(
        "Укажите путь к тестовому датасету и запустите валидацию."
    )
    parent.val_summary_label.setWordWrap(True)
    ac.addWidget(parent.val_summary_label)

    parent.val_table = QTableWidget()
    parent.val_table.setColumnCount(6)
    parent.val_table.setHorizontalHeaderLabels(
        ["Класс", "Precision", "Recall", "F1", "mAP50", "mAP50-95"]
    )
    parent.val_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.val_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    parent.val_table.setMinimumHeight(150)
    ac.addWidget(parent.val_table, 1)

    val_btn_row = QHBoxLayout()
    parent.btn_run_validation = QPushButton("▶ Запустить валидацию")
    parent.btn_run_validation.setStyleSheet(
        "background-color: #00897B; color: white; font-weight: bold;"
    )
    parent.btn_stop_validation = QPushButton("⏹ Остановить валидацию")
    parent.btn_stop_validation.setEnabled(False)
    val_btn_row.addWidget(parent.btn_run_validation)
    val_btn_row.addWidget(parent.btn_stop_validation)
    val_btn_row.addStretch()
    ac.addLayout(val_btn_row)

    parent.btn_run_audit = QPushButton("Запустить аудит модели")
    parent.btn_stop_audit = QPushButton("⏹ Остановить")
    parent.btn_stop_audit.setEnabled(False)

    _make_page(
        parent.toolbox,
        "5. Аудит модели (Loss Inspection) и валидация YOLO",
        "**Loss Inspection:** модель прогоняется по каждой GT-разметке. "
        "**Двойной клик по строке** — двойное окно сравнения (GT ↔ Pred). "
        "**Валидация (YOLO val):** укажите путь к `data.yaml` тестового "
        "датасета (или папке), выберите сплит, запустите. "
        "Метрики P/R/F1/mAP50/mAP50-95 по каждому классу выдаёт "
        "Ultralytics. **Работает по filtered_pairs.**",
        audit_content,
        buttons=[parent.btn_run_audit, parent.btn_stop_audit],
        min_height=700,
        content_stretch=1,
    )

    # ============================================================
