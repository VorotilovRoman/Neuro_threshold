"""Shared threshold/gradient ranges and pixel morphology configuration."""

from copy import deepcopy
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from utils_new.autoreseach_utils.search_space import (
    specs_for,
    search_kind,
    POST_SPECS,
    combination_count,
    default_settings,
)
from utils_new.morphology import SUPPORTED_OPERATIONS
from ui.morphology_ui import MorphologyWidget


class AutoSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)

        self.resize(720, 620)
        state = deepcopy(current_settings or default_settings())
        self.kind = search_kind(state)
        self.specs = specs_for(self.kind)
        self.setWindowTitle(
            "Диапазоны "
            + ("порогового" if self.kind == "threshold" else "градиентного")
            + " автоподбора"
        )
        self._ranges = {}
        self._enabled = {}
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        for method, specs in self.specs.items():
            page = QWidget()
            form = QFormLayout(page)
            check = QCheckBox("Включить метод")
            check.setChecked(state["methods"][method]["enabled"])
            form.addRow(check)
            self._enabled[method] = check
            for key, spec in specs.items():
                row = QHBoxLayout()
                widgets = []
                for caption, value in zip(
                    ("От", "До", "Шаг"), state["methods"][method]["ranges"][key]
                ):
                    box = QSpinBox() if spec[5] else QDoubleSpinBox()
                    if not spec[5]:
                        box.setDecimals(2 if len(spec) > 7 and spec[7] == 0.01 else 6)
                    box.setRange(
                        spec[3]
                        if caption != "Шаг"
                        else (
                            1 if spec[5] else (spec[7] if len(spec) > 7 else 0.000001)
                        ),
                        spec[4] if caption != "Шаг" else 10000,
                    )
                    box.setValue(value)
                    row.addWidget(QLabel(caption))
                    row.addWidget(box)
                    widgets.append(box)
                form.addRow(key, row)
                self._ranges[(method, key)] = widgets
            tabs.addTab(page, method)
        self._post_ranges = {}
        if self.kind == "gradient":
            page = QWidget()
            form = QFormLayout(page)
            for key, spec in POST_SPECS.items():
                row = QHBoxLayout()
                widgets = []
                for caption, value in zip(
                    ("От", "До", "Шаг"), state["post_ranges"][key]
                ):
                    box = QSpinBox() if spec[5] else QDoubleSpinBox()
                    if not spec[5]:
                        box.setDecimals(2 if len(spec) > 7 and spec[7] == 0.01 else 6)
                    box.setRange(
                        spec[3]
                        if caption != "Шаг"
                        else (
                            1 if spec[5] else (spec[7] if len(spec) > 7 else 0.000001)
                        ),
                        spec[4] if caption != "Шаг" else 10000,
                    )
                    box.setValue(value)
                    row.addWidget(QLabel(caption))
                    row.addWidget(box)
                    widgets.append(box)
                form.addRow(
                    "Порог градиента" if key == "threshold" else "Мин. площадь, %", row
                )
                self._post_ranges[key] = widgets
            self.fill = QComboBox()
            self.fill.addItems(["Без заливки", "С заливкой", "Оба варианта"])
            fills = state["fill_contours"]
            self.fill.setCurrentIndex(2 if len(fills) == 2 else int(fills[0]))
            form.addRow("Заливка контуров", self.fill)
            tabs.addTab(page, "Бинаризация и объекты")
        page = QWidget()
        form = QFormLayout(page)
        self.morphology = MorphologyWidget(contour_available=False)
        self.morphology.set_settings(state["morphology"])
        form.addRow(self.morphology)
        for spin in self.morphology.findChildren(QSpinBox):
            spin.setEnabled(False)
            spin.setToolTip("Размер задаётся диапазоном ниже.")
        form.addRow(
            QLabel(
                "Форма и порядок — в панели выше. Размеры для перебора задаются ниже, в px."
            )
        )
        self._kernels = {}
        for op in SUPPORTED_OPERATIONS:
            row = QHBoxLayout()
            widgets = []
            for caption, value in zip(("От", "До", "Шаг"), state["kernel_ranges"][op]):
                box = QSpinBox()
                box.setRange(1 if caption == "Шаг" else 0, 4096)
                box.setValue(value)
                row.addWidget(QLabel(caption))
                row.addWidget(box)
                widgets.append(box)
            form.addRow(op + " (px)", row)
            self._kernels[op] = widgets
        self.invert = QComboBox()
        self.invert.addItems(["Без инверсии", "С инверсией", "Оба варианта"])
        self.invert.setCurrentIndex(
            2 if len(state["inversions"]) == 2 else int(state["inversions"][0])
        )
        form.addRow("Инверсия", self.invert)
        self.mode = QComboBox()
        for label, key in [("BBox", "detect"), ("OBB", "obb"), ("Segment", "segment")]:
            self.mode.addItem(label, key)
        self.mode.setCurrentIndex(self.mode.findData(state["annotation_mode"]))
        form.addRow("Разметка", self.mode)
        self.hull = QCheckBox("Выпуклая оболочка (OBB/Segment)")
        self.hull.setChecked(state["use_hull"])
        form.addRow(self.hull)
        tabs.addTab(page, "Морфология и результат")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        state = default_settings(self.kind)
        for method in self.specs:
            state["methods"][method] = {
                "enabled": self._enabled[method].isChecked(),
                "ranges": {
                    key: [box.value() for box in self._ranges[(method, key)]]
                    for key in self.specs[method]
                },
            }
        state["morphology"] = self.morphology.get_settings()
        state["kernel_ranges"] = {
            op: [box.value() for box in widgets]
            for op, widgets in self._kernels.items()
        }
        state["inversions"] = [[False], [True], [False, True]][
            self.invert.currentIndex()
        ]
        state["annotation_mode"] = self.mode.currentData()
        state["use_hull"] = self.hull.isChecked()
        if self.kind == "gradient":
            state["post_ranges"] = {
                key: [box.value() for box in widgets]
                for key, widgets in self._post_ranges.items()
            }
            state["fill_contours"] = [[False], [True], [False, True]][
                self.fill.currentIndex()
            ]
        combination_count(state)
        return state

    def accept(self):
        try:
            count = combination_count(self.get_settings())
            if not count:
                raise ValueError("Включите хотя бы один метод.")
        except (ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Диапазоны", str(exc))
            return
        super().accept()
