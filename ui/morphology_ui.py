# -*- coding: utf-8 -*-
"""Универсальная панель для utils_new.morphology.

Пример подключения:
    from ui.morphology_ui import MorphologyWidget
    self.morphology_widget = MorphologyWidget(contour_available=False)
    layout.addWidget(self.morphology_widget)
    self.morphology_widget.parameters_changed.connect(self.schedule_update)
    result = apply_morphology(image, self.morphology_widget.get_steps(),
                              shape=self.morphology_widget.kernel_shape())

Четыре операции всегда присутствуют. Размер 0/1 px = отключено,
>=2 px = реальный размер квадратного ядра. Кнопки ↑/↓ изменяют
порядок применения, который входит в сериализуемые настройки.
"""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from utils_new.morphology import SUPPORTED_OPERATIONS, SUPPORTED_SHAPES, MorphologyStep

_LABELS = {
    "close": "Закрытие",
    "open": "Открытие",
    "erode": "Эрозия",
    "dilate": "Расширение",
}


class MorphologyWidget(QGroupBox):
    """Переиспользуемая панель настройки морфологии (без обработки изображений).

    Состояние можно передать в JSON: ``get_settings()`` / ``set_settings()``.
    Специальных связей с вкладками и системой пресетов нет.
    ``contour_available=False`` скрывает управление областью — удобно для
    морфологии бинарной маски по всему снимку (Threshold Methods).
    """

    parameters_changed = pyqtSignal()

    def __init__(self, parent=None, *, contour_available=True,
                 contour_default=False, initial_order=None, title="Морфология"):
        super().__init__(title, parent)
        self._loading = True
        self._order = list(initial_order or SUPPORTED_OPERATIONS)
        if len(self._order) != len(SUPPORTED_OPERATIONS) or set(self._order) != set(SUPPORTED_OPERATIONS):
            raise ValueError("initial_order должен содержать close/open/erode/dilate по одному разу")
        self._rows = {}
        self._sizes = {}
        self._up = {}
        self._down = {}

        main = QVBoxLayout(self)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Форма ядра:"))
        self.shape_combo = QComboBox(self)
        self.shape_combo.addItems(list(SUPPORTED_SHAPES))
        shape_row.addWidget(self.shape_combo, 1)
        main.addLayout(shape_row)

        hint = QLabel("Размер ядра, px (0/1 — выкл.); ↑↓ — порядок операций", self)
        hint.setWordWrap(True)
        main.addWidget(hint)

        self.operations_layout = QVBoxLayout()
        self.operations_layout.setSpacing(3)
        for key in SUPPORTED_OPERATIONS:
            row = QWidget(self)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            label = QLabel(_LABELS[key], row)
            label.setMinimumWidth(80)
            layout.addWidget(label)
            size = QSpinBox(row)
            size.setRange(0, 4096)
            size.setSingleStep(1)
            size.setSuffix(" px")
            size.setValue(0)
            size.setToolTip("0/1 — операция выключена; 2..4096 — размер ядра в пикселях")
            layout.addWidget(size, 1)
            up = QToolButton(row)
            up.setText("↑")
            up.setToolTip("Выполнить раньше")
            down = QToolButton(row)
            down.setText("↓")
            down.setToolTip("Выполнить позже")
            layout.addWidget(up)
            layout.addWidget(down)
            self._rows[key] = row
            self._sizes[key] = size
            self._up[key] = up
            self._down[key] = down
            size.valueChanged.connect(self._notify)
            up.clicked.connect(lambda _checked=False, op=key: self._move(op, -1))
            down.clicked.connect(lambda _checked=False, op=key: self._move(op, 1))

        main.addLayout(self.operations_layout)
        self._reorder_widgets()

        self.contour_checkbox = QCheckBox("Только по контуру объекта", self)
        self.contour_checkbox.setChecked(bool(contour_default))
        self.contour_checkbox.setToolTip(
            "При включении применяется внутренняя полоса контура маски; "
            "при выключении — вся область объекта."
        )
        main.addWidget(self.contour_checkbox)
        width_row = QHBoxLayout()
        self.contour_width_label = QLabel("Ширина контура:", self)
        width_row.addWidget(self.contour_width_label)
        self.contour_width_spin = QSpinBox(self)
        self.contour_width_spin.setRange(0, 4096)
        self.contour_width_spin.setSpecialValueText("Авто")
        self.contour_width_spin.setSuffix(" px")
        self.contour_width_spin.setToolTip(
            "Авто — половина наибольшего активного ядра, "
            "либо явно заданная ширина в пикселях"
        )
        width_row.addWidget(self.contour_width_spin, 1)
        main.addLayout(width_row)
        self.set_contour_available(contour_available)
        self._update_width_enabled()

        self.shape_combo.currentIndexChanged.connect(self._notify)
        self.contour_checkbox.toggled.connect(self._on_contour_changed)
        self.contour_width_spin.valueChanged.connect(self._notify)
        self._loading = False

    def _notify(self, *_):
        if not self._loading:
            self.parameters_changed.emit()

    def _on_contour_changed(self, *_):
        self._update_width_enabled()
        self._notify()

    def _update_width_enabled(self):
        can_edit = self._contour_available and self.contour_checkbox.isChecked()
        self.contour_width_label.setEnabled(can_edit)
        self.contour_width_spin.setEnabled(can_edit)

    def set_contour_available(self, available: bool):
        """Если False, режим контура не используется и не показывается."""
        self._contour_available = bool(available)
        self.contour_checkbox.setVisible(self._contour_available)
        self.contour_width_label.setVisible(self._contour_available)
        self.contour_width_spin.setVisible(self._contour_available)
        self._update_width_enabled()

    def _reorder_widgets(self):
        for key in self._order:
            widget = self._rows[key]
            self.operations_layout.removeWidget(widget)
            self.operations_layout.addWidget(widget)
        for i, key in enumerate(self._order):
            self._up[key].setEnabled(i > 0)
            self._down[key].setEnabled(i < len(self._order) - 1)

    def _move(self, key: str, direction: int):
        index = self._order.index(key)
        other = index + direction
        if 0 <= other < len(self._order):
            self._order[index], self._order[other] = self._order[other], self._order[index]
            self._reorder_widgets()
            self._notify()

    def set_order(self, operations):
        ops = list(operations)
        if len(ops) != 4 or set(ops) != set(SUPPORTED_OPERATIONS):
            raise ValueError("Нужны ровно четыре уникальные операции")
        if ops != self._order:
            self._order = ops
            self._reorder_widgets()
            self._notify()

    def operation_order(self):
        return list(self._order)

    def kernel_shape(self):
        return self.shape_combo.currentText()

    def set_kernel_shape(self, shape):
        idx = self.shape_combo.findText(str(shape))
        if idx < 0:
            raise ValueError(f"Неизвестная форма ядра {shape!r}")
        self.shape_combo.setCurrentIndex(idx)

    def set_operation_size(self, operation: str, kernel_px: int):
        self._sizes[operation].setValue(int(kernel_px))

    def operation_size(self, operation: str) -> int:
        return self._sizes[operation].value()

    def has_active_operations(self) -> bool:
        return any(size.value() > 1 for size in self._sizes.values())

    def get_steps(self):
        return [MorphologyStep(key, self._sizes[key].value()) for key in self._order]

    def contour_only(self):
        return self._contour_available and self.contour_checkbox.isChecked()

    def contour_width_px(self):
        value = self.contour_width_spin.value()
        return value if value > 0 else None

    def get_settings(self):
        return {
            "version": 1,
            "units": "px",
            "shape": self.kernel_shape(),
            "operations": [{"operation": key, "kernel_px": self._sizes[key].value()}
                           for key in self._order],
            "contour_only": self.contour_only(),
            "contour_width_px": self.contour_width_px(),
        }

    def set_settings(self, state: dict):
        """Восстановить порядок, размеры, форму и параметры контура из JSON."""
        if not isinstance(state, dict):
            raise TypeError("morphology settings должны быть словарём")
        saved_loading = self._loading
        self._loading = True
        try:
            if "shape" in state:
                self.set_kernel_shape(state["shape"])
            if "operations" in state:
                operations = state["operations"]
                names = [step["operation"] for step in operations]
                self.set_order(names)
                for step in operations:
                    self.set_operation_size(step["operation"], int(step["kernel_px"]))
            if "contour_only" in state:
                self.contour_checkbox.setChecked(bool(state["contour_only"]))
            if "contour_width_px" in state:
                self.contour_width_spin.setValue(int(state["contour_width_px"] or 0))
            self._update_width_enabled()
        finally:
            self._loading = saved_loading
        self._notify()


    def set_legacy_yolo(self, erosion=1, dilation=1, shape="Rectangle", contour_only=True):
        """Старый YOLO preset: размеры уже заданы в пикселях."""
        self.set_settings({
            "shape": shape,
            "operations": [
                {"operation": "erode", "kernel_px": int(erosion)},
                {"operation": "dilate", "kernel_px": int(dilation)},
                {"operation": "close", "kernel_px": 0},
                {"operation": "open", "kernel_px": 0},
            ],
            "contour_only": bool(contour_only),
            "contour_width_px": None,
        })


__all__ = ["MorphologyWidget"]
