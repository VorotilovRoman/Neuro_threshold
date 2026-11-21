"""Shared page framing and scaled training-result image label."""

import re
from html import escape

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class TrainingImageLabel(QLabel):
    """QLabel, который масштабирует QPixmap под свой размер
    с сохранением пропорций и перерисовывает при resize."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet("background-color: white;")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._pixmap = None
        self.setText("(нет данных)")

    def set_image(self, pixmap):
        self._pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.setText("(нет данных)")
            self.setPixmap(QPixmap())
        else:
            self.setText("")
            self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def sizeHint(self):
        return QSize(800, 400)


def _make_page(
    toolbox, title, description, content, buttons, min_height=240, content_stretch=1
):
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(6, 6, 6, 6)

    if description:
        description = re.sub(
            r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(description)
        ).replace("\n", "<br>")
        d = QLabel(description)
        d.setTextFormat(Qt.RichText)
        d.setWordWrap(True)
        d.setStyleSheet("color: #555; font-size: 13px;")
        d.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(d)

    if content is not None:
        content.setMinimumHeight(min_height)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(content, content_stretch)

    if buttons:
        br = QHBoxLayout()
        for b in buttons:
            b.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            br.addWidget(b)
        br.addStretch()
        layout.addLayout(br)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    scroll.setFrameShape(QFrame.NoFrame)
    toolbox.addItem(scroll, title)
    return scroll


def _mk_min_spin(rng, suffix="", special="Нет", tooltip=""):
    w = QSpinBox()
    w.setRange(*rng)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    w.setMinimumWidth(80)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_max_spin(rng, suffix="", special="Нет", tooltip=""):
    w = QSpinBox()
    w.setRange(*rng)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    w.setMinimumWidth(80)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_min_dspin(rng, decimals=2, suffix="", special="Нет", tooltip=""):
    w = QDoubleSpinBox()
    w.setRange(*rng)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    w.setMinimumWidth(80)
    if tooltip:
        w.setToolTip(tooltip)
    return w


def _mk_max_dspin(rng, decimals=2, suffix="", special="Нет", tooltip=""):
    w = QDoubleSpinBox()
    w.setRange(*rng)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    w.setSpecialValueText(special)
    w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    w.setMinimumWidth(80)
    if tooltip:
        w.setToolTip(tooltip)
    return w
