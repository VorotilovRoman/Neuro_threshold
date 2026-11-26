"""Prompt input and preview overlay shared by annotation and DL tabs."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPolygonF
from ui.image_view import SmartGraphicsViewNew
from utils_new.annotations.adapters import from_legacy
from utils_new.annotations.model import ImageSize


class SamPromptView(SmartGraphicsViewNew):
    prompts_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sam_tool = None
        self.sam_points, self.sam_labels = [], []
        self.sam_box = self._sam_start = None
        self.sam_preview = []

    def clear_sam_prompts(self):
        self.sam_points, self.sam_labels = [], []
        self.sam_box = self._sam_start = None
        self.prompts_changed.emit()
        self.viewport().update()

    def _sam_point(self, event, *, dragging=False):
        point = self.mapToScene(event.pos())
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        if not dragging and not (0 <= point.x() < self.img_width and 0 <= point.y() < self.img_height):
            return None
        return (max(0, min(point.x(), self.img_width)), max(0, min(point.y(), self.img_height)))

    def mousePressEvent(self, event):
        if self.sam_tool in ("positive", "negative") and event.button() in (Qt.LeftButton, Qt.RightButton):
            point = self._sam_point(event)
            if point is not None:
                self.sam_points.append(point)
                self.sam_labels.append(0 if event.button() == Qt.RightButton or self.sam_tool == "negative" else 1)
                self.prompts_changed.emit()
                self.viewport().update()
            event.accept()
            return
        if self.sam_tool == "box" and event.button() == Qt.LeftButton:
            self._sam_start = self._sam_point(event)
            if self._sam_start is not None:
                self.sam_box = (*self._sam_start, *self._sam_start)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.sam_tool == "box" and self._sam_start is not None:
            point = self._sam_point(event, dragging=True)
            if point is not None:
                x, y = self._sam_start
                self.sam_box = (min(x, point[0]), min(y, point[1]), max(x, point[0]), max(y, point[1]))
                self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.sam_tool == "box" and event.button() == Qt.LeftButton and self._sam_start is not None:
            self.mouseMoveEvent(event)
            self._sam_start = None
            if self.sam_box[2] <= self.sam_box[0] or self.sam_box[3] <= self.sam_box[1]:
                self.sam_box = None
            self.prompts_changed.emit()
            self.viewport().update()
            event.accept()
            return
        if self.sam_tool in ("positive", "negative") and event.button() in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        if self.sam_tool in ("positive", "negative"):
            event.accept()
            return
        super().contextMenuEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.sam_tool is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.sam_tool is not None:
            self.sam_tool = self._sam_start = None
            self.set_drawing_tool(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not hasattr(self, "sam_preview"):
            return
        painter.save()
        try:
            pen = QPen(QColor("#00c8ff"), 2, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 200, 255, 50))
            if self.img_width > 0 and self.img_height > 0:
                size = ImageSize(self.img_width, self.img_height)
                for ann in self.sam_preview:
                    polygon = from_legacy(ann, size).points_px
                    painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in polygon]))
            painter.setBrush(Qt.NoBrush)
            if self.sam_box is not None:
                x1, y1, x2, y2 = self.sam_box
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            radius = 5 / max(abs(self.transform().m11()), 1e-6)
            for (x, y), label in zip(self.sam_points, self.sam_labels):
                color = QColor("#26d84b" if label else "#ff4858")
                pen.setColor(color)
                painter.setPen(pen)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, y), radius, radius)
        finally:
            painter.restore()
