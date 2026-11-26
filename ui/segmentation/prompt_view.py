"""Shared vector prompts for classical interactive methods and SAM models."""

from runtime_torch import prepare_windows_torch

prepare_windows_torch()

import math
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainterPath, QPen, QPolygonF
from ui.image_view import SmartGraphicsViewNew
from utils_new.annotations.adapters import from_legacy
from utils_new.annotations.model import ImageSize


class PromptInputView(SmartGraphicsViewNew):
    prompts_changed = pyqtSignal()
    tool_changed = pyqtSignal()
    box_drawn = pyqtSignal(object)
    stroke_added = pyqtSignal(float, float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.prompt_tool = None
        self.prompt_points, self.prompt_labels, self.prompt_strokes = [], [], []
        self.prompt_box = self._prompt_start = None
        self.prompt_preview = []
        self.full_image = True
        self.brush_radius = 0
        self.stroke_serial = 0
        self._stroke = self._stroke_button = None
        self._box_before = None

    def set_prompt_tool(self, tool):
        if self._prompt_start is not None and self._box_before is not None:
            self.prompt_box, self.full_image = self._box_before
        self._box_before = None
        self.prompt_tool = tool
        self._prompt_start = self._stroke = self._stroke_button = None
        self.set_drawing_tool(None)
        if tool:
            self.set_edit_mode(False)
            self.setDragMode(self.NoDrag)
            self.setCursor(Qt.CrossCursor)
        self.tool_changed.emit()
        self.viewport().update()

    def clear_prompts(self):
        self.prompt_points, self.prompt_labels, self.prompt_strokes = [], [], []
        self.prompt_box = self._prompt_start = None
        self._stroke = self._stroke_button = None
        self._box_before = None
        self.full_image = True
        self.prompts_changed.emit()
        self.viewport().update()

    def effective_box(self):
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        return (0, 0, self.img_width, self.img_height) if self.full_image else self.prompt_box

    def set_prompt_box(self, box):
        x1, y1, x2, y2 = box
        self.prompt_box = (max(0, min(x1, self.img_width)), max(0, min(y1, self.img_height)),
                           max(0, min(x2, self.img_width)), max(0, min(y2, self.img_height)))
        self._prompt_start = self._box_before = None
        self.full_image = False
        self.prompts_changed.emit()
        self.viewport().update()

    def _point(self, event, *, dragging=False, box=False):
        point = self.mapToScene(event.pos())
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        if not dragging and not (0 <= point.x() < self.img_width and 0 <= point.y() < self.img_height):
            return None
        # Points address pixel centers; ROI's far edge can reach width/height.
        return (max(0, min(point.x(), self.img_width - (0 if box else 1))),
                max(0, min(point.y(), self.img_height - (0 if box else 1))))

    def _add_point(self, point, *, final=False):
        stroke = self._stroke
        if stroke is None or (not final and stroke['points'] and point == stroke['points'][-1]):
            return
        previous = stroke['points'][-1] if stroke['points'] else point
        if not stroke['points'] or point != stroke['points'][-1]:
            stroke['points'].append(point)
        # Sample sparse model prompts along a continuous visual/seed stroke.
        length = math.dist(previous, point)
        count = max(1, math.ceil(length / 8))
        for step in range(1, count + 1):
            ratio = step / count
            sample = (previous[0] + ratio * (point[0] - previous[0]),
                      previous[1] + ratio * (point[1] - previous[1]))
            last = stroke.get('sample_last')
            if last is not None and math.dist(last, sample) < (1 if final else 8):
                continue
            keep = [i for i, old in enumerate(self.prompt_points)
                    if self.prompt_labels[i] == stroke['label'] or math.dist(old, sample) > max(1, self.brush_radius)]
            self.prompt_points = [self.prompt_points[i] for i in keep]
            self.prompt_labels = [self.prompt_labels[i] for i in keep]
            if self.prompt_points and self.prompt_labels[-1] == stroke['label'] and math.dist(self.prompt_points[-1], sample) < 1:
                continue
            self.prompt_points.append(sample)
            self.prompt_labels.append(stroke['label'])
            stroke['sample_last'] = sample
        self.stroke_added.emit(*point, stroke['label'], self.stroke_serial)
        self.prompts_changed.emit()
        self.viewport().update()

    def mousePressEvent(self, event):
        if self.prompt_tool in ('positive', 'negative') and event.button() in (Qt.LeftButton, Qt.RightButton):
            point = self._point(event)
            if point is not None:
                self.stroke_serial += 1
                self._stroke_button = event.button()
                self._stroke = {'label': int(event.button() != Qt.RightButton and self.prompt_tool == 'positive'),
                                'points': [], 'radius': self.brush_radius}
                self.prompt_strokes.append(self._stroke)
                self._add_point(point)
            event.accept()
            return
        if self.prompt_tool == 'box' and event.button() == Qt.LeftButton:
            self._prompt_start = self._point(event, box=True)
            if self._prompt_start is not None:
                self._box_before = (self.prompt_box, self.full_image)
                self.prompt_box = (*self._prompt_start, *self._prompt_start)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._stroke is not None:
            point = self._point(event, dragging=True)
            if point is not None:
                self._add_point(point)
            event.accept()
            return
        if self.prompt_tool == 'box' and self._prompt_start is not None:
            point = self._point(event, dragging=True, box=True)
            if point is not None:
                x, y = self._prompt_start
                self.prompt_box = (min(x, point[0]), min(y, point[1]), max(x, point[0]), max(y, point[1]))
                self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._stroke is not None and event.button() == self._stroke_button:
            point = self._point(event, dragging=True)
            if point is not None:
                self._add_point(point, final=True)
            self._stroke = self._stroke_button = None
            event.accept()
            return
        if self.prompt_tool == 'box' and event.button() == Qt.LeftButton and self._prompt_start is not None:
            self.mouseMoveEvent(event)
            self._prompt_start = None
            previous, self._box_before = self._box_before, None
            if self.prompt_box is not None and self.prompt_box[2] > self.prompt_box[0] and self.prompt_box[3] > self.prompt_box[1]:
                self.full_image = False
                self.box_drawn.emit(self.prompt_box)
                self.prompts_changed.emit()
            else:
                if previous is not None:
                    self.prompt_box, self.full_image = previous
            self.viewport().update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        if self.prompt_tool in ('positive', 'negative'):
            event.accept()
            return
        super().contextMenuEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.prompt_tool is not None:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.prompt_tool is not None:
            self.set_prompt_tool(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not hasattr(self, 'prompt_preview'):
            return
        painter.save()
        try:
            pen = QPen(QColor('#00c8ff'), 2, Qt.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 200, 255, 50))
            if self.img_width > 0 and self.img_height > 0:
                size = ImageSize(self.img_width, self.img_height)
                for ann in self.prompt_preview:
                    polygon = from_legacy(ann, size).points_px
                    painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in polygon]))
            painter.setBrush(Qt.NoBrush)
            box = self.prompt_box if self._prompt_start is not None else self.effective_box()
            if box is not None:
                x1, y1, x2, y2 = box
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            zoom = max(abs(self.transform().m11()), 1e-6)
            for stroke in self.prompt_strokes:
                color = QColor('#26d84b' if stroke['label'] else '#ff4858')
                points, radius = stroke['points'], stroke['radius']
                if not points:
                    continue
                pen = QPen(color, 2 * radius + 1 if radius else 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                pen.setCosmetic(not bool(radius))
                painter.setPen(pen)
                painter.setBrush(color)
                if len(points) == 1:
                    painter.drawEllipse(QPointF(*points[0]), radius or 5 / zoom, radius or 5 / zoom)
                else:
                    painter.setBrush(Qt.NoBrush)
                    path = QPainterPath(QPointF(*points[0]))
                    for point in points[1:]:
                        path.lineTo(QPointF(*point))
                    painter.drawPath(path)
        finally:
            painter.restore()
