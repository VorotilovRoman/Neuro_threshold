# -*- coding: utf-8 -*-
"""Read-only Qt-представление датасета, использующее новое smart_view_new.

Не наследует методы редактирования в поведении мыши: геометрия не меняется
при щелчках/перетаскивании, но выделение, панорамирование и zoom доступны.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QGraphicsView
from utils_new.smart_view_new import SmartGraphicsViewNew


class DatasetReviewGraphicsView(SmartGraphicsViewNew):
    """Щелчок выделяет объект; перетаскивание ЛКМ в любой точке панорамирует вид."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_enabled = True
        # Read-only means no annotation editing; the entire viewport can
        # initiate a ScrollHandDrag, including the interior of an annotation.
        self.drawing_mode = False
        self.edit_mode = False
        self._setup_pan_mode()
        self._read_only_press_pos = None
        self._read_only_press_hit = -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            # Match the dataset editor's wheel-click reset, without activating
            # any drawing or edit action.
            self._read_only_press_pos = None
            self._read_only_press_hit = -1
            self.reset_view()
            event.accept()
            return

        if event.button() == Qt.RightButton and self.on_annotation_right_clicked_callback:
            scene_pos = self.mapToScene(event.pos())
            if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                idx, _ = self._find_hit((scene_pos.x(), scene_pos.y()), self.selected_index)
                if idx >= 0:
                    self.on_annotation_right_clicked_callback(idx, event.globalPos())
                    event.accept()
                    return

        if event.button() == Qt.LeftButton:
            self._read_only_press_pos = event.pos()
            self._read_only_press_hit = -1
            if self.selection_enabled and self.img_width > 0 and self.img_height > 0:
                scene_pos = self.mapToScene(event.pos())
                if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                    idx, _ = self._find_hit((scene_pos.x(), scene_pos.y()), self.selected_index)
                    self._read_only_press_hit = idx

        # Do not consume a hit on an annotation. QGraphicsView starts
        # ScrollHandDrag here for BOTH annotated and unannotated pixels.
        QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        # Complete the Qt hand-drag before delivering a click selection.
        QGraphicsView.mouseReleaseEvent(self, event)
        if event.button() != Qt.LeftButton:
            return
        press_pos = self._read_only_press_pos
        hit = self._read_only_press_hit
        self._read_only_press_pos = None
        self._read_only_press_hit = -1
        if press_pos is None or hit < 0 or not self.selection_enabled:
            return
        # Selection occurs only on a click; a mouse drag MUST pan instead.
        if (event.pos() - press_pos).manhattanLength() <= QApplication.startDragDistance():
            self.set_selected_index(hit)

    def mouseDoubleClickEvent(self, event):
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def contextMenuEvent(self, event):
        # Меню просмотра остаётся доступно через contextMenuEvent новой утилиты.
        super().contextMenuEvent(event)
