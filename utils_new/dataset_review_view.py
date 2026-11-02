# -*- coding: utf-8 -*-
"""Read-only Qt-представление датасета, использующее новое smart_view_new.

Не наследует методы редактирования в поведении мыши: геометрия не меняется
при щелчках/перетаскивании, но выделение, панорамирование и zoom доступны.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView
from utils_new.smart_view_new import SmartGraphicsViewNew


class DatasetReviewGraphicsView(SmartGraphicsViewNew):
    """Щелчок выделяет объект; перетаскивание в остальном панорамирует вид."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selection_enabled = True

    def mousePressEvent(self, event):
        # Режим просмотра обходит SmartGraphicsView.mousePressEvent;
        # отдельно возвращаем его стандартный reset zoom по средней кнопке.
        if event.button() == Qt.MiddleButton:
            self.reset_view()
            event.accept()
            return
        if self.selection_enabled and event.button() == Qt.LeftButton and self.img_width > 0 and self.img_height > 0:
            scene_pos = self.mapToScene(event.pos())
            if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                idx, _ = self._find_hit((scene_pos.x(), scene_pos.y()), self.selected_index)
                if idx >= 0:
                    self.set_selected_index(idx)
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
        # Вызов базового QGraphicsView, минуя обработчики рисования/редактирования.
        QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseDoubleClickEvent(self, event):
        QGraphicsView.mouseDoubleClickEvent(self, event)

    def contextMenuEvent(self, event):
        # Меню просмотра остаётся доступно через contextMenuEvent новой утилиты.
        super().contextMenuEvent(event)
