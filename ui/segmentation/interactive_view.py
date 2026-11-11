"""ROI and foreground/background input over the canonical image viewer.

The annotation editor's polygon/OBB gestures stay separate from segmentation
markers. Coordinates delivered to the tab are working-image pixels after zoom.
"""
from PyQt5.QtCore import Qt

from ui.image_view import SmartGraphicsViewNew


class SegmentationInputView(SmartGraphicsViewNew):
    def _input_point(self, event):
        point = self.mapToScene(event.pos())
        if self.img_width <= 0 or self.img_height <= 0:
            return None
        return (max(0,min(int(point.x()),self.img_width-1)),
                max(0,min(int(point.y()),self.img_height-1)))

    def mousePressEvent(self, event):
        if event.button()==Qt.LeftButton and self.drawing_mode and self.current_tool in ('rect','fg','bg'):
            point = self._input_point(event)
            if point is None:
                return
            self.drawing = True
            self.start_point = self.current_point = point
            if self.current_tool=='rect':
                self.temp_rect = (*point,*point)
            else:
                self._add_scribble(*point,temp=True)
            self.viewport().update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_tool in ('rect','fg','bg'):
            point = self._input_point(event)
            if point is not None:
                self.current_point = point
                if self.current_tool=='rect':
                    self.temp_rect = (*self.start_point,*point)
                else:
                    self._add_scribble(*point,temp=True)
                self.viewport().update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.LeftButton and self.drawing and self.current_tool in ('rect','fg','bg'):
            tool = self.current_tool
            point = self._input_point(event)
            start = self.start_point
            self.drawing = False
            self.temp_rect = self.temp_scribble = None
            if point is not None:
                if tool=='rect' and start is not None:
                    rect=(min(start[0],point[0]),min(start[1],point[1]),
                          max(start[0],point[0]),max(start[1],point[1]))
                    if self.on_rect_drawn_callback:
                        self.on_rect_drawn_callback(rect)
                else:
                    self._add_scribble(*point)
            self.viewport().update()
            event.accept()
            return
        super().mouseReleaseEvent(event)
