from import_libs_internal import *

ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics.utils.ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass

class SmartGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self._current_pixmap = None
        self._last_pixmap_size = None
        self._save_basename = "image"

        # Режимы
        self.drawing_mode = False
        self.current_tool = None
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None

        self.edit_mode = False
        self.edit_handle = None
        self.edit_start_norm = None
        self.edit_original_ann = None
        self.edit_original_points = None
        self.edit_poly_idx = -1
        self.edit_obb_center = None
        self.edit_obb_w = None
        self.edit_obb_h = None
        self.edit_obb_angle = None
        self.edit_start_angle = None

        self.img_width = 0
        self.img_height = 0
        self.annotations = []
        self.selected_index = -1

        self.cursor_scene_pos = None

        # Коллбэки
        self.on_rect_drawn_callback = None
        self.on_scribble_added_callback = None
        self.on_display_update_callback = None
        self.on_reset_tool_callback = None
        self.on_annotation_modified_callback = None
        self.on_selection_changed_callback = None
        self.on_log_callback = None

        self._suppress_context_menu = False

        self._setup_pan_mode()

    def _setup_pan_mode(self):
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setCursor(QCursor(Qt.OpenHandCursor))

    def _setup_drawing_mode(self):
        self.setDragMode(QGraphicsView.NoDrag)
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None

    def _setup_edit_mode(self):
        self.setDragMode(QGraphicsView.NoDrag)
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None

    def set_suggested_save_name(self, name):
        self._save_basename = name

    def set_drawing_tool(self, tool):
        self.current_tool = tool
        if tool is not None:
            self.edit_mode = False
            self.drawing_mode = True
            self._setup_drawing_mode()
            self.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.drawing_mode = False
            self.cursor_scene_pos = None
            if not self.edit_mode:
                self._setup_pan_mode()
            self.viewport().update()
        self.viewport().update()

    def set_edit_mode(self, enabled):
        if enabled == self.edit_mode:
            return
        self.edit_mode = enabled
        if enabled:
            self.drawing_mode = False
            self.current_tool = None
            self._setup_edit_mode()
            self.setCursor(QCursor(Qt.ArrowCursor))
        else:
            self.edit_handle = None
            self.edit_start_norm = None
            self.edit_original_ann = None
            self.edit_original_points = None
            self.edit_poly_idx = -1
            self.edit_obb_center = None
            self.edit_obb_w = None
            self.edit_obb_h = None
            self.edit_obb_angle = None
            self.edit_start_angle = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
            self._setup_pan_mode()
        if self._current_pixmap:
            self.viewport().update()

    def set_annotations(self, annotations, img_width, img_height):
        self.annotations = annotations if annotations is not None else []
        self.img_width = img_width
        self.img_height = img_height
        if self.img_width > 0 and self.img_height > 0:
            for idx, ann in enumerate(self.annotations):
                if ann[0] == 'obb':
                    self._get_points_pixel(ann)

    def set_selected_index(self, idx):
        if idx == self.selected_index:
            return
        self.selected_index = idx
        if self.on_selection_changed_callback:
            self.on_selection_changed_callback(idx)

        if idx != -1 and idx < len(self.annotations) and self.annotations[idx][0] == 'obb':
            pts_pixel = self._get_points_pixel(self.annotations[idx])
            cx, cy, w, h, angle = self._obb_to_params(pts_pixel)
            self.edit_obb_center = (cx, cy)
            self.edit_obb_w = w
            self.edit_obb_h = h
            self.edit_obb_angle = angle
        else:
            self.edit_obb_center = None
            self.edit_obb_w = None
            self.edit_obb_h = None
            self.edit_obb_angle = None

        self.viewport().update()

    def set_callbacks(self, on_rect_drawn=None, on_scribble_added=None,
                      on_display_update=None, on_reset_tool=None,
                      on_annotation_modified=None, on_selection_changed=None,
                      on_log=None):
        self.on_rect_drawn_callback = on_rect_drawn
        self.on_scribble_added_callback = on_scribble_added
        self.on_display_update_callback = on_display_update
        self.on_reset_tool_callback = on_reset_tool
        self.on_annotation_modified_callback = on_annotation_modified
        self.on_selection_changed_callback = on_selection_changed
        self.on_log_callback = on_log

    def set_image_data(self, width, height, fg_scribbles=None, bg_scribbles=None):
        self.img_width = width
        self.img_height = height

    def set_pixmap(self, pixmap, preserve_view=True):
        if pixmap.isNull():
            self.pixmap_item.setPixmap(pixmap)
            self._last_pixmap_size = None
            self._current_pixmap = None
            return
        self._current_pixmap = pixmap
        new_size = pixmap.size()
        self.pixmap_item.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        if preserve_view and self._last_pixmap_size == new_size:
            return
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._last_pixmap_size = new_size

    def reset_view(self):
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def save_current_image(self):
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        suggested = f"{self._save_basename}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", suggested, "PNG Image (*.png)"
        )
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            self._current_pixmap.save(file_path, "PNG")

    # ----- Вспомогательные методы -----
    def _get_points_pixel(self, ann):
        if self.img_width == 0 or self.img_height == 0:
            return []
        typ = ann[0]
        try:
            if typ == 'detect':
                _, _, cx, cy, w, h = ann
                x1 = int((cx - w/2) * self.img_width)
                y1 = int((cy - h/2) * self.img_height)
                x2 = int((cx + w/2) * self.img_width)
                y2 = int((cy + h/2) * self.img_height)
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif typ in ('obb', 'segment'):
                points = ann[2]
                pts = []
                for i in range(0, len(points), 2):
                    px = int(points[i] * self.img_width)
                    py = int(points[i+1] * self.img_height)
                    pts.append((px, py))
                return pts
        except Exception:
            pass
        return []

    def point_in_polygon(self, pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        if n < 3:
            return False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # ---------- OBB геометрия ----------
    def _obb_to_params(self, pts_pixel):
        if len(pts_pixel) != 4:
            return (0, 0, 1, 1, 0)
        if ULTRALYTICS_AVAILABLE:
            pts_flat = [coord for pt in pts_pixel for coord in pt]
            boxes = torch.tensor([pts_flat], dtype=torch.float32)
            xywhr = xyxyxyxy2xywhr(boxes)[0]
            cx, cy, w, h, angle = xywhr.tolist()
            return (cx, cy, w, h, angle)
        else:
            return self._obb_to_params_fallback(pts_pixel)

    def _obb_to_points(self, cx, cy, w, h, angle):
        if ULTRALYTICS_AVAILABLE:
            xywhr = torch.tensor([[cx, cy, w, h, angle]], dtype=torch.float32)
            boxes = xywhr2xyxyxyxy(xywhr)
            pts_flat = boxes[0].reshape(-1).tolist()
            if len(pts_flat) >= 8:
                pts_pixel = [(pts_flat[i], pts_flat[i + 1]) for i in range(0, 8, 2)]
                return pts_pixel
        return self._obb_to_points_fallback(cx, cy, w, h, angle)

    def _obb_to_params_fallback(self, pts_pixel):
        cx = sum(p[0] for p in pts_pixel) / 4.0
        cy = sum(p[1] for p in pts_pixel) / 4.0
        w1 = math.hypot(pts_pixel[1][0] - pts_pixel[0][0], pts_pixel[1][1] - pts_pixel[0][1])
        w2 = math.hypot(pts_pixel[2][0] - pts_pixel[3][0], pts_pixel[2][1] - pts_pixel[3][1])
        w = (w1 + w2) / 2
        h1 = math.hypot(pts_pixel[2][0] - pts_pixel[1][0], pts_pixel[2][1] - pts_pixel[1][1])
        h2 = math.hypot(pts_pixel[0][0] - pts_pixel[3][0], pts_pixel[0][1] - pts_pixel[3][1])
        h = (h1 + h2) / 2
        angle = math.atan2(pts_pixel[1][1] - pts_pixel[0][1], pts_pixel[1][0] - pts_pixel[0][0])
        return (cx, cy, w, h, angle)

    def _obb_to_points_fallback(self, cx, cy, w, h, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = w / 2 * cos_a
        dy = w / 2 * sin_a
        dx2 = -h / 2 * sin_a
        dy2 = h / 2 * cos_a
        p0 = (cx + dx + dx2, cy + dy + dy2)
        p1 = (cx - dx + dx2, cy - dy + dy2)
        p2 = (cx - dx - dx2, cy - dy - dy2)
        p3 = (cx + dx - dx2, cy + dy - dy2)
        return [p0, p1, p2, p3]

    def _get_obb_handles(self, pts_pixel):
        if len(pts_pixel) != 4:
            return {}
        cx, cy, w, h, angle = self._obb_to_params(pts_pixel)
        corners = self._obb_to_points(cx, cy, w, h, angle)
        mid01 = ((corners[0][0]+corners[1][0])/2, (corners[0][1]+corners[1][1])/2)
        mid12 = ((corners[1][0]+corners[2][0])/2, (corners[1][1]+corners[2][1])/2)
        mid23 = ((corners[2][0]+corners[3][0])/2, (corners[2][1]+corners[3][1])/2)
        mid30 = ((corners[3][0]+corners[0][0])/2, (corners[3][1]+corners[0][1])/2)
        return {
            'corner0': corners[0], 'corner1': corners[1], 'corner2': corners[2], 'corner3': corners[3],
            'edge0': mid01, 'edge1': mid12, 'edge2': mid23, 'edge3': mid30
        }

    def _hit_test_obb_handles(self, pts_pixel, pos_pixel):
        handles = self._get_obb_handles(pts_pixel)
        for name, pt in handles.items():
            if abs(pt[0] - pos_pixel[0]) <= 8 and abs(pt[1] - pos_pixel[1]) <= 8:
                return name
        if self.point_in_polygon(pos_pixel, pts_pixel):
            return 'move'
        return None

    def _get_obb_rotation_sector_for_annotation(self, ann, pts_pixel, idx):
        if pts_pixel is None:
            if ann is None:
                return None
            pts_pixel = self._get_points_pixel(ann)
        if len(pts_pixel) != 4:
            return None
        cx, cy, w, h, angle = self._obb_to_params(pts_pixel)
        radius = max(w, h) * 0.25
        start_angle = angle - math.radians(45)
        span_angle = math.radians(90)
        return (cx, cy, radius, start_angle, span_angle)

    def _hit_test_obb_rotation_sector(self, ann, pts_pixel, idx, pos_pixel):
        sector = self._get_obb_rotation_sector_for_annotation(ann, pts_pixel, idx)
        if sector is None:
            return False
        cx, cy, radius, start_angle_rad, span_angle_rad = sector
        dx = pos_pixel[0] - cx
        dy = pos_pixel[1] - cy
        dist = math.hypot(dx, dy)
        if dist > radius:
            return False
        point_angle = math.atan2(-dy, dx)
        if point_angle < 0:
            point_angle += 2 * math.pi
        start = start_angle_rad % (2 * math.pi)
        end = (start_angle_rad + span_angle_rad) % (2 * math.pi)
        if start < end:
            inside = start <= point_angle <= end
        else:
            inside = point_angle >= start or point_angle <= end
        return inside

    def _hit_test_point(self, point, pos_pixel):
        x, y = point
        px, py = pos_pixel
        return abs(x - px) <= 8 and abs(y - py) <= 8

    def _hit_test_detect(self, ann, pos_pixel):
        points = self._get_points_pixel(ann)
        for i, pt in enumerate(points):
            if self._hit_test_point(pt, pos_pixel):
                return ('point', i)
        if len(points) == 4:
            x1, y1 = points[0]
            x2, y2 = points[2]
            if x1 <= pos_pixel[0] <= x2 and y1 <= pos_pixel[1] <= y2:
                return ('move', -1)
        return None

    def _hit_test_segment(self, ann, pos_pixel):
        points = self._get_points_pixel(ann)
        for i, pt in enumerate(points):
            if self._hit_test_point(pt, pos_pixel):
                return ('point', i)
        if len(points) >= 3 and self.point_in_polygon(pos_pixel, points):
            return ('move', -1)
        return None

    # ----- События мыши -----
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.reset_view()
            event.accept()
            return

        if event.button() == Qt.RightButton:
            if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                self._suppress_context_menu = False
                event.ignore()
                return
            if self.drawing_mode or self.edit_mode:
                if self.on_reset_tool_callback:
                    self.on_reset_tool_callback()
                self.set_edit_mode(False)
                self.set_selected_index(-1)
                self._suppress_context_menu = True
                event.accept()
                return
            self._suppress_context_menu = False
            event.ignore()
            return

        if event.button() == Qt.LeftButton:
            if event.modifiers() == Qt.ControlModifier:
                scene_pos = self.mapToScene(event.pos())
                if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                    pos_pixel = (scene_pos.x(), scene_pos.y())
                    for i, ann in enumerate(self.annotations):
                        typ = ann[0]
                        hit = None
                        if typ == 'detect':
                            hit = self._hit_test_detect(ann, pos_pixel)
                        elif typ == 'obb':
                            pts_pix = self._get_points_pixel(ann)
                            hit = self._hit_test_obb_handles(pts_pix, pos_pixel)
                        elif typ == 'segment':
                            hit = self._hit_test_segment(ann, pos_pixel)
                        if hit is not None:
                            self.set_selected_index(i)
                            self.set_edit_mode(True)
                            event.accept()
                            self.viewport().update()
                            return
                event.accept()
                return

            if self.edit_mode:
                try:
                    scene_pos = self.mapToScene(event.pos())
                    if not (0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height):
                        super().mousePressEvent(event)
                        return
                    pos_pixel = (scene_pos.x(), scene_pos.y())
                    nx = scene_pos.x() / (self.img_width or 1)
                    ny = scene_pos.y() / (self.img_height or 1)

                    for i, ann in enumerate(self.annotations):
                        if ann[0] == 'obb':
                            pts_pix = self._get_points_pixel(ann)
                            if self._hit_test_obb_rotation_sector(ann, pts_pix, i, pos_pixel):
                                self.set_selected_index(i)
                                if not self.edit_mode:
                                    self.set_edit_mode(True)
                                self.edit_handle = 'rotate'
                                self.edit_start_norm = (nx, ny)
                                self.edit_original_ann = ann
                                self.edit_original_points = pts_pix
                                cx, cy, w, h, angle = self._obb_to_params(pts_pix)
                                self.edit_obb_center = (cx, cy)
                                self.edit_obb_w = w
                                self.edit_obb_h = h
                                self.edit_obb_angle = angle
                                start_mouse_angle = math.atan2(-(scene_pos.y() - cy), scene_pos.x() - cx)
                                self.edit_start_angle = start_mouse_angle
                                event.accept()
                                self.viewport().update()
                                return

                    idx = self.selected_index
                    hit = None
                    if idx != -1 and idx < len(self.annotations):
                        ann = self.annotations[idx]
                        if ann[0] == 'detect':
                            hit = self._hit_test_detect(ann, pos_pixel)
                        elif ann[0] == 'obb':
                            pts_pix = self._get_points_pixel(ann)
                            hit = self._hit_test_obb_handles(pts_pix, pos_pixel)
                        elif ann[0] == 'segment':
                            hit = self._hit_test_segment(ann, pos_pixel)

                    if hit is None:
                        for i, ann in enumerate(self.annotations):
                            typ = ann[0]
                            h = None
                            if typ == 'detect':
                                h = self._hit_test_detect(ann, pos_pixel)
                            elif typ == 'obb':
                                pts_pix = self._get_points_pixel(ann)
                                h = self._hit_test_obb_handles(pts_pix, pos_pixel)
                            elif typ == 'segment':
                                h = self._hit_test_segment(ann, pos_pixel)
                            if h is not None:
                                idx = i
                                hit = h
                                break

                    if idx != -1 and hit is not None:
                        self.set_selected_index(idx)
                        if isinstance(hit, tuple):
                            self.edit_handle = hit[0]
                            if len(hit) > 1:
                                self.edit_poly_idx = hit[1]
                            else:
                                self.edit_poly_idx = -1
                        else:
                            self.edit_handle = hit
                            self.edit_poly_idx = -1
                        self.edit_start_norm = (nx, ny)
                        self.edit_original_ann = self.annotations[idx]
                        self.edit_original_points = self._get_points_pixel(self.edit_original_ann)
                        if self.annotations[idx][0] == 'obb' and self.edit_handle != 'rotate':
                            cx, cy, w, h, angle = self._obb_to_params(self.edit_original_points)
                            self.edit_obb_center = (cx, cy)
                            self.edit_obb_w = w
                            self.edit_obb_h = h
                            self.edit_obb_angle = angle
                        event.accept()
                        self.viewport().update()
                        return
                    else:
                        self.set_selected_index(-1)
                        self.set_edit_mode(False)
                        event.accept()
                        self.viewport().update()
                        return
                except Exception:
                    event.accept()
                    return
            elif self.drawing_mode and self.current_tool is not None:
                event.accept()
                self._start_drawing(event)
                return
            else:
                super().mousePressEvent(event)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            if self.drawing_mode and not self.edit_mode:
                scene_pos = self.mapToScene(event.pos())
                if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                    self.cursor_scene_pos = (scene_pos.x(), scene_pos.y())
                else:
                    self.cursor_scene_pos = None
                self.viewport().update()
            else:
                self.cursor_scene_pos = None

            if self.edit_mode and self.edit_handle is not None and self.selected_index != -1:
                if self.edit_original_ann is None:
                    event.accept()
                    return
                scene_pos = self.mapToScene(event.pos())
                nx = scene_pos.x() / (self.img_width or 1)
                ny = scene_pos.y() / (self.img_height or 1)
                dx = nx - self.edit_start_norm[0]
                dy = ny - self.edit_start_norm[1]

                ann = self.edit_original_ann
                typ = ann[0]

                if typ == 'detect':
                    _, cls, cx, cy, w, h = ann
                    x1 = cx - w / 2
                    x2 = cx + w / 2
                    y1 = cy - h / 2
                    y2 = cy + h / 2
                    handle = self.edit_handle
                    if handle == 'move':
                        x1 += dx; x2 += dx; y1 += dy; y2 += dy
                    elif handle == 'point':
                        idx = self.edit_poly_idx
                        if idx == 0:
                            x1 += dx; y1 += dy
                        elif idx == 1:
                            x2 += dx; y1 += dy
                        elif idx == 2:
                            x2 += dx; y2 += dy
                        elif idx == 3:
                            x1 += dx; y2 += dy
                    else:
                        event.accept()
                        return
                    x1 = max(0., min(x1, 1.))
                    x2 = max(0., min(x2, 1.))
                    y1 = max(0., min(y1, 1.))
                    y2 = max(0., min(y2, 1.))
                    if x1 + 0.001 >= x2:
                        if handle in ('point', 'tl', 'bl') or (handle == 'move' and dx < 0):
                            x1 = x2 - 0.001
                        else:
                            x2 = x1 + 0.001
                    if y1 + 0.001 >= y2:
                        if handle in ('point', 'tl', 'tr') or (handle == 'move' and dy < 0):
                            y1 = y2 - 0.001
                        else:
                            y2 = y1 + 0.001
                    new_ann = ('detect', cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                    self.annotations[self.selected_index] = new_ann
                    self.edit_start_norm = (nx, ny)
                    self.edit_original_ann = new_ann
                    if self.on_annotation_modified_callback:
                        self.on_annotation_modified_callback(self.selected_index, new_ann)
                    if self.on_display_update_callback:
                        self.on_display_update_callback()
                    event.accept()
                    self.viewport().update()
                    return

                elif typ == 'obb':
                    if self.edit_handle == 'rotate':
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        mouse_angle = math.atan2(-(scene_pos.y() - cy), scene_pos.x() - cx)
                        delta = mouse_angle - self.edit_start_angle
                        new_angle = self.normalize_angle_rad(self.edit_obb_angle - delta)

                        new_points = self._obb_to_points(cx, cy, w, h, new_angle)
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('obb', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_start_angle = mouse_angle
                        self.edit_obb_angle = new_angle
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return

                    elif self.edit_handle == 'move':
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        new_cx = cx + dx * self.img_width
                        new_cy = cy + dy * self.img_height
                        new_cx = max(0, min(new_cx, self.img_width))
                        new_cy = max(0, min(new_cy, self.img_height))
                        new_points = self._obb_to_points(new_cx, new_cy, w, h, angle)
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('obb', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_obb_center = (new_cx, new_cy)
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        self.edit_start_norm = (nx, ny)
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return

                    elif self.edit_handle.startswith('corner'):
                        idx = int(self.edit_handle[6:])
                        opposite_idx = (idx + 2) % 4

                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)

                        points = self.edit_original_points
                        opp_pt = points[opposite_idx]
                        new_pt = (scene_pos.x(), scene_pos.y())

                        new_cx = (opp_pt[0] + new_pt[0]) / 2.0
                        new_cy = (opp_pt[1] + new_pt[1]) / 2.0

                        vx = new_pt[0] - new_cx
                        vy = new_pt[1] - new_cy
                        lx = vx * cos_a + vy * sin_a
                        ly = -vx * sin_a + vy * cos_a

                        MIN_SIZE = 2.0
                        new_w = max(2.0 * abs(lx), MIN_SIZE)
                        new_h = max(2.0 * abs(ly), MIN_SIZE)
                        new_w = min(new_w, self.img_width)
                        new_h = min(new_h, self.img_height)

                        new_points = self._obb_to_points(new_cx, new_cy, new_w, new_h, angle)
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('obb', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann

                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        self.edit_obb_center = (new_cx, new_cy)
                        self.edit_obb_w = new_w
                        self.edit_obb_h = new_h

                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return

                    elif self.edit_handle.startswith('edge'):
                        idx = int(self.edit_handle[4:])
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)
                        mx = scene_pos.x() - cx
                        my = scene_pos.y() - cy
                        lx = mx * cos_a + my * sin_a
                        ly = -mx * sin_a + my * cos_a
                        if idx == 0:
                            new_h = max(ly * 2, 1)
                        elif idx == 1:
                            new_w = max(lx * 2, 1)
                        elif idx == 2:
                            new_h = max(-ly * 2, 1)
                        else:
                            new_w = max(-lx * 2, 1)
                        new_w = min(new_w, self.img_width)
                        new_h = min(new_h, self.img_height)
                        new_points = self._obb_to_points(cx, cy, new_w, new_h, angle)
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('obb', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        self.edit_obb_w = new_w
                        self.edit_obb_h = new_h
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return

                elif typ == 'segment':
                    if self.edit_handle == 'point' and self.edit_poly_idx != -1:
                        points = self.edit_original_points
                        new_points = list(points)
                        oldx, oldy = points[self.edit_poly_idx]
                        newx = oldx + dx * self.img_width
                        newy = oldy + dy * self.img_height
                        newx = max(0, min(newx, self.img_width))
                        newy = max(0, min(newy, self.img_height))
                        new_points[self.edit_poly_idx] = (newx, newy)
                        norm_pts = []
                        for (px, py) in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('segment', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_start_norm = (nx, ny)
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return
                    elif self.edit_handle == 'move':
                        points = self.edit_original_points
                        new_points = [(p[0] + dx * self.img_width, p[1] + dy * self.img_height) for p in points]
                        new_points = [(max(0, min(px, self.img_width)), max(0, min(py, self.img_height))) for px, py in new_points]
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('segment', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_start_norm = (nx, ny)
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        self.viewport().update()
                        return

            elif self.drawing_mode and self.drawing and self.current_tool is not None:
                event.accept()
                self._continue_drawing(event)
                self.viewport().update()
                return

            super().mouseMoveEvent(event)
        except Exception:
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.edit_mode and self.edit_handle is not None:
                self.edit_handle = None
                self.edit_start_norm = None
                self.edit_original_ann = None
                self.edit_original_points = None
                self.edit_poly_idx = -1
                self.edit_obb_center = None
                self.edit_obb_w = None
                self.edit_obb_h = None
                self.edit_obb_angle = None
                self.edit_start_angle = None
                event.accept()
                return
            elif self.drawing_mode and self.drawing and self.current_tool is not None:
                event.accept()
                self._finish_drawing(event)
                self.drawing = False
                self.start_point = None
                self.current_point = None
                self.temp_rect = None
                self.temp_scribble = None
                self.viewport().update()
                if self.on_display_update_callback:
                    self.on_display_update_callback()
                return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.cursor_scene_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def contextMenuEvent(self, event):
        if self._suppress_context_menu:
            self._suppress_context_menu = False
            return
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        menu = QMenu(self)
        save_action = menu.addAction("Сохранить изображение")
        action = menu.exec_(event.globalPos())
        if action == save_action:
            self.save_current_image()

    # ----- Рисование -----
    def _start_drawing(self, event):
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if 0 <= x < self.img_width and 0 <= y < self.img_height:
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.drawing = True
            if self.current_tool in ("fg", "bg"):
                self._add_scribble(x, y, temp=True)

    def _continue_drawing(self, event):
        if self.start_point is None:
            return
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if 0 <= x < self.img_width and 0 <= y < self.img_height:
            self.current_point = (x, y)
            if self.current_tool == "rect":
                self.temp_rect = (self.start_point[0], self.start_point[1], x, y)
                if self.on_display_update_callback:
                    self.on_display_update_callback()
            elif self.current_tool in ("fg", "bg"):
                self._add_scribble(x, y, temp=True)

    def _finish_drawing(self, event):
        if self.current_tool == "rect" and self.start_point and self.current_point:
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            x1 = max(0, min(x1, self.img_width - 1))
            y1 = max(0, min(y1, self.img_height - 1))
            x2 = max(0, min(x2, self.img_width - 1))
            y2 = max(0, min(y2, self.img_height - 1))
            rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if self.on_rect_drawn_callback:
                self.on_rect_drawn_callback(rect)
        # Для штрихов ничего делать не нужно, они уже добавлены
        self.temp_scribble = None

    def _add_scribble(self, x, y, temp=False):
        if temp:
            self.temp_scribble = (x, y, 5)
        if self.on_scribble_added_callback:
            self.on_scribble_added_callback(x, y, self.current_tool)

    def normalize_angle_rad(self, angle):
        angle = math.fmod(angle, 2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        if angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def drawForeground(self, painter: QPainter, rect):
        if self.drawing and self.start_point and self.current_point and self.current_tool == "rect":
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

        # Временный штрих
        if self.drawing and self.current_tool in ("fg", "bg") and self.temp_scribble:
            x, y, r = self.temp_scribble
            color = QColor(0, 255, 0) if self.current_tool == "fg" else QColor(0, 0, 255)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(QPointF(x, y), r, r)

        if self.edit_mode and 0 <= self.selected_index < len(self.annotations):
            ann = self.annotations[self.selected_index]
            typ = ann[0]
            edit_color_bgr = settings.get_color('edit_points')
            edit_color_rgb = (edit_color_bgr[2], edit_color_bgr[1], edit_color_bgr[0])
            painter.setPen(QPen(QColor(*edit_color_rgb), 2))

            if typ == 'detect':
                points = self._get_points_pixel(ann)
                for (px, py) in points:
                    painter.drawEllipse(QPointF(px, py), 6, 6)

            elif typ == 'obb':
                points = self._get_points_pixel(ann)
                handles = self._get_obb_handles(points)
                for name, pt in handles.items():
                    painter.drawEllipse(QPointF(pt[0], pt[1]), 6, 6)
                sector = self._get_obb_rotation_sector_for_annotation(ann, points, self.selected_index)
                if sector:
                    cx, cy, radius, start_angle_rad, span_angle_rad = sector
                    start_deg = math.degrees(start_angle_rad) % 360
                    span_deg = math.degrees(span_angle_rad)
                    rectf = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
                    painter.setPen(QPen(QColor(0, 0, 255), 2))
                    painter.setBrush(QBrush(QColor(0, 0, 255, 80)))
                    painter.drawPie(rectf, int(start_deg * 16), int(span_deg * 16))

            elif typ == 'segment':
                points = self._get_points_pixel(ann)
                for (px, py) in points:
                    painter.drawEllipse(QPointF(px, py), 6, 6)

        if self.drawing_mode and not self.edit_mode and self.cursor_scene_pos is not None:
            x, y = self.cursor_scene_pos
            crosshair_color_bgr = settings.get_color('crosshair')
            crosshair_color_rgb = (crosshair_color_bgr[2], crosshair_color_bgr[1], crosshair_color_bgr[0])
            painter.setPen(QPen(QColor(*crosshair_color_rgb), 1))
            painter.drawLine(QPointF(0, y), QPointF(self.img_width, y))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.img_height))