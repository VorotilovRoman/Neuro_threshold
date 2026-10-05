# smart_view.py
from import_libs_internal import *
import math

from utils.settings import settings   # ← добавить


ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics.utils.ops import xyxyxyxy2xywhr, xywhr2xyxyxyxy  # noqa: F401
    ULTRALYTICS_AVAILABLE = True
except (ImportError, AttributeError):
    pass


class SmartGraphicsView(QGraphicsView):
    """
    Интерактивный виджет для отображения изображений и разметки.

    Инструменты рисования:
        • rect    — axis-aligned bounding box (detect)
        • segment — полигон (клики по вершинам, замыкание — клик по первой)
        • obb     — oriented bounding box (3 клика: угол → ширина → высота)

    Режимы редактирования:
        • detect  — 4 угловых маркера, перетаскивание за угол / за тело
        • segment — перетаскивание вершин / за тело
        • obb     — 4 угловых квадрата, 4 кружка на серединах сторон,
                    круглая ручка вращения ВНУТРИ (у верхней стороны),
                    перетаскивание за тело

    Выделение и навигация:
        • Ctrl+клик по объекту     — выделить + перейти в edit
        • клик по объекту в edit   — выделить + начать перетаскивание
        • клик вне объектов в edit — выйти из edit
        • перетаскивание за тело   — перемещение
    """

    # ==================================================================
    # Конструктор
    # ==================================================================
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

        # ----- Режимы -----
        self.drawing_mode = False
        self.current_tool = None
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None

        # ----- Состояние рисования segment -----
        self.temp_polygon_points = []
        self.temp_polygon_cursor = None

        # ----- Состояние рисования OBB -----
        self.obb_draw_stage = 0
        self.obb_draw_p1 = None
        self.obb_draw_p2 = None
        self.obb_draw_p3 = None
        self.obb_draw_cursor = None

        # ----- Состояние редактирования -----
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

        # ----- Вращение OBB -----
        self.rotate_start_mouse_angle = None
        self.rotate_start_obb_angle = None
        self.rotate_last_valid_angle = None

        self.img_width = 0
        self.img_height = 0
        self.annotations = []
        self.selected_index = -1

        self.cursor_scene_pos = None

        # ----- Коллбэки -----
        self.on_rect_drawn_callback = None
        self.on_obb_drawn_callback = None
        self.on_segment_drawn_callback = None
        self.on_scribble_added_callback = None
        self.on_display_update_callback = None
        self.on_reset_tool_callback = None
        self.on_annotation_modified_callback = None
        self.on_selection_changed_callback = None
        self.on_log_callback = None

        self._suppress_context_menu = False

        self._setup_pan_mode()

    # ==================================================================
    # Настройка режимов
    # ==================================================================
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
        self.temp_polygon_points = []
        self.temp_polygon_cursor = None
        self.obb_draw_stage = 0
        self.obb_draw_p1 = None
        self.obb_draw_p2 = None
        self.obb_draw_p3 = None
        self.obb_draw_cursor = None

    def _setup_edit_mode(self):
        self.setDragMode(QGraphicsView.NoDrag)
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None
        self.temp_polygon_points = []
        self.temp_polygon_cursor = None
        self.obb_draw_stage = 0

    def set_suggested_save_name(self, name):
        self._save_basename = name

    # ==================================================================
    # Управление инструментами
    # ==================================================================
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
            self.temp_polygon_points = []
            self.temp_polygon_cursor = None
            self.obb_draw_stage = 0
            if not self.edit_mode:
                self._setup_pan_mode()
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
            self._clear_edit_state()
            self.setCursor(QCursor(Qt.OpenHandCursor))
            self._setup_pan_mode()
        if self._current_pixmap:
            self.viewport().update()

    def _clear_edit_state(self):
        """Сброс всего состояния редактирования в одно место."""
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
        self.rotate_start_mouse_angle = None
        self.rotate_start_obb_angle = None
        self.rotate_last_valid_angle = None

    def set_annotations(self, annotations, img_width, img_height):
        self.annotations = annotations if annotations is not None else []
        self.img_width = img_width
        self.img_height = img_height

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

    def set_callbacks(self, on_rect_drawn=None, on_obb_drawn=None,
                      on_segment_drawn=None, on_scribble_added=None,
                      on_display_update=None, on_reset_tool=None,
                      on_annotation_modified=None, on_selection_changed=None,
                      on_log=None):
        self.on_rect_drawn_callback = on_rect_drawn
        self.on_obb_drawn_callback = on_obb_drawn
        self.on_segment_drawn_callback = on_segment_drawn
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

    # ==================================================================
    # Вспомогательные геометрические методы
    # ==================================================================
    def _get_points_pixel(self, ann):
        if self.img_width == 0 or self.img_height == 0:
            return []
        typ = ann[0]
        try:
            if typ == 'detect':
                _, _, cx, cy, w, h = ann
                x1 = int((cx - w / 2) * self.img_width)
                y1 = int((cy - h / 2) * self.img_height)
                x2 = int((cx + w / 2) * self.img_width)
                y2 = int((cy + h / 2) * self.img_height)
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif typ in ('obb', 'segment'):
                points = ann[2]
                pts = []
                for i in range(0, len(points), 2):
                    px = int(points[i] * self.img_width)
                    py = int(points[i + 1] * self.img_height)
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

    def _hit_test_point(self, point, pos_pixel, radius=8):
        x, y = point
        px, py = pos_pixel
        return abs(x - px) <= radius and abs(y - py) <= radius

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

    # ==================================================================
    # Единый диспетчер hit-testing
    # ==================================================================
    def _hit_test_any(self, ann, pos_pixel):
        """Возвращает имя хендла/кортеж для конкретной аннотации."""
        typ = ann[0]
        if typ == 'detect':
            return self._hit_test_detect(ann, pos_pixel)
        elif typ == 'obb':
            pts_pix = self._get_points_pixel(ann)
            return self._hit_test_obb_handles(pts_pix, pos_pixel)
        elif typ == 'segment':
            return self._hit_test_segment(ann, pos_pixel)
        return None

    def _find_hit(self, pos_pixel, prefer_idx=-1):
        """
        Ищет объект под курсором. Сначала проверяет prefer_idx (если он
        задан), затем — остальные. Возвращает (idx, hit) или (-1, None).
        """
        n = len(self.annotations)
        order = []
        if 0 <= prefer_idx < n:
            order.append(prefer_idx)
        for i in range(n):
            if i != prefer_idx:
                order.append(i)
        for i in order:
            hit = self._hit_test_any(self.annotations[i], pos_pixel)
            if hit is not None:
                return i, hit
        return -1, None

    # ==================================================================
    # OBB: чистая математика без torch/ultralytics
    # ==================================================================
    def _obb_to_params(self, pts_pixel):
        """
        4 угла (пиксели) → (cx, cy, w, h, angle_rad).

        Порядок углов (как их возвращает _obb_to_points):
            p0 = c + v1 + v2  (право-низ при angle=0)
            p1 = c - v1 + v2  (лево-низ)
            p2 = c - v1 - v2  (лево-верх)
            p3 = c + v1 - v2  (право-верх)
        angle = направление v1 (p1 → p0).
        """
        if len(pts_pixel) != 4:
            return (0.0, 0.0, 1.0, 1.0, 0.0)

        try:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts_pixel
        except Exception:
            return (0.0, 0.0, 1.0, 1.0, 0.0)

        cx = (x0 + x1 + x2 + x3) / 4.0
        cy = (y0 + y1 + y2 + y3) / 4.0

        w1 = math.hypot(x1 - x0, y1 - y0)
        w2 = math.hypot(x2 - x3, y2 - y3)
        w = (w1 + w2) / 2.0

        h1 = math.hypot(x2 - x1, y2 - y1)
        h2 = math.hypot(x3 - x0, y3 - y0)
        h = (h1 + h2) / 2.0

        angle = math.atan2(y0 - y1, x0 - x1)

        for v in (cx, cy, w, h, angle):
            if not math.isfinite(v):
                return (0.0, 0.0, 1.0, 1.0, 0.0)

        if w < 1e-3:
            w = 1.0
        if h < 1e-3:
            h = 1.0

        return (float(cx), float(cy), float(w), float(h), float(angle))

    def _obb_to_points(self, cx, cy, w, h, angle):
        def _finite(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(f):
                return None
            if abs(f) > 1e7:
                return None
            return f

        vals = [_finite(cx), _finite(cy), _finite(w), _finite(h), _finite(angle)]
        if any(v is None for v in vals):
            return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        cx, cy, w, h, angle = vals
        w = max(abs(w), 1e-3)
        h = max(abs(h), 1e-3)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        half_w = w / 2.0
        half_h = h / 2.0

        v1x, v1y = half_w * cos_a, half_w * sin_a
        v2x, v2y = -half_h * sin_a, half_h * cos_a

        p0 = (cx + v1x + v2x, cy + v1y + v2y)
        p1 = (cx - v1x + v2x, cy - v1y + v2y)
        p2 = (cx - v1x - v2x, cy - v1y - v2y)
        p3 = (cx + v1x - v2x, cy + v1y - v2y)

        out = []
        for px, py in (p0, p1, p2, p3):
            if not (math.isfinite(px) and math.isfinite(py)):
                return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            out.append((float(px), float(py)))
        return out

    def _obb_points_in_bounds(self, points, eps=0.5):
        W, H = self.img_width, self.img_height
        for px, py in points:
            if px < -eps or px > W + eps or py < -eps or py > H + eps:
                return False
        return True

    def _obb_world_half_extent(self, w, h, angle):
        cos_a = abs(math.cos(angle))
        sin_a = abs(math.sin(angle))
        hw = (w / 2.0) * cos_a + (h / 2.0) * sin_a
        hh = (w / 2.0) * sin_a + (h / 2.0) * cos_a
        return hw, hh

    def _get_obb_handles(self, pts_pixel):
        """
        Возвращает словарь хендлов OBB:
            corner0..corner3 — углы (квадраты)
            edge0..edge3     — середины сторон (жёлтые кружки)
            rotate           — круглая ручка вращения ВНУТРИ OBB,
                               сдвинутая от середины верхней стороны
                               (между corner0 и corner1) в сторону центра.
        """
        if len(pts_pixel) != 4:
            return {}

        cx, cy, w, h, angle = self._obb_to_params(pts_pixel)
        corners = self._obb_to_points(cx, cy, w, h, angle)
        if len(corners) != 4:
            return {}

        mid01 = ((corners[0][0] + corners[1][0]) / 2,
                 (corners[0][1] + corners[1][1]) / 2)
        mid12 = ((corners[1][0] + corners[2][0]) / 2,
                 (corners[1][1] + corners[2][1]) / 2)
        mid23 = ((corners[2][0] + corners[3][0]) / 2,
                 (corners[2][1] + corners[3][1]) / 2)
        mid30 = ((corners[3][0] + corners[0][0]) / 2,
                 (corners[3][1] + corners[0][1]) / 2)

        # Ручка вращения — ВНУТРИ бокса, ближе к верхней стороне.
        # Направление: от mid01 к центру.
        to_cx = cx - mid01[0]
        to_cy = cy - mid01[1]
        dist = math.hypot(to_cx, to_cy)
        if dist < 1e-6:
            rot_handle = mid01
        else:
            ux = to_cx / dist
            uy = to_cy / dist
            # Отступ: не больше 30% от меньшей стороны и не больше 30 px,
            # но минимум 12 px, чтобы не сливаться с edge0.
            min_side = min(w, h)
            offset = max(min(min_side * 0.30, 30.0), 12.0)
            # Не выходим за половину меньшей стороны (иначе уедем за центр).
            offset = min(offset, min_side * 0.45)
            rot_handle = (mid01[0] + ux * offset, mid01[1] + uy * offset)

        return {
            'corner0': corners[0],
            'corner1': corners[1],
            'corner2': corners[2],
            'corner3': corners[3],
            'edge0': mid01,
            'edge1': mid12,
            'edge2': mid23,
            'edge3': mid30,
            'rotate': rot_handle,
        }

    def _hit_test_obb_handles(self, pts_pixel, pos_pixel):
        handles = self._get_obb_handles(pts_pixel)
        if not handles:
            return None

        # 1) Ручка вращения — приоритет (проверяем первой).
        rot = handles.get('rotate')
        if rot and abs(rot[0] - pos_pixel[0]) <= 11 \
                and abs(rot[1] - pos_pixel[1]) <= 11:
            return 'rotate'

        # 2) Углы.
        for name in ('corner0', 'corner1', 'corner2', 'corner3'):
            pt = handles.get(name)
            if pt and self._hit_test_point(pt, pos_pixel, radius=9):
                return name

        # 3) Середины сторон.
        for name in ('edge0', 'edge1', 'edge2', 'edge3'):
            pt = handles.get(name)
            if pt and self._hit_test_point(pt, pos_pixel, radius=8):
                return name

        # 4) Тело OBB.
        if self.point_in_polygon(pos_pixel, pts_pixel):
            return 'move'
        return None

    # ------------------------------------------------------------------
    # Ограничение OBB рамками изображения при рисовании
    # ------------------------------------------------------------------
    def _max_height_inside(self, x1, y1, x2, y2, dir_x, dir_y):
        W, H = self.img_width, self.img_height
        max_h = float('inf')

        for base_x, base_y in ((x1, y1), (x2, y2)):
            if abs(dir_x) > 1e-9:
                if dir_x > 0:
                    t = (W - base_x) / dir_x
                else:
                    t = (0 - base_x) / dir_x
                if t < max_h:
                    max_h = t
            else:
                if base_x < 0 or base_x > W:
                    return 0.0

            if abs(dir_y) > 1e-9:
                if dir_y > 0:
                    t = (H - base_y) / dir_y
                else:
                    t = (0 - base_y) / dir_y
                if t < max_h:
                    max_h = t
            else:
                if base_y < 0 or base_y > H:
                    return 0.0

        if max_h < 0:
            max_h = 0.0
        return max_h

    def _compute_obb_preview_params(self):
        if (self.obb_draw_p1 is None or self.obb_draw_p2 is None
                or self.obb_draw_cursor is None):
            return None

        x1, y1 = self.obb_draw_p1
        x2, y2 = self.obb_draw_p2
        x3, y3 = self.obb_draw_cursor

        width = math.hypot(x2 - x1, y2 - y1)
        if width < 1e-6 or not math.isfinite(width):
            return None

        angle = math.atan2(y2 - y1, x2 - x1)
        if not math.isfinite(angle):
            return None

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        nx = -sin_a
        ny = cos_a

        proj = (x3 - x1) * nx + (y3 - y1) * ny
        sign = 1.0 if proj >= 0 else -1.0

        dir_x = nx * sign
        dir_y = ny * sign

        desired_h = abs((x3 - x1) * sin_a - (y3 - y1) * cos_a)
        if not math.isfinite(desired_h) or desired_h < 1.0:
            desired_h = 1.0

        max_h = self._max_height_inside(x1, y1, x2, y2, dir_x, dir_y)
        height = min(desired_h, max_h)
        if height < 1.0:
            height = 1.0

        cx = x1 + (width / 2.0) * cos_a + (height / 2.0) * dir_x
        cy = y1 + (width / 2.0) * sin_a + (height / 2.0) * dir_y

        return (cx, cy, width, height, angle)

    # ==================================================================
    # События мыши
    # ==================================================================
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        # --- Колёсико: сброс зума ---
        if event.button() == Qt.MiddleButton:
            self.reset_view()
            event.accept()
            return

        # --- Правый клик: выход из режимов / контекстное меню ---
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

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        # --- Ctrl+клик: выделить объект и войти в edit ---
        if event.modifiers() == Qt.ControlModifier:
            scene_pos = self.mapToScene(event.pos())
            if 0 <= scene_pos.x() < self.img_width and 0 <= scene_pos.y() < self.img_height:
                pos_pixel = (scene_pos.x(), scene_pos.y())
                idx, hit = self._find_hit(pos_pixel)
                if idx != -1:
                    self.set_selected_index(idx)
                    self.set_edit_mode(True)
                    event.accept()
                    QTimer.singleShot(0, self.viewport().update)
                    return
            event.accept()
            return

        # --- Режим редактирования: выделение + drag через единый _find_hit ---
        if self.edit_mode:
            scene_pos = self.mapToScene(event.pos())
            if not (0 <= scene_pos.x() < self.img_width
                    and 0 <= scene_pos.y() < self.img_height):
                super().mousePressEvent(event)
                return

            pos_pixel = (scene_pos.x(), scene_pos.y())
            nx = scene_pos.x() / (self.img_width or 1)
            ny = scene_pos.y() / (self.img_height or 1)

            # Приоритет — уже выделенному объекту.
            idx, hit = self._find_hit(pos_pixel, self.selected_index)

            if idx == -1:
                # Клик вне объектов — выходим из edit-режима.
                self.set_selected_index(-1)
                self.set_edit_mode(False)
                event.accept()
                QTimer.singleShot(0, self.viewport().update)
                return

            # Применяем выделение (если сменился объект).
            if idx != self.selected_index:
                self.set_selected_index(idx)

            # Универсальная инициализация drag для любого типа хендла.
            if isinstance(hit, tuple):
                self.edit_handle = hit[0]
                self.edit_poly_idx = hit[1] if len(hit) > 1 else -1
            else:
                self.edit_handle = hit
                self.edit_poly_idx = -1

            self.edit_start_norm = (nx, ny)
            self.edit_original_ann = self.annotations[idx]
            self.edit_original_points = self._get_points_pixel(self.edit_original_ann)

            ann = self.annotations[idx]
            if ann[0] == 'obb':
                cx, cy, w, h, angle = self._obb_to_params(self.edit_original_points)
                self.edit_obb_center = (cx, cy)
                self.edit_obb_w = w
                self.edit_obb_h = h
                self.edit_obb_angle = angle
                if self.edit_handle == 'rotate':
                    self.rotate_start_mouse_angle = math.atan2(
                        pos_pixel[1] - cy, pos_pixel[0] - cx)
                    self.rotate_start_obb_angle = angle
                    self.rotate_last_valid_angle = angle

            event.accept()
            QTimer.singleShot(0, self.viewport().update)
            return

        # --- Режим рисования: старт нового объекта ---
        if self.drawing_mode and self.current_tool is not None:
            event.accept()
            self._start_drawing(event)
            return

        # --- Обычный режим (без edit) — panning ---
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            # Обновляем позицию курсора для crosshair при рисовании
            if self.drawing_mode and not self.edit_mode:
                scene_pos = self.mapToScene(event.pos())
                if 0 <= scene_pos.x() < self.img_width \
                        and 0 <= scene_pos.y() < self.img_height:
                    self.cursor_scene_pos = (scene_pos.x(), scene_pos.y())
                else:
                    self.cursor_scene_pos = None
                QTimer.singleShot(0, self.viewport().update)

            # --- Обновление при рисовании ---
            if self.drawing_mode and self.current_tool is not None:
                if self.current_tool == 'rect':
                    if self.drawing and self.start_point is not None:
                        self._continue_drawing(event)
                elif self.current_tool == 'segment':
                    self._update_segment_cursor(event)
                elif self.current_tool == 'obb':
                    self._update_obb_draw(event)

            # --- Режим редактирования (drag) ---
            if self.edit_mode and self.edit_handle is not None \
                    and self.selected_index != -1:
                if self.edit_original_ann is None:
                    event.accept()
                    return

                scene_pos = self.mapToScene(event.pos())
                nx = scene_pos.x() / (self.img_width or 1)
                ny = scene_pos.y() / (self.img_height or 1)
                dx = nx - self.edit_start_norm[0]
                dy = ny - self.edit_start_norm[1]

                clamped_sx = max(0.0, min(float(scene_pos.x()), float(self.img_width)))
                clamped_sy = max(0.0, min(float(scene_pos.y()), float(self.img_height)))

                ann = self.edit_original_ann
                typ = ann[0]

                # ---------- detect ----------
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
                        idx_pt = self.edit_poly_idx
                        if idx_pt == 0:
                            x1 += dx; y1 += dy
                        elif idx_pt == 1:
                            x2 += dx; y1 += dy
                        elif idx_pt == 2:
                            x2 += dx; y2 += dy
                        elif idx_pt == 3:
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
                    new_ann = ('detect', cls, (x1 + x2) / 2, (y1 + y2) / 2,
                               x2 - x1, y2 - y1)
                    self.annotations[self.selected_index] = new_ann
                    self.edit_start_norm = (nx, ny)
                    self.edit_original_ann = new_ann
                    if self.on_annotation_modified_callback:
                        self.on_annotation_modified_callback(self.selected_index, new_ann)
                    if self.on_display_update_callback:
                        self.on_display_update_callback()
                    event.accept()
                    QTimer.singleShot(0, self.viewport().update)
                    return

                # ---------- OBB ----------
                elif typ == 'obb':
                    if self.edit_handle == 'rotate':
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h

                        current_mouse_angle = math.atan2(
                            scene_pos.y() - cy, scene_pos.x() - cx)
                        delta_angle = current_mouse_angle - self.rotate_start_mouse_angle
                        new_angle = self.normalize_angle_rad(
                            self.rotate_start_obb_angle + delta_angle)

                        trial_points = self._obb_to_points(cx, cy, w, h, new_angle)
                        if self._obb_points_in_bounds(trial_points):
                            self.rotate_last_valid_angle = new_angle
                            applied_angle = new_angle
                        else:
                            if self.rotate_last_valid_angle is None:
                                event.accept()
                                QTimer.singleShot(0, self.viewport().update)
                                return
                            applied_angle = self.rotate_last_valid_angle

                        new_points = self._obb_to_points(cx, cy, w, h, applied_angle)
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / self.img_width)
                            norm_pts.append(py / self.img_height)
                        new_ann = ('obb', ann[1], norm_pts)
                        self.annotations[self.selected_index] = new_ann
                        self.edit_obb_angle = applied_angle
                        self.edit_original_ann = new_ann
                        self.edit_original_points = new_points
                        if self.on_annotation_modified_callback:
                            self.on_annotation_modified_callback(self.selected_index, new_ann)
                        if self.on_display_update_callback:
                            self.on_display_update_callback()
                        event.accept()
                        QTimer.singleShot(0, self.viewport().update)
                        return

                    elif self.edit_handle == 'move':
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle

                        hw, hh = self._obb_world_half_extent(w, h, angle)
                        W, H = self.img_width, self.img_height

                        new_cx = cx + dx * W
                        new_cy = cy + dy * H
                        if hw <= W / 2.0:
                            new_cx = max(hw, min(new_cx, W - hw))
                        else:
                            new_cx = W / 2.0
                        if hh <= H / 2.0:
                            new_cy = max(hh, min(new_cy, H - hh))
                        else:
                            new_cy = H / 2.0

                        new_points = self._obb_to_points(new_cx, new_cy, w, h, angle)
                        if not self._obb_points_in_bounds(new_points):
                            event.accept()
                            QTimer.singleShot(0, self.viewport().update)
                            return
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(px / W)
                            norm_pts.append(py / H)
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
                        QTimer.singleShot(0, self.viewport().update)
                        return

                    elif self.edit_handle.startswith('corner'):
                        idx_c = int(self.edit_handle[6:])
                        opposite_idx = (idx_c + 2) % 4
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)
                        points = self.edit_original_points
                        opp_pt = points[opposite_idx]
                        new_pt = (clamped_sx, clamped_sy)

                        new_cx = (opp_pt[0] + new_pt[0]) / 2.0
                        new_cy = (opp_pt[1] + new_pt[1]) / 2.0

                        vx = new_pt[0] - new_cx
                        vy = new_pt[1] - new_cy
                        lx = vx * cos_a + vy * sin_a
                        ly = -vx * sin_a + vy * cos_a

                        MIN_SIZE = 2.0
                        new_w = max(2.0 * abs(lx), MIN_SIZE)
                        new_h = max(2.0 * abs(ly), MIN_SIZE)
                        max_dim = math.hypot(self.img_width, self.img_height)
                        new_w = min(new_w, max_dim)
                        new_h = min(new_h, max_dim)

                        new_points = self._obb_to_points(new_cx, new_cy, new_w, new_h, angle)
                        if not self._obb_points_in_bounds(new_points):
                            event.accept()
                            QTimer.singleShot(0, self.viewport().update)
                            return
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(max(0.0, min(1.0, px / self.img_width)))
                            norm_pts.append(max(0.0, min(1.0, py / self.img_height)))
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
                        QTimer.singleShot(0, self.viewport().update)
                        return

                    elif self.edit_handle.startswith('edge'):
                        # Перетаскивание маркера на середине стороны.
                        #
                        # Локальные координаты мыши относительно центра OBB:
                        #   lx — проекция вдоль длинной стороны w,
                        #   ly — проекция вдоль короткой стороны h.
                        #
                        # Положение середин сторон в локальных координатах:
                        #   edge0 (нижняя, между corner0 и corner1) → (0, +h/2)
                        #   edge1 (левая,  между corner1 и corner2) → (-w/2, 0)
                        #   edge2 (верхняя, между corner2 и corner3) → (0, -h/2)
                        #   edge3 (правая, между corner3 и corner0) → (+w/2, 0)
                        #
                        # Поэтому знаки для новых w/h берутся с учётом того,
                        # с какой стороны от центра находится тянущаяся сторона.
                        idx_e = int(self.edit_handle[4:])
                        cx, cy = self.edit_obb_center
                        w = self.edit_obb_w
                        h = self.edit_obb_h
                        angle = self.edit_obb_angle
                        cos_a = math.cos(angle)
                        sin_a = math.sin(angle)
                        mx = clamped_sx - cx
                        my = clamped_sy - cy
                        lx = mx * cos_a + my * sin_a
                        ly = -mx * sin_a + my * cos_a

                        new_w = w
                        new_h = h
                        if idx_e == 0:
                            # Нижняя сторона: ly положительна при движении вниз.
                            new_h = max(ly * 2, 1)
                        elif idx_e == 1:
                            # Левая сторона: lx отрицательна, берём -lx.
                            new_w = max(-lx * 2, 1)
                        elif idx_e == 2:
                            # Верхняя сторона: ly отрицательна, берём -ly.
                            new_h = max(-ly * 2, 1)
                        else:
                            # Правая сторона: lx положительна.
                            new_w = max(lx * 2, 1)

                        max_dim = math.hypot(self.img_width, self.img_height)
                        new_w = min(new_w, max_dim)
                        new_h = min(new_h, max_dim)
                        new_points = self._obb_to_points(cx, cy, new_w, new_h, angle)
                        if not self._obb_points_in_bounds(new_points):
                            event.accept()
                            QTimer.singleShot(0, self.viewport().update)
                            return
                        norm_pts = []
                        for px, py in new_points:
                            norm_pts.append(max(0.0, min(1.0, px / self.img_width)))
                            norm_pts.append(max(0.0, min(1.0, py / self.img_height)))
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
                        QTimer.singleShot(0, self.viewport().update)
                        return

                # ---------- segment ----------
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
                        QTimer.singleShot(0, self.viewport().update)
                        return
                    elif self.edit_handle == 'move':
                        points = self.edit_original_points
                        new_points = [(p[0] + dx * self.img_width,
                                       p[1] + dy * self.img_height) for p in points]
                        new_points = [(max(0, min(px, self.img_width)),
                                       max(0, min(py, self.img_height)))
                                      for px, py in new_points]
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
                        QTimer.singleShot(0, self.viewport().update)
                        return

            super().mouseMoveEvent(event)
        except Exception as e:
            print(f"[mouseMoveEvent] {type(e).__name__}: {e}")
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.edit_mode and self.edit_handle is not None:
                self._clear_edit_state()
                event.accept()
                return

            elif self.drawing_mode and self.drawing and self.current_tool == "rect":
                event.accept()
                self._finish_drawing(event)
                self.drawing = False
                self.start_point = None
                self.current_point = None
                self.temp_rect = None
                self.temp_scribble = None
                QTimer.singleShot(0, self.viewport().update)
                if self.on_display_update_callback:
                    self.on_display_update_callback()
                return

        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.cursor_scene_pos = None
        QTimer.singleShot(0, self.viewport().update)
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

    # ==================================================================
    # Рисование новых объектов
    # ==================================================================
    def _start_drawing(self, event):
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if not (0 <= x < self.img_width and 0 <= y < self.img_height):
            return

        if self.current_tool == "rect":
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.drawing = True
            QTimer.singleShot(0, self.viewport().update)

        elif self.current_tool == "segment":
            self.temp_polygon_points.append((x, y))
            self.temp_polygon_cursor = (x, y)
            if len(self.temp_polygon_points) >= 3:
                first = self.temp_polygon_points[0]
                if math.hypot(x - first[0], y - first[1]) <= 10:
                    self._finish_segment()
                    return
            self.drawing = True
            QTimer.singleShot(0, self.viewport().update)

        elif self.current_tool == "obb":
            if self.obb_draw_stage == 0:
                self.obb_draw_p1 = (x, y)
                self.obb_draw_stage = 1
                self.obb_draw_cursor = (x, y)
                self.drawing = True
            elif self.obb_draw_stage == 1:
                if math.hypot(x - self.obb_draw_p1[0],
                              y - self.obb_draw_p1[1]) < 3:
                    return
                self.obb_draw_p2 = (x, y)
                self.obb_draw_stage = 2
                self.obb_draw_cursor = (x, y)
            elif self.obb_draw_stage == 2:
                self.obb_draw_p3 = (x, y)
                self._finish_obb()
                return
            QTimer.singleShot(0, self.viewport().update)

    def _update_segment_cursor(self, event):
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if 0 <= x < self.img_width and 0 <= y < self.img_height:
            self.temp_polygon_cursor = (x, y)
            QTimer.singleShot(0, self.viewport().update)

    def _update_obb_draw(self, event):
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if 0 <= x < self.img_width and 0 <= y < self.img_height:
            self.obb_draw_cursor = (x, y)
            QTimer.singleShot(0, self.viewport().update)

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
        self.temp_scribble = None

    def _finish_segment(self):
        pts = self.temp_polygon_points
        if len(pts) < 3:
            self.temp_polygon_points = []
            self.temp_polygon_cursor = None
            self.drawing = False
            QTimer.singleShot(0, self.viewport().update)
            return
        norm_pts = []
        for px, py in pts:
            norm_pts.append(px / self.img_width)
            norm_pts.append(py / self.img_height)
        self.temp_polygon_points = []
        self.temp_polygon_cursor = None
        self.drawing = False
        QTimer.singleShot(0, self.viewport().update)
        if self.on_segment_drawn_callback:
            self.on_segment_drawn_callback(norm_pts)

    def _finish_obb(self):
        params = self._compute_obb_preview_params()
        if params is None:
            self._reset_obb_draw()
            return

        cx, cy, width, height, angle = params
        points = self._obb_to_points(cx, cy, width, height, angle)

        norm_pts = []
        for px, py in points:
            nx = max(0.0, min(1.0, px / self.img_width))
            ny = max(0.0, min(1.0, py / self.img_height))
            norm_pts.append(nx)
            norm_pts.append(ny)

        self._reset_obb_draw()
        if self.on_obb_drawn_callback:
            self.on_obb_drawn_callback(norm_pts)

    def _reset_obb_draw(self):
        self.obb_draw_stage = 0
        self.obb_draw_p1 = None
        self.obb_draw_p2 = None
        self.obb_draw_p3 = None
        self.obb_draw_cursor = None
        self.drawing = False
        QTimer.singleShot(0, self.viewport().update)

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

    # ==================================================================
    # Отрисовка поверх сцены
    # ==================================================================
    def drawForeground(self, painter: QPainter, rect):
        try:
            self._draw_foreground_impl(painter, rect)
        except Exception as e:
            print(f"[drawForeground] {type(e).__name__}: {e}")

    def _draw_foreground_impl(self, painter: QPainter, rect):
        # --- Временный прямоугольник (rect) ---
        if self.drawing and self.start_point and self.current_point \
                and self.current_tool == "rect":
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(Qt.NoBrush)
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            painter.drawRect(QRectF(min(x1, x2), min(y1, y2),
                                    abs(x2 - x1), abs(y2 - y1)))

        # --- Временный полигон (segment) ---
        if self.current_tool == "segment" and self.temp_polygon_points:
            painter.setPen(QPen(QColor(0, 200, 0), 2))
            painter.setBrush(QBrush(QColor(0, 200, 0, 80)))
            pts = [QPointF(px, py) for px, py in self.temp_polygon_points]
            if len(pts) >= 2:
                painter.drawPolyline(pts)
            if self.temp_polygon_cursor:
                painter.drawLine(pts[-1], QPointF(*self.temp_polygon_cursor))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            for px, py in self.temp_polygon_points:
                painter.drawEllipse(QPointF(px, py), 4, 4)
            if len(self.temp_polygon_points) >= 3:
                first = self.temp_polygon_points[0]
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(first[0], first[1]), 12, 12)

        # --- Временный OBB при рисовании ---
        if self.current_tool == "obb":
            if self.obb_draw_stage == 1 and self.obb_draw_p1 \
                    and self.obb_draw_cursor:
                x1, y1 = self.obb_draw_p1
                x2, y2 = self.obb_draw_cursor
                painter.setPen(QPen(QColor(0, 200, 0), 2, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                painter.setBrush(QBrush(QColor(0, 200, 0)))
                painter.setPen(QPen(QColor(0, 200, 0), 1))
                painter.drawEllipse(QPointF(x1, y1), 5, 5)
                painter.drawEllipse(QPointF(x2, y2), 5, 5)

            elif self.obb_draw_stage == 2:
                params = self._compute_obb_preview_params()
                if params is not None:
                    try:
                        cx, cy, width, height, angle = params
                        pts = self._obb_to_points(cx, cy, width, height, angle)

                        qpts = [QPointF(px, py) for px, py in pts]
                        qpts.append(qpts[0])
                        painter.setPen(QPen(QColor(0, 200, 0), 2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawPolyline(qpts)

                        if width > 2.0 and height > 2.0:
                            path = QPainterPath()
                            path.moveTo(pts[0][0], pts[0][1])
                            for px, py in pts[1:]:
                                path.lineTo(px, py)
                            path.closeSubpath()
                            painter.fillPath(path, QBrush(QColor(0, 200, 0, 60)))

                        if self.obb_draw_cursor is not None:
                            x3, y3 = self.obb_draw_cursor
                            painter.setPen(QPen(QColor(255, 0, 0), 2))
                            painter.setBrush(QBrush(QColor(255, 0, 0)))
                            painter.drawEllipse(QPointF(x3, y3), 5, 5)
                    except Exception as e:
                        print(f"[OBB preview] skip: {e}")

        # --- Временный штрих (fg/bg) ---
        if self.drawing and self.current_tool in ("fg", "bg") and self.temp_scribble:
            x, y, r = self.temp_scribble
            color = QColor(0, 255, 0) if self.current_tool == "fg" else QColor(0, 0, 255)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(QPointF(x, y), r, r)

        # --- Хендлы в режиме редактирования ---
        if self.edit_mode and 0 <= self.selected_index < len(self.annotations):
            ann = self.annotations[self.selected_index]
            typ = ann[0]
            edit_color_bgr = settings.get_color('edit_points')
            edit_color_rgb = (edit_color_bgr[2], edit_color_bgr[1], edit_color_bgr[0])
            painter.setPen(QPen(QColor(*edit_color_rgb), 2))

            if typ == 'detect':
                points = self._get_points_pixel(ann)
                for (px, py) in points:
                    painter.setBrush(QBrush(QColor(*edit_color_rgb)))
                    painter.drawEllipse(QPointF(px, py), 6, 6)

            elif typ == 'obb':
                points = self._get_points_pixel(ann)
                handles = self._get_obb_handles(points)
                if handles:
                    # Углы — квадраты
                    for name in ('corner0', 'corner1', 'corner2', 'corner3'):
                        pt = handles.get(name)
                        if pt is None:
                            continue
                        painter.setBrush(QBrush(QColor(*edit_color_rgb)))
                        painter.setPen(QPen(QColor(*edit_color_rgb), 2))
                        painter.drawRect(QRectF(pt[0] - 5, pt[1] - 5, 10, 10))
                    # Середины сторон — жёлтые кружки
                    for name in ('edge0', 'edge1', 'edge2', 'edge3'):
                        pt = handles.get(name)
                        if pt is None:
                            continue
                        painter.setBrush(QBrush(QColor(255, 255, 0)))
                        painter.setPen(QPen(QColor(180, 180, 0), 1))
                        painter.drawEllipse(QPointF(pt[0], pt[1]), 4, 4)
                    # Ручка вращения — крупный синий кружок внутри OBB
                    rot = handles.get('rotate')
                    if rot:
                        # Тонкая линия от центра к ручке — визуальная связь.
                        cx, cy, _, _, _ = self._obb_to_params(points)
                        painter.setPen(QPen(QColor(0, 0, 255, 90), 1, Qt.DashLine))
                        painter.drawLine(QPointF(cx, cy), QPointF(rot[0], rot[1]))
                        # Синий круг с заливкой
                        painter.setPen(QPen(QColor(0, 0, 255), 2))
                        painter.setBrush(QBrush(QColor(0, 0, 255, 140)))
                        painter.drawEllipse(QPointF(rot[0], rot[1]), 9, 9)
                        # Белая иконка «дуга вращения»
                        painter.setPen(QPen(QColor(255, 255, 255), 2))
                        painter.drawArc(QRectF(rot[0] - 5, rot[1] - 5, 10, 10),
                                        30 * 16, 120 * 16)

            elif typ == 'segment':
                points = self._get_points_pixel(ann)
                for (px, py) in points:
                    painter.setBrush(QBrush(QColor(*edit_color_rgb)))
                    painter.drawEllipse(QPointF(px, py), 6, 6)

        # --- Crosshair при рисовании ---
        if self.drawing_mode and not self.edit_mode \
                and self.cursor_scene_pos is not None:
            x, y = self.cursor_scene_pos
            crosshair_color_bgr = settings.get_color('crosshair')
            crosshair_color_rgb = (crosshair_color_bgr[2],
                                   crosshair_color_bgr[1],
                                   crosshair_color_bgr[0])
            painter.setPen(QPen(QColor(*crosshair_color_rgb), 1))
            painter.drawLine(QPointF(0, y), QPointF(self.img_width, y))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.img_height))