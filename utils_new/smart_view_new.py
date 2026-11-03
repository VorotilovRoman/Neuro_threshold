# -*- coding: utf-8 -*-
"""Интерактивный Qt-просмотрщик разметки: detect, OBB, segment.

Независимая ветка старого utils.smart_view. Старый файл НЕ импортируется.
Все функции мыши, хендлы, инструменты и callback API сохранены.

Отличия:
    * базовое изображение не содержит разметку, её рисует drawForeground;
    * косметические QPen сохраняют ширину линии при масштабировании;
    * подписи и маркеры имеют размеры в логических пикселях окна;
    * скрытие рамок/меток и настраиваемая заливка на независимых Qt-слоях;
    * размер QPixmap не обязан совпадать с рабочими размерами изображения;
    * импорты Qt указаны явно, без циклического import_libs_internal.

Формат аннотаций:
    ('detect', class_id, cx, cy, width, height)
    ('obb'|'segment', class_id, [x1, y1, x2, y2, ...])
Все координаты нормализованы к рабочей ширине / высоте изображения.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer
from PyQt5.QtGui import (
    QBrush, QColor, QCursor, QFont, QFontMetricsF,
    QPainter, QPen, QPixmap, QPolygonF, QTransform,
)
from PyQt5.QtWidgets import (
    QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QMenu,
)

from utils.settings import settings


@dataclass(frozen=True)
class AnnotationDisplayStyle:
    """Визуальные размеры в логических пикселях Qt, а не пикселях снимка."""
    line_width_px: float = 1.0
    selected_line_width_px: float = 1.8
    preview_line_width_px: float = 1.25
    handle_outline_width_px: float = 1.0
    font_size_px: int = 11
    label_padding_px: float = 2.0
    label_gap_px: float = 3.0
    label_background_alpha: int = 160
    # 'auto' = не показывать подписи слишком мелких объектов; выделенный — всегда
    # 'all', 'selected', 'none' также поддерживаются
    label_mode: str = 'auto'
    auto_label_min_width_px: float = 34.0
    auto_label_min_height_px: float = 17.0
    # Показывать ли контуры и подписи независимо друг от друга.
    show_frames: bool = True
    show_labels: bool = True
    # Совместимое имя переключателя V3/V20: заливка произвольным цветом.
    # По умолчанию белая, непрозрачная; исходное изображение не меняется.
    fill_white: bool = False
    fill_color: str = "#FFFFFF"
    # Ранее существовавшая мягкая заливка цветом класса (в отличие от белой).
    fill_annotations: bool = False
    fill_alpha: int = 24
    # Оставлено для обратной совместимости с API V3. Новая вкладка всегда
    # рисует классы по dataset_class_palette (нет класса -> класс 0).
    use_class_colors: bool = True


def set_fill_color_button_preview(button, color):
    """Общий Qt-предпросмотр выбранного цвета без создания новой утилиты.

    Работает и при выключенном переключателе заливки. Отображает цвет
    самой кнопкой, обеспечивая контраст надписи и рамку у белого фона.
    """
    qcolor = QColor(color)
    if not qcolor.isValid():
        raise ValueError(f'Некорректный цвет заливки: {color!r}')
    name = qcolor.name().upper()
    brightness = (qcolor.red() * 299 + qcolor.green() * 587 + qcolor.blue() * 114) / 1000
    text_color = '#000000' if brightness >= 145 else '#FFFFFF'
    button.setText(f'Цвет: {name}')
    button.setStyleSheet(
        f'QPushButton {{ background-color: {name}; color: {text_color}; '
        'border: 1px solid #777777; border-radius: 4px; padding: 3px 6px; }}'
    )


class SmartGraphicsViewNew(QGraphicsView):
    """
    Интерактивный виджет для отображения изображений и разметки.

    Инструменты рисования:
        • rect         — axis-aligned bounding box (detect)
        • segment      — полигон (клики по вершинам, замыкание — клик по первой)
        • segment_line — свободное рисование полигона: зажать ЛКМ, вести,
                         отпустить — контур автоматически замыкается на начало
        • obb          — oriented bounding box (3 клика: угол → ширина → высота)

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

    # ------------------------------------------------------------------
    # Размеры элементов управления.
    # *_PX — это ЭКРАННЫЕ пиксели. Они не зависят от зума: при отрисовке
    # делятся на текущий масштаб scene, чтобы визуально сохранять размер.
    # ------------------------------------------------------------------
    # Отрисовка (радиусы кружков/полустороны квадратов и толщина линий)
    DRAW_DETECT_POINT_PX   = 5      # угол bbox detect
    DRAW_SEGMENT_POINT_PX  = 5      # вершина polygon
    DRAW_OBB_CORNER_PX     = 5      # угол OBB (квадрат со стороной 2*PX)
    DRAW_OBB_EDGE_PX       = 4      # середина стороны OBB (жёлтый кружок)
    DRAW_OBB_ROTATE_PX     = 8      # радиус синей ручки вращения
    DRAW_LINE_WIDTH_PX     = 1.6    # толщина контуров фигур и линий хендлов

    # Хит-тестирование (радиус попадания)
    HIT_POINT_PX           = 8      # detect углы, segment вершины
    HIT_OBB_CORNER_PX      = 9      # углы OBB
    HIT_OBB_EDGE_PX        = 8      # середины сторон OBB
    HIT_OBB_ROTATE_PX      = 11     # ручка вращения

    # Двойной клик по сегменту:
    # попадание по вершине (удаление) — заметно больше визуального маркера,
    # чтобы по «точечке» было легко попасть мышью.
    HIT_SEGMENT_POINT_DBL_PX = 20   # экранные пиксели
    HIT_SEGMENT_EDGE_DBL_PX  = 20    # экранные пиксели (попадание по ребру)

    # Доля меньшей стороны объекта, выше которой маркер не увеличивается.
    # Защита от ситуации «объект крошечный, а маркер в нём занимает пол-объекта».
    MARKER_MAX_FRACTION    = 0.22
    # Минимальное число вершин полигона сегментации. Если пользователь
    # нарисовал меньше — недостающие точки добавляются на самых длинных
    # сегментах (середины сторон). Относится и к "segment", и к "segment_line".
    MIN_SEGMENT_POINTS     = 5
    # Минимальное расстояние между сохраняемыми точками при свободном
    # рисовании segment_line (в пикселях изображения).
    FREEHAND_MIN_STEP_PX   = 3
    HIT_CLOSE_POLYGON_PX  = 10

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
        self.display_style = AnnotationDisplayStyle()
        self._class_color_provider = None
        self._class_name_provider = None

        # ----- Режимы -----
        self.drawing_mode = False
        self.current_tool = None
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None
        self.temp_scribble = None

        # ----- Состояние рисования segment (кликами) -----
        self.temp_polygon_points = []
        self.temp_polygon_cursor = None

        # ----- Состояние свободного рисования segment_line -----
        self.freehand_points = []
        self.freehand_drawing = False

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
        self.on_annotation_right_clicked_callback = None
        self.on_log_callback = None

        self._suppress_context_menu = False

        self._setup_pan_mode()

        # Общий конфиг settings/settings.json для старых и новых утилит.
        # Новые настройки размеров применяем при создании виджета и после
        # каждого сохранения во вкладке «Настройки», включая сброс.
        self.apply_global_display_settings()
        settings.settings_changed.connect(self.apply_global_display_settings)

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
        self.freehand_points = []
        self.freehand_drawing = False
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
        self.freehand_points = []
        self.freehand_drawing = False
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
            self.freehand_points = []
            self.freehand_drawing = False
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
        """Обновляет векторную разметку; QPixmap при этом не перегенерируется."""
        # Нормализуем только classless записи; корректные аннотации не
        # копируем, чтобы не менять типичный путь передачи между view и tab.
        source = annotations if annotations is not None else []
        normalized = [self._normalize_classless_annotation(a) for a in source]
        self.annotations = (source if all(x is y for x, y in zip(normalized, source))
                            else normalized)
        iw = max(0, int(img_width or 0))
        ih = max(0, int(img_height or 0))
        if (iw, ih) != (self.img_width, self.img_height):
            self.img_width, self.img_height = iw, ih
            self._sync_scene_geometry()
        self.viewport().update()

    @staticmethod
    def _normalize_classless_annotation(ann):
        """Датасет: записи без класса преобразуются в единый формат с 0.

        detect: ('detect',cx,cy,w,h) -> ('detect',0,cx,cy,w,h)
        obb/segment: ('segment',points) -> ('segment',0,points)
        Также None либо отрицательный class_id означает класс 0.
        Не меняет геометрию или тип корректных записей.
        """
        if not isinstance(ann, (list, tuple)) or not ann:
            return ann
        typ = ann[0]
        if typ == 'detect':
            if len(ann) == 5:
                return ('detect', 0, *ann[1:])
            if len(ann) == 6:
                try:
                    if ann[1] is not None and int(ann[1]) >= 0:
                        return ann
                except (TypeError, ValueError, OverflowError):
                    pass
                return ('detect', 0, *ann[2:])
        elif typ in ('obb', 'segment'):
            if len(ann) == 2:
                return (typ, 0, ann[1])
            if len(ann) == 3:
                try:
                    if ann[1] is not None and int(ann[1]) >= 0:
                        return ann
                except (TypeError, ValueError, OverflowError):
                    pass
                return (typ, 0, ann[2])
        return ann

    def set_display_style(self, style=None, **overrides):
        """Настройки Qt-отрисовки. Можно передать dataclass или ключевые поля."""
        if style is None:
            style = self.display_style
        if not isinstance(style, AnnotationDisplayStyle):
            raise TypeError('style должен быть AnnotationDisplayStyle')
        if overrides:
            style = replace(style, **overrides)
        if style.label_mode not in ('all', 'selected', 'none', 'auto'):
            raise ValueError('label_mode: all / selected / none / auto')
        if min(style.line_width_px, style.selected_line_width_px,
               style.preview_line_width_px, style.handle_outline_width_px) <= 0:
            raise ValueError('Толщина линий должна быть положительной')
        if style.font_size_px < 1:
            raise ValueError('Размер шрифта должен быть положительным')
        self.display_style = style
        self.viewport().update()

    def apply_global_display_settings(self, _new_settings=None):
        """Читает общие настройки пикселей и палитры классов.

        Не затрагивает локальные переключатели скрытия рамок / меток и
        пользовательской заливки. Изменения применяются по settings_changed.
        """
        line_px = float(settings.get_line_thickness_px())
        font_px = int(settings.get_font_size_px())
        self.set_display_style(
            line_width_px=line_px,
            selected_line_width_px=max(1.5, line_px * 1.8),
            preview_line_width_px=max(1.0, line_px * 1.25),
            handle_outline_width_px=max(1.0, min(2.0, line_px)),
            font_size_px=font_px,
            use_class_colors=True,
        )

    def set_annotation_visibility(self, *, show_frames=None, show_labels=None,
                                  fill_white=None, fill_color=None):
        """Независимые параметры отображения (не изменяют данные аннотаций).

        show_frames=False: скрыть ВСЕ постоянные рамки/контуры;
        show_labels=False: скрыть подписи независимо от режима label_mode;
        fill_white=True: непрозрачная заливка области каждого объекта цветом
        fill_color (по умолчанию #FFFFFF). Имя fill_white сохранено для
        совместимости со старыми вкладками. fill_color можно менять при
        выключенной заливке: цвет сохранится для последующего включения.
        None оставляет соответствующий параметр прежним.

        Временные контуры инструмента и маркеры редактирования не скрываются:
        они необходимы для интерактивного редактирования даже при скрытых рамках.
        """
        overrides = {}
        if show_frames is not None:
            overrides['show_frames'] = bool(show_frames)
        if show_labels is not None:
            overrides['show_labels'] = bool(show_labels)
        if fill_white is not None:
            overrides['fill_white'] = bool(fill_white)
        if fill_color is not None:
            chosen = QColor(fill_color)
            if not chosen.isValid():
                raise ValueError(f'Некорректный цвет заливки: {fill_color!r}')
            # Заливка непрозрачная; alpha не влияет на исходный QPixmap.
            overrides['fill_color'] = chosen.name().upper()
        if overrides:
            self.set_display_style(**overrides)

    def set_class_color_provider(self, provider):
        """Совместимость: старый provider сохраняется, но цвета берутся
        из редактируемой палитры settings.dataset_class_palette."""
        self._class_color_provider = provider
        self.viewport().update()

    def set_class_name_provider(self, provider):
        """Необязательная функция: provider(class_id) -> текст подписи."""
        self._class_name_provider = provider
        self.viewport().update()

    def refresh_annotations(self):
        """Перерисовать слой аннотаций без перестройки базового QPixmap."""
        self.viewport().update()

    def set_selected_index(self, idx):
        if idx == self.selected_index:
            return
        self.selected_index = idx
        if self.on_selection_changed_callback:
            self.on_selection_changed_callback(idx)

        if 0 <= idx < len(self.annotations) and self.annotations[idx][0] == 'obb':
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
                      on_annotation_right_clicked=None,
                      on_log=None):
        self.on_rect_drawn_callback = on_rect_drawn
        self.on_obb_drawn_callback = on_obb_drawn
        self.on_segment_drawn_callback = on_segment_drawn
        self.on_scribble_added_callback = on_scribble_added
        self.on_display_update_callback = on_display_update
        self.on_reset_tool_callback = on_reset_tool
        self.on_annotation_modified_callback = on_annotation_modified
        self.on_selection_changed_callback = on_selection_changed
        self.on_annotation_right_clicked_callback = on_annotation_right_clicked
        self.on_log_callback = on_log

    def set_image_data(self, width, height, fg_scribbles=None, bg_scribbles=None):
        # Сохраняем старый API. Неиспользуемые scribbles — для совместимости.
        self.img_width = int(width or 0)
        self.img_height = int(height or 0)
        self._sync_scene_geometry()
        self.viewport().update()

    def _sync_scene_geometry(self):
        """Делает систему координат сцены равной пикселям *рабочего* изображения.

        numpy_to_qpixmap может уменьшать QPixmap для экономии памяти.
        Поэтому масштабируется только pixmap_item, не геометрия аннотаций.
        """
        pixmap = self.pixmap_item.pixmap()
        if pixmap.isNull():
            return
        pw, ph = pixmap.width(), pixmap.height()
        if pw <= 0 or ph <= 0:
            return
        iw = self.img_width or pw
        ih = self.img_height or ph
        self.pixmap_item.setTransform(QTransform.fromScale(iw / pw, ih / ph))
        self.setSceneRect(QRectF(0, 0, iw, ih))

    def set_pixmap(self, pixmap, preserve_view=True):
        if pixmap is None or pixmap.isNull():
            self.pixmap_item.setPixmap(pixmap if pixmap is not None else QPixmap())
            self.pixmap_item.setTransform(QTransform())
            self._last_pixmap_size = None
            self._current_pixmap = None
            self.viewport().update()
            return
        self._current_pixmap = pixmap
        new_size = (pixmap.width(), pixmap.height(),
                    self.img_width or pixmap.width(),
                    self.img_height or pixmap.height())
        self.pixmap_item.setPixmap(pixmap)
        self._sync_scene_geometry()
        if not (preserve_view and self._last_pixmap_size == new_size):
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._last_pixmap_size = new_size
        self.viewport().update()

    def reset_view(self):
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.viewport().update()

    def save_current_image(self):
        """Сохраняет видимую область просмотра ВМЕСТЕ с векторной разметкой.

        Для сохранения исходных пикселей без разметки есть метод
        save_source_image(). Функция сохраняет экранный вид, а не исходный
        размер изображения; это явно обозначено в контекстном меню.
        """
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить текущий вид с разметкой',
            f'{self._save_basename}_view.png', 'PNG Image (*.png)')
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            self.viewport().grab().save(file_path, 'PNG')

    def save_source_image(self):
        """Сохраняет базовый pixmap без векторного слоя разметки."""
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить изображение без разметки',
            f'{self._save_basename}.png', 'PNG Image (*.png)')
        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            self._current_pixmap.save(file_path, 'PNG')

    # ==================================================================
    # Вспомогательные геометрические методы
    # ==================================================================
    def _scene_per_px(self):
        """Сколько scene-единиц соответствует одному экранному пикселю
        при текущем масштабе view. 1.0 — масштаб 1:1.
        Уменьшается при приближении (zoom in), растёт при отдалении."""
        try:
            s = self.transform().m11()
        except Exception:
            return 1.0
        if s is None or not math.isfinite(s) or s <= 1e-9:
            return 1.0
        return 1.0 / s

    def _get_points_pixel(self, ann):
        """Получить вершины в координатах сцены БЕЗ округления до int.

        Так исключается накопление ошибок при последовательном редактировании.
        """
        if self.img_width <= 0 or self.img_height <= 0:
            return []
        typ = ann[0]
        try:
            if typ == 'detect':
                _, _, cx, cy, w, h = ann
                x1 = (cx - w / 2) * self.img_width
                y1 = (cy - h / 2) * self.img_height
                x2 = (cx + w / 2) * self.img_width
                y2 = (cy + h / 2) * self.img_height
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            if typ in ('obb', 'segment'):
                points = ann[2]
                if len(points) % 2 != 0:
                    return []
                return [(float(points[i]) * self.img_width,
                         float(points[i + 1]) * self.img_height)
                        for i in range(0, len(points), 2)]
        except (IndexError, TypeError, ValueError, OverflowError):
            return []
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

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2):
        """Кратчайшее расстояние от точки (px,py) до отрезка (x1,y1)-(x2,y2)."""
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        cx = x1 + t * dx
        cy = y1 + t * dy
        return math.hypot(px - cx, py - cy)

    def _hit_test_point(self, point, pos_pixel, radius=None):
        if radius is None:
            # 8 экранных пикселей, переведённых в scene-единицы.
            radius = self.HIT_POINT_PX * self._scene_per_px()
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

            spx = self._scene_per_px()
            min_side = min(w, h)

            # Постоянные 22 экранных пикселя от середины верхней стороны,
            # с ограничением сверху — иначе ручка уедет за центр мелкого OBB.
            offset = 22.0 * spx
            offset = max(offset, 4.0 * spx)          # нижняя планка
            offset = min(offset, min_side * 0.35)    # не дальше 35% от стороны

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

        spx = self._scene_per_px()
        rot_r    = self.HIT_OBB_ROTATE_PX * spx
        corner_r = self.HIT_OBB_CORNER_PX * spx
        edge_r   = self.HIT_OBB_EDGE_PX   * spx

        # 1) Ручка вращения — приоритет.
        rot = handles.get('rotate')
        if rot and abs(rot[0] - pos_pixel[0]) <= rot_r \
                and abs(rot[1] - pos_pixel[1]) <= rot_r:
            return 'rotate'

        # 2) Углы.
        for name in ('corner0', 'corner1', 'corner2', 'corner3'):
            pt = handles.get(name)
            if pt and self._hit_test_point(pt, pos_pixel, radius=corner_r):
                return name

        # 3) Середины сторон.
        for name in ('edge0', 'edge1', 'edge2', 'edge3'):
            pt = handles.get(name)
            if pt and self._hit_test_point(pt, pos_pixel, radius=edge_r):
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

        # --- Правый клик: аннотация / выход из режимов / контекстное меню ---
        if event.button() == Qt.RightButton:
            if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                self._suppress_context_menu = False
                event.ignore()
                return

            # Если курсор попал в выделенную аннотацию — показать меню
            # действий с объектом (Edit / Convert to OBB / Delete).
            # Приоритет отдаём уже выделенному объекту, но если он не под
            # курсором, разрешаем найти другой — родитель синхронизирует
            # выделение сам.
            if (self._current_pixmap is not None
                    and not self._current_pixmap.isNull()
                    and self.img_width > 0 and self.img_height > 0):
                scene_pos = self.mapToScene(event.pos())
                if (0 <= scene_pos.x() < self.img_width
                        and 0 <= scene_pos.y() < self.img_height):
                    pos_pixel = (scene_pos.x(), scene_pos.y())
                    idx_hit, _hit = self._find_hit(pos_pixel, self.selected_index)
                    if idx_hit != -1 and self.on_annotation_right_clicked_callback:
                        self._suppress_context_menu = True
                        self.on_annotation_right_clicked_callback(
                            idx_hit, event.globalPos())
                        event.accept()
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

        # --- Режим редактирования ИЛИ клик по объекту ---
        # Если инструмент рисования активен — клик обычно рисует, НО:
        #   • если клик попал по уже выделенному объекту — приоритет
        #     отдаём редактированию (удобно сразу после рисования);
        #   • если инструмент не активен — клик по объекту выделяет
        #     и включает edit-mode.
        tool_active = self.drawing_mode and self.current_tool is not None

        # «Клик по выделенному» — короткая разведка для случая, когда
        # инструмент активен, но пользователь явно тыкает в свой объект.
        click_on_selected = False
        if tool_active and not self.edit_mode and 0 <= self.selected_index < len(self.annotations):
            scene_probe = self.mapToScene(event.pos())
            if (0 <= scene_probe.x() < self.img_width
                    and 0 <= scene_probe.y() < self.img_height):
                probe_pixel = (scene_probe.x(), scene_probe.y())
                if self._hit_test_any(
                        self.annotations[self.selected_index],
                        probe_pixel) is not None:
                    click_on_selected = True

        if self.edit_mode or not tool_active or click_on_selected:
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
                if self.edit_mode:
                    # Клик вне объектов — выходим из edit-режима.
                    self.set_selected_index(-1)
                    self.set_edit_mode(False)
                    # Если инструмент рисования активен — сразу начинаем
                    # новый объект, чтобы пользователь не «терял» клик.
                    if tool_active:
                        self.drawing = False
                        self._start_drawing(event)
                    event.accept()
                    QTimer.singleShot(0, self.viewport().update)
                    return
                # Не в edit-mode и не попали в объект — отдаём panning.
                super().mousePressEvent(event)
                return

            # Нашли объект: выделяем и входим в edit-mode.
            if idx != self.selected_index:
                self.set_selected_index(idx)
            if not self.edit_mode:
                self.set_edit_mode(True)

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
                elif self.current_tool == 'segment_line':
                    self._update_segment_line(event)
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

            elif self.drawing_mode and self.drawing and self.current_tool == "segment_line":
                # Отпустили мышь — замыкаем контур на первую точку.
                event.accept()
                self._finish_segment_line()
                return

        super().mouseReleaseEvent(event)

    # ==================================================================
    # Двойной клик: удаление / вставка точек сегмента
    # ==================================================================
    def mouseDoubleClickEvent(self, event):
        # Обрабатываем только ЛКМ.
        if event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return

        # Не мешаем активному инструменту рисования.
        if self.drawing_mode and self.current_tool is not None:
            super().mouseDoubleClickEvent(event)
            return

        if self._current_pixmap is None or self._current_pixmap.isNull():
            super().mouseDoubleClickEvent(event)
            return
        if self.img_width <= 0 or self.img_height <= 0:
            super().mouseDoubleClickEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        if not (0 <= scene_pos.x() < self.img_width
                and 0 <= scene_pos.y() < self.img_height):
            super().mouseDoubleClickEvent(event)
            return

        pos_pixel = (float(scene_pos.x()), float(scene_pos.y()))
        if self._try_segment_double_click(pos_pixel):
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def _try_segment_double_click(self, pos_pixel):
        """Обрабатывает двойной клик по сегментации.

        • клик в увеличенной невидимой зоне вершины — удаляет вершину
          (если после удаления останется ≥ 3 точки);
        • клик рядом с ребром между вершинами — вставляет новую точку
          в проекцию клика на это ребро.

        Возвращает True, если что-то сделали.
        """
        # Порядок обхода: сначала выделенный объект, потом все остальные.
        order = []
        if 0 <= self.selected_index < len(self.annotations):
            order.append(self.selected_index)
        for i in range(len(self.annotations)):
            if i != self.selected_index:
                order.append(i)

        spx = self._scene_per_px()
        point_r = self.HIT_SEGMENT_POINT_DBL_PX * spx
        edge_r  = self.HIT_SEGMENT_EDGE_DBL_PX  * spx

        for idx in order:
            ann = self.annotations[idx]
            if not ann or ann[0] != 'segment':
                continue

            norm_flat = list(ann[2])
            if len(norm_flat) < 6 or len(norm_flat) % 2 != 0:
                continue
            n = len(norm_flat) // 2

            # Пиксельные координаты без округления до int — чтобы не
            # терять точность при повторных правках.
            pts_pixel = [
                (norm_flat[2 * i] * self.img_width,
                 norm_flat[2 * i + 1] * self.img_height)
                for i in range(n)
            ]

            # ----- 1) Попадание по вершине → удалить. -----
            best_i = -1
            best_d = None
            for i, pt in enumerate(pts_pixel):
                d = math.hypot(pt[0] - pos_pixel[0], pt[1] - pos_pixel[1])
                if d <= point_r and (best_d is None or d < best_d):
                    best_d = d
                    best_i = i

            if best_i != -1:
                if n <= 3:
                    if self.on_log_callback:
                        self.on_log_callback(
                            "Нельзя удалить точку: в полигоне минимум 3 вершины.")
                    return True
                new_flat = list(norm_flat)
                del new_flat[2 * best_i:2 * best_i + 2]
                self._apply_segment_change(
                    idx, new_flat,
                    f"Удалена точка #{best_i + 1} сегмента "
                    f"(осталось {len(new_flat) // 2})."
                )
                return True

            # ----- 2) Попадание по ребру → вставить точку. -----
            best_edge_i = -1
            best_edge_d = None
            best_edge_xy = None
            for i in range(n):
                x1, y1 = pts_pixel[i]
                x2, y2 = pts_pixel[(i + 1) % n]
                d = self._point_to_segment_dist(
                    pos_pixel[0], pos_pixel[1], x1, y1, x2, y2)
                if d <= edge_r and (best_edge_d is None or d < best_edge_d):
                    # Проекция клика на отрезок — чтобы точка встала
                    # ровно туда, куда указал пользователь.
                    dx = x2 - x1
                    dy = y2 - y1
                    seg_len2 = dx * dx + dy * dy
                    if seg_len2 < 1e-9:
                        nx, ny = x1, y1
                    else:
                        t = ((pos_pixel[0] - x1) * dx
                             + (pos_pixel[1] - y1) * dy) / seg_len2
                        t = max(0.0, min(1.0, t))
                        nx, ny = x1 + t * dx, y1 + t * dy
                    best_edge_d = d
                    best_edge_i = i
                    best_edge_xy = (nx, ny)

            if best_edge_i != -1:
                nx_norm = max(0.0, min(1.0, best_edge_xy[0] / self.img_width))
                ny_norm = max(0.0, min(1.0, best_edge_xy[1] / self.img_height))
                new_flat = list(norm_flat)
                pos = 2 * (best_edge_i + 1)
                new_flat.insert(pos, nx_norm)
                new_flat.insert(pos + 1, ny_norm)
                self._apply_segment_change(
                    idx, new_flat,
                    f"Добавлена точка на ребре {best_edge_i + 1} "
                    f"(всего {len(new_flat) // 2})."
                )
                return True

        return False

    def _apply_segment_change(self, idx, norm_flat, log_msg):
        """Применяет новое множество нормализованных точек сегмента
        и уведомляет родителя."""
        ann = self.annotations[idx]
        new_ann = ('segment', ann[1], list(norm_flat))
        self.annotations[idx] = new_ann
        # Синхронизируем состояние редактирования — если пользователь
        # продолжит тащить точки без переоткрытия сцены.
        self.edit_original_ann = new_ann
        self.edit_original_points = self._get_points_pixel(new_ann)
        # Если правка пришла в объект, который во view не выделен —
        # переключаем выделение и включаем edit-mode, чтобы пользователь
        # оставался в контексте редактирования.
        if self.selected_index != idx:
            self.set_selected_index(idx)
        if not self.edit_mode:
            self.set_edit_mode(True)

        if self.on_annotation_modified_callback:
            self.on_annotation_modified_callback(idx, new_ann)
        if self.on_display_update_callback:
            self.on_display_update_callback()
        if self.on_log_callback:
            self.on_log_callback(log_msg)

        QTimer.singleShot(0, self.viewport().update)


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
        save_view_action = menu.addAction('Сохранить текущий вид с разметкой')
        save_source_action = menu.addAction('Сохранить изображение без разметки')
        action = menu.exec_(event.globalPos())
        if action == save_view_action:
            self.save_current_image()
        elif action == save_source_action:
            self.save_source_image()

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
                if math.hypot(x - first[0], y - first[1]) <= self.HIT_CLOSE_POLYGON_PX * self._scene_per_px():
                    self._finish_segment()
                    return
            self.drawing = True
            QTimer.singleShot(0, self.viewport().update)

        elif self.current_tool == "segment_line":
            # Первый клик — начинаем собирать траекторию.
            self.freehand_points = [(x, y)]
            self.freehand_drawing = True
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

    def _update_segment_line(self, event):
        """Свободное рисование: накапливает точки, пока ЛКМ зажата."""
        if not self.freehand_drawing:
            return
        scene_pos = self.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        if not (0 <= x < self.img_width and 0 <= y < self.img_height):
            return

        # Фильтр по минимальному расстоянию: не плодим почти одинаковые
        # точки, когда мышь двигается медленно и Qt шлёт события часто.
        if self.freehand_points:
            lx, ly = self.freehand_points[-1]
            if math.hypot(x - lx, y - ly) < self.FREEHAND_MIN_STEP_PX:
                return

        self.freehand_points.append((x, y))
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
        # Минимум 3 вершины — иначе это не полигон, а случайный клик.
        if len(pts) < 3:
            self.temp_polygon_points = []
            self.temp_polygon_cursor = None
            self.drawing = False
            QTimer.singleShot(0, self.viewport().update)
            return

        # NEW: добиваем до минимального количества точек.
        pts = self._ensure_min_polygon_points(pts)

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

    def _ensure_min_polygon_points(self, pts):
        """Если вершин меньше MIN_SEGMENT_POINTS — вставляет середины
        на самых длинных сегментах (с учётом замыкающего ребра), пока
        точек не станет >= MIN_SEGMENT_POINTS.

        pts — список пар (x, y) в пикселях. Возвращает новый список.
        """
        pts = list(pts)
        target = int(self.MIN_SEGMENT_POINTS)
        guard = 0
        while len(pts) < target and len(pts) >= 2 and guard < 100:
            guard += 1
            n = len(pts)
            best_i = 0
            best_len = -1.0
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                d = math.hypot(x2 - x1, y2 - y1)
                if d > best_len:
                    best_len = d
                    best_i = i
            if best_len <= 0.0:
                # Все вершины совпали — расширять некуда.
                break
            x1, y1 = pts[best_i]
            x2, y2 = pts[(best_i + 1) % n]
            mid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            pts.insert(best_i + 1, mid)
        return pts

    def _finish_segment_line(self):
        """Завершает свободный сегмент: прореживает точки, добивает
        до минимального количества, нормирует, отдаёт в
        on_segment_drawn_callback. Автозамыкание контура выполняет
        cv2.polylines(isClosed=True) в родителе."""
        pts = list(self.freehand_points)
        self.freehand_points = []
        self.freehand_drawing = False
        self.drawing = False
        QTimer.singleShot(0, self.viewport().update)

        # Минимум 3 вершины — иначе это не полигон, а случайный мазок.
        if len(pts) < 3:
            return

        # Прореживание: убираем близкие точки, сохраняя форму.
        # approxPolyDP — идеальный вариант: допуск 2 пикселя сохраняет
        # изгибы, но выкидывает «дрожание» руки.
        try:
            pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            simplified = cv2.approxPolyDP(pts_np, epsilon=2.0, closed=True)
            if simplified is not None and len(simplified) >= 3:
                pts = [(int(p[0][0]), int(p[0][1])) for p in simplified]
        except Exception:
            # cv2/np могут быть недоступны в этом окружении — тогда
            # используем сырые точки. Это не критично.
            pass

        if len(pts) < 3:
            return

        # NEW: добиваем до минимального количества точек.
        pts = self._ensure_min_polygon_points(pts)

        norm_pts = []
        for px, py in pts:
            nx = max(0.0, min(1.0, px / self.img_width))
            ny = max(0.0, min(1.0, py / self.img_height))
            norm_pts.append(nx)
            norm_pts.append(ny)

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
    # ==================================================================
    # Единый Qt-векторный слой разметки / временных фигур / хендлов
    # ==================================================================
    @staticmethod
    def _color_from_bgr(bgr):
        return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))

    def _annotation_color(self, cls_id, selected=False):
        """Цвет выделения общий; обычные рамки — из палитры классов.

        Отдельный settings.colors.annotation не применяется к датасету.
        Неуказанный, некорректный или отрицательный id = класс 0.
        """
        if selected:
            return self._color_from_bgr(settings.get_color('selected'))
        return self._color_from_bgr(settings.get_dataset_class_color(cls_id))

    def _annotation_label(self, cls_id):
        if self._class_name_provider is not None:
            try:
                return str(self._class_name_provider(cls_id))
            except Exception:
                pass
        return str(cls_id)

    @staticmethod
    def _cosmetic_pen(color, width_px=1.0, style=Qt.SolidLine):
        pen = QPen(color)
        pen.setStyle(style)
        pen.setCosmetic(True)
        pen.setWidthF(max(0.1, float(width_px)))
        return pen

    def drawForeground(self, painter: QPainter, rect):
        try:
            self._draw_foreground_impl(painter, rect)
        except Exception as e:
            print(f'[SmartGraphicsViewNew.drawForeground] {type(e).__name__}: {e}')

    def _draw_foreground_impl(self, painter: QPainter, rect):
        painter.save()
        try:
            # Аннотации не рисуются в QPixmap и не меняют исходные пиксели.
            self._draw_annotations_layer(painter, rect)
            self._draw_preview_layer(painter)
            self._draw_edit_handles(painter)
            self._draw_crosshair(painter)
        finally:
            painter.restore()

    def _draw_annotations_layer(self, painter, visible_rect):
        """Три независимых слоя: заливка -> контуры -> метки.

        Отрисовка в три прохода нужна, чтобы заливка соседнего объекта
        не перекрывала уже нарисованные рамки и подписи других объектов.
        При полном отключении визуальных элементов объекты по-прежнему
        доступны для hit-testing и редактирования.
        """
        if self.img_width <= 0 or self.img_height <= 0 or not self.annotations:
            return
        style = self.display_style
        if not (style.fill_white or style.fill_annotations or
                style.show_frames or (style.show_labels and style.label_mode != 'none')):
            return

        # Выделенный объект рисуем поверх остальных (для рамки и подписи).
        indices = [i for i in range(len(self.annotations)) if i != self.selected_index]
        if 0 <= self.selected_index < len(self.annotations):
            indices.append(self.selected_index)

        visible = []
        for idx in indices:
            ann = self.annotations[idx]
            if not ann or ann[0] not in ('detect', 'obb', 'segment'):
                continue
            points = self._get_points_pixel(ann)
            if len(points) < 3:
                continue
            try:
                poly = QPolygonF([QPointF(x, y) for x, y in points])
                bounds = poly.boundingRect().normalized()
            except (OverflowError, TypeError, ValueError):
                continue
            if not bounds.intersects(visible_rect):
                continue
            selected = idx == self.selected_index
            try:
                cls_id = int(ann[1])
            except (TypeError, ValueError, OverflowError):
                cls_id = 0
            if cls_id < 0:
                cls_id = 0
            visible.append((ann[0], poly, bounds, cls_id, selected))

        # 1. Заливаем геометрию выбранным непрозрачным цветом.
        # Qt-перерисовка не затрагивает QPixmap / оригинальные пиксели.
        if style.fill_white or style.fill_annotations:
            painter.setPen(Qt.NoPen)
            for typ, poly, bounds, cls_id, selected in visible:
                if style.fill_white:
                    fill = QColor(style.fill_color)
                    fill.setAlpha(255)
                else:
                    fill = self._annotation_color(cls_id, selected)
                    fill.setAlpha(max(0, min(255, int(style.fill_alpha))))
                painter.setBrush(QBrush(fill))
                if typ == 'detect':
                    painter.drawRect(bounds)
                else:
                    painter.drawPolygon(poly)

        # 2. Рамки и контуры отключаются независимо от заливки и меток.
        if style.show_frames:
            painter.setBrush(Qt.NoBrush)
            for typ, poly, bounds, cls_id, selected in visible:
                color = self._annotation_color(cls_id, selected)
                width = (style.selected_line_width_px if selected
                         else style.line_width_px)
                painter.setPen(self._cosmetic_pen(color, width))
                if typ == 'detect':
                    painter.drawRect(bounds)
                else:
                    painter.drawPolygon(poly)

        # 3. Метки всегда поверх заливок/контуров. Дополнительно применяется
        # существующий label_mode (auto / all / selected / none).
        if style.show_labels:
            for typ, poly, bounds, cls_id, selected in visible:
                if self._label_is_visible(bounds, selected):
                    self._paint_screen_label(
                        painter, self._annotation_label(cls_id), bounds,
                        self._color_from_bgr(settings.get_color('label_text')),
                        self._annotation_color(cls_id, selected=False),
                    )

    def _label_is_visible(self, bounds, selected):
        style = self.display_style
        if style.label_mode == 'none':
            return False
        if style.label_mode == 'selected':
            return selected
        if style.label_mode == 'all' or selected:
            return True
        scale = 1.0 / self._scene_per_px()
        return (bounds.width() * scale >= style.auto_label_min_width_px
                and bounds.height() * scale >= style.auto_label_min_height_px)

    def _paint_screen_label(self, painter, label, bounds, color, background_color=None):
        style = self.display_style
        spx = self._scene_per_px()
        font = QFont()
        font.setPixelSize(style.font_size_px)
        fm = QFontMetricsF(font)
        padding = max(0.0, style.label_padding_px)
        gap = max(0.0, style.label_gap_px)
        # Размер плашки в экранных px, до масштабирования painter.
        text_width = fm.horizontalAdvance(label)
        width_px = text_width + 2.0 * padding
        height_px = fm.height() + 2.0 * padding
        vw = self.viewport().width()
        vh = self.viewport().height()
        if vw <= 0 or vh <= 0:
            return
        # Позиции сравниваем во VIEW-пикселях, не в пикселях изображения.
        top_left = self.mapFromScene(bounds.topLeft())
        bottom_left = self.mapFromScene(bounds.bottomLeft())
        x_view = max(0.0, min(float(top_left.x()), max(0.0, vw - width_px)))
        top_y_view = float(top_left.y()) - height_px - gap
        below_y_view = float(bottom_left.y()) + gap
        if top_y_view >= 0:
            y_view = top_y_view
        elif below_y_view + height_px <= vh:
            y_view = below_y_view
        else:
            # При отсутствии места используем минимальное перекрытие.
            y_view = max(0.0, min(top_y_view, max(0.0, vh - height_px)))
        # mapToScene учитывает панорамирование и зум, а painter.scale(1/zoom)
        # гарантирует неизменный физический размер текста/подложки.
        start_scene = self.mapToScene(int(round(x_view)), int(round(y_view)))
        painter.save()
        try:
            painter.translate(start_scene)
            painter.scale(spx, spx)
            if style.label_background_alpha > 0:
                # Плашка датасета окрашивается по классу, как в старом
                # draw_class_label(); текст — независимый colors.label_text.
                # Для других вызовов сохраняется контрастный fallback.
                if background_color is not None:
                    bg = QColor(background_color)
                else:
                    luminance = (0.2126 * color.red() + 0.7152 * color.green()
                                 + 0.0722 * color.blue())
                    bg = QColor(255, 255, 255) if luminance < 140 else QColor(0, 0, 0)
                bg.setAlpha(max(0, min(255, style.label_background_alpha)))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(bg))
                painter.drawRect(QRectF(0, 0, width_px, height_px))
            painter.setBrush(Qt.NoBrush)
            painter.setFont(font)
            painter.setPen(self._cosmetic_pen(color, 1.0))
            painter.drawText(QPointF(padding, padding + fm.ascent()), label)
        finally:
            painter.restore()

    def _draw_preview_layer(self, painter):
        s = self.display_style
        spx = self._scene_per_px()
        green = self._color_from_bgr(settings.get_color('crosshair'))
        selected_color = self._color_from_bgr(settings.get_color('selected'))
        green_pen = self._cosmetic_pen(green, s.preview_line_width_px)
        painter.setPen(green_pen)
        painter.setBrush(Qt.NoBrush)

        # detect: перетаскивание прямоугольника
        if self.drawing and self.start_point is not None and self.current_point is not None \
                and self.current_tool == 'rect':
            x1, y1 = self.start_point
            x2, y2 = self.current_point
            painter.drawRect(QRectF(min(x1, x2), min(y1, y2),
                                    abs(x2-x1), abs(y2-y1)))

        # segment: клики по вершинам
        if self.current_tool == 'segment' and self.temp_polygon_points:
            pts = [QPointF(x, y) for x, y in self.temp_polygon_points]
            if len(pts) >= 2:
                painter.drawPolyline(QPolygonF(pts))
            if self.temp_polygon_cursor is not None:
                painter.drawLine(pts[-1], QPointF(*self.temp_polygon_cursor))
            painter.setBrush(QBrush(green))
            for p in pts:
                painter.drawEllipse(p, 3.0 * spx, 3.0 * spx)
            if len(pts) >= 3:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(self._cosmetic_pen(selected_color, s.preview_line_width_px))
                painter.drawEllipse(pts[0], 7.0 * spx, 7.0 * spx)

        # segment_line: свободная линия
        if self.current_tool == 'segment_line' and self.freehand_points:
            pts = [QPointF(x, y) for x, y in self.freehand_points]
            painter.setPen(green_pen)
            painter.setBrush(Qt.NoBrush)
            if len(pts) >= 2:
                painter.drawPolyline(QPolygonF(pts))
                painter.setPen(self._cosmetic_pen(QColor(green.red(), green.green(), green.blue(), 160),
                                                  s.preview_line_width_px, Qt.DashLine))
                painter.drawLine(pts[-1], pts[0])
            painter.setPen(self._cosmetic_pen(selected_color, s.preview_line_width_px))
            painter.drawEllipse(pts[0], 5.0 * spx, 5.0 * spx)

        # OBB: 1-й и 2-й этапы, preview векторный
        if self.current_tool == 'obb':
            if self.obb_draw_stage == 1 and self.obb_draw_p1 is not None \
                    and self.obb_draw_cursor is not None:
                p1 = QPointF(*self.obb_draw_p1)
                p2 = QPointF(*self.obb_draw_cursor)
                painter.setPen(self._cosmetic_pen(green, s.preview_line_width_px, Qt.DashLine))
                painter.drawLine(p1, p2)
                painter.setBrush(QBrush(green))
                painter.drawEllipse(p1, 4.0*spx, 4.0*spx)
                painter.drawEllipse(p2, 4.0*spx, 4.0*spx)
            elif self.obb_draw_stage == 2:
                params = self._compute_obb_preview_params()
                if params is not None:
                    cx, cy, width, height, angle = params
                    pts = self._obb_to_points(cx, cy, width, height, angle)
                    if len(pts) == 4:
                        painter.setPen(green_pen)
                        painter.setBrush(Qt.NoBrush)
                        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
                        if self.obb_draw_cursor is not None:
                            painter.setBrush(QBrush(selected_color))
                            painter.drawEllipse(QPointF(*self.obb_draw_cursor), 4.0*spx, 4.0*spx)

        # Зарезервированные подсказки fg/bg для старых обработчиков.
        if self.drawing and self.current_tool in ('fg', 'bg') and self.temp_scribble:
            x, y, r = self.temp_scribble
            col = QColor(0, 255, 0) if self.current_tool == 'fg' else QColor(0, 0, 255)
            painter.setPen(self._cosmetic_pen(col, s.preview_line_width_px))
            painter.setBrush(QBrush(col))
            painter.drawEllipse(QPointF(x, y), r*spx, r*spx)

    def _draw_edit_handles(self, painter):
        if not self.edit_mode or not (0 <= self.selected_index < len(self.annotations)):
            return
        ann = self.annotations[self.selected_index]
        typ = ann[0]
        points = self._get_points_pixel(ann)
        if not points:
            return
        spx = self._scene_per_px()
        outline = self.display_style.handle_outline_width_px
        edit_color = self._color_from_bgr(settings.get_color('edit_points'))
        painter.setPen(self._cosmetic_pen(edit_color, outline))
        painter.setBrush(QBrush(edit_color))

        if typ in ('detect', 'segment'):
            xs = [p[0] for p in points]; ys = [p[1] for p in points]
            min_side = min(max(xs)-min(xs), max(ys)-min(ys))
            target = self.DRAW_DETECT_POINT_PX if typ == 'detect' else self.DRAW_SEGMENT_POINT_PX
            r = min(target*spx, max(0., min_side)*self.MARKER_MAX_FRACTION)
            r = max(r, 2.0*spx)
            for x,y in points:
                painter.drawEllipse(QPointF(x,y), r,r)
            return

        if typ != 'obb':
            return
        handles = self._get_obb_handles(points)
        if not handles:
            return
        cx, cy, w, h, _ = self._obb_to_params(points)
        min_side = max(1.0,min(w,h))
        corner_half = max(2.0*spx,min(self.DRAW_OBB_CORNER_PX*spx,
                                      min_side*self.MARKER_MAX_FRACTION))
        for name in ('corner0','corner1','corner2','corner3'):
            pt = handles.get(name)
            if pt is not None:
                painter.setPen(self._cosmetic_pen(edit_color, outline))
                painter.setBrush(QBrush(edit_color))
                painter.drawRect(QRectF(pt[0]-corner_half,pt[1]-corner_half,
                                        2*corner_half,2*corner_half))
        edge_r = max(1.5*spx,min(self.DRAW_OBB_EDGE_PX*spx,
                                 min_side*self.MARKER_MAX_FRACTION))
        for name in ('edge0','edge1','edge2','edge3'):
            pt=handles.get(name)
            if pt is not None:
                painter.setPen(self._cosmetic_pen(edit_color, outline))
                painter.setBrush(QBrush(edit_color.lighter(125)))
                painter.drawEllipse(QPointF(*pt),edge_r,edge_r)
        rot=handles.get('rotate')
        if rot is not None:
            rot_r=max(4.0*spx,min(self.DRAW_OBB_ROTATE_PX*spx,min_side*0.4))
            connector = QColor(edit_color)
            connector.setAlpha(100)
            painter.setPen(self._cosmetic_pen(connector, outline, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(QPointF(cx,cy),QPointF(*rot))
            painter.setPen(self._cosmetic_pen(edit_color, outline))
            rot_fill = QColor(edit_color)
            rot_fill.setAlpha(140)
            painter.setBrush(QBrush(rot_fill))
            painter.drawEllipse(QPointF(*rot),rot_r,rot_r)
            painter.setPen(self._cosmetic_pen(edit_color.lighter(160), outline))
            painter.drawArc(QRectF(rot[0]-rot_r*.55,rot[1]-rot_r*.55,
                                   1.1*rot_r,1.1*rot_r),30*16,120*16)

    def _draw_crosshair(self,painter):
        if not (self.drawing_mode and not self.edit_mode and self.cursor_scene_pos is not None):
            return
        x,y=self.cursor_scene_pos
        color=self._color_from_bgr(settings.get_color('crosshair'))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self._cosmetic_pen(color,1.0))
        painter.drawLine(QPointF(0,y),QPointF(self.img_width,y))
        painter.drawLine(QPointF(x,0),QPointF(x,self.img_height))


# Совместимый с прежними импортами псевдоним для потребителей новой утилиты.
SmartGraphicsView = SmartGraphicsViewNew

def numpy_to_qpixmap(img_bgr, max_side=8192):
    """Display-only conversion. Geometry stays in source-image coordinates."""
    from PyQt5.QtGui import QImage
    if img_bgr is None:
        return QPixmap()
    img = np.asarray(img_bgr)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = float(max_side) / max(h, w)
        img = cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))),
                         interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        from utils_new.dataset_image_stats_new import ImageStats
        img = ImageStats.to_uint8(img)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        fmt = QImage.Format_Grayscale8
        stride = w
    elif img.ndim == 3 and img.shape[2] == 3:
        fmt = QImage.Format_BGR888
        stride = 3*w
    elif img.ndim == 3 and img.shape[2] == 4:
        fmt = QImage.Format_ARGB32
        stride = 4*w
    else:
        raise ValueError(f'Unsupported display shape: {img.shape}')
    # Deep-copy QImage: numpy data must not be freed while Qt holds the pixels.
    qimage = QImage(img.data, w, h, stride, fmt).copy()
    return QPixmap.fromImage(qimage)
