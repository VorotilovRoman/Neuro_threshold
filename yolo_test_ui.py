# yolo_test_ui.py
"""
UI-слой приложения YOLO Inspect + AE/VAE.

Содержит:
  • cv_to_qpixmap — конвертер numpy→QPixmap;
  • ZoomableImageView — виджет с зумом/панорамой/центрированием;
  • YoloInspectWindow — базовое окно: разметка UI, загрузка моделей
    (YOLO и AE/VAE), загрузка изображений и папок, навигация,
    сохранение снимков и карты ошибок.

Вся вычислительная логика (модели AE/VAE, инференс YOLO, применение
AE/VAE, авто/ручные трансформации) — в yolo_test.py.
"""
import os
import sys
import contextlib

# Гасим предупреждения OpenCV (в т.ч. TIFFReadDirectory: Unknown field ...)
# ДО импорта cv2, иначе переменная не подхватится.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

# Дополнительно — программный уровень (на случай, если env-переменной мало).
try:
    cv2.setLogLevel(0)  # cv2.LOG_LEVEL_SILENT
except Exception:
    pass

import numpy as np
import torch
from ultralytics import YOLO

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QVBoxLayout, QHBoxLayout, QSplitter, QTextEdit,
    QScrollArea, QSizePolicy, QMessageBox, QListWidget, QListWidgetItem,
    QCheckBox, QSlider,
)


# ============================================================
# Утилиты
# ============================================================
@contextlib.contextmanager
def _suppress_stderr():
    """
    Временно подменяет fd 2 на /dev/null.
    Нужно, чтобы libtiff не сыпал 'TIFFReadDirectory: Unknown field...'
    через нативный stderr в обход логгера OpenCV.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    try:
        saved = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved, stderr_fd)
        except OSError:
            pass
        os.close(saved)
        os.close(devnull)


def cv_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    if img_bgr is None:
        return QPixmap()
    if len(img_bgr.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ============================================================
# Виджет с зумом и панорамой
# ============================================================
class ZoomableImageView(QWidget):
    """
    Отображение QPixmap с поддержкой:
      • зум колесом мыши (к позиции курсора);
      • панорамирование перетаскиванием ЛКМ или СКМ;
      • центрирование вида по клику СКМ (без перетаскивания);
      • двойной клик — тоже центрирование;
      • автоподгонка масштаба под размер виджета.

    Реализация безопасна для Qt5/PyQt5:
      • в wheelEvent используется pos()/position(), но не localPos();
      • в paintEvent рисуется только видимая часть pixmap — без
        аллокации огромных промежуточных буферов при сильном зуме.
    """

    def __init__(self, parent=None, placeholder_text=""):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(1, 1)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        self._pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._placeholder = placeholder_text

        self._panning = False
        self._pan_start_pos = None
        self._pan_start_offset = None

        self._mid_active = False
        self._mid_press_pos = None
        self._mid_moved = False
        self._mid_start_offset = None

        self._bg_color = QColor(30, 30, 30)
        self._fg_color = QColor(170, 170, 170)

        self._min_scale = 0.02
        self._max_scale = 32.0
        self._zoom_step = 1.15

    # -------- публичный API --------
    def has_image(self) -> bool:
        return self._pixmap is not None and not self._pixmap.isNull()

    def set_image_pixmap(self, pixmap: QPixmap, reset_view: bool = True):
        old_size = None
        if self._pixmap is not None and not self._pixmap.isNull():
            old_size = (self._pixmap.width(), self._pixmap.height())

        self._pixmap = pixmap

        new_size = None
        if pixmap is not None and not pixmap.isNull():
            new_size = (pixmap.width(), pixmap.height())

        if reset_view or old_size != new_size:
            self._scale = 1.0
            self._offset = QPointF(0.0, 0.0)

        self.update()

    def clear_image(self):
        self._pixmap = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.update()

    def set_placeholder(self, text: str):
        self._placeholder = text
        if not self.has_image():
            self.update()

    def reset_view(self):
        """Центрирование: сброс зума и панорамы к виду «вписать в окно»."""
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.update()

    # -------- внутреннее --------
    def _base_scale(self) -> float:
        if not self.has_image():
            return 1.0
        pw = self._pixmap.width()
        ph = self._pixmap.height()
        if pw <= 0 or ph <= 0:
            return 1.0
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return 1.0
        return min(w / pw, h / ph)

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), self._bg_color)

            if not self.has_image():
                painter.setPen(self._fg_color)
                painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
                return

            W = self.width()
            H = self.height()
            if W <= 0 or H <= 0:
                return

            base = self._base_scale()
            total = base * self._scale
            if total <= 0.0 or not np.isfinite(total):
                return

            pw = self._pixmap.width()
            ph = self._pixmap.height()
            sw = pw * total
            sh = ph * total

            x = (W - sw) / 2.0 + self._offset.x()
            y = (H - sh) / 2.0 + self._offset.y()

            vx1 = max(0.0, x)
            vy1 = max(0.0, y)
            vx2 = min(float(W), x + sw)
            vy2 = min(float(H), y + sh)
            if vx2 <= vx1 or vy2 <= vy1:
                return

            sx1 = (vx1 - x) / total
            sy1 = (vy1 - y) / total
            sx2 = (vx2 - x) / total
            sy2 = (vy2 - y) / total

            target_rect = QRectF(vx1, vy1, vx2 - vx1, vy2 - vy1)
            source_rect = QRectF(sx1, sy1, sx2 - sx1, sy2 - sy1)

            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawPixmap(target_rect, self._pixmap, source_rect)
        finally:
            painter.end()

    # -------- события --------
    def _event_pos_f(self, event) -> QPointF:
        """Возвращает позицию курсора в координатах виджета.
        Совместимо с Qt5 (pos()) и Qt6 (position())."""
        if hasattr(event, "position"):
            try:
                return QPointF(event.position())
            except Exception:
                pass
        return QPointF(event.pos())

    def wheelEvent(self, event):
        if not self.has_image():
            event.ignore()
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        pos = self._event_pos_f(event)

        factor = self._zoom_step if delta > 0 else (1.0 / self._zoom_step)
        old_scale = self._scale
        new_scale = max(self._min_scale, min(self._max_scale, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-9:
            event.accept()
            return

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        base = self._base_scale()
        old_total = base * old_scale
        new_total = base * new_scale
        if old_total <= 0.0:
            event.accept()
            return

        px_off = (pos.x() - cx - self._offset.x()) / old_total
        py_off = (pos.y() - cy - self._offset.y()) / old_total

        new_off_x = pos.x() - cx - px_off * new_total
        new_off_y = pos.y() - cy - py_off * new_total

        self._scale = new_scale
        self._offset = QPointF(new_off_x, new_off_y)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if not self.has_image():
            event.ignore()
            return

        if event.button() == Qt.LeftButton:
            self._panning = True
            self._pan_start_pos = self._event_pos_f(event)
            self._pan_start_offset = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

        elif event.button() == Qt.MiddleButton:
            self._mid_active = True
            self._mid_moved = False
            self._mid_press_pos = self._event_pos_f(event)
            self._mid_start_offset = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.has_image():
            return

        pos = self._event_pos_f(event)

        if self._panning and self._pan_start_pos is not None:
            delta = pos - self._pan_start_pos
            self._offset = self._pan_start_offset + delta
            self.update()

        if self._mid_active and self._mid_press_pos is not None:
            delta = pos - self._mid_press_pos
            if delta.manhattanLength() > 4:
                self._mid_moved = True
            self._offset = self._mid_start_offset + delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()

        elif event.button() == Qt.MiddleButton:
            if self._mid_active and not self._mid_moved:
                self.reset_view()
            self._mid_active = False
            self._mid_press_pos = None
            self._mid_start_offset = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()

        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.has_image():
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()


# ============================================================
# Базовое окно приложения
# ============================================================
class YoloInspectWindow(QMainWindow):
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Inspect + AE/VAE")
        self.resize(1500, 900)

        self.model = None
        self.model_path = ""

        # --- AE/VAE ---
        self.ae_model = None
        self.ae_cfg = None
        self.ae_model_path = ""
        self.ae_device = torch.device("cpu")
        self.ae_arch = "conv"
        self._ae_ckpt_threshold = None
        self._ae_ckpt_percentile = None

        self._ae_err_hw = None
        self._ae_valid_region = None
        self._ae_base_bgr = None
        self._ae_threshold = float("nan")

        self._error_pixmap = None
        self._error_bgr = None
        self._updating_objects_list = False

        self.image_paths = []
        self.current_index = -1

        self.original_image = None
        self.displayed_image = None
        self.display_mode = "original"
        self.current_detections = []
        self.annotated_image = None       # результат r.plot(), без трансформаций
        self._manual_ops = []             # накопленные ручные ops

        self.current_pixmap = None

        self._build_ui()

    # --------------------------------------------------------
    # Построение UI
    # --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ==== YOLO ====
        top = QHBoxLayout()

        grp_model = QGroupBox("Модель YOLO")
        m_lay = QHBoxLayout(grp_model)
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("Путь к .pt-файлу модели…")
        self.btn_load_model = QPushButton("Загрузить модель")
        m_lay.addWidget(self.model_edit, 1)
        m_lay.addWidget(self.btn_load_model)
        top.addWidget(grp_model, 3)

        grp_params = QGroupBox("Параметры инференса YOLO")
        p_lay = QHBoxLayout(grp_params)

        p_lay.addWidget(QLabel("conf:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.75)
        p_lay.addWidget(self.conf_spin)

        p_lay.addWidget(QLabel("iou:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setValue(0.45)
        p_lay.addWidget(self.iou_spin)

        p_lay.addWidget(QLabel("imgsz:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        p_lay.addWidget(self.imgsz_spin)

        p_lay.addWidget(QLabel("device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "0", "mps"])
        p_lay.addWidget(self.device_combo)

        # Новый чекбокс — рисуется ли разметка YOLO на снимке.
        self.chk_show_boxes = QCheckBox("Показывать боксы")
        self.chk_show_boxes.setChecked(False)
        self.chk_show_boxes.setToolTip(
            "Если выключено — снимок показывается без разметки YOLO\n"
            "(боксы/сегментация/метки классов). Авто-обрезка и доворот\n"
            "продолжают работать по найденным объектам."
        )
        p_lay.addWidget(self.chk_show_boxes)

        p_lay.addStretch()
        top.addWidget(grp_params, 4)
        root.addLayout(top)

        # ==== AE/VAE ====
        ae_row = QHBoxLayout()

        grp_ae_model = QGroupBox("Модель AE/VAE")
        ae_m = QHBoxLayout(grp_ae_model)
        self.ae_model_edit = QLineEdit()
        self.ae_model_edit.setReadOnly(True)
        self.ae_model_edit.setPlaceholderText("Путь к .pt-файлу модели AE/VAE…")
        self.btn_load_ae = QPushButton("Загрузить AE/VAE")
        self.btn_clear_ae = QPushButton("Очистить AE/VAE")
        self.btn_clear_ae.setStyleSheet(
            "QPushButton { background-color: #c0392b; color: white; font-weight: 600; }"
        )
        self.btn_clear_ae.setToolTip("Выгрузить текущую модель AE/VAE и убрать карту ошибок.")
        self.btn_apply_ae = QPushButton("Применить к снимку")
        self.btn_apply_ae.setStyleSheet(
            "QPushButton { background-color: #8e44ad; color: white; font-weight: 600; }"
        )
        ae_m.addWidget(self.ae_model_edit, 1)
        ae_m.addWidget(self.btn_load_ae)
        ae_m.addWidget(self.btn_clear_ae)
        ae_m.addWidget(self.btn_apply_ae)
        # Кнопка сохранения карты ошибок создаётся позже — в блоке авто-действий.
        ae_row.addWidget(grp_ae_model, 3)

        grp_ae_params = QGroupBox("Параметры AE/VAE")
        ae_p = QHBoxLayout(grp_ae_params)

        ae_p.addWidget(QLabel("Порог, перцентиль:"))
        self.ae_perc_spin = QDoubleSpinBox()
        self.ae_perc_spin.setRange(50.0, 100.0)
        self.ae_perc_spin.setSingleStep(0.5)
        self.ae_perc_spin.setDecimals(1)
        self.ae_perc_spin.setValue(99.0)
        ae_p.addWidget(self.ae_perc_spin)

        self.chk_ae_use_ckpt_thr = QCheckBox("Порог из чекпойнта")
        self.chk_ae_use_ckpt_thr.setChecked(True)
        self.chk_ae_use_ckpt_thr.setToolTip(
            "Если в чекпойнте есть готовый порог (threshold_value),\n"
            "использовать его. Иначе — перцентиль слева."
        )
        ae_p.addWidget(self.chk_ae_use_ckpt_thr)

        self.chk_ae_overlay = QCheckBox("Наложить на снимок")
        self.chk_ae_overlay.setChecked(True)
        ae_p.addWidget(self.chk_ae_overlay)

        self.chk_ae_show_mask = QCheckBox("Подсветить > порога")
        self.chk_ae_show_mask.setChecked(False)
        ae_p.addWidget(self.chk_ae_show_mask)

        ae_p.addWidget(QLabel("Прозрачность:"))
        self.ae_alpha_slider = QSlider(Qt.Horizontal)
        self.ae_alpha_slider.setRange(0, 100)
        self.ae_alpha_slider.setValue(50)
        self.ae_alpha_slider.setFixedWidth(120)
        self.ae_alpha_slider.setToolTip(
            "0 — карта полностью видна,\n"
            "100 — карта невидима."
        )
        ae_p.addWidget(self.ae_alpha_slider)

        self.ae_alpha_label = QLabel("50")
        self.ae_alpha_label.setFixedWidth(28)
        ae_p.addWidget(self.ae_alpha_label)

        ae_p.addStretch()
        ae_row.addWidget(grp_ae_params, 4)
        root.addLayout(ae_row)

        # ==== Навигация ====
        nav = QHBoxLayout()
        self.btn_load_files = QPushButton("Загрузить файлы…")
        self.btn_load_folder = QPushButton("Загрузить папку…")
        nav.addWidget(self.btn_load_files)
        nav.addWidget(self.btn_load_folder)

        nav.addSpacing(16)

        self.btn_prev = QPushButton("◀ Предыдущий")
        self.btn_next = QPushButton("Следующий ▶")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)

        nav.addSpacing(16)

        self.image_info_label = QLabel("Изображения не загружены")
        self.image_info_label.setStyleSheet("QLabel { color: #555; }")
        nav.addWidget(self.image_info_label, 1)

        root.addLayout(nav)

        # ==== Центральная часть ====
        splitter = QSplitter(Qt.Horizontal)

        self.image_label = ZoomableImageView(placeholder_text="Изображение не загружено")
        self.image_label.setToolTip(
            "Колесо мыши — зум к курсору\n"
            "ЛКМ/СКМ drag — панорама\n"
            "СКМ клик — центрировать\n"
            "Двойной клик — центрировать"
        )

        self.error_label = ZoomableImageView(placeholder_text="Карта ошибок (AE/VAE)")
        self.error_label.setToolTip(
            "Колесо мыши — зум к курсору\n"
            "ЛКМ/СКМ drag — панорама\n"
            "СКМ клик — центрировать\n"
            "Двойной клик — центрировать"
        )
        self.error_label.setVisible(False)

        self._img_splitter = QSplitter(Qt.Horizontal)
        self._img_splitter.addWidget(self.image_label)
        self._img_splitter.addWidget(self.error_label)
        self._img_splitter.setSizes([500, 500])

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self._img_splitter)
        splitter.addWidget(self.scroll)

        # --- Правый блок ---
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(540)
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(0, 0, 0, 0)
        r_lay.setSpacing(6)

        grp_obj = QGroupBox("Найденные объекты")
        ol = QVBoxLayout(grp_obj)
        self.objects_list = QListWidget()
        self.objects_list.setSelectionMode(QListWidget.SingleSelection)
        self.objects_list.setToolTip(
            "Выберите объект для авто-обрезки/доворота.\n"
            "Если ничего не выбрано — берётся самый уверенный."
        )
        ol.addWidget(self.objects_list)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Отступ (px):"))
        self.crop_padding_spin = QSpinBox()
        self.crop_padding_spin.setRange(-500, 500)
        self.crop_padding_spin.setValue(0)
        self.crop_padding_spin.setToolTip(
            "Положительное значение расширяет область объекта,\n"
            "отрицательное — сужает её."
        )
        pad_row.addWidget(self.crop_padding_spin)
        pad_row.addStretch()
        ol.addLayout(pad_row)

        r_lay.addWidget(grp_obj, 2)

        grp_act = QGroupBox("Авто-действия с изображением")
        al = QVBoxLayout(grp_act)

        self.chk_auto_crop = QCheckBox("✂ Авто-обрезка по объекту")
        self.chk_auto_crop.setChecked(False)
        self.chk_auto_crop.setToolTip(
            "Автоматически обрезать снимок по границам выбранного объекта.\n"
            "Применяется при загрузке нового снимка и при изменении отступа."
        )
        al.addWidget(self.chk_auto_crop)

        self.chk_auto_rotate = QCheckBox("🔄 Авто-доворот объекта")
        self.chk_auto_rotate.setChecked(False)
        self.chk_auto_rotate.setToolTip(
            "Автоматически довернуть снимок так, чтобы длинная сторона объекта\n"
            "стала горизонтальной. Если включена и авто-обрезка — обрежет\n"
            "уже по выровненному объекту."
        )
        al.addWidget(self.chk_auto_rotate)

        manual_label = QLabel("Ручные трансформации:")
        manual_label.setStyleSheet("QLabel { color: #666; margin-top: 4px; }")
        al.addWidget(manual_label)

        row1 = QHBoxLayout()
        self.btn_rot_cw = QPushButton("↻ 90° по часовой")
        self.btn_rot_ccw = QPushButton("↺ 90° против часовой")
        row1.addWidget(self.btn_rot_cw)
        row1.addWidget(self.btn_rot_ccw)
        al.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_flip_v = QPushButton("⇅ Отразить по вертикали")
        self.btn_flip_h = QPushButton("⇆ Отразить по горизонтали")
        row2.addWidget(self.btn_flip_v)
        row2.addWidget(self.btn_flip_h)
        al.addLayout(row2)

        al.addSpacing(4)

        self.btn_reset = QPushButton("Сбросить (показать оригинал)")
        al.addWidget(self.btn_reset)

        # Сохранение снимка и карты ошибок — друг под другом.
        self.btn_save = QPushButton("💾 Сохранить изображение…")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: 600; }"
        )
        al.addWidget(self.btn_save)

        self.btn_save_errmap = QPushButton("🗺 Сохранить карту ошибок…")
        self.btn_save_errmap.setStyleSheet(
            "QPushButton { background-color: #16a085; color: white; font-weight: 600; }"
        )
        self.btn_save_errmap.setToolTip(
            "Сохранить текущую карту ошибок AE/VAE в файл."
        )
        al.addWidget(self.btn_save_errmap)

        r_lay.addWidget(grp_act)

        r_lay.addWidget(QLabel("Лог:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        r_lay.addWidget(self.log, 1)

        splitter.addWidget(right)
        splitter.setSizes([1000, 460])
        root.addWidget(splitter, 1)

        self.statusBar().showMessage("Готов")

        # ==== Сигналы ====
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_ae.clicked.connect(self.load_ae_model)
        self.btn_clear_ae.clicked.connect(self.clear_ae_model)
        self.btn_apply_ae.clicked.connect(lambda: self.apply_ae(silent=False))
        self.btn_save_errmap.clicked.connect(self.save_error_map)
        self.btn_load_files.clicked.connect(self.load_files)
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_rot_cw.clicked.connect(lambda: self.manual_transform("rot_cw"))
        self.btn_rot_ccw.clicked.connect(lambda: self.manual_transform("rot_ccw"))
        self.btn_flip_v.clicked.connect(lambda: self.manual_transform("flip_v"))
        self.btn_flip_h.clicked.connect(lambda: self.manual_transform("flip_h"))
        self.btn_reset.clicked.connect(self.reset_view)
        self.btn_save.clicked.connect(self.save_image)

        self.chk_auto_crop.toggled.connect(self._on_auto_transform_changed)
        self.chk_auto_rotate.toggled.connect(self._on_auto_transform_changed)
        self.crop_padding_spin.valueChanged.connect(self._on_padding_changed)

        # Тумблер отображения разметки YOLO — обработчик в yolo_test.py.
        self.chk_show_boxes.toggled.connect(self._on_show_boxes_changed)

        # YOLO-параметры: при изменении — перезапуск инференса
        # (обработчик с debounce определён в yolo_test.py).
        self.conf_spin.valueChanged.connect(self._on_yolo_params_changed)
        self.iou_spin.valueChanged.connect(self._on_yolo_params_changed)
        self.imgsz_spin.valueChanged.connect(self._on_yolo_params_changed)
        self.device_combo.currentTextChanged.connect(self._on_yolo_params_changed)

        self.objects_list.currentRowChanged.connect(self._on_object_selection_changed)

        self.chk_ae_overlay.toggled.connect(self._on_ae_render_params_changed)
        self.chk_ae_show_mask.toggled.connect(self._on_ae_render_params_changed)
        self.ae_alpha_slider.valueChanged.connect(self._on_ae_alpha_changed)
        self.chk_ae_use_ckpt_thr.toggled.connect(self._on_ae_threshold_params_changed)
        self.ae_perc_spin.valueChanged.connect(self._on_ae_threshold_params_changed)

        self._update_nav_buttons()
        self._update_actions_enabled()

    # --------------------------------------------------------
    # Вспомогательные методы UI
    # --------------------------------------------------------
    def _log(self, msg: str):
        self.log.append(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_nav_buttons(self):
        n = len(self.image_paths)
        self.btn_prev.setEnabled(n > 0 and self.current_index > 0)
        self.btn_next.setEnabled(n > 0 and self.current_index < n - 1)

    def _update_actions_enabled(self):
        has_image = self.displayed_image is not None
        self.btn_save.setEnabled(has_image)
        self.btn_rot_cw.setEnabled(has_image)
        self.btn_rot_ccw.setEnabled(has_image)
        self.btn_flip_v.setEnabled(has_image)
        self.btn_flip_h.setEnabled(has_image)
        self.btn_reset.setEnabled(has_image and self.display_mode != "original")
        self.btn_apply_ae.setEnabled(has_image and self.ae_model is not None)
        self.btn_clear_ae.setEnabled(self.ae_model is not None)
        self.btn_save_errmap.setEnabled(self._error_bgr is not None)

    # --------------------------------------------------------
    # Чтение изображений (TIFF — через PIL/tifffile, остальное — cv2)
    # --------------------------------------------------------
    @staticmethod
    def _read_via_pil(path):
        """Читает изображение через PIL. Возвращает BGR/BGRA/grayscale или None."""
        try:
            from PIL import Image
            with Image.open(path) as pil_img:
                arr = np.array(pil_img)
            if arr is None or arr.size == 0:
                return None
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception:
            return None

    @staticmethod
    def _read_via_tifffile(path):
        """Читает TIFF через tifffile. Возвращает BGR/BGRA/grayscale или None."""
        try:
            import tifffile
        except ImportError:
            return None
        try:
            arr = tifffile.imread(path)
            if arr is None or arr.size == 0:
                return None
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception:
            return None

    @staticmethod
    def _normalize_to_uint8(img):
        """
        Приводит изображение к uint8 (0..255), как в image_io.normalize_to_uint8.
        Корректно обрабатывает 16-бит, float (0..1 и произвольные диапазоны)
        и знаковые целочисленные типы.
        """
        if img.dtype == np.uint8:
            return img
        if img.dtype == np.uint16:
            return (img / 256).astype(np.uint8)
        if img.dtype in (np.float32, np.float64):
            if img.size == 0:
                return img.astype(np.uint8)
            mx = float(img.max())
            if mx <= 1.0:
                return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            mn = float(img.min())
            if mx - mn < 1e-9:
                return np.zeros(img.shape, dtype=np.uint8)
            return np.clip((img - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
        # int16 / int32 / прочее — нормализация по фактическому диапазону.
        f = img.astype(np.float32)
        if f.size == 0:
            return f.astype(np.uint8)
        mn = float(f.min()); mx = float(f.max())
        if mx - mn < 1e-9:
            return np.zeros(img.shape, dtype=np.uint8)
        return np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

    def _read_image(self, path):
        """
        Читает изображение с фолбэками:
          • TIFF  — сначала PIL, затем tifffile, затем cv2 (с заглушённым stderr);
          • прочие — сначала cv2, затем PIL.
        Результат всегда 3-канальный BGR uint8.
        """
        try:
            ext = os.path.splitext(path)[1].lower()

            if ext in ('.tif', '.tiff'):
                img = self._read_via_pil(path)
                if img is None:
                    img = self._read_via_tifffile(path)
                if img is None:
                    with _suppress_stderr():
                        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            else:
                with _suppress_stderr():
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    img = self._read_via_pil(path)

            if img is None or img.size == 0:
                return None

            img = self._normalize_to_uint8(img)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] != 3:
                img = img[:, :, :3]
            return img
        except Exception:
            return None

    def _set_image(self, img_bgr: np.ndarray):
        self.current_pixmap = cv_to_qpixmap(img_bgr)
        self.image_label.set_image_pixmap(self.current_pixmap)

    def _clear_ae_view(self):
        self._error_pixmap = None
        self._error_bgr = None
        self._ae_err_hw = None
        self._ae_valid_region = None
        self._ae_base_bgr = None
        self._ae_threshold = float("nan")
        self.error_label.clear_image()
        self.error_label.setVisible(False)
        self._update_actions_enabled()

    # --------------------------------------------------------
    # Загрузка модели YOLO
    # --------------------------------------------------------
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели YOLO", "",
            "YOLO weights (*.pt *.onnx *.engine);;Все файлы (*)"
        )
        if not path:
            return
        try:
            self.statusBar().showMessage(f"Загрузка модели: {path}")
            QApplication.processEvents()
            self.model = YOLO(path)
            self.model_path = path
            self.model_edit.setText(path)
            names = self.model.names
            self.statusBar().showMessage(f"Модель загружена: {len(names)} классов")
            self._log(f"[YOLO] {path}")
            self._log(f"[YOLO классы] {dict(names)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{e}")
            self.statusBar().showMessage("Ошибка загрузки модели")
            return
        self._update_actions_enabled()
        self._auto_apply_all()

    # --------------------------------------------------------
    # Загрузка модели AE/VAE (conv и unet)
    # --------------------------------------------------------
    def load_ae_model(self):
        # Ленивый импорт, чтобы избежать циклической зависимости
        from yolo_test import build_ae_model_from_cfg, detect_arch

        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите чекпойнт модели AE/VAE", "",
            "PyTorch checkpoint (*.pt *.pth);;Все файлы (*)"
        )
        if not path:
            return
        try:
            self.statusBar().showMessage(f"Загрузка AE/VAE: {path}")
            QApplication.processEvents()

            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location="cpu")

            if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
                raise RuntimeError(
                    "В чекпойнте нет ожидаемых полей 'model' и 'config'."
                )

            cfg = dict(ckpt["config"])
            arch = detect_arch(cfg)

            required = ["model_type", "in_channels", "img_h", "img_w"]
            if arch == "unet":
                required.append("latent_channels")
            else:
                required.append("latent_dim")
            for k in required:
                if k not in cfg:
                    raise RuntimeError(
                        f"В config чекпойнта нет ключа '{k}' (arch={arch})."
                    )

            device = self._resolve_torch_device()
            model = build_ae_model_from_cfg(cfg)
            model.load_state_dict(ckpt["model"])
            model.eval()

            try:
                model.to(device)
            except Exception as e:
                self._log(f"[AE/VAE] Не удалось перевести модель на {device}: {e}. "
                          f"Использую CPU.")
                device = torch.device("cpu")
                model.to(device)

            self.ae_model = model
            self.ae_cfg = cfg
            self.ae_model_path = path
            self.ae_device = device
            self.ae_arch = arch
            self.ae_model_edit.setText(path)

            self._ae_ckpt_threshold = ckpt.get("threshold_value", None)
            self._ae_ckpt_percentile = ckpt.get("threshold_percentile", None)

            thr_info = ""
            if self._ae_ckpt_threshold is not None:
                thr_info = f", ckpt-threshold={self._ae_ckpt_threshold:.6f}"
                if self._ae_ckpt_percentile is not None:
                    thr_info += f" @ p{self._ae_ckpt_percentile}"

            if arch == "unet":
                latent_info = f"latent_channels={cfg['latent_channels']}"
            else:
                latent_info = f"latent_dim={cfg['latent_dim']}"

            self._log(f"[AE/VAE] {path}")
            self._log(f"[AE/VAE] arch={arch}, type={cfg['model_type']}, "
                      f"{latent_info}, "
                      f"size=({cfg['img_h']},{cfg['img_w']}), "
                      f"color={cfg.get('color_mode')}, "
                      f"keep_aspect={cfg.get('keep_aspect')}, "
                      f"device={device}{thr_info}")
            self.statusBar().showMessage(f"AE/VAE загружен ({arch})")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить AE/VAE:\n{e}")
            self.statusBar().showMessage("Ошибка загрузки AE/VAE")
            return

        self._update_actions_enabled()
        self._auto_apply_all()

    # --------------------------------------------------------
    # Очистка модели AE/VAE
    # --------------------------------------------------------
    def clear_ae_model(self):
        if self.ae_model is None:
            return

        prev_path = self.ae_model_path

        try:
            if self.ae_model is not None:
                self.ae_model.to("cpu")
        except Exception:
            pass
        self.ae_model = None
        self.ae_cfg = None
        self.ae_model_path = ""
        self.ae_device = torch.device("cpu")
        self.ae_arch = "conv"
        self._ae_ckpt_threshold = None
        self._ae_ckpt_percentile = None

        self.ae_model_edit.clear()

        self._clear_ae_view()
        if self.original_image is not None:
            if self.model is not None:
                self._run_inference_on_current()
            else:
                self.displayed_image = self.original_image.copy()
                self.display_mode = "original"
                self._set_image(self.displayed_image)
                self._update_objects_list()
            self._apply_auto_transforms()

        self._update_actions_enabled()
        self.statusBar().showMessage("AE/VAE очищен")
        if prev_path:
            self._log(f"[AE/VAE] Очищено: {prev_path}")
        else:
            self._log("[AE/VAE] Очищено")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------------------------------------------------
    # Загрузка изображений / папки
    # --------------------------------------------------------
    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;Все файлы (*)"
        )
        if files:
            self._set_images(files)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return
        found = []
        for r, _, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(self.IMAGE_EXTS):
                    found.append(os.path.join(r, f))
        if not found:
            QMessageBox.information(self, "Пусто", "В выбранной папке нет изображений.")
            return
        self._set_images(found)

    def _set_images(self, paths):
        self.image_paths = list(paths)
        self.current_index = 0 if paths else -1
        self._log(f"Загружено изображений: {len(self.image_paths)}")
        self._show_current()

    # --------------------------------------------------------
    # Навигация
    # --------------------------------------------------------
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._show_current()

    def _show_current(self):
        if self.current_index < 0 or not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        img = self._read_image(path)

        self.current_detections = []
        self.display_mode = "original"
        self._clear_ae_view()

        if img is None:
            self.original_image = None
            self.displayed_image = None
            self.current_pixmap = None
            self.image_label.clear_image()
            self.image_label.set_placeholder(f"Не удалось прочитать:\n{path}")
            self.objects_list.clear()
            self.image_info_label.setText(
                f"{self.current_index + 1} / {len(self.image_paths)}: "
                f"{os.path.basename(path)} (ошибка)"
            )
            self._update_nav_buttons()
            self._update_actions_enabled()
            return

        self.original_image = img
        self._auto_apply_all()

        fname = os.path.basename(path)
        h, w = img.shape[:2]
        self.image_info_label.setText(
            f"{self.current_index + 1} / {len(self.image_paths)}: {fname}  ({w}×{h})"
        )
        self.statusBar().showMessage(fname)
        self._update_nav_buttons()
        self._update_actions_enabled()

    # --------------------------------------------------------
    # Сохранение изображения
    # --------------------------------------------------------
    def save_image(self):
        if self.displayed_image is None:
            return

        if 0 <= self.current_index < len(self.image_paths):
            src = self.image_paths[self.current_index]
            base = os.path.splitext(os.path.basename(src))[0]
            start_dir = os.path.dirname(src)
        else:
            base = "image"
            start_dir = ""

        if self.display_mode == "cropped":
            suffix = "_crop"
        elif self.display_mode == "rotated":
            suffix = "_rot"
        elif self.display_mode == "manual":
            suffix = "_edit"
        else:
            suffix = ""

        default_name = f"{base}{suffix}.png"
        default_path = os.path.join(start_dir, default_name) if start_dir else default_name

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;Все файлы (*)"
        )
        if not save_path:
            return
        try:
            ok = cv2.imwrite(save_path, self.displayed_image)
            if not ok:
                raise RuntimeError("cv2.imwrite вернул False")
            self._log(f"Сохранено: {save_path}")
            self.statusBar().showMessage(f"Сохранено: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    # --------------------------------------------------------
    # Сохранение карты ошибок
    # --------------------------------------------------------
    def save_error_map(self):
        if self._error_bgr is None:
            QMessageBox.information(
                self, "Нет карты ошибок",
                "Сначала примените модель AE/VAE к снимку."
            )
            return

        if 0 <= self.current_index < len(self.image_paths):
            src = self.image_paths[self.current_index]
            base = os.path.splitext(os.path.basename(src))[0]
            start_dir = os.path.dirname(src)
        else:
            base = "image"
            start_dir = ""

        default_name = f"{base}_errmap.png"
        default_path = (
            os.path.join(start_dir, default_name) if start_dir else default_name
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить карту ошибок", default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;"
            "TIFF (*.tif *.tiff);;Все файлы (*)"
        )
        if not save_path:
            return
        try:
            ok = cv2.imwrite(save_path, self._error_bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite вернул False")
            self._log(f"Карта ошибок сохранена: {save_path}")
            self.statusBar().showMessage(f"Карта ошибок сохранена: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось сохранить карту ошибок:\n{e}")