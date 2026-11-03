# -*- coding: utf-8 -*-
"""Read-only pair-validation dialog using canonical annotations and Qt vectors.

Displays the original uint8 preview in a DatasetReviewGraphicsView without
rasterizing annotation outlines. Masks use the global dataset class palette.
"""
import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget, QFrame,
                             QShortcut, QLineEdit, QSizePolicy)

from utils_new.dataset_io_new import _read_with_fallback
from utils_new.dataset_validation_new import ImageStats, load_annotations, read_mask_with_fallback
from utils_new.dataset_review_view import DatasetReviewGraphicsView
from utils_new.dataset_mask_preview_new import blend_index_mask


def _to_qpixmap(bgr):
    """Convert a BGR preview to an owned QPixmap without scaling its pixels."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Preview must be BGR uint8')
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _palette_color(class_id):
    """Read current application class palette, with deterministic fallback."""
    try:
        from utils.settings import settings
        return settings.get_dataset_class_color(int(class_id))
    except (AttributeError, ImportError, ValueError):
        palette = ((20, 180, 20), (220, 135, 0), (0, 185, 215),
                   (195, 35, 225), (60, 120, 230), (210, 155, 105))
        return palette[int(class_id) % len(palette)]


class AutoFitReviewView(DatasetReviewGraphicsView):
    """Review viewer that fits the image *after* Qt has laid out the viewport.

    SmartGraphicsViewNew.set_pixmap() calls fitInView before the dialog is shown;
    the initial viewport is still small. Fitting again on show and resize fixes
    this without raster scaling or changing existing mouse interaction.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_fit = True
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def fit_image(self):
        self._auto_fit = True
        if self.viewport().width() > 20 and self.viewport().height() > 20:
            self.reset_view()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.fit_image)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and self.isVisible():
            QTimer.singleShot(0, self._fit_if_enabled)

    def _fit_if_enabled(self):
        if self._auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        # Don't override manually zoomed image when dialog is resized.
        self._auto_fit = False
        super().wheelEvent(event)

    def zoom_actual_pixels(self):
        self._auto_fit = False
        self.resetTransform()
        self.centerOn(self.pixmap_item)
        self.viewport().update()


class PairValidationDialog(QDialog):
    """Large, modern and read-only validator with Fit/1:1/fullscreen controls."""

    def __init__(self, parent, pair, dataset_type, validator):
        super().__init__(parent)
        self.dataset_type = int(dataset_type)
        self.validator = validator
        self.view = None
        image_path = str(pair[0])
        aux_path = str(pair[1]) if len(pair) > 1 and pair[1] else None
        self.setWindowTitle(f'Проверка пары — {os.path.basename(image_path)}')
        self.setMinimumSize(800, 580)
        screen_rect = QApplication.desktop().availableGeometry(self)
        self.resize(min(1500, max(800, int(screen_rect.width() * 0.90))),
                    min(1000, max(580, int(screen_rect.height() * 0.88))))

        image = _read_with_fallback(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            QMessageBox.warning(self, 'Ошибка', f'Не удалось прочитать изображение:\n{image_path}')
            self.reject()
            return
        display = ImageStats.to_display_bgr(image)
        h, w = display.shape[:2]
        annotations = []
        mask_status = None
        mask_short_status = None

        if self.dataset_type == 0 and aux_path:
            try:
                annotations = load_annotations(aux_path, w, h,
                                               format_hint=validator.input_format)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, 'Аннотации', f'Ошибка чтения аннотаций: {exc}')
        elif self.dataset_type == 1 and aux_path:
            mask = read_mask_with_fallback(aux_path)
            if mask is None:
                mask_status = '⚠ Маска не прочитана — проверьте путь/формат файла'
                mask_short_status = '⚠ Маска не прочитана'
            else:
                if mask.shape != (h, w):
                    mask_status = (f'⚠ Размер маски {mask.shape[1]}×{mask.shape[0]} '
                                   f'не совпадает со снимком {w}×{h}')
                    mask_short_status = '⚠ Размеры не совпадают'
                # Mapping keys are SOURCE pixel values (not class numbers).
                # Discard unselected values, leaving raw mask untouched.
                remap = getattr(parent, 'mask_class_remap', None) or None
                display, report = blend_index_mask(
                    display, mask, _palette_color,
                    opacity=0.6, pixel_offset=1, background_value=0,
                    pixel_to_class=remap,
                )
                detail = report.description()
                if report.is_empty:
                    mask_status = '⚠ Маска не содержит видимых классов. ' + detail
                    mask_short_status = '⚠ Нет видимых классов'
                elif mask_status is not None:
                    mask_status += '; ' + detail
                else:
                    mask_status = 'Маска наложена. ' + detail
                    mask_short_status = f'Маска: {report.percentage:.1f}% пикселей'
        elif self.dataset_type == 1:
            mask_status = '⚠ К паре не привязан файл маски'
            mask_short_status = '⚠ Нет файла маски'

        valid, error, color_type, bit_type = validator.validate_full(pair)
        ctext = {'color': 'Цветное', 'gray_real': 'Ч/Б (1 канал)',
                 'gray_visual': 'Визуально Ч/Б', 'unknown': '—'}.get(color_type, '—')
        btext = {'uint8_png': 'uint8 PNG', 'uint8_other': 'uint8 (другой)',
                 'uint16': 'uint16', 'int16': 'int16', 'float': 'float',
                 'other': 'other', 'unknown': '—'}.get(bit_type, '—')

        root = QVBoxLayout(self)
        root.setContentsMargins(11, 10, 11, 9)
        root.setSpacing(7)

        # Compact status, control toolbar; image consumes all remaining height.
        top = QHBoxLayout()
        top.setSpacing(9)
        status = QLabel('✓  Корректная пара' if valid else '✕  Ошибка пары')
        status.setObjectName('validationStatusOk' if valid else 'validationStatusError')
        status.setToolTip('Валидация пройдена' if valid else str(error))
        top.addWidget(status)
        title = QLabel(os.path.basename(image_path))
        title.setObjectName('validationTitle')
        title.setToolTip(image_path)
        top.addWidget(title)
        top.addStretch(1)
        dims = QLabel(f'{w} × {h} px   ·   {btext}   ·   {ctext}')
        dims.setObjectName('validationMeta')
        top.addWidget(dims)
        root.addLayout(top)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.fit_btn = QPushButton('Вписать в окно')
        self.fit_btn.setToolTip('Масштабировать изображение до области просмотра (F)')
        self.actual_btn = QPushButton('1:1')
        self.actual_btn.setToolTip('Исходный масштаб, 1 пиксель изображения = 1 пиксель экрана (1)')
        self.fullscreen_btn = QPushButton('На весь экран')
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.setToolTip('Переключить полноэкранный режим (F11)')
        bar.addWidget(self.fit_btn)
        bar.addWidget(self.actual_btn)
        bar.addWidget(self.fullscreen_btn)
        bar.addStretch(1)
        if self.dataset_type == 0:
            count_label = QLabel(f'Аннотаций: {len(annotations)}')
        elif self.dataset_type == 1:
            count_label = QLabel(mask_short_status or '⚠ Маска отсутствует')
        else:
            count_label = QLabel('Без аннотаций')
        count_label.setObjectName('validationMeta')
        if self.dataset_type == 1:
            count_label.setToolTip(mask_status or '')
        bar.addWidget(count_label)
        root.addLayout(bar)

        self.view = AutoFitReviewView(self)
        self.view.label_mode = 'all'
        self.view.selection_enabled = bool(annotations)
        self.view.set_pixmap(_to_qpixmap(display))
        self.view.set_annotations(annotations, w, h)
        root.addWidget(self.view, 1)

        bottom = QHBoxLayout()
        self.details_btn = QPushButton('Подробности ▾')
        self.details_btn.setCheckable(True)
        bottom.addWidget(self.details_btn)
        if not valid:
            err = QLabel(str(error))
            err.setObjectName('validationErrorText')
            err.setToolTip(str(error))
            bottom.addWidget(err)
        bottom.addStretch(1)
        close = QPushButton('Закрыть')
        close.clicked.connect(self.accept)
        bottom.addWidget(close)
        root.addLayout(bottom)
        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 2, 0, 2)
        for name, path in (('Снимок', image_path), ('Разметка/маска', aux_path or '—')):
            row = QHBoxLayout()
            label = QLabel(name + ':')
            label.setMinimumWidth(100)
            editor = QLineEdit(path)
            editor.setReadOnly(True)
            editor.setToolTip(path)
            row.addWidget(label)
            row.addWidget(editor, 1)
            details_layout.addLayout(row)
        self.details.setVisible(False)
        root.addWidget(self.details)
        self.details_btn.toggled.connect(self._toggle_details)
        self.fit_btn.clicked.connect(self.view.fit_image)
        self.actual_btn.clicked.connect(self.view.zoom_actual_pixels)
        self.fullscreen_btn.toggled.connect(self._set_fullscreen)
        self._shortcut_fit = QShortcut(QKeySequence('F'), self)
        self._shortcut_fit.activated.connect(self.view.fit_image)
        self._shortcut_actual = QShortcut(QKeySequence('1'), self)
        self._shortcut_actual.activated.connect(self.view.zoom_actual_pixels)
        self._shortcut_fullscreen = QShortcut(QKeySequence('F11'), self)
        self._shortcut_fullscreen.activated.connect(
            lambda: self.fullscreen_btn.setChecked(not self.fullscreen_btn.isChecked()))

        self.setStyleSheet("""
            QLabel#validationStatusOk { padding: 6px 10px; font-weight: 700;
                color: #187b4a; border: 1px solid #187b4a; border-radius: 6px; }
            QLabel#validationStatusError { padding: 6px 10px; font-weight: 700;
                color: #bb3333; border: 1px solid #bb3333; border-radius: 6px; }
            QLabel#validationTitle { font-weight: 650; font-size: 14px; }
            QLabel#validationMeta { color: palette(mid); }
            QLabel#validationErrorText { color: #bb3333; }
            QPushButton { padding: 5px 11px; min-height: 24px; }
        """)
        # Deliberately defer initial fit until after the first show/layout pass.
        QTimer.singleShot(0, self.view.fit_image)

    def _toggle_details(self, visible):
        self.details.setVisible(bool(visible))
        self.details_btn.setText('Подробности ▴' if visible else 'Подробности ▾')
        if self.view is not None and self.view._auto_fit:
            QTimer.singleShot(0, self.view.fit_image)

    def _set_fullscreen(self, fullscreen):
        self.showFullScreen() if fullscreen else self.showNormal()
        self.fullscreen_btn.setText('Обычное окно' if fullscreen else 'На весь экран')
        if self.view is not None and self.view._auto_fit:
            QTimer.singleShot(0, self.view.fit_image)
