# -*- coding: utf-8 -*-
"""Общие настройки проекта: обычные аннотации и палитра классов датасета.

Один settings/settings.json для старых и новых утилит. Цвета сохраняются
в BGR (OpenCV), в QColorDialog и UI отображаются в RGB/HEX.
"""
from __future__ import annotations

from copy import deepcopy

from path_setup import setup_project_path

setup_project_path()

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QColorDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.settings_service import (
    DEFAULT_DATASET_CLASS_PALETTE_BGR,
    FONT_SIZE_PX_MAX,
    FONT_SIZE_PX_MIN,
    LINE_THICKNESS_PX_MAX,
    LINE_THICKNESS_PX_MIN,
    apply_theme,
    settings,
)


class SettingsWidget(QWidget):
    """Редактор единого конфига с отдельной палитрой разметки датасета."""

    COLOR_LABELS = {
        'annotation': 'Обычная аннотация (без классов)',
        'selected': 'Выделенный объект',
        'crosshair': 'Перекрестие и контур рисования',
        'edit_points': 'Точки редактирования',
        'label_text': 'Текст подписей',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._draft_primary = None
        self._draft_colors = {}
        self._draft_palette = []
        self.color_buttons = {}
        self.palette_buttons = {}
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(14)

        heading = QLabel('Настройки системы')
        heading.setStyleSheet('font-size: 19px; font-weight: 700;')
        layout.addWidget(heading)
        hint = QLabel('Настройки сохраняются в общем settings/settings.json и применяются к открытым вкладкам.')
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # --- Тема ---
        appearance = QGroupBox('Оформление приложения')
        appearance_layout = QVBoxLayout(appearance)
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel('Тема:'))
        from PyQt5.QtWidgets import (
            QComboBox,  # локально, чтобы не расширять публичный API
        )
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['тёмная', 'светлая'])
        theme_row.addWidget(self.theme_combo, 1)
        appearance_layout.addLayout(theme_row)
        primary_row = QHBoxLayout()
        primary_row.addWidget(QLabel('Акцент приложения:'))
        self.primary_btn = QPushButton()
        self.primary_btn.setMinimumWidth(125)
        self.primary_btn.clicked.connect(self.on_primary_color_clicked)
        primary_row.addWidget(self.primary_btn)
        primary_row.addStretch()
        appearance_layout.addLayout(primary_row)
        layout.addWidget(appearance)

        # --- Цвета обычных аннотаций и общих элементов управления ---
        general = QGroupBox('Обычные аннотации и общие элементы')
        general_layout = QVBoxLayout(general)
        description = QLabel(
            'Цвет «Обычная аннотация» используется вне формата разметки датасета. '
            'Цвет выделения, перекрестия, точек редактирования и текста общий '
            'для соответствующих инструментов. Цвета классов задаются ниже отдельно.'
        )
        description.setWordWrap(True)
        general_layout.addWidget(description)
        general_grid = QGridLayout()
        general_grid.setHorizontalSpacing(16)
        general_grid.setVerticalSpacing(8)
        for row, (key, name) in enumerate(self.COLOR_LABELS.items()):
            title = QLabel(name)
            btn = QPushButton()
            btn.setMinimumWidth(118)
            btn.setProperty('color_key', key)
            btn.setToolTip('Открыть полную палитру Qt: HEX, RGB и выбор оттенка')
            btn.clicked.connect(self.on_color_clicked)
            general_grid.addWidget(title, row, 0)
            general_grid.addWidget(btn, row, 1)
            self.color_buttons[key] = btn
        general_grid.setColumnStretch(0, 1)
        general_layout.addLayout(general_grid)
        layout.addWidget(general)

        # --- Палитра разметки датасета ---
        dataset = QGroupBox('Палитра классов датасета')
        dataset_layout = QVBoxLayout(dataset)
        dataset_info = QLabel(
            'Палитра классов 0–9 из исходной вкладки разметки. Каждый цвет можно '
            'изменить: нажмите на образец для открытия QColorDialog. '
            'Аннотация без класса получает класс 0. Если class_id больше числа '
            'цветов, палитра повторяется по кругу. Можно добавлять свои классы.'
        )
        dataset_info.setWordWrap(True)
        dataset_layout.addWidget(dataset_info)
        self.palette_grid = QGridLayout()
        self.palette_grid.setHorizontalSpacing(12)
        self.palette_grid.setVerticalSpacing(10)
        dataset_layout.addLayout(self.palette_grid)
        self.palette_count_label = QLabel()
        dataset_layout.addWidget(self.palette_count_label)
        palette_actions = QHBoxLayout()
        self.add_class_btn = QPushButton('+ Добавить цвет класса')
        self.add_class_btn.clicked.connect(self.add_class_color)
        self.remove_class_btn = QPushButton('Удалить последний')
        self.remove_class_btn.clicked.connect(self.remove_last_class_color)
        self.reset_palette_btn = QPushButton('Восстановить исходную палитру')
        self.reset_palette_btn.clicked.connect(self.reset_class_palette)
        palette_actions.addWidget(self.add_class_btn)
        palette_actions.addWidget(self.remove_class_btn)
        palette_actions.addStretch()
        palette_actions.addWidget(self.reset_palette_btn)
        dataset_layout.addLayout(palette_actions)
        layout.addWidget(dataset)

        # --- Геометрия шрифта и линий ---
        size_group = QGroupBox('Размеры отображения')
        size_layout = QVBoxLayout(size_group)
        self.font_slider, self.font_spin = self._add_slider_row(
            size_layout, 'Размер шрифта подписей', FONT_SIZE_PX_MIN, FONT_SIZE_PX_MAX,
            'Логические пиксели Qt; при масштабировании снимка размер не меняется.'
        )
        self.thickness_slider, self.thickness_spin = self._add_slider_row(
            size_layout, 'Толщина обычной рамки', LINE_THICKNESS_PX_MIN,
            LINE_THICKNESS_PX_MAX,
            'Логические пиксели Qt для обычной рамки; выделение рисуется толще.'
        )
        size_hint = QLabel(
            'Для старых OpenCV-вкладок прежние функции используют эти значения '
            'как коэффициенты относительно своего базового масштаба.'
        )
        size_hint.setWordWrap(True)
        size_layout.addWidget(size_hint)
        layout.addWidget(size_group)

        # --- Предпросмотр ---
        preview_group = QGroupBox('Предпросмотр разметки')
        preview_layout = QVBoxLayout(preview_group)
        self.preview_box = QLabel('Класс 0  ·  пример подписи')
        self.preview_box.setAlignment(Qt.AlignCenter)
        self.preview_box.setMinimumHeight(82)
        self.preview_box.setWordWrap(True)
        preview_layout.addWidget(self.preview_box)
        layout.addWidget(preview_group)
        self.font_slider.valueChanged.connect(self._refresh_preview)
        self.thickness_slider.valueChanged.connect(self._refresh_preview)

        # --- Навигация ---
        navigation = QGroupBox('Горячие клавиши навигации')
        navigation_layout = QGridLayout(navigation)
        navigation_layout.addWidget(QLabel('Предыдущий снимок:'), 0, 0)
        self.nav_prev_edit = QLineEdit()
        self.nav_prev_edit.setPlaceholderText('Num+4')
        navigation_layout.addWidget(self.nav_prev_edit, 0, 1)
        navigation_layout.addWidget(QLabel('Следующий снимок:'), 1, 0)
        self.nav_next_edit = QLineEdit()
        self.nav_next_edit.setPlaceholderText('Num+6')
        navigation_layout.addWidget(self.nav_next_edit, 1, 1)
        navigation_help = QLabel('Примеры: Num+4, Num+6, Left, Right. Горячие клавиши не изменены.')
        navigation_help.setWordWrap(True)
        navigation_layout.addWidget(navigation_help, 2, 0, 1, 2)
        layout.addWidget(navigation)
        layout.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        reset_btn = QPushButton('Сбросить все настройки')
        reset_btn.clicked.connect(self.reset_settings)
        footer.addWidget(reset_btn)
        footer.addStretch()
        save_btn = QPushButton('Сохранить изменения')
        save_btn.setDefault(True)
        save_btn.setMinimumWidth(180)
        save_btn.clicked.connect(self.save_settings)
        footer.addWidget(save_btn)
        root.addLayout(footer)

    @staticmethod
    def _add_slider_row(layout, title, lo, hi, tooltip):
        row = QHBoxLayout()
        caption = QLabel(title)
        caption.setMinimumWidth(200)
        row.addWidget(caption)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setSingleStep(1)
        slider.setPageStep(max(1, (hi - lo) // 8))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(max(1, (hi - lo) // 8))
        slider.setToolTip(tooltip)
        row.addWidget(slider, 1)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSuffix(' px')
        spin.setMinimumWidth(94)
        spin.setToolTip('Точное значение в логических пикселях')
        row.addWidget(spin)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        layout.addLayout(row)
        return slider, spin

    @staticmethod
    def _to_qcolor(bgr):
        return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))

    @staticmethod
    def _to_bgr(color):
        return [color.blue(), color.green(), color.red()]

    @staticmethod
    def _paint_swatch(button, bgr):
        color = SettingsWidget._to_qcolor(bgr)
        rgb_hex = color.name(QColor.HexRgb).upper()
        luminance = 0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
        fg = '#FFFFFF' if luminance < 135 else '#171A20'
        button.setText(rgb_hex)
        button.setStyleSheet(
            'QPushButton { background-color: %s; color: %s; '
            'border: 1px solid rgba(128,128,128,120); border-radius: 9px; '
            'padding: 8px 12px; font-weight: 700; } '
            'QPushButton:hover { border: 2px solid %s; }'
            % (rgb_hex, fg, fg)
        )
        button.setToolTip('Цвет %s (RGB). Нажмите, чтобы открыть Qt-палитру.' % rgb_hex)

    def _choose_color(self, bgr, title):
        # Не нативный системный диалог, а полноценный Qt QColorDialog:
        # основные и пользовательские образцы + HSV + HEX/RGB.
        dialog = QColorDialog(self._to_qcolor(bgr), self)
        dialog.setWindowTitle(title)
        dialog.setOption(QColorDialog.DontUseNativeDialog, True)
        for i, entry in enumerate(self._draft_palette[:QColorDialog.customCount()]):
            QColorDialog.setCustomColor(i, self._to_qcolor(entry))
        if dialog.exec_() != QColorDialog.Accepted:
            return None
        picked = dialog.selectedColor()
        return self._to_bgr(picked) if picked.isValid() else None

    def _rebuild_palette_grid(self):
        while self.palette_grid.count():
            item = self.palette_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.palette_buttons.clear()
        for class_id, bgr in enumerate(self._draft_palette):
            tile = QWidget(self)
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(1, 2, 1, 2)
            tile_layout.setSpacing(4)
            name = QLabel('Класс %d' % class_id)
            name.setAlignment(Qt.AlignCenter)
            tile_layout.addWidget(name)
            swatch = QPushButton()
            swatch.setMinimumWidth(95)
            swatch.setAccessibleName('Выбрать цвет класса %d' % class_id)
            swatch.clicked.connect(lambda checked=False, idx=class_id: self._edit_class_color(idx))
            self._paint_swatch(swatch, bgr)
            tile_layout.addWidget(swatch)
            self.palette_grid.addWidget(tile, class_id // 5, class_id % 5)
            self.palette_buttons[class_id] = swatch
        self.palette_count_label.setText('Настраиваемых цветов: %d' % len(self._draft_palette))
        self.remove_class_btn.setEnabled(len(self._draft_palette) > len(DEFAULT_DATASET_CLASS_PALETTE_BGR))
        self._refresh_preview()

    def _edit_class_color(self, class_id):
        if not (0 <= class_id < len(self._draft_palette)):
            return
        chosen = self._choose_color(self._draft_palette[class_id], 'Цвет класса %d' % class_id)
        if chosen is not None:
            self._draft_palette[class_id] = chosen
            self._paint_swatch(self.palette_buttons[class_id], chosen)
            self._refresh_preview()

    def add_class_color(self):
        default = DEFAULT_DATASET_CLASS_PALETTE_BGR[
            len(self._draft_palette) % len(DEFAULT_DATASET_CLASS_PALETTE_BGR)
        ]
        self._draft_palette.append(list(default))
        self._rebuild_palette_grid()

    def remove_last_class_color(self):
        if len(self._draft_palette) > len(DEFAULT_DATASET_CLASS_PALETTE_BGR):
            self._draft_palette.pop()
            self._rebuild_palette_grid()

    def reset_class_palette(self):
        """Сбрасывает только черновик палитры, не другие настройки."""
        self._draft_palette = deepcopy(DEFAULT_DATASET_CLASS_PALETTE_BGR)
        self._rebuild_palette_grid()

    def _refresh_preview(self, *_):
        if not self._draft_palette:
            return
        frame = self._to_qcolor(self._draft_palette[0]).name()
        label_color = self._to_qcolor(
            self._draft_colors.get('label_text', settings.get_color('label_text'))
        ).name()
        size = self.font_slider.value()
        width = self.thickness_slider.value()
        self.preview_box.setStyleSheet(
            'QLabel { border: %dpx solid %s; border-radius: 9px; '
            'font-size: %dpx; font-weight: 600; color: %s; padding: 12px; }'
            % (width, frame, size, label_color)
        )

    def load_current_settings(self):
        """Данные с диска не изменяются до нажатия «Сохранить»."""
        curr = settings.current
        self._draft_primary = list(settings.get_primary_color())
        self._draft_colors = {
            name: list(settings.get_color(name)) for name in self.COLOR_LABELS
        }
        self._draft_palette = settings.get_dataset_class_palette()
        self.theme_combo.setCurrentText('тёмная' if curr.get('theme', 'dark') == 'dark' else 'светлая')
        self._paint_swatch(self.primary_btn, list(reversed(self._draft_primary)))
        for name, button in self.color_buttons.items():
            self._paint_swatch(button, self._draft_colors[name])
        self.font_spin.setValue(settings.get_font_size_px())
        self.thickness_spin.setValue(settings.get_line_thickness_px())
        nav = curr.get('navigation_shortcuts', {}) or {}
        self.nav_prev_edit.setText(str(nav.get('prev', 'Num+4')))
        self.nav_next_edit.setText(str(nav.get('next', 'Num+6')))
        self._rebuild_palette_grid()
        self._refresh_preview()

    def on_primary_color_clicked(self):
        bgr = list(reversed(self._draft_primary))
        chosen = self._choose_color(bgr, 'Основной цвет темы')
        if chosen is not None:
            self._draft_primary = list(reversed(chosen))
            self._paint_swatch(self.primary_btn, chosen)

    def on_color_clicked(self):
        button = self.sender()
        key = button.property('color_key') if button is not None else None
        if key not in self._draft_colors:
            return
        chosen = self._choose_color(self._draft_colors[key], self.COLOR_LABELS[key])
        if chosen is not None:
            self._draft_colors[key] = chosen
            self._paint_swatch(button, chosen)
            self._refresh_preview()

    def reset_settings(self):
        settings.save(deepcopy(settings.defaults))
        self.load_current_settings()
        apply_theme(QApplication.instance())
        QMessageBox.information(self, 'Настройки', 'Значения по умолчанию восстановлены.')

    def save_settings(self):
        # Сохраняем все неизвестные дополнительные поля конфига.
        new_settings = deepcopy(settings.current)
        new_settings['theme'] = 'dark' if self.theme_combo.currentText() == 'тёмная' else 'light'
        new_settings['primary_color'] = list(self._draft_primary)  # RGB, как раньше
        colors = deepcopy(new_settings.get('colors', {}) or {})
        colors.update(deepcopy(self._draft_colors))  # BGR, как раньше
        new_settings['colors'] = colors
        new_settings['dataset_class_palette'] = deepcopy(self._draft_palette)  # BGR
        new_settings['font_size_px'] = int(self.font_spin.value())
        new_settings['line_thickness_px'] = int(self.thickness_spin.value())
        nav = dict(new_settings.get('navigation_shortcuts', {}) or {})
        nav.update({
            'prev': self.nav_prev_edit.text().strip() or 'Num+4',
            'next': self.nav_next_edit.text().strip() or 'Num+6',
        })
        new_settings['navigation_shortcuts'] = nav
        if not settings.save(new_settings):
            QMessageBox.critical(self, 'Настройки', 'Не удалось сохранить настройки. Проверьте путь и параметры файла.')
            return
        apply_theme(QApplication.instance())
        QMessageBox.information(self, 'Настройки', 'Настройки сохранены.')
