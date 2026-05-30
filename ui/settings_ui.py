# settings_ui.py
import sys
import os
from path_setup import setup_project_path
setup_project_path()

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QPushButton, QColorDialog, QMessageBox)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from utils.settings import settings, apply_theme
from PyQt5.QtWidgets import QApplication


class SettingsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.load_current_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Тема
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Тема:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["темная", "светлая"])
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        # Основной цвет темы
        primary_layout = QHBoxLayout()
        primary_layout.addWidget(QLabel("Основной цвет темы:"))
        self.primary_btn = QPushButton("выбрать цвет")
        self.primary_btn.clicked.connect(self.on_primary_color_clicked)
        primary_layout.addWidget(self.primary_btn)
        layout.addLayout(primary_layout)

        # Цвета аннотаций (остаются)
        colors_group = QWidget()
        colors_layout = QVBoxLayout(colors_group)
        colors_layout.addWidget(QLabel("Цвета аннотаций:"))
        self.color_buttons = {}
        color_names = {
            "annotation": "Цвет аннотации",
            "selected": "Цвет выделенной аннотации",
            "crosshair": "Цвет перекрестия",
            "edit_points": "Цвет точек редактирования",
            "label_text": "Цвет подписей"
        }
        for key, label in color_names.items():
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(QLabel(label))
            btn = QPushButton("выбрать цвет")
            btn.setProperty("color_key", key)
            btn.clicked.connect(self.on_color_clicked)
            btn_layout.addWidget(btn)
            colors_layout.addLayout(btn_layout)
            self.color_buttons[key] = btn
        layout.addWidget(colors_group)

        # Размер шрифта
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Размер шрифта:"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(["авто", "маленький", "средний", "большой"])
        font_layout.addWidget(self.font_combo)
        layout.addLayout(font_layout)

        # Толщина рамки
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Толщина рамки:"))
        self.thickness_combo = QComboBox()
        self.thickness_combo.addItems(["авто", "маленький", "средний", "большой"])
        thickness_layout.addWidget(self.thickness_combo)
        layout.addLayout(thickness_layout)

        # Кнопка сброса
        reset_btn = QPushButton("Сбросить на значения по умолчанию")
        reset_btn.clicked.connect(self.reset_settings)
        layout.addWidget(reset_btn)

        # Кнопка сохранить
        save_btn = QPushButton("Сохранить изменения")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        layout.addStretch()

    def load_current_settings(self):
        curr = settings.current
        # Тема
        theme = curr.get("theme", "dark")
        self.theme_combo.setCurrentText("темная" if theme == "dark" else "светлая")
        # Основной цвет
        primary = curr.get("primary_color", [42, 130, 218])
        self.primary_btn.setStyleSheet(f"background-color: rgb({primary[0]}, {primary[1]}, {primary[2]});")
        # Цвета аннотаций
        colors = curr.get("colors", {})
        for key, btn in self.color_buttons.items():
            bgr = colors.get(key, [0, 0, 0])
            rgb = (bgr[2], bgr[1], bgr[0])
            btn.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});")
        # Размер шрифта
        font_mode = curr.get("font_size_mode", "auto")
        if font_mode == "auto":
            self.font_combo.setCurrentText("авто")
        elif font_mode == "small":
            self.font_combo.setCurrentText("маленький")
        elif font_mode == "medium":
            self.font_combo.setCurrentText("средний")
        elif font_mode == "large":
            self.font_combo.setCurrentText("большой")
        # Толщина рамки
        thick_mode = curr.get("line_thickness_mode", "auto")
        if thick_mode == "auto":
            self.thickness_combo.setCurrentText("авто")
        elif thick_mode == "small":
            self.thickness_combo.setCurrentText("маленький")
        elif thick_mode == "medium":
            self.thickness_combo.setCurrentText("средний")
        elif thick_mode == "large":
            self.thickness_combo.setCurrentText("большой")

    def on_primary_color_clicked(self):
        current = settings.get_primary_color()
        qcolor = QColor(current[0], current[1], current[2])
        color = QColorDialog.getColor(qcolor, self, "Выберите основной цвет темы")
        if color.isValid():
            if not hasattr(self, "_temp_primary"):
                self._temp_primary = None
            self._temp_primary = (color.red(), color.green(), color.blue())
            self.primary_btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def on_color_clicked(self):
        btn = self.sender()
        key = btn.property("color_key")
        current_color = settings.get_color(key)  # BGR
        qcolor = QColor(current_color[2], current_color[1], current_color[0])
        color = QColorDialog.getColor(qcolor, self, f"Выберите цвет для {key}")
        if color.isValid():
            if not hasattr(self, "_temp_colors"):
                self._temp_colors = {}
            self._temp_colors[key] = (color.blue(), color.green(), color.red())  # BGR
            btn.setStyleSheet(f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});")

    def reset_settings(self):
        """Сброс к значениям по умолчанию"""
        settings.save(settings.defaults)
        self.load_current_settings()
        if hasattr(self, "_temp_colors"):
            del self._temp_colors
        if hasattr(self, "_temp_primary"):
            self._temp_primary = None
        QMessageBox.information(self, "Настройки", "Настройки сброшены к значениям по умолчанию.")

    def save_settings(self):
        theme = "dark" if self.theme_combo.currentText() == "темная" else "light"
        colors = settings.current.get("colors", {}).copy()
        if hasattr(self, "_temp_colors"):
            for key, bgr in self._temp_colors.items():
                colors[key] = list(bgr)
        font_text = self.font_combo.currentText()
        if font_text == "авто":
            font_mode = "auto"
        elif font_text == "маленький":
            font_mode = "small"
        elif font_text == "средний":
            font_mode = "medium"
        else:
            font_mode = "large"

        thick_text = self.thickness_combo.currentText()
        if thick_text == "авто":
            thick_mode = "auto"
        elif thick_text == "маленький":
            thick_mode = "small"
        elif thick_text == "средний":
            thick_mode = "medium"
        else:
            thick_mode = "large"

        primary_color = settings.current.get("primary_color", [42, 130, 218])
        if hasattr(self, "_temp_primary") and self._temp_primary is not None:
            primary_color = list(self._temp_primary)

        new_settings = {
            "theme": theme,
            "primary_color": primary_color,
            "colors": colors,
            "font_size_mode": font_mode,
            "line_thickness_mode": thick_mode
        }
        settings.save(new_settings)
        apply_theme(QApplication.instance())
        QMessageBox.information(self, "Настройки", "Настройки сохранены.")
        if hasattr(self, "_temp_colors"):
            del self._temp_colors
        if hasattr(self, "_temp_primary"):
            self._temp_primary = None