"""Compact prompt controls shared by SAM and classical interactive segmentation."""

from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class PromptTools(QWidget):
    tool_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.positive, self.negative, self.box = (QPushButton(text) for text in ('+ Объект', '− Фон', 'Рамка'))
        for name, button in zip(('positive', 'negative', 'box'), (self.positive, self.negative, self.box)):
            button.setCheckable(True)
            button.clicked.connect(lambda _, name=name: self.tool_requested.emit(name))
            row.addWidget(button)
        layout.addLayout(row)
        for button in (self.positive, self.negative):
            button.setToolTip('Клик — точка; зажатая кнопка — непрерывный штрих. ПКМ рисует фон. Escape возвращает навигацию.')
        row = QHBoxLayout()
        self.full_image = QCheckBox('Весь снимок')
        self.full_image.setChecked(True)
        self.full_image.setToolTip('Использовать весь снимок как рамку. Рисование рамки или выбор объекта отключают этот режим.')
        self.selected_box, self.clear = QPushButton('Бокс из объекта'), QPushButton('Очистить')
        self.selected_box.setToolTip('Взять рамку активного объекта из списка или с изображения.')
        for widget in (self.full_image, self.selected_box, self.clear):
            row.addWidget(widget)
        layout.addLayout(row)
        self.merge = QCheckBox('Объединять все объекты')
        self.merge.setToolTip('Одна общая BBox/OBB; для Segment — выпуклая оболочка всех фрагментов, включая промежутки. '
                              'Фрагменты группируются до фильтрации по площади. Маска не изменяется.')
        layout.addWidget(self.merge)

    def bind_view(self, view):
        def full_changed(checked):
            view.full_image = checked
            view.prompts_changed.emit()
            view.viewport().update()

        def sync():
            blocker = QSignalBlocker(self.full_image)
            self.full_image.setChecked(view.full_image)
            del blocker
            for name in ('positive', 'negative', 'box'):
                getattr(self, name).setChecked(view.prompt_tool == name)

        self.full_image.toggled.connect(full_changed)
        view.prompts_changed.connect(sync)
        view.tool_changed.connect(sync)
        sync()
