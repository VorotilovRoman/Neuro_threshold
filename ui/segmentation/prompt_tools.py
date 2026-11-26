"""Compact prompt controls shared by SAM and classical interactive segmentation."""

from PyQt5.QtCore import QSignalBlocker, pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


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
        self.region = QComboBox()
        self.region.addItem('Весь снимок (100%)', 100)
        for percent in (50, 60, 70, 80, 90):
            self.region.addItem(f'Центр снимка ({percent}%)', percent)
        self.region.addItem('Своя рамка', None)
        self.region.setToolTip('Процент задаёт ширину и высоту центральной рамки относительно снимка. '
                               '50% — половина ширины и половина высоты. Можно нарисовать свою рамку или взять её из объекта.')
        self.selected_box, self.clear = QPushButton('Бокс из объекта'), QPushButton('Очистить')
        self.selected_box.setToolTip('Взять рамку активного объекта из списка или с изображения.')
        for widget in (self.region, self.selected_box, self.clear):
            row.addWidget(widget)
        layout.addLayout(row)
        self.merge = QCheckBox('Объединять все объекты')
        self.merge.setToolTip('Одна общая BBox/OBB; для Segment — выпуклая оболочка всех фрагментов, включая промежутки. '
                              'Фрагменты группируются до фильтрации по площади. Маска не изменяется.')
        layout.addWidget(self.merge)

    def bind_view(self, view):
        def region_changed(_):
            view.set_region_percent(self.region.currentData())

        def sync():
            blocker = QSignalBlocker(self.region)
            self.region.setCurrentIndex(self.region.findData(view.region_percent))
            del blocker
            for name in ('positive', 'negative', 'box'):
                getattr(self, name).setChecked(view.prompt_tool == name)

        self.region.currentIndexChanged.connect(region_changed)
        view.prompts_changed.connect(sync)
        view.tool_changed.connect(sync)
        sync()
