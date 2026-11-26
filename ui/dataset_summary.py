"""Compact dataset overview rendered only from background inspection metadata."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class DatasetSummary(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        heading = QHBoxLayout()
        title = QLabel('Обзор датасета')
        title.setStyleSheet('font-weight: 600;')
        self.status = QLabel('Снимки не загружены')
        self.status.setWordWrap(True)
        heading.addWidget(title)
        heading.addStretch()
        heading.addWidget(self.status)
        layout.addLayout(heading)
        grid = QGridLayout()
        grid.setSpacing(5)
        self.values, self.captions = {}, {}
        tips = {
            'images': 'Число загруженных снимков. Наличие разметки не требуется.',
            'labelled': 'Снимки с найденным файлом TXT или маски, включая пустой фон и файлы с ошибками.',
            'unlabelled': 'Снимки, для которых нет файла разметки. Это нормальное состояние.',
            'with_objects': 'Снимки с корректной разметкой, содержащей объекты; это число снимков, а не объектов.',
            'background': 'Пустой TXT или маска только с нулевым фоном. Файл существует, объектов нет.',
            'errors': 'Все проверенные снимки с ошибкой чтения, разметки или размера маски. Это отдельный показатель.',
        }
        for index, (key, caption) in enumerate((('images', 'Снимки'), ('labelled', 'С TXT'),
                                                ('unlabelled', 'Без разметки'), ('with_objects', 'С объектами'),
                                                ('background', 'Размеченный фон'), ('errors', 'С ошибками'))):
            card = QFrame()
            card.setObjectName('datasetMetric')
            card.setStyleSheet('QFrame#datasetMetric { border: 1px solid palette(mid); border-radius: 5px; }')
            row = QHBoxLayout(card)
            row.setContentsMargins(8, 5, 8, 5)
            label, value = QLabel(caption), QLabel('0')
            label.setWordWrap(True)
            value.setStyleSheet('font-weight: 700; font-size: 16px;')
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(label, 1)
            row.addWidget(value)
            card.setToolTip(tips[key])
            grid.addWidget(card, index // 3, index % 3)
            self.values[key], self.captions[key] = value, label
        for column in range(3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)
        self.properties_label, self.annotation_label = QLabel(), QLabel()
        for label in (self.properties_label, self.annotation_label):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(label)

    def update_summary(self, total, dataset_type, result=None, *, loading=False, checking=False, progress=(0, 0)):
        self.captions['labelled'].setText('С масками' if dataset_type == 1 else 'С TXT')
        summary = result.get('summary', {}) if result and total else {}
        for key, label in self.values.items():
            value = '—' if total or loading else '0'
            if key == 'images':
                value = '—' if loading else str(total)
            elif not loading and summary:
                value = str(summary[key])
            label.setText(value)
        if loading:
            self.status.setText('Загрузка списка снимков…')
        elif not total:
            self.status.setText('Снимки не загружены')
        elif summary:
            self.status.setText(f"Проверено: {summary['checked']}/{total}")
        elif checking:
            self.status.setText(f'Проверка: {progress[0]}/{total}…')
        else:
            self.status.setText('Проверка не завершена — отменена или произошла ошибка')
        self.status.setToolTip('Проверено — все обработанные снимки, включая снимки с ошибками.')
        if not summary or loading:
            self.properties_label.setText('Свойства снимков появятся после фоновой проверки.' if total or loading else '')
            self.annotation_label.setText('')
            return
        colors, bits, types = result['colors'], result['bits'], result['types']
        color_names = (('color', 'цветные'), ('gray_real', 'Ч/Б, 1 канал'), ('gray_visual', 'визуально Ч/Б'), ('unknown', 'не определён'))
        bit_names = (('uint8_png', '8 бит PNG'), ('uint8_other', '8 бит, другой формат'), ('uint16', '16 бит'),
                     ('int16', 'int16'), ('float', 'float'), ('other', 'другой тип'), ('unknown', 'не определена'))
        color_text = '; '.join(f'{caption}: {colors[key]}' for key, caption in color_names if colors.get(key))
        bit_text = '; '.join(f'{caption}: {bits[key]}' for key, caption in bit_names if bits.get(key))
        self.properties_label.setText(f'Цвет: {color_text or "—"}\nРазрядность: {bit_text or "—"}')
        if dataset_type == 2:
            self.annotation_label.setText('Режим «Без разметки»: TXT и маски не используются.')
        else:
            kinds = '; '.join(f'{caption}: {types[key]}' for key, caption in
                              (('detect', 'BBox'), ('obb', 'OBB'), ('segment', 'Segment'), ('mixed', 'смешанная')) if types.get(key))
            self.annotation_label.setText(f"Классы объектов: {summary['classes']}" + (f' · Снимки по типу разметки: {kinds}' if kinds else ''))
        self.annotation_label.setToolTip('Без разметки — нет файла. Размеченный фон — файл есть, объектов нет. Ошибки учитываются отдельно.')
