"""Read-only object inspector: stable IDs, inclusion checks and one focused object."""

import numpy as np
from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    pyqtSignal,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from utils_new.annotations.model import ImageSize
from utils_new.annotations.modes import MODE_LABELS
from utils_new.annotations.object_info import ObjectInfo
from utils_new.dataset_yolo_labels_new import serialize_yolo_labels

SORT_ROLE = Qt.UserRole + 1
SEARCH_ROLE = Qt.UserRole + 2


class ObjectTableModel(QAbstractTableModel):
    checks_changed = pyqtSignal(object)
    HEADERS = ("✓", "№", "Тип", "Класс", "Габариты", "Доля, %")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.objects = []
        self.checked = set()
        self.size = (1, 1)
        self.normalized_units = False
        self.geometry = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.objects)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        if orientation == Qt.Horizontal and role == Qt.ToolTipRole:
            return (
                "Отметка включает объект в отображение и сохранение.",
                "Номер в полном результате. Сортировка и поиск его не меняют.",
                "Тип разметки датасета.",
                "ID класса. Эти методы выделяют объекты класса 0.",
                "Ширина × высота осевого охватывающего прямоугольника в выбранных единицах.",
                "Площадь геометрии / площадь снимка. Для BBox/OBB — площадь рамки, для Segment — полигона; это не площадь маски.",
            )[section]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        info = self.objects[index.row()]
        column = index.column()
        shape = self.geometry[index.row()]
        if role == Qt.CheckStateRole and column == 0:
            return Qt.Checked if info.index in self.checked else Qt.Unchecked
        if role == Qt.UserRole:
            return info.index
        if role == SEARCH_ROLE:
            return f"№{info.index + 1} #{info.index + 1} {MODE_LABELS[info.normalized.kind]} класс {info.normalized.class_id}".casefold()
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)
        if role == Qt.ToolTipRole:
            return self.headerData(column, Qt.Horizontal, Qt.ToolTipRole)
        if role == SORT_ROLE:
            return (
                int(info.index in self.checked),
                info.index,
                MODE_LABELS[info.normalized.kind],
                info.normalized.class_id,
                shape.width * shape.height,
                info.area_fraction,
            )[column]
        if role == Qt.DisplayRole:
            precision = 6 if self.normalized_units else 1
            return (
                "",
                info.index + 1,
                MODE_LABELS[info.normalized.kind],
                info.normalized.class_id,
                f"{shape.width:.{precision}f} × {shape.height:.{precision}f}",
                f"{info.area_fraction * 100:.3f}",
            )[column]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return flags | Qt.ItemIsUserCheckable if index.column() == 0 else flags

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.CheckStateRole or index.column() != 0:
            return False
        checked = set(self.checked)
        object_id = self.objects[index.row()].index
        if value == Qt.Checked:
            checked.add(object_id)
        else:
            checked.discard(object_id)
        self.set_checked(checked)
        return True

    def set_objects(self, objects):
        self.beginResetModel()
        self.objects = list(objects)
        self.checked = {info.index for info in objects}
        self.geometry = [info.geometry(self.size) for info in self.objects]
        self.endResetModel()

    def set_units(self, size, normalized=False):
        self.size = size
        self.normalized_units = normalized
        self.geometry = [info.geometry(size) for info in self.objects]
        if self.objects:
            self.dataChanged.emit(
                self.index(0, 4), self.index(len(self.objects) - 1, 4)
            )

    def set_checked(self, indices):
        updated = set(indices) & {info.index for info in self.objects}
        if updated == self.checked:
            return
        self.checked = updated
        if self.objects:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self.objects) - 1, 0),
                [Qt.CheckStateRole, SORT_ROLE],
            )
        self.checks_changed.emit(sorted(updated))


class ObjectFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = ""
        self.setSortRole(SORT_ROLE)

    def filterAcceptsRow(self, row, parent):
        text = self.sourceModel().index(row, 0, parent).data(SEARCH_ROLE)
        return all(token in text for token in self.query.casefold().split())

    def set_query(self, text):
        self.query = text
        self.invalidateFilter()


class VertexTableModel(QAbstractTableModel):
    """Virtual table: large contours don't create a QWidget/item per vertex."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []

    def set_points(self, points, normalized=False):
        self.beginResetModel()
        self.points = points
        self.precision = 6 if normalized else 2
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.points)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 3

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return ("Вершина", "X", "Y")[section]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            if index.column() == 0:
                return index.row() + 1
            return f"{self.points[index.row()][index.column() - 1]:.{self.precision}f}"


class AnnotationObjectsWidget(QWidget):
    checked_changed = pyqtSignal(object)
    focused_changed = pyqtSignal(int)
    zoom_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_size = self.working_size = (1, 1)
        self.focused_index = -1
        self._syncing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Объекты на снимке")
        title.setStyleSheet("font-weight: bold")
        layout.addWidget(title)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: №, тип, класс")
        self.search.setAccessibleName("Поиск объектов")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip(
            "Поиск меняет только таблицу. Отметки отображения и экспорта сохраняются."
        )
        layout.addWidget(self.search)
        row = QHBoxLayout()
        row.addWidget(QLabel("Отметить:"))
        self.all_button = QPushButton("Все")
        self.none_button = QPushButton("Ни одного")
        self.invert_button = QPushButton("Инверсия")
        for button in (self.all_button, self.none_button, self.invert_button):
            row.addWidget(button)
            button.setToolTip(
                "Применяется ко всем объектам, включая скрытые поиском строки."
            )
        layout.addLayout(row)
        self.units = QComboBox()
        self.units.addItem("Координаты: исходник (px)", "source")
        self.units.addItem("Координаты: обработка (px)", "working")
        self.units.addItem("Координаты: нормализованные (0…1)", "normalized")
        self.units.setAccessibleName("Единицы геометрии объектов")
        layout.addWidget(self.units)
        self.model = ObjectTableModel(self)
        self.proxy = ObjectFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setAccessibleName("Объекты на снимке")
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setMinimumSectionSize(28)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().resizeSection(0, 32)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(1, Qt.AscendingOrder)
        self.table.setMinimumHeight(150)
        self.table.setMaximumHeight(260)
        layout.addWidget(self.table)
        self.empty_label = QLabel("Нет объектов.")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        self.details_toggle = QToolButton()
        self.details_toggle.setText("Свойства объекта")
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.details_toggle.setArrowType(Qt.RightArrow)
        self.details_toggle.setCheckable(True)
        layout.addWidget(self.details_toggle)
        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        self.properties = QLabel("Выберите строку или объект на снимке.")
        self.properties.setWordWrap(True)
        self.properties.setTextInteractionFlags(Qt.TextSelectableByMouse)
        details_layout.addWidget(self.properties)
        self.vertices = VertexTableModel(self)
        self.vertex_table = QTableView()
        self.vertex_table.setModel(self.vertices)
        self.vertex_table.setAccessibleName("Вершины выбранного объекта")
        self.vertex_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.vertex_table.verticalHeader().hide()
        self.vertex_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vertex_table.setMaximumHeight(125)
        details_layout.addWidget(self.vertex_table)
        self.zoom_button = QPushButton("Показать на снимке")
        self.zoom_button.setToolTip(
            "Приблизить отмеченный объект. Также доступно двойным щелчком по строке."
        )
        details_layout.addWidget(self.zoom_button)
        layout.addWidget(self.details)
        self.details.hide()
        row = QHBoxLayout()
        row.addWidget(QLabel("Копировать YOLO:"))
        self.copy_focused = QPushButton("Объект")
        self.copy_checked = QPushButton("Отмеченные")
        for button in (self.copy_focused, self.copy_checked):
            button.setToolTip(
                "Копировать координаты в буфер обмена. Файлы не изменяются; формат совпадает с сохранением TXT."
            )
            row.addWidget(button)
        layout.addLayout(row)
        self.model.checks_changed.connect(self._checks_changed)
        self.search.textChanged.connect(self._filter_changed)
        self.units.currentIndexChanged.connect(self._units_changed)
        self.table.selectionModel().currentRowChanged.connect(self._current_changed)
        self.table.doubleClicked.connect(lambda _: self._zoom())
        self.zoom_button.clicked.connect(self._zoom)
        self.all_button.clicked.connect(
            lambda: self.set_checked_indices(range(len(self.model.objects)))
        )
        self.none_button.clicked.connect(lambda: self.set_checked_indices([]))
        self.invert_button.clicked.connect(
            lambda: self.set_checked_indices(
                set(range(len(self.model.objects))) - self.model.checked
            )
        )
        self.copy_focused.clicked.connect(lambda: self._copy([self.focused_index]))
        self.copy_checked.clicked.connect(lambda: self._copy(self.checked_indices))
        self.details_toggle.toggled.connect(self._toggle_details)
        self._update_summary()
        self._update_details()

    @property
    def checked_indices(self):
        return sorted(self.model.checked)

    def set_objects(self, values, source_size, working_size):
        objects = [
            ObjectInfo.from_tuple(index, value) for index, value in enumerate(values)
        ]
        self._syncing = True
        try:
            self.focused_index = -1
            self.source_size, self.working_size = (
                tuple(source_size),
                tuple(working_size),
            )
            self.model.set_objects(objects)
            self._units_changed()
        finally:
            self._syncing = False
        self._update_summary()
        self._update_details()

    def clear(self):
        self.set_objects([], (1, 1), (1, 1))

    def set_checked_indices(self, indices):
        self.model.set_checked(indices)

    def focus_object(self, object_id, *, reveal=False):
        object_id = object_id if 0 <= object_id < len(self.model.objects) else -1
        changed = object_id != self.focused_index
        self.focused_index = object_id
        self._syncing = True
        try:
            proxy_index = self.proxy.mapFromSource(self.model.index(object_id, 0))
            if object_id >= 0 and not proxy_index.isValid() and reveal:
                self.search.clear()
                proxy_index = self.proxy.mapFromSource(self.model.index(object_id, 0))
            self.table.setCurrentIndex(proxy_index)
            if proxy_index.isValid():
                self.table.scrollTo(proxy_index)
        finally:
            self._syncing = False
        self._update_details()
        if changed:
            self.focused_changed.emit(object_id)

    def _filter_changed(self, text):
        focused = self.focused_index
        self._syncing = True
        try:
            self.proxy.set_query(text)
        finally:
            self._syncing = False
        self.focus_object(focused)
        self._update_summary()

    def _current_changed(self, current, previous):
        if not self._syncing and current.isValid():
            self.focus_object(current.data(Qt.UserRole))

    def _checks_changed(self, indices):
        self._update_summary()
        self._update_details()
        self.checked_changed.emit(indices)

    def _units_changed(self, *_):
        unit = self.units.currentData()
        size = {
            "source": self.source_size,
            "working": self.working_size,
            "normalized": (1, 1),
        }[unit]
        self.model.set_units(size, unit == "normalized")
        self._update_details()

    def _update_summary(self):
        count = len(self.model.objects)
        self.summary.setText(
            f"Всего: {count} · отмечено: {len(self.model.checked)} · в таблице: {self.proxy.rowCount()}"
        )
        self.empty_label.setVisible(self.proxy.rowCount() == 0)
        self.empty_label.setText(
            "Нет совпадений. Отметки сохранены." if count else "Нет объектов."
        )
        for button in (self.all_button, self.none_button, self.invert_button):
            button.setEnabled(count > 0)
        self.copy_checked.setEnabled(bool(self.model.checked))

    def _update_details(self):
        valid = 0 <= self.focused_index < len(self.model.objects)
        self.copy_focused.setEnabled(valid)
        self.zoom_button.setEnabled(valid and self.focused_index in self.model.checked)
        if not valid:
            self.details_toggle.setText("Свойства объекта")
            self.properties.setText("Выберите строку или объект на снимке.")
            self.vertices.set_points([])
            return
        info = self.model.objects[self.focused_index]
        geometry = self.model.geometry[self.focused_index]
        normalized = self.model.normalized_units
        precision = 6 if normalized else 2
        suffix = "" if normalized else " px"
        fmt = lambda values: "; ".join(f"{value:.{precision}f}" for value in values)
        included = (
            "отмечен"
            if self.focused_index in self.model.checked
            else "исключён из отображения и экспорта"
        )
        self.details_toggle.setText(f"Свойства объекта №{info.index + 1}")
        text = (
            f"№{info.index + 1} · {MODE_LABELS[info.normalized.kind]} · класс {info.normalized.class_id} · {included}\n"
            f"Исходник: {self.source_size[0]} × {self.source_size[1]} px; обработка: {self.working_size[0]} × {self.working_size[1]} px\n"
            f"Центр габаритов: ({fmt(geometry.center)}){suffix}\n"
            f"Границы X₀; Y₀; X₁; Y₁: {fmt(geometry.bounds)}{suffix}\n"
            f"Площадь геометрии: {geometry.area:.{precision}f}{' px²' if not normalized else ' от снимка'}; вершин: {len(geometry.points)}"
        )
        if info.normalized.kind == "obb":
            points = info.geometry(self.source_size).points
            edges = np.roll(points, -1, axis=0) - points
            lengths = np.linalg.norm(edges, axis=1)
            edge = edges[np.argmax(lengths)]
            angle = np.degrees(np.arctan2(edge[1], edge[0])) % 180
            sides = sorted(
                ((lengths[0] + lengths[2]) / 2, (lengths[1] + lengths[3]) / 2),
                reverse=True,
            )
            text += f"\nСтороны OBB в исходнике: {sides[0]:.2f} × {sides[1]:.2f} px"
            text += (
                f"\nУгол длинного ребра в исходнике: {angle:.2f}° (от оси X, Y вниз)"
            )
        self.properties.setText(text)
        self.vertices.set_points(geometry.points, normalized)

    def _toggle_details(self, checked):
        self.details.setVisible(checked)
        self.details_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _zoom(self):
        if self.focused_index in self.model.checked:
            self.zoom_requested.emit(self.focused_index)

    def yolo_text(self, indices):
        annotations = [
            self.model.objects[index].source_annotation(self.source_size)
            for index in sorted(set(indices))
            if 0 <= index < len(self.model.objects)
        ]
        return serialize_yolo_labels(annotations, ImageSize(*self.source_size))

    def _copy(self, indices):
        text = self.yolo_text(indices)
        if text:
            QApplication.clipboard().setText(text)
