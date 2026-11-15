"""Read-only comparison tables; sorting/filtering never changes the evaluation."""

import math
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView, QHeaderView, QTableView
from utils_new.annotations.modes import MODE_LABELS

SORT_ROLE = Qt.UserRole + 1
KIND_LABELS = {"threshold": "Пороговые", "gradient": "Градиентные"}
PERCENT_FIELDS = {
    "score",
    "precision",
    "recall",
    "avg_iou_tp",
    "coverage",
    "mean_f1",
    "f1",
}
SETTING_COLUMNS = [
    (
        "rank",
        "Место",
        "Позиция по измеренному качеству на всём эталоне. Без полного покрытия место не назначается.",
    ),
    ("kind", "Вид", "Семейство классических методов."),
    (
        "method",
        "Метод",
        "Метод сегментации. Полные параметры выбранной строки показаны справа.",
    ),
    (
        "score",
        "F1",
        "Баланс точности и полноты: 100% — все объекты найдены без лишних; выше — лучше. Не mAP.",
    ),
    (
        "precision",
        "Точность",
        "Доля правильных объектов среди предсказаний. Высокая точность означает меньше лишних объектов.",
    ),
    (
        "recall",
        "Полнота",
        "Доля найденных объектов среди всех объектов эталона. Высокая полнота означает меньше пропусков.",
    ),
    ("fp", "Лишние", "FP: сколько лишних объектов выделено на проверенных снимках."),
    (
        "fn",
        "Пропущено",
        "FN: сколько эталонных объектов не найдено на проверенных снимках.",
    ),
    (
        "avg_iou_tp",
        "IoU совпадений",
        "Насколько точно совпали рамки/контуры найденных объектов. Высокий IoU означает лучшую локализацию.",
    ),
    (
        "evaluated_images",
        "Снимков проверено",
        "Успешно проверенные снимки / все снимки эталона. Для рекомендации нужны все снимки.",
    ),
    (
        "annotation_mode",
        "Разметка",
        "BBox — осевая рамка; OBB — повёрнутая рамка; Segment — контур.",
    ),
]
IMAGE_COLUMNS = [
    ("image_id", "Снимок", "Имя снимка в эталонном датасете."),
    (
        "status",
        "Состояние",
        "Ошибка или отсутствие вычисления не считается успешным пустым предсказанием.",
    ),
    ("true_objects", "В эталоне", "Сколько объектов размечено вручную."),
    ("predicted_objects", "Найдено", "Сколько объектов выделил метод."),
    ("tp", "Совпало", "TP: найденные и однозначно сопоставленные объекты."),
    ("fp", "Лишние", "FP: объекты без соответствия в ручной разметке."),
    ("fn", "Пропущено", "FN: эталонные объекты без соответствия в результате метода."),
    ("f1", "F1 снимка", "Качество выбранного пресета на этом снимке."),
    (
        "error",
        "Причина / что исправить",
        "Подробная причина отсутствия корректного результата.",
    ),
]


class ComparisonTableModel(QAbstractTableModel):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.frame = pd.DataFrame()

    def set_frame(self, frame):
        self.beginResetModel()
        self.frame = frame.reset_index(drop=True).copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return self.columns[section][1]
            if role == Qt.ToolTipRole:
                return self.columns[section][2]
        if role == Qt.DisplayRole:
            return section + 1
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self.frame.iloc[index.row()]
        key, _, help_text = self.columns[index.column()]
        value = row.get(key)
        missing = value is None or pd.isna(value)
        if role == Qt.UserRole:
            return row.get("recipe_key")
        if role == Qt.ToolTipRole:
            return "\n".join(
                filter(
                    None,
                    [
                        help_text,
                        str(row.get("exclusion_reason", "")),
                        str(row.get("error", "")),
                    ],
                )
            )
        if role == SORT_ROLE:
            if missing:
                return math.inf if key == "rank" else -math.inf
            if isinstance(value, (int, float)) or key in (
                "rank",
                "tp",
                "fp",
                "fn",
                "evaluated_images",
                "true_objects",
                "predicted_objects",
            ):
                return float(value)
            return str(value).casefold()
        if role != Qt.DisplayRole:
            return None
        if missing:
            return "—"
        if key in PERCENT_FIELDS:
            return f"{float(value) * 100:.1f}%"
        if key == "kind":
            return KIND_LABELS.get(value, value)
        if key == "annotation_mode":
            return MODE_LABELS.get(value, value)
        if key == "status":
            return {
                "ok": "Проверено",
                "error": "Ошибка",
                "missing": "Нет вычисления",
            }.get(value, value)
        if key == "evaluated_images":
            return f"{int(value)} / {int(row['expected_images'])}"
        if key in ("rank", "tp", "fp", "fn", "true_objects", "predicted_objects"):
            return str(int(value))
        return str(value)


class ComparisonTable(QTableView):
    def __init__(self, columns=SETTING_COLUMNS, parent=None):
        super().__init__(parent)
        self.source = ComparisonTableModel(columns, self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.source)
        self.proxy.setSortRole(SORT_ROLE)
        self.setModel(self.proxy)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def set_frame(self, frame):
        self.source.set_frame(frame)
        self.resizeColumnsToContents()
        for column in range(self.source.columnCount()):
            self.setColumnWidth(column, min(270, max(70, self.columnWidth(column))))

    def selected_frame(self):
        indices = sorted(
            {
                self.proxy.mapToSource(index).row()
                for index in self.selectionModel().selectedRows()
            }
        )
        return self.source.frame.iloc[indices].copy()

    def select_recipe(self, key):
        if "recipe_key" not in self.source.frame:
            return False
        for row in range(self.source.rowCount()):
            if self.source.frame.iloc[row]["recipe_key"] == key:
                self.selectRow(
                    self.proxy.mapFromSource(self.source.index(row, 0)).row()
                )
                return True
        return False
