from import_libs_external import *


class ImageNavigationWidget(QWidget):
    load_images = pyqtSignal()
    load_folder = pyqtSignal()
    prev = pyqtSignal()
    next = pyqtSignal()
    goto_page = pyqtSignal(int)   # новый сигнал для перехода к конкретному изображению (индекс с 1)
    resize_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        self._load_images_btn = QPushButton("Load Images")
        self._load_folder_btn = QPushButton("Load Folder")
        self._prev_btn = QPushButton("◀ Previous")
        self._next_btn = QPushButton("Next ▶")
        self._page_spin = QSpinBox()
        self._page_spin.setRange(1, 1)          # будет обновляться при загрузке изображений
        self._page_spin.setFixedWidth(80)
        self._page_spin.setAlignment(Qt.AlignCenter)
        self._page_spin.setSuffix(" / ?")       # временный суффикс, обновится при set_total
        self._resize_cb = QCheckBox("Resize to max 1024px")
        self._resize_cb.setChecked(True)

        layout.addWidget(self._load_images_btn)
        layout.addWidget(self._load_folder_btn)
        layout.addWidget(self._prev_btn)
        layout.addWidget(self._page_spin)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._resize_cb)
        layout.addStretch()

        # Сигналы
        self._load_images_btn.clicked.connect(self.load_images.emit)
        self._load_folder_btn.clicked.connect(self.load_folder.emit)
        self._prev_btn.clicked.connect(self.prev.emit)
        self._next_btn.clicked.connect(self.next.emit)
        self._page_spin.valueChanged.connect(self._on_page_changed)
        self._resize_cb.toggled.connect(self.resize_toggled.emit)

        self._total = 0
        self._current = 0   # индекс с 0

    def _on_page_changed(self, value):
        """При изменении номера страницы (с 1) испускаем сигнал goto_page."""
        if value != self._current + 1:   # предотвращаем рекурсию при программной установке
            self.goto_page.emit(value)

    def set_navigation_enabled(self, enabled):
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._page_spin.setEnabled(enabled)

    def set_prev_enabled(self, enabled):
        self._prev_btn.setEnabled(enabled)

    def set_next_enabled(self, enabled):
        self._next_btn.setEnabled(enabled)

    def set_resize_checked(self, checked):
        self._resize_cb.setChecked(checked)

    def is_resize_enabled(self):
        return self._resize_cb.isChecked()

    def set_current_index(self, idx, total):
        """
        Устанавливает текущий отображаемый индекс и общее количество изображений.
        idx – индекс с 0 (внутреннее представление).
        """
        self._current = idx
        self._total = total
        self._page_spin.blockSignals(True)
        self._page_spin.setRange(1, max(1, total))
        self._page_spin.setValue(idx + 1)
        self._page_spin.setSuffix(f" / {total}" if total > 0 else " / ?")
        self._page_spin.blockSignals(False)

        # Обновляем состояние кнопок (обычно это делает родитель, но можно и здесь)
        self._prev_btn.setEnabled(total > 0 and idx > 0)
        self._next_btn.setEnabled(total > 0 and idx < total - 1)