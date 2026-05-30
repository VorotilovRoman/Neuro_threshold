from import_libs_external import *


class LogWidget(QWidget):
    cleared = pyqtSignal()

    def __init__(self, show_clear_btn=True, show_progress=True, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        if show_clear_btn:
            header = QHBoxLayout()
            header.addWidget(QLabel("Лог выполнения"))
            header.addStretch()
            self._clear_btn = QPushButton("Очистить лог")
            self._clear_btn.clicked.connect(self._on_clear)
            header.addWidget(self._clear_btn)
            layout.addLayout(header)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFontFamily("Courier New")
        layout.addWidget(self._text)

        if show_progress:
            self._progress = QProgressBar()
            self._progress.setRange(0, 100)
            self._progress.setValue(0)
            self._progress.setTextVisible(True)
            layout.addWidget(self._progress)

    def _on_clear(self):
        self._text.clear()
        self.cleared.emit()

    def log(self, message):
        self._text.append(message)

    def clear(self):
        self._text.clear()

    def set_progress(self, value, max_val=None, fmt=None):
        if hasattr(self, '_progress'):
            if max_val is not None:
                self._progress.setRange(0, max_val)
            self._progress.setValue(value)
            if fmt:
                self._progress.setFormat(fmt)

    @property
    def text(self):
        return self._text