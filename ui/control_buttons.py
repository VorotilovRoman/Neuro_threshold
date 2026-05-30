# widgets/control_buttons.py
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt5.QtCore import pyqtSignal

class ControlButtons(QWidget):
    generate = pyqtSignal()
    cancel = pyqtSignal()
    save = pyqtSignal()

    def __init__(self, show_generate=True, show_cancel=True, show_save=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        if show_generate:
            self._generate_btn = QPushButton("Generate")
            self._generate_btn.clicked.connect(self.generate.emit)
            layout.addWidget(self._generate_btn)

        if show_cancel:
            self._cancel_btn = QPushButton("Cancel")
            self._cancel_btn.clicked.connect(self.cancel.emit)
            layout.addWidget(self._cancel_btn)

        if show_save:
            self._save_btn = QPushButton("Save")
            self._save_btn.clicked.connect(self.save.emit)
            layout.addWidget(self._save_btn)

        layout.addStretch()

    def set_generate_enabled(self, enabled):
        if hasattr(self, '_generate_btn'):
            self._generate_btn.setEnabled(enabled)

    def set_cancel_enabled(self, enabled):
        if hasattr(self, '_cancel_btn'):
            self._cancel_btn.setEnabled(enabled)

    def set_save_enabled(self, enabled):
        if hasattr(self, '_save_btn'):
            self._save_btn.setEnabled(enabled)