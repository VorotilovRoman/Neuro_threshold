"""Common Qt preset and panel actions; schema-specific binding stays separate."""
from PyQt5.QtWidgets import QInputDialog, QMessageBox


def update_preset_combo(self):
    previous = self.preset_combo.blockSignals(True)
    try:
        self.preset_combo.clear()
        self.preset_manager.load_presets()
        for name in self.preset_manager.get_preset_names():
            self.preset_combo.addItem(name)
        self.preset_combo.setCurrentText('default')
    finally:
        self.preset_combo.blockSignals(previous)


def save_as_preset(self):
    name, ok = QInputDialog.getText(self, "Сохранить пресет", "Введите имя пресета:")
    if ok and name:
        if name in self.preset_manager.get_preset_names():
            reply = QMessageBox.question(
                self, "Перезаписать пресет",
                f"Пресет '{name}' уже существует. Перезаписать?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return
            self.preset_manager.presets[name] = self.preset_manager.get_current_settings()
            self.preset_manager.save_presets()
        else:
            self.preset_manager.add_preset(name)
        self.update_preset_combo()
        self.preset_combo.setCurrentText(name)
        self.log(f"Сохранён пресет: {name}")

def delete_current_preset(self):
    current = self.preset_combo.currentText()
    reply = QMessageBox.question(
        self, "Удалить пресет",
        f"Удалить пресет '{current}'?",
        QMessageBox.Yes | QMessageBox.No
    )
    if reply == QMessageBox.Yes:
        if self.preset_manager.delete_preset(current):
            self.update_preset_combo()
            self.log(f"Удалён пресет: {current}")

def toggle_log(self, checked):
    self.log_widget.setVisible(checked)
    self.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")

def toggle_histogram(self, checked):
    self.hist_container.setVisible(checked)
    self.toggle_hist_btn.setText("Скрыть гистограмму" if checked else "Показать гистограмму")



def log(owner, message):
    owner.log_widget.log(message)
    print(message)
