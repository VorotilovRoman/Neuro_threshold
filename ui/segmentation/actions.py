"""Shared user actions for segmentation windows and their current presets."""

from copy import deepcopy

from PyQt5.QtWidgets import QInputDialog, QMessageBox


def update_preset_combo(owner):
    previous = owner.preset_combo.blockSignals(True)
    try:
        owner.preset_combo.clear()
        owner.preset_manager.load_presets()
        owner.preset_combo.addItems(owner.preset_manager.get_preset_names())
        owner.preset_combo.setCurrentText("default")
    finally:
        owner.preset_combo.blockSignals(previous)


def apply_preset(owner, index):
    name = owner.preset_combo.itemText(index)
    if not name:
        return
    try:
        if owner.preset_manager.apply_preset(name):
            owner.log(f"Применён пресет: {name}")
    except (OSError, ValueError, TypeError) as exc:
        owner.log(f"Ошибка пресета {name}: {exc}")
        QMessageBox.warning(owner, "Не удалось применить пресет", str(exc))


def save_as_preset(owner):
    name, ok = QInputDialog.getText(owner, "Сохранить пресет", "Введите имя пресета:")
    if not ok or not name:
        return
    manager = owner.preset_manager
    if name in manager.presets:
        if (
            QMessageBox.question(
                owner,
                "Перезаписать пресет",
                f"Пресет '{name}' уже существует. Перезаписать?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            != QMessageBox.Yes
        ):
            return
    previous = deepcopy(manager.presets)
    try:
        manager.add_preset(name)
    except (OSError, ValueError, TypeError) as exc:
        manager.presets = previous
        QMessageBox.critical(owner, "Не удалось сохранить пресет", str(exc))
        return
    owner.update_preset_combo()
    owner.preset_combo.setCurrentText(name)
    owner.log(f"Сохранён пресет: {name}")


def delete_current_preset(owner):
    name = owner.preset_combo.currentText()
    if (
        QMessageBox.question(
            owner,
            "Удалить пресет",
            f"Удалить пресет '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        != QMessageBox.Yes
    ):
        return
    manager = owner.preset_manager
    previous = deepcopy(manager.presets)
    try:
        if manager.delete_preset(name):
            owner.update_preset_combo()
            owner.log(f"Удалён пресет: {name}")
    except (OSError, ValueError, TypeError) as exc:
        manager.presets = previous
        QMessageBox.critical(owner, "Не удалось удалить пресет", str(exc))


def toggle_log(owner, checked):
    owner.log_widget.setVisible(checked)
    owner.toggle_log_btn.setText("Скрыть лог" if checked else "Показать лог")


def toggle_histogram(owner, checked):
    owner.hist_container.setVisible(checked)
    owner.toggle_hist_btn.setText(
        "Скрыть гистограмму" if checked else "Показать гистограмму"
    )


def log(owner, message):
    owner.log_widget.log(message)
    print(message)
