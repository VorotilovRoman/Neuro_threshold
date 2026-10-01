# ui/image_navigation.py
from import_libs_external import *

from utils.settings import settings

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


# ============================================================
# Горячие клавиши навигации по снимкам (Num+4 / Num+6 и др.)
# ============================================================
from PyQt5.QtCore import QEvent, QObject


def parse_key_spec(spec):
    """
    Парсит строку вида:
        "4"          -> (Qt.Key_4, False)
        "Num+4"      -> (Qt.Key_4, True)
        "Keypad+6"   -> (Qt.Key_6, True)
        "Left"       -> (Qt.Key_Left, False)
        "Space"      -> (Qt.Key_Space, False)
    Возвращает (Qt.Key, requires_keypad_modifier) либо (None, False).
    """
    if not spec:
        return None, False
    s = str(spec).strip()
    low = s.lower()

    requires_keypad = False
    for prefix in ("num+", "numpad+", "keypad+", "kp+"):
        if low.startswith(prefix):
            requires_keypad = True
            s = s[len(prefix):]
            low = s.lower()
            break

    # Односимвольная цифра или буква
    if len(s) == 1:
        key = getattr(Qt, f"Key_{s.upper()}", None)
        if key is not None:
            return key, requires_keypad

    # Именованные константы
    for name in (f"Key_{s}", f"Key_{s.capitalize()}", f"Key_{s.upper()}"):
        key = getattr(Qt, name, None)
        if key is not None:
            return key, requires_keypad

    return None, False


class NavigationShortcutInstaller(QObject):
    """
    Устанавливает горячие клавиши prev/next для конкретной вкладки.

    Логика:
      • event filter ставится на QApplication — ловит все нажатия клавиш
        независимо от того, на каком дочернем виджете фокус;
      • срабатывает только если вкладка сейчас активна в QTabWidget;
      • при фокусе в QLineEdit/QSpinBox/QTextEdit не-numpad клавиши
        игнорируются, чтобы не мешать вводу;
      • настройки перечитываются по сигналу settings.settings_changed.
    """
    def __init__(self, tab_widget, prev_callback, next_callback, parent=None):
        super().__init__(parent or tab_widget)
        self._tab = tab_widget
        self._prev_cb = prev_callback
        self._next_cb = next_callback
        self._prev_key = (Qt.Key_4, True)
        self._next_key = (Qt.Key_6, True)
        self._reload_from_settings()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        try:
            settings.settings_changed.connect(self._on_settings_changed)
        except Exception:
            pass

    # --- Настройки ---
    def _reload_from_settings(self):
        cfg = settings.current.get("navigation_shortcuts", {}) or {}
        pk, pkp = parse_key_spec(cfg.get("prev", "Num+4"))
        nk, nkp = parse_key_spec(cfg.get("next", "Num+6"))
        self._prev_key = (pk if pk is not None else Qt.Key_4, pkp)
        self._next_key = (nk if nk is not None else Qt.Key_6, nkp)

    def _on_settings_changed(self, *_):
        self._reload_from_settings()

    # --- Проверки ---
    def _is_active_tab(self):
        # Ищем QTabWidget, в котором живёт эта вкладка
        p = self._tab
        while p is not None:
            if isinstance(p, QTabWidget):
                try:
                    return p.currentWidget() is self._tab
                except Exception:
                    return False
            p = p.parentWidget()
        # Если QTabWidget не найден (окно как отдельное) — просто по видимости
        return self._tab.isVisible()

    @staticmethod
    def _focus_is_text_input():
        from PyQt5.QtWidgets import QAbstractSpinBox
        w = QApplication.focusWidget()
        return isinstance(w, (QLineEdit, QTextEdit, QAbstractSpinBox))

    # --- Обработчик ---
    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and self._is_active_tab():
            key = event.key()
            mods = event.modifiers()
            is_keypad = bool(mods & Qt.KeypadModifier)

            pk, pkp = self._prev_key
            nk, nkp = self._next_key

            # Prev
            if key == pk and (not pkp or is_keypad):
                if pkp or not self._focus_is_text_input():
                    self._prev_cb()
                    return True
            # Next
            if key == nk and (not nkp or is_keypad):
                if nkp or not self._focus_is_text_input():
                    self._next_cb()
                    return True
        return super().eventFilter(obj, event)