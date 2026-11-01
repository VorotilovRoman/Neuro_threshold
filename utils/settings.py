# settings.py
from import_libs_external import *
from copy import deepcopy
from path_setup import get_project_root


# Исходные BGR-цвета классов 0..9 из utils.segmentation_ops.CLASS_COLORS.
# Этот модуль не импортирует segmentation_ops, чтобы избежать циклического импорта.
DEFAULT_DATASET_CLASS_PALETTE_BGR = [
    [0, 0, 200], [0, 130, 220], [0, 150, 150], [0, 140, 0],
    [170, 110, 0], [200, 0, 0], [180, 0, 180], [110, 0, 110],
    [19, 69, 139], [100, 100, 0],
]

FONT_SIZE_PX_MIN, FONT_SIZE_PX_MAX = 8, 48
LINE_THICKNESS_PX_MIN, LINE_THICKNESS_PX_MAX = 1, 12
FONT_SIZE_PX_DEFAULT, LINE_THICKNESS_PX_DEFAULT = 11, 1



class SettingsManager(QObject):
    settings_changed = pyqtSignal(dict)

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__()
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Используем path_setup для получения корня проекта
        root = get_project_root()
        self.settings_file = os.path.join(root, "settings", "settings.json")

        self.defaults = {
            "theme": "dark",
            "primary_color": [42, 130, 218],
            "colors": {
                "annotation": [255, 0, 0],
                "selected": [0, 255, 255],
                "crosshair": [0, 255, 0],
                "edit_points": [0, 0, 255],
                "label_text": [255, 0, 0]
            },
            # Размеры в логических пикселях Qt (для старых OpenCV вкладок
            # getter-множители вычисляются относительно прежних базовых размеров).
            "font_size_px": FONT_SIZE_PX_DEFAULT,
            "line_thickness_px": LINE_THICKNESS_PX_DEFAULT,
            # Старые поля сохраняем: старый settings.json мигрирует без потери.
            "font_size_mode": "auto",
            "line_thickness_mode": "auto",
            # Датасет имеет отдельную палитру: все десять прежних цветов.
            "dataset_class_palette": deepcopy(DEFAULT_DATASET_CLASS_PALETTE_BGR),
            # Старый V3-флаг сохраняется для обратной совместимости.
            # Новая вкладка разметки всегда использует class_palette.
            "annotation_view": {"use_class_colors": True},
            "navigation_shortcuts": {
                "prev": "Num+4",
                "next": "Num+6"
            }
        }
        self.current = self.load()

    @staticmethod
    def _with_defaults(data, defaults):
        """Добавляет недостающие значения, сохраняя незнакомые ключи.

        В частности, старые settings.json с неполной секцией colors не
        теряют существующие цвета при добавлении новых настроек.
        """
        if not isinstance(data, dict):
            return deepcopy(defaults)
        result = deepcopy(data)
        for key, default in defaults.items():
            if key not in result:
                result[key] = deepcopy(default)
            elif isinstance(default, dict):
                result[key] = SettingsManager._with_defaults(result[key], default)
        return result

    @staticmethod
    def _migrate_legacy_size_settings(data):
        """Перевести старые режимы в пиксели, только если px ещё не заданы."""
        if not isinstance(data, dict):
            return data
        result = deepcopy(data)
        if "font_size_px" not in result:
            result["font_size_px"] = {
                "auto": 11, "small": 8, "medium": 13, "large": 16,
            }.get(result.get("font_size_mode", "auto"), 11)
        if "line_thickness_px" not in result:
            result["line_thickness_px"] = {
                "auto": 1, "small": 1, "medium": 1, "large": 2,
            }.get(result.get("line_thickness_mode", "auto"), 1)
        return result

    def load(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._with_defaults(
                    self._migrate_legacy_size_settings(data), self.defaults
                )
            except Exception as e:
                print(f"Error loading settings: {e}")
                return deepcopy(self.defaults)
        else:
            data = deepcopy(self.defaults)
            self.save(data)
            return data

    def save(self, settings=None):
        if settings is None:
            settings = self.current
        # Копия предотвращает изменение defaults через current.
        settings = deepcopy(settings)
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.current = settings
            self.settings_changed.emit(settings)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get_color(self, name):
        colors = self.current.get("colors", {}) or {}
        return tuple(colors.get(name, self.defaults["colors"].get(name, [0, 0, 0])))

    def get_primary_color(self):
        return tuple(self.current.get("primary_color", [42, 130, 218]))

    def get_theme(self):
        return self.current.get("theme", "dark")

    @staticmethod
    def _safe_int(value, fallback, low, high):
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError, OverflowError):
            return fallback

    def get_font_size_px(self):
        return self._safe_int(
            self.current.get("font_size_px", FONT_SIZE_PX_DEFAULT),
            FONT_SIZE_PX_DEFAULT, FONT_SIZE_PX_MIN, FONT_SIZE_PX_MAX
        )

    def get_line_thickness_px(self):
        return self._safe_int(
            self.current.get("line_thickness_px", LINE_THICKNESS_PX_DEFAULT),
            LINE_THICKNESS_PX_DEFAULT, LINE_THICKNESS_PX_MIN, LINE_THICKNESS_PX_MAX
        )

    def get_dataset_class_palette(self):
        """Полная палитра классов, в формате BGR, минимум 10 цветов."""
        raw = self.current.get("dataset_class_palette")
        if not isinstance(raw, list):
            raw = []
        n = max(len(DEFAULT_DATASET_CLASS_PALETTE_BGR), len(raw))
        result = []
        for i in range(n):
            default = DEFAULT_DATASET_CLASS_PALETTE_BGR[i % len(DEFAULT_DATASET_CLASS_PALETTE_BGR)]
            candidate = raw[i] if i < len(raw) else default
            if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
                candidate = default
            try:
                bgr = [max(0, min(255, int(channel))) for channel in candidate]
            except (TypeError, ValueError, OverflowError):
                bgr = list(default)
            result.append(bgr)
        return result

    def get_dataset_class_color(self, class_id):
        """BGR цвет класса; отсутствующий / отрицательный класс -> класс 0.

        Для class_id >= длины палитры сохраняется циклическая раскраска.
        """
        try:
            cls = int(class_id)
        except (ValueError, TypeError, OverflowError):
            cls = 0
        cls = max(0, cls)
        palette = self.get_dataset_class_palette()
        return tuple(palette[cls % len(palette)])

    def get_font_scale_factor(self):
        # Обратная совместимость с OpenCV-утилитами, ожидающими множитель.
        return self.get_font_size_px() / float(FONT_SIZE_PX_DEFAULT)

    def get_line_thickness_factor(self):
        # Для старого get_display_params() — множитель от его базовой толщины.
        # Точная пиксельная толщина гарантируется новым Qt view.
        return self.get_line_thickness_px() / float(LINE_THICKNESS_PX_DEFAULT)


def adjust_color(color, h_shift=0, s_shift=0, l_shift=0):
    r, g, b = [x/255.0 for x in color]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + h_shift) % 1.0
    l = max(0.0, min(1.0, l + l_shift))
    s = max(0.0, min(1.0, s + s_shift))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))


def generate_palette_for_theme(theme, primary_color):
    palette = QPalette()
    accent = QColor(*primary_color)

    comp_rgb = adjust_color(primary_color, h_shift=0.5, s_shift=0, l_shift=0)
    complement = QColor(*comp_rgb)  # используется для mid/shadow

    if theme == "dark":
        bg_rgb = adjust_color(primary_color, h_shift=0, s_shift=-0.1, l_shift=-0.1)
        dark = QColor(*bg_rgb)
        darker = dark.darker(105)
        text = QColor(255, 255, 255)
        button = dark.lighter(105)

        mid_rgb = adjust_color(comp_rgb, h_shift=0, s_shift=-0.3, l_shift=-0.2)
        mid = QColor(*mid_rgb)
        shadow_rgb = adjust_color(comp_rgb, h_shift=0, s_shift=-0.2, l_shift=-0.2)
        shadow = QColor(*shadow_rgb)

        highlight = accent
        highlighted_text = QColor(0, 0, 0) if highlight.lightness() > 128 else QColor(255, 255, 255)

        palette.setColor(QPalette.Window, dark)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, darker)
        palette.setColor(QPalette.AlternateBase, dark)
        palette.setColor(QPalette.ToolTipBase, text)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, button)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, accent)
        palette.setColor(QPalette.Highlight, highlight)
        palette.setColor(QPalette.HighlightedText, highlighted_text)
        palette.setColor(QPalette.Mid, mid)
        palette.setColor(QPalette.Shadow, shadow)
    else:
        bg_rgb = adjust_color(primary_color, h_shift=0, s_shift=-0.8, l_shift=0.7)
        light = QColor(*bg_rgb)
        lighter = light.darker(102)
        text = QColor(0, 0, 0)
        button = light.lighter(105)

        mid_rgb = adjust_color(comp_rgb, h_shift=0, s_shift=-0.5, l_shift=0.3)
        mid = QColor(*mid_rgb)
        shadow_rgb = adjust_color(comp_rgb, h_shift=0, s_shift=-0.3, l_shift=0.1)
        shadow = QColor(*shadow_rgb)

        highlight = accent
        highlighted_text = QColor(255, 255, 255) if highlight.lightness() < 128 else QColor(0, 0, 0)

        palette.setColor(QPalette.Window, light)
        palette.setColor(QPalette.WindowText, text)
        palette.setColor(QPalette.Base, lighter)
        palette.setColor(QPalette.AlternateBase, light)
        palette.setColor(QPalette.ToolTipBase, text)
        palette.setColor(QPalette.ToolTipText, text)
        palette.setColor(QPalette.Text, text)
        palette.setColor(QPalette.Button, button)
        palette.setColor(QPalette.ButtonText, text)
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, accent)
        palette.setColor(QPalette.Highlight, highlight)
        palette.setColor(QPalette.HighlightedText, highlighted_text)
        palette.setColor(QPalette.Mid, mid)
        palette.setColor(QPalette.Shadow, shadow)

    return palette


def apply_theme(app):
    theme = settings.get_theme()
    primary_color = settings.get_primary_color()
    palette = generate_palette_for_theme(theme, primary_color)
    app.setStyle("Fusion")
    app.setPalette(palette)


settings = SettingsManager()