# settings.py
from import_libs_external import *
from path_setup import get_project_root


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
            "font_size_mode": "auto",
            "line_thickness_mode": "auto"
        }
        self.current = self.load()

    def load(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in self.defaults.items():
                    if key not in data:
                        data[key] = value
                return data
            except Exception as e:
                print(f"Error loading settings: {e}")
                return self.defaults
        else:
            self.save(self.defaults)
            return self.defaults

    def save(self, settings=None):
        if settings is None:
            settings = self.current
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.current = settings
            self.settings_changed.emit(settings)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get_color(self, name):
        return tuple(self.current["colors"].get(name, [0, 0, 0]))

    def get_primary_color(self):
        return tuple(self.current.get("primary_color", [42, 130, 218]))

    def get_theme(self):
        return self.current.get("theme", "dark")

    def get_font_scale_factor(self):
        mode = self.current.get("font_size_mode", "auto")
        if mode == "auto":
            return 1.0
        elif mode == "small":
            return 0.7
        elif mode == "medium":
            return 1.2
        elif mode == "large":
            return 1.5
        return 1.0

    def get_line_thickness_factor(self):
        mode = self.current.get("line_thickness_mode", "auto")
        if mode == "auto":
            return 1.0
        elif mode == "small":
            return 0.7
        elif mode == "medium":
            return 1.0
        elif mode == "large":
            return 1.3
        return 1.0


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