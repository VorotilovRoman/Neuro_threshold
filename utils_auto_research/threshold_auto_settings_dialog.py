from import_libs_external import *

class AutoSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Research Settings")
        self.setMinimumWidth(850)
        self.setMinimumHeight(650)

        self.current_settings = current_settings if current_settings else self._default_settings()
        self.settings = self._default_settings()

        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        self.tabs = QTabWidget()
        scroll_layout.addWidget(self.tabs)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._setup_tabs()

        if self.current_settings:
            self._load_settings()

    def _safe_name(self, name):
        """Преобразует строку с пробелами и скобками в безопасное имя атрибута."""
        return name.replace(' ', '_').replace('(', '_').replace(')', '_')

    def _default_settings(self):
        return {
            'Simple Threshold': {'enabled': True, 'start': 1, 'end': 254, 'step': 1},
            'Otsu': {'enabled': True},
            'Triangle': {'enabled': True},
            'Adaptive Mean': {'enabled': True, 'window_start': 3, 'window_end': 201, 'window_step': 10, 'c_start': -20, 'c_end': 20, 'c_step': 5},
            'Adaptive Gauss': {'enabled': True, 'window_start': 3, 'window_end': 201, 'window_step': 10, 'c_start': -20, 'c_end': 20, 'c_step': 5},
            'Niblack': {'enabled': True, 'window_start': 3, 'window_end': 201, 'window_step': 10, 'k_start': -1.0, 'k_end': 1.0, 'k_step': 0.1},
            'Sauvola': {'enabled': True, 'window_start': 3, 'window_end': 201, 'window_step': 10, 'k_start': -1.0, 'k_end': 1.0, 'k_step': 0.1, 'r_start': 1, 'r_end': 255, 'r_step': 20},
            'ISODATA': {'enabled': True, 'init_start': 0, 'init_end': 255, 'init_step': 1},
            'Background Symmetry': {'enabled': True, 'excess_start': 0.01, 'excess_end': 1.0, 'excess_step': 0.01},
            'Row Adaptive': {'enabled': True, 'window_start': 10, 'window_end': 500, 'window_step': 10, 'k_start': 0.0, 'k_end': 1.0, 'k_step': 0.1},
            'Morphology': {
                'enabled': False,
                'invert_enabled': False,
                'close_enabled': False,
                'close_start': 0.0,
                'close_end': 0.10,
                'close_step': 0.01,
                'open_enabled': False,
                'open_start': 0.0,
                'open_end': 0.10,
                'open_step': 0.01
            }
        }

    def _setup_tabs(self):
        self._add_simple_threshold_tab()
        self._add_otsu_tab()
        self._add_triangle_tab()
        self._add_adaptive_tab("Adaptive Mean")
        self._add_adaptive_tab("Adaptive Gauss")
        self._add_niblack_tab()
        self._add_sauvola_tab()
        self._add_isodata_tab()
        self._add_background_symmetry_tab()
        self._add_row_adaptive_tab()
        self._add_morphology_tab()   # новая вкладка

    def _add_simple_threshold_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        group = QGroupBox("Simple Threshold")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Simple Threshold']['enabled'])

        form = QFormLayout(group)
        start = QSpinBox(); start.setRange(0, 255); start.setValue(self.current_settings['Simple Threshold']['start'])
        end = QSpinBox(); end.setRange(0, 255); end.setValue(self.current_settings['Simple Threshold']['end'])
        step = QSpinBox(); step.setRange(1, 255); step.setValue(self.current_settings['Simple Threshold']['step'])
        form.addRow("Start", start)
        form.addRow("End", end)
        form.addRow("Step", step)
        layout.addWidget(group)

        self.simple_threshold_group = group
        self.simple_threshold_start = start
        self.simple_threshold_end = end
        self.simple_threshold_step = step

        self.tabs.addTab(tab, "Simple Threshold")

    def _add_otsu_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Otsu")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Otsu']['enabled'])
        layout.addWidget(group)
        self.otsu_group = group
        self.tabs.addTab(tab, "Otsu")

    def _add_triangle_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Triangle")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Triangle']['enabled'])
        layout.addWidget(group)
        self.triangle_group = group
        self.tabs.addTab(tab, "Triangle")

    def _add_adaptive_tab(self, method_name):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox(method_name)
        group.setCheckable(True)
        group.setChecked(self.current_settings[method_name]['enabled'])

        form = QFormLayout(group)
        ws_start = QSpinBox(); ws_start.setRange(1, 1000); ws_start.setValue(self.current_settings[method_name]['window_start'])
        ws_end = QSpinBox(); ws_end.setRange(1, 1000); ws_end.setValue(self.current_settings[method_name]['window_end'])
        ws_step = QSpinBox(); ws_step.setRange(1, 100); ws_step.setValue(self.current_settings[method_name]['window_step'])
        c_start = QSpinBox(); c_start.setRange(-100, 100); c_start.setValue(self.current_settings[method_name]['c_start'])
        c_end = QSpinBox(); c_end.setRange(-100, 100); c_end.setValue(self.current_settings[method_name]['c_end'])
        c_step = QSpinBox(); c_step.setRange(1, 20); c_step.setValue(self.current_settings[method_name]['c_step'])

        form.addRow("Window start", ws_start)
        form.addRow("Window end", ws_end)
        form.addRow("Window step", ws_step)
        form.addRow("C start", c_start)
        form.addRow("C end", c_end)
        form.addRow("C step", c_step)

        layout.addWidget(group)

        safe_name = self._safe_name(method_name)
        setattr(self, f"{safe_name}_group", group)
        setattr(self, f"{safe_name}_ws_start", ws_start)
        setattr(self, f"{safe_name}_ws_end", ws_end)
        setattr(self, f"{safe_name}_ws_step", ws_step)
        setattr(self, f"{safe_name}_c_start", c_start)
        setattr(self, f"{safe_name}_c_end", c_end)
        setattr(self, f"{safe_name}_c_step", c_step)

        self.tabs.addTab(tab, method_name)

    def _add_niblack_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Niblack")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Niblack']['enabled'])

        form = QFormLayout(group)
        ws_start = QSpinBox(); ws_start.setRange(1, 1000); ws_start.setValue(self.current_settings['Niblack']['window_start'])
        ws_end = QSpinBox(); ws_end.setRange(1, 1000); ws_end.setValue(self.current_settings['Niblack']['window_end'])
        ws_step = QSpinBox(); ws_step.setRange(1, 100); ws_step.setValue(self.current_settings['Niblack']['window_step'])
        k_start = QDoubleSpinBox(); k_start.setRange(-5, 5); k_start.setSingleStep(0.01); k_start.setValue(self.current_settings['Niblack']['k_start'])
        k_end = QDoubleSpinBox(); k_end.setRange(-5, 5); k_end.setSingleStep(0.01); k_end.setValue(self.current_settings['Niblack']['k_end'])
        k_step = QDoubleSpinBox(); k_step.setRange(0.01, 1); k_step.setSingleStep(0.01); k_step.setValue(self.current_settings['Niblack']['k_step'])

        form.addRow("Window start", ws_start)
        form.addRow("Window end", ws_end)
        form.addRow("Window step", ws_step)
        form.addRow("k start", k_start)
        form.addRow("k end", k_end)
        form.addRow("k step", k_step)

        layout.addWidget(group)

        self.niblack_group = group
        self.niblack_ws_start = ws_start
        self.niblack_ws_end = ws_end
        self.niblack_ws_step = ws_step
        self.niblack_k_start = k_start
        self.niblack_k_end = k_end
        self.niblack_k_step = k_step

        self.tabs.addTab(tab, "Niblack")

    def _add_sauvola_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Sauvola")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Sauvola']['enabled'])

        form = QFormLayout(group)
        ws_start = QSpinBox(); ws_start.setRange(1, 1000); ws_start.setValue(self.current_settings['Sauvola']['window_start'])
        ws_end = QSpinBox(); ws_end.setRange(1, 1000); ws_end.setValue(self.current_settings['Sauvola']['window_end'])
        ws_step = QSpinBox(); ws_step.setRange(1, 100); ws_step.setValue(self.current_settings['Sauvola']['window_step'])
        k_start = QDoubleSpinBox(); k_start.setRange(-5, 5); k_start.setSingleStep(0.01); k_start.setValue(self.current_settings['Sauvola']['k_start'])
        k_end = QDoubleSpinBox(); k_end.setRange(-5, 5); k_end.setSingleStep(0.01); k_end.setValue(self.current_settings['Sauvola']['k_end'])
        k_step = QDoubleSpinBox(); k_step.setRange(0.01, 1); k_step.setSingleStep(0.01); k_step.setValue(self.current_settings['Sauvola']['k_step'])
        r_start = QSpinBox(); r_start.setRange(1, 1000); r_start.setValue(self.current_settings['Sauvola']['r_start'])
        r_end = QSpinBox(); r_end.setRange(1, 1000); r_end.setValue(self.current_settings['Sauvola']['r_end'])
        r_step = QSpinBox(); r_step.setRange(1, 50); r_step.setValue(self.current_settings['Sauvola']['r_step'])

        form.addRow("Window start", ws_start)
        form.addRow("Window end", ws_end)
        form.addRow("Window step", ws_step)
        form.addRow("k start", k_start)
        form.addRow("k end", k_end)
        form.addRow("k step", k_step)
        form.addRow("R start", r_start)
        form.addRow("R end", r_end)
        form.addRow("R step", r_step)

        layout.addWidget(group)

        self.sauvola_group = group
        self.sauvola_ws_start = ws_start
        self.sauvola_ws_end = ws_end
        self.sauvola_ws_step = ws_step
        self.sauvola_k_start = k_start
        self.sauvola_k_end = k_end
        self.sauvola_k_step = k_step
        self.sauvola_r_start = r_start
        self.sauvola_r_end = r_end
        self.sauvola_r_step = r_step

        self.tabs.addTab(tab, "Sauvola")

    def _add_isodata_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("ISODATA")
        group.setCheckable(True)
        group.setChecked(self.current_settings['ISODATA']['enabled'])

        form = QFormLayout(group)
        init_start = QSpinBox(); init_start.setRange(0, 255); init_start.setValue(self.current_settings['ISODATA']['init_start'])
        init_end = QSpinBox(); init_end.setRange(0, 255); init_end.setValue(self.current_settings['ISODATA']['init_end'])
        init_step = QSpinBox(); init_step.setRange(1, 255); init_step.setValue(self.current_settings['ISODATA']['init_step'])
        form.addRow("Initial start", init_start)
        form.addRow("Initial end", init_end)
        form.addRow("Step", init_step)
        layout.addWidget(group)

        self.isodata_group = group
        self.isodata_init_start = init_start
        self.isodata_init_end = init_end
        self.isodata_init_step = init_step

        self.tabs.addTab(tab, "ISODATA")

    def _add_background_symmetry_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Background Symmetry")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Background Symmetry']['enabled'])

        form = QFormLayout(group)
        excess_start = QDoubleSpinBox(); excess_start.setRange(0.01, 1.0); excess_start.setSingleStep(0.01); excess_start.setValue(self.current_settings['Background Symmetry']['excess_start'])
        excess_end = QDoubleSpinBox(); excess_end.setRange(0.01, 1.0); excess_end.setSingleStep(0.01); excess_end.setValue(self.current_settings['Background Symmetry']['excess_end'])
        excess_step = QDoubleSpinBox(); excess_step.setRange(0.01, 0.1); excess_step.setSingleStep(0.01); excess_step.setValue(self.current_settings['Background Symmetry']['excess_step'])
        form.addRow("Excess start", excess_start)
        form.addRow("Excess end", excess_end)
        form.addRow("Step", excess_step)
        layout.addWidget(group)

        self.background_group = group
        self.background_excess_start = excess_start
        self.background_excess_end = excess_end
        self.background_excess_step = excess_step

        self.tabs.addTab(tab, "Background Symmetry")

    def _add_row_adaptive_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Row Adaptive")
        group.setCheckable(True)
        group.setChecked(self.current_settings['Row Adaptive']['enabled'])

        form = QFormLayout(group)
        ws_start = QSpinBox(); ws_start.setRange(1, 1000); ws_start.setValue(self.current_settings['Row Adaptive']['window_start'])
        ws_end = QSpinBox(); ws_end.setRange(1, 1000); ws_end.setValue(self.current_settings['Row Adaptive']['window_end'])
        ws_step = QSpinBox(); ws_step.setRange(1, 100); ws_step.setValue(self.current_settings['Row Adaptive']['window_step'])
        k_start = QDoubleSpinBox(); k_start.setRange(0.0, 1.0); k_start.setSingleStep(0.01); k_start.setValue(self.current_settings['Row Adaptive']['k_start'])
        k_end = QDoubleSpinBox(); k_end.setRange(0.0, 1.0); k_end.setSingleStep(0.01); k_end.setValue(self.current_settings['Row Adaptive']['k_end'])
        k_step = QDoubleSpinBox(); k_step.setRange(0.01, 0.1); k_step.setSingleStep(0.01); k_step.setValue(self.current_settings['Row Adaptive']['k_step'])
        form.addRow("Window start", ws_start)
        form.addRow("Window end", ws_end)
        form.addRow("Window step", ws_step)
        form.addRow("k start", k_start)
        form.addRow("k end", k_end)
        form.addRow("k step", k_step)
        layout.addWidget(group)

        self.row_adaptive_group = group
        self.row_adaptive_ws_start = ws_start
        self.row_adaptive_ws_end = ws_end
        self.row_adaptive_ws_step = ws_step
        self.row_adaptive_k_start = k_start
        self.row_adaptive_k_end = k_end
        self.row_adaptive_k_step = k_step

        self.tabs.addTab(tab, "Row Adaptive")

    def _add_morphology_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Группа "Morphology operations"
        group = QGroupBox("Morphology operations")
        group.setCheckable(True)
        group.setChecked(self.current_settings.get('Morphology', {}).get('enabled', False))
        form = QFormLayout(group)

        # Инверсия
        invert_check = QCheckBox("Invert mask")
        invert_check.setChecked(self.current_settings.get('Morphology', {}).get('invert_enabled', False))
        form.addRow("", invert_check)

        # Закрытие
        close_check = QCheckBox("Enable closing")
        close_check.setChecked(self.current_settings.get('Morphology', {}).get('close_enabled', False))
        form.addRow("", close_check)

        close_start = QDoubleSpinBox()
        close_start.setRange(0, 0.10)
        close_start.setSingleStep(0.01)
        close_start.setValue(self.current_settings.get('Morphology', {}).get('close_start', 0.0))
        close_end = QDoubleSpinBox()
        close_end.setRange(0, 0.10)
        close_end.setSingleStep(0.01)
        close_end.setValue(self.current_settings.get('Morphology', {}).get('close_end', 0.10))
        close_step = QDoubleSpinBox()
        close_step.setRange(0.001, 0.05)
        close_step.setSingleStep(0.001)
        close_step.setValue(self.current_settings.get('Morphology', {}).get('close_step', 0.01))
        form.addRow("Close start", close_start)
        form.addRow("Close end", close_end)
        form.addRow("Close step", close_step)

        # Открытие
        open_check = QCheckBox("Enable opening")
        open_check.setChecked(self.current_settings.get('Morphology', {}).get('open_enabled', False))
        form.addRow("", open_check)

        open_start = QDoubleSpinBox()
        open_start.setRange(0, 0.10)
        open_start.setSingleStep(0.01)
        open_start.setValue(self.current_settings.get('Morphology', {}).get('open_start', 0.0))
        open_end = QDoubleSpinBox()
        open_end.setRange(0, 0.10)
        open_end.setSingleStep(0.01)
        open_end.setValue(self.current_settings.get('Morphology', {}).get('open_end', 0.10))
        open_step = QDoubleSpinBox()
        open_step.setRange(0.001, 0.05)
        open_step.setSingleStep(0.001)
        open_step.setValue(self.current_settings.get('Morphology', {}).get('open_step', 0.01))
        form.addRow("Open start", open_start)
        form.addRow("Open end", open_end)
        form.addRow("Open step", open_step)

        layout.addWidget(group)
        self.tabs.addTab(tab, "Morphology")

        # Сохраняем виджеты
        self.morphology_group = group
        self.morphology_invert_check = invert_check
        self.morphology_close_check = close_check
        self.morphology_close_start = close_start
        self.morphology_close_end = close_end
        self.morphology_close_step = close_step
        self.morphology_open_check = open_check
        self.morphology_open_start = open_start
        self.morphology_open_end = open_end
        self.morphology_open_step = open_step

    def _load_settings(self):
        # Simple Threshold
        st = self.current_settings['Simple Threshold']
        self.simple_threshold_group.setChecked(st['enabled'])
        self.simple_threshold_start.setValue(st['start'])
        self.simple_threshold_end.setValue(st['end'])
        self.simple_threshold_step.setValue(st['step'])

        # Otsu
        self.otsu_group.setChecked(self.current_settings['Otsu']['enabled'])

        # Triangle
        self.triangle_group.setChecked(self.current_settings['Triangle']['enabled'])

        # Adaptive Mean
        am = self.current_settings['Adaptive Mean']
        safe_am = self._safe_name("Adaptive Mean")
        getattr(self, f"{safe_am}_group").setChecked(am['enabled'])
        getattr(self, f"{safe_am}_ws_start").setValue(am['window_start'])
        getattr(self, f"{safe_am}_ws_end").setValue(am['window_end'])
        getattr(self, f"{safe_am}_ws_step").setValue(am['window_step'])
        getattr(self, f"{safe_am}_c_start").setValue(am['c_start'])
        getattr(self, f"{safe_am}_c_end").setValue(am['c_end'])
        getattr(self, f"{safe_am}_c_step").setValue(am['c_step'])

        # Adaptive Gauss
        ag = self.current_settings['Adaptive Gauss']
        safe_ag = self._safe_name("Adaptive Gauss")
        getattr(self, f"{safe_ag}_group").setChecked(ag['enabled'])
        getattr(self, f"{safe_ag}_ws_start").setValue(ag['window_start'])
        getattr(self, f"{safe_ag}_ws_end").setValue(ag['window_end'])
        getattr(self, f"{safe_ag}_ws_step").setValue(ag['window_step'])
        getattr(self, f"{safe_ag}_c_start").setValue(ag['c_start'])
        getattr(self, f"{safe_ag}_c_end").setValue(ag['c_end'])
        getattr(self, f"{safe_ag}_c_step").setValue(ag['c_step'])

        # Niblack
        nb = self.current_settings['Niblack']
        self.niblack_group.setChecked(nb['enabled'])
        self.niblack_ws_start.setValue(nb['window_start'])
        self.niblack_ws_end.setValue(nb['window_end'])
        self.niblack_ws_step.setValue(nb['window_step'])
        self.niblack_k_start.setValue(nb['k_start'])
        self.niblack_k_end.setValue(nb['k_end'])
        self.niblack_k_step.setValue(nb['k_step'])

        # Sauvola
        sv = self.current_settings['Sauvola']
        self.sauvola_group.setChecked(sv['enabled'])
        self.sauvola_ws_start.setValue(sv['window_start'])
        self.sauvola_ws_end.setValue(sv['window_end'])
        self.sauvola_ws_step.setValue(sv['window_step'])
        self.sauvola_k_start.setValue(sv['k_start'])
        self.sauvola_k_end.setValue(sv['k_end'])
        self.sauvola_k_step.setValue(sv['k_step'])
        self.sauvola_r_start.setValue(sv['r_start'])
        self.sauvola_r_end.setValue(sv['r_end'])
        self.sauvola_r_step.setValue(sv['r_step'])

        # ISODATA
        iso = self.current_settings['ISODATA']
        self.isodata_group.setChecked(iso['enabled'])
        self.isodata_init_start.setValue(iso['init_start'])
        self.isodata_init_end.setValue(iso['init_end'])
        self.isodata_init_step.setValue(iso['init_step'])

        # Background Symmetry
        bs = self.current_settings['Background Symmetry']
        self.background_group.setChecked(bs['enabled'])
        self.background_excess_start.setValue(bs['excess_start'])
        self.background_excess_end.setValue(bs['excess_end'])
        self.background_excess_step.setValue(bs['excess_step'])

        # Row Adaptive
        ra = self.current_settings['Row Adaptive']
        self.row_adaptive_group.setChecked(ra['enabled'])
        self.row_adaptive_ws_start.setValue(ra['window_start'])
        self.row_adaptive_ws_end.setValue(ra['window_end'])
        self.row_adaptive_ws_step.setValue(ra['window_step'])
        self.row_adaptive_k_start.setValue(ra['k_start'])
        self.row_adaptive_k_end.setValue(ra['k_end'])
        self.row_adaptive_k_step.setValue(ra['k_step'])

        # Morphology
        morph = self.current_settings.get('Morphology', {})
        self.morphology_group.setChecked(morph.get('enabled', False))
        self.morphology_invert_check.setChecked(morph.get('invert_enabled', False))
        self.morphology_close_check.setChecked(morph.get('close_enabled', False))
        self.morphology_close_start.setValue(morph.get('close_start', 0.0))
        self.morphology_close_end.setValue(morph.get('close_end', 0.10))
        self.morphology_close_step.setValue(morph.get('close_step', 0.01))
        self.morphology_open_check.setChecked(morph.get('open_enabled', False))
        self.morphology_open_start.setValue(morph.get('open_start', 0.0))
        self.morphology_open_end.setValue(morph.get('open_end', 0.10))
        self.morphology_open_step.setValue(morph.get('open_step', 0.01))

    def get_settings(self):
        settings = {}

        # Simple Threshold
        settings['Simple Threshold'] = {
            'enabled': self.simple_threshold_group.isChecked(),
            'start': self.simple_threshold_start.value(),
            'end': self.simple_threshold_end.value(),
            'step': self.simple_threshold_step.value()
        }

        # Otsu
        settings['Otsu'] = {'enabled': self.otsu_group.isChecked()}

        # Triangle
        settings['Triangle'] = {'enabled': self.triangle_group.isChecked()}

        # Adaptive Mean
        safe_am = self._safe_name("Adaptive Mean")
        settings['Adaptive Mean'] = {
            'enabled': getattr(self, f"{safe_am}_group").isChecked(),
            'window_start': getattr(self, f"{safe_am}_ws_start").value(),
            'window_end': getattr(self, f"{safe_am}_ws_end").value(),
            'window_step': getattr(self, f"{safe_am}_ws_step").value(),
            'c_start': getattr(self, f"{safe_am}_c_start").value(),
            'c_end': getattr(self, f"{safe_am}_c_end").value(),
            'c_step': getattr(self, f"{safe_am}_c_step").value()
        }

        # Adaptive Gauss
        safe_ag = self._safe_name("Adaptive Gauss")
        settings['Adaptive Gauss'] = {
            'enabled': getattr(self, f"{safe_ag}_group").isChecked(),
            'window_start': getattr(self, f"{safe_ag}_ws_start").value(),
            'window_end': getattr(self, f"{safe_ag}_ws_end").value(),
            'window_step': getattr(self, f"{safe_ag}_ws_step").value(),
            'c_start': getattr(self, f"{safe_ag}_c_start").value(),
            'c_end': getattr(self, f"{safe_ag}_c_end").value(),
            'c_step': getattr(self, f"{safe_ag}_c_step").value()
        }

        # Niblack
        settings['Niblack'] = {
            'enabled': self.niblack_group.isChecked(),
            'window_start': self.niblack_ws_start.value(),
            'window_end': self.niblack_ws_end.value(),
            'window_step': self.niblack_ws_step.value(),
            'k_start': self.niblack_k_start.value(),
            'k_end': self.niblack_k_end.value(),
            'k_step': self.niblack_k_step.value()
        }

        # Sauvola
        settings['Sauvola'] = {
            'enabled': self.sauvola_group.isChecked(),
            'window_start': self.sauvola_ws_start.value(),
            'window_end': self.sauvola_ws_end.value(),
            'window_step': self.sauvola_ws_step.value(),
            'k_start': self.sauvola_k_start.value(),
            'k_end': self.sauvola_k_end.value(),
            'k_step': self.sauvola_k_step.value(),
            'r_start': self.sauvola_r_start.value(),
            'r_end': self.sauvola_r_end.value(),
            'r_step': self.sauvola_r_step.value()
        }

        # ISODATA
        settings['ISODATA'] = {
            'enabled': self.isodata_group.isChecked(),
            'init_start': self.isodata_init_start.value(),
            'init_end': self.isodata_init_end.value(),
            'init_step': self.isodata_init_step.value()
        }

        # Background Symmetry
        settings['Background Symmetry'] = {
            'enabled': self.background_group.isChecked(),
            'excess_start': self.background_excess_start.value(),
            'excess_end': self.background_excess_end.value(),
            'excess_step': self.background_excess_step.value()
        }

        # Row Adaptive
        settings['Row Adaptive'] = {
            'enabled': self.row_adaptive_group.isChecked(),
            'window_start': self.row_adaptive_ws_start.value(),
            'window_end': self.row_adaptive_ws_end.value(),
            'window_step': self.row_adaptive_ws_step.value(),
            'k_start': self.row_adaptive_k_start.value(),
            'k_end': self.row_adaptive_k_end.value(),
            'k_step': self.row_adaptive_k_step.value()
        }

        # Morphology
        settings['Morphology'] = {
            'enabled': self.morphology_group.isChecked(),
            'invert_enabled': self.morphology_invert_check.isChecked(),
            'close_enabled': self.morphology_close_check.isChecked(),
            'close_start': self.morphology_close_start.value(),
            'close_end': self.morphology_close_end.value(),
            'close_step': self.morphology_close_step.value(),
            'open_enabled': self.morphology_open_check.isChecked(),
            'open_start': self.morphology_open_start.value(),
            'open_end': self.morphology_open_end.value(),
            'open_step': self.morphology_open_step.value()
        }

        return settings