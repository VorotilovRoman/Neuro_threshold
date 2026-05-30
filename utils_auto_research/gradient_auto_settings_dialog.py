from import_libs_external import *


class GradientAutoSettingsDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Research Settings (Gradient Methods)")
        self.setMinimumWidth(850)
        self.setMinimumHeight(650)

        self.parent_window = parent
        self.log("Инициализация диалога настроек автоподбора градиентных методов")

        try:
            self.current_settings = current_settings if current_settings else self._default_settings()
            # Дополняем недостающие ключи из дефолта
            default = self._default_settings()
            for key in default:
                if key not in self.current_settings:
                    self.current_settings[key] = default[key]
                    self.log(f"Добавлен отсутствующий ключ '{key}' в настройки")
                elif isinstance(default[key], dict) and isinstance(self.current_settings[key], dict):
                    for subkey in default[key]:
                        if subkey not in self.current_settings[key]:
                            self.current_settings[key][subkey] = default[key][subkey]
                            self.log(f"Добавлен отсутствующий ключ '{key}.{subkey}'")

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

            self.log("Диалог настроек успешно создан")
        except Exception as e:
            self.log(f"ОШИБКА при создании диалога настроек: {e}\n{traceback.format_exc()}")
            raise

    def log(self, message):
        try:
            if self.parent_window and hasattr(self.parent_window, 'log'):
                self.parent_window.log(message)
            else:
                print(message)
        except Exception:
            print(message)

    def _safe_name(self, name):
        return name.replace(' ', '_').replace('(', '_').replace(')', '_')

    def _default_settings(self):
        return {
            'Sobel': {
                'enabled': True,
                'ksize_start': 3, 'ksize_end': 7, 'ksize_step': 2,
                'scale_start': 1, 'scale_end': 3, 'scale_step': 1
            },
            'Scharr': {
                'enabled': True,
                'scale_start': 1, 'scale_end': 3, 'scale_step': 1
            },
            'Laplacian': {
                'enabled': True,
                'ksize_start': 3, 'ksize_end': 7, 'ksize_step': 2,
                'scale_start': 1, 'scale_end': 3, 'scale_step': 1
            },
            'Canny': {
                'enabled': True,
                'thresh1_start': 30, 'thresh1_end': 150, 'thresh1_step': 20,
                'thresh2_start': 100, 'thresh2_end': 250, 'thresh2_step': 30,
                'aperture_start': 3, 'aperture_end': 5, 'aperture_step': 2
            },
            'Prewitt': {'enabled': True},
            'Roberts': {'enabled': True},
            'Kirsch': {'enabled': True},
            'Binarization': {
                'enabled': True,
                'threshold_start': 0, 'threshold_end': 255, 'threshold_step': 10
            },
            'ObjectFiltering': {
                'enabled': True,
                'fill_contours': [True],
                'min_area_percent_start': 0.0, 'min_area_percent_end': 0.1, 'min_area_percent_step': 0.01
            },
            'Morphology': {
                'enabled': True,
                'close_enabled': True,
                'close_start': 0.0, 'close_end': 0.05, 'close_step': 0.005,
                'open_enabled': True,
                'open_start': 0.0, 'open_end': 0.05, 'open_step': 0.005,
                'kernel_shapes': ["Rectangle", "Ellipse", "Cross"]
            },
            'Invert': [False]
        }

    def _setup_tabs(self):
        try:
            self.log("Создание вкладок диалога настроек")
            self._add_method_tab("Sobel")
            self._add_method_tab("Scharr")
            self._add_method_tab("Laplacian")
            self._add_method_tab("Canny")
            self._add_method_tab("Prewitt")
            self._add_method_tab("Roberts")
            self._add_method_tab("Kirsch")
            self._add_binarization_tab()
            self._add_object_filtering_tab()
            self._add_morphology_tab()
            self._add_invert_tab()
            self.log("Вкладки успешно созданы")
        except Exception as e:
            self.log(f"Ошибка при создании вкладок: {e}\n{traceback.format_exc()}")
            raise

    def _add_method_tab(self, method_name):
        try:
            # Если метода нет в current_settings, используем дефолтные
            if method_name not in self.current_settings:
                self.current_settings[method_name] = self._default_settings()[method_name]
                self.log(f"Метод '{method_name}' отсутствовал в настройках, добавлен со значениями по умолчанию")

            tab = QWidget()
            layout = QVBoxLayout(tab)
            group = QGroupBox(method_name)
            group.setCheckable(True)
            group.setChecked(self.current_settings[method_name]['enabled'])

            form = QFormLayout(group)
            params = self.current_settings[method_name]

            for key in sorted(params.keys()):
                if key == 'enabled':
                    continue
                if key.endswith('_start'):
                    base = key[:-6]
                    start_widget = self._create_param_widget(params[key])
                    end_widget = self._create_param_widget(params[f'{base}_end'])
                    step_widget = self._create_param_widget(params[f'{base}_step'], is_step=True)
                    hbox = QHBoxLayout()
                    hbox.addWidget(QLabel("Start:"))
                    hbox.addWidget(start_widget)
                    hbox.addWidget(QLabel("End:"))
                    hbox.addWidget(end_widget)
                    hbox.addWidget(QLabel("Step:"))
                    hbox.addWidget(step_widget)
                    form.addRow(base, hbox)
                    setattr(self, f"{method_name}_{base}_start", start_widget)
                    setattr(self, f"{method_name}_{base}_end", end_widget)
                    setattr(self, f"{method_name}_{base}_step", step_widget)

            layout.addWidget(group)
            self.tabs.addTab(tab, method_name)
            setattr(self, f"{method_name}_group", group)
        except Exception as e:
            self.log(f"Ошибка при добавлении вкладки {method_name}: {e}\n{traceback.format_exc()}")
            raise

    def _create_param_widget(self, value, is_step=False):
        if isinstance(value, float):
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setSingleStep(0.001)
            if is_step:
                w.setRange(0.001, 1.0)
            else:
                w.setRange(-1000, 1000)
            w.setValue(value)
            return w
        else:
            w = QSpinBox()
            if is_step:
                w.setRange(1, 1000)
            else:
                w.setRange(-10000, 10000)
            w.setValue(value)
            return w

    def _add_binarization_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            group = QGroupBox("Binarization threshold")
            group.setCheckable(True)
            group.setChecked(self.current_settings['Binarization']['enabled'])
            form = QFormLayout(group)
            start = QSpinBox(); start.setRange(0, 255); start.setValue(self.current_settings['Binarization']['threshold_start'])
            end = QSpinBox(); end.setRange(0, 255); end.setValue(self.current_settings['Binarization']['threshold_end'])
            step = QSpinBox(); step.setRange(1, 255); step.setValue(self.current_settings['Binarization']['threshold_step'])
            form.addRow("Start", start)
            form.addRow("End", end)
            form.addRow("Step", step)
            layout.addWidget(group)
            self.tabs.addTab(tab, "Binarization")
            self.binarization_group = group
            self.binarization_start = start
            self.binarization_end = end
            self.binarization_step = step
        except Exception as e:
            self.log(f"Ошибка при создании вкладки Binarization: {e}\n{traceback.format_exc()}")
            raise

    def _add_object_filtering_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            group = QGroupBox("Object filtering")
            group.setCheckable(True)
            group.setChecked(self.current_settings['ObjectFiltering']['enabled'])
            form = QFormLayout(group)

            # Fill contours – всегда показываем два чекбокса (True и False)
            fill_group = QGroupBox("Fill contours")
            fill_layout = QVBoxLayout()
            self.fill_checkboxes = []
            for val in [True, False]:
                cb = QCheckBox(f"Fill = {val}")
                fill_layout.addWidget(cb)
                self.fill_checkboxes.append(cb)
            fill_group.setLayout(fill_layout)
            form.addRow(fill_group)

            # Min area percent (float 0.0-0.1)
            min_area_start = QDoubleSpinBox()
            min_area_start.setRange(0.0, 0.1)
            min_area_start.setSingleStep(0.001)
            min_area_start.setValue(self.current_settings['ObjectFiltering']['min_area_percent_start'])
            min_area_end = QDoubleSpinBox()
            min_area_end.setRange(0.0, 0.1)
            min_area_end.setSingleStep(0.001)
            min_area_end.setValue(self.current_settings['ObjectFiltering']['min_area_percent_end'])
            min_area_step = QDoubleSpinBox()
            min_area_step.setRange(0.001, 0.1)
            min_area_step.setSingleStep(0.001)
            min_area_step.setValue(self.current_settings['ObjectFiltering']['min_area_percent_step'])
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel("Start (%):")); hbox.addWidget(min_area_start)
            hbox.addWidget(QLabel("End (%):")); hbox.addWidget(min_area_end)
            hbox.addWidget(QLabel("Step (%):")); hbox.addWidget(min_area_step)
            form.addRow("Min area %", hbox)

            layout.addWidget(group)
            self.tabs.addTab(tab, "Object filtering")
            self.object_filtering_group = group
            self.object_filtering_min_area_start = min_area_start
            self.object_filtering_min_area_end = min_area_end
            self.object_filtering_min_area_step = min_area_step
        except Exception as e:
            self.log(f"Ошибка при создании вкладки Object filtering: {e}\n{traceback.format_exc()}")
            raise

    def _add_morphology_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            group = QGroupBox("Morphology")
            group.setCheckable(True)
            group.setChecked(self.current_settings['Morphology']['enabled'])
            form = QFormLayout(group)

            # Kernel shapes – всегда показываем все три формы
            shapes_group = QGroupBox("Kernel shapes")
            shapes_layout = QVBoxLayout()
            self.shape_checkboxes = []
            all_shapes = ["Rectangle", "Ellipse", "Cross"]
            for shape in all_shapes:
                cb = QCheckBox(shape)
                shapes_layout.addWidget(cb)
                self.shape_checkboxes.append(cb)
            shapes_group.setLayout(shapes_layout)
            form.addRow(shapes_group)

            # Closing
            close_cb = QCheckBox("Enable closing")
            close_cb.setChecked(self.current_settings['Morphology']['close_enabled'])
            form.addRow("", close_cb)
            close_start = QDoubleSpinBox()
            close_start.setRange(0, 0.05)
            close_start.setSingleStep(0.001)
            close_start.setValue(self.current_settings['Morphology']['close_start'])
            close_end = QDoubleSpinBox()
            close_end.setRange(0, 0.05)
            close_end.setSingleStep(0.001)
            close_end.setValue(self.current_settings['Morphology']['close_end'])
            close_step = QDoubleSpinBox()
            close_step.setRange(0.001, 0.01)
            close_step.setSingleStep(0.001)
            close_step.setValue(self.current_settings['Morphology']['close_step'])
            hbox_close = QHBoxLayout()
            hbox_close.addWidget(QLabel("Start:")); hbox_close.addWidget(close_start)
            hbox_close.addWidget(QLabel("End:")); hbox_close.addWidget(close_end)
            hbox_close.addWidget(QLabel("Step:")); hbox_close.addWidget(close_step)
            form.addRow("Close factor", hbox_close)

            # Opening
            open_cb = QCheckBox("Enable opening")
            open_cb.setChecked(self.current_settings['Morphology']['open_enabled'])
            form.addRow("", open_cb)
            open_start = QDoubleSpinBox()
            open_start.setRange(0, 0.05)
            open_start.setSingleStep(0.001)
            open_start.setValue(self.current_settings['Morphology']['open_start'])
            open_end = QDoubleSpinBox()
            open_end.setRange(0, 0.05)
            open_end.setSingleStep(0.001)
            open_end.setValue(self.current_settings['Morphology']['open_end'])
            open_step = QDoubleSpinBox()
            open_step.setRange(0.001, 0.01)
            open_step.setSingleStep(0.001)
            open_step.setValue(self.current_settings['Morphology']['open_step'])
            hbox_open = QHBoxLayout()
            hbox_open.addWidget(QLabel("Start:")); hbox_open.addWidget(open_start)
            hbox_open.addWidget(QLabel("End:")); hbox_open.addWidget(open_end)
            hbox_open.addWidget(QLabel("Step:")); hbox_open.addWidget(open_step)
            form.addRow("Open factor", hbox_open)

            layout.addWidget(group)
            self.tabs.addTab(tab, "Morphology")
            self.morphology_group = group
            self.morphology_close_cb = close_cb
            self.morphology_close_start = close_start
            self.morphology_close_end = close_end
            self.morphology_close_step = close_step
            self.morphology_open_cb = open_cb
            self.morphology_open_start = open_start
            self.morphology_open_end = open_end
            self.morphology_open_step = open_step
        except Exception as e:
            self.log(f"Ошибка при создании вкладки Morphology: {e}\n{traceback.format_exc()}")
            raise

    def _add_invert_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            group = QGroupBox("Invert mask")
            form = QFormLayout(group)
            self.invert_checkboxes = []
            for val in [False, True]:
                cb = QCheckBox(f"Invert = {val}")
                form.addRow(cb)
                self.invert_checkboxes.append(cb)
            layout.addWidget(group)
            self.tabs.addTab(tab, "Invert")
            self.invert_group = group
        except Exception as e:
            self.log(f"Ошибка при создании вкладки Invert: {e}\n{traceback.format_exc()}")
            raise

    def _load_settings(self):
        try:
            self.log("Загрузка сохранённых настроек в диалог")
            # Загрузка методов
            for method in ['Sobel', 'Scharr', 'Laplacian', 'Canny', 'Prewitt', 'Roberts', 'Kirsch']:
                group = getattr(self, f"{method}_group")
                group.setChecked(self.current_settings[method]['enabled'])
                for key in self.current_settings[method]:
                    if key == 'enabled':
                        continue
                    if key.endswith('_start'):
                        base = key[:-6]
                        start_w = getattr(self, f"{method}_{base}_start")
                        end_w = getattr(self, f"{method}_{base}_end")
                        step_w = getattr(self, f"{method}_{base}_step")
                        start_w.setValue(self.current_settings[method][key])
                        end_w.setValue(self.current_settings[method][f'{base}_end'])
                        step_w.setValue(self.current_settings[method][f'{base}_step'])

            # Binarization
            bin_settings = self.current_settings['Binarization']
            self.binarization_group.setChecked(bin_settings['enabled'])
            self.binarization_start.setValue(bin_settings['threshold_start'])
            self.binarization_end.setValue(bin_settings['threshold_end'])
            self.binarization_step.setValue(bin_settings['threshold_step'])

            # Object filtering
            obj_settings = self.current_settings['ObjectFiltering']
            self.object_filtering_group.setChecked(obj_settings['enabled'])
            self.object_filtering_min_area_start.setValue(obj_settings['min_area_percent_start'])
            self.object_filtering_min_area_end.setValue(obj_settings['min_area_percent_end'])
            self.object_filtering_min_area_step.setValue(obj_settings['min_area_percent_step'])
            # Восстанавливаем состояние чекбоксов fill_contours
            fill_values = obj_settings.get('fill_contours', [True])
            for i, cb in enumerate(self.fill_checkboxes):
                # i=0 -> False, i=1 -> True? Порядок: сначала False, потом True
                # Лучше определить значение по тексту чекбокса
                cb_text = cb.text().split('=')[1].strip()
                cb_value = (cb_text == 'True')
                cb.setChecked(cb_value in fill_values)

            # Morphology
            morph = self.current_settings['Morphology']
            self.morphology_group.setChecked(morph['enabled'])
            self.morphology_close_cb.setChecked(morph['close_enabled'])
            self.morphology_close_start.setValue(morph['close_start'])
            self.morphology_close_end.setValue(morph['close_end'])
            self.morphology_close_step.setValue(morph['close_step'])
            self.morphology_open_cb.setChecked(morph['open_enabled'])
            self.morphology_open_start.setValue(morph['open_start'])
            self.morphology_open_end.setValue(morph['open_end'])
            self.morphology_open_step.setValue(morph['open_step'])
            # Восстанавливаем состояние чекбоксов kernel_shapes
            kernel_shapes = morph.get('kernel_shapes', ["Rectangle", "Ellipse", "Cross"])
            for cb in self.shape_checkboxes:
                cb.setChecked(cb.text() in kernel_shapes)

            # Invert
            invert_values = self.current_settings.get('Invert', [False])
            for cb in self.invert_checkboxes:
                cb_text = cb.text().split('=')[1].strip()
                cb_value = (cb_text == 'True')
                cb.setChecked(cb_value in invert_values)

            self.log("Настройки успешно загружены")
        except Exception as e:
            self.log(f"Ошибка при загрузке настроек: {e}\n{traceback.format_exc()}")
            raise

    def get_settings(self):
        try:
            settings = {}
            for method in ['Sobel', 'Scharr', 'Laplacian', 'Canny', 'Prewitt', 'Roberts', 'Kirsch']:
                group = getattr(self, f"{method}_group")
                s = {'enabled': group.isChecked()}
                default = self._default_settings()[method]
                for key in default:
                    if key == 'enabled':
                        continue
                    if key.endswith('_start'):
                        base = key[:-6]
                        s[key] = getattr(self, f"{method}_{base}_start").value()
                        s[f'{base}_end'] = getattr(self, f"{method}_{base}_end").value()
                        s[f'{base}_step'] = getattr(self, f"{method}_{base}_step").value()
                settings[method] = s

            # Binarization
            settings['Binarization'] = {
                'enabled': self.binarization_group.isChecked(),
                'threshold_start': self.binarization_start.value(),
                'threshold_end': self.binarization_end.value(),
                'threshold_step': self.binarization_step.value()
            }

            # Object filtering
            fill_values = []
            for cb in self.fill_checkboxes:
                if cb.isChecked():
                    val_str = cb.text().split('=')[1].strip()
                    fill_values.append(val_str == 'True')
            if not fill_values:
                fill_values = [False]
            settings['ObjectFiltering'] = {
                'enabled': self.object_filtering_group.isChecked(),
                'fill_contours': fill_values,
                'min_area_percent_start': self.object_filtering_min_area_start.value(),
                'min_area_percent_end': self.object_filtering_min_area_end.value(),
                'min_area_percent_step': self.object_filtering_min_area_step.value()
            }

            # Morphology
            kernel_shapes = [cb.text() for cb in self.shape_checkboxes if cb.isChecked()]
            if not kernel_shapes:
                kernel_shapes = ["Rectangle"]  # fallback
            settings['Morphology'] = {
                'enabled': self.morphology_group.isChecked(),
                'close_enabled': self.morphology_close_cb.isChecked(),
                'close_start': self.morphology_close_start.value(),
                'close_end': self.morphology_close_end.value(),
                'close_step': self.morphology_close_step.value(),
                'open_enabled': self.morphology_open_cb.isChecked(),
                'open_start': self.morphology_open_start.value(),
                'open_end': self.morphology_open_end.value(),
                'open_step': self.morphology_open_step.value(),
                'kernel_shapes': kernel_shapes
            }

            # Invert
            invert_values = []
            for cb in self.invert_checkboxes:
                if cb.isChecked():
                    val_str = cb.text().split('=')[1].strip()
                    invert_values.append(val_str == 'True')
            if not invert_values:
                invert_values = [False]
            settings['Invert'] = invert_values

            return settings
        except Exception as e:
            self.log(f"Ошибка при получении настроек: {e}\n{traceback.format_exc()}")
            raise