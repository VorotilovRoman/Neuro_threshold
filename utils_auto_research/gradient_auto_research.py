from import_libs_external import *

class GradientAutoResearch:
    def __init__(self, parent_window, search_all=False):
        self.parent = parent_window
        self.search_all = search_all

    def run(self):
        if not self.parent.display_images:
            self.parent.log("Нет загруженных изображений для автоподбора.")
            return

        # Сохраняем текущие настройки интерфейса
        saved_settings = self._save_current_ui_state()

        # Фиксированные настройки для поиска
        self.parent.draw_combo.setCurrentText("Segmentation (Polygon)")
        self.parent.hull_checkbox.setChecked(False)

        # Получаем настройки автоподбора
        settings = self.parent.auto_settings
        if settings is None:
            self.parent.log("Настройки автоподбора не найдены. Используются настройки по умолчанию.")
            settings = self._default_settings()
        else:
            # Дополняем отсутствующие ключи значениями по умолчанию (для совместимости со старыми файлами)
            settings = self._merge_with_defaults(settings)

        # Список изображений
        if self.search_all:
            image_indices = list(range(len(self.parent.display_images)))
        else:
            image_indices = [self.parent.current_index]

        # Генерируем все комбинации
        all_combos = self._generate_combinations(settings)
        total_combos = len(all_combos) * len(image_indices)

        progress = QProgressDialog("Автоподбор градиентных методов...", "Отмена", 0, total_combos, self.parent)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        processed = 0
        try:
            for img_idx in image_indices:
                self.parent.current_index = img_idx
                self.parent.display_current_image()
                img_path = self.parent.image_paths[img_idx]
                base_name = os.path.splitext(img_path)[0]
                report_path = base_name + "_gradient_report.txt"

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("Отчёт автоподбора градиентных методов\n")
                    f.write(f"Изображение: {img_path}\n")
                    f.write("Фиксированные настройки: отрисовка Contours (simple), hull выкл\n\n")

                    for combo in all_combos:
                        if progress.wasCanceled():
                            break
                        progress.setValue(processed)
                        progress.setLabelText(f"Обработка {processed+1}/{total_combos}: {combo['desc']}")

                        try:
                            # Применяем метод и параметры
                            self.parent.method_combo.setCurrentText(combo['method'])
                            self._apply_method_params(combo['method'], combo['params'])
                            self.parent.threshold_slider.setValue(combo.get('threshold', 127))
                            self.parent.fill_contours_checkbox.setChecked(combo.get('fill_contours', True))
                            self.parent.min_area_spinbox.setValue(combo.get('min_area_percent', 0.0))
                            self.parent.invert_checkbox.setChecked(combo.get('invert', False))
                            self.parent.close_kernel_slider.setValue(int(combo.get('close_factor', 0.0) * 200))
                            self.parent.open_kernel_slider.setValue(int(combo.get('open_factor', 0.0) * 200))
                            self.parent.kernel_shape_combo.setCurrentText(combo.get('kernel_shape', 'Rectangle'))

                            self.parent.display_current_image()
                            objects = self.parent.current_objects_full
                            count = len(objects)

                            f.write(f"=== {combo['desc']} | Объектов: {count} ===\n")
                            if count > 0:
                                img_h, img_w = self.parent.display_images[img_idx].shape[:2]
                                for obj in objects:
                                    if len(obj) == 4:
                                        x, y, w, h = obj
                                        cx = (x + w/2) / img_w
                                        cy = (y + h/2) / img_h
                                        nw = w / img_w
                                        nh = h / img_h
                                        f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                                    else:
                                        f.write(f"# Нестандартный объект: {obj}\n")
                            f.write("\n")
                            f.flush()
                            QApplication.processEvents()
                        except Exception as e:
                            f.write(f"!!! Ошибка: {e}\n\n")
                            self.parent.log(f"Ошибка при {combo['desc']}: {e}")
                        finally:
                            processed += 1
                            if progress.wasCanceled():
                                break

                if not progress.wasCanceled():
                    self.parent.log(f"Сохранён отчёт: {report_path}")
                if progress.wasCanceled():
                    break

            if progress.wasCanceled():
                self.parent.log("Автоподбор прерван пользователем.")
            else:
                self.parent.log(f"Автоподбор завершён. Обработано {len(image_indices)} изображений.")
        except Exception as e:
            self.parent.log(f"Ошибка при автоподборе: {e}")
        finally:
            self._restore_ui_state(saved_settings)
            progress.close()

    def _merge_with_defaults(self, settings):
        """Дополняет переданные настройки недостающими ключами из дефолтных."""
        default = self._default_settings()
        merged = settings.copy()
        for key in default:
            if key not in merged:
                merged[key] = default[key]
                self.parent.log(f"Добавлен отсутствующий ключ '{key}' в настройки автоподбора")
            elif isinstance(default[key], dict) and isinstance(merged[key], dict):
                # Рекурсивное дополнение для вложенных словарей
                for subkey in default[key]:
                    if subkey not in merged[key]:
                        merged[key][subkey] = default[key][subkey]
                        self.parent.log(f"Добавлен отсутствующий ключ '{key}.{subkey}' в настройки")
        return merged

    def _save_current_ui_state(self):
        return {
            'method': self.parent.method_combo.currentIndex(),
            'draw_mode': self.parent.draw_combo.currentIndex(),
            'threshold': self.parent.threshold_slider.value(),
            'fill_contours': self.parent.fill_contours_checkbox.isChecked(),
            'min_area_percent': self.parent.min_area_spinbox.value(),
            'invert': self.parent.invert_checkbox.isChecked(),
            'close_kernel': self.parent.close_kernel_slider.value(),
            'open_kernel': self.parent.open_kernel_slider.value(),
            'kernel_shape': self.parent.kernel_shape_combo.currentIndex(),
            'hull': self.parent.hull_checkbox.isChecked(),
            'current_index': self.parent.current_index
        }

    def _restore_ui_state(self, state):
        self.parent.method_combo.setCurrentIndex(state['method'])
        self.parent.draw_combo.setCurrentIndex(state['draw_mode'])
        self.parent.threshold_slider.setValue(state['threshold'])
        self.parent.fill_contours_checkbox.setChecked(state['fill_contours'])
        self.parent.min_area_spinbox.setValue(state['min_area_percent'])
        self.parent.invert_checkbox.setChecked(state['invert'])
        self.parent.close_kernel_slider.setValue(state['close_kernel'])
        self.parent.open_kernel_slider.setValue(state['open_kernel'])
        self.parent.kernel_shape_combo.setCurrentIndex(state['kernel_shape'])
        self.parent.hull_checkbox.setChecked(state['hull'])
        self.parent.current_index = state['current_index']
        self.parent.display_current_image()

    def _default_settings(self):
        return {
            'Sobel': {'enabled': True, 'ksize_start': 3, 'ksize_end': 7, 'ksize_step': 2,
                      'scale_start': 1, 'scale_end': 3, 'scale_step': 1},
            'Scharr': {'enabled': True, 'scale_start': 1, 'scale_end': 3, 'scale_step': 1},
            'Laplacian': {'enabled': True, 'ksize_start': 3, 'ksize_end': 7, 'ksize_step': 2,
                          'scale_start': 1, 'scale_end': 3, 'scale_step': 1},
            'Canny': {'enabled': True, 'thresh1_start': 30, 'thresh1_end': 150, 'thresh1_step': 20,
                      'thresh2_start': 100, 'thresh2_end': 250, 'thresh2_step': 30,
                      'aperture_start': 3, 'aperture_end': 5, 'aperture_step': 2},
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

    def _generate_combinations(self, settings):
        combos = []
        gradient_methods = ['Sobel', 'Scharr', 'Laplacian', 'Canny', 'Prewitt', 'Roberts', 'Kirsch']
        for method in gradient_methods:
            method_settings = settings.get(method, {})
            if not method_settings.get('enabled', False):
                continue
            method_params = self._generate_method_params(method, method_settings)

            # Бинаризация
            bin_settings = settings.get('Binarization', {})
            if bin_settings.get('enabled', True):
                thresh_values = self._range_values(
                    bin_settings.get('threshold_start', 0),
                    bin_settings.get('threshold_end', 255),
                    bin_settings.get('threshold_step', 10)
                )
            else:
                thresh_values = [127]  # фиксированное значение по умолчанию

            # Object filtering
            obj_settings = settings.get('ObjectFiltering', {})
            if obj_settings.get('enabled', True):
                fill_values = obj_settings.get('fill_contours', [True])
                if isinstance(fill_values, bool):
                    fill_values = [fill_values]
                min_area_values = self._range_values(
                    obj_settings.get('min_area_percent_start', 0.0),
                    obj_settings.get('min_area_percent_end', 0.1),
                    obj_settings.get('min_area_percent_step', 0.01)
                )
            else:
                fill_values = [True]  # фиксированное значение (заливка включена)
                min_area_values = [0.0]  # фиксированное значение (0%)

            # Morphology
            morph_settings = settings.get('Morphology', {})
            if morph_settings.get('enabled', True):
                if morph_settings.get('close_enabled', True):
                    close_values = self._range_values(
                        morph_settings.get('close_start', 0.0),
                        morph_settings.get('close_end', 0.05),
                        morph_settings.get('close_step', 0.005)
                    )
                else:
                    close_values = [0.0]
                if morph_settings.get('open_enabled', True):
                    open_values = self._range_values(
                        morph_settings.get('open_start', 0.0),
                        morph_settings.get('open_end', 0.05),
                        morph_settings.get('open_step', 0.005)
                    )
                else:
                    open_values = [0.0]
                kernel_shapes = morph_settings.get('kernel_shapes', ["Rectangle"])
            else:
                close_values = [0.0]
                open_values = [0.0]
                kernel_shapes = ["Rectangle"]

            # Инверсия
            invert_settings = settings.get('Invert', [False])
            if isinstance(invert_settings, bool):
                invert_settings = [invert_settings]
            # Если Invert как группа не предусмотрена, просто используем как есть
            # (в настройках Invert - это список, не группа с enabled)
            invert_values = invert_settings

            for params in method_params:
                for thresh in thresh_values:
                    for fill in fill_values:
                        for min_area in min_area_values:
                            for close in close_values:
                                for openf in open_values:
                                    for shape in kernel_shapes:
                                        for invert in invert_values:
                                            desc = (f"Метод: {method}, {self._params_desc(params)}, "
                                                    f"порог={thresh}, заливка={'Да' if fill else 'Нет'}, "
                                                    f"min_area={min_area:.3f}%, close={close:.3f}, open={openf:.3f}, "
                                                    f"ядро={shape}, инверсия={'Да' if invert else 'Нет'}")
                                            combos.append({
                                                'method': method,
                                                'params': params,
                                                'desc': desc,
                                                'threshold': thresh,
                                                'fill_contours': fill,
                                                'min_area_percent': min_area,
                                                'close_factor': close,
                                                'open_factor': openf,
                                                'kernel_shape': shape,
                                                'invert': invert
                                            })
        return combos


    def _generate_method_params(self, method, method_settings):
        params_list = [{}]
        if method == 'Sobel':
            ksize_list = self._range_values(method_settings.get('ksize_start', 3), method_settings.get('ksize_end', 7), method_settings.get('ksize_step', 2))
            scale_list = self._range_values(method_settings.get('scale_start', 1), method_settings.get('scale_end', 3), method_settings.get('scale_step', 1))
            params_list = [{'ksize': k, 'scale': s} for k in ksize_list for s in scale_list]
        elif method == 'Scharr':
            scale_list = self._range_values(method_settings.get('scale_start', 1), method_settings.get('scale_end', 3), method_settings.get('scale_step', 1))
            params_list = [{'scale': s} for s in scale_list]
        elif method == 'Laplacian':
            ksize_list = self._range_values(method_settings.get('ksize_start', 3), method_settings.get('ksize_end', 7), method_settings.get('ksize_step', 2))
            scale_list = self._range_values(method_settings.get('scale_start', 1), method_settings.get('scale_end', 3), method_settings.get('scale_step', 1))
            params_list = [{'ksize': k, 'scale': s} for k in ksize_list for s in scale_list]
        elif method == 'Canny':
            t1_list = self._range_values(method_settings.get('thresh1_start', 30), method_settings.get('thresh1_end', 150), method_settings.get('thresh1_step', 20))
            t2_list = self._range_values(method_settings.get('thresh2_start', 100), method_settings.get('thresh2_end', 250), method_settings.get('thresh2_step', 30))
            aperture_list = self._range_values(method_settings.get('aperture_start', 3), method_settings.get('aperture_end', 5), method_settings.get('aperture_step', 2))
            params_list = [{'threshold1': t1, 'threshold2': t2, 'aperture': ap} for t1 in t1_list for t2 in t2_list for ap in aperture_list]
        elif method in ('Prewitt', 'Roberts', 'Kirsch'):
            params_list = [{}]
        return params_list

    def _range_values(self, start, end, step):
        values = []
        if isinstance(step, float):
            val = start
            while val <= end + 1e-9:
                values.append(round(val, 3))
                val += step
        else:
            val = start
            while val <= end:
                values.append(val)
                val += step
        values = list(dict.fromkeys(values))
        return values

    def _params_desc(self, params):
        if not params:
            return "без параметров"
        parts = []
        for k, v in params.items():
            parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _apply_method_params(self, method, params):
        if method == 'Sobel':
            if 'ksize' in params:
                self.parent.sobel_kernel.setValue(params['ksize'])
            if 'scale' in params:
                self.parent.sobel_scale.setValue(params['scale'])
        elif method == 'Scharr':
            if 'scale' in params:
                self.parent.sobel_scale.setValue(params['scale'])
        elif method == 'Laplacian':
            if 'ksize' in params:
                self.parent.lap_kernel.setValue(params['ksize'])
            if 'scale' in params:
                self.parent.lap_scale.setValue(params['scale'])
        elif method == 'Canny':
            if 'threshold1' in params:
                self.parent.canny_thresh1.setValue(params['threshold1'])
            if 'threshold2' in params:
                self.parent.canny_thresh2.setValue(params['threshold2'])
            if 'aperture' in params:
                self.parent.canny_aperture.setValue(params['aperture'])