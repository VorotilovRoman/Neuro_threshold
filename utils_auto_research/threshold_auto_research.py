from import_libs_external import *

class AutoResearch:
    def __init__(self, main_window, search_all=False):
        self.main = main_window
        self.search_all = search_all

    def run(self):
        if not self.main.display_images:
            self.main.log("Нет загруженных изображений для автоподбора.")
            return

        # Сохраняем текущие настройки интерфейса
        saved_settings = {
            'method': self.main.method_combo.currentIndex(),
            'draw_mode': self.main.draw_combo.currentIndex(),
            'invert': self.main.invert_checkbox.isChecked(),
            'hull': self.main.hull_checkbox.isChecked(),
            'close_kernel': self.main.close_kernel_slider.value(),
            'open_kernel': self.main.open_kernel_slider.value(),
            'kernel_shape': self.main.kernel_shape_combo.currentIndex(),
            'threshold': self.main.threshold_slider.value(),
            'current_index': self.main.current_index
        }

        # Фиксированные настройки для перебора
        self.main.draw_combo.setCurrentText("Segmentation (Polygon)")

        self.main.hull_checkbox.setChecked(False)
        self.main.kernel_shape_combo.setCurrentText("Rectangle")
        self.main.threshold_slider.blockSignals(True)

        # Получаем настройки автоподбора
        settings = self.main.auto_settings
        if settings is None:
            self.main.log("Настройки автоподбора не найдены. Используются настройки по умолчанию (все методы с дефолтными диапазонами).")
            settings = self._default_settings()

        # Определяем список изображений для обработки
        if self.search_all:
            image_indices = list(range(len(self.main.display_images)))
        else:
            image_indices = [self.main.current_index]

        # Предварительно подсчитываем общее количество комбинаций
        total_combos = 0
        for idx in image_indices:
            self.main.current_index = idx
            combos = self._generate_combinations(settings)
            total_combos += len(combos)

        # Прогресс-диалог
        progress = QProgressDialog("Автоподбор параметров...", "Отмена", 0, total_combos, self.main)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        processed = 0

        try:
            for img_idx in image_indices:
                self.main.current_index = img_idx
                self.main.display_current_image()  # обновляем изображение в памяти

                img_path = self.main.image_paths[img_idx]
                base_name = os.path.splitext(img_path)[0]
                report_path = base_name + "_full_report.txt"

                combos = self._generate_combinations(settings)

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("Отчёт автоподбора (полный перебор методов и параметров)\n")
                    f.write(f"Изображение: {img_path}\n")
                    f.write("Фиксированные настройки: отрисовка Contours (simple), ядро Rectangle, hull выкл\n\n")

                    for combo in combos:
                        if progress.wasCanceled():
                            break

                        progress.setValue(processed)
                        progress.setLabelText(f"Обработка {processed+1}/{total_combos}: {combo['desc']}")

                        try:
                            self.main.method_combo.setCurrentText(combo['method'])
                            self._apply_method_params(combo['method'], combo['params'])

                            self.main.invert_checkbox.setChecked(combo['invert'])
                            self.main.close_kernel_slider.setValue(int(combo['close_factor'] * 100))
                            self.main.open_kernel_slider.setValue(int(combo['open_factor'] * 100))

                            self.main.display_current_image()

                            objects = self.main.current_objects_full
                            count = len(objects)

                            # Формируем строку параметров в одну строку
                            param_line = f"=== {combo['desc']}, инверсия={'Да' if combo['invert'] else 'Нет'}, close={combo['close_factor']:.2f}, open={combo['open_factor']:.2f} | Количество объектов: {count} ===\n"
                            f.write(param_line)

                            if count > 0:
                                img_h, img_w = self.main.display_images[img_idx].shape[:2]
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
                            self.main.log(f"Ошибка при обработке {combo['desc']}: {e}")
                        finally:
                            processed += 1

                        if progress.wasCanceled():
                            break

                # Логируем сохранение отчёта после обработки всех комбинаций для текущего изображения
                if not progress.wasCanceled():
                    self.main.log(f"Сохранён отчёт для {os.path.basename(img_path)}: {report_path}")

                if progress.wasCanceled():
                    break

            if progress.wasCanceled():
                self.main.log("Автоподбор прерван пользователем.")
            else:
                self.main.log(f"Автоподбор завершён. Обработано {len(image_indices)} изображений. Отчёты сохранены в папках с изображениями.")

        except Exception as e:
            self.main.log(f"Ошибка при автоподборе: {e}")
        finally:
            # Восстанавливаем настройки интерфейса
            self.main.threshold_slider.blockSignals(False)
            self.main.method_combo.setCurrentIndex(saved_settings['method'])
            self.main.draw_combo.setCurrentIndex(saved_settings['draw_mode'])
            self.main.invert_checkbox.setChecked(saved_settings['invert'])
            self.main.hull_checkbox.setChecked(saved_settings['hull'])
            self.main.close_kernel_slider.setValue(saved_settings['close_kernel'])
            self.main.open_kernel_slider.setValue(saved_settings['open_kernel'])
            self.main.kernel_shape_combo.setCurrentIndex(saved_settings['kernel_shape'])
            self.main.threshold_slider.setValue(saved_settings['threshold'])
            self.main.current_index = saved_settings['current_index']
            self.main.display_current_image()
            progress.close()

    def _default_settings(self):
        """Возвращает словарь настроек по умолчанию (все методы включены с типичными диапазонами)."""
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

    def _generate_combinations(self, settings):
        """Генерирует список комбинаций параметров на основе переданных настроек,
        включая инверсию и морфологию."""
        bin_combos = self._generate_binarization_combos(settings)

        morph = settings.get('Morphology', {})
        morph_enabled = morph.get('enabled', False)

        invert_values = [False]
        if morph_enabled and morph.get('invert_enabled', False):
            invert_values = [False, True]

        close_values = [0.0]
        if morph_enabled and morph.get('close_enabled', False):
            close_start = morph['close_start']
            close_end = morph['close_end']
            close_step = morph['close_step']
            close_values = []
            val = close_start
            while val <= close_end + 1e-9:
                close_values.append(round(val, 2))
                val += close_step
            close_values = list(set(close_values))

        open_values = [0.0]
        if morph_enabled and morph.get('open_enabled', False):
            open_start = morph['open_start']
            open_end = morph['open_end']
            open_step = morph['open_step']
            open_values = []
            val = open_start
            while val <= open_end + 1e-9:
                open_values.append(round(val, 2))
                val += open_step
            open_values = list(set(open_values))

        all_combos = []
        for bc in bin_combos:
            for invert in invert_values:
                for close in close_values:
                    for openf in open_values:
                        new_desc = f"{bc['desc']}, инверсия={'Да' if invert else 'Нет'}, close={close:.2f}, open={openf:.2f}"
                        all_combos.append({
                            'method': bc['method'],
                            'params': bc['params'],
                            'desc': new_desc,
                            'invert': invert,
                            'close_factor': close,
                            'open_factor': openf
                        })
        return all_combos

    def _generate_binarization_combos(self, settings):
        """Генерирует только комбинации бинаризации (без морфологии/инверсии)."""
        combos = []

        # Simple Threshold
        if settings.get('Simple Threshold', {}).get('enabled', False):
            st = settings['Simple Threshold']
            for thresh in range(st['start'], st['end'] + 1, st['step']):
                combos.append({
                    'method': 'Simple Threshold',
                    'params': {'thresh': thresh},
                    'desc': f'Метод: Simple Threshold, порог={thresh}'
                })

        # Otsu
        if settings.get('Otsu', {}).get('enabled', False):
            combos.append({
                'method': 'Otsu',
                'params': {},
                'desc': 'Метод: Otsu'
            })

        # Triangle
        if settings.get('Triangle', {}).get('enabled', False):
            combos.append({
                'method': 'Triangle',
                'params': {},
                'desc': 'Метод: Triangle'
            })

        # Adaptive Mean
        am = settings.get('Adaptive Mean', {})
        if am.get('enabled', False):
            for ws in range(am['window_start'], am['window_end']+1, am['window_step']):
                w = ws if ws % 2 == 1 else ws + 1
                for c in range(am['c_start'], am['c_end']+1, am['c_step']):
                    combos.append({
                        'method': 'Adaptive Mean',
                        'params': {'ws': w, 'c': c},
                        'desc': f'Метод: Adaptive Mean, окно={w}, C={c}'
                    })

        # Adaptive Gauss
        ag = settings.get('Adaptive Gauss', {})
        if ag.get('enabled', False):
            for ws in range(ag['window_start'], ag['window_end']+1, ag['window_step']):
                w = ws if ws % 2 == 1 else ws + 1
                for c in range(ag['c_start'], ag['c_end']+1, ag['c_step']):
                    combos.append({
                        'method': 'Adaptive Gauss',
                        'params': {'ws': w, 'c': c},
                        'desc': f'Метод: Adaptive Gauss, окно={w}, C={c}'
                    })

        # Niblack
        nb = settings.get('Niblack', {})
        if nb.get('enabled', False):
            for ws in range(nb['window_start'], nb['window_end']+1, nb['window_step']):
                w = ws if ws % 2 == 1 else ws + 1
                k_steps = int((nb['k_end'] - nb['k_start']) / nb['k_step']) + 1
                for i in range(k_steps):
                    k_val = nb['k_start'] + i * nb['k_step']
                    combos.append({
                        'method': 'Niblack',
                        'params': {'ws': w, 'k': round(k_val, 2)},
                        'desc': f'Метод: Niblack, окно={w}, k={round(k_val, 2)}'
                    })

        # Sauvola
        sv = settings.get('Sauvola', {})
        if sv.get('enabled', False):
            for ws in range(sv['window_start'], sv['window_end']+1, sv['window_step']):
                w = ws if ws % 2 == 1 else ws + 1
                k_steps = int((sv['k_end'] - sv['k_start']) / sv['k_step']) + 1
                for i in range(k_steps):
                    k_val = sv['k_start'] + i * sv['k_step']
                    for r in range(sv['r_start'], sv['r_end']+1, sv['r_step']):
                        combos.append({
                            'method': 'Sauvola',
                            'params': {'ws': w, 'k': round(k_val, 2), 'r': r},
                            'desc': f'Метод: Sauvola, окно={w}, k={round(k_val, 2)}, R={r}'
                        })

        # ISODATA
        iso = settings.get('ISODATA', {})
        if iso.get('enabled', False):
            for init in range(iso['init_start'], iso['init_end']+1, iso['init_step']):
                combos.append({
                    'method': 'ISODATA',
                    'params': {'init': init},
                    'desc': f'Метод: ISODATA, начальный порог={init}'
                })

        # Background Symmetry
        bs = settings.get('Background Symmetry', {})
        if bs.get('enabled', False):
            steps = int((bs['excess_end'] - bs['excess_start']) / bs['excess_step']) + 1
            for i in range(steps):
                excess = bs['excess_start'] + i * bs['excess_step']
                combos.append({
                    'method': 'Background Symmetry',
                    'params': {'excess': round(excess, 2)},
                    'desc': f'Метод: Background Symmetry, excess={round(excess, 2)}'
                })

        # Row Adaptive
        ra = settings.get('Row Adaptive', {})
        if ra.get('enabled', False):
            for ws in range(ra['window_start'], ra['window_end']+1, ra['window_step']):
                k_steps = int((ra['k_end'] - ra['k_start']) / ra['k_step']) + 1
                for i in range(k_steps):
                    k_val = ra['k_start'] + i * ra['k_step']
                    combos.append({
                        'method': 'Row Adaptive',
                        'params': {'ws': ws, 'k': round(k_val, 2)},
                        'desc': f'Метод: Row Adaptive, окно={ws}, k={round(k_val, 2)}'
                    })

        return combos

    def _apply_method_params(self, method, params):
        """Применяет параметры метода к соответствующим элементам управления."""
        if method == 'Simple Threshold':
            self.main.threshold_slider.setValue(params['thresh'])
        elif method in ['Adaptive Mean', 'Adaptive Gauss']:
            self.main.adaptive_win.setValue(params['ws'])
            self.main.adaptive_c.setValue(params['c'])
        elif method == 'Niblack':
            self.main.niblack_win.setValue(params['ws'])
            k_slider = int(params['k'] * 100)
            if k_slider < -100:
                k_slider = -100
            if k_slider > 100:
                k_slider = 100
            self.main.niblack_k.setValue(k_slider)
        elif method == 'Sauvola':
            self.main.sauvola_win.setValue(params['ws'])
            k_slider = int(params['k'] * 100)
            if k_slider < -100:
                k_slider = -100
            if k_slider > 100:
                k_slider = 100
            self.main.sauvola_k.setValue(k_slider)
            self.main.sauvola_r.setValue(params['r'])
        elif method == 'ISODATA':
            self.main.isodata_init.setValue(params['init'])
        elif method == 'Background Symmetry':
            excess_slider = int(params['excess'] * 100)
            if excess_slider < 1:
                excess_slider = 1
            if excess_slider > 100:
                excess_slider = 100
            self.main.bg_excess.setValue(excess_slider)
        elif method == 'Row Adaptive':
            self.main.row_win.setValue(params['ws'])
            k_slider = int(params['k'] * 100)
            if k_slider < 0:
                k_slider = 0
            if k_slider > 100:
                k_slider = 100
            self.main.row_k.setValue(k_slider)
        # Otsu, Triangle не имеют параметров