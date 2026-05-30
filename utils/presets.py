from import_libs_internal import *
from path_setup import get_project_root


class PresetManager:
    def __init__(self, main_window=None, preset_file="presets.json"):
        self.main = main_window
        root = get_project_root()
        presets_dir = os.path.join(root, "settings", "presets")
        os.makedirs(presets_dir, exist_ok=True)
        self.presets_file = os.path.join(presets_dir, preset_file)
        self.presets = {}
        self.load_presets()

    def load_presets(self):
        if os.path.exists(self.presets_file):
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    self.presets = json.load(f)
            except Exception as e:
                print(f"Error loading presets: {e}")
                self.presets = {}
        if not self.presets and self.main is not None:
            self.presets["default"] = self.get_current_settings()
            self.save_presets()

    def save_presets(self):
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving presets: {e}")

    # В методе get_current_settings – убедимся, что инверсия сохраняется
    def get_current_settings(self):
        if self.main is None:
            raise RuntimeError("get_current_settings() called without main_window")
        main = self.main
        method = main.method_combo.currentText()
        method = normalize_method_name(method)
        params = {}

        if method == "Simple Threshold":
            params["threshold"] = main.threshold_slider.value()
        elif method in ["Adaptive Mean", "Adaptive Gauss"]:
            params["window"] = main.adaptive_win.value()
            params["c"] = main.adaptive_c.value()
        elif method == "Niblack":
            params["window"] = main.niblack_win.value()
            params["k"] = main.niblack_k.value() / 100.0
        elif method == "Sauvola":
            params["window"] = main.sauvola_win.value()
            params["k"] = main.sauvola_k.value() / 100.0
            params["r"] = main.sauvola_r.value()
        elif method == "ISODATA":
            params["init"] = main.isodata_init.value()
        elif method == "Background Symmetry":
            params["excess"] = main.bg_excess.value() / 100.0
        elif method == "Row Adaptive":
            params["window"] = main.row_win.value()
            params["k"] = main.row_k.value() / 100.0

        settings = {
            "method": method,
            "params": params,
            "invert": main.invert_checkbox.isChecked(),  # ключ invert
            "close_factor": main.close_kernel_slider.value() / 100.0,  # ключ close_factor
            "open_factor": main.open_kernel_slider.value() / 100.0,  # ключ open_factor
            "draw_mode": main.draw_combo.currentText(),
            "use_hull": main.hull_checkbox.isChecked(),
            "kernel_shape": main.kernel_shape_combo.currentText()  # добавим форму ядра
        }
        print(settings)
        return settings

    def apply_preset(self, name):
        if self.main is None:
            raise RuntimeError("apply_preset() called without main_window")
        if name not in self.presets:
            return False
        preset = self.presets[name]
        main = self.main

        # Логируем для отладки
        print(f"Applying preset '{name}': invert={preset.get('invert', False)}")

        widgets_to_block = [
            main.method_combo, main.threshold_slider, main.adaptive_win, main.adaptive_c,
            main.niblack_win, main.niblack_k, main.sauvola_win, main.sauvola_k, main.sauvola_r,
            main.isodata_init, main.bg_excess, main.row_win, main.row_k,
            main.close_kernel_slider, main.open_kernel_slider, main.draw_combo,
            main.hull_checkbox, main.invert_checkbox, main.kernel_shape_combo
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        # Установка метода бинаризации
        method_text = preset["method"]
        normalized = normalize_method_name(method_text)
        idx = main.method_combo.findText(normalized)
        if idx < 0:
            idx = main.method_combo.findText(method_text)
        if idx >= 0:
            main.method_combo.setCurrentIndex(idx)
        else:
            if self.main:
                self.main.log(f"Предупреждение: метод '{method_text}' не найден в списке.")

        # Установка параметров бинаризации
        params = preset.get("params", {})
        if normalized == "Simple Threshold":
            if "threshold" in params:
                main.threshold_slider.setValue(params["threshold"])
            main.threshold_value_label.setText(str(main.threshold_slider.value()))
        elif normalized in ["Adaptive Mean", "Adaptive Gauss"]:
            win = params.get("window", 25)
            if win % 2 == 0:
                win += 1
            main.adaptive_win.setValue(win)
            main.adaptive_c.setValue(params.get("c", 3))
        elif normalized == "Niblack":
            main.niblack_win.setValue(params.get("window", 25))
            k_val = int(params.get("k", 0.2) * 100)
            main.niblack_k.setValue(k_val)
            main.niblack_k_label.setText(f"{main.niblack_k.value() / 100:.2f}")
        elif normalized == "Sauvola":
            main.sauvola_win.setValue(params.get("window", 25))
            k_val = int(params.get("k", 0.2) * 100)
            main.sauvola_k.setValue(k_val)
            main.sauvola_r.setValue(params.get("r", 128))
            main.sauvola_k_label.setText(f"{main.sauvola_k.value() / 100:.2f}")
            main.sauvola_r_label.setText(str(main.sauvola_r.value()))
        elif normalized == "ISODATA":
            main.isodata_init.setValue(params.get("init", 128))
        elif normalized == "Background Symmetry":
            excess_val = int(params.get("excess", 0.2) * 100)
            main.bg_excess.setValue(excess_val)
            main.bg_excess_label.setText(f"{main.bg_excess.value() / 100:.2f}")
        elif normalized == "Row Adaptive":
            main.row_win.setValue(params.get("window", 50))
            k_val = int(params.get("k", 0.5) * 100)
            main.row_k.setValue(k_val)
            main.row_k_label.setText(f"{main.row_k.value() / 100:.2f}")

        # Применение инверсии, морфологии, отрисовки
        main.invert_checkbox.setChecked(preset.get("invert", False))
        main.close_kernel_slider.setValue(int(preset.get("close_factor", 0.0) * 100))
        main.open_kernel_slider.setValue(int(preset.get("open_factor", 0.0) * 100))
        main.draw_combo.setCurrentText(preset.get("draw_mode", "Contours (simple)"))
        main.hull_checkbox.setChecked(preset.get("use_hull", False))
        # Форма ядра
        kernel_shape = preset.get("kernel_shape", "Rectangle")
        idx_shape = main.kernel_shape_combo.findText(kernel_shape)
        if idx_shape >= 0:
            main.kernel_shape_combo.setCurrentIndex(idx_shape)

        # Обновление текстовых меток
        main.close_kernel_label.setText(f"{main.close_kernel_slider.value() / 100:.2f}")
        main.open_kernel_label.setText(f"{main.open_kernel_slider.value() / 100:.2f}")

        # Разблокировка сигналов
        for w in widgets_to_block:
            w.blockSignals(False)

        # Принудительное обновление видимости контейнеров параметров
        main.on_method_changed(main.method_combo.currentIndex())

        # Принудительное обновление изображения – вызываем display_current_image напрямую
        main.display_current_image()
        return True


    def add_preset(self, name):
        if self.main is None:
            raise RuntimeError("add_preset() called without main_window")
        if not name:
            return False
        self.presets[name] = self.get_current_settings()
        self.save_presets()
        return True

    def add_preset_dict(self, name, preset_dict):
        if not name:
            return False
        self.presets[name] = preset_dict
        self.save_presets()
        return True

    def delete_preset(self, name):
        if name not in self.presets:
            return False
        del self.presets[name]
        self.save_presets()
        if not self.presets and self.main is not None:
            self.presets["default"] = self.get_current_settings()
            self.save_presets()
        return True

    def get_preset_names(self):
        return list(self.presets.keys())


class GradientPresetManager(PresetManager):
    def get_current_settings(self):
        main = self.main
        return {
            'method': main.method_combo.currentText(),
            'sobel_kernel': main.sobel_kernel.value(),
            'sobel_scale': main.sobel_scale.value(),
            'lap_kernel': main.lap_kernel.value(),
            'lap_scale': main.lap_scale.value(),
            'canny_thresh1': main.canny_thresh1.value(),
            'canny_thresh2': main.canny_thresh2.value(),
            'canny_aperture': main.canny_aperture.value(),
            'threshold': main.threshold_slider.value(),
            'fill_contours': main.fill_contours_checkbox.isChecked(),
            'min_area_percent': main.min_area_spinbox.value(),
            'invert': main.invert_checkbox.isChecked(),
            'close_factor': main.close_kernel_slider.value() / 200.0,
            'open_factor': main.open_kernel_slider.value() / 200.0,
            'kernel_shape': main.kernel_shape_combo.currentText(),
            'draw_mode': main.draw_combo.currentText(),
            'use_hull': main.hull_checkbox.isChecked()
        }

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        preset = self.presets[name]
        main = self.main

        widgets_to_block = [
            main.method_combo, main.sobel_kernel, main.sobel_scale,
            main.lap_kernel, main.lap_scale, main.canny_thresh1,
            main.canny_thresh2, main.canny_aperture, main.threshold_slider,
            main.fill_contours_checkbox, main.min_area_spinbox,
            main.invert_checkbox, main.close_kernel_slider, main.open_kernel_slider,
            main.kernel_shape_combo, main.draw_combo, main.hull_checkbox
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        main.method_combo.setCurrentText(preset.get('method', 'Sobel'))
        main.sobel_kernel.setValue(preset.get('sobel_kernel', 3))
        main.sobel_scale.setValue(preset.get('sobel_scale', 1))
        main.lap_kernel.setValue(preset.get('lap_kernel', 3))
        main.lap_scale.setValue(preset.get('lap_scale', 1))
        main.canny_thresh1.setValue(preset.get('canny_thresh1', 50))
        main.canny_thresh2.setValue(preset.get('canny_thresh2', 150))
        main.canny_aperture.setValue(preset.get('canny_aperture', 3))
        main.threshold_slider.setValue(preset.get('threshold', 127))
        main.fill_contours_checkbox.setChecked(preset.get('fill_contours', True))
        main.min_area_spinbox.setValue(preset.get('min_area_percent', 0.0))
        main.invert_checkbox.setChecked(preset.get('invert', False))
        main.close_kernel_slider.setValue(int(preset.get('close_factor', 0.0) * 200))
        main.open_kernel_slider.setValue(int(preset.get('open_factor', 0.0) * 200))
        main.kernel_shape_combo.setCurrentText(preset.get('kernel_shape', 'Rectangle'))
        main.draw_combo.setCurrentText(preset.get('draw_mode', 'Contours (simple)'))
        main.hull_checkbox.setChecked(preset.get('use_hull', False))

        main.threshold_label.setText(str(main.threshold_slider.value()))
        main.close_kernel_label.setText(f"{main.close_kernel_slider.value()/200:.3f}")
        main.open_kernel_label.setText(f"{main.open_kernel_slider.value()/200:.3f}")

        for w in widgets_to_block:
            w.blockSignals(False)

        main.on_method_changed(main.method_combo.currentIndex())
        main.schedule_update()
        return True


class TraditionalMLPresetManager(PresetManager):
    def get_current_settings(self):
        main = self.main
        # Сохраняем пути для всех типов моделей
        model_paths = getattr(main, 'model_paths', {})
        settings = {
            'model': main.model_combo.currentText(),
            'n_clusters': main.kmeans_clusters.value() if hasattr(main, 'kmeans_clusters') else 2,
            'svm_kernel': main.svm_kernel_combo.currentText() if hasattr(main, 'svm_kernel_combo') else 'rbf',
            'svm_c': main.svm_c.value() if hasattr(main, 'svm_c') else 1.0,
            'rf_trees': main.rf_trees.value() if hasattr(main, 'rf_trees') else 100,
            'xgb_trees': main.xgb_trees.value() if hasattr(main, 'xgb_trees') else 100,
            'invert': main.invert_checkbox.isChecked(),
            'close_factor': main.close_kernel_slider.value() / 100.0,
            'open_factor': main.open_kernel_slider.value() / 100.0,
            'draw_mode': main.draw_combo.currentText(),
            'use_hull': main.hull_checkbox.isChecked(),
            'model_paths': model_paths.copy()   # сохраняем весь словарь путей
        }
        return settings

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        preset = self.presets[name]
        main = self.main

        widgets = [main.model_combo, main.invert_checkbox, main.close_kernel_slider,
                   main.open_kernel_slider, main.draw_combo, main.hull_checkbox]
        if hasattr(main, 'kmeans_clusters'):
            widgets.append(main.kmeans_clusters)
        if hasattr(main, 'svm_kernel_combo'):
            widgets.append(main.svm_kernel_combo)
        if hasattr(main, 'svm_c'):
            widgets.append(main.svm_c)
        if hasattr(main, 'rf_trees'):
            widgets.append(main.rf_trees)
        if hasattr(main, 'xgb_trees'):
            widgets.append(main.xgb_trees)
        for w in widgets:
            w.blockSignals(True)

        main.model_combo.setCurrentText(preset.get('model', 'K‑Means'))
        if hasattr(main, 'kmeans_clusters'):
            main.kmeans_clusters.setValue(preset.get('n_clusters', 2))
        if hasattr(main, 'svm_kernel_combo'):
            main.svm_kernel_combo.setCurrentText(preset.get('svm_kernel', 'rbf'))
        if hasattr(main, 'svm_c'):
            main.svm_c.setValue(preset.get('svm_c', 1.0))
        if hasattr(main, 'rf_trees'):
            main.rf_trees.setValue(preset.get('rf_trees', 100))
        if hasattr(main, 'xgb_trees'):
            main.xgb_trees.setValue(preset.get('xgb_trees', 100))
        main.invert_checkbox.setChecked(preset.get('invert', False))
        main.close_kernel_slider.setValue(int(preset.get('close_factor', 0.0) * 100))
        main.open_kernel_slider.setValue(int(preset.get('open_factor', 0.0) * 100))
        main.draw_combo.setCurrentText(preset.get('draw_mode', 'Contours (simple)'))
        main.hull_checkbox.setChecked(preset.get('use_hull', False))

        # Восстанавливаем пути к моделям
        if 'model_paths' in preset:
            main.model_paths = preset['model_paths'].copy()
        else:
            main.model_paths = {}
        main._update_model_path_display()   # обновить поле пути для текущей модели
        # Если путь для текущей модели существует – загружаем модель
        main._auto_load_model_for_current_type()

        main.close_kernel_label.setText(f"{main.close_kernel_slider.value() / 100:.2f}")
        main.open_kernel_label.setText(f"{main.open_kernel_slider.value() / 100:.2f}")

        for w in widgets:
            w.blockSignals(False)

        main.on_model_changed(main.model_combo.currentIndex())
        main.schedule_update()
        return True


class DeepLearningPresetManager(PresetManager):
    def get_current_settings(self):
        main = self.main
        settings = {'model': main.model_combo.currentText(),
                    'input_size': main.input_size.value() if hasattr(main, 'input_size') else 512,
                    'sam_prompt': main.sam_prompt_combo.currentText() if hasattr(main, 'sam_prompt_combo') else 'box',
                    'device': main.device_combo.currentText(), 'invert': main.invert_checkbox.isChecked(),
                    'close_factor': main.close_kernel_slider.value() / 100.0,
                    'open_factor': main.open_kernel_slider.value() / 100.0, 'draw_mode': main.draw_combo.currentText(),
                    'use_hull': main.hull_checkbox.isChecked(),
                    'unet_model_path': main.unet_settings._model_path.text().strip(),
                    'deeplab_model_path': main.deeplab_settings._model_path.text().strip(),
                    'yolo_model_path': main.yolo_settings._model_path.text().strip(),
                    'onnx_model_path': main.custom_settings._model_path.text().strip(),
                    'yolo_conf': main.yolo_settings._conf.value(), 'yolo_iou': main.yolo_settings._iou.value(),
                    'yolo_imgsz': main.yolo_settings._imgsz.value(), 'yolo_save': main.yolo_settings._save.isChecked(),
                    'unet_encoder': main.unet_settings._encoder.currentText(),
                    'unet_input_size': main.unet_settings._input_size.currentText(),
                    'deeplab_backbone': main.deeplab_settings._backbone.currentText(),
                    'deeplab_output_stride': main.deeplab_settings._output_stride.currentText()}
        # Для U-Net и DeepLabV3+ можно также сохранить выбранные значения энкодеров и т.д.
        # (они и так заблокированы и восстанавливаются из метаданных модели, но тоже можно сохранить)
        return settings

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        preset = self.presets[name]
        main = self.main

        widgets = [
            main.model_combo, main.device_combo, main.invert_checkbox,
            main.close_kernel_slider, main.open_kernel_slider,
            main.draw_combo, main.hull_checkbox,
            main.yolo_settings._conf, main.yolo_settings._iou,
            main.yolo_settings._imgsz, main.yolo_settings._save,
            main.unet_settings._model_path, main.deeplab_settings._model_path,
            main.yolo_settings._model_path, main.custom_settings._model_path,
            main.unet_settings._encoder, main.unet_settings._input_size,
            main.deeplab_settings._backbone, main.deeplab_settings._output_stride
        ]
        for w in widgets:
            w.blockSignals(True)

        # Основные параметры
        main.model_combo.setCurrentText(preset.get('model', 'U‑Net'))
        main.device_combo.setCurrentText(preset.get('device', 'CPU'))
        if hasattr(main, 'input_size'):
            main.input_size.setValue(preset.get('input_size', 512))
        if hasattr(main, 'sam_prompt_combo'):
            main.sam_prompt_combo.setCurrentText(preset.get('sam_prompt', 'box'))
        main.invert_checkbox.setChecked(preset.get('invert', False))
        main.close_kernel_slider.setValue(int(preset.get('close_factor', 0.0) * 100))
        main.open_kernel_slider.setValue(int(preset.get('open_factor', 0.0) * 100))
        main.draw_combo.setCurrentText(preset.get('draw_mode', 'Contours (simple)'))
        main.hull_checkbox.setChecked(preset.get('use_hull', False))

        # Пути к моделям
        if 'unet_model_path' in preset:
            main.unet_settings._model_path.setText(preset['unet_model_path'])
        if 'deeplab_model_path' in preset:
            main.deeplab_settings._model_path.setText(preset['deeplab_model_path'])
        if 'yolo_model_path' in preset:
            main.yolo_settings._model_path.setText(preset['yolo_model_path'])
        if 'onnx_model_path' in preset:
            main.custom_settings._model_path.setText(preset['onnx_model_path'])

        # Параметры YOLO
        if 'yolo_conf' in preset:
            main.yolo_settings._conf.setValue(preset['yolo_conf'])
        if 'yolo_iou' in preset:
            main.yolo_settings._iou.setValue(preset['yolo_iou'])
        if 'yolo_imgsz' in preset:
            main.yolo_settings._imgsz.setValue(preset['yolo_imgsz'])
        if 'yolo_save' in preset:
            main.yolo_settings._save.setChecked(preset['yolo_save'])

        # Параметры U-Net (заблокированные поля)
        if 'unet_encoder' in preset:
            main.unet_settings._encoder.setCurrentText(preset['unet_encoder'])
        if 'unet_input_size' in preset:
            main.unet_settings._input_size.setCurrentText(preset['unet_input_size'])
        # Параметры DeepLabV3+
        if 'deeplab_backbone' in preset:
            main.deeplab_settings._backbone.setCurrentText(preset['deeplab_backbone'])
        if 'deeplab_output_stride' in preset:
            main.deeplab_settings._output_stride.setCurrentText(preset['deeplab_output_stride'])

        # Обновляем текстовые метки
        main.close_kernel_label.setText(f"{main.close_kernel_slider.value() / 100:.2f}")
        main.open_kernel_label.setText(f"{main.open_kernel_slider.value() / 100:.2f}")

        for w in widgets:
            w.blockSignals(False)

        # Переключаем видимость виджетов модели
        main.on_model_changed(main.model_combo.currentIndex())

        # Если путь к текущей модели существует – загружаем модель (перезагружаем с новыми параметрами)
        current_model = main.model_combo.currentText()
        path = ""
        if current_model == "U-Net":
            path = main.unet_settings._model_path.text().strip()
        elif current_model == "DeepLabV3+":
            path = main.deeplab_settings._model_path.text().strip()
        elif current_model == "YOLO-seg":
            path = main.yolo_settings._model_path.text().strip()
        elif current_model == "Custom ONNX":
            path = main.custom_settings._model_path.text().strip()
        if path and os.path.exists(path):
            main.load_model_from_widget(current_model)

        main.schedule_update()
        return True

class InteractivePresetManager(PresetManager):
    def get_current_settings(self):
        main = self.main
        return {
            'invert': main.invert_checkbox.isChecked(),
            'close_factor': main.close_kernel_slider.value() / 100.0,
            'open_factor': main.open_kernel_slider.value() / 100.0,
            'draw_mode': main.draw_combo.currentText(),
            'use_hull': main.hull_checkbox.isChecked()
        }

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        preset = self.presets[name]
        main = self.main
        widgets = [main.invert_checkbox, main.close_kernel_slider, main.open_kernel_slider,
                   main.draw_combo, main.hull_checkbox]
        for w in widgets:
            w.blockSignals(True)
        main.invert_checkbox.setChecked(preset.get('invert', False))
        main.close_kernel_slider.setValue(int(preset.get('close_factor', 0.0) * 100))
        main.open_kernel_slider.setValue(int(preset.get('open_factor', 0.0) * 100))
        main.draw_combo.setCurrentText(preset.get('draw_mode', 'Contours (simple)'))
        main.hull_checkbox.setChecked(preset.get('use_hull', False))
        main.close_kernel_label.setText(f"{main.close_kernel_slider.value()/100:.2f}")
        main.open_kernel_label.setText(f"{main.open_kernel_slider.value()/100:.2f}")
        for w in widgets:
            w.blockSignals(False)
        main.schedule_update()
        return True