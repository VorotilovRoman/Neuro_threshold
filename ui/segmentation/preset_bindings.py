"""Preset persistence orchestration and current pixel-based method bindings."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSlider,
    QSpinBox,
)

from path_setup import get_project_root
from utils_new.presets_store import read_presets, write_presets
from utils_new.segmentation.presets import make_preset, validate_preset


@contextmanager
def blocked_signals(widgets):
    previous = [
        (widget, widget.blockSignals(True)) for widget in dict.fromkeys(widgets)
    ]
    try:
        yield
    finally:
        for widget, state in reversed(previous):
            widget.blockSignals(state)


def control_value(widget):
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    return widget.value()


def set_control_value(widget, value):
    if isinstance(widget, QCheckBox):
        widget.setChecked(value)
    elif isinstance(widget, QComboBox):
        widget.setCurrentIndex(widget.findText(value))
    elif isinstance(widget, QLineEdit):
        widget.setText(value)
    else:
        widget.setValue(value)


def validate_control(widget, value):
    if isinstance(widget, QCheckBox):
        valid = isinstance(value, bool)
    elif isinstance(widget, QComboBox):
        valid = isinstance(value, str) and widget.findText(value) >= 0
    elif isinstance(widget, QLineEdit):
        valid = isinstance(value, str)
    else:
        valid = (
            type(value) in (int, float)
            and widget.minimum() <= value <= widget.maximum()
            and (isinstance(widget, QDoubleSpinBox) or type(value) is int)
        )
    if not valid:
        raise ValueError(f"Недопустимое значение параметра: {value!r}")


class PresetManager:
    """File collection and CRUD; subclasses define their own window state."""

    def __init__(self, main_window=None, preset_file="presets.json"):
        self.main = main_window
        self.presets_file = str(
            Path(get_project_root()) / "settings" / "presets" / preset_file
        )
        self.presets = {}
        self.load_error = None
        self.load_presets()

    def _validate(self, state):
        if not isinstance(state, dict):
            raise ValueError("Пресет должен быть словарём.")

    def load_presets(self):
        self.load_error = None
        try:
            self.presets = (
                read_presets(self.presets_file)
                if Path(self.presets_file).exists()
                else {}
            )
            for state in self.presets.values():
                self._validate(state)
        except (OSError, ValueError, TypeError) as exc:
            self.load_error = str(exc)
            self.presets = {}
            message = f"Не удалось прочитать пресеты {self.presets_file}: {exc}"
            if self.main is not None and hasattr(self.main, "log"):
                self.main.log(message)
            else:
                print(message)
        if not self.presets and self.main is not None:
            # In-memory default; opening a window never overwrites invalid/old files.
            self.presets["default"] = self.get_current_settings()

    def save_presets(self):
        for state in self.presets.values():
            self._validate(state)
        write_presets(self.presets_file, self.presets)

    def get_current_settings(self):
        raise NotImplementedError("A window-specific preset binding is required")

    def apply_preset(self, name):
        raise NotImplementedError("A window-specific preset binding is required")

    def add_preset(self, name):
        return self.add_preset_dict(name, self.get_current_settings())

    def add_preset_dict(self, name, preset_dict):
        if not name:
            return False
        self._validate(preset_dict)
        self.presets[name] = deepcopy(preset_dict)
        self.save_presets()
        return True

    def delete_preset(self, name):
        if name not in self.presets:
            return False
        del self.presets[name]
        if not self.presets and self.main is not None:
            self.presets["default"] = self.get_current_settings()
        self.save_presets()
        return True

    def get_preset_names(self):
        return list(self.presets)


class MethodPresetManager(PresetManager):
    kind = None
    fields = {}

    def _validate(self, state):
        validate_preset(state, self.kind)
        missing = self.fields.keys() - state.keys()
        if missing:
            raise ValueError(f'В пресете отсутствуют параметры: {sorted(missing)}')


    def _controls(self):
        result = {}
        for key, path in self.fields.items():
            widget = self.main
            for part in path.split("."):
                widget = getattr(widget, part)
            result[key] = widget
        return result

    def get_current_settings(self):
        main = self.main
        state = make_preset(
            self.kind,
            **{key: control_value(widget) for key, widget in self._controls().items()},
        )
        state.update(
            morphology=main.morphology_widget.get_settings(),
            invert=main.invert_checkbox.isChecked(),
            draw_mode=main.draw_combo.currentText(),
            use_hull=main.hull_checkbox.isChecked(),
        )
        self._capture_extra(state)
        self._validate(state)
        return state

    def _capture_extra(self, state):
        pass

    def _extra_controls(self, state):
        return {}

    def _before_apply(self, state):
        pass

    def _after_apply(self, state):
        self.main.schedule_update()

    def apply_preset(self, name):
        if name not in self.presets:
            return False
        state = self.presets[name]
        self._validate(state)
        main = self.main
        controls = self._controls()
        values = {key: state[key] for key in controls if key in state}
        extra = self._extra_controls(state)
        updates = [(controls[key], value) for key, value in values.items()] + list(
            extra.items()
        )
        updates += [
            (main.invert_checkbox, state["invert"]),
            (main.draw_combo, state["draw_mode"]),
            (main.hull_checkbox, state["use_hull"]),
        ]
        for widget, value in updates:
            validate_control(widget, value)
        with blocked_signals(
            [*(widget for widget, _ in updates), main.morphology_widget]
        ):
            self._before_apply(state)
            for widget, value in updates:
                set_control_value(widget, value)
            main.morphology_widget.set_settings(state["morphology"])
        self._after_apply(state)
        return True


class ThresholdPresetManager(MethodPresetManager):
    kind = "threshold"
    fields = {"method": "method_combo"}
    parameter_controls = {
        "Simple Threshold": {"threshold": ("threshold_slider", 1)},
        "Adaptive Mean": {"window": ("adaptive_win", 1), "c": ("adaptive_c", 1)},
        "Adaptive Gauss": {"window": ("adaptive_win", 1), "c": ("adaptive_c", 1)},
        "Niblack": {"window": ("niblack_win", 1), "k": ("niblack_k", 100)},
        "Sauvola": {
            "window": ("sauvola_win", 1),
            "k": ("sauvola_k", 100),
            "r": ("sauvola_r", 1),
        },
        "ISODATA": {"init": ("isodata_init", 1)},
        "Background Symmetry": {"excess": ("bg_excess", 100)},
        "Row Adaptive": {"window": ("row_win", 1), "k": ("row_k", 100)},
    }

    def _capture_extra(self, state):
        state["params"] = self.main.get_current_threshold_params()

    def _extra_controls(self, state):
        return {
            getattr(self.main, attr): round(state["params"][key] * scale)
            for key, (attr, scale) in self.parameter_controls.get(
                state["method"], {}
            ).items()
            if key in state["params"]
        }

    def _before_apply(self, state):
        # Visibility/reset comes first; then restore the actual preset parameters.
        self.main.method_combo.setCurrentText(state["method"])
        self.main.on_method_changed(self.main.method_combo.currentIndex())

    def _after_apply(self, state):
        main = self.main
        main.threshold_value_label.setText(str(main.threshold_slider.value()))
        for control, label, scale in [
            ("niblack_k", "niblack_k_label", 100),
            ("sauvola_k", "sauvola_k_label", 100),
            ("sauvola_r", "sauvola_r_label", 1),
            ("bg_excess", "bg_excess_label", 100),
            ("row_k", "row_k_label", 100),
        ]:
            value = getattr(main, control).value() / scale
            getattr(main, label).setText(
                str(int(value)) if scale == 1 else f"{value:.2f}"
            )
        if main._lazy_cache is not None:
            main.display_current_image()


class GradientPresetManager(MethodPresetManager):
    kind = "gradient"
    fields = {
        name: name
        for name in (
            "sobel_kernel",
            "sobel_scale",
            "lap_kernel",
            "lap_scale",
            "canny_thresh1",
            "canny_thresh2",
            "canny_aperture",
        )
    }
    fields.update(
        method="method_combo",
        threshold="threshold_slider",
        fill_contours="fill_contours_checkbox",
        min_area_percent="min_area_spinbox",
    )

    def _after_apply(self, state):
        self.main.threshold_label.setText(str(self.main.threshold_slider.value()))
        self.main.on_method_changed(self.main.method_combo.currentIndex())
        self.main.schedule_update()


class InteractivePresetManager(MethodPresetManager):
    kind = "interactive"
    fields = {"method": "method_combo"}

    def _parameter_controls(self):
        return {
            name: widget
            for name, widget in vars(self.main.params_widget).items()
            if isinstance(
                widget, (QSpinBox, QDoubleSpinBox, QSlider, QComboBox, QCheckBox)
            )
        }

    def _capture_extra(self, state):
        state["method_params"] = {
            name: control_value(widget)
            for name, widget in self._parameter_controls().items()
        }

    def _extra_controls(self, state):
        controls = self._parameter_controls()
        values = state.get("method_params")
        if not isinstance(values, dict) or values.keys() != controls.keys():
            raise ValueError('Пресет должен содержать полный набор параметров интерактивных методов.')
        return {controls[name]: value for name, value in values.items()}

    def _after_apply(self, state):
        self.main.on_method_changed(self.main.method_combo.currentIndex())
        self.main.schedule_update()


class TraditionalMLPresetManager(MethodPresetManager):
    kind = "traditional_ml"
    fields = dict(
        model="model_combo",
        n_clusters="kmeans_clusters",
        svm_kernel="svm_kernel_combo",
        svm_c="svm_c",
        rf_trees="rf_trees",
        xgb_trees="xgb_trees",
        color_feature="color_feature",
        texture_feature="texture_feature",
        spatial_feature="spatial_feature",
        use_superpixels="use_superpixels",
        superpixel_size="superpixel_size",
    )

    def _capture_extra(self, state):
        state["model_paths"] = self.main.model_paths.copy()

    def _after_apply(self, state):
        self.main.model_paths = deepcopy(state.get("model_paths", {}))
        self.main.on_superpixel_toggled(
            2 if self.main.use_superpixels.isChecked() else 0
        )
        self.main.on_model_changed(self.main.model_combo.currentIndex())
        self.main.schedule_update()


class DeepLearningPresetManager(MethodPresetManager):
    kind = "deep_learning"
    fields = dict(
        model="model_combo",
        device="device_combo",
        unet_model_path="unet_settings._model_path",
        deeplab_model_path="deeplab_settings._model_path",
        yolo_model_path="yolo_settings._model_path",
        onnx_model_path="custom_settings._model_path",
        yolo_conf="yolo_settings._conf",
        yolo_iou="yolo_settings._iou",
        yolo_imgsz="yolo_settings._imgsz",
        yolo_save="yolo_settings._save",
        unet_encoder="unet_settings._encoder",
        unet_input_size="unet_settings._input_size",
        deeplab_backbone="deeplab_settings._backbone",
        deeplab_output_stride="deeplab_settings._output_stride",
        unet_threshold="unet_settings._threshold",
        deeplab_threshold="deeplab_settings._threshold",
    )

    def _after_apply(self, state):
        main = self.main
        main.on_model_changed(main.model_combo.currentIndex())
        paths = {
            "U-Net": "unet_model_path",
            "DeepLabV3+": "deeplab_model_path",
            "YOLO-seg": "yolo_model_path",
            "Custom ONNX": "onnx_model_path",
        }
        path = state.get(paths.get(state["model"], ""), "")
        if path and Path(path).is_file():
            main.load_model_from_widget(state["model"])
        main.schedule_update()
