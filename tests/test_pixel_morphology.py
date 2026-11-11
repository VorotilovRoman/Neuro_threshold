"""Pixel units, current preset schemas and live method pipelines."""

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils_new.segmentation.presets import make_preset, validate_preset
from utils_new.settings_store import load_settings, normalize_settings, save_settings

ROOT = Path(__file__).resolve().parents[1]
REMOVED = {
    f"utils.{name}"
    for name in (
        "DatasetGenerator",
        "annotation_converter",
        "aug_utils",
        "augmentation_base",
        "augmentation_random",
        "coco_export",
        "coco_io",
        "dataset_validation",
        "unet_export",
        "unet_io",
        "yaml_io",
        "settings",
        "presets",
        "smart_view",
    )
}
REMOVED |= {"utils_new.smart_view_new", "utils_new.dataset_review_view"}


class PixelPresetTests(unittest.TestCase):
    def test_current_schema_rejects_relative_ambiguous_or_incomplete_morphology(self):
        state = make_preset(
            "threshold", method="Simple Threshold", params={"threshold": 128}
        )
        state["morphology"]["operations"].reverse()
        state["morphology"]["operations"][0]["kernel_px"] = 13
        validate_preset(state, "threshold")
        mutations = [
            lambda s: s.update(close_factor=0.02),
            lambda s: s.update(schema_version=1),
            lambda s: s["morphology"].update(units="percent"),
            lambda s: s["morphology"]["operations"].pop(),
            lambda s: s["morphology"]["operations"][0].update(kernel_px=4.5),
            lambda s: s["morphology"]["operations"][0].update(kernel_px=5000),
            lambda s: s["morphology"]["operations"][0].update(operation="close"),
        ]
        for change in mutations:
            invalid = copy.deepcopy(state)
            change(invalid)
            with self.assertRaises((ValueError, TypeError)):
                validate_preset(invalid, "threshold")
        with self.assertRaises(ValueError):
            validate_preset(state, "gradient")
        for path in (ROOT / "settings/presets").glob("*.json"):
            if path.name == "presets_aeb.json":
                continue
            for preset in json.loads(path.read_text()).values():
                validate_preset(preset)

    def test_settings_require_current_schema_and_atomic_failure_keeps_file(self):
        defaults = {
            "schema_version": 1,
            "font_size_px": 11,
            "line_thickness_px": 1,
            "colors": {"annotation": [1, 2, 3]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_settings(path, defaults, defaults)
            self.assertEqual(load_settings(path, defaults), defaults)
            before = path.read_bytes()
            for invalid in (
                {"font_size_mode": "large"},
                {**defaults, "font_size_px": 4},
                {**defaults, "colors": []},
                {**defaults, "font_size_px": True},
            ):
                with self.assertRaises(ValueError):
                    save_settings(path, invalid, defaults)
                self.assertEqual(path.read_bytes(), before)
            value = normalize_settings(
                {**defaults, "font_size_mode": "large"}, defaults
            )
            self.assertNotIn("font_size_mode", value)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_removed_utilities_have_no_code_or_registry_consumers(self):
        for module in REMOVED:
            self.assertFalse((ROOT / (module.replace(".", "/") + ".py")).exists())
        for path in ROOT.rglob("*.py"):
            if path.relative_to(ROOT).parts[0] in ("tests", "Docs", ".git"):
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [
                        (node.module or "") + "." + n.name for n in node.names
                    ]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = [node.value]
                else:
                    continue
                self.assertFalse(REMOVED.intersection(names), f"{path}:{node.lineno}")

    def test_ensemble_executes_pixel_erosion_from_shared_threshold_presets(self):
        from utils.ensemble_utils import get_detections_from_preset

        gray = np.full((80, 120), 255, np.uint8)
        gray[20:60, 30:90] = 0
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        preset = make_preset(
            "threshold",
            method="Simple Threshold",
            params={"threshold": 128},
            annotation_mode='detect',
        )
        initial = get_detections_from_preset(image, gray, preset)
        preset["morphology"]["operations"][2]["kernel_px"] = 13
        eroded = get_detections_from_preset(image, gray, preset)
        self.assertEqual(len(initial), 1)
        self.assertEqual(len(eroded), 1)
        self.assertAlmostEqual(eroded[0][3], initial[0][3] - 12 / 120)
        self.assertAlmostEqual(eroded[0][4], initial[0][4] - 12 / 80)


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI/runtime suite"
)
class PixelMorphologyGuiTests(unittest.TestCase):
    def run_gui(self, code):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                NO_ALBUMENTATIONS_UPDATE="1",
                XDG_CACHE_HOME=directory,
                XDG_CONFIG_HOME=directory,
                MPLCONFIGDIR=directory,
                YOLO_CONFIG_DIR=directory,
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_five_pipelines_apply_exact_pixels_and_roundtrip_four_ordered_operations(
        self,
    ):
        self.run_gui("""
from pathlib import Path
import tempfile,json,importlib
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
from main import ImagePlayer
from ui.morphology_ui import MorphologyWidget
from utils_new.morphology import apply_morphology
app=QApplication([])
keys=['threshold','gradient','interactive','traditional_ml','deep_learning']
w=ImagePlayer([(key,key) for key in keys])
mask=np.zeros((80,160),np.uint8);mask[15:65,30:130]=255;mask[35:40,70:75]=0;mask[4:6,6:8]=255
codes={'close':cv2.MORPH_CLOSE,'open':cv2.MORPH_OPEN,'erode':cv2.MORPH_ERODE,'dilate':cv2.MORPH_DILATE}
def pixels(view):
    image=view.pixmap_item.pixmap().toImage().convertToFormat(QImage.Format_Grayscale8)
    pointer=image.bits();pointer.setsize(image.byteCount())
    return np.frombuffer(pointer,np.uint8).reshape(image.height(),image.bytesPerLine())[:,:image.width()].copy()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'frame.png';cv2.imwrite(str(path),cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR))
    for key,owner in zip(keys,w._tab_widgets):
        panel=owner.morphology_widget;assert isinstance(panel,MorphologyWidget)
        assert not hasattr(owner,'close_kernel_slider')
        owner._load_images([str(path)]);owner.invert_checkbox.setChecked(False)
        module=importlib.import_module('tabs.'+key+'_methods')
        if key=='threshold':owner.method_combo.setCurrentText('Simple Threshold')
        for shape in ['Rectangle','Ellipse','Cross']:
            panel.set_kernel_shape(shape)
            shape_code={'Rectangle':cv2.MORPH_RECT,'Ellipse':cv2.MORPH_ELLIPSE,'Cross':cv2.MORPH_CROSS}[shape]
            for operation in codes:
                for step in codes:panel.set_operation_size(step,13 if step==operation else 0)
                owner.mask=mask.copy();owner.prediction_mask=mask.copy()
                with patch.object(module,'apply_threshold_method',return_value=(cv2.bitwise_not(mask),128),create=True),patch.object(module,'apply_gradient_method',return_value=mask,create=True):
                    owner.display_current_image()
                expected=cv2.morphologyEx(mask,codes[operation],cv2.getStructuringElement(shape_code,(13,13)))
                np.testing.assert_array_equal(pixels(owner.morph_view),expected,err_msg=key+':'+shape+':'+operation)
        panel.set_kernel_shape('Ellipse')
        panel.set_order(['dilate','close','erode','open'])
        for operation,value in zip(panel.operation_order(),[7,13,3,5]):panel.set_operation_size(operation,value)
        state=panel.get_settings();owner.preset_manager.presets_file=str(Path(directory)/(key+'.json'))
        owner.preset_manager.add_preset('pixels')
        saved=json.loads(Path(owner.preset_manager.presets_file).read_text())['pixels']
        assert saved['morphology']==state and saved['schema_version']==3
        panel.set_order(['close','open','erode','dilate']);panel.set_operation_size('close',2)
        owner.preset_manager.load_presets();assert owner.preset_manager.apply_preset('pixels')
        assert panel.get_settings()==state
        # 13 px stays 13 px after a frame-size change, not a percentage of the frame.
        larger=Path(directory)/'larger.png';cv2.imwrite(str(larger),np.zeros((180,360,3),np.uint8))
        owner._load_images([str(larger)]);assert panel.get_settings()==state
        if key in ['threshold','gradient']:
            assert not owner.auto_button.isEnabled() and owner.auto_button.toolTip()
        cache=getattr(owner,'_lazy_cache',None)
        if cache:cache.shutdown()
from PyQt5.QtCore import QTimer
from ui.application import exec_window
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")

    def test_ensemble_preview_keeps_nested_params_and_applies_pixel_morphology(self):
        self.run_gui("""
import cv2,numpy as np
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from main import ImagePlayer
from ui.application import exec_window
from utils_new.segmentation.presets import make_preset
import tabs.demo_ensemble_methods as module
app=QApplication([])
w=ImagePlayer([('yolo_demo','Ensemble')]);owner=w._tab_widgets[0]
gray=np.full((80,120),255,np.uint8);gray[20:60,30:90]=0
owner.current_image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);owner.current_gray=gray
preset=make_preset('threshold',method='Simple Threshold',params={'threshold':127},annotation_mode='detect')
preset['morphology']['operations'][2]['kernel_px']=13
owner.preset_manager.presets={'pixels':preset};owner.active_preset_name='pixels'
assert owner.get_current_preset_settings()['params']=={'threshold':127}
with patch.object(module,'apply_morphology',wraps=module.apply_morphology) as morphology,patch.object(owner,'log') as log:
    owner.apply_preset_to_image()
    morphology.assert_called_once()
    assert morphology.call_args.args[1]==preset['morphology']['operations']
    assert not any('Ошибка' in str(call) for call in log.call_args_list)
assert len(owner.current_objects)==1
box=owner.current_objects[0]
assert abs(box[4]-48/120)<1e-8 and abs(box[5]-28/80)<1e-8
assert not owner.preset_result_view.pixmap_item.pixmap().isNull()
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")

    def test_preset_application_restores_threshold_parameters_and_rejects_old_state_before_mutation(
        self,
    ):
        self.run_gui("""
import tempfile,json,copy
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QSignalSpy
from main import ImagePlayer
app=QApplication([])
w=ImagePlayer([(key,key) for key in ['threshold','interactive','traditional_ml','deep_learning']])
t,interactive,ml,dl=w._tab_widgets
with tempfile.TemporaryDirectory() as directory:
    for owner in w._tab_widgets:owner.preset_manager.presets_file=str(Path(directory)/(type(owner).__name__+'.json'))
    t.method_combo.setCurrentText('Niblack');t.niblack_win.setValue(31);t.niblack_k.setValue(37)
    t.morphology_widget.set_operation_size('erode',13)
    t.preset_manager.add_preset('exact');before=t.preset_manager.get_current_settings()
    t.method_combo.setCurrentText('Simple Threshold')
    t.preset_manager.apply_preset('exact')
    assert t.method_combo.currentText()=='Niblack' and t.niblack_win.value()==31 and t.niblack_k.value()==37
    assert t.preset_manager.get_current_settings()==before
    blocked=t.morphology_widget.blockSignals(True)
    t.preset_manager.apply_preset('exact');assert t.morphology_widget.signalsBlocked()
    t.morphology_widget.blockSignals(blocked)
    invalid=copy.deepcopy(before);invalid['morphology']['operations'][0]['kernel_px']=1.5
    t.preset_manager.presets['invalid']=invalid
    try:t.preset_manager.apply_preset('invalid');raise AssertionError('invalid preset accepted')
    except ValueError:pass
    assert t.preset_manager.get_current_settings()==before
    from PyQt5.QtWidgets import QMessageBox
    t.preset_combo.addItem('invalid')
    with patch.object(QMessageBox,'warning') as warning:
        t.on_preset_activated(t.preset_combo.findText('invalid'));warning.assert_called_once()
    assert t.preset_manager.get_current_settings()==before
    path=Path(directory)/'old.json';path.write_text(json.dumps({'old':{'close_factor':.1}}));raw=path.read_bytes()
    t.preset_manager.presets_file=str(path);t.preset_manager.load_presets()
    assert t.preset_manager.load_error and path.read_bytes()==raw
    assert 'default' in t.preset_manager.presets
    interactive.params_widget.grabcut_iters.setValue(9)
    interactive.preset_manager.add_preset('params');interactive.params_widget.grabcut_iters.setValue(1)
    interactive.preset_manager.apply_preset('params');assert interactive.params_widget.grabcut_iters.value()==9
    ml.use_superpixels.setChecked(True);ml.superpixel_size.setValue(70)
    ml.preset_manager.add_preset('features');ml.use_superpixels.setChecked(False);ml.superpixel_size.setValue(20)
    ml.preset_manager.apply_preset('features');assert ml.use_superpixels.isChecked() and ml.superpixel_size.value()==70 and ml.superpixel_size.isEnabled()
    dl.unet_settings._threshold.setValue(.73);dl.preset_manager.add_preset('network')
    dl.unet_settings._threshold.setValue(.2);dl.preset_manager.apply_preset('network')
    assert dl.unet_settings._threshold.value()==.73
from PyQt5.QtCore import QTimer
from ui.application import exec_window
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")


if __name__ == "__main__":
    unittest.main()
