"""Regression coverage for the final utils consumers and real background UI actions."""

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from utils_new.annotations.boxes import ensemble_detections
from utils_new.segmentation.presets import make_preset
from utils_new.segmentation.threshold import get_detections_from_preset

ROOT = Path(__file__).resolve().parents[1]


class EnsembleAndSearchTests(unittest.TestCase):
    def test_classes_zero_weights_and_weighted_nms(self):
        a = (2, 0.45, 0.5, 0.4, 0.4, 0.9)
        b = (2, 0.5, 0.5, 0.4, 0.4, 0.8)
        other = (7, 0.45, 0.5, 0.4, 0.4, 0.7)
        result = ensemble_detections([[a, other], [b]], [0.1, 1.0], method="nms")
        self.assertEqual({row[0] for row in result}, {2})  # class 7 weighted below .1
        self.assertAlmostEqual(result[0][1], 0.5)
        result = ensemble_detections([[a, other], [b]], [1.0, 0.0], method="nms")
        self.assertEqual({row[0] for row in result}, {2, 7})
        self.assertEqual(ensemble_detections([[a]], [0]), [])
        result = ensemble_detections([[a], [other]], [1, 1], method="wbf")
        self.assertEqual({row[0] for row in result}, {2, 7})

    def test_wbf_confidence_coordinates_and_source_votes(self):
        a = (3, 0.4, 0.5, 0.4, 0.4, 0.8)
        b = (3, 0.5, 0.5, 0.4, 0.4, 0.4)
        rows = ensemble_detections([[a, a, a], [b]], [2, 1], iou_threshold=0.5)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][1], (0.4 * 1.6 + 0.5 * 0.4) / 2)
        self.assertAlmostEqual(rows[0][-1], 2 / 3)
        self.assertEqual(
            rows, ensemble_detections([[a, a, a], [b]], [20, 10], iou_threshold=0.5)
        )
        absent = ensemble_detections([[a], []], [2, 1])
        self.assertAlmostEqual(absent[0][-1], 1.6 / 3)

    def test_containment_requires_full_box_and_parameters_are_validated(self):
        a = (0, 0.4, 0.5, 0.4, 0.4, 0.8)
        b = (0, 0.6, 0.5, 0.4, 0.4, 0.8)
        self.assertEqual(
            len(
                ensemble_detections(
                    [[a], [b]], [1, 1], iou_threshold=0.9, use_containment=True
                )
            ),
            2,
        )
        nested = (0, 0.4, 0.5, 0.1, 0.1, 0.8)
        self.assertEqual(
            len(
                ensemble_detections(
                    [[a], [nested]], [1, 1], iou_threshold=0.9, use_containment=True
                )
            ),
            1,
        )
        for weights, options in (
            ([-1], {}),
            ([float("nan")], {}),
            ([], {}),
            ([1], {"method": "unknown"}),
            ([1], {"conf_threshold": 1.1}),
        ):
            with self.assertRaises(ValueError):
                ensemble_detections([[a]], weights, **options)

    def test_preset_score_class_and_polygon_boxes(self):
        gray = np.full((60, 120), 255, np.uint8)
        gray[10:45, 30:80] = 0
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for mode in ("detect", "obb", "segment"):
            preset = make_preset(
                "threshold",
                method="Simple Threshold",
                params={"threshold": 128},
                annotation_mode=mode,
            )
            rows = get_detections_from_preset(image, gray, preset, 0.43, class_id=8)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], 8)
            self.assertAlmostEqual(rows[0][-1], 0.43)
            self.assertGreater(rows[0][3], 0.3)

    def test_search_uses_original_pixel_grid_and_handles_constructor_failure(self):
        from tabs.yolo_find_img import ScanThread

        class Boxes:
            xyxy = np.array([[90.0, 90.0, 200.0, 200.0]])
            cls = np.array([5])
            conf = np.array([0.91])

            def __len__(self):
                return 1

        seen = []

        def predict(image, **options):
            seen.append(image.shape)
            return [SimpleNamespace(boxes=Boxes())]

        class Model:
            names = {5: "target"}
            __call__ = staticmethod(predict)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "снимок.png"
            cv2.imwrite(str(path), np.zeros((800, 1600), np.uint16))
            worker = ScanThread(
                directory, "stub.pt", 5, 0.25, 0.45, 640, "cpu", edge_margin=75
            )
            outputs = []
            worker.results_ready.connect(outputs.append)
            with (
                patch.dict(
                    sys.modules, ultralytics=SimpleNamespace(YOLO=lambda _: Model())
                ),
                patch("runtime_policy.configure_ultralytics"),
            ):
                worker.run()
            self.assertEqual(seen, [(800, 1600, 3)])
            self.assertEqual(outputs[0][0][0], str(path))
            self.assertEqual(worker.checked, 1)
            failed = ScanThread(directory, "bad.pt", 5, 0.25, 0.45, 640, "cpu")
            result = []
            failed.results_ready.connect(result.append)
            with (
                patch.dict(
                    sys.modules,
                    ultralytics=SimpleNamespace(
                        YOLO=lambda _: (_ for _ in ()).throw(ValueError("bad weights"))
                    ),
                ),
                patch("runtime_policy.configure_ultralytics"),
            ):
                failed.run()
            self.assertIn("bad weights", failed.error)
            self.assertEqual(result, [[]])

    def test_old_utilities_and_registry_entries_have_no_consumers(self):
        removed = {"utils.image_io", "utils.ensemble_utils", "utils_aeb.yolo_aeb_utils"}
        for name in removed:
            self.assertFalse((ROOT / (name.replace(".", "/") + ".py")).exists())
        for path in ROOT.rglob("*.py"):
            if path.relative_to(ROOT).parts[0] in ("tests", "docs", ".git"):
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    targets = [node.module]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    targets = [node.value]
                self.assertFalse(
                    removed.intersection(targets),
                    f"{path}:{getattr(node, 'lineno', 0)}",
                )


class AutoencoderTests(unittest.TestCase):
    def test_actual_architectures_shapes_and_explicit_deterministic_vae(self):
        import torch
        from utils_new.segmentation.autoencoder import build_ae_model_from_cfg, infer_autoencoder

        image = np.full((20, 60, 3), 73, np.uint8)
        for arch in ("conv", "conv_gn", "unet", "unet_v2"):
            for kind in ("ae", "vae"):
                with self.subTest(arch=arch, kind=kind):
                    cfg = dict(
                        arch=arch,
                        model_type=kind,
                        in_channels=1,
                        img_h=32,
                        img_w=48,
                        latent_dim=4,
                        latent_channels=4,
                        base=8,
                        vae_use_mean=True,
                    )
                    model = build_ae_model_from_cfg(cfg)
                    first = infer_autoencoder(image, model, cfg, torch.device("cpu"))
                    second = infer_autoencoder(image, model, cfg, torch.device("cpu"))
                    self.assertEqual(first["error"].shape, (32, 48))
                    self.assertTrue(np.isfinite(first["error"]).all())
                    np.testing.assert_array_equal(first["error"], second["error"])
        cfg["img_w"] = 31
        with self.assertRaises(ValueError):
            build_ae_model_from_cfg(cfg)

    def test_invalid_output_and_edge_channels_are_not_silently_compared(self):
        import torch
        from utils_new.segmentation.autoencoder import infer_autoencoder

        cfg = dict(in_channels=1, img_h=32, img_w=32, keep_aspect=True)

        class Bad(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((1, 2, 32, 32)), None

        with self.assertRaisesRegex(ValueError, "Выход AE/VAE"):
            infer_autoencoder(np.zeros((20, 40, 3), np.uint8), Bad(), cfg, "cpu")
        cfg.update(model_in_channels=2, use_edge_channel=True)

        class Good(torch.nn.Module):
            def forward(self, x):
                return x[:, :1], None

        value = infer_autoencoder(np.zeros((20, 40, 3), np.uint8), Good(), cfg, "cpu")
        self.assertEqual(value["input_shape"], (2, 32, 32))
        self.assertTrue(np.all(value["error"] == 0))


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in Qt tests")
class MigratedTabsGuiTests(unittest.TestCase):
    def gui(self, code):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory,
                MPLCONFIGDIR=directory,
                YOLO_CONFIG_DIR=directory,
                NO_ALBUMENTATIONS_UPDATE="1",
                PYTHONFAULTHANDLER="1",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code
                    + "\nfrom PyQt5.QtCore import QCoreApplication,QEvent\nfor widget in app.topLevelWidgets():\n widget.close();widget.deleteLater()\nQCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)\n",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=55,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_classical_ensemble_real_button_class_mapping_and_failed_yolo(self):
        self.gui("""
import tempfile,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.demo_ensemble_methods import YoloDemoWindow
from utils_new.segmentation.presets import make_preset
app=QApplication([]);w=YoloDemoWindow();w.show();w.auto_run_yolo=False
def until(condition):
 t=QElapsedTimer();t.start()
 while not condition() and t.elapsed()<8000:QTest.qWait(10)
 assert condition()
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'frame.png';image=np.full((80,160,3),255,np.uint8);image[20:60,40:120]=0;cv2.imwrite(str(path),image)
 w.load_images_from_paths([str(path)]);w.update_timer.stop()
 preset=make_preset('threshold',method='Simple Threshold',params={'threshold':128},annotation_mode='detect')
 w.preset_manager.presets={'unit':preset};w.update_ensemble_weights_ui()
 w.preset_weight_spins[0][1].setValue(1);w.preset_weight_spins[0][2].setValue(.6);w.preset_class_spins['unit'].setValue(7)
 w.yolo_weight_spin.setValue(0);w.run_ensemble_btn.click();until(lambda:w.ensemble_thread is None)
 assert w.ensemble_result_view.annotations[0][1]==7
 assert 'conf=0.600' in w.ensemble_analysis_text.toPlainText()
 assert w.run_ensemble_btn.isEnabled() and not w.stop_btn.isEnabled()
 # Rebuilding rows preserves user values and is safe through deferred Qt deletion.
 for _ in range(5):w.update_ensemble_weights_ui();app.processEvents()
 assert w.preset_class_spins['unit'].value()==7 and w.preset_weight_spins[0][2].value()==.6
 before=sys.excepthook
 with patch.dict(sys.modules,ultralytics=SimpleNamespace(YOLO=lambda _:(_ for _ in ()).throw(ValueError('bad weights')))),patch('runtime_policy.configure_ultralytics'):
  w.yolo_settings._model_path.setText(str(path));w.run_yolo_btn.click();until(lambda:not w.yolo_inference_running)
  assert w.run_ensemble_btn.isEnabled() and not w.ensemble_mode
 assert sys.excepthook is before
w.close();app.processEvents()
""")

    def test_search_failure_buttons_and_same_folder_copy(self):
        self.gui("""
import tempfile,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QListWidgetItem
from PyQt5.QtCore import Qt,QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.yolo_find_img import FindImagesWindow
app=QApplication([]);w=FindImagesWindow();w.show()
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'image.png';cv2.imwrite(str(path),np.zeros((30,60),np.uint8));weights=Path(directory)/'model.pt';weights.write_text('bad')
 w.current_root_dir=directory
 # The path starts a class-reading task too: patch before either background import.
 with patch.dict(sys.modules,ultralytics=SimpleNamespace(YOLO=lambda _:(_ for _ in ()).throw(ValueError('bad weights')))),patch('runtime_policy.configure_ultralytics'):
  w.yolo_settings._model_path.setText(str(weights))
  w.scan_btn.click();timer=QElapsedTimer();timer.start()
  while not w.scan_btn.isEnabled() and timer.elapsed()<8000:QTest.qWait(10)
  assert w.scan_btn.isEnabled() and not w.stop_btn.isEnabled()
  assert 'bad weights' in w.scan_status.text(),w.scan_status.text()
  while w._class_task and w._class_task.isRunning() and timer.elapsed()<8000:QTest.qWait(10)
  assert not w._class_task or not w._class_task.isRunning()
 item=QListWidgetItem('image');item.setData(Qt.UserRole,str(path));item.setCheckState(Qt.Checked);w.results_list.addItem(item)
 w.target_folder_edit.setText(directory)
 with patch.object(QMessageBox,'information'):w.copy_selected_files()
 assert not (Path(directory)/'image_1.png').exists()
w.close();app.processEvents();assert w._closing
""")

    def test_ae_real_apply_invalid_output_and_unreadable_next_frame(self):
        self.gui("""
import sys,tempfile
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.yolo_test_aeb import YoloInspectWindow
app=QApplication([]);w=YoloInspectWindow();w.show()
assert 'torch' not in sys.modules,'opening AE tab eagerly loaded Torch'
import torch
class Identity(torch.nn.Module):
 def forward(self,x):return x,None
class Bad(torch.nn.Module):
 def forward(self,x):return torch.zeros((1,5,32,48)),None
def until(condition):
 timer=QElapsedTimer();timer.start()
 while not condition() and timer.elapsed()<8000:QTest.qWait(10)
 assert condition(),w.processing_status.text()
with tempfile.TemporaryDirectory() as directory:
 good=Path(directory)/'good.png';cv2.imwrite(str(good),np.full((30,80,3),60,np.uint8));bad=Path(directory)/'bad.png';bad.write_text('bad')
 w._load_images([str(good),str(bad)]);w.update_timer.stop();w.ae_model=Identity();w.ae_cfg=dict(in_channels=3,img_h=32,img_w=48)
 w.btn_apply_ae.click();until(lambda:w._ae_task is None)
 assert w._ae_err_hw.shape==(32,48) and w._error_bgr is not None
 w.ae_model=Bad();w.btn_apply_ae.click();until(lambda:w._ae_task is None)
 assert 'Выход AE/VAE' in w.processing_status.text() and w._ae_err_hw is None
 assert w.btn_apply_ae.isEnabled()
 w.goto_image(2)
 assert w.original_image is None and w.displayed_image is None and w._error_bgr is None
 assert w.yolo_view.pixmap_item.pixmap().isNull() and not w.current_detections
 w.yolo_view.set_annotations([('detect',0,.5,.5,.2,.2)],80,30)
 w.clear_images();assert not w.yolo_view.annotations and w.yolo_view.img_width==0
 assert not w.btn_stop_processing.isEnabled()
w.close();app.processEvents()
""")

    def test_close_embedded_tabs_and_cancelled_late_model_results(self):
        self.gui("""
import tempfile,threading,time
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QElapsedTimer,QTimer
from PyQt5.QtTest import QTest
from main import ImagePlayer
app=QApplication([]);window=ImagePlayer([('yolo_demo','Ensemble'),('yolo_sort','Find'),('yolo_aeb','AE')]);window.show()
w=window._tab_widgets[2];entered,release,done=threading.Event(),threading.Event(),threading.Event()
class Model:
 names={0:'object'}
 def __call__(self,*args,**kwargs):
  entered.set();release.wait(4);done.set();return []
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'image.png';cv2.imwrite(str(path),np.zeros((40,80,3),np.uint8))
 w._load_images([str(path)]);w.update_timer.stop();w.model=Model();w._run_yolo_inference()
 timer=QElapsedTimer();timer.start()
 while not entered.is_set() and timer.elapsed()<2000:QTest.qWait(10)
 assert entered.is_set()
 beats=[];heartbeat=QTimer();heartbeat.timeout.connect(lambda:beats.append(1));heartbeat.start(5);QTest.qWait(60);assert len(beats)>2
 worker=w._yolo_task;window.close();assert worker.closed and w._closing
 release.set();assert done.wait(2)
 for _ in range(10):QTest.qWait(10)
 assert not w.current_detections
 assert all(tab._closing for tab in window._tab_widgets)
heartbeat.stop();app.processEvents()
""")

    def test_background_thumbnails_preserve_list_selection_and_sorting(self):
        self.gui("""
import tempfile
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.yolo_find_img import FindImagesWindow
app=QApplication([]);w=FindImagesWindow();w.show()
with tempfile.TemporaryDirectory() as directory:
 first=Path(directory)/'first.png';second=Path(directory)/'second.png'
 for path in (first,second):cv2.imwrite(str(path),np.full((50,500,3),50,np.uint8))
 w.on_file_done(str(first),.5,True);w.on_file_done(str(second),.9,True)
 w.results_list.item(0).setCheckState(Qt.Checked)
 assert w.get_selected_files()==[str(first)]
 timer=QElapsedTimer();timer.start()
 while len(w.thumbnail_widgets)<2 and timer.elapsed()<5000:QTest.qWait(10)
 assert len(w.thumbnail_widgets)==2
 assert w.thumbnail_widgets[str(first)][0].isChecked()
 w.sort_by_confidence();assert w.get_selected_files()==[str(first)]
 assert w.results_list.item(0).data(Qt.UserRole)==str(second)
 w.toggle_view_mode();assert w.get_selected_files()==[str(first)]
 w.close();app.processEvents()
""")
