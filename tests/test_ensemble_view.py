"""Ensemble display migration: clean pixels, vector annotations and read-only input."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class EnsembleWorkerDisplayTests(unittest.TestCase):
    def test_yolo_worker_emits_unpainted_pixels_and_unchanged_detection_values(self):
        from tabs.demo_ensemble_methods import YoloInferThread

        image = np.full((80, 160, 3), 47, np.uint8)
        box = SimpleNamespace(cls=[3], conf=[0.91], xyxy=np.array([[40, 20, 120, 60]]))
        result = SimpleNamespace(
            boxes=[box],
            names={3: "car"},
            plot=Mock(side_effect=AssertionError("Raster overlay is obsolete")),
        )
        model = Mock(return_value=[result])
        fake_ultralytics = SimpleNamespace(YOLO=Mock(return_value=model))
        old_hook = sys.excepthook
        try:
            worker = YoloInferThread("stub.pt", image, 0.25, 0.45, 640, "cpu", False)
            outputs, finished, labels = [], [], []
            worker.class_names_signal.connect(labels.append)
            worker.result_signal.connect(lambda *args: outputs.append(args))
            worker.finished_signal.connect(lambda *args: finished.append(args))
            with (
                patch.dict(sys.modules, ultralytics=fake_ultralytics),
                patch("runtime_policy.configure_ultralytics"),
                patch("tabs.demo_ensemble_methods.ULTRALYTICS_AVAILABLE", True),
            ):
                worker.run()
            self.assertEqual(finished[0][0], True)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(labels, [{3: "car"}])
            np.testing.assert_array_equal(outputs[0][0], image)
            self.assertIsNot(outputs[0][0], image)
            self.assertEqual(
                outputs[0][1], ["3 0.500000 0.500000 0.500000 0.500000 0.910000"]
            )
            result.plot.assert_not_called()
        finally:
            sys.excepthook = old_hook


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI/runtime suite"
)
class EnsembleViewGuiTests(unittest.TestCase):
    def run_gui(self, code):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                NO_ALBUMENTATIONS_UPDATE="1",
                XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory,
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

    def test_all_four_panels_use_vectors_and_update_geometry_clear_and_export(self):
        self.run_gui("""
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication,QFileDialog
from main import ImagePlayer
from ui.annotation_review_view import DatasetReviewGraphicsView
from ui.application import exec_window
from ui.image_conversion import numpy_to_qpixmap
from utils_new.segmentation.presets import make_preset
import tabs.demo_ensemble_methods as module
app=QApplication([]);window=ImagePlayer([('yolo_demo','Ensemble')]);window.show()
owner=window._tab_widgets[0];app.processEvents()
assert all(type(view) is DatasetReviewGraphicsView for view in owner.view_widgets.values())
assert 'utils.smart_view' not in __import__('sys').modules
def pixels(view):
    image=view.pixmap_item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
    pointer=image.bits();pointer.setsize(image.byteCount())
    return np.frombuffer(pointer,np.uint8).reshape(image.height(),image.bytesPerLine())[:,:image.width()*3].reshape(image.height(),image.width(),3)[:,:,::-1].copy()
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);path=root/'frame.png'
    gray=np.full((80,160),255,np.uint8);gray[20:60,40:120]=37
    original=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR);cv2.imwrite(str(path),original)
    path.with_suffix('.txt').write_text('2 0.5 0.5 0.5 0.5\\n')
    owner.load_images_from_paths([str(path)]);owner.update_timer.stop()
    assert owner.preview_view.annotations==[('detect',2,.5,.5,.5,.5)]
    np.testing.assert_array_equal(pixels(owner.preview_view),original)
    owner.set_yolo_class_names({3:'car'});assert owner.model_result_view._annotation_label(3)=='car'
    owner.display_yolo_result(original,['3 0.5 0.5 0.5 0.5 0.91'])
    assert owner.model_result_view.annotations==[('detect',3,.5,.5,.5,.5)]
    assert 'conf=0.910' in owner.model_analysis_text.toPlainText()
    np.testing.assert_array_equal(pixels(owner.model_result_view),original)
    preset=make_preset('threshold',method='Simple Threshold',params={'threshold':127},annotation_mode='detect')
    owner.preset_manager.presets={'pixels':preset};owner.active_preset_name='pixels'
    owner.apply_preset_to_image()
    assert owner.preset_result_view.annotations==owner.current_objects and len(owner.current_objects)==1
    np.testing.assert_array_equal(pixels(owner.preset_result_view),original)
    owner.on_ensemble_result([(4,.5,.5,.5,.5,.93)])
    assert owner.ensemble_result_view.annotations==[('detect',4,.5,.5,.5,.5)]
    assert 'conf=0.930' in owner.ensemble_analysis_text.toPlainText()
    np.testing.assert_array_equal(pixels(owner.ensemble_result_view),original)
    for name,view in owner.view_widgets.items():
        assert (view.img_width,view.img_height)==(160,80)
        target=root/(name+'.png')
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'')):
            view.save_source_image()
        np.testing.assert_array_equal(cv2.imread(str(target)),original)
        # Display limits must not change the working coordinate system.
        large=np.zeros((77,211,3),np.uint8)
        with patch.object(module,'numpy_to_qpixmap',side_effect=lambda data:numpy_to_qpixmap(data,max_side=64)):
            owner._show_image_in_view(view,large,[('detect',0,.5,.5,.4,.4)])
        bounds=view.pixmap_item.mapRectToScene(view.pixmap_item.boundingRect())
        assert (view.img_width,view.img_height)==(211,77)
        assert abs(bounds.width()-211)<1e-6 and abs(bounds.height()-77)<1e-6
    second=root/'second.png';cv2.imwrite(str(second),np.zeros((180,320,3),np.uint8))
    owner.load_images_from_paths([str(path),str(second)]);owner.update_timer.stop()
    owner.next_image();owner.update_timer.stop()
    assert (owner.preview_view.img_width,owner.preview_view.img_height)==(320,180)
    for name in ['model_result_view','preset_result_view','ensemble_result_view']:
        view=getattr(owner,name);assert view.pixmap_item.pixmap().isNull() and not view.annotations
        assert (view.img_width,view.img_height)==(0,0)
    owner.clear_images()
    for view in owner.view_widgets.values():
        assert view.pixmap_item.pixmap().isNull() and not view.annotations
        assert (view.img_width,view.img_height)==(0,0)
QTimer.singleShot(10,window.close);assert exec_window(window,app)==0
""")

    def test_read_only_pan_zoom_reset_and_selection_do_not_edit_boxes(self):
        self.run_gui("""
import numpy as np
from PyQt5.QtCore import Qt,QPoint,QPointF,QTimer
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from tabs.demo_ensemble_methods import YoloDemoWindow
from ui.annotation_review_view import DatasetReviewGraphicsView
from ui.application import exec_window
app=QApplication([]);window=YoloDemoWindow();window.show();app.processEvents()
image=np.full((120,240,3),90,np.uint8);annotation=('detect',2,.5,.5,.4,.4)
fits={}
for view in window.view_widgets.values():
    window._show_image_in_view(view,image,[annotation]);view.reset_view();app.processEvents()
    fit=view.transform().m11();fits[view]=fit;center=view.viewport().rect().center()
    event=QWheelEvent(QPointF(center),QPointF(view.viewport().mapToGlobal(center)),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
    QApplication.sendEvent(view.viewport(),event)
    assert view.transform().m11()>fit
    QTest.mouseClick(view.viewport(),Qt.MiddleButton);assert abs(view.transform().m11()-fit)<1e-6
    view.scale(4,4);app.processEvents()
    center=view.viewport().rect().center();before=view.mapToScene(center)
    global_start=view.viewport().mapToGlobal(center)
    start=window.mapFromGlobal(global_start);end=start+QPoint(35,20)
    QTest.mousePress(window.windowHandle(),Qt.LeftButton,Qt.NoModifier,start)
    QTest.mouseMove(window.windowHandle(),end,30)
    QTest.mouseRelease(window.windowHandle(),Qt.LeftButton,Qt.NoModifier,end)
    app.processEvents();after=view.mapToScene(center)
    assert abs(before.x()-after.x())>1 or abs(before.y()-after.y())>1
    assert view.annotations==[annotation] and not view.drawing_mode and not view.edit_mode
    view.scale(2,2)
QTest.mouseClick(window.reset_zoom_all_btn,Qt.LeftButton);app.processEvents()
for view in window.view_widgets.values():
    assert abs(view.transform().m11()-fits[view])<1e-6 and view.annotations==[annotation]
QTimer.singleShot(10,window.close);assert exec_window(window,app)==0
""")


if __name__ == "__main__":
    unittest.main()
