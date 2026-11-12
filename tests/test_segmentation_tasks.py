"""Numerical and GUI regressions for shared background tasks, without real weights."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = """
import threading
from pathlib import Path
from unittest.mock import patch
import tempfile
import cv2,numpy as np
from PyQt5.QtCore import QTimer,QElapsedTimer
from PyQt5.QtTest import QTest,QSignalSpy
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
from main import ImagePlayer
from ui.application import exec_window
from utils_new.segmentation import operations
from utils_new.segmentation.result import ResultStatus
app=QApplication([]);gui_thread=threading.get_ident()
def until(condition,timeout=10000):
    timer=QElapsedTimer();timer.start()
    while not condition() and timer.elapsed()<timeout:QTest.qWait(10)
    assert condition(),'condition timed out'
"""


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI/runtime suite"
)
class SegmentationTaskGuiTests(unittest.TestCase):
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
                [sys.executable, "-c", HEADER + code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_interactive_worker_gui_heartbeat_stale_frame_and_close(self):
        self.run_gui("""
w=ImagePlayer([('interactive','Interactive')]);w.show();owner=w._tab_widgets[0]
entered=threading.Event();release=threading.Event();exited=threading.Event();threads=[]
def work(method,image,gray,rect,fg,bg,params,cancel):
    threads.append(threading.get_ident());entered.set();release.wait(5);exited.set()
    mask=np.zeros(gray.shape,np.uint8);mask[10:30,10:30]=255;return mask
with tempfile.TemporaryDirectory() as directory,patch.object(operations,'interactive',side_effect=work):
    paths=[]
    for i,shape in enumerate([(80,160),(96,80)]):
        path=Path(directory)/(str(i)+'.png');cv2.imwrite(str(path),np.full((*shape,3),47,np.uint8));paths.append(str(path))
    owner._load_images(paths);owner.update_timer.stop();owner.rect=(5,5,35,35)
    owner.run_segmentation();until(entered.is_set)
    assert owner.jobs.busy and not owner.controls_panel.isEnabled()
    heartbeat=[];QTimer.singleShot(0,lambda:heartbeat.append(True));until(lambda:bool(heartbeat))
    assert all(thread!=gui_thread for thread in threads)
    owner.next_image();owner.update_timer.stop();release.set();until(lambda:not owner.jobs.busy)
    QTest.qWait(80)
    assert owner.current_index==1 and owner.mask is None and owner.result_state.status==ResultStatus.NOT_RUN
    assert owner.controls_panel.isEnabled() and not owner.save_button.isEnabled()
    entered.clear();release.clear();exited.clear();owner.rect=(5,5,35,35)
    owner.run_segmentation();until(entered.is_set)
    settled=QSignalSpy(owner.jobs.settled);owner.close();assert owner.jobs.closed
    release.set();until(exited.is_set);QTest.qWait(30)
    assert len(settled)==0 and owner.mask is None
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")

    def test_dl_failure_retry_and_model_load_survives_image_reload(self):
        self.run_gui("""
w=ImagePlayer([('deep_learning','DL')]);w.show();owner=w._tab_widgets[0]
class Model:
    metadata={}
    fail=False
    calls=[]
    def predict(self,image,**parameters):
        self.calls.append((threading.get_ident(),parameters.copy()))
        if self.fail:raise RuntimeError('native prediction failed')
        return np.zeros(image.shape[:2],np.uint8)
model=Model()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'frame.png';cv2.imwrite(str(path),np.zeros((80,160,3),np.uint8))
    owner._load_images([str(path)]);owner.update_timer.stop();owner.segmentor=model
    owner.yolo_settings._conf.setValue(.37);owner.run_segmentation()
    until(lambda:not owner.jobs.busy and owner.result_state.status==ResultStatus.READY)
    assert model.calls[-1][0]!=gui_thread and model.calls[-1][1]['conf']==.37
    assert owner.save_button.isEnabled() and owner.progress_bar.value()==100
    count=len(model.calls);owner.on_global_settings_changed({});assert len(model.calls)==count
    model.fail=True;owner.prediction_mask=None
    with patch.object(QMessageBox,'critical') as error:
        owner.run_segmentation();until(lambda:not owner.jobs.busy)
        error.assert_called_once()
    assert owner.result_state.status==ResultStatus.ERROR and owner.prediction_mask is None
    assert not owner.save_button.isEnabled() and owner.controls_panel.isEnabled()
    owner.display_current_image();QTest.qWait(80);assert len(model.calls)==count+1
    model.fail=False;owner.yolo_settings._conf.setValue(.42)
    until(lambda:not owner.jobs.busy and owner.result_state.status==ResultStatus.READY)
    weights=Path(directory)/'stub.pt';weights.write_bytes(b'stub')
    owner.yolo_settings._model_path.setText(str(weights));owner.update_timer.stop()
    entered=threading.Event();release=threading.Event();loaded=Model()
    def load(name,path,device,cancel):
        assert threading.get_ident()!=gui_thread;entered.set();release.wait(5);return loaded
    with patch.object(operations,'load_dl',side_effect=load):
        owner.load_model_from_widget('YOLO-seg');until(entered.is_set)
        owner._load_images([str(path)]);owner.update_timer.stop()
        release.set();until(lambda:owner.segmentor is loaded and not owner.jobs.busy and owner.result_state.status==ResultStatus.READY)
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")

    def test_real_ml_fit_predict_normalization_and_model_roundtrip(self):
        self.run_gui("""
w=ImagePlayer([('traditional_ml','ML')]);w.show();owner=w._tab_widgets[0]
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'frame.png';image=np.full((40,60,3),220,np.uint8);image[10:30,15:45]=20
    cv2.imwrite(str(path),image);owner._load_images([str(path)])
    owner.model_combo.setCurrentText('Decision Tree');owner.color_feature.setChecked(True)
    owner.texture_feature.setChecked(False);owner.spatial_feature.setChecked(False)
    owner.use_superpixels.setChecked(False);owner.normalize_features.setChecked(True)
    owner.update_timer.stop();owner.train_model()
    until(lambda:owner.model is not None and not owner.jobs.busy and owner.result_state.status==ResultStatus.READY,15000)
    assert owner.model.__class__.__name__=='DecisionTreeClassifier' and 'scaler' in owner.model_metadata
    expected=owner.prediction_mask.copy();assert expected[20,30]==255 and expected[0,0]==0
    model_path=Path(directory)/'tree.joblib'
    with patch.object(QFileDialog,'getSaveFileName',return_value=(str(model_path),'')),patch.object(QMessageBox,'information'):
        owner.save_model()
    assert model_path.is_file()
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(model_path),'')):
        owner.load_model()
    until(lambda:not owner.jobs.busy and owner.result_state.status==ResultStatus.READY,15000)
    np.testing.assert_array_equal(owner.prediction_mask,expected)
    assert owner.model_metadata['normalize_features'] and owner.normalize_features.isChecked()
    owner.clear_images()
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(model_path),'')),patch.object(QMessageBox,'warning') as warning:
        owner.load_model();until(lambda:not owner.jobs.busy and owner.model is not None)
        warning.assert_not_called()
    assert owner.result_state.status==ResultStatus.NO_IMAGE
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")

    def test_real_kmeans_superpixels_and_cluster_choice_reuses_prediction(self):
        self.run_gui("""
w=ImagePlayer([('traditional_ml','ML')]);w.show();owner=w._tab_widgets[0]
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'frame.png';image=np.full((60,80,3),220,np.uint8);image[15:45,20:60]=20
    cv2.imwrite(str(path),image);owner._load_images([str(path)])
    owner.model_combo.setCurrentText('K-Means Clustering');owner.kmeans_clusters.setValue(2)
    owner.color_feature.setChecked(True);owner.texture_feature.setChecked(False);owner.spatial_feature.setChecked(False)
    owner.use_superpixels.setChecked(True);owner.superpixel_size.setValue(10);owner.update_timer.stop()
    owner.train_model()
    until(lambda:not owner.jobs.busy and owner.result_state.status==ResultStatus.READY,15000)
    assert owner.cluster_combo.count()==2 and owner.model_metadata['use_superpixels']
    first=owner.prediction_mask.copy();assert first.any() and not first.all()
    with patch.object(operations,'predict_ml',wraps=operations.predict_ml) as predict:
        owner.cluster_combo.setCurrentIndex(1-owner.cluster_combo.currentIndex());owner.display_current_image();owner.update_timer.stop()
        predict.assert_not_called()
    np.testing.assert_array_equal(first.astype(np.int16)+owner.prediction_mask,255)
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
""")


class NumericalOperationTests(unittest.TestCase):
    def test_cancelled_preparation_does_not_start_feature_extraction(self):
        from concurrent.futures import CancelledError
        from threading import Event
        from unittest.mock import patch

        import numpy as np

        from utils_new.segmentation import operations

        cancel = Event()
        cancel.set()
        with patch.object(operations, "features") as extract:
            with self.assertRaises(CancelledError):
                operations.prepare_training(
                    [np.zeros((10, 10, 3), np.uint8)],
                    [np.zeros((10, 10), np.uint8)],
                    {},
                    cancel,
                )
            extract.assert_not_called()

    def test_interactive_dispatch_preserves_backend_parameters(self):
        from threading import Event
        from unittest.mock import patch

        import numpy as np

        from utils_new.segmentation import operations
        from utils_ops import interactive_ops

        image = np.zeros((20, 30, 3), np.uint8)
        gray = np.zeros((20, 30), np.uint8)
        params = dict(
            superpixel_size=20,
            compactness=5,
            sigma=2,
            color_sigma=7,
            lambda_val=0.6,
            sigma_color=11,
        )
        with patch.object(
            interactive_ops, "supercut_segmentation", return_value=gray
        ) as compute:
            operations.interactive(
                "SuperCut", image, gray, (1, 1, 10, 10), gray, gray, params, Event()
            )
            self.assertNotIn("color_sigma", compute.call_args.kwargs)
            self.assertEqual(compute.call_args.kwargs["sigma"], 2)

    def test_integer_class_ids_are_not_truncated_before_binary_union(self):
        from threading import Event
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        from utils_new.segmentation import operations
        from utils_new.segmentation.result import binary_mask

        image = np.zeros((2, 3, 3), np.uint8)
        gray = np.zeros((2, 3), np.uint8)
        labels = np.array([0, 256, 257, 0, 1, 2], np.int64)
        model = SimpleNamespace(predict=lambda features: labels)
        with patch.object(
            operations, "features", return_value=(np.zeros((6, 1)), None)
        ):
            result = operations.predict_ml(
                image, gray, model, None, {"use_superpixels": False}, False, Event()
            )
        np.testing.assert_array_equal(result, labels.reshape(2, 3))
        np.testing.assert_array_equal(
            binary_mask(result, gray.shape), [[0, 255, 255], [0, 255, 255]]
        )
