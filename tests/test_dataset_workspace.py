"""Overlap geometry, background quality and real Qt button regressions."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils_new.annotations.processing import find_strong_overlaps
from utils_new.dataset_audit.annotation_review import AnnotationReviewJob
from utils_new.dataset_audit.quality import QualityJob
from utils_new.dataset_audit.scans import ScanJob

ROOT = Path(__file__).resolve().parents[1]


class DatasetWorkspaceTests(unittest.TestCase):
    def test_histogram_handles_area_rounding_noise_without_losing_counts(self):
        from utils_new.dataset_audit.analysis import histogram_edges

        values = [16.0, np.nextafter(16.0, 17.0)] * 3
        edges = histogram_edges(values)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertEqual(int(np.histogram(values, bins=edges)[0].sum()), len(values))
        self.assertEqual(
            int(
                np.histogram(
                    [0.0, 10.0, 100.0], bins=histogram_edges([0.0, 10.0, 100.0])
                )[0].sum()
            ),
            3,
        )

    def test_overlap_metric_threshold_classes_and_non_square_polygons(self):
        annotations = [
            ("detect", 0, 0.5, 0.5, 0.8, 0.8),
            ("detect", 1, 0.5, 0.5, 0.2, 0.2),
        ]
        self.assertEqual(set(find_strong_overlaps(annotations, 200, 100)), {0, 1})
        self.assertFalse(find_strong_overlaps(annotations, 200, 100, metric="iou"))
        self.assertFalse(find_strong_overlaps(annotations, 200, 100, same_class=True))
        annotations = [
            ("segment", 0, [0.1, 0.1, 0.8, 0.1, 0.8, 0.6, 0.1, 0.6]),
            ("obb", 0, [0.1, 0.1, 0.8, 0.1, 0.8, 0.6, 0.1, 0.6]),
        ]
        warnings = find_strong_overlaps(
            annotations, 800, 200, threshold=1, metric="iou", same_class=True
        )
        self.assertAlmostEqual(warnings[0][0][1], 1)
        shift = [("detect", 0, 0.5, 0.5, 0.4, 0.4), ("detect", 0, 0.6, 0.5, 0.4, 0.4)]
        self.assertTrue(find_strong_overlaps(shift, 200, 100, threshold=0.7))
        self.assertFalse(find_strong_overlaps(shift, 200, 100, threshold=0.8))
        for threshold in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                find_strong_overlaps(shift, 200, 100, threshold)

    def test_quality_rows_errors_thresholds_cancel_and_basic_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dark = root / "тёмный.png"
            pattern = root / "pattern.png"
            broken = root / "bad.png"
            cv2.imwrite(str(dark), np.zeros((40, 80), np.uint8))
            checker = ((np.indices((40, 80)).sum(axis=0) % 2) * 255).astype(np.uint8)
            cv2.imwrite(str(pattern), checker)
            broken.write_text("broken")
            pairs = [(str(path), None) for path in (dark, pattern, broken)]
            result = QualityJob(pairs).run()
            self.assertEqual(result["checked"], 2)
            self.assertEqual(result["unreadable"], 1)
            self.assertIn("Тёмный", result["rows"][0]["issues"])
            self.assertGreater(
                result["rows"][1]["sharpness"], result["rows"][0]["sharpness"]
            )
            self.assertTrue(result["rows"][2]["error"])
            cancelled = QualityJob(pairs)
            cancelled.cancel()
            self.assertTrue(cancelled.run()["cancelled"])
            with self.assertRaises(ValueError):
                QualityJob(pairs, {"dark": 240, "bright": 20})
            result = ScanJob("basic", pairs[:2], dict(dataset_type=2)).run()
            self.assertIn("quality", result)
            self.assertIn("duplicates", result)
            self.assertNotIn("heatmap", result)

    def test_annotation_scan_uses_memory_and_preserves_valid_inferred_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            label = image.with_suffix(".txt")
            cv2.imwrite(str(image), np.zeros((40, 80), np.uint8))
            label.write_text("0 .1 .1 .7 .1 .5 .5 .1 .6\n0 .1 .1 .7 .1 .5 .5 .1 .6\n")
            result = AnnotationReviewJob([(str(image), str(label))]).run()
            self.assertFalse(result["errors"])
            self.assertEqual(len(result["records"][str(image)]), 2)
            result = AnnotationReviewJob(
                [(str(image), str(label))], memory={str(image): ([], (80, 40))}
            ).run()
            self.assertFalse(result["records"][str(image)])
            job = AnnotationReviewJob([(str(image), str(label))])
            job.cancel()
            self.assertTrue(job.run()["cancelled"])


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in Qt tests")
class DatasetWorkspaceGuiTests(unittest.TestCase):
    def test_editor_and_viewer_share_rows_indices_and_configurable_palette(self):
        self.gui(r'''
import tempfile
from pathlib import Path
import cv2,numpy as np
from copy import deepcopy
from PyQt5.QtWidgets import QApplication,QListWidget
from PyQt5.QtCore import Qt,QCoreApplication,QEvent
from PyQt5.QtTest import QSignalSpy
from main import ImagePlayer
from ui.annotation_review import update_annotation_list
from ui.settings_service import settings
app=QApplication([]);window=ImagePlayer([('labeler','Editor'),('test','Viewer')])
palette=deepcopy(settings.current['dataset_class_palette'])
try:
 window.show();app.processEvents()
 with tempfile.TemporaryDirectory() as directory:
  image=Path(directory)/'source.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8))
  image.with_suffix('.txt').write_text('1 .5 .5 .4 .4\n2 .1 .1 .3 .1 .3 .3 .1 .3\n3 .5 .1 .8 .1 .7 .3\n')
  settings.current['dataset_class_palette'][1]=[12,34,56]
  rows=[]
  for owner in window._tab_widgets:
   owner._load_images([str(image)]);objects=owner.overlap_review.objects
   assert objects.count()==3
   rows.append([objects.item(i).text().split('  [OVERLAP:')[0].removeprefix('⚠ ') for i in range(3)])
   assert [objects.item(i).data(Qt.UserRole) for i in range(3)]==[0,1,2]
   view=getattr(owner,'image_view',None) or owner.viewer
   assert view._annotation_color(1).getRgb()[:3]==(56,34,12)
  assert rows[0]==rows[1],rows
  objects=window._tab_widgets[0].object_list;objects.setCurrentRow(1)
  changes=QSignalSpy(objects.currentRowChanged)
  objects.blockSignals(True)
  update_annotation_list(objects,[('detect',1,.5,.5,.4,.4),('detect',)],80,40)
  assert objects.signalsBlocked() and len(changes)==0
  assert objects.count()==2 and 'некорректная' in objects.item(1).text()
  objects.blockSignals(False)
  update_annotation_list(objects,[],80,40)
  assert not objects.signalsBlocked() and not objects.count() and len(changes)==0
finally:
 settings.current['dataset_class_palette']=palette
 window.close();window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
''')

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
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=50,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_quality_button_and_basic_scan_no_torch_and_errors_are_guarded(self):
        self.gui("""
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.dataset_analysis import DatasetAnalysisWindow
app=QApplication([]);w=DatasetAnalysisWindow();w.show()
def until(condition):
 timer=QElapsedTimer();timer.start()
 while not condition() and timer.elapsed()<10000:QTest.qWait(10)
 assert condition(),w.analysis_status.text()
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'sample.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));label=image.with_suffix('.txt');label.write_text('0 .5 .5 .5 .5\\n')
 w._apply_loaded_pairs([(str(image),str(label))],0,'auto',class_names_from_yaml={0:'Object'})
 assert w.toolbox.count()==9
 assert "Ошибка операции анализа" not in w.log_text.toPlainText()
 w.btn_run_quality.click();until(lambda:w.scans.worker is None)
 assert w._last_quality and w._last_quality['checked']==1
 assert w.quality_table.rowCount()==1
 assert 'torch' not in sys.modules, 'basic analysis imported Torch'
 with patch.object(w,'_draw_quality_plots',side_effect=ValueError('plot failure')):
  w.btn_run_quality.click();until(lambda:w.scans.worker is None)
  assert 'plot failure' in w.analysis_status.text()
 w.btn_run_basic.click();until(lambda:w.scans.worker is None)
 assert w._last_quality and w._last_dup is not None and w._last_heatmap
 # Decorated filter/export actions must tolerate clicked(bool) as well.
 w.btn_apply_filters.click();assert not w._scan_busy
 w.filter_min_area.setValue(999999)
 report=w._collect_report_data();assert report['filter_params']['min_area']==0
 assert report['general']['Фильтры изменены после применения']
 w.btn_apply_filters.click();assert not w._get_analysis_pairs()
 w.scope_combo.setCurrentIndex(0);assert len(w._get_analysis_pairs())==1
w.close();app.processEvents()
""")

    def test_overlaps_in_both_editors_follow_threshold_edits_and_image_index(self):
        self.gui("""
import tempfile
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from tabs.viewing_dataset import ViewingDataset
from tabs.layout_dataset import Labeler
app=QApplication([])
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'image.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));image.with_suffix('.txt').write_text('0 .5 .5 .4 .4\\n0 .6 .5 .4 .4\\n')
 for cls in (ViewingDataset,Labeler):
  owner=cls();owner.show();owner._load_images([str(image)]);review=owner.overlap_review
  review.threshold.setValue(70)
  for _ in range(30):QTest.qWait(10)
  assert all('OVERLAP' in review.objects.item(i).text() for i in range(2))
  assert not review.images.item(0).icon().isNull()
  review.threshold.setValue(90)
  for _ in range(30):QTest.qWait(10)
  assert all('OVERLAP' not in review.objects.item(i).text() for i in range(2))
  assert review.images.item(0).icon().isNull()
  review.threshold.setValue(70);owner.current_annotations=owner.current_annotations[:1];owner._revalidate_current()
  assert not review.current_warnings
  assert review.images.item(0).icon().isNull()
  owner.close();app.processEvents()
""")

    def test_preparation_file_warnings_and_pair_dialog_objects(self):
        self.gui("""
import tempfile
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from ui.pair_validation_dialog_new import PairValidationDialog
app=QApplication([]);owner=DatasetPreparationWindow();owner.show()
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'image.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));label=image.with_suffix('.txt');label.write_text('0 .5 .5 .4 .4\\n0 .51 .5 .4 .4\\n')
 pairs=[(str(image),str(label))]
 owner._apply_loaded_pairs(pairs,0,'auto',class_names_from_yaml={0:'Object'})
 for _ in range(35):QTest.qWait(10)
 assert not owner.file_list.item(0).icon().isNull()
 dialog=PairValidationDialog(owner,pairs[0],0,owner.validator)
 dialog.show();app.processEvents()
 assert dialog.overlap_review.objects.count()==2
 assert 'OVERLAP' in dialog.overlap_review.objects.item(0).text()
 dialog.close();owner.close();app.processEvents()
""")

    def test_scan_invalidation_cancellation_and_close_ignore_late_delivery(self):
        self.gui("""
import tempfile,threading
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from tabs.dataset_analysis import DatasetAnalysisWindow
from ui.dataset_audit.tasks import ScanWorker
app=QApplication([])
entered,release,done=threading.Event(),threading.Event(),threading.Event()
class Job:
 def __init__(self,*args,**kwargs):pass
 def cancel(self):pass
 def run(self):
  entered.set();release.wait(3);done.set();return {'rows':[],'checked':999,'unreadable':0,'candidates':0,'resolution':None,'brightness':None}
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'frame.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));image.with_suffix('.txt').write_text('0 .5 .5 .5 .5\\n')
 for action in ('invalidate','close'):
  entered.clear();release.clear();done.clear()
  owner=DatasetAnalysisWindow();owner.show();owner._apply_loaded_pairs([(str(image),str(image.with_suffix('.txt')))],0,'auto',class_names_from_yaml={0:'Object'})
  with patch.object(ScanWorker,'job_type',Job):
   owner.btn_run_quality.click();assert entered.wait(2)
   worker=owner.scans.worker
   if action=='invalidate':owner._invalidate_analysis_results()
   else:owner.close()
   assert worker.closed
   release.set();assert done.wait(2)
   for _ in range(10):QTest.qWait(10)
   assert owner._last_quality is None
   assert owner.scans.worker is None
  owner.close();app.processEvents()
""")
