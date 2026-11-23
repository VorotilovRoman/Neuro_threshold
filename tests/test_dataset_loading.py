"""Responsive Qt loading, bounded decoding, cancellation and stale-result regressions."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from utils_new.import_export.catalog import DatasetImportJob
from utils_new.import_export.images import write_image
from utils_new.dataset.pairs import PairInspectionJob

ROOT = Path(__file__).resolve().parents[1]


class DiscoveryTests(unittest.TestCase):
    def test_folder_discovery_reads_names_only_and_keeps_images_without_labels(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            for name in ("one", "two", "name.with.dots 2026.10.03 02.05.33"):
                (images / (name + ".png")).touch()
            (labels / "one.txt").write_text("0 .5 .5 .5 .5\n")
            (labels / "name.with.dots 2026.10.03 02.05.33.TXT").touch()
            with patch(
                "utils_new.import_export.images.read_image",
                side_effect=AssertionError("eager image decoding"),
            ):
                result = DatasetImportJob(
                    "folder",
                    [str(images)],
                    dict(dataset_type=0, annotations=str(labels)),
                ).run()
            self.assertEqual(len(result["catalog"].pairs), 3)
            self.assertEqual(sum(p[1] is None for p in result["catalog"].pairs), 1)

    def test_inspection_snapshot_and_cancel_do_not_drop_unlabelled_images(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "one.png"
            write_image(source, np.zeros((20, 40, 3), np.uint8))
            label = source.with_suffix(".txt")
            label.write_text("3 .5 .5 .5 .5\n")
            result = PairInspectionJob(
                [(str(source), str(label)), (str(source), None)]
            ).run()
            self.assertEqual(result["valid"], 2)
            self.assertEqual(result["classes"], [3])
            self.assertEqual(result["rows"][1]["tag"], "unlabelled")
            self.assertEqual(sum(result["colors"].values()), 2)
            cancelled = PairInspectionJob([(str(source), str(label))])
            cancelled.cancel()
            self.assertTrue(cancelled.run()["cancelled"])


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "Qt integration opt-in"
)
class BackgroundLoadingTests(unittest.TestCase):
    def gui(self, body):
        header = """
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from threading import Event,get_ident
from time import monotonic
import numpy as np
from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5.QtCore import QTimer,QThread
from PyQt5.QtTest import QTest
import sys
sys.path.insert(0,str(Path.cwd()/'tests'))
from dataset_test_support import wait_dataset
from utils_new.import_export.images import write_image
app=QApplication([])
def until(condition):
 deadline=monotonic()+8
 while not condition() and monotonic()<deadline:QTest.qWait(5)
 assert condition(),'Timed out waiting for dataset state'
def fixture(root,name='one'):
 image=root/(name+'.png');write_image(image,np.zeros((24,40,3),np.uint8))
 label=image.with_suffix('.txt');label.write_text('0 .5 .5 .5 .5\\n')
 return str(image),str(label)
"""
        result = subprocess.run(
            [sys.executable, "-c", header + body],
            cwd=ROOT,
            env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
            capture_output=True,
            text=True,
            timeout=35,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_slow_metadata_does_not_block_list_navigation_timer_cancel_or_retry(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from utils_new.dataset import pairs
owner=DatasetPreparationWindow();owner.show();app.processEvents()
started,release=Event(),Event();threads=[];ticks=[]
timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(5)
original=pairs.build_pair_display
def slow(*args,**kwargs):
 threads.append(get_ident());started.set();release.wait(5);return original(*args,**kwargs)
with TemporaryDirectory() as temporary,patch.object(pairs,'build_pair_display',slow):
 pair=fixture(Path(temporary))
 owner._apply_loaded_pairs([pair],0,'auto',class_names_from_yaml={0:'object'})
 until(started.is_set)
 assert owner.file_list.count()==1 and owner.btn_folder.isEnabled()
 assert not owner._dataset_ready() and owner._dataset_tasks.rows['inspect'][0].isVisible()
 QTest.qWait(100);assert len(ticks)>=3
 assert threads[0]!=get_ident()
 owner._dataset_tasks.cancel('inspect');release.set();wait_dataset(owner)
 assert not owner._dataset_ready()
 retry=owner._dataset_tasks.rows['inspect'][3];assert retry.text()=='Повторить'
 retry.click();wait_dataset(owner);assert owner._dataset_ready()
 assert owner._pair_display_cache[tuple(pair)]['is_valid']
owner.close();timer.stop()
""")

    def test_stale_delivery_and_close_are_safe_even_if_io_ignores_cancel(self):
        self.gui("""
from ui.dataset_loading import dataset_tasks
owner=QMainWindow();owner.log=lambda text:None;owner.show()
started,release,done=Event(),Event(),Event();results=[];delivery=[]
class Slow:
 def __init__(self,value):self.value=value
 def cancel(self):pass
 def run(self):
  if self.value=='old':started.set();release.wait(5);done.set()
  return {'value':self.value}
tasks=dataset_tasks(owner)
def receive(value):results.append(value['value']);delivery.append(QThread.currentThread() is app.thread())
tasks.start('load',Slow,'old',label='Old',finished=receive);until(started.is_set)
tasks.start('load',Slow,'new',label='New',finished=receive);wait_dataset(owner)
assert results==['new'] and all(delivery)
release.set();until(done.is_set);QTest.qWait(30);assert results==['new']
release.clear();started.clear()
tasks.start('load',Slow,'old',label='Close',finished=receive);until(started.is_set)
# QMainWindow has no product close hook; destroy exercises the controller's guard.
owner.close();tasks.close();owner.deleteLater();app.processEvents()
release.set();QTest.qWait(40);assert results==['new']
""")

    def test_lazy_editor_decode_is_off_thread_bounded_and_close_never_waits_for_io(
        self,
    ):
        self.gui("""
import tabs.layout_dataset as editor
owner=editor.Labeler();owner.show();started,release,done=Event(),Event(),Event()
original=editor.load_one_image_item;threads=[]
def slow(path,*args,**kwargs):
 threads.append(get_ident());started.set();release.wait(5)
 try:return original(path,*args,**kwargs)
 finally:done.set()
with TemporaryDirectory() as temporary,patch.object(editor,'load_one_image_item',slow):
 root=Path(temporary);paths=[fixture(root,str(i))[0] for i in range(25)]
 before=monotonic();owner._load_images(paths);assert monotonic()-before<.5
 until(started.is_set);QTest.qWait(40)
 assert owner.image_paths==paths and not any(value is not None for value in owner.all_annotations)
 assert len(threads)<=2 and all(value!=get_ident() for value in threads)
 before=monotonic();owner.close();assert monotonic()-before<.5
 release.set();until(done.is_set)
""")

    def test_pending_navigation_never_creates_an_empty_annotation_or_edits_old_frame(
        self,
    ):
        self.gui("""
import tabs.layout_dataset as editor
owner=editor.Labeler();owner.show()
started,release=Event(),Event();original=editor.load_one_image_item
with TemporaryDirectory() as temporary:
 root=Path(temporary);first=fixture(root,'first');second=fixture(root,'second')
 Path(second[1]).write_text('4 .5 .5 .3 .3\\n')
 def slow(path,*args,**kwargs):
  if path==first[0]:started.set();release.wait(5)
  return original(path,*args,**kwargs)
 with patch.object(editor,'load_one_image_item',slow):
  owner._load_images([first[0],second[0]]);until(started.is_set)
  assert not owner.save_button.isEnabled() and not owner.image_view.isEnabled()
  owner.next_image();until(lambda:owner._current_frame_ready())
  assert owner.current_index==1 and owner.current_annotations[0][1]==4
  assert owner.all_annotations[0] is None
  assert owner.save_button.isEnabled()
  release.set();wait_dataset(owner)
  assert owner.current_index==1 and owner.current_annotations[0][1]==4
  owner.prev_image();wait_dataset(owner)
  assert owner.current_annotations[0][1]==0
 owner.close()
""")

    def test_a_failed_frame_can_be_retried_without_reloading_the_dataset(self):
        self.gui("""
from tabs.layout_dataset import Labeler
owner=Labeler();owner.show()
with TemporaryDirectory() as temporary:
 image=Path(temporary)/'frame.png';image.write_bytes(b'broken')
 owner._load_images([str(image)]);wait_dataset(owner)
 assert not owner.save_button.isEnabled()
 retry=owner._dataset_tasks.rows['frame'][3]
 assert retry.text()=='Повторить' and retry.isEnabled()
 write_image(image,np.zeros((24,40,3),np.uint8))
 retry.click();wait_dataset(owner)
 assert owner._current_frame_ready() and owner.save_button.isEnabled()
 owner.close()
""")

    def test_old_overlap_warnings_are_cleared_before_a_new_inspection_finishes(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from utils_new.dataset.pairs import PairInspectionJob
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary);image,old_label=fixture(root)
 Path(old_label).write_text('0 .5 .5 .5 .5\\n0 .5 .5 .5 .5\\n')
 new_label=root/'new.txt';new_label.write_text('0 .5 .5 .2 .2\\n')
 owner._apply_loaded_pairs([(image,old_label)],0,'auto',class_names_from_yaml={0:'Object'});wait_dataset(owner)
 review=owner._overlap_review_panel
 assert review._records[image]
 started,release=Event(),Event();original=PairInspectionJob.run
 def slow(job):started.set();release.wait(5);return original(job)
 with patch.object(PairInspectionJob,'run',slow):
  owner._start_pair_inspection();until(started.is_set)
  assert not review._records and not review._pairs
  release.set();wait_dataset(owner);assert review._records[image]
  release.clear();started.clear()
  owner._apply_loaded_pairs([(image,str(new_label))],0,'auto',class_names_from_yaml={0:'Object'})
  until(started.is_set)
  assert not review._records and not review._pairs
  assert owner.file_list.item(0).icon().isNull()
  release.set();wait_dataset(owner)
 assert not review._records[image]
 owner.close()
""")

    def test_filter_snapshot_finishes_in_background_and_rejects_previous_dataset(self):
        self.gui("""
from tabs.dataset_analysis import DatasetAnalysisWindow
from utils_new.dataset_audit.analysis import DatasetFilterJob
owner=DatasetAnalysisWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary);first=fixture(root,'first');second=fixture(root,'second')
 owner._apply_loaded_pairs([first],0,'auto',class_names_from_yaml={0:'Object'});wait_dataset(owner)
 original=DatasetFilterJob.run;started,release=Event(),Event();threads=[]
 def slow(job):
  threads.append(get_ident())
  if job.pairs[0][0]==first[0]:started.set();release.wait(5)
  return original(job)
 with patch.object(DatasetFilterJob,'run',slow):
  owner.apply_filters_and_update_metrics();until(started.is_set)
  assert owner.btn_folder.isEnabled()
  owner._apply_loaded_pairs([second],0,'auto',class_names_from_yaml={0:'Object'});wait_dataset(owner)
  release.set();QTest.qWait(50)
  assert owner.filtered_pairs==[second]
  assert threads and all(thread!=get_ident() for thread in threads)
  assert owner._pair_class_cache[tuple(second)]==0
owner.close()
""")
