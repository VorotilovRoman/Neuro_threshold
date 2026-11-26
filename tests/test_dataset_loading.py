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
    def test_summary_separates_images_files_objects_background_and_errors(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = []
            for name, text in (('object', '2 .5 .5 .5 .5\n'), ('background', ''),
                               ('unlabelled', None), ('invalid', 'bad row')):
                image = root / (name + '.png')
                write_image(image, np.zeros((20, 40, 3), np.uint8))
                label = image.with_suffix('.txt')
                if text is not None:
                    label.write_text(text)
                pairs.append((str(image), str(label) if text is not None else None))
            broken = root / 'broken.png'
            broken.write_bytes(b'not an image')
            label = broken.with_suffix('.txt')
            label.write_text('0 .5 .5 .5 .5\n')
            pairs.append((str(broken), str(label)))
            result = PairInspectionJob(pairs).run()
            self.assertEqual(result['summary'], dict(images=5, checked=5, labelled=4, unlabelled=1,
                                                     errors=2, with_objects=1, background=1, classes=1))
            self.assertEqual(result['valid'], 3)

    def test_zero_mask_is_background_and_not_an_object_class(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = []
            for index in range(3):
                image, mask = root / f'image{index}.png', root / f'mask{index}.png'
                write_image(image, np.zeros((20, 40, 3), np.uint8))
                pixels = np.zeros((20, 40), np.uint8)
                if index == 0:
                    pixels[5:10, 10:20] = 4
                write_image(mask, pixels)
                pairs.append((str(image), str(mask) if index < 2 else None))
            result = PairInspectionJob(pairs, dataset_type=1, include_empty=False).run()
            self.assertEqual(result['summary'], dict(images=3, checked=3, labelled=2, unlabelled=1,
                                                     errors=0, with_objects=1, background=1, classes=1))
            self.assertEqual(result['rows'][1]['tag'], 'empty')
            self.assertIn('image + mask', result['rows'][1]['text'])

    def test_folder_labels_follow_relative_subfolders_and_explicit_split(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = root / 'images', root / 'labels'
            for split in ('train', 'val'):
                (images / split).mkdir(parents=True)
                (labels / split).mkdir(parents=True)
                (images / split / 'frame.with.dots.PNG').touch()
                (labels / split / 'frame.with.dots.TXT').write_text('0 .5 .5 .5 .5\n')
            for split in ('train', 'val'):
                result = DatasetImportJob('folder', [str(images / split)], dict(dataset_type=0, annotations=str(labels))).run()
                self.assertEqual(len(result['catalog'].pairs), 1)
                image, label = result['catalog'].pairs[0]
                self.assertIsNotNone(label)
                self.assertEqual(Path(image).parent.name, Path(label).parent.name)
            train = DatasetImportJob('folder', [str(images / 'train')], dict(dataset_type=0, annotations=str(labels))).run()
            self.assertEqual(Path(train['catalog'].pairs[0][1]).parent.name, 'train')

    def test_nested_label_fallback_rejects_ambiguous_names(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = root / 'pictures', root / 'labels'
            images.mkdir()
            (images / 'one.png').touch()
            for name in ('first', 'second'):
                (labels / name).mkdir(parents=True)
                (labels / name / 'one.txt').touch()
            with self.assertRaisesRegex(ValueError, 'Несколько меток'):
                DatasetImportJob('folder', [str(images)], dict(dataset_type=0, annotations=str(labels))).run()
            (labels / 'second' / 'one.txt').unlink()
            logs = []
            job = DatasetImportJob('folder', [str(images)], dict(dataset_type=0, annotations=str(labels)))
            job.log = logs.append
            result = job.run()
            self.assertTrue(result['catalog'].pairs[0][1].endswith('first/one.txt'))
            self.assertTrue(any('с разметкой: 1' in line for line in logs))

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

    def test_preparation_folder_then_txt_updates_rows_and_classes(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from PyQt5.QtCore import Qt
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary);images=root/'images';labels=root/'labels'
 (images/'train').mkdir(parents=True);(labels/'train').mkdir(parents=True)
 write_image(images/'train'/'part.with.dots.png',np.zeros((24,40,3),np.uint8))
 (labels/'train'/'part.with.dots.TXT').write_text('3 .5 .5 .5 .5\\n')
 with patch('ui.dataset_loader.source_dialogs.get_directory',return_value=str(images/'train')):
  owner.btn_folder.click();wait_dataset(owner)
 assert owner.file_list.count()==1 and '[Без разметки]' in owner.file_list.item(0).text()
 owner.input_format_combo.setCurrentIndex(owner.input_format_combo.findData('detect'));wait_dataset(owner)
 with patch('ui.dataset_loader.source_dialogs.get_directory',return_value=str(labels)):
  owner.btn_labels.click();wait_dataset(owner)
 assert '[detect]' in owner.file_list.item(0).text(),owner.file_list.item(0).text()
 assert owner.file_list.item(0).data(Qt.UserRole+1)[1].endswith('train/part.with.dots.TXT')
 assert owner.original_ids==[3],owner.original_ids
 assert owner.input_format=='detect'
 assert owner._dataset_ready()
 assert 'с разметкой: 1' in owner.log_text.toPlainText()
owner.close()
""")

    def test_preparation_summary_for_200_images_without_labels_and_cancelled_recheck(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from utils_new.dataset.pairs import PairInspectionJob
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary)
 for index in range(200):write_image(root/(str(index)+'.png'),np.zeros((12,20,3),np.uint8))
 owner.images_folder=str(root);owner.scan_pairs();wait_dataset(owner)
 widget=owner.dataset_summary
 assert {key:label.text() for key,label in widget.values.items()}==dict(
  images='200',labelled='0',unlabelled='200',with_objects='0',background='0',errors='0')
 assert widget.status.text()=='Проверено: 200/200'
 assert owner._overlap_review_panel.isHidden()
 assert '8 бит PNG: 200' in widget.properties_label.text()
 started,release=Event(),Event();original=PairInspectionJob.run
 def slow(job):started.set();release.wait(5);return original(job)
 with patch.object(PairInspectionJob,'run',slow):
  owner._start_pair_inspection();until(started.is_set)
  assert widget.values['labelled'].text()=='—'
  assert 'Проверка:' in widget.status.text()
  owner._dataset_tasks.cancel('inspect');release.set();wait_dataset(owner)
 assert 'не завершена' in widget.status.text()
 assert widget.values['errors'].text()=='—'
owner.close()
""")

    def test_preparation_color_conversion_refreshes_inspection_and_display(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from utils_new.import_export.images import _read_with_fallback
import cv2
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary);image=root/'color.png'
 pixels=np.zeros((24,40,3),np.uint8);pixels[:,:,1]=220;write_image(image,pixels)
 owner.images_folder=str(root);owner.scan_pairs();wait_dataset(owner)
 assert owner.file_list.item(0).data(Qt.UserRole+3)=='color'
 previous=owner._inspection_revision
 with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):
  owner.btn_convert_gray.click()
 assert not owner._dataset_ready() and not owner.btn_convert_rgb.isEnabled()
 wait_dataset(owner)
 assert owner._inspection_revision>previous
 assert _read_with_fallback(str(image),cv2.IMREAD_UNCHANGED).ndim==2
 assert owner.file_list.item(0).data(Qt.UserRole+3)=='gray_real',owner.file_list.item(0).text()
 assert owner.color_type_stats.get('gray_real')==1
 with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):
  owner.btn_convert_rgb.click();wait_dataset(owner)
 assert _read_with_fallback(str(image),cv2.IMREAD_UNCHANGED).shape[2]==3
 assert owner.file_list.item(0).data(Qt.UserRole+3)=='gray_visual'
 assert owner._dataset_ready()
owner.close()
""")

    def test_preparation_excludes_unlabelled_by_default_and_counts_selected_sources(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from PyQt5.QtWidgets import QMessageBox
import json
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 root=Path(temporary)
 first=fixture(root,'object');empty=fixture(root,'empty');Path(empty[1]).write_text('')
 unknown=fixture(root,'unknown');Path(unknown[1]).unlink();unknown=(unknown[0],None)
 owner._apply_loaded_pairs([first,empty,unknown],0,'auto',class_names_from_yaml={0:'object'});wait_dataset(owner)
 assert not owner.include_unlabelled_images.isChecked()
 owner.train_slider.setValue(100);owner.val_slider.setValue(0);owner.test_slider.setValue(0)
 owner.resize_keep.setChecked(True)
 for include_empty in (False,True):
  for include_unlabelled in (False,True):
   owner.include_empty_annotations.setChecked(include_empty)
   owner.include_unlabelled_images.setChecked(include_unlabelled)
   expected={'object'}|({'empty'} if include_empty else set())|({'unknown'} if include_unlabelled else set())
   assert owner.train_count_label.text()==str(len(expected)),owner.train_count_label.text()
   owner.output_folder=str(root/('out_'+str(include_empty)+'_'+str(include_unlabelled)))
   owner.update_generate_button_state();assert owner.generate_btn.isEnabled()
   with patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as error,patch.object(QMessageBox,'warning') as warning:
    owner.on_generate()
    assert owner.generator_thread is not None
    assert owner.generator_thread.engine.include_unlabelled==include_unlabelled
    assert owner.generator_thread.wait(15000);app.processEvents()
    error.assert_not_called();warning.assert_not_called()
   actual={path.stem for path in (Path(owner.output_folder)/'images/train').iterdir()}
   assert actual==expected,(actual,expected)
 owner._apply_loaded_pairs([unknown],0,'auto');wait_dataset(owner)
 owner.include_empty_annotations.setChecked(False);owner.include_unlabelled_images.setChecked(False)
 assert owner.train_count_label.text()=='0' and not owner.generate_btn.isEnabled()
 owner.include_unlabelled_images.setChecked(True)
 assert owner.train_count_label.text()=='1' and owner.generate_btn.isEnabled()
 owner._apply_loaded_pairs([unknown],2,'auto');wait_dataset(owner)
 owner.include_unlabelled_images.setChecked(False)
 assert owner.train_count_label.text()=='1' and not owner.include_unlabelled_images.isEnabled()
owner.close()
""")

    def test_preparation_png_conversion_refreshes_paths_bit_depth_and_labels(self):
        self.gui("""
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
owner=DatasetPreparationWindow();owner.show()
with TemporaryDirectory() as temporary:
 for extension in ('tiff','png'):
  root=Path(temporary)/extension;root.mkdir();image=root/('image.'+extension);label=root/'image.txt'
  write_image(image,np.full((24,40),32000,np.uint16));label.write_text('2 .5 .5 .5 .5\\n')
  owner.images_folder=str(root);owner.labels_folder=str(root);owner.scan_pairs();wait_dataset(owner)
  assert owner.file_list.item(0).data(Qt.UserRole+4)=='uint16'
  def choose(dialog):
   next(button for button in dialog.buttons() if button.text().startswith('Только')).click()
  with patch.object(QMessageBox,'exec_',choose):
   owner.btn_convert_8bit_png.click();wait_dataset(owner)
  pair=owner.file_list.item(0).data(Qt.UserRole+1)
  assert pair[0]==str(root/'image.png'),pair
  assert pair[1]==str(label) and label.is_file()
  assert image.exists()==(extension=='png')
  assert owner.file_list.item(0).data(Qt.UserRole+4)=='uint8_png'
  assert '[detect]' in owner.file_list.item(0).text()
  assert owner.original_ids==[2] and owner._dataset_ready()
owner.close()
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
