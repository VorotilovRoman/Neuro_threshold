"""Reviewed sources: explicit labels, missing annotations, real dialogs and native YOLO."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import yaml

from utils_new.import_export.dataset_sources import (
    defaults,
    preview,
    relocate,
    write_manifest,
)
from utils_new.import_export.images import write_image
from utils_new.import_export.yolo_yaml import load_yolo_yaml
from utils_new.import_export.mask_yaml import load_unet_yaml, load_training_sources
from utils_new.import_export.ground_truth import load_ground_truth

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    for name in ("a", "b"):
        path = root / "images/train/nested" / (name + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_image(path, np.zeros((24, 32, 3), np.uint8))
        path.with_suffix(".txt").write_text("1 .5 .5 .5 .5\n")
    labels = root / "labels/train/nested"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 .5 .5 .25 .25\n")
    (labels / "extra.txt").write_text("0 .5 .5 .25 .25\n")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            dict(path=str(root), train="images/train", names=["selected", "old"])
        )
    )
    return data


class ImportPlans(unittest.TestCase):
    def test_voc_preview_and_saved_selection_survive_session_cleanup(self):
        import shutil
        from utils_new.import_export.dataset_sources import persist_source

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            approved = write_manifest(
                preview(defaults([source], "yolo"), "labels"), root / "temporary"
            )[0]
            persistent = persist_source(approved, root / "run/dataset_source")
            shutil.rmtree(root / "temporary")
            self.assertEqual(len(load_yolo_yaml(persistent)[0]), 2)
            images = root / "JPEGImages"
            masks = root / "SegmentationClass"
            listing = root / "ImageSets/Segmentation/train.txt"
            images.mkdir()
            masks.mkdir()
            listing.parent.mkdir(parents=True)
            write_image(images / "one.jpg", np.zeros((24, 32, 3), np.uint8))
            write_image(masks / "one.png", np.ones((24, 32), np.uint8))
            listing.write_text("one\n")
            voc = root / "voc.yaml"
            voc.write_text(
                yaml.safe_dump(
                    dict(path=str(root), train_txt=str(listing), num_classes=2)
                )
            )
            planned = preview(defaults([voc], "unet"), "declared")
            self.assertEqual(planned["summaries"][0]["matched"], 1)
            view = write_manifest(planned, root / "voc_view")[0]
            self.assertEqual(len(load_training_sources(view)["splits"]["train"]), 1)
            from utils_new.import_export.mask_yaml import write_yolo_txt_for_unet_pairs
            from utils_new.import_export.yolo_labels import load_annotations
            converted = write_yolo_txt_for_unet_pairs(load_unet_yaml(view)['pairs'], root / 'converted', class_names={})
            self.assertEqual(len(load_annotations(converted[0][1], 32, 24)), 1)

    def test_source_choice_missing_labels_and_originals_survive_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            spec = defaults([source], "yolo")
            selected = preview(spec, "labels")
            summary = selected["summaries"][0]
            self.assertEqual(
                (
                    summary["images"],
                    summary["matched"],
                    summary["missing"],
                    summary["files"],
                    summary["conflicts"],
                ),
                (2, 1, 1, 2, 1),
            )
            path = write_manifest(selected, root / "approved")[0]
            logs = []
            pairs = load_yolo_yaml(path, logs.append)[0]
            self.assertEqual(len(pairs), 2)
            self.assertTrue(pairs[0][1].endswith("labels/train/nested/a.txt"))
            self.assertIsNone(pairs[1][1])
            self.assertTrue(any("без разметки=1" in line for line in logs))
            truth = load_ground_truth(path)
            self.assertEqual(len(truth.samples), 1)
            self.assertEqual(truth.samples[0].annotations[0][1], 0)
            for name, content in before.items():
                self.assertEqual(Path(name).read_bytes(), content)
            beside = write_manifest(preview(spec, "beside"), root / "beside")[0]
            self.assertEqual(len([p for p in load_yolo_yaml(beside)[0] if p[1]]), 2)
            empty = write_manifest(preview(spec, "none"), root / "none")[0]
            self.assertTrue(all(label is None for _, label in load_yolo_yaml(empty)[0]))

    def test_root_other_paths_and_split_membership_are_explicit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            source = fixture(first)
            fixture(second)
            spec = relocate(defaults([source], "yolo"), second)
            plan = preview(spec, "labels")
            self.assertTrue(
                all(e["image"].startswith(str(second)) for e in plan["entries"])
            )
            custom = second / "custom/nested"
            custom.mkdir(parents=True)
            (custom / "b.txt").write_text("0 .5 .5 .2 .2\n")
            spec["rows"][0]["annotations"] = [str(custom.parent)]
            plan = preview(spec, "other")
            self.assertIsNone(plan["entries"][0]["annotation"])
            self.assertTrue(
                plan["entries"][1]["annotation"].endswith("custom/nested/b.txt")
            )

    def test_unet_keeps_unlabelled_images_but_training_uses_only_real_masks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            masks = root / "original_masks/nested"
            masks.mkdir(parents=True)
            write_image(masks / "a.png", np.ones((24, 32), np.uint8))
            source.write_text(
                yaml.safe_dump(
                    dict(
                        path=str(root),
                        train_images="images/train",
                        train_masks="original_masks",
                        num_classes=2,
                        class_names=["background", "object"],
                    )
                )
            )
            plan = preview(defaults([source], "unet"), "declared")
            path = write_manifest(plan, root / "approved")[0]
            self.assertEqual(len(load_unet_yaml(path)["pairs"]), 2)
            self.assertIsNone(load_unet_yaml(path)["pairs"][1][1])
            self.assertEqual(len(load_training_sources(path)["splits"]["train"]), 1)

    def test_coco_root_json_and_empty_images_are_preserved(self):
        from utils_new.import_export.coco import load_coco

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            cfg = root / "annotations.json"
            cfg.write_text(
                json.dumps(
                    dict(
                        images=[
                            dict(
                                id=i,
                                file_name=f"images/train/nested/{name}.jpg",
                                width=32,
                                height=24,
                            )
                            for i, name in enumerate(("a", "b"))
                        ],
                        categories=[dict(id=1, name="object")],
                        annotations=[
                            dict(id=1, image_id=0, category_id=1, bbox=[4, 4, 8, 8])
                        ],
                    )
                )
            )
            plan = preview(defaults([cfg], "coco"), "json")
            self.assertEqual(
                (plan["summaries"][0]["images"], plan["summaries"][0]["files"]), (2, 1)
            )
            approved = write_manifest(plan, root / "approved")[0]
            items, _, _ = load_coco(approved)
            self.assertEqual(len(items), 2)
            self.assertEqual(len(items[1]["annotations"]), 0)

    def test_subset_export_preserves_missing_labels_instead_of_inventing_background(
        self,
    ):
        from utils_new.import_export.dataset_export import export_subset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            approved = write_manifest(
                preview(defaults([source], "yolo"), "labels"), root / "view"
            )[0]
            pairs = load_yolo_yaml(approved)[0]
            output = export_subset(pairs, root / "export", {0: "selected", 1: "old"})
            restored = load_yolo_yaml(output)[0]
            self.assertEqual(len(restored), 2)
            self.assertIsNone(restored[1][1])
            self.assertFalse((root / "export/labels/train/b.txt").exists())

    def test_native_yolo_reads_approved_txt_and_does_not_fall_back_for_missing(self):
        from utils_new.import_export.dataset_sources import enable_ultralytics_sources

        enable_ultralytics_sources()
        from ultralytics.data.build import YOLODataset
        from ultralytics.utils import DEFAULT_CFG

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture(root)
            approved = write_manifest(
                preview(defaults([source], "yolo"), "labels"), root / "approved"
            )[0]
            data = yaml.safe_load(Path(approved).read_text())
            dataset = YOLODataset(
                img_path=data["train"],
                data=data,
                imgsz=32,
                augment=False,
                batch_size=2,
                rect=False,
                cache=False,
                stride=32,
                hyp=deepcopy(DEFAULT_CFG),
            )
            self.assertEqual(dataset.labels[0]["cls"].ravel().tolist(), [0.0])
            self.assertEqual(dataset.labels[1]["cls"].size, 0)
            self.assertTrue(
                dataset.label_files[1].startswith(str(root / "approved/unlabelled"))
            )
            import pickle

            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored.label_files, dataset.label_files)


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI checks")
class ImportDialogTests(unittest.TestCase):
    def gui(self, code):
        with TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                NO_ALBUMENTATIONS_UPDATE="1",
                MPLCONFIGDIR=directory,
                XDG_CACHE_HOME=directory,
                XDG_CONFIG_HOME=directory,
                YOLO_CONFIG_DIR=directory,
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            prefix = "import sys;sys.path.insert(0,'tests')\nfrom test_dataset_import import fixture\n"
            result = subprocess.run(
                [sys.executable, "-c", prefix + code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dialog_updates_paths_counts_root_and_cancel_has_no_side_effects(self):
        self.gui("""
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QDialogButtonBox
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from ui.dataset_import import DatasetImportDialog
app=QApplication([])
def ready(dialog):
 timer=QElapsedTimer();timer.start()
 while dialog.plan is None and timer.elapsed()<10000:QTest.qWait(20)
 assert dialog.plan is not None,dialog.notice.text()
with TemporaryDirectory() as directory:
 root=Path(directory);source=fixture(root)
 dialog=DatasetImportDialog([str(source)],'yolo');dialog.show();ready(dialog)
 assert dialog.table.item(0,3).text()=='2'
 assert dialog.table.item(0,4).text()=='1'
 assert dialog.table.item(0,5).text()=='1'
 dialog.mode.setCurrentIndex(dialog.mode.findData('beside'));ready(dialog)
 assert dialog.table.item(0,4).text()=='2'
 assert 'images/train' in dialog.annotation_fields[0].text()
 other=root/'relocated';other.mkdir();fixture(other)
 dialog.root.setText(str(other));ready(dialog)
 assert str(other) in dialog.image_fields[0].text()
 dialog.mode.setCurrentIndex(dialog.mode.findData('none'));ready(dialog)
 assert dialog.table.item(0,5).text()=='2'
 dialog.reject();assert dialog.result()==0
 assert not list(root.glob('neuro-dataset-*'))
app.processEvents()
""")

    def test_real_yaml_actions_on_all_dataset_tabs_keep_selected_and_missing_labels(
        self,
    ):
        self.gui("""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from PyQt5.QtCore import QTimer,QCoreApplication,QEvent
from ui.dataset_import import DatasetImportDialog
from tabs.layout_dataset import Labeler
from tabs.viewing_dataset import ViewingDataset
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from tabs.dataset_analysis import DatasetAnalysisWindow
from tabs.slice_dataset_sahi import SahiSlicerWindow
app=QApplication([])
def accept():
 dialog=app.activeModalWidget()
 if isinstance(dialog,DatasetImportDialog) and dialog.plan is not None:dialog.accept()
timer=QTimer();timer.timeout.connect(accept);timer.start(10)
with TemporaryDirectory() as directory:
 root=Path(directory);source=fixture(root)
 import numpy as np
 from utils_new.import_export.images import write_image
 masks=root/'masks/train/nested';masks.mkdir(parents=True)
 write_image(masks/'a.png',np.ones((24,32),np.uint8))
 for cls in (Labeler,ViewingDataset,DatasetPreparationWindow,DatasetAnalysisWindow,SahiSlicerWindow):
  window=cls();window.show()
  with patch.object(QFileDialog,'getOpenFileName',return_value=(str(source),'')),patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as errors:
   window.btn_load_yaml.click();wait_dataset(window);errors.assert_not_called()
  if hasattr(window,'image_paths'):
   assert len(window.image_paths)==2
   assert window._yaml_label_map[str(root/'images/train/nested/b.jpg')] is None
   if isinstance(window,ViewingDataset):
    assert window._get_annotations(1)==[]
    assert not window._yaml_mask_map
   else:
    window.current_index=1;window.display_current_image();wait_dataset(window);assert window.current_annotations==[]
  else:
   assert len(window.pairs)==2,window.pairs
   assert window.pairs[1][1] is None
  from PyQt5.QtWidgets import QTextEdit
  texts=window.findChildren(QTextEdit)
  assert any('[ИСТОЧНИК]' in text.toPlainText() for text in texts)
  window.close();window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
timer.stop();app.processEvents()
""")

    def test_real_coco_and_unet_imports_share_confirmation_and_keep_empty_images(self):
        self.gui("""
import json,yaml
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from PyQt5.QtCore import QTimer,QCoreApplication,QEvent
from ui.dataset_import import DatasetImportDialog
from tabs.layout_dataset import Labeler
from tabs.viewing_dataset import ViewingDataset
from tabs.preparing_dataset_yaml import DatasetPreparationWindow
from tabs.dataset_analysis import DatasetAnalysisWindow
from utils_new.import_export.images import write_image
app=QApplication([])
def accept():
 d=app.activeModalWidget()
 if isinstance(d,DatasetImportDialog) and d.plan is not None:d.accept()
 elif isinstance(d,QMessageBox):
  button=next((b for b in d.buttons() if b.text()=='Файлы JSON…'),None)
  if button:button.click()
timer=QTimer();timer.timeout.connect(accept);timer.start(10)
with TemporaryDirectory() as directory:
 root=Path(directory);fixture(root)
 masks=root/'masks/train/nested';masks.mkdir(parents=True)
 write_image(masks/'a.png',np.ones((24,32),np.uint8))
 source=root/'masks_data.yaml';source.write_text(yaml.safe_dump(dict(path=str(root),train_images='images/train',train_masks='masks/train',class_names=['background','object'],num_classes=2)))
 coco=root/'annotations.json';coco.write_text(json.dumps(dict(images=[dict(id=i,file_name=f'images/train/nested/{name}.jpg',width=32,height=24) for i,name in enumerate(('a','b'))],categories=[dict(id=1,name='object')],annotations=[])))
 for cls in (Labeler,ViewingDataset,DatasetPreparationWindow,DatasetAnalysisWindow):
  window=cls();window.show()
  with patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as errors:
   with patch.object(QFileDialog,'getOpenFileName',return_value=(str(source),'')):
    window.btn_load_unet.click();wait_dataset(window)
   errors.assert_not_called()
   assert len(window.image_paths if hasattr(window,'image_paths') else window.pairs)==2
   with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(coco)],'')):
    window.btn_load_coco.click();wait_dataset(window)
   errors.assert_not_called()
   assert len(window.image_paths if hasattr(window,'image_paths') else window.pairs)==2
  window.close();window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
timer.stop();app.processEvents()
""")

    def test_deeplab_preset_keeps_selected_paths_after_temporary_view_is_deleted(self):
        self.gui("""
import shutil,yaml
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget
from ui.dataset_source import set_dataset_path
from utils_new.import_export.dataset_sources import defaults,preview,write_manifest
from utils_new.import_export.mask_yaml import load_training_sources
from utils_new.import_export.images import write_image
app=QApplication([])
with TemporaryDirectory() as directory:
 root=Path(directory);source=fixture(root)
 masks=root/'masks/train/nested';masks.mkdir(parents=True)
 write_image(masks/'a.png',np.ones((24,32),np.uint8))
 source.write_text(yaml.safe_dump(dict(path=str(root),train_images='images/train',train_masks='masks/train',class_names=['background','object'],num_classes=2)))
 approved=write_manifest(preview(defaults([source],'unet'),'declared'),root/'temporary')[0]
 widget=DeepLabV3TrainWidget();set_dataset_path(widget.deeplab_yaml,approved)
 target=root/'preset.txt';widget.preset_path.setText(str(target))
 with patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as error:
  widget._save_preset();error.assert_not_called()
 saved=next(line.partition(':')[2].strip() for line in target.read_text().splitlines() if line.startswith('deeplab_yaml:'))
 shutil.rmtree(root/'temporary')
 assert Path(saved).is_file()
 assert len(load_training_sources(saved)['splits']['train'])==1
 widget.close();app.processEvents()
""")

    def test_training_fields_display_original_source_and_pass_approved_view(self):
        self.gui("""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QFileDialog
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from PyQt5.QtCore import QTimer
from ui.dataset_import import DatasetImportDialog
from ui.dataset_source import dataset_path
from ui.DL_train_settings_base import YOLOTrainWidget
app=QApplication([])
def accept():
 d=app.activeModalWidget()
 if isinstance(d,DatasetImportDialog) and d.plan is not None:d.accept()
timer=QTimer();timer.timeout.connect(accept);timer.start(10)
with TemporaryDirectory() as directory:
 source=fixture(Path(directory));widget=YOLOTrainWidget();widget.show()
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(source),'')):
  widget.browse_yaml_btn.click()
 assert widget.dataset_yaml.text()==str(source)
 approved=dataset_path(widget.dataset_yaml)
 assert approved!=str(source) and Path(approved).is_file()
 widget.dataset_yaml.setText('manual.yaml');assert dataset_path(widget.dataset_yaml)=='manual.yaml'
 widget.close()
timer.stop();app.processEvents()
""")


if __name__ == "__main__":
    unittest.main()
