"""Arbitrary landmark schemas and image-level classes must roundtrip without loss."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from concurrent.futures import CancelledError
import json
import os
import subprocess
import sys
import unittest
import numpy as np

from utils_new.annotations.model import (
    Annotation,
    PoseAnnotation,
    PoseSchema,
    ImageClassification,
    ImageSize,
)
from utils_new.annotations.geometry import rectangle_points
from utils_new.import_export.images import write_image
from utils_new.import_export.task_datasets import (
    TaskDataset,
    TaskSample,
    load_task_dataset,
    save_task_project,
    export_task_dataset,
    decode_pose,
    encode_pose,
    merge_task_datasets,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, task="pose"):
    samples = []
    schema = PoseSchema(
        ("tip_a", "tip_b", "center", "control"), ((0, 2), (2, 1), (2, 3)), (1, 0, 2, 3)
    )
    for i in range(5):
        path = root / f"{i}.png"
        write_image(
            path, np.random.default_rng(i).integers(0, 256, (48, 80, 3), dtype=np.uint8)
        )
        pose = PoseAnnotation(
            Annotation("detect", 0, rectangle_points(5, 5, 75, 43)),
            np.array([[10, 20, 2], [70, 20, 1], [40, 22, 2], [0, 0, 0]]),
        )
        samples.append(
            TaskSample(
                str(path),
                "train" if i < 2 else "val",
                ImageClassification(i % 2 if task == "classify" else None, True),
                poses=[pose] if task == "pose" else None,
                size=ImageSize(80, 48),
            )
        )
    return TaskDataset(
        task,
        {0: "part"} if task == "pose" else {0: "normal", 1: "defect"},
        samples,
        schema if task == "pose" else None,
    )


class TaskDatasetContracts(unittest.TestCase):
    def test_arbitrary_schema_and_visibility_zero_at_image_origin(self):
        for count in (1, 4, 21, 37):
            schema = PoseSchema(tuple(f"p{i}" for i in range(count)))
            pts = np.zeros((count, 3))
            pts[0] = [0, 0, 2]
            pose = PoseAnnotation(
                Annotation("detect", 0, rectangle_points(0, 0, 80, 48)), pts
            )
            value = encode_pose([pose], ImageSize(80, 48), schema)
            decoded = decode_pose(value, ImageSize(80, 48), schema)[0]
            np.testing.assert_array_equal(decoded.keypoints, pts)
        for bad in (
            dict(names=("a", "a")),
            dict(names=("a", "b"), edges=((0, 2),)),
            dict(names=("a", "b", "c"), flip_idx=(1, 2, 0)),
        ):
            with self.assertRaises(ValueError):
                PoseSchema(**bad)
        with self.assertRaises(ValueError):
            decode_pose("0 .5 .5 .5 .5 0 0 3", ImageSize(80, 48), PoseSchema(("x",)))

    def test_yolo_coco_and_project_roundtrips_and_unlabelled_are_not_background(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            # Merely viewing a missing TXT cannot invent a background label.
            extra = root / "unlabelled.png"
            write_image(extra, np.zeros((48, 80, 3), np.uint8))
            data.samples.append(
                TaskSample(str(extra), poses=[], size=ImageSize(80, 48))
            )
            export_task_dataset(data, root / "yolo")
            loaded = load_task_dataset(root / "yolo/data.yaml", "pose")
            self.assertEqual(len(loaded.samples), 5)
            self.assertEqual(loaded.schema, data.schema)
            np.testing.assert_allclose(
                loaded.load_poses(loaded.samples[0])[0].keypoints,
                data.samples[0].poses[0].keypoints,
                atol=1e-7,
            )
            export_task_dataset(data, root / "coco", coco=True)
            coco = load_task_dataset(root / "coco/annotations.json", "pose")
            self.assertEqual(coco.schema, data.schema)
            self.assertEqual(len(coco.samples), 5)
            np.testing.assert_array_equal(
                coco.samples[0].poses[0].keypoints, data.samples[0].poses[0].keypoints
            )
            save_task_project(data, root / "project.json")
            project = load_task_dataset(root / "project.json", "pose")
            self.assertEqual(len(project.samples), 6)
            self.assertEqual(project.schema, data.schema)
            with self.assertRaises(FileExistsError):
                export_task_dataset(data, root / "yolo")
            stop = Event()
            stop.set()
            with self.assertRaises(CancelledError):
                export_task_dataset(data, root / "cancel", cancel=stop)
            self.assertFalse((root / "cancel").exists())

    def test_invalid_lazy_txt_never_enters_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            sample = data.samples[0]
            sample.poses = None
            label = root / "bad.txt"
            label.write_text("99 .5 .5 .5 .5 " + "0 0 0 " * 4, encoding="utf-8")
            sample.label_path = str(label)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "неизвестный класс"):
                    data.load_poses(sample)
                self.assertIsNone(sample.poses)

    def test_coco_empty_background_survives_project_merge_and_export(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            data.samples[0].poses = []
            data.samples[0].classification = ImageClassification(None, False)
            data.samples[0].pose_labeled = True
            export_task_dataset(data, root / "coco", coco=True)
            loaded = load_task_dataset(root / "coco/annotations.json", "pose")
            self.assertTrue(loaded.samples[0].has_pose_labels)
            self.assertFalse(loaded.samples[0].classification.reviewed)
            save_task_project(loaded, root / "session.json")
            loaded = load_task_dataset(root / "session.json", "pose")
            export_task_dataset(loaded, root / "yolo")
            self.assertEqual(
                len(load_task_dataset(root / "yolo/data.yaml", "pose").samples), 5
            )
            from copy import deepcopy

            unlabelled = deepcopy(loaded)
            unlabelled.samples[0].pose_labeled = False
            merged = merge_task_datasets(unlabelled, loaded)
            self.assertTrue(merged.samples[0].pose_labeled)
            cfg = json.loads((root / "coco/annotations.json").read_text())
            cfg["images"].append(dict(cfg["images"][0]))
            (root / "bad.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "ID снимка"):
                load_task_dataset(root / "bad.json", "pose")

    def test_classification_folder_export_preserves_names_mapping_and_unlabelled(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root, "classify")
            data.samples[-1].classification = ImageClassification(None)
            export_task_dataset(data, root / "cls")
            loaded = load_task_dataset(root / "cls", "classify")
            self.assertEqual(loaded.classes, {0: "defect", 1: "normal"})
            self.assertEqual(len(loaded.samples), 4)
            self.assertEqual(
                [
                    loaded.classes[s.classification.class_id] for s in loaded.samples
                ].count("defect"),
                2,
            )
            save_task_project(data, root / "cls.json")
            project = load_task_dataset(root / "cls.json", "classify")
            self.assertIsNone(project.samples[-1].classification.class_id)
            from utils_new.configuration.training import build_train_args

            params = dict(
                model="yolo11n-cls.pt", data=str(root / "cls"), task="classify"
            )
            self.assertEqual(build_train_args(params)["data"], str(root / "cls"))
            params["data"] = str(root / "cls.json")
            with self.assertRaises(ValueError):
                build_train_args(params)

    def test_merge_preserves_ids_rejects_conflicts_and_detects_split_leakage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            a = fixture(root / "a", "classify")
            b = fixture(root / "b", "classify")
            b.classes = {0: "defect", 1: "normal"}
            merged = merge_task_datasets(a, b)
            self.assertEqual(merged.classes, a.classes)
            self.assertEqual(len(merged.samples), 10)
            self.assertEqual(merged.samples[5].classification.class_id, 1)
            from copy import deepcopy

            bad = deepcopy(a)
            bad.samples[0].split = "val"
            with self.assertRaisesRegex(ValueError, "split"):
                merge_task_datasets(a, bad)
            # Same bytes in a different split cannot be exported as independent data.
            duplicate = root / "duplicate.png"
            duplicate.write_bytes(Path(a.samples[0].path).read_bytes())
            a.samples.append(TaskSample(str(duplicate), "val", ImageClassification(0)))
            with self.assertRaisesRegex(ValueError, "разные split"):
                export_task_dataset(a, root / "leak")
            self.assertFalse((root / "leak").exists())

    def test_empty_pose_txt_remains_labelled_background_after_project_save(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            image = root / "empty.png"
            write_image(image, np.zeros((48, 80, 3), np.uint8))
            label = image.with_suffix(".txt")
            label.write_text("")
            data.samples.append(TaskSample(str(image), "train", label_path=str(label)))
            save_task_project(data, root / "session.json")
            restored = load_task_dataset(root / "session.json", "pose")
            self.assertIsNotNone(restored.samples[-1].label_path)
            self.assertEqual(restored.samples[-1].poses, [])
            export_task_dataset(restored, root / "out")
            exported = load_task_dataset(root / "out/data.yaml", "pose")
            self.assertEqual(len(exported.samples), 6)
            self.assertEqual(
                len([s for s in exported.samples if exported.load_poses(s) == []]), 1
            )

    def test_pose_training_validates_shape_and_mirroring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(root)
            data.schema = replace_schema = PoseSchema(
                data.schema.names, data.schema.edges
            )
            export_task_dataset(data, root / "pose")
            from utils_new.configuration.training import build_train_args

            params = dict(
                model="yolo11n-pose.pt",
                data=str(root / "pose/data.yaml"),
                task="pose",
                disable_aug=False,
            )
            with self.assertRaisesRegex(ValueError, "flip_idx"):
                build_train_args(params)
            self.assertEqual(build_train_args(dict(params, fliplr=0))["task"], "pose")


class NativeTaskTraining(unittest.TestCase):
    def test_native_training_checkpoints_reload_without_json_and_predictions(self):
        code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json, subprocess,sys
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_pose_classification import fixture
from utils_new.import_export.task_datasets import export_task_dataset
from utils_new.import_export.training import read_model_meta
import ultralytics
cfgroot=Path(ultralytics.__file__).parent/'cfg/models/11'
with TemporaryDirectory() as directory:
 root=Path(directory)
 for task,suffix in [('classify','cls'),('pose','pose')]:
  data=fixture(root/task,task);export_task_dataset(data,root/(task+'_export'))
  cfg=dict(model=str(cfgroot/f'yolo11-{suffix}.yaml'),task=task,
           data=str(root/(task+'_export')) if task=='classify' else str(root/(task+'_export/data.yaml')),
           epochs=1,batch=2,imgsz=64,pretrained=False,workers=0,amp=False,plots=False,
           project=str(root/'runs'),name=task,device='cpu',disable_aug=True)
  path=root/f'{task}.json';path.write_text(json.dumps(cfg))
  result=subprocess.run([sys.executable,'-m','utils_new.training.yolo','--params_file',str(path)],text=True,capture_output=True,timeout=180)
  assert result.returncode==0,(result.stdout+result.stderr)[-6000:]
  weights=root/'runs'/task/'weights/best.pt';meta=read_model_meta(weights);assert meta['task']==task
  assert meta['validation_metrics']
  if task=='pose':assert len(meta['pose_schema']['names'])==4
  weights.with_name('best_meta.json').unlink()
  from utils_new.import_export.model_load import load_yolo
  from utils_new.import_export.images import read_image
  model=load_yolo(str(weights));prediction=model.predict(read_image(data.samples[0].path),verbose=False,imgsz=64,device='cpu')[0]
  assert prediction.probs is not None if task=='classify' else prediction.keypoints is not None
  if task=='pose':assert len(model.neuro_metadata['pose_schema']['names'])==4
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr)[-8000:])


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in Qt")
class TaskAnnotationGUI(unittest.TestCase):
    def test_both_tabs_schema_edit_points_undo_class_batch_and_navigation(self):
        code = r"""
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QElapsedTimer,Qt
from PyQt5.QtTest import QTest
from tabs.layout_dataset import Labeler
from tabs.viewing_dataset import ViewingDataset
from utils_new.import_export.images import write_image
from utils_new.annotations.model import PoseSchema
import numpy as np,sys
app=QApplication([]);a=Labeler();b=ViewingDataset();a.resize(1400,950);a.show()
assert 'torch' not in sys.modules
p=a.task_modes.pose;a.task_modes.selector.setCurrentIndex(1)
def wait(p):
 timer=QElapsedTimer();timer.start()
 while p.job is not None and timer.elapsed()<10000:QTest.qWait(10)
 assert p.job is None,p.status.text()
with TemporaryDirectory() as directory:
 root=Path(directory);paths=[]
 for i in range(3):
  path=root/f'{i}.png';write_image(path,np.full((100,160,3),60+i,np.uint8));paths.append(str(path))
 p.dataset.schema=PoseSchema(('left','right','center','tip'),((0,2),(2,1),(2,3)),(1,0,2,3))
 p.load_images(paths);wait(p);assert p.dirty
 assert p.image.shape==(100,160,3),p.status.text()
 p.add_bbox((10,10,150,90))
 from PyQt5.QtCore import QPointF
 p.set_tool('points');position=p.view.mapFromScene(QPointF(25,30))
 QTest.mouseClick(p.view.viewport(),Qt.LeftButton,pos=position);app.processEvents()
 assert p.dataset.samples[0].poses[0].keypoints[0,2]==2
 assert p.table.currentRow()==1
 p.remember();p.move_point(0,25,30,True);p.remember();p.move_point(1,135,30,True)
 assert p.dataset.samples[0].poses[0].keypoints[1,2]==2
 p.undo();assert p.dataset.samples[0].poses[0].keypoints[1,2]==0
 p.undo(True);assert p.dataset.samples[0].poses[0].keypoints[1,2]==2
 p.table.setCurrentCell(1,0);p.set_point_visibility(1);assert p.dataset.samples[0].poses[0].keypoints[1,2]==1
 p.view.viewport().update();app.processEvents()
 # Removing the final object is an explicitly labeled empty image, including undo.
 p.delete_pose();assert p.dataset.samples[0].has_pose_labels and not p.dataset.samples[0].poses
 p.undo();assert len(p.dataset.samples[0].poses)==1
 p.navigate(1);wait(p);assert not p.dataset.samples[1].poses
 p.navigate(0);wait(p);assert len(p.dataset.samples[0].poses)==1
 from utils_new.import_export.task_datasets import save_task_project
 save_task_project(p.dataset,root/'pose.json');p.dirty=False
 b.task_modes.selector.setCurrentIndex(1);v=b.task_modes.pose;v.load_source(str(root/'pose.json'));wait(v)
 assert not v.edit.isChecked() and v.view.pose_data[0].keypoints.shape==(4,3)
 c=a.task_modes.classify;a.task_modes.selector.setCurrentIndex(2);c.load_images(paths);wait(c)
 c.images.item(0).setSelected(True);c.images.item(1).setSelected(True)
 c.class_combo.setCurrentIndex(1);c.assign_class()
 assert c.dataset.samples[0].classification.class_id==1 and c.dataset.samples[1].classification.class_id==1
 c.undo();assert c.dataset.samples[0].classification.class_id is None
 c.append.setChecked(True);c.load_images([paths[2]]);wait(c)
 assert c.dirty and c.dataset.samples[1].classification.class_id==1
 # Cancelled image reads must not mutate a live sample through a worker.
 from threading import Event
 import ui.task_annotations as task_ui
 started=Event();release=Event();original=task_ui.read_image_with_fallback
 def slow_read(path):
  started.set();release.wait(5);return original(path)
 task_ui.read_image_with_fallback=slow_read
 p.dataset.samples[1].poses=None
 p.navigate(1)
 assert started.wait(5)
 p.cancel_job();release.set();QTest.qWait(100)
 assert p.dataset.samples[1].poses is None
 task_ui.read_image_with_fallback=original
 c.dirty=False;v.dirty=False
 a.task_modes.selector.setCurrentIndex(0);assert not a.current_annotations
 a.close();b.close();app.processEvents()
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
