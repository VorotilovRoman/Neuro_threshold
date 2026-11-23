"""Regression coverage for strict objects, safe export and dataset audit lifecycle."""

import os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager
import cv2, numpy as np, yaml
from utils_new.dataset_audit.analysis import (
    DatasetAnalyzer,
    FilterParams,
    ClassImbalanceAnalyzer,
    get_image_brightness,
)
from utils_new.dataset_audit.objects import read_objects, load_pair_annotations
from utils_new.dataset_audit.checks import (
    DuplicateDetector,
    LeakageChecker,
    ObjectDistributionHeatmap,
)
from utils_new.import_export.dataset_export import export_subset, infer_split
from utils_new.dataset_audit.splits import split_memberships
from utils_new.dataset_audit.matching import match_detections, project_embeddings
from utils_new.dataset_audit.jobs import (
    LossInspectionJob,
    ReportJob,
    EmbeddingClusterJob,
    YoloValidationJob,
)

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def injected_module(name, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name="image.png", value=30, shape=(40, 80)):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full(shape, value, np.uint8))
        return path

    def pair(self, name="image.png", text="0 .5 .5 .5 .5\n"):
        image = self.image(name)
        label = image.with_suffix(".txt")
        label.write_text(text)
        return (str(image), str(label))

    def test_numerical_modules_import_without_qt_torch_or_models(self):
        code = """
import importlib, sys
for name in ('analysis', 'objects', 'checks', 'splits', 'matching', 'jobs'):
    importlib.import_module('utils_new.dataset_audit.' + name)
for name in ('dataset_export','atomic'):
    importlib.import_module('utils_new.import_export.' + name)
for prefix in ('PyQt5', 'torch', 'ultralytics', 'open_clip', 'transformers', 'matplotlib.pyplot'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonsquare_xywhr_and_mask_components_have_real_geometry(self):
        pair = self.pair(text="2 .5 .5 .4 .2 0.7853981633974483\n")
        annotations = load_pair_annotations(*pair, format_hint="obb_xywhr")
        points = np.array(annotations[0][2]).reshape(-1, 2) * [80, 40]
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        np.testing.assert_allclose(sorted(lengths), [8, 8, 32, 32], rtol=1e-6)
        mask = np.zeros((40, 80), np.uint16)
        mask[1:3, 2:4] = 5
        mask[5:7, 5:8] = 5
        mask[12, 13] = 9
        mask_path = self.root / "mask.png"
        cv2.imwrite(str(mask_path), mask)
        size, objects = read_objects((pair[0], str(mask_path)), 1)
        self.assertEqual(size, (80, 40))
        self.assertEqual(sorted(o.area_px for o in objects), [1, 4, 6])
        self.assertEqual([o.class_id for o in objects], [5, 5, 9])
        np.testing.assert_allclose(
            objects[-1].center, [(13 + 0.5) / 80, (12 + 0.5) / 40]
        )
        result = DatasetAnalyzer(1).filter_pairs(
            [(pair[0], str(mask_path))], {5, 9}, 2, FilterParams()
        )
        self.assertEqual(result["total_objects"], 3)
        self.assertEqual(sum(result["object_areas"]), 11)

    def test_missing_invalid_empty_and_excluded_labels_are_not_equivalent(self):
        valid = self.pair()
        empty = self.pair("empty.png", "")
        other = self.pair("other.png", "1 .5 .5 .1 .1\n")
        bad = self.pair("bad.png", "bad data\n")
        missing = (str(self.image("missing.png")), None)
        result = DatasetAnalyzer().filter_pairs(
            [valid, empty, other, bad, missing],
            {0},
            2,
            FilterParams(),
            include_empty=True,
        )
        self.assertEqual(result["filtered_pairs"], [valid, empty, missing])
        self.assertEqual(result["empty_images"], 1)
        self.assertEqual(result["total_objects"], 1)
        self.assertEqual(result['unlabelled_images'], 1)
        self.assertEqual(len(result["errors"]), 1)
        result = DatasetAnalyzer(2).filter_pairs([missing], set(), 0, FilterParams())
        self.assertEqual(result["filtered_pairs"], [missing])
        self.assertTrue(
            DatasetAnalyzer().filter_pairs(
                [valid], {0}, 1, FilterParams(min_w=100, max_w=50)
            )["error"]
        )

    def test_brightness_unicode_depth_cache_and_zero_count_class_ids(self):
        image = self.image("снимок.png", 0)
        self.assertEqual(get_image_brightness(str(image)), 0)
        cv2.imwrite(str(image), np.full((40, 80), 200, np.uint8))
        os.utime(image, None)
        self.assertEqual(get_image_brightness(str(image)), 200)
        deep = self.root / "deep.tiff"
        cv2.imwrite(str(deep), np.full((10, 20), 32768, np.uint16))
        self.assertEqual(get_image_brightness(str(deep)), 128)
        result = ClassImbalanceAnalyzer.analyze({99: 0, 7: 20, 3: 2})
        self.assertEqual(result["majority_class"], 7)
        self.assertEqual(result["minority_class"], 3)

    def test_duplicate_aliases_and_transitive_near_groups(self):
        a = self.image("a.png", 10)
        b = self.image("b.png", 20)
        c = self.image("c.png", 30)
        hashes = {str(a): 0, str(b): 1, str(c): 2}

        class Hash:
            def __init__(self, value):
                self.value = value

            def __sub__(self, other):
                return abs(self.value - other.value)

        with patch.object(
            DuplicateDetector, "_phash", side_effect=lambda path: Hash(hashes[path])
        ):
            result = DuplicateDetector.find_duplicates(
                [(str(a), None), (str(a), None), (str(b), None), (str(c), None)],
                phash_threshold=1,
            )
        self.assertEqual(result["exact"], [])
        self.assertEqual(len(result["near"]), 1)
        self.assertEqual(len(result["near"][0]), 3)
        b.write_bytes(a.read_bytes())
        with patch.object(DuplicateDetector, "_phash", return_value=None):
            result = DuplicateDetector.find_duplicates(
                [(str(a), None), (str(a), None), (str(b), None)]
            )
        self.assertEqual(result["exact"], [[str(a), str(b)]])

    def test_split_components_and_yaml_overlapping_paths_and_manifests(self):
        self.assertEqual(infer_split("some/training/contest.png"), "unknown")
        self.assertEqual(infer_split("C:\\data\\validation\\contest.png"), "val")
        first = self.pair("pictures/a.png")
        second = self.pair("pictures/b.png")
        (self.root / "selected.txt").write_text("pictures/a.png\n")
        cfg = self.root / "data.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "path": str(self.root),
                    "train": "selected.txt",
                    "val": ["pictures/a.png", "pictures/b.png"],
                }
            )
        )
        membership = split_memberships(cfg, [first, second])
        self.assertEqual(membership[first[0]], {"train", "val"})
        self.assertEqual(membership[second[0]], {"val"})
        report = LeakageChecker.check([], [first, second], membership)
        self.assertEqual(report["leaked_count"], 1)
        with self.assertRaises(ValueError):
            export_subset(
                [first], self.root / "out", {0: "Object"}, split_by_image=membership
            )

    def test_split_yaml_errors_have_a_readable_value_error(self):
        cfg = self.root / "bad.yaml"
        for text in ("train: [", "train: 17"):
            cfg.write_text(text)
            with self.assertRaises(ValueError):
                split_memberships(cfg, [])

    def test_heatmap_format_and_mask_centers_match_object_statistics(self):
        pair = self.pair()
        result = ObjectDistributionHeatmap.compute([pair], grid_size=8)
        self.assertEqual(result["total_objects"], 1)
        self.assertEqual(result["heatmap"][4, 4], 1)
        label = Path(pair[1])
        label.write_text("bad\n")
        result = ObjectDistributionHeatmap.compute([pair])
        self.assertEqual(result["total_objects"], 0)
        self.assertEqual(len(result["errors"]), 1)

    def test_export_remaps_names_and_uses_image_stem_with_exact_split(self):
        pair = self.pair("val/picture.png", "7 .5 .5 .2 .2\n")
        label = Path(pair[1])
        label.rename(label.with_name("different.txt"))
        pair = (pair[0], str(label.with_name("different.txt")))
        destination = self.root / "out"
        name = "dog: red\nnext"
        config = Path(export_subset([pair], destination, {7: name}))
        data = yaml.safe_load(config.read_text())
        self.assertEqual(data["names"], {0: name})
        self.assertTrue((destination / "images/val/picture.png").exists())
        self.assertTrue(
            (destination / "labels/val/picture.txt").read_text().startswith("0 ")
        )
        self.assertTrue(Path(pair[0]).exists())
        self.assertTrue(Path(pair[1]).exists())

    def test_export_masks_roundtrip_and_failures_leave_no_partial_dataset(self):
        image = self.image()
        mask = np.zeros((40, 80), np.uint16)
        mask[:10, :10] = 5
        mask[20:30, 20:30] = 9
        mask_path = self.root / "mask.png"
        cv2.imwrite(str(mask_path), mask)
        out = self.root / "masks_out"
        export_subset(
            [(str(image), str(mask_path))], out, {5: "A", 9: "B"}, dataset_type=1
        )
        result = cv2.imread(str(out / "masks/train/image.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(set(np.unique(result)), {0, 1, 2})
        self.assertFalse((out / "labels/train/image.txt").exists())
        from utils_new.import_export.catalog import load_unet_catalog

        catalog = load_unet_catalog(out / "data.yaml")
        self.assertEqual(catalog.class_names, {1: "A", 2: "B"})
        self.assertEqual(len(catalog.pairs), 1)
        before = (out / "data.yaml").read_bytes()
        with self.assertRaises(ValueError):
            export_subset([(str(image), str(mask_path))], out, {5: "A"}, dataset_type=1)
        self.assertEqual((out / "data.yaml").read_bytes(), before)
        first = self.pair("one/a.png")
        second = self.pair("two/a.png")
        target = self.root / "collision"
        with self.assertRaises(ValueError):
            export_subset([first, second], target, {0: "A"})
        self.assertFalse(target.exists())
        bad = self.pair("bad.png", "wrong data")
        target = self.root / "bad_out"
        target.mkdir()
        with self.assertRaises(ValueError):
            export_subset([bad], target, {0: "A"})
        self.assertEqual(list(target.iterdir()), [])

    def test_one_to_one_class_matching_projection_and_atomic_report(self):
        box = [0, 0, 10, 10]
        result = match_detections([box, box], [0, 0], [box], [0])
        self.assertEqual(result["false_negatives"], 1)
        result = match_detections([box], [0], [box], [1])
        self.assertEqual((result["false_negatives"], result["false_positives"]), (1, 1))
        self.assertEqual(project_embeddings([[1, 2, 3]]).shape, (1, 2))
        self.assertEqual(project_embeddings([[1], [2]]).shape, (2, 2))
        with self.assertRaises(ValueError):
            project_embeddings([[float("nan")]])
        path = self.root / "report.md"
        path.write_text("previous")
        job = ReportJob({"general": {"Pairs": 1}}, str(path), {})
        with patch(
            "utils_new.import_export.atomic.os.replace", side_effect=OSError("disk")
        ):
            self.assertIn("disk", job.run()["error"])
        self.assertEqual(path.read_text(), "previous")
        self.assertIsNone(job.run()["error"])
        self.assertIn("Pairs", path.read_text())
        previous = path.read_bytes()
        job.cancel()
        self.assertTrue(job.run()["cancelled"])
        self.assertEqual(path.read_bytes(), previous)

    def test_model_audit_distinguishes_misses_wrong_classes_and_negatives(self):
        import types

        pair = self.pair()

        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)

            def cpu(self):
                return self

            def numpy(self):
                return self.data

        class Boxes:
            def __init__(self, classes):
                self.xyxy = Tensor([[20, 10, 60, 30] for _ in classes])
                self.conf = Tensor([0.9] * len(classes))
                self.cls = Tensor(classes)
                self.n = len(classes)

            def __len__(self):
                return self.n

        class Model:
            def __init__(self, path):
                pass

            def __call__(self, *args, **kwargs):
                return [
                    types.SimpleNamespace(
                        boxes=Boxes(classes), obb=None, orig_shape=(40, 80)
                    )
                ]

        module = types.ModuleType("ultralytics")
        module.YOLO = Model
        for text, classes, expected in [
            ("0 .5 .5 .5 .5\n", [0], False),
            ("0 .5 .5 .5 .5\n", [1], True),
            ("", [], False),
        ]:
            Path(pair[1]).write_text(text)
            with (
                injected_module("ultralytics", module),
                patch("runtime_torch.require_torch", return_value=object()),
                patch("runtime_policy.configure_ultralytics"),
            ):
                result = LossInspectionJob([pair], "mock.pt", device="cpu").run()
            self.assertIsNone(result["error"])
            self.assertEqual(len(result["all_results"]), 1)
            self.assertEqual(result["all_results"][0]["is_suspicious"], expected)

    def test_validation_logs_metrics_cancellation_and_handler_cleanup(self):
        import logging, types

        logger = logging.getLogger("dataset-validation-regression")
        logger.setLevel(logging.INFO)
        config = self.root / "data.yaml"
        config.write_text("names: [Object]")
        module = types.ModuleType("ultralytics")
        utilities = types.ModuleType("ultralytics.utils")
        utilities.LOGGER = logger

        class Model:
            def __init__(self, path):
                pass

            def add_callback(self, event, callback):
                self.callback = callback

            def val(self, **kwargs):
                logger.info("validation batch")
                self.callback()
                if broken:
                    raise ValueError("bad validation data")
                box = types.SimpleNamespace(
                    mp=0.8,
                    mr=0.5,
                    map50=0.7,
                    map=0.6,
                    map75=0.65,
                    ap_class_index=[0],
                    p=[0.8],
                    r=[0.5],
                    f1=[0.61],
                    ap50=[0.7],
                    ap=[0.6],
                )
                return types.SimpleNamespace(box=box, names={0: "Object"})

        module.YOLO = Model
        initial = list(logger.handlers)
        for broken, cancelled in [(False, False), (True, False), (False, True)]:
            job = YoloValidationJob("mock.pt", str(config), device="cpu")
            messages = []
            job.log = messages.append
            if cancelled:
                job.request_stop()
            with (
                injected_module("ultralytics", module),
                injected_module("ultralytics.utils", utilities),
                patch("runtime_torch.require_torch", return_value=object()),
                patch("runtime_policy.configure_ultralytics"),
            ):
                result = job.run()
            self.assertEqual(logger.handlers, initial)
            self.assertTrue(any("validation batch" in message for message in messages))
            if broken or cancelled:
                self.assertTrue(result["error"])
                self.assertIsNone(result["metrics"])
            else:
                self.assertEqual(result["metrics"]["summary"]["Precision"], 0.8)
                self.assertEqual(
                    result["metrics"]["per_class"][0]["class"], "0: Object"
                )

    def test_embedding_single_image_runs_without_download_or_pca_failure(self):
        import torch, types

        image = self.image()

        class Model:
            def to(self, device):
                return self

            def eval(self):
                return self

            def encode_image(self, batch):
                return torch.ones((len(batch), 4))

        module = types.ModuleType("open_clip")
        module.create_model_and_transforms = lambda *args, **kwargs: (
            Model(),
            None,
            lambda image: torch.ones((3, 4, 4)),
        )
        with (
            injected_module("open_clip", module),
            patch("runtime_torch.require_torch", return_value=torch),
            patch("utils_new.dataset_audit.jobs.CLIP_AVAILABLE", True),
            patch("utils_new.dataset_audit.jobs._CLIP_BACKEND", "open_clip"),
            patch("utils_new.dataset_audit.jobs.UMAP_AVAILABLE", False),
            patch("utils_new.dataset_audit.jobs.HDBSCAN_AVAILABLE", False),
        ):
            result = EmbeddingClusterJob([(str(image), None)], device="cpu").run()
        self.assertIsNone(result["error"])
        self.assertEqual(result["embeddings_2d"], [[0.0, 0.0]])
        self.assertEqual(result["clusters"], [-1])


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI opt-in")
class DatasetAuditGuiTests(unittest.TestCase):
    def run_gui(self, code):
        with tempfile.TemporaryDirectory() as config:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CACHE_HOME=config,
                XDG_CONFIG_HOME=config,
                MPLCONFIGDIR=config,
                YOLO_CONFIG_DIR=config,
                NO_ALBUMENTATIONS_UPDATE="1",
                PYTHONFAULTHANDLER="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_yaml_filters_masks_heatmap_and_empty_selection(self):
        self.run_gui("""
from pathlib import Path
import tempfile,cv2,numpy as np,yaml
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from tabs.dataset_analysis import DatasetAnalysisWindow
from PyQt5.QtTest import QTest
from PyQt5.QtCore import QElapsedTimer
app=QApplication([]);w=DatasetAnalysisWindow();w.show()
def settled():
 timer=QElapsedTimer();timer.start()
 while w.scans.worker is not None and timer.elapsed()<5000:QTest.qWait(10)
 assert w.scans.worker is None,w.analysis_status.text()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 for split,name in [('alpha','a'),('beta','b')]:
  folder=root/split;folder.mkdir();cv2.imwrite(str(folder/(name+'.png')),np.full((40,80),50,np.uint8));(folder/(name+'.txt')).write_text('0 .5 .5 .5 .5\\n')
 cfg=root/'data.yaml';cfg.write_text(yaml.safe_dump({'path':str(root),'train':'alpha','val':'beta','names':{0:'Object'}}))
 w._load_yolo_yaml_from_path(str(cfg));wait_dataset(w);assert len(w.filtered_pairs)==2 and w._last_filter_stats['total_objects']==2
 assert set().union(*w._analysis_splits.values())=={'train','val'}
 w.run_heatmap_analysis();settled();assert w._last_heatmap['total_objects']==2
 w.filter_min_area.setValue(100000);w.apply_filters_and_update_metrics();wait_dataset(w);assert not w.filtered_pairs and not w._get_analysis_pairs()
 assert w._last_heatmap is None and w.audit_table.rowCount()==0
 with patch.object(QMessageBox,'warning') as warning:w.run_quality_analysis();warning.assert_called_once()
 w.filter_min_area.setValue(0);w.apply_filters_and_update_metrics();wait_dataset(w);assert len(w.filtered_pairs)==2
 mask=root/'alpha'/'a_mask.png';array=np.zeros((40,80),np.uint8);array[3:6,3:6]=7;array[10:12,10:12]=7;cv2.imwrite(str(mask),array)
 w._apply_loaded_pairs(pairs=[(str(root/'alpha'/'a.png'),str(mask))],input_type=1,input_format='auto',class_names_override={7:'Mask'},source_label='Test mask')
 wait_dataset(w)
 assert w._last_filter_stats['total_objects']==2 and w.filtered_pairs
 w.run_heatmap_analysis();settled();assert w._last_heatmap['total_objects']==2
 # Cancellation must not become a successful empty/full-dataset diagnosis.
 with patch('utils_new.dataset_audit.checks.DuplicateDetector.find_duplicates',return_value={'cancelled':True}):
  w.run_duplicate_analysis();settled();assert 'отмен' in w.dup_summary_label.text()
 with patch('utils_new.dataset_audit.checks.ObjectDistributionHeatmap.compute',return_value={'cancelled':True}):
  w.run_heatmap_analysis();settled();assert 'отмен' in w.heatmap_summary_label.text()
 w._on_audit_finished({'cancelled':True});assert 'отменён' in w.audit_summary_label.text()
w.close();app.processEvents()
""")

    def test_stale_worker_and_close_ignore_late_results(self):
        self.run_gui("""
from threading import Event
import time
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from main import ImagePlayer
from ui.dataset_audit.workers import LossInspectionWorker
app=QApplication([]);window=ImagePlayer([('dataset_analysis','Analysis')]);window.show();owner=window._tab_widgets[0]
started,release,done=Event(),Event(),Event()
class Job:
 def __init__(self):pass
 def cancel(self):pass
 def run(self):
  started.set();release.wait(5);done.set();return {'error':None,'suspicious':[{'img_path':'old'}]}
with patch.object(LossInspectionWorker,'job_type',Job):
 worker=LossInspectionWorker();owner._audit_thread=worker;worker.finished_signal.connect(owner._on_audit_finished);owner._start_audit_worker(worker)
 assert started.wait(2)
 old=owner._dataset_revision;owner._invalidate_analysis_results()
 worker.finished_signal.emit({'error':None,'suspicious':[{'img_path':'stale'}],'_dataset_revision':old})
 assert owner._last_audit_results==[] and owner.btn_run_audit.isEnabled()
 window.close();assert worker.closed and owner._closing
 release.set();assert done.wait(2)
 for _ in range(10):app.processEvents();time.sleep(.01)
 assert owner._last_audit_results==[]
""")

    def test_reentry_and_closing_main_during_scan_are_deferred_safely(self):
        self.run_gui("""
from pathlib import Path
import tempfile,cv2,numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
import sys as _dataset_test_sys
_dataset_test_sys.path.insert(0, str(__import__('pathlib').Path.cwd() / 'tests'))
from dataset_test_support import wait_dataset
from main import ImagePlayer
app=QApplication([]);window=ImagePlayer([('dataset_analysis','Analysis')]);window.show();owner=window._tab_widgets[0]
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'frame.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));image.with_suffix('.txt').write_text('0 .5 .5 .5 .5\\n')
 owner._apply_loaded_pairs([(str(image),str(image.with_suffix('.txt')))],0,'auto',class_names_from_yaml={0:'Object'})
 wait_dataset(owner)
 from threading import Event
 from unittest.mock import patch
 from utils_new.dataset_audit.analysis import DatasetFilterJob
 started,release=Event(),Event()
 original=DatasetFilterJob.run
 def slow(job):
  started.set();release.wait(5);return original(job)
 with patch.object(DatasetFilterJob,'run',slow):
  owner.apply_filters_and_update_metrics()
  assert started.wait(2)
  assert owner.btn_folder.isEnabled() and not owner._scan_busy
  QTimer.singleShot(0,window.close)
  for _ in range(5):app.processEvents()
  assert owner._closing and not window.isVisible()
  release.set()

""")
