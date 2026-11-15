"""Dataset-level recommendations, geometry, coverage and explicit reference workflow."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from threading import Event
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import yaml
from utils_new.autoreseach_utils.search_space import default_settings, iter_presets
from utils_new.autoreseach_utils.search import run_search
from utils_new.autoreseach_utils.reports import read_report, parse_reports
from utils_new.autoreseach_utils.ground_truth import (
    load_ground_truth,
    match_sample,
    digest_file,
)
from utils_new.autoreseach_utils.geometry_metrics import (
    normalize_annotations,
    evaluate_image,
    match_overlaps,
)
from utils_new.autoreseach_utils.evaluation import evaluate_dataset
from utils_new.autoreseach_utils.evaluation_export import export_evaluation
from utils_new.autoreseach_utils.analysis import analyze
from utils_new.autoreseach_utils.metrics_core import compute_metrics, parse_objects

ROOT = Path(__file__).resolve().parents[1]
BOX = ("detect", 0, 0.4, 0.4, 0.4, 0.4)


def preset(kind="threshold", value=80, mode="detect"):
    state = default_settings(kind)
    state["annotation_mode"] = mode
    if kind == "threshold":
        state["methods"]["Simple Threshold"]["ranges"]["threshold"] = [value, value, 1]
    return next(iter_presets(state))


class ReportEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "manual"
        self.images = self.dataset / "images"
        self.labels = self.dataset / "labels"
        self.images.mkdir(parents=True)
        self.labels.mkdir()
        self.paths = []
        for index in range(3):
            path = self.images / f"{index}.png"
            image = np.full((100, 100), 230, np.uint8)
            if index < 2:
                image[20:60, 20:60] = 40
            cv2.imwrite(str(path), image)
            (self.labels / f"{index}.txt").write_text(
                "2 .4 .4 .4 .4\n" if index < 2 else ""
            )
            self.paths.append(path)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, index, records, *, folder="reports", resize=True, image_path=None):
        destination = self.root / folder
        destination.mkdir(parents=True, exist_ok=True)
        kind = records[0][0]["kind"]
        path = destination / (
            str(index)
            + ("_full_report.txt" if kind == "threshold" else "_gradient_report.txt")
        )
        header = dict(
            record="header",
            schema_version=1,
            kind=kind + "_search",
            image_path=image_path or str(self.paths[index]),
            source_size=[100, 100],
            working_size=[100, 100],
            source_sha256=digest_file(self.paths[index]),
            processing=dict(resize_enabled=resize, max_side=1024, pipeline_version=1),
            planned=len(records),
        )
        data = [header]
        for state, annotations in records:
            record = dict(
                record="result",
                preset=state,
                status="ok" if annotations is not None else "error",
            )
            if annotations is None:
                record["error"] = "controlled failure"
            else:
                record.update(annotations=annotations, objects=[])
            data.append(record)
        data.append(dict(record="footer", complete=True, processed=len(records)))
        path.write_text("\n".join(json.dumps(value) for value in data) + "\n")
        return path

    def frame(self, paths):
        return parse_reports(
            paths, strict=True, include_failures=True, load_adjacent_truth=False
        )

    def test_complete_recipes_beat_cherry_picked_images_and_failures_are_visible(self):
        good, bad, missing, failed = (
            preset(value=80),
            preset(value=100),
            preset(value=120),
            preset(value=140),
        )
        paths = []
        for index in range(3):
            truth = [BOX] if index < 2 else []
            records = [
                (good, truth),
                (bad, [BOX] if index in (0, 2) else []),
                (failed, None),
            ]
            if index != 1:
                records.append((missing, truth))
            paths.append(self.report(index, records))
            paths.append(
                self.report(index, [(preset("gradient"), truth if index != 1 else [])])
            )
        result = evaluate_dataset(self.frame(paths), self.dataset)
        self.assertEqual(result.reference_images, 3)
        self.assertEqual(result.reference_objects, 2)
        rows = result.rankings.set_index("params")
        best = result.rankings.iloc[0]
        self.assertEqual(json.loads(best.params)["threshold"], 80)
        self.assertEqual(best.actual_f1, 1)
        self.assertEqual((best.tp, best.fp, best.fn), (2, 0, 0))
        bad_row = next(
            row
            for _, row in result.rankings.iterrows()
            if row.kind == "threshold" and json.loads(row.params)["threshold"] == 100
        )
        self.assertEqual(bad_row.micro_f1, 0.5)
        self.assertEqual((bad_row.tp, bad_row.fp, bad_row.fn), (1, 1, 1))
        self.assertEqual(bad_row.negative_accuracy, 0)
        incomplete = result.rankings[~result.rankings.eligible]
        self.assertEqual(len(incomplete), 2)
        self.assertTrue(incomplete.score.isna().all())
        self.assertTrue(incomplete["rank"].isna().all())
        self.assertEqual(len(result.details), 15)
        self.assertEqual(set(result.methods.kind), {"threshold", "gradient"})

    def test_class_semantics_are_explicit_and_missing_reports_do_not_look_like_negatives(
        self,
    ):
        paths = [self.report(i, [(preset(), [BOX] if i < 2 else [])]) for i in range(3)]
        frame = self.frame(paths)
        self.assertEqual(
            evaluate_dataset(frame, self.dataset).rankings.iloc[0].micro_f1, 1
        )
        aware = evaluate_dataset(frame, self.dataset, class_aware=True).rankings.iloc[0]
        self.assertEqual((aware.micro_f1, aware.tp, aware.fp, aware.fn), (0, 0, 2, 2))
        missing = evaluate_dataset(
            frame[frame.image_path != str(self.paths[2])], self.dataset
        )
        self.assertFalse(missing.rankings.eligible.any())
        self.assertTrue(missing.methods.empty)
        self.assertEqual(missing.details.iloc[-1]["status"], "missing")

    def test_real_polygon_obb_geometry_threshold_boundary_and_maximum_matching(self):
        triangle = [("segment", 0, [0.2, 0.2, 0.6, 0.2, 0.2, 0.6])]
        square = [BOX]
        score = evaluate_image(
            normalize_annotations(square, (100, 100)),
            normalize_annotations(triangle, (100, 100)),
            threshold=0.75,
        )
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 1, 1))
        self.assertEqual(
            evaluate_image(
                normalize_annotations(square, (100, 100)),
                normalize_annotations(triangle, (100, 100)),
                threshold=0.5,
            )["tp"],
            1,
        )
        shape = [("obb", 0, [0.4, 0.1, 0.7, 0.4, 0.4, 0.7, 0.1, 0.4])]
        value = evaluate_image(
            normalize_annotations(shape, (200, 100)),
            normalize_annotations(shape, (1000, 500)),
            threshold=0.95,
        )
        self.assertEqual(value["f1"], 1)
        matched = match_overlaps([[0.9, 0.6], [0.8, 0]], 0.5)
        self.assertEqual(matched["tp"], 2)
        self.assertEqual(match_overlaps([[0.5]], 0.5)["tp"], 1)
        self.assertEqual(compute_metrics([(0, 0.4, 0.4, 0.4, 0.4)], []), (0.0, 0, 1, 0))
        with self.assertRaises(ValueError):
            parse_objects("broken")
        concave = [("segment", 0, [0, 0, 1, 0, 1, 0.2, 0.2, 0.2, 0.2, 1, 0, 1])]
        self.assertAlmostEqual(
            evaluate_image(
                normalize_annotations(concave, (10, 10)),
                normalize_annotations(concave, (10, 10)),
            )["f1"],
            1,
        )

    def test_reference_yaml_lists_relocation_duplicates_and_label_only_folder(self):
        selected = self.dataset / "selection.txt"
        selected.write_text("images/0.png\nimages/2.png\n")
        config = self.dataset / "data.yaml"
        config.write_text(
            yaml.safe_dump({"path": str(self.dataset), "train": "selection.txt"})
        )
        truth = load_ground_truth(config)
        self.assertEqual(len(truth.samples), 2)
        self.assertEqual(match_sample(r"C:\old\images\0.png", truth).key, "0.png")
        paths = [self.report(i, [(preset(), [BOX] if i < 2 else [])]) for i in range(3)]
        result = evaluate_dataset(self.frame(paths), self.labels)
        self.assertEqual(result.rankings.iloc[0].micro_f1, 1)
        self.assertTrue(any("только TXT" in value for value in result.diagnostics))
        for split in ("train", "val"):
            image = self.images / split / "same.png"
            image.parent.mkdir()
            image.write_bytes(self.paths[0].read_bytes())
            label = self.labels / split / "same.txt"
            label.parent.mkdir()
            label.write_text("0 .4 .4 .4 .4")
        truth = load_ground_truth(self.dataset)
        self.assertEqual(
            match_sample("C:/copy/images/val/same.png", truth).key, "val/same.png"
        )
        with self.assertRaises(ValueError):
            match_sample("C:/flat/same.png", truth)
        (self.labels / "1.txt").unlink()
        with self.assertRaises(ValueError):
            load_ground_truth(self.dataset)

    def test_duplicate_results_are_idempotent_conflicts_and_corrupt_reports_are_rejected(
        self,
    ):
        paths = [self.report(i, [(preset(), [BOX] if i < 2 else [])]) for i in range(3)]
        clone = self.root / "clone.txt"
        clone.write_bytes(paths[0].read_bytes())
        result = evaluate_dataset(self.frame([*paths, clone]), self.dataset)
        self.assertEqual(result.rankings.iloc[0].tp, 2)
        data = [json.loads(line) for line in clone.read_text().splitlines()]
        data[1]["annotations"] = []
        clone.write_text("\n".join(json.dumps(value) for value in data))
        with self.assertRaises(ValueError):
            evaluate_dataset(self.frame([*paths, clone]), self.dataset)
        clone.write_text(paths[0].read_text().rsplit("\n", 2)[0])
        with self.assertRaises(ValueError):
            self.frame([*paths, clone])
        clone.write_text(paths[0].read_text() + json.dumps(data[1]))
        with self.assertRaises(ValueError):
            read_report(clone, load_adjacent_truth=False)
        self.assertEqual(len(read_report(paths[0], load_adjacent_truth=False)), 1)

    def test_processing_conditions_and_changed_pixels_cannot_share_a_good_score(self):
        paths = []
        for i in range(3):
            paths.append(
                self.report(i, [(preset(), [BOX] if i < 2 else [])], resize=i < 2)
            )
        result = evaluate_dataset(self.frame(paths), self.dataset)
        self.assertEqual(len(result.rankings), 2)
        self.assertFalse(result.rankings.eligible.any())
        for i in range(3):
            paths[i] = self.report(i, [(preset(), [BOX] if i < 2 else [])])
        frame = self.frame(paths)
        cv2.imwrite(str(self.paths[0]), np.zeros((100, 100), np.uint8))
        result = evaluate_dataset(frame, self.dataset)
        self.assertEqual(result.rankings.iloc[0].failed_images, 1)
        self.assertIn("Содержимое", result.details.iloc[0]["error"])

    def test_no_in_sample_regression_can_override_measured_scores(self):
        paths = [
            self.report(
                i, [(preset(value=80), [BOX] if i < 2 else []), (preset(value=100), [])]
            )
            for i in range(3)
        ]
        frame = self.frame(paths)
        # This legacy exploratory helper must also remain empirical with >=10 rows.
        scored = pd.concat([frame, frame], ignore_index=True)
        scored["mean_f1"] = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0] * 2
        scored["avg_iou_tp"] = 1.0
        top = analyze(scored, skip_metrics=True, use_diversity=False, per_method=False)
        self.assertAlmostEqual(top.iloc[0].actual_f1, 1.0)
        np.testing.assert_allclose(top.pred_f1, top.actual_f1)

    def test_safe_export_full_details_and_cooperative_cancellation(self):
        paths = [self.report(i, [(preset(), [BOX] if i < 2 else [])]) for i in range(3)]
        frame = self.frame(paths)
        result = evaluate_dataset(frame, self.dataset)
        output = self.root / "evaluation"
        export_evaluation(result, output)
        metadata = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(metadata["images"], 3)
        self.assertEqual(len(pd.read_csv(output / "image_details.csv")), 3)
        self.assertIn("Лучший проверенный пресет", (output / "summary.md").read_text())
        with self.assertRaises(ValueError):
            export_evaluation(result, output)
        with patch(
            "utils_new.autoreseach_utils.evaluation_export.os.replace",
            side_effect=OSError("disk"),
        ):
            with self.assertRaises(OSError):
                export_evaluation(result, self.root / "failed")
        self.assertFalse((self.root / "failed").exists())
        self.assertFalse(list(self.root.glob(".report-evaluation-*")))
        cancelled = Event()
        with self.assertRaises(InterruptedError):
            evaluate_dataset(
                frame,
                self.dataset,
                progress=lambda *args: cancelled.set(),
                cancel_check=cancelled.is_set,
            )

    def test_real_search_both_kinds_to_external_manual_reference_and_presets(self):
        paths = []
        for kind in ("threshold", "gradient"):
            state = default_settings(kind)
            if kind == "threshold":
                state["methods"]["Simple Threshold"]["ranges"]["threshold"] = [
                    80,
                    100,
                    20,
                ]
            else:
                state["methods"]["Sobel"]["ranges"]["ksize"] = [3, 3, 2]
                state["methods"]["Sobel"]["ranges"]["scale"] = [1, 1, 1]
            output = run_search(list(map(str, self.paths)), state, Event())
            self.assertFalse(output["errors"])
            paths.extend(output["reports"])
        frame = self.frame(paths)
        result = evaluate_dataset(frame, self.dataset)
        self.assertEqual(set(result.methods.kind), {"threshold", "gradient"})
        self.assertTrue(result.rankings.eligible.all())
        self.assertEqual(result.rankings.iloc[0].micro_f1, 1)
        for state in result.rankings.preset_json.map(json.loads):
            self.assertEqual(state["preprocessing"]["max_side"], 1024)


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI opt-in")
class ReportEvaluationGuiTests(unittest.TestCase):
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

    def test_explicit_reference_mixed_reports_immutable_source_and_export(self):
        self.run_gui("""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
import time,cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from tabs.model_research import ModelResearchApp
from utils_new.autoreseach_utils.search import run_search
from utils_new.autoreseach_utils.search_space import default_settings
app=QApplication([]);w=ModelResearchApp();w.show()
with TemporaryDirectory() as directory:
 root=Path(directory);images=root/'images';labels=root/'labels';images.mkdir();labels.mkdir()
 image=images/'sample.png';pixels=np.full((100,100),230,np.uint8);pixels[20:60,20:60]=40;cv2.imwrite(str(image),pixels);(labels/'sample.txt').write_text('0 .4 .4 .4 .4')
 w.set_ground_truth_source(root)
 reports=[]
 for kind in ('threshold','gradient'):
  state=default_settings(kind)
  if kind=='threshold':state['methods']['Simple Threshold']['ranges']['threshold']=[80,100,20]
  else:
   state['methods']['Sobel']['ranges']['ksize']=[3,3,2];state['methods']['Sobel']['ranges']['scale']=[1,1,1]
  reports.extend(run_search([str(image)],state,Event())['reports'])
 assert w._load_report_paths(reports[:1]);assert w._load_report_paths(reports[1:])
 total=len(w._report_source);w.df_full=w.df_full.iloc[:0]
 w.run_analysis();deadline=time.monotonic()+10
 while w.evaluation_controller.busy and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
 assert not w.evaluation_controller.busy and w._evaluation_result is not None
 result=w._evaluation_result;assert len(result.rankings)==total and set(result.methods.kind)=={'threshold','gradient'}
 assert result.rankings.eligible.all() and w.results_widget.rank_table.isVisible()
 output=root/'evaluation';output.mkdir()
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(output)),patch.object(QMessageBox,'information'):w.save_evaluation()
 assert (output/'rankings.csv').is_file()
 with patch.object(w,'add_to_presets') as save:w.results_widget.add_btn.click();assert len(save.call_args.args[0])==1
 w.evaluation_iou.setValue(.75);assert w._evaluation_result is None and not w.btn_export_evaluation.isEnabled()
 w.clear_report_inputs();assert w._report_source is None
w.close();app.processEvents()
""")

    def test_resize_condition_is_restored_before_applying_exported_recipe(self):
        self.run_gui("""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from main import ImagePlayer
from utils_new.autoreseach_utils.search import run_search,evaluate
from utils_new.autoreseach_utils.search_space import default_settings
from utils_new.autoreseach_utils.reports import parse_reports,preset_from_row
app=QApplication([]);w=ImagePlayer([('threshold','Threshold'),('gradient','Gradient')]);w.show()
with TemporaryDirectory() as directory:
 image=Path(directory)/'large.png';pixels=np.full((700,1400),230,np.uint8);pixels[100:400,200:800]=40;cv2.imwrite(str(image),pixels)
 for owner,kind in zip(w._tab_widgets,('threshold','gradient')):
  owner._load_images([str(image)]);owner.update_timer.stop();assert owner.current_gray.shape[1]==1024
  state=default_settings(kind)
  if kind=='threshold':state['methods']['Simple Threshold']['ranges']['threshold']=[80,80,1]
  else:
   state['methods']['Sobel']['ranges']['ksize']=[3,3,2];state['methods']['Sobel']['ranges']['scale']=[1,1,1]
  reports=run_search([str(image)],state,Event(),resize_enabled=False)['reports'];row=parse_reports(reports,load_adjacent_truth=False).iloc[0];recipe=preset_from_row(row)
  owner.preset_manager.presets={'raw':recipe};owner.preset_manager.apply_preset('raw');owner.update_timer.stop();owner.display_current_image()
  assert not owner.nav_widget.is_resize_enabled() and owner.current_gray.shape[1]==1400
  np.testing.assert_array_equal(owner.final_mask,evaluate(owner.current_gray,owner.original_image,recipe)[0])
w.close();app.processEvents()
""")

    def test_input_change_and_main_close_cancel_and_ignore_background_reply(self):
        self.run_gui("""
from threading import Event
from unittest.mock import patch
import time,pandas as pd
from PyQt5.QtWidgets import QApplication
from main import ImagePlayer
from utils_new.autoreseach_utils.search_space import default_settings,iter_presets
import json
app=QApplication([]);w=ImagePlayer([('reports','Reports')]);w.show();owner=w._tab_widgets[0]
started,release=Event(),Event()
def blocked(*args,**kwargs):started.set();release.wait(5);raise ValueError('obsolete')
frame=pd.DataFrame([dict(preset_json=json.dumps(next(iter_presets(default_settings()))),source_size='[10,10]',working_size='[10,10]',annotations_json='[]',image_path='x.png')])
owner.df_full=frame;owner.set_ground_truth_source('/unneeded')
with patch('ui.autoreseach_ui.evaluation_controller.evaluate_dataset',side_effect=blocked):
 owner.run_analysis();assert started.wait(2)
 owner.set_ground_truth_source('/new-input');assert owner._evaluation_result is None
 w.close();assert owner.evaluation_controller.closed
 release.set()
 for _ in range(10):app.processEvents();time.sleep(.01)
 assert owner._evaluation_result is None
""")
