"""Guided report workflow, removed tools, responsive IO and read-only result views."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event
import unittest
import numpy as np
import test_report_evaluation as fixtures
from utils_new.autoreseach_utils.workflow import (
    discover_reports,
    load_collection,
    input_guidance,
    result_guidance,
    selected_presets,
)
from utils_new.autoreseach_utils.reports import read_report
from utils_new.autoreseach_utils.evaluation import evaluate_dataset
from utils_new.autoreseach_utils.ground_truth import load_ground_truth
from utils_new.annotations.boxes import yolo_to_xyxy, xyxy_to_yolo, iou_xyxy
from utils.ensemble_utils import non_max_suppression

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReportEvaluationTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def reports(self):
        return [
            self.fixture.report(
                i, [(fixtures.preset(), [fixtures.BOX] if i < 2 else [])]
            )
            for i in range(3)
        ]

    def test_ready_states_counts_discovery_and_cancellation(self):
        self.assertFalse(input_guidance(None, None)[0])
        self.assertIn("Шаг 1", input_guidance(None, None)[1])
        self.assertIn("Шаг 2", input_guidance(self.fixture.dataset, None)[1])
        paths = self.reports()
        collection = load_collection([*paths, paths[0]])
        self.assertEqual(
            (
                len(collection.paths),
                collection.image_count,
                collection.threshold_files,
                collection.gradient_files,
            ),
            (3, 3, 3, 0),
        )
        self.assertTrue(input_guidance(self.fixture.dataset, collection)[0])
        self.assertFalse(input_guidance(self.fixture.root / "absent", collection)[0])
        self.assertEqual(
            discover_reports(self.fixture.root / "reports"), list(map(str, paths))
        )
        with self.assertRaises(InterruptedError):
            load_collection(paths, cancel_check=lambda: True)
        with self.assertRaises(InterruptedError):
            read_report(paths[0], cancel_check=lambda: True)
        stopped = Event()
        with self.assertRaises(InterruptedError):
            load_collection(
                paths, cancel_check=stopped.is_set, progress=lambda *args: stopped.set()
            )
        (self.fixture.root / "broken_full_report.txt").write_text("bad report")
        with self.assertRaises(ValueError):
            load_collection([*paths, self.fixture.root / "broken_full_report.txt"])

    def test_negative_only_reference_is_not_a_method_recommendation(self):
        path = self.fixture.report(2, [(fixtures.preset(), [])])
        collection = load_collection([path])
        truth = load_ground_truth(self.fixture.dataset)
        negative = type(truth)((truth.samples[2],), truth.source)
        result = evaluate_dataset(collection.frame, negative)
        self.assertTrue(result.rankings.eligible.all())
        self.assertFalse(result.rankings.recommendable.any())
        self.assertTrue(result.rankings.score.isna().all())
        self.assertTrue(result.rankings["rank"].isna().all())
        self.assertTrue(result.methods.empty)
        self.assertIn("нужны снимки с объектами", result_guidance(result)[0])
        with self.assertRaises(ValueError):
            selected_presets(result.rankings)

    def test_missing_and_zero_quality_have_actionable_explanations(self):
        partial = load_collection(self.reports()[:1])
        result = evaluate_dataset(partial.frame, self.fixture.dataset)
        self.assertIn("Победитель пока не определён", result_guidance(result)[0])
        self.assertIn("одинаковыми диапазонами", result_guidance(result)[2])
        failed = load_collection(
            [self.fixture.report(i, [(fixtures.preset(), [])]) for i in range(3)]
        )
        result = evaluate_dataset(failed.frame, self.fixture.dataset)
        self.assertIn("Подходящие настройки не найдены", result_guidance(result)[0])
        self.assertIn("инверсию", result_guidance(result)[2])

    def test_only_complete_recipes_are_persistable_and_duplicates_are_not_removed_as_failures(
        self,
    ):
        collection = load_collection(self.reports())
        result = evaluate_dataset(collection.frame, self.fixture.dataset)
        states = selected_presets(result.methods)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["params"]["threshold"], 80)
        frame = result.rankings.copy()
        frame["eligible"] = False
        frame["recommendable"] = False
        with self.assertRaises(ValueError):
            selected_presets(frame)
        failed = self.fixture.report(0, [(fixtures.preset(), None)], folder="failed")
        self.assertEqual(load_collection([failed]).failed_results, 1)
        self.assertEqual(read_report(failed)[0]["status"], "error")

    def test_ensemble_box_geometry_is_retained_and_nms_still_suppresses_duplicate_objects(
        self,
    ):
        coords = yolo_to_xyxy(0.5, 0.5, 0.4, 0.2)
        np.testing.assert_allclose(coords, [0.3, 0.4, 0.7, 0.6])
        np.testing.assert_allclose(xyxy_to_yolo(*coords), [0.5, 0.5, 0.4, 0.2])
        self.assertAlmostEqual(iou_xyxy(coords, coords), 1.0)
        self.assertEqual(iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)
        boxes, scores = non_max_suppression(
            [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8, 0.7]
        )
        self.assertEqual(boxes, [[0, 0, 10, 10], [20, 20, 30, 30]])
        self.assertEqual(scores, [0.9, 0.7])

    def test_removed_analysis_modules_have_no_imports_or_registry_exports(self):
        import import_libs_internal

        removed = {
            "utils_new.autoreseach_utils." + name
            for name in ("analysis", "prepare", "metrics_core")
        }
        for module in removed:
            self.assertFalse(
                (ROOT / Path(*module.split("."))).with_suffix(".py").exists()
            )
        for module, _ in import_libs_internal._EXPORTS.values():
            self.assertNotIn(module, removed)
        for folder in ("tabs", "ui", "utils", "utils_new"):
            for path in (ROOT / folder).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, removed, str(path))
                    if isinstance(node, ast.Import):
                        self.assertTrue(
                            all(alias.name not in removed for alias in node.names),
                            str(path),
                        )


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI opt-in")
class WorkflowGuiTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReportEvaluationTests()
        self.fixture.setUp()
        self.paths = []
        for i in range(3):
            records = [
                (fixtures.preset(value=80), [fixtures.BOX] if i < 2 else []),
                (fixtures.preset(value=100), []),
            ]
            if i == 0:
                records.append((fixtures.preset(value=120), [fixtures.BOX]))
            self.paths.append(self.fixture.report(i, records))
            self.paths.append(
                self.fixture.report(
                    i, [(fixtures.preset("gradient"), [fixtures.BOX] if i < 2 else [])]
                )
            )

    def tearDown(self):
        self.fixture.tearDown()

    def run_gui(self, code):
        code = code.replace("REPORT_PATHS", repr(list(map(str, self.paths)))).replace(
            "REFERENCE_PATH", repr(str(self.fixture.dataset))
        )
        with tempfile.TemporaryDirectory() as config:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                NO_ALBUMENTATIONS_UPDATE="1",
                XDG_CACHE_HOME=config,
                XDG_CONFIG_HOME=config,
                MPLCONFIGDIR=config,
                YOLO_CONFIG_DIR=config,
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

    def test_guidance_empty_inputs_async_load_cancel_and_inline_errors(self):
        self.run_gui("""
from threading import Event
from unittest.mock import patch
import time
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from tabs.model_research import ModelResearchApp
from utils_new.autoreseach_utils.workflow import load_collection
app=QApplication([]);w=ModelResearchApp();w.show()
assert not w.btn_analyze.isEnabled() and 'Шаг 1' in w.lbl_evaluation_status.text()
w.btn_help.click();assert w.btn_help.isChecked()
w.set_ground_truth_source(REFERENCE_PATH);assert not w.btn_analyze.isEnabled() and 'Шаг 2' in w.lbl_evaluation_status.text()
started,release=Event(),Event();ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(1)
def loading(*args,**kwargs):started.set();release.wait(5);return load_collection(*args,**kwargs)
with patch('ui.autoreseach_ui.evaluation_controller.load_collection',side_effect=loading):
 assert w._load_report_paths(REPORT_PATHS);assert started.wait(2)
 assert w.btn_stop_evaluation.isEnabled() and not w.btn_analyze.isEnabled()
 until=time.monotonic()+.05
 while time.monotonic()<until:app.processEvents();time.sleep(.002)
 assert ticks
 w.clear_report_inputs();release.set();until=time.monotonic()+5
 while w.evaluation_controller.busy and time.monotonic()<until:app.processEvents();time.sleep(.002)
 assert w._report_source is None and not w.btn_analyze.isEnabled()
assert w._load_report_paths(REPORT_PATHS)
until=time.monotonic()+10
while w.evaluation_controller.busy and time.monotonic()<until:app.processEvents();time.sleep(.002)
assert w.btn_analyze.isEnabled();previous=w._collection
assert w._load_report_paths(['/not/a/valid/report.txt'])
until=time.monotonic()+10
while w.evaluation_controller.busy and time.monotonic()<until:app.processEvents();time.sleep(.002)
assert w._collection is previous and 'Что сделать' in w.lbl_evaluation_status.text()
assert not hasattr(w,'btn_cluster') and not hasattr(w,'df_full')
w.close();app.processEvents()
""")

    def test_readonly_views_sorting_presets_and_per_image_problem_filters(self):
        self.run_gui("""
import time,json,pandas as pd
from unittest.mock import patch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from tabs.model_research import ModelResearchApp
from utils_new.autoreseach_utils.evaluation import evaluate_dataset
app=QApplication([]);w=ModelResearchApp();w.show();w.set_ground_truth_source(REFERENCE_PATH);w._load_report_paths(REPORT_PATHS)
def wait():
 until=time.monotonic()+10
 while w.evaluation_controller.busy and time.monotonic()<until:app.processEvents();time.sleep(.002)
 assert not w.evaluation_controller.busy
wait()
with patch('ui.autoreseach_ui.evaluation_controller.evaluate_dataset',wraps=evaluate_dataset) as evaluate:
 assert w.run_analysis();wait();result=w._evaluation_result;assert result is not None
 app.processEvents()
 view=w.results_widget;original=result.rankings.copy(deep=True)
 viewport=w.centralWidget().viewport()
 assert viewport.rect().contains(view.add_btn.mapTo(viewport,view.add_btn.rect().center()))
 assert view.methods_table.source.rowCount()==2
 incomplete=result.rankings[~result.rankings.eligible].iloc[0]
 view.tabs.setCurrentIndex(1);assert view.rank_table.select_recipe(incomplete.recipe_key)
 assert not view.add_btn.isEnabled() and 'Неполное' in view.properties.toPlainText()
 view.tabs.setCurrentIndex(2);view.problem_filter.setCurrentIndex(4)
 assert len(view.image_table.source.frame)==2 and set(view.image_table.source.frame.status)=={'missing'}
 bad=result.rankings[(result.rankings.kind=='threshold')&result.rankings.params.map(lambda value:json.loads(value).get('threshold')==100)].iloc[0]
 view.tabs.setCurrentIndex(1);view.rank_table.sortByColumn(3,Qt.DescendingOrder);view.rank_table.select_recipe(bad.recipe_key)
 assert 'Порог: 100' in view.properties.toPlainText() and view.add_btn.isEnabled()
 with patch.object(w,'add_to_presets') as save:
  view.add_btn.click();assert save.call_args.args[0].iloc[0].recipe_key==bad.recipe_key
 view.tabs.setCurrentIndex(2);view.problem_filter.setCurrentIndex(3)
 assert len(view.image_table.source.frame)==2 and (view.image_table.source.frame.fn>0).all()
 view.kind_filter.setCurrentIndex(1);assert set(view.rank_table.source.frame.kind)=={'threshold'}
 pd.testing.assert_frame_equal(result.rankings,original);assert evaluate.call_count==1
w.close();app.processEvents()
""")
