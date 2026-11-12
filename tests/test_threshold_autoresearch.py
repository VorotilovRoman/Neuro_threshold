"""Numerical parity, report integrity and GUI isolation of threshold research."""

import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
import cv2, numpy as np
from utils_new.autoreseach_utils.search_space import (
    default_settings,
    combination_count,
    iter_presets,
)
from utils_new.autoreseach_utils.threshold_search import evaluate, run_search
from utils_new.autoreseach_utils.reports import (
    read_report,
    parse_reports,
    preset_from_row,
)
from utils_new.autoreseach_utils.prepare import (
    add_meanap_metrics,
    filter_invalid_params,
    cluster_open_close,
)
from utils_new.autoreseach_utils.analysis import analyze, global_top_diverse

ROOT = Path(__file__).resolve().parents[1]


class SearchTests(unittest.TestCase):
    def settings(self):
        state = default_settings()
        state["methods"]["Simple Threshold"]["ranges"]["threshold"] = [80, 100, 20]
        return state

    def test_lazy_ranges_exact_count_and_invalid_steps(self):
        state = self.settings()
        state["kernel_ranges"]["erode"] = [0, 13, 13]
        state["inversions"] = [False, True]
        self.assertEqual(combination_count(state), 8)
        presets = list(iter_presets(state))
        self.assertEqual(len(presets), 8)
        self.assertEqual(presets[-1]["morphology"]["operations"][2]["kernel_px"], 13)
        state["methods"]["Simple Threshold"]["ranges"]["threshold"][2] = 0
        with self.assertRaises(ValueError):
            combination_count(state)
        state = default_settings()
        state["methods"]["Adaptive Mean"]["enabled"] = True
        state["methods"]["Adaptive Mean"]["ranges"]["window"] = [4, 6, 2]
        with self.assertRaises(ValueError):
            combination_count(state)

    def test_numerical_order_and_all_modes_match_existing_pipeline(self):
        from utils_new.threshold_processing import apply_threshold_method
        from utils_new.morphology import apply_morphology
        from utils_new.segmentation.postprocessing import extract_objects

        gray = np.full((64, 128), 230, np.uint8)
        gray[10:50, 25:100] = 40
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        state = self.settings()
        state["kernel_ranges"]["close"] = [13, 13, 1]
        state["kernel_ranges"]["erode"] = [3, 3, 1]
        state["inversions"] = [False, True]
        for mode in ("detect", "obb", "segment"):
            state["annotation_mode"] = mode
            for preset in iter_presets(state):
                raw, _ = apply_threshold_method(
                    gray, preset["method"], preset["params"]
                )
                expected = apply_morphology(
                    cv2.bitwise_not(raw),
                    preset["morphology"]["operations"],
                    shape=preset["morphology"]["shape"],
                )
                if preset["invert"]:
                    expected = cv2.bitwise_not(expected)
                actual, objects = evaluate(gray, image, preset)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(
                    objects, extract_objects(image, expected, mode, False)[1]
                )

    def test_full_report_analysis_presets_and_cancellation_preserve_old_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            gray = np.full((64, 128), 230, np.uint8)
            gray[10:50, 25:100] = 40
            cv2.imwrite(str(path), gray)
            path.with_suffix(".txt").write_text(
                "0 0.488281 0.468750 0.585938 0.625000\n"
            )
            result = run_search([str(path)], self.settings(), Event())
            self.assertFalse(result["errors"])
            report = Path(result["reports"][0])
            old = report.read_bytes()
            rows = read_report(report)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["ground_truth_available"])
            df = parse_reports([report])
            filtered, _ = filter_invalid_params(df)
            self.assertEqual(len(filtered), 2)
            scored, _ = add_meanap_metrics(df)
            top = analyze(scored, skip_metrics=True, use_diversity=False)
            self.assertEqual(len(top), 2)
            self.assertEqual(
                {preset_from_row(r)["params"]["threshold"] for _, r in top.iterrows()},
                {80, 100},
            )
            self.assertGreater(top.actual_f1.min(), 0.99)
            cancel = Event()
            partial = run_search(
                [str(path)], self.settings(), cancel, lambda *args: cancel.set()
            )
            self.assertTrue(partial["canceled"])
            self.assertEqual(report.read_bytes(), old)
            with self.assertRaises(ValueError):
                read_report(partial["reports"][0])
            self.assertEqual(
                len(read_report(partial["reports"][0], allow_partial=True)), 1
            )

    def test_missing_empty_and_invalid_truth_and_failed_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.png"
            cv2.imwrite(str(path), np.full((64, 128), 230, np.uint8))
            report = run_search([str(path)], self.settings(), Event())["reports"][0]
            df = parse_reports([report])
            self.assertFalse(df.ground_truth_available.any())
            with self.assertRaises(ValueError):
                add_meanap_metrics(df)
            path.with_suffix(".txt").write_text("")
            scored, _ = add_meanap_metrics(parse_reports([report]))
            self.assertTrue((scored.mean_f1 == 1).all())
            path.with_suffix(".txt").write_text("broken\n")
            with self.assertRaises(ValueError):
                read_report(report)
            with patch(
                "utils_new.autoreseach_utils.threshold_search.evaluate",
                side_effect=RuntimeError("compute failed"),
            ):
                failed = run_search([str(path)], self.settings(), Event())
            path.with_suffix(".txt").write_text("")
            self.assertEqual(failed["failed"], 2)
            self.assertEqual(read_report(failed["reports"][0]), [])

    def test_regression_retains_full_presets_and_all_morphology_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            cv2.imwrite(str(path), np.full((64, 128), 230, np.uint8))
            path.with_suffix(".txt").write_text("")
            state = self.settings()
            state["methods"]["Simple Threshold"]["ranges"]["threshold"] = [1, 12, 1]
            report = run_search([str(path)], state, Event())["reports"][0]
            df, _ = add_meanap_metrics(parse_reports([report]))
            df["cluster"] = 0
            df, _ = cluster_open_close(df, n_clusters=2)
            result = analyze(df, skip_metrics=True)
            self.assertEqual(len(result), 5)
            import pandas as pd
            combined = pd.concat([df,df],ignore_index=True)
            global_top = global_top_diverse(combined,k=10,score_col='mean_f1')
            self.assertEqual(len(global_top),10)
            self.assertEqual(global_top.preset_json.nunique(),10)
            for _, row in result.iterrows():
                self.assertEqual(
                    preset_from_row(row)["params"], json.loads(row["params"])
                )


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI opt-in")
class SearchGuiTests(unittest.TestCase):
    def test_dialog_worker_progress_state_preservation_and_report_window(self):
        code = """
import tempfile,time
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QTimer
from tabs.threshold_methods import ThresholdWindow
from tabs.model_research import ModelResearchApp
from ui.autoreseach_ui.settings_dialog import AutoSettingsDialog
from utils_new.autoreseach_utils.search_space import default_settings
app=QApplication([]);owner=ThresholdWindow();owner.show()
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'frame.png';image=np.full((80,160),230,np.uint8);image[20:60,40:120]=40;cv2.imwrite(str(path),image);path.with_suffix('.txt').write_text('0 .5 .5 .5 .5\\n')
 owner._load_images([str(path)]);owner.draw_combo.setCurrentIndex(1);owner.update_timer.stop();owner.display_current_image();owner.objects_widget.focus_object(0)
 owner.settings_file=str(Path(directory)/'settings.json');owner.auto_settings=default_settings();owner.auto_settings['methods']['Simple Threshold']['ranges']['threshold']=[70,90,20]
 dlg=AutoSettingsDialog(owner,owner.auto_settings);assert dlg.get_settings()==owner.auto_settings;dlg.close()
 before=(owner.get_current_threshold_params(),owner.current_index,owner.objects_widget.checked_indices,owner.objects_widget.focused_index,owner.current_base_image.copy())
 ticks=[];timer=QTimer();timer.timeout.connect(lambda:ticks.append(1));timer.start(1)
 with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):owner.run_auto_optimization()
 deadline=time.monotonic()+15
 while owner.auto_search.busy and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
 assert not owner.auto_search.busy and ticks
 after=(owner.get_current_threshold_params(),owner.current_index,owner.objects_widget.checked_indices,owner.objects_widget.focused_index)
 assert before[:4]==after;np.testing.assert_array_equal(before[4],owner.current_base_image)
 report=path.with_name(path.stem+'_full_report.txt');research=ModelResearchApp();df=research.parse_reports([report]);assert len(df)==2
 research.df_full=df;research.display_dataframe(df)
 from utils_new.autoreseach_utils.prepare import add_meanap_metrics
 from utils_new.autoreseach_utils.analysis import analyze
 from ui.segmentation.preset_bindings import ThresholdPresetManager
 import json
 scored,_=add_meanap_metrics(df);top=analyze(scored,skip_metrics=True)
 with patch.object(QMessageBox,'critical') as errors:
  research.run_analysis();errors.assert_not_called()
 assert len(research.results_widget.top5_df)==2
 manager=ThresholdPresetManager(None);manager.presets_file=str(Path(directory)/'presets.json');manager.presets={}
 with patch('tabs.model_research.ThresholdPresetManager',return_value=manager),patch.object(QMessageBox,'information'):
  research.add_to_presets(top)
 saved=json.loads(Path(manager.presets_file).read_text());assert len(saved)==2
 assert {json.dumps(v,sort_keys=True) for v in saved.values()}==set(top.preset_json)
 research.close()
owner.close();app.processEvents()
"""
        with tempfile.TemporaryDirectory() as config:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CACHE_HOME=config,
                XDG_CONFIG_HOME=config,
                MPLCONFIGDIR=config,
                YOLO_CONFIG_DIR=config,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_closing_during_search_ignores_late_completion(self):
        code = """
from threading import Event
from unittest.mock import patch
import time,tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.threshold_methods import ThresholdWindow
app=QApplication([]);window=ThresholdWindow();window.show()
started,release,done=Event(),Event(),Event()
def work(*args,**kwargs):
    started.set();release.wait(5);done.set();return {'reports':[],'errors':[],'canceled':True,'processed':0,'failed':0}
with tempfile.TemporaryDirectory() as directory:
    window.settings_file=str(Path(directory)/'settings.json');window.image_paths=['unused.png']
    with patch('ui.autoreseach_ui.controls.run_search',side_effect=work),patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):
        window.run_auto_optimization();assert started.wait(2)
        window.close();assert window.auto_search.cancel.is_set() and window.auto_search.closed
        release.set();assert done.wait(2)
        for _ in range(10):app.processEvents();time.sleep(.01)
"""
        with tempfile.TemporaryDirectory() as config:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CACHE_HOME=config,
                XDG_CONFIG_HOME=config,
                MPLCONFIGDIR=config,
                YOLO_CONFIG_DIR=config,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
