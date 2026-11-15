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
from utils_new.autoreseach_utils.search import evaluate, run_search
from utils_new.autoreseach_utils.reports import (
    read_report,
    parse_reports,
    preset_from_row,
)
from utils_new.autoreseach_utils.evaluation import evaluate_dataset


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
            self.assertEqual(rows[0]["status"], "ok")
            df = parse_reports([report])
            top = evaluate_dataset(df, directory).rankings
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

    def test_explicit_truth_negatives_and_failed_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.png"
            cv2.imwrite(str(path), np.full((64, 128), 230, np.uint8))
            report = run_search([str(path)], self.settings(), Event())["reports"][0]
            frame = parse_reports([report])
            with self.assertRaises(ValueError):
                evaluate_dataset(frame, directory)
            path.with_suffix(".txt").write_text("")
            result = evaluate_dataset(frame, directory)
            self.assertTrue((result.rankings.micro_f1 == 1).all())
            self.assertTrue(result.methods.empty)
            self.assertTrue(result.rankings["rank"].isna().all())
            path.with_suffix(".txt").write_text("broken")
            with self.assertRaises(ValueError):
                evaluate_dataset(frame, directory)
            # Loading predictions is independent of an unrelated adjacent TXT.
            self.assertEqual(len(read_report(report)), 2)
            with patch(
                "utils_new.autoreseach_utils.search.evaluate",
                side_effect=RuntimeError("compute failed"),
            ):
                failed = run_search([str(path)], self.settings(), Event())
            rows = read_report(failed["reports"][0])
            self.assertEqual(failed["failed"], 2)
            self.assertEqual([row["status"] for row in rows], ["error", "error"])

    def test_exact_recipes_keep_all_parameters_and_duplicate_inputs_do_not_change_quality(
        self,
    ):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            image = np.full((64, 128), 230, np.uint8)
            image[16:48, 32:96] = 0
            cv2.imwrite(str(path), image)
            path.with_suffix(".txt").write_text("0 .5 .5 .5 .5")
            state = self.settings()
            state["methods"]["Simple Threshold"]["ranges"]["threshold"] = [1, 12, 1]
            report = run_search([str(path)], state, Event())["reports"][0]
            frame = parse_reports([report])
            result = evaluate_dataset(frame, directory).rankings
            combined = evaluate_dataset(
                pd.concat([frame, frame], ignore_index=True), directory
            ).rankings
            self.assertEqual(len(result), 12)
            self.assertEqual(len(combined), 12)
            self.assertEqual(result.preset_json.nunique(), 12)
            for _, row in result.iterrows():
                self.assertEqual(
                    preset_from_row(row)["params"], json.loads(row["params"])
                )
                self.assertTrue(
                    all(
                        op + "_px" in row for op in ("close", "open", "erode", "dilate")
                    )
                )
            pd.testing.assert_series_equal(result.score, combined.score)


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
 from tabs.gradient_methods import GradientMethodsWindow
 from ui.autoreseach_ui.controls import AutoResearchPanel
 gradient=GradientMethodsWindow();gradient.settings_file=str(Path(directory)/'gradient-settings.json')
 assert type(owner.auto_research_panel) is AutoResearchPanel and type(gradient.auto_research_panel) is AutoResearchPanel
 assert not hasattr(owner,'auto_enable_checkbox') and not hasattr(gradient,'auto_enable_checkbox')
 for attr in ('auto_button','configure_auto_button','search_all_checkbox'):
  assert getattr(owner,attr).text()==getattr(gradient,attr).text()
  assert getattr(owner,attr).isEnabled() and getattr(gradient,attr).isEnabled()
 gradient.close()
 with patch('tabs.threshold_methods.AutoSettingsDialog') as settings_dialog:
  settings_dialog.return_value.exec_.return_value=0
  owner.configure_auto_button.click();settings_dialog.assert_called_once_with(owner,owner.auto_settings)
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
 report=path.with_name(path.stem+'_full_report.txt');research=ModelResearchApp();research.set_ground_truth_source(directory)
 assert research._load_report_paths([report])
 deadline=time.monotonic()+10
 while research.evaluation_controller.busy and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
 assert len(research._report_source)==2
 from ui.segmentation.preset_bindings import ThresholdPresetManager
 import json
 with patch.object(QMessageBox,'critical') as errors:
  research.run_analysis();deadline=time.monotonic()+10
  while research.evaluation_controller.busy and time.monotonic()<deadline:app.processEvents();time.sleep(.002)
  assert not research.evaluation_controller.busy;errors.assert_not_called()
 top=research._evaluation_result.rankings;assert len(top)==2
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
