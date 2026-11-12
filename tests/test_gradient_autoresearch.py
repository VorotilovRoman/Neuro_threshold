"""Gradient research parity, finite search spaces and mixed report/preset routing."""

import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch
import cv2, numpy as np
from utils_new.autoreseach_utils.search_space import (
    default_settings,
    combination_count,
    iter_presets,
    method_params,
)
from utils_new.autoreseach_utils.search import evaluate, run_search
from utils_new.autoreseach_utils.reports import (
    parse_reports,
    preset_from_row,
    read_report,
)
from utils_new.autoreseach_utils.prepare import (
    add_meanap_metrics,
    filter_invalid_params,
    add_cluster_labels,
    cluster_open_close,
)
from utils_new.autoreseach_utils.analysis import analyze, global_top_diverse

ROOT = Path(__file__).resolve().parents[1]


class GradientSearchTests(unittest.TestCase):
    def test_canny_count_excludes_invalid_pairs_and_preserves_domains(self):
        threshold = default_settings()
        threshold["methods"]["Niblack"]["enabled"] = True
        threshold["methods"]["Niblack"]["ranges"]["k"] = [0.001, 0.001, 0.01]
        with self.assertRaises(ValueError):
            combination_count(threshold)
        threshold["methods"]["Niblack"]["ranges"]["k"] = [0.12, 0.12, 0.01]
        threshold["methods"]["Niblack"]["ranges"]["window"] = [203, 203, 2]
        with self.assertRaises(ValueError):
            combination_count(threshold)
        state = default_settings("gradient")
        for value in state["methods"].values():
            value["enabled"] = False
        canny = state["methods"]["Canny"]
        canny["enabled"] = True
        canny["ranges"] = {
            "threshold1": [0, 10, 10],
            "threshold2": [0, 10, 10],
            "aperture": [3, 7, 2],
        }
        self.assertEqual(combination_count(state), 9)
        presets = list(iter_presets(state))
        self.assertEqual(len(presets), 9)
        self.assertTrue(all(p["canny_thresh1"] <= p["canny_thresh2"] for p in presets))
        canny["ranges"]["threshold1"] = [20, 20, 1]
        self.assertEqual(combination_count(state), 0)
        canny["ranges"]["aperture"] = [4, 6, 2]
        with self.assertRaises(ValueError):
            combination_count(state)

    def test_seven_methods_match_manual_live_pipeline_with_fill_filter_invert(self):
        from utils_ops.gradient_ops import apply_gradient_method
        from utils_new.morphology import apply_morphology

        gray = np.full((64, 96), 220, np.uint8)
        gray[8:50, 20:80] = 20
        gray[24:32, 40:48] = 220
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for method in default_settings("gradient")["methods"]:
            state = default_settings("gradient")
            for key, value in state["methods"].items():
                value["enabled"] = key == method
                for param, values in value["ranges"].items():
                    values[1] = values[0]
            state["inversions"] = [False, True]
            state["fill_contours"] = [False, True]
            state["post_ranges"]["min_area_percent"] = [0, 0.5, 0.5]
            state["kernel_ranges"]["close"] = [13, 13, 1]
            for preset in iter_presets(state):
                gradient = apply_gradient_method(
                    cv2.bitwise_not(gray), method, method_params(preset)
                )
                _, expected = cv2.threshold(
                    gradient, preset["threshold"], 255, cv2.THRESH_BINARY
                )
                expected = apply_morphology(
                    expected,
                    preset["morphology"]["operations"],
                    shape=preset["morphology"]["shape"],
                )
                contours, _ = cv2.findContours(
                    expected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if (
                        cv2.contourArea(contour)
                        < preset["min_area_percent"] / 100 * gray.size
                    ):
                        cv2.drawContours(expected, [contour], -1, 0, cv2.FILLED)
                if preset["fill_contours"]:
                    contours, _ = cv2.findContours(
                        expected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    expected = np.zeros_like(gray)
                    cv2.drawContours(expected, contours, -1, 255, cv2.FILLED)
                if preset["invert"]:
                    expected = cv2.bitwise_not(expected)
                with self.subTest(
                    method=method, fill=preset["fill_contours"], invert=preset["invert"]
                ):
                    np.testing.assert_array_equal(
                        evaluate(gray, image, preset)[0], expected
                    )

    def test_postprocessing_variants_reuse_the_numerical_base(self):
        from utils_new.autoreseach_utils.search import RawMaskCache
        from utils_new.autoreseach_utils import gradient_processing

        gray = np.full((32, 48), 210, np.uint8)
        gray[8:24, 12:36] = 20
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for kind in ("threshold", "gradient"):
            state = default_settings(kind)
            for method in state["methods"].values():
                for values in method["ranges"].values():
                    values[1] = values[0]
            state["kernel_ranges"]["close"] = [0, 13, 13]
            state["inversions"] = [False, True]
            if kind == "gradient":
                state["post_ranges"]["threshold"] = [50, 100, 50]
            cache = RawMaskCache(gray)
            target = (
                "utils_new.autoreseach_utils.search.apply_threshold_method"
                if kind == "threshold"
                else "utils_new.autoreseach_utils.gradient_processing.apply_gradient_method"
            )
            from utils_new.autoreseach_utils import search

            function = (
                search.apply_threshold_method
                if kind == "threshold"
                else gradient_processing.apply_gradient_method
            )
            with patch(target, wraps=function) as operation:
                outputs = [
                    evaluate(gray, image, p, raw_mask=cache.get(p))[0]
                    for p in iter_presets(state)
                ]
                self.assertEqual(operation.call_count, 1)
            for result, preset in zip(outputs, iter_presets(state)):
                np.testing.assert_array_equal(result, evaluate(gray, image, preset)[0])

    def test_mixed_reports_ranking_clustering_cancellation_and_preset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            image = np.full((64, 96), 220, np.uint8)
            image[8:50, 20:80] = 20
            cv2.imwrite(str(path), image)
            path.with_suffix(".txt").write_text("0 .52 .45 .6 .65\n")
            settings = default_settings()
            settings["methods"]["Simple Threshold"]["ranges"]["threshold"] = [
                80,
                100,
                20,
            ]
            threshold = run_search([str(path)], settings, Event())["reports"][0]
            state = default_settings("gradient")
            gradient = run_search([str(path)], state, Event())["reports"][0]
            self.assertTrue(gradient.endswith("_gradient_report.txt"))
            self.assertNotEqual(gradient, threshold)
            df = parse_reports([threshold, gradient])
            self.assertEqual(set(df.kind), {"threshold", "gradient"})
            filtered, _ = filter_invalid_params(df)
            self.assertEqual(len(df), len(filtered))
            scored, _ = add_meanap_metrics(df)
            scored, _ = add_cluster_labels(scored, output_dir=directory)
            scored, _ = cluster_open_close(scored, n_clusters=2)
            top = analyze(scored, skip_metrics=True)
            self.assertEqual(set(top.kind), {"threshold", "gradient"})
            for _, row in top.iterrows():
                self.assertEqual(preset_from_row(row)["kind"], row.kind)
            self.assertTrue(
                all(
                    "threshold" in json.loads(v)
                    for v in scored.loc[scored.kind == "gradient", "params"]
                )
            )
            global_top = global_top_diverse(scored, k=20, score_col="mean_f1")
            self.assertEqual(global_top.preset_json.nunique(), len(scored))
            old = Path(gradient).read_bytes()
            cancel = Event()
            result = run_search([str(path)], state, cancel, lambda *args: cancel.set())
            self.assertTrue(result["canceled"])
            self.assertEqual(Path(gradient).read_bytes(), old)
            with self.assertRaises(ValueError):
                read_report(result["reports"][0])
            with patch(
                "utils_new.autoreseach_utils.search.evaluate",
                side_effect=RuntimeError("failed"),
            ):
                result = run_search([str(path)], state, Event())
                self.assertEqual(result["failed"], 9)
            self.assertEqual(read_report(result["reports"][0]), [])


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI opt-in")
class UnifiedSearchGuiTests(unittest.TestCase):
    def test_both_tabs_live_parity_state_and_mixed_presets_go_to_correct_stores(self):
        code = """
import json,tempfile,time
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from main import ImagePlayer
from ui.autoreseach_ui.settings_dialog import AutoSettingsDialog
from utils_new.autoreseach_utils.search_space import default_settings,iter_presets
from utils_new.autoreseach_utils.search import evaluate
from utils_new.autoreseach_utils.prepare import add_meanap_metrics
from utils_new.autoreseach_utils.analysis import analyze
from tabs.model_research import ModelResearchApp
from ui.segmentation.preset_bindings import ThresholdPresetManager,GradientPresetManager
app=QApplication([]);w=ImagePlayer([('threshold','Threshold'),('gradient','Gradient')]);w.show()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);path=root/'frame.png';image=np.full((64,96),220,np.uint8);image[8:50,20:80]=20;cv2.imwrite(str(path),image);path.with_suffix('.txt').write_text('0 .52 .45 .6 .65\\n')
 for owner,kind in zip(w._tab_widgets,('threshold','gradient')):
  owner.settings_file=str(root/(kind+'.json'));owner._load_images([str(path)]);owner.update_timer.stop()
  state=default_settings(kind);owner.auto_settings=state
  if kind=='threshold':state['methods']['Simple Threshold']['ranges']['threshold']=[80,80,1]
  else:
   state['methods']['Sobel']['enabled']=False;state['methods']['Canny']['enabled']=True
   state['methods']['Canny']['ranges']['threshold1']=[70,70,1];state['methods']['Canny']['ranges']['threshold2']=[180,180,1]
   state['post_ranges']['min_area_percent']=[.5,.5,.5]
  dialog=AutoSettingsDialog(owner,state);assert dialog.get_settings()==state;dialog.close()
  assert owner.auto_button.isEnabled() and owner.configure_auto_button.isEnabled()
  preset=next(iter_presets(state));owner.preset_manager.presets={'check':preset};assert owner.preset_manager.apply_preset('check');owner.update_timer.stop();owner.display_current_image()
  np.testing.assert_array_equal(owner.final_mask,evaluate(owner.current_gray,owner.original_image,preset)[0])
  if kind=='gradient':assert owner.canny_label1.text()=='70' and owner.canny_label2.text()=='180' and owner.min_area_spinbox.value()==.5
  owner.objects_widget.set_checked_indices([]);owner.objects_widget.focus_object(-1)
  before=(owner.current_base_image.copy(),owner.final_mask.copy(),owner.current_index,owner.draw_combo.currentData(),owner.objects_widget.checked_indices)
  with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):owner.run_auto_optimization()
  until=time.monotonic()+15
  while owner.auto_search.busy and time.monotonic()<until:app.processEvents();time.sleep(.002)
  assert not owner.auto_search.busy
  np.testing.assert_array_equal(owner.current_base_image,before[0]);np.testing.assert_array_equal(owner.final_mask,before[1]);assert (owner.current_index,owner.draw_combo.currentData(),owner.objects_widget.checked_indices)==before[2:]
 research=ModelResearchApp();df=research.parse_reports(list(root.glob('*_report.txt')));assert set(df.kind)=={'threshold','gradient'}
 research.df_full=df
 with patch.object(QMessageBox,'critical') as error:research.run_analysis();error.assert_not_called()
 top=research.results_widget.top5_df;assert set(top.kind)=={'threshold','gradient'}
 managers={}
 for kind,cls in [('threshold',ThresholdPresetManager),('gradient',GradientPresetManager)]:
  manager=cls(None);manager.presets_file=str(root/(kind+'_presets.json'));manager.presets={};managers[kind]=manager
 with patch('tabs.model_research.ThresholdPresetManager',return_value=managers['threshold']),patch('tabs.model_research.GradientPresetManager',return_value=managers['gradient']),patch.object(QMessageBox,'information'):
  research.add_to_presets(top)
 for kind,manager in managers.items():
  saved=json.loads(Path(manager.presets_file).read_text());assert saved and all(value['kind']==kind for value in saved.values())
  assert {json.dumps(value,sort_keys=True) for value in saved.values()}==set(top.loc[top.kind==kind,'preset_json'])
 research.close()
w.close();app.processEvents()
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
                timeout=50,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
