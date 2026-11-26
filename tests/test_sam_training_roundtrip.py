"""Opt-in real SAM/SAM2 training and both UI hosts; local weights only, no network.

NEURO_RUN_GUI_TESTS=1 NEURO_SAM_TEST_WEIGHTS=/path/sam2.1_hiera_tiny.pt
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p test_sam_training_roundtrip.py -v
NEURO_SAM1_TEST_WEIGHTS optionally selects local SAM ViT-B weights for the second test.
"""

import csv
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "Requires opt-in Qt")
class SamNativeRoundtripTests(unittest.TestCase):
    def until(self, app, condition):
        deadline = time.monotonic() + 60
        from PyQt5.QtTest import QTest
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(5)
        self.assertTrue(condition(), "Native SAM/Qt operation did not settle")

    @unittest.skipUnless(os.environ.get("NEURO_SAM_TEST_WEIGHTS"), "Requires local SAM2.1 tiny weights")
    def test_sam2_training_metrics_and_both_application_tabs(self):
        self.check_roundtrip("sam2.1-tiny", os.environ["NEURO_SAM_TEST_WEIGHTS"])

    @unittest.skipUnless(os.environ.get("NEURO_SAM1_TEST_WEIGHTS"), "Requires local SAM ViT-B weights")
    def test_sam1_training_metrics_and_both_application_tabs(self):
        self.check_roundtrip("sam-vit-b", os.environ["NEURO_SAM1_TEST_WEIGHTS"])

    @unittest.skipUnless(os.environ.get("NEURO_SAM_TEST_WEIGHTS"), "Requires local SAM2.1 tiny weights")
    def test_sam2_coco_rle_and_index_mask_training_use_identical_targets(self):
        from runtime_torch import require_torch
        from utils_new.import_export.coco import encode_rle
        from utils_new.import_export.images import write_image
        from utils_new.training.sam import run_training
        torch = require_torch()
        threads = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, threads)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.zeros((64, 96), np.uint8)
            labels[10:30, 10:35] = 1
            labels[16:24, 18:28] = 0  # A real target hole must survive both loaders.
            labels[34:58, 55:85] = 2
            for split, marker in (("train", 1), ("val", 2)):
                (root / "images" / split).mkdir(parents=True)
                (root / "masks" / split).mkdir(parents=True)
                image = np.repeat((labels > 0).astype(np.uint8)[:, :, None], 3, axis=2) * 180
                image[0, 0, 0] = marker
                write_image(root / f"images/{split}/image.png", image)
                write_image(root / f"masks/{split}/image.png", labels)
                payload = dict(images=[dict(id=1, file_name=f"images/{split}/image.png", width=96, height=64)],
                    annotations=[dict(id=cid, image_id=1, category_id=cid, iscrowd=0,
                                      segmentation=encode_rle(labels == cid)) for cid in (1, 2)],
                    categories=[dict(id=cid, name=str(cid)) for cid in (1, 2)])
                (root / f"{split}.json").write_text(json.dumps(payload))
            source = root / "masks.yaml"
            source.write_text(f"path: {root}\ntrain: images/train\nval: images/val\n"
                              "train_masks: masks/train\nval_masks: masks/val\nnum_classes: 3\n"
                              "class_names: [background, a, b]\nmask_class_offset: 0\n")
            rows = []
            for name, source_path, val_path in (("coco", root / "train.json", root / "val.json"),
                                                ("indexed", source, "")):
                cfg = dict(source=str(source_path), val_source=str(val_path),
                    model_path=os.environ["NEURO_SAM_TEST_WEIGHTS"], project_path=str(root / "runs"),
                    name=name, epochs=1, device="cpu", use_amp=False, accum_steps=1, prompt_type="box")
                self.assertEqual(run_training(cfg, log=lambda _: None), 0)
                summary = json.loads((root / "runs" / name / "metrics.json").read_text())
                self.assertEqual(summary["last"]["train_objects"], 2)
                self.assertEqual(summary["last"]["val_objects"], 2)
                rows.append(summary["last"])
                gc.collect()
            for key in ("train_loss", "val_loss", "val_iou", "val_dice"):
                self.assertAlmostEqual(rows[0][key], rows[1][key], places=6)

    def check_roundtrip(self, model_key, weights):
        from runtime_torch import require_torch
        from utils_new.dataset.sam import SamSample, sample_prompt
        from utils_new.import_export.images import write_image
        from utils_new.segmentation.backends.sam import LocalSAM, SamPrompt
        from utils_new.training.sam import run_training
        torch = require_torch()
        before_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, before_threads)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split, value in (("train", 1), ("val", 2)):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                image = np.zeros((64, 96, 3), np.uint8)
                image[10:30, 10:35] = [160, 180, 220]
                image[34:58, 55:85] = [220, 160, 180]
                image[0, 0, 0] = value
                write_image(root / "images" / split / "sample.png", image)
                (root / "labels" / split / "sample.txt").write_text(
                    "0 " + " ".join(str(v) for xy in ((10, 10), (35, 10), (35, 30), (10, 30)) for v in (xy[0]/96, xy[1]/64)) + "\n" +
                    "0 " + " ".join(str(v) for xy in ((55, 34), (85, 34), (85, 58), (55, 58)) for v in (xy[0]/96, xy[1]/64)) + "\n")
            source = root / "data.yaml"
            source.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames: [part]\n")
            cfg = dict(source=str(source), model_path=weights, model_key=model_key,
                       project_path=str(root / "runs"), name="checked", epochs=2, device="cpu",
                       accum_steps=2, prompt_type="box", train_prompt_encoder=True, use_amp=False)
            self.assertEqual(run_training(cfg, log=lambda _: None), 0)
            run = root / "runs/checked"
            initial = torch.load(cfg["model_path"], map_location="cpu", weights_only=True)
            current = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
            initial, current = initial.get("model", initial), current.get("model", current)
            self.assertTrue(all(torch.equal(v, current[k]) for k, v in initial.items() if k.startswith("image_encoder.")))
            decoder = "mask_decoder." if model_key.startswith("sam-vit-") else "sam_mask_decoder."
            prompt_encoder = "prompt_encoder." if model_key.startswith("sam-vit-") else "sam_prompt_encoder."
            self.assertTrue(any(not torch.equal(v, current[k]) for k, v in initial.items() if k.startswith(decoder)))
            self.assertTrue(any(not torch.equal(v, current[k]) for k, v in initial.items() if k.startswith(prompt_encoder)))
            del initial, current
            summary = json.loads((run / "metrics.json").read_text())
            meta = json.loads((run / "best_meta.json").read_text())
            with (run / "results.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["best"], meta["metrics"])
            self.assertEqual(meta["epoch"], meta["metrics"]["epoch"])
            self.assertEqual(summary["last"]["val_objects"], 2)
            self.assertEqual(summary["last"]["train_updates"], 1)
            self.assertAlmostEqual(float(rows[-1]["val_iou"]), summary["last"]["val_iou"])
            sample = SamSample(str(root / "images/val/sample.png"), "yolo", str(root / "labels/val/sample.txt"))
            from utils_new.configuration.sam_training import sam_training_config
            checked_cfg = sam_training_config(cfg)
            masks, valid = sample.masks(checked_cfg)
            backend = LocalSAM.load(model_key, run / "best.pt", device="cpu")
            rng = np.random.default_rng(checked_cfg["seed"])
            import cv2
            bgr = cv2.cvtColor(sample.rgb(), cv2.COLOR_RGB2BGR)
            measured = []
            for target in masks:
                box, points, labels = sample_prompt(target, valid, checked_cfg, rng, training=False)
                result = backend.predict(bgr, SamPrompt(tuple(map(tuple, points)), tuple(labels), tuple(box)), min_score=0)
                predicted = result.masks[0] & valid
                overlap = (predicted & target).sum()
                measured.append((overlap / (predicted | target).sum(),
                                 (2 * overlap + 1e-6) / (predicted.sum() + target.sum() + 1e-6)))
            np.testing.assert_allclose(np.mean(measured, axis=0),
                [meta["metrics"]["val_iou"], meta["metrics"]["val_dice"]], atol=1e-6, rtol=0)
            del backend
            gc.collect()

            from PyQt5.QtWidgets import QApplication, QMessageBox
            from tabs.layout_dataset import Labeler
            from tabs.deep_learning_methods import DeepLearningWindow
            app = QApplication.instance() or QApplication([])
            for constructor in (Labeler, DeepLearningWindow):
                with self.subTest(tab=constructor.__name__):
                    owner = constructor()
                    warnings = []
                    try:
                        if isinstance(owner, Labeler):
                            owner.auto_backend_combo.setCurrentIndex(1)
                        else:
                            owner.model_combo.setCurrentIndex(owner.model_combo.findText("SAM"))
                        owner._load_images([sample.image])
                        controller, panel = owner.sam_controller, owner.sam_panel
                        self.until(app, lambda: controller.host.image_ready() and controller.host.can_start())
                        panel.model.setCurrentIndex(panel.model.findData("sam3"))
                        with patch("ui.sam.controller.source_dialogs.get_open_file", return_value=(str(run / "best.pt"), "")):
                            controller.browse_weights()
                        self.assertEqual(panel.model.currentData(), model_key)
                        self.assertIn("val IoU", panel.checkpoint_info.text())
                        panel.device.setCurrentText("cpu")
                        panel.score.setValue(0)
                        with patch.object(QMessageBox, "warning", side_effect=lambda *args: warnings.append(args[2])):
                            panel.load.click()
                            self.until(app, lambda: controller.job is None)
                            self.assertIsNotNone(controller.backend, warnings)
                            self.assertTrue(panel.load.property("modelLoaded"))
                            controller.view.set_prompt_box((10, 10, 35, 30))
                            panel.run.click()
                            self.until(app, lambda: controller.job is None)
                            self.assertIsNotNone(controller.preview_result, warnings)
                            self.assertTrue(panel.accept.isEnabled(), panel.status.text())
                            expected = controller.preview_result.combined_mask()
                            panel.accept.click()
                        self.assertFalse(warnings, warnings)
                        if isinstance(owner, Labeler):
                            self.assertGreater(len(owner.current_annotations), 0)
                        else:
                            np.testing.assert_array_equal(owner.prediction_mask, expected)
                    finally:
                        owner.close()
                        app.processEvents()
                        owner.deleteLater()
                        app.processEvents()
                        del owner
                        gc.collect()
