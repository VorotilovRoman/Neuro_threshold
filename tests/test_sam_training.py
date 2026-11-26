"""Instance targets, leakage prevention, differentiable loss and resumable worker."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from utils_new.configuration.sam_training import sam_training_config
from utils_new.dataset.sam import SamSample, discover_samples, sample_prompt, training_samples
from utils_new.import_export.coco import encode_rle
from utils_new.import_export.images import write_image
from utils_new.training.protocol import StopRequest, worker_arguments


class SamTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for split, value in (("train", 100), ("val", 150)):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            write_image(self.root / "images" / split / "image.png", np.full((12, 20, 3), value, np.uint8))
            (self.root / "labels" / split / "image.txt").write_text("0 0.1 0.2 0.7 0.2 0.7 0.8 0.1 0.8\n")
        self.source = self.root / "data.yaml"
        self.source.write_text(f"path: {self.root}\ntrain: images/train\nval: images/val\nnames: [part]\n")
        self.weights = self.root / "initial.pt"
        self.weights.write_bytes(b"placeholder")
        self.params = dict(source=str(self.source), model_path=str(self.weights), device="cpu",
                           project_path=str(self.root / "runs"), accum_steps=2, epochs=1,
                           use_amp=False, prompt_type="mixed")

    def config(self, **extra):
        return sam_training_config(dict(self.params, **extra))

    def test_config_rejects_sam3_nonfinite_and_unsupported_batch(self):
        for change in (dict(model_key="sam3"), dict(lr=float("nan")), dict(batch_size=2),
                       dict(accum_steps=0), dict(validation_fraction=0), dict(max_side=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(**change)
        self.assertEqual(self.config(batch_size=-1)["batch_size"], 1)
        self.assertIn("utils_new.training.sam", worker_arguments("sam", params_file="p.json"))

    def test_four_corner_segment_preserved_and_bbox_rejected(self):
        train, val = training_samples(self.config(), log=lambda _: None)
        self.assertEqual((len(train), len(val)), (1, 1))
        masks, valid = train[0].masks(self.config())
        self.assertEqual(len(masks), 1)
        self.assertTrue(valid.all())
        Path(train[0].target).write_text("0 0.5 0.5 0.4 0.4\n")
        with self.assertRaises(ValueError):
            train[0].masks(self.config())

    def test_coco_rle_keeps_holes_and_multipart_is_one_instance(self):
        image = str(self.root / "images/train/image.png")
        target = np.zeros((12, 20), np.uint8)
        target[1:11, 1:19] = 1
        target[4:8, 6:14] = 0
        sample = SamSample(image, "coco", [{"segmentation": encode_rle(target)}])
        masks, _ = sample.masks(self.config())
        np.testing.assert_array_equal(masks[0], target)
        sample = SamSample(image, "coco", [{"segmentation": [[1, 1, 4, 1, 4, 4, 1, 4], [12, 7, 16, 7, 16, 10, 12, 10]]}])
        masks, _ = sample.masks(self.config())
        self.assertEqual(len(masks), 1)
        self.assertTrue(masks[0][2, 2] and masks[0][8, 13])
        for ann in ({"bbox": [1, 1, 4, 4]}, {"iscrowd": 1, "segmentation": encode_rle(target)}):
            with self.assertRaises(ValueError):
                SamSample(image, "coco", [ann]).masks(self.config())

    def test_index_masks_split_instances_and_preserve_ignore(self):
        labels = np.zeros((12, 20), np.uint8)
        labels[1:4, 1:4] = 2
        labels[7:10, 12:16] = 2
        labels[:, -1] = 255
        path = self.root / "mask.png"
        write_image(path, labels)
        sample = SamSample(str(self.root / "images/train/image.png"), "unet", str(path))
        masks, valid = sample.masks(self.config())
        self.assertEqual(len(masks), 2)
        self.assertFalse(valid[:, -1].any())

    def test_duplicate_image_bytes_rejected_across_splits(self):
        (self.root / "images/val/image.png").write_bytes((self.root / "images/train/image.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "Утечка"):
            training_samples(self.config(), log=lambda _: None)

    def test_explicit_validation_yaml_uses_only_val_not_train_or_test(self):
        self.source.write_text(self.source.read_text() + "test: images/train\n")
        train, val = training_samples(self.config(val_source=str(self.source)), log=lambda _: None)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(val[0].split, "val")

    def test_reviewed_sources_remain_authoritative_and_coco_discovery_keeps_rle(self):
        from utils_new.import_export.dataset_sources import defaults, preview, write_manifest
        beside = self.root / "images/train/image.txt"
        beside.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        # Competing sidecar must not override the source approved in the shared dialog.
        plan = preview(defaults([self.source], "yolo"), "labels")
        approved = write_manifest(plan, self.root / "approved")[0]
        sample = next(s for s in discover_samples(approved, log=lambda _: None) if s.split == "train")
        self.assertEqual(Path(sample.target), self.root / "labels/train/image.txt")
        mask = np.zeros((12, 20), np.uint8)
        mask[2:7, 3:11] = 1
        source = self.root / "coco.json"
        source.write_text(json.dumps(dict(images=[dict(id=1, file_name="images/train/image.png", width=20, height=12)],
            annotations=[dict(id=1, image_id=1, category_id=1, segmentation=encode_rle(mask))],
            categories=[dict(id=1, name="part")])))
        samples = discover_samples(source)
        self.assertEqual(len(samples), 1)
        np.testing.assert_array_equal(samples[0].masks(self.config())[0][0], mask)

    def test_prompt_labels_match_instance_and_ignore_pixels_are_never_background(self):
        mask = np.zeros((12, 20), bool)
        mask[2:8, 4:13] = True
        valid = np.ones_like(mask)
        valid[:, :3] = False
        cfg = self.config(prompt_type="mixed")
        rng = np.random.default_rng(11)
        for _ in range(30):
            box, points, labels = sample_prompt(mask, valid, cfg, rng, training=True)
            if box is not None:
                self.assertTrue(0 <= box[0] < box[2] <= 20 and 0 <= box[1] < box[3] <= 12)
            for (x, y), label in zip(points.astype(int), labels):
                self.assertTrue(valid[y, x])
                self.assertEqual(bool(mask[y, x]), bool(label))
        a = sample_prompt(mask, valid, cfg, np.random.default_rng(42), training=False)
        b = sample_prompt(mask, valid, cfg, np.random.default_rng(42), training=False)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_loss_has_gradients_and_ignored_pixels_do_not_change_metrics(self):
        from runtime_torch import require_torch
        from utils_new.training.sam import mask_loss
        torch = require_torch()
        logits = torch.randn(3, 12, 20, requires_grad=True)
        scores = torch.randn(3, requires_grad=True)
        target = torch.zeros(12, 20)
        target[2:8, 4:13] = 1
        valid = torch.ones_like(target)
        valid[0] = 0
        loss, iou, dice = mask_loss(logits, scores, target, valid)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(scores.grad.abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, 0].abs().sum()), 0)
        changed = logits.detach().clone()
        changed[:, 0] = 100
        other = mask_loss(changed, scores.detach(), target, valid)
        self.assertAlmostEqual(float(loss.detach()), float(other[0]), places=6)
        self.assertEqual((iou, dice), other[1:])

    def engine_patches(self):
        from runtime_torch import require_torch
        torch = require_torch()

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.1))

        class TinyTrainer:
            def __init__(self, model, *args):
                self.model = model

            def mode(self, training):
                self.model.train(training)

            def encode(self, image):
                self.image = torch.as_tensor(image[:, :, 0].copy()).float() / 255

            def logits(self, prompt):
                return (self.model.weight * self.image)[None].repeat(3, 1, 1), self.model.weight.repeat(3)

        def load(key, weights, device, **kwargs):
            model = TinyModel()
            if str(weights).endswith("resume.pt"):
                model.load_state_dict(torch.load(weights, weights_only=True)["model"])
            return model

        return patch("utils_new.training.sam.PromptTrainer", TinyTrainer), patch("utils_new.segmentation.backends.sam.load_prompt_model", load)

    def test_resume_matches_continuous_training_and_rejects_edited_labels(self):
        from runtime_torch import require_torch
        from utils_new.training.sam import run_training
        torch = require_torch()
        first, second = self.engine_patches()
        with first, second:
            self.assertEqual(run_training(dict(self.params, epochs=2, name="continuous"), log=lambda _: None), 0)
            self.assertEqual(run_training(dict(self.params, name="first"), log=lambda _: None), 0)
            resume = self.root / "runs/first/last.resume.pt"
            self.weights.unlink()  # Full resume must be independent of initial weights.
            self.assertEqual(run_training(dict(self.params, resume=str(resume), epochs=2, name="resumed"), log=lambda _: None), 0)
            a = torch.load(self.root / "runs/continuous/last.pt", weights_only=True)
            b = torch.load(self.root / "runs/resumed/last.pt", weights_only=True)
            torch.testing.assert_close(a["model"]["weight"], b["model"]["weight"], rtol=0, atol=0)
            self.assertNotEqual(float(a["model"]["weight"]), 0.1)
            self.assertTrue((self.root / "runs/resumed/best.pt").is_file())
            label = self.root / "labels/train/image.txt"
            label.write_text(label.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(ValueError, "разметка"):
                run_training(dict(self.params, resume=str(resume), epochs=2), log=lambda _: None)

    def test_cancel_does_not_mark_partial_epoch_as_complete(self):
        from utils_new.training.sam import run_training
        stop = StopRequest()
        def log(line):
            if "Эпоха 1 train: 1/1" in line:
                stop.set()
        first, second = self.engine_patches()
        with first, second:
            self.assertEqual(run_training(self.params, stop=stop, log=log), 130)
        root = self.root / "runs/sam"
        self.assertTrue((root / "interrupted.pt").is_file())
        self.assertFalse((root / "last.resume.pt").exists())
        self.assertFalse(json.loads((root / "interrupted_meta.json").read_text())["complete_epoch"])

    def test_reviewed_import_survives_closed_dialog_and_resume(self):
        import shutil
        from utils_new.import_export.dataset_sources import defaults, preview, write_manifest
        from utils_new.training.sam import run_training
        imported = self.root / "temporary-import"
        source = write_manifest(preview(defaults([self.source], "yolo"), "labels"), imported)[0]
        first, second = self.engine_patches()
        with first, second:
            run_training(dict(self.params, source=source, name="persisted"), log=lambda _: None)
            run = self.root / "runs/persisted"
            cfg = json.loads((run / "params.json").read_text())
            self.assertNotEqual(cfg["source"], source)
            shutil.rmtree(imported)
            cfg.update(resume=str(run / "last.resume.pt"), epochs=2, name="continued")
            self.assertEqual(run_training(cfg, log=lambda _: None), 0)


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "Opt-in Qt")
class SamTrainingGuiTests(unittest.TestCase):
    def test_training_tab_uses_real_worker_and_default_tiny_batch_one(self):
        from PyQt5.QtWidgets import QApplication
        from tabs.deep_learning_training import YoloTrainWindow
        app = QApplication.instance() or QApplication([])
        window = YoloTrainWindow()
        try:
            panel = window.sam_train_widget
            self.assertEqual(panel.model.currentData(), "sam2.1-tiny")
            self.assertEqual(panel.model.findData("sam3"), -1)
            self.assertEqual(panel.batch.value(), 1)
            self.assertFalse(panel.auto_batch.isEnabled())
            self.assertEqual(window._training_parts("sam"), ("sam_process", "sam_train_widget", "SAM"))
            app.processEvents()
        finally:
            window.close()
