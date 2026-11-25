"""Prompt, local checkpoint and mask contracts; no real weights or network."""

from concurrent.futures import CancelledError
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from utils_new.segmentation.backends.sam import LocalSAM, SamPrompt, SamResult


class SamBackendTests(unittest.TestCase):
    def backend(self):
        backend = LocalSAM()
        backend.model, backend.model_key = object(), "sam2.1-tiny"
        backend.torch = SimpleNamespace(inference_mode=nullcontext)
        masks = np.zeros((3, 12, 20), bool)
        masks[1, 2:10, 3:16] = True
        backend.predictor = SimpleNamespace(set_image=Mock(),
            predict=Mock(return_value=(masks, np.array([0.1, 0.9, 0.4]), None)))
        return backend

    def test_prompt_validation_rejects_ignored_and_invalid_inputs(self):
        cases = [SamPrompt(), SamPrompt(((2, 2),), (0,)), SamPrompt(((20, 2),), (1,)),
                 SamPrompt(((2, float("nan")),), (1,)), SamPrompt(((2, 2),), (2,)),
                 SamPrompt(((2, 2),), ()), SamPrompt(box=(8, 1, 3, 8)),
                 SamPrompt(box=(0, 0, 21, 12)), SamPrompt(text="object"),
                 SamPrompt(((2, 2),), (1,), text="object")]
        for prompt in cases:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                prompt.validated(20, 12)
        SamPrompt(((2, 2), (3, 4)), (1, 0), (0, 0, 20, 12)).validated(20, 12)
        SamPrompt(text="metal screw", box=(0, 0, 20, 12)).validated(20, 12, supports_text=True)

    def test_rgb_encoding_cache_best_mask_and_point_labels(self):
        backend = self.backend()
        image = np.zeros((12, 20, 3), np.uint8)
        image[:, :, 0] = 12
        image[:, :, 2] = 251
        prompt = SamPrompt(((2, 3), (6, 8)), (1, 0), (0, 0, 20, 12))
        result = backend.predict(image, prompt, image_key="image-1", min_score=0.5)
        self.assertEqual(result.masks.shape, (1, 12, 20))
        self.assertEqual(float(result.scores[0]), 0.9)
        np.testing.assert_array_equal(backend.predictor.set_image.call_args.args[0][0, 0], [251, 0, 12])
        np.testing.assert_array_equal(backend.predictor.predict.call_args.kwargs["point_labels"], [1, 0])
        backend.predict(image, SamPrompt(box=(2, 3, 10, 11)), image_key="image-1")
        self.assertEqual(backend.predictor.set_image.call_count, 1)
        backend.predict(image, prompt, image_key="image-2")
        self.assertEqual(backend.predictor.set_image.call_count, 2)

    def test_gray_uint16_no_input_mutation_and_cancelled_encoding_not_cached(self):
        backend = self.backend()
        image = np.full((12, 20), 65535, np.uint16)
        prompt = SamPrompt(box=(1, 1, 10, 11))
        backend.predict(image, prompt)
        rgb = backend.predictor.set_image.call_args.args[0]
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(rgb.shape, (12, 20, 3))
        self.assertTrue((rgb == 255).all())
        self.assertTrue((image == 65535).all())
        cancel = Event()
        backend.predictor.set_image.side_effect = lambda _: cancel.set()
        with self.assertRaises(CancelledError):
            backend.predict(image, prompt, image_key="new", cancel=cancel)
        self.assertIsNone(backend._image_key)

    def test_score_filter_and_output_shape_validation(self):
        backend = self.backend()
        image = np.zeros((12, 20), np.uint8)
        prompt = SamPrompt(box=(1, 1, 10, 11))
        self.assertEqual(len(backend.predict(image, prompt, min_score=0.95).masks), 0)
        backend.predictor.predict.return_value = (np.zeros((1, 5, 7)), np.array([0.9]), None)
        with self.assertRaisesRegex(ValueError, "некорректные"):
            backend.predict(image, prompt)
        with self.assertRaises(ValueError):
            backend.predict(image, prompt, min_score=float("nan"))

    def test_shared_mask_conversion_preserves_class_and_normalized_geometry(self):
        mask = np.zeros((1, 30, 40), bool)
        mask[:, 4:24, 5:35] = True
        mask[:, 10:15, 15:20] = False  # YOLO cannot represent this hole.
        result = SamResult(mask, np.array([0.9]))
        for kind in ("segment", "detect", "obb"):
            annotations = result.annotations(class_id=7, kind=kind)
            self.assertEqual(len(annotations), 1)
            self.assertEqual(annotations[0][:2], (kind, 7))
            coordinates = annotations[0][2:] if kind == "detect" else annotations[0][2]
            self.assertTrue(all(0 <= value <= 1 for value in coordinates))
        self.assertEqual(result.annotations(min_area=10000), [])

    def test_sam_and_sam2_builders_receive_only_local_paths(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        model = Mock()
        model.to.return_value = model
        package = SimpleNamespace(sam_model_registry={"vit_b": Mock(return_value=model)}, SamPredictor=Mock())
        builder = Mock(return_value=model)
        predictor = Mock()
        modules = {"segment_anything": package, "sam2.build_sam": SimpleNamespace(build_sam2=builder),
                   "sam2.sam2_image_predictor": SimpleNamespace(SAM2ImagePredictor=predictor)}
        with TemporaryDirectory() as directory:
            weights = Path(directory) / "custom weights.pth"
            weights.write_bytes(b"checkpoint")
            with patch("utils_new.segmentation.backends.sam.require_torch", return_value=torch), \
                    patch("utils_new.segmentation.backends.sam.importlib.import_module", side_effect=modules.__getitem__):
                LocalSAM.load("sam-vit-b", weights)
                package.sam_model_registry["vit_b"].assert_called_once_with(checkpoint=str(weights))
                LocalSAM.load("sam2.1-base-plus", weights, device="cpu")
                builder.assert_called_once_with("configs/sam2.1/sam2.1_hiera_b+.yaml", str(weights), device="cpu")
            with self.assertRaises(FileNotFoundError):
                LocalSAM.load("sam-vit-b", weights.with_name("missing.pth"))

    def test_explicit_cuda_does_not_silently_fall_back(self):
        with TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pt"
            weights.write_bytes(b"checkpoint")
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch("utils_new.segmentation.backends.sam.require_torch", return_value=torch), \
                    self.assertRaisesRegex(RuntimeError, "CUDA"):
                LocalSAM.load("sam2.1-tiny", weights, device="cuda:0")

    def test_sam3_groups_clicks_into_one_object_and_requires_matching_mode(self):
        backend = self.backend()
        backend.model_key = "sam3"
        boxes = np.array([[0, 0, 10, 10, 0.9, 0]])
        backend.predictor.inference_features = Mock(return_value=(np.zeros((1, 12, 20)), boxes))
        backend.predictor.reset_image = Mock()
        backend.predictor.features = object()
        image = np.zeros((12, 20, 3), np.uint8)
        backend.predict(image, SamPrompt(((1, 1), (2, 2)), (1, 0)))
        arguments = backend.predictor.inference_features.call_args.kwargs
        self.assertEqual(arguments["points"].shape, (1, 2, 2))
        self.assertEqual(arguments["labels"].shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "режим"):
            backend.predict(image, SamPrompt(text="screw"))

    def test_sam3_text_logits_use_zero_threshold_and_scores_include_presence(self):
        from runtime_torch import require_torch

        torch = require_torch()
        backend = self.backend()
        backend.torch, backend.model_key, backend.prompt_mode = torch, "sam3", "text"
        backend.predictor.reset_image = Mock()
        backend.predictor.features = object()
        backend.predictor._prepare_geometric_prompts = Mock(return_value=(None, None))
        # Positive logits below .5 are foreground; negative logits are background.
        logits = torch.full((1, 2, 3, 5), -0.1)
        logits[:, 0] = 0.1
        backend.predictor._inference_features = Mock(return_value={
            "pred_masks": logits, "pred_logits": torch.tensor([[[3.0], [0.0]]]),
            "presence_logit_dec": torch.tensor([[3.0]])})
        result = backend.predict(np.zeros((12, 20, 3), np.uint8), SamPrompt(text="screw"), min_score=0.8)
        self.assertEqual(result.masks.shape, (1, 12, 20))
        self.assertTrue(result.masks.all())
        self.assertAlmostEqual(float(result.scores[0]), float(torch.sigmoid(torch.tensor(3.0)) ** 2), places=5)

    def test_sam3_load_rejects_wrong_checkpoint_before_building(self):
        backend = LocalSAM()
        backend.torch = SimpleNamespace(load=Mock(return_value={"yolo": object()}))
        with self.assertRaisesRegex(ValueError, "SAM3"):
            backend._load_sam3(Path("local.pt"), None)

    def test_sam3_text_loader_uses_selected_local_tokenizer_without_official_sam3(self):
        backend = LocalSAM()
        backend.model_key, backend.prompt_mode = "sam3", "text"
        backend.torch = SimpleNamespace(load=Mock(return_value={"detector.backbone.weight": object()}))
        tokenizer = Mock()
        model = SimpleNamespace(backbone=SimpleNamespace(language_backbone=SimpleNamespace()))
        predictor = Mock()
        builders = SimpleNamespace(build_sam3_image_model=Mock(return_value=model))
        modules = {
            "ultralytics.models.sam.predict": SimpleNamespace(SAM3SemanticPredictor=Mock(return_value=predictor)),
            "ultralytics.models.sam.build_sam3": builders,
            "clip": SimpleNamespace(simple_tokenizer=SimpleNamespace(SimpleTokenizer=Mock(return_value=tokenizer))),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sam3.pt"
            bpe = Path(directory) / "custom.gz"
            bpe.write_bytes(b"tokenizer")
            with patch("utils_new.segmentation.backends.sam.importlib.import_module", side_effect=modules.__getitem__):
                backend._load_sam3(path, bpe)
            builders.build_sam3_image_model.assert_called_once_with(str(path))
            modules["clip"].simple_tokenizer.SimpleTokenizer.assert_called_once_with(bpe_path=str(bpe))
            self.assertIs(model.backbone.language_backbone.tokenizer, tokenizer)
            predictor.setup_model.assert_called_once_with(model=model, verbose=False)

    def test_sam3_interactive_size_matches_patch_grid_and_does_not_import_clip(self):
        backend = LocalSAM()
        backend.model_key = "sam3"
        backend.torch = SimpleNamespace(load=Mock(return_value={
            "detector.backbone.weight": object(), "tracker.weight": object()}))
        model, predictor = object(), Mock()
        predictors = SimpleNamespace(SAM3Predictor=Mock(return_value=predictor))
        builders = SimpleNamespace(build_interactive_sam3=Mock(return_value=model))
        modules = {"ultralytics.models.sam.predict": predictors,
                   "ultralytics.models.sam.build_sam3": builders}
        with patch("utils_new.segmentation.backends.sam.importlib.import_module", side_effect=modules.__getitem__):
            backend._load_sam3(Path("local.pt"), None)
        options = predictors.SAM3Predictor.call_args.kwargs["overrides"]
        self.assertEqual(options["imgsz"], 1008)
        self.assertEqual(options["imgsz"] % 14, 0)


if __name__ == "__main__":
    unittest.main()
