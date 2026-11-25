"""Offline detector contracts, ROI coordinates and multi-object SAM2 composition."""

from concurrent.futures import CancelledError
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from utils_new.segmentation.backends.grounding_dino import GroundedSAM2, GroundingDetections, LocalGroundingDINO
from utils_new.segmentation.backends.sam import SamPrompt, SamResult


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class GroundingTests(unittest.TestCase):
    def detector(self, boxes=None, scores=None, phrases=None):
        detector = LocalGroundingDINO()
        detector.device = "cpu"
        detector.torch = SimpleNamespace(inference_mode=nullcontext)
        inputs = Inputs(input_ids=np.array([[1, 2]]))
        detector.processor = Mock(return_value=inputs)
        detector.processor.post_process_grounded_object_detection.return_value = [{
            "boxes": np.array(boxes if boxes is not None else [[2, 3, 20, 30]], dtype=float).reshape(-1, 4),
            "scores": np.asarray(scores if scores is not None else [.9]),
            "text_labels": phrases if phrases is not None else ["screw"]}]
        detector.model = Mock()
        return detector

    def test_roi_coordinates_rgb_nms_and_phrase_associations(self):
        detector = self.detector([[-5, -3, 20, 30], [0, 0, 19, 29], [35, 40, 80, 100], [22, 22, 22, 30]],
                                 [.9, .8, .75, .99], ["screw", "duplicate", "bolt", "degenerate"])
        image = np.zeros((100, 120, 3), np.uint8)
        image[:, :, 0], image[:, :, 2] = 12, 240
        result = detector.detect(image, "Metal Screw", roi=(10, 20, 70, 80))
        np.testing.assert_allclose(result.boxes, [[10, 20, 30, 50], [45, 60, 70, 80]])
        self.assertEqual(result.phrases, ("screw", "bolt"))
        image_input = detector.processor.call_args.kwargs["images"]
        self.assertEqual(image_input.size, (60, 60))
        self.assertEqual(image_input.getpixel((0, 0)), (240, 0, 12))
        self.assertEqual(detector.processor.call_args.kwargs["text"], "metal screw.")
        post = detector.processor.post_process_grounded_object_detection.call_args.kwargs
        self.assertEqual(post["target_sizes"], [(60, 60)])
        self.assertEqual(post["threshold"], .35)
        self.assertEqual(post["text_threshold"], .25)

    def test_confidence_and_maximum_count_are_applied_separately_from_sam_iou(self):
        detector = self.detector([[0, 0, 5, 5], [10, 10, 20, 20], [30, 30, 40, 40]],
                                 [.2, .9, .8], ["low", "first", "second"])
        result = detector.detect(np.zeros((50, 50), np.uint16), "bolt.", box_threshold=.5,
                                 text_threshold=.7, max_objects=1)
        self.assertEqual(result.phrases, ("first",))
        self.assertEqual(detector.processor.post_process_grounded_object_detection.call_args.kwargs["text_threshold"], .7)
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((50, 50)), "bolt", box_threshold=float("nan"))

    def test_empty_and_invalid_outputs(self):
        detector = self.detector([], [], [])
        self.assertEqual(detector.detect(np.zeros((10, 20), np.uint8), "screw").boxes.shape, (0, 4))
        detector.processor.post_process_grounded_object_detection.return_value[0]["boxes"] = np.zeros((1, 4))
        with self.assertRaisesRegex(ValueError, "некорректные"):
            detector.detect(np.zeros((10, 20), np.uint8), "screw")

    def test_cancel_before_and_after_detector_prevents_later_work(self):
        detector = self.detector()
        cancel = Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            detector.detect(np.zeros((40, 50), np.uint8), "screw", cancel=cancel)
        detector.model.assert_not_called()
        cancel.clear()
        detector.model.side_effect = lambda **inputs: cancel.set()
        with self.assertRaises(CancelledError):
            detector.detect(np.zeros((40, 50), np.uint8), "screw", cancel=cancel)
        detector.processor.post_process_grounded_object_detection.assert_not_called()

    def test_local_loader_never_uses_remote_ids_or_custom_kernels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
                (root / name).write_text("{}")
            config = SimpleNamespace(model_type="grounding-dino", use_timm_backbone=False)
            model = Mock()
            model.to.return_value = model
            model.eval.return_value = model
            transformers = SimpleNamespace(AutoConfig=Mock(), AutoProcessor=Mock(), AutoModelForZeroShotObjectDetection=Mock())
            transformers.AutoConfig.from_pretrained.return_value = config
            transformers.AutoModelForZeroShotObjectDetection.from_pretrained.return_value = model
            torch = SimpleNamespace(float32="float32")
            with patch("utils_new.segmentation.backends.grounding_dino.require_torch", return_value=torch), \
                    patch("utils_new.segmentation.backends.grounding_dino.importlib.import_module", return_value=transformers):
                backend = LocalGroundingDINO.load(root)
            for factory in (transformers.AutoConfig, transformers.AutoProcessor, transformers.AutoModelForZeroShotObjectDetection):
                call = factory.from_pretrained.call_args
                self.assertEqual(call.args[0], str(root))
                self.assertTrue(call.kwargs["local_files_only"])
                self.assertFalse(call.kwargs["trust_remote_code"])
            self.assertTrue(config.disable_custom_kernels)
            self.assertFalse(config.use_pretrained_backbone)
            self.assertTrue(transformers.AutoModelForZeroShotObjectDetection.from_pretrained.call_args.kwargs["use_safetensors"])
            self.assertIs(backend.model, model)

    def test_incomplete_model_bundle_fails_before_importing_transformers(self):
        with TemporaryDirectory() as directory, \
                patch("utils_new.segmentation.backends.grounding_dino.importlib.import_module") as imports:
            with self.assertRaisesRegex(ValueError, "полная локальная папка"):
                LocalGroundingDINO.load(directory)
            imports.assert_not_called()

    def composed(self, detections):
        backend = GroundedSAM2()
        backend.detector = SimpleNamespace(detect=Mock(return_value=detections))
        masks = np.ones((1, 40, 50), bool)
        backend.sam = SimpleNamespace(predict=Mock(return_value=SamResult(masks, np.array([.8]))))
        return backend

    def test_every_detected_box_is_segmented_with_one_shared_embedding_key(self):
        backend = self.composed(GroundingDetections(np.array([[2, 3, 20, 30], [25, 4, 45, 35]]),
                                                   np.array([.91, .72]), ("screw", "screw")))
        image = np.zeros((40, 50, 3), np.uint8)
        result = backend.predict(image, SamPrompt(text="screw", box=(1, 2, 49, 39)), image_key="frame-1", min_score=.6)
        self.assertEqual(result.masks.shape, (2, 40, 50))
        np.testing.assert_allclose(result.scores, [.8, .8])
        np.testing.assert_allclose(result.detection_scores, [.91, .72])
        self.assertEqual(result.phrases, ("screw", "screw"))
        self.assertEqual(result.detected_count, 2)
        for call in backend.sam.predict.call_args_list:
            self.assertIs(call.args[0], image)
            self.assertEqual(call.kwargs["image_key"], "frame-1")
            self.assertEqual(call.kwargs["min_score"], .6)
            self.assertFalse(call.args[1].text)
        self.assertEqual(backend.sam.predict.call_args_list[1].args[1].box, (25, 4, 45, 35))

    def test_no_detections_preserves_empty_result_without_segmenting(self):
        backend = self.composed(GroundingDetections(np.empty((0, 4)), np.empty(0), ()))
        result = backend.predict(np.zeros((40, 50), np.uint8), SamPrompt(text="screw"))
        self.assertEqual(result.masks.shape, (0, 40, 50))
        backend.sam.predict.assert_not_called()

    def test_detected_boxes_remain_visible_in_counts_if_sam_quality_filters_all_masks(self):
        backend = self.composed(GroundingDetections(np.array([[2, 3, 20, 30]]), np.array([.9]), ("screw",)))
        backend.sam.predict.return_value = SamResult(np.empty((0, 40, 50), bool), np.empty(0))
        result = backend.predict(np.zeros((40, 50)), SamPrompt(text="screw"), min_score=.95)
        self.assertEqual(result.detected_count, 1)
        self.assertEqual(result.masks.shape, (0, 40, 50))

    def test_cancellation_between_objects_and_invalid_prompts(self):
        backend = self.composed(GroundingDetections(np.array([[2, 3, 20, 30], [25, 4, 45, 35]]),
                                                   np.array([.91, .72]), ("screw", "screw")))
        cancel = Event()
        def predict(*args, **kwargs):
            cancel.set()
            return SamResult(np.ones((1, 40, 50), bool), np.array([.8]))
        backend.sam.predict.side_effect = predict
        with self.assertRaises(CancelledError):
            backend.predict(np.zeros((40, 50)), SamPrompt(text="screw"), cancel=cancel)
        self.assertEqual(backend.sam.predict.call_count, 1)
        with self.assertRaises(ValueError):
            backend.predict(np.zeros((40, 50)), SamPrompt(((2, 2),), (1,), text="screw"))


if __name__ == "__main__":
    unittest.main()
