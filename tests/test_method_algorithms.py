"""Algorithm contracts on controlled images, independent of pretrained weights."""

import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from utils_ops import interactive_ops as ops


class InteractiveAlgorithms(unittest.TestCase):
    def setUp(self):
        self.image = np.full((48, 96, 3), 220, np.uint8)
        self.image[12:36, 48:80] = (25, 35, 45)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.fg = np.zeros(self.gray.shape, np.uint8)
        self.bg = self.fg.copy()
        self.fg[20:24, 60:64] = 255
        self.bg[4:8, 4:8] = 255

    def test_grabcut_both_keeps_hard_seeds_and_does_not_modify_inputs(self):
        bg = self.bg.copy()
        bg[16:19, 51:54] = (
            255  # contrary color mark INSIDE ROI must still stay background
        )
        initial = np.full(self.gray.shape, cv2.GC_PR_BGD, np.uint8)
        before = initial.copy()
        mask = ops.grabcut_segmentation(
            self.image, (44, 8, 84, 40), initial, self.fg, bg
        )
        np.testing.assert_array_equal(initial, before)
        self.assertTrue(np.all(mask[self.fg > 0] == 255))
        self.assertTrue(np.all(mask[bg > 0] == 0))

    def test_grabcut_both_without_roi_really_segments(self):
        mask = ops.grabcut_segmentation(
            self.image, fg_scribbles=self.fg, bg_scribbles=self.bg
        )
        self.assertGreater(np.count_nonzero(mask), np.count_nonzero(self.fg) * 10)
        self.assertGreater(np.mean(mask[14:34, 50:78]), 250)

    def test_grabcut_invalid_roi_or_seeds_do_not_fallback(self):
        with self.assertRaises(ValueError):
            ops.grabcut_segmentation(self.image, (0, 0, 96, 48))
        with self.assertRaises(ValueError):
            ops.grabcut_segmentation(
                self.image, (44, 8, 84, 40), mode="GC_INIT_WITH_MASK"
            )

    def test_snake_uses_rows_columns_on_non_square_image_without_hidden_morphology(
        self,
    ):
        def unchanged(image, initial, **kwargs):
            self.assertLess(initial[:, 0].max(), 40)
            self.assertGreater(initial[:, 1].min(), 40)
            return initial

        with patch("skimage.segmentation.active_contour", side_effect=unchanged):
            result = ops.active_contour_segmentation(
                self.gray, (48, 12, 80, 36), init_type="Rectangle"
            )
        expected = np.zeros_like(result)
        expected[12:36, 48:80] = 255
        np.testing.assert_array_equal(result, expected)
        with self.assertRaises(ValueError):
            ops.active_contour_segmentation(self.gray, (48, 12, 80, 36), gamma=0)

    def test_all_seeded_methods_preserve_hard_marks(self):
        results = [
            ops.watershed_segmentation(
                self.gray, fg_scribbles=self.fg, bg_scribbles=self.bg, mode="scribbles"
            ),
            ops.random_walker_segmentation(
                self.image, np.where(self.fg, 1, np.where(self.bg, 2, 0)), mode="bf"
            ),
            ops.lazy_snapping(self.image, self.fg, self.bg, superpixel_size=8, sigma=0),
        ]
        for result in results:
            self.assertTrue(np.all(result[self.fg > 0] == 255))
            self.assertTrue(np.all(result[self.bg > 0] == 0))

    def test_graph_methods_cut_real_image_and_report_errors(self):
        for function in (ops.supercut_segmentation, ops.onecut_segmentation):
            result = function(self.image, (48, 12, 80, 36), superpixel_size=8, sigma=0)
            self.assertGreater(np.mean(result[16:32, 52:76]), 240)
            self.assertLess(np.mean(result[:8]), 10)
            with (
                patch("networkx.minimum_cut", side_effect=RuntimeError("test failure")),
                self.assertRaisesRegex(RuntimeError, "test failure"),
            ):
                function(self.image, (48, 12, 80, 36), superpixel_size=8)
        with self.assertRaises(ValueError):
            ops.onecut_segmentation(self.image, (0, 0, 96, 48))

    def test_watershed_min_distance_is_peak_spacing_not_area(self):
        with patch(
            "skimage.feature.peak_local_max", return_value=np.array([[20, 60]])
        ) as peaks:
            ops.watershed_segmentation(self.gray, min_distance=7)
        self.assertEqual(peaks.call_args.kwargs["min_distance"], 7)

    def test_morphological_chan_vese_finds_uniform_region(self):
        result = ops.morphological_chan_vese_segmentation(
            self.gray, (44, 8, 84, 40), iterations=30
        )
        expected = np.zeros_like(result, bool)
        expected[12:36, 48:80] = True
        self.assertGreater(
            np.count_nonzero((result > 0) & expected)
            / np.count_nonzero((result > 0) | expected),
            0.9,
        )

    def test_onecut_smoothing_reaches_backend(self):
        from utils_new.segmentation.operations import interactive

        with patch.object(
            ops, "onecut_segmentation", return_value=self.gray
        ) as implementation:
            interactive(
                "OneCut",
                self.image,
                self.gray,
                (1, 1, 20, 20),
                self.fg,
                self.bg,
                {"sigma": 2},
                threading.Event(),
            )
        self.assertEqual(implementation.call_args.kwargs["sigma"], 2)


class ThresholdAndGradientAlgorithms(unittest.TestCase):
    def test_threshold_local_formulas_match_reference_interior(self):
        from skimage.filters import threshold_niblack, threshold_sauvola

        from utils_new.threshold_processing import niblack_threshold, sauvola_threshold

        image = np.random.default_rng(4).integers(0, 256, (37, 61), dtype=np.uint8)
        # The classic mean+k*std convention is the NEGATIVE of skimage's k.
        for mask, threshold in (
            (niblack_threshold(image, 7, -0.2)[0], threshold_niblack(image, 7, k=0.2)),
            (
                sauvola_threshold(image, 7, 0.3, 128)[0],
                threshold_sauvola(image, 7, k=0.3, r=128),
            ),
        ):
            np.testing.assert_array_equal(
                mask[4:-4, 4:-4],
                ((image > threshold) * 255).astype(np.uint8)[4:-4, 4:-4],
            )

    def test_all_thresholds_share_polarity_and_report_actual_threshold(self):
        from utils_new.threshold_processing import (
            apply_threshold_method,
            row_adaptive_threshold,
        )

        image = np.tile(np.array([20] * 10 + [220] * 10, np.uint8), (20, 1))
        mask, _ = row_adaptive_threshold(image, window_size=20)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[0, -1], 255)
        for name in (
            "Simple Threshold",
            "Otsu",
            "Triangle",
            "Adaptive Mean",
            "Adaptive Gauss",
            "Niblack",
            "Sauvola",
            "ISODATA",
            "Background Symmetry",
            "Row Adaptive",
        ):
            result, threshold = apply_threshold_method(image, name, {})
            self.assertEqual(result.shape, image.shape)
            self.assertTrue(np.isin(result, [0, 255]).all())
            if threshold is not None:
                self.assertTrue(0 <= threshold <= 255)
                np.testing.assert_array_equal(
                    result, (image > threshold).astype(np.uint8) * 255
                )
        for bg, fg in ((240, 20), (20, 240)):
            image.fill(bg)
            image[5:15, 5:15] = fg
            mask, threshold = apply_threshold_method(image, "Background Symmetry", {})
            self.assertLess(min(bg, fg), threshold)
            self.assertGreater(max(bg, fg), threshold)
            self.assertNotEqual(mask[0, 0], mask[10, 10])

    def test_kirsch_is_signed_maximum_of_eight_compass_directions(self):
        from utils_ops.gradient_ops import apply_gradient_method

        # Low amplitude avoids saturation hiding the max(abs) error.
        image = np.random.default_rng(6).integers(0, 12, (17, 23), dtype=np.uint8)
        actual = apply_gradient_method(image, "Kirsch", {})
        ring = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
        for y, x in ((3, 5), (8, 11), (12, 19)):
            patch_values = image[y - 1 : y + 2, x - 1 : x + 2].astype(float)
            responses = []
            for direction in range(8):
                kernel = np.full((3, 3), -3.0)
                kernel[1, 1] = 0
                for step in range(3):
                    kernel[ring[(direction + step) % 8]] = 5
                responses.append(np.sum(kernel * patch_values))
            self.assertEqual(actual[y, x], np.clip(max(responses), 0, 255))

    def test_shared_threshold_backend_is_used_by_ensemble(self):
        from utils_new import threshold_processing
        from utils_ops import segmentation_ops

        self.assertIs(
            segmentation_ops.apply_threshold_method,
            threshold_processing.apply_threshold_method,
        )


class TraditionalLearningContracts(unittest.TestCase):
    def test_manual_labels_override_otsu_and_missing_labels_are_errors(self):
        import tempfile
        from pathlib import Path

        from utils_new.segmentation import operations
        from utils_new.segmentation.training_labels import manual_target

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            # BRIGHT foreground: inverse Otsu would learn the opposite labels.
            image = np.zeros((20, 30, 3), np.uint8)
            image[5:15, 10:20] = 220
            cv2.imwrite(str(path), image)
            with self.assertRaisesRegex(ValueError, "найдено 0"):
                manual_target(path, image.shape[:2])
            path.with_suffix(".txt").write_text("0 0.5 0.5 0.3333333333 0.5\n")
            spec = {
                "model_type": "Decision Tree",
                "training_source": "Ручная разметка YOLO",
                "use_superpixels": False,
                "use_intensity": True,
                "use_texture": False,
                "use_spatial": False,
                "normalize_features": True,
                "superpixel_size": 10,
            }
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x, y = operations.prepare_training(
                [image], [gray], spec, threading.Event(), [str(path)]
            )
            model, metadata = operations.train_prepared(x, y, spec, threading.Event())
            result = operations.predict_ml(
                image, gray, model, metadata, spec, False, threading.Event()
            )
            self.assertEqual(result[10, 15], 1)
            self.assertEqual(result[0, 0], 0)
            with self.assertRaisesRegex(ValueError, "Признаки отличаются"):
                operations.predict_ml(
                    image,
                    gray,
                    model,
                    metadata,
                    dict(spec, use_texture=True),
                    False,
                    threading.Event(),
                )
            path.with_suffix(".txt").write_text("broken label")
            with self.assertRaisesRegex(ValueError, "Некорректная"):
                manual_target(path, gray.shape)

    def test_tiny_superpixels_and_constant_k_search(self):
        from utils_ops.traditional_ml_ops import (
            compute_elbow_kmeans,
            create_superpixels,
            find_optimal_k_elbow,
        )

        segments = create_superpixels(np.zeros((3, 5, 3), np.uint8), 100)
        self.assertEqual(segments.shape, (3, 5))
        self.assertEqual(find_optimal_k_elbow(compute_elbow_kmeans(np.ones((3, 1)))), 1)
