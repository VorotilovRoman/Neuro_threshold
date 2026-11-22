"""Cross-domain contracts for reorganized utilities and confirmed data-loss bugs."""

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from utils_new.annotations import ImageSize
from utils_new.annotations.masks import remap_mask_indices, mask_to_annotations
from utils_new.annotations.metrics import maximum_iou_assignment
from utils_new.dataset.validation import PairValidator
from utils_new.dataset_audit.matching import match_detections
from utils_new.dataset_audit.objects import read_objects
from utils_new.images.cache import LazyImageCache
from utils_new.images.session import load_one_image_item, load_segmentation_image
from utils_new.import_export.catalog import collect_mask_classes
from utils_new.import_export.images import read_image_with_fallback, write_image
from utils_new.import_export.yolo_labels import (
    read_yolo_labels,
    load_annotations,
    write_yolo_labels,
)
from utils_new.import_export.yolo_yaml import load_yolo_yaml, build_yolo_save_target_map
from utils_new.import_export.mask_yaml import load_unet_yaml, is_unet_yaml
from utils_new.import_export.json_objects import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


class UtilityContracts(unittest.TestCase):
    def test_all_utilities_have_domain_packages_and_no_qt_dependencies(self):
        files = list((ROOT / "utils_new").rglob("*.py"))
        self.assertEqual(
            [p.name for p in (ROOT / "utils_new").glob("*.py")], ["__init__.py"]
        )
        for path in files:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                self.assertFalse(
                    any(
                        i.startswith(("PyQt5", "ui.", "ui_tabs.", "tabs."))
                        for i in imports
                    ),
                    path,
                )

    def test_image_lists_keep_split_membership_and_nested_save_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "nested").mkdir(parents=True)
            for name in ("a", "b", "extra"):
                write_image(
                    root / "images" / "nested" / (name + ".png"),
                    np.zeros((20, 40), np.uint8),
                )
            (root / "train.txt").write_text("images/nested/a.png\n")
            (root / "val.txt").write_text("./images/nested/b.png\n")
            data = root / "data.yaml"
            data.write_text(
                yaml.safe_dump(
                    {"train": "train.txt", "val": "val.txt", "names": ["object"]}
                )
            )
            pairs, names, base, counts = load_yolo_yaml(data)
            self.assertEqual({Path(p[0]).stem for p in pairs}, {"a", "b"})
            self.assertEqual(counts, {"train": 1, "val": 1})
            targets = build_yolo_save_target_map(data, [p[0] for p in pairs])
            self.assertEqual(
                Path(targets[str(root / "images/nested/a.png")])
                .relative_to(root)
                .as_posix(),
                "labels/train/nested/a.txt",
            )
            self.assertEqual(
                Path(targets[str(root / "images/nested/b.png")])
                .relative_to(root)
                .as_posix(),
                "labels/val/nested/b.txt",
            )
            # The saved canonical target must also be found on the next load.
            for target in targets.values():
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_text("0 .5 .5 .2 .2\n")
            self.assertTrue(all(label for image, label in load_yolo_yaml(data)[0]))

    def test_import_audit_and_reference_resolve_the_same_nested_lists(self):
        from utils_new.import_export.ground_truth import load_ground_truth
        from utils_new.import_export.yolo_yaml import resolve_split_sources
        from utils_new.dataset_audit.splits import split_memberships

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "lists/sub").mkdir(parents=True)
            for name in ("a", "b"):
                write_image(
                    root / "images" / (name + ".png"), np.zeros((10, 20), np.uint8)
                )
                (root / "images" / (name + ".txt")).write_text("0 .5 .5 .2 .2\n")
            (root / "lists/train.txt").write_text("./sub/nested.txt\n")
            (root / "lists/sub/nested.txt").write_text("../../images/a.png\n")
            config = root / "data.yaml"
            config.write_text("train: lists/train.txt\nvalid: images/b.png\n")
            pairs = load_yolo_yaml(config)[0]
            memberships = split_memberships(config, pairs)
            self.assertEqual(memberships[str(root / "images/a.png")], {"train"})
            self.assertEqual(memberships[str(root / "images/b.png")], {"val"})
            self.assertEqual(
                {s.image_path for s in load_ground_truth(config).samples},
                {p[0] for p in pairs},
            )
            # Validation aliases round-trip labels saved to the canonical val folder.
            (root / "images/b.txt").unlink()
            target = Path(
                build_yolo_save_target_map(config, [str(root / "images/b.png")])[
                    str(root / "images/b.png")
                ]
            )
            target.parent.mkdir(parents=True)
            target.write_text("0 .5 .5 .2 .2\n")
            self.assertEqual(
                dict(load_yolo_yaml(config)[0])[str(root / "images/b.png")], str(target)
            )
            (root / "lists/sub/nested.txt").write_text("../train.txt\n")
            with self.assertRaisesRegex(ValueError, "Циклический"):
                resolve_split_sources("lists/train.txt", root)

    def test_mask_yaml_relative_path_and_invalid_document(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config" / "masks.yaml").write_text(
                "train_images: images/train\nclass_names: [background, object]\n"
            )
            with patch("os.getcwd", return_value=str(root)):
                value = load_unet_yaml(root / "config" / "masks.yaml")
            self.assertEqual(Path(value["root"]), root / "config")
            invalid = root / "list.yaml"
            invalid.write_text("- not\n- a mapping\n")
            self.assertFalse(is_unet_yaml(invalid))
            with self.assertRaises(ValueError):
                load_unet_yaml(invalid)

    def test_parser_validator_and_audit_agree_on_quadrilaterals_and_mixed_rows(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            label = image.with_suffix(".txt")
            write_image(image, np.zeros((40, 80), np.uint8))
            pair = (str(image), str(label))
            label.write_text("\ufeff2 .1 .1 .8 .1 .6 .7 .2 .6\n", encoding="utf-8")
            self.assertEqual(
                read_yolo_labels(label, ImageSize(80, 40), strict=True)[0].kind,
                "segment",
            )
            validator = PairValidator(0, "auto")
            self.assertTrue(validator.validate(pair)[0])
            self.assertEqual(
                PairValidator.detect_annotation_type(label, "auto", (80, 40)), "segment"
            )
            self.assertEqual(
                len(read_objects(pair)[1]), 1
            )  # Diagnostic note is not an error.
            label.write_text("0 .5 .5 .2 .2\n1 .1 .1 .4 .1 .3 .3\n")
            validator.set_input_format("mixed")
            self.assertTrue(validator.validate(pair)[0])
            # Editing the file invalidates cached validation without a manual clear.
            label.write_text("0 nan .5 .2 .2\n")
            self.assertFalse(validator.validate(pair)[0])
            with self.assertRaises(ValueError):
                read_yolo_labels(label, ImageSize(80, 40), strict=True)
            self.assertTrue(validator.validate((str(image), None))[0])
            self.assertTrue(PairValidator(2).validate((str(image), None))[0])

    def test_mask_class_collection_does_not_truncate_32bit_ids(self):
        original = np.array([[0, 1000, 70000]], np.int32)
        with patch(
            "utils_new.import_export.catalog.read_index_mask", return_value=original
        ):
            values, pairs = collect_mask_classes([("image", "mask")])
        self.assertEqual(values, [0, 1000, 70000])
        self.assertEqual(pairs, [("image", "mask")])
        np.testing.assert_array_equal(original, [[0, 1000, 70000]])
        with self.assertRaises(ValueError):
            remap_mask_indices(original, {0.5: 1})
        with self.assertRaises(TypeError):
            remap_mask_indices(original, {1000: 0}, dtype=np.float32)
        with self.assertRaises(ValueError):
            mask_to_annotations(np.zeros((2, 2), np.uint8), holes="invalid")

    def test_mask_preview_resampling_keeps_large_ids(self):
        from utils_new.annotations.rendering import blend_index_mask

        source = np.array([[0, 2**40]], dtype=np.int64)
        preview, info = blend_index_mask(
            np.zeros((2, 4, 3), np.uint8), source, lambda cls: (0, 200, 0), opacity=1
        )
        self.assertEqual(info.values, (0, 2**40))
        self.assertEqual(info.rendered_pixels, 4)
        np.testing.assert_array_equal(
            preview[:, 2:], np.full((2, 2, 3), (0, 200, 0), np.uint8)
        )
        np.testing.assert_array_equal(source, [[0, 2**40]])
        with self.assertRaises(ValueError):
            blend_index_mask(
                np.zeros((2, 4, 3), np.uint8),
                source,
                lambda cls: (0, 0, 0),
                opacity=float("nan"),
            )

    def test_mask_overlay_adapters_use_same_geometry_and_blending(self):
        from utils_new.annotations import MaskEncoding
        from utils_new.annotations.rendering import blend_index_mask, overlay_class_mask
        from utils_new.annotations.processing import apply_class_mask_overlay

        image = np.full((4, 4, 3), 20, np.uint8)
        mask = np.ones((2, 2), np.int64)
        annotations = [
            ("detect", 0, 0.5, 0.5, 1.0, 1.0),
            ("detect", 1, 0.5, 0.5, 0.5, 0.5),
        ]
        palette = lambda cls: (200, 0, 0) if cls == 0 else (0, 200, 0)
        common, info = blend_index_mask(
            image,
            mask,
            palette,
            annotations=annotations,
            prefer_annotation_classes=True,
            annotation_order="specific",
        )
        np.testing.assert_array_equal(
            apply_class_mask_overlay(
                image, mask, 0.5, palette, annotations=annotations
            ),
            common,
        )
        from utils_new.annotations import from_legacy

        values = [from_legacy(a, ImageSize(4, 4)) for a in annotations]
        np.testing.assert_array_equal(
            overlay_class_mask(
                image, mask, palette, annotations=values, encoding=MaskEncoding()
            ),
            common,
        )
        self.assertEqual(info.rendered_pixels, 16)
        np.testing.assert_array_equal(common[1, 1], [10, 110, 10])
        np.testing.assert_array_equal(image, np.full((4, 4, 3), 20, np.uint8))

    def test_matching_uses_maximum_cardinality_in_both_audits_and_reports(self):
        gt = [[0, 0, 10, 10], [6, 0, 16, 10]]
        pred = [[2, 0, 12, 10], [0, 0, 6, 10]]
        result = match_detections(gt, [0, 0], pred, [0, 0], threshold=0.3)
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["false_negatives"], 0)
        rows, cols = maximum_iou_assignment([[2 / 3, 0.6], [3 / 7, 0]], 0.3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(maximum_iou_assignment([[0]], 0)[0]), 0)
        with self.assertRaises(ValueError):
            maximum_iou_assignment([[1.1]], 0.5)

    def test_atomic_writers_preserve_previous_bytes_and_cleanup_temporary_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            labels = root / "image.txt"
            config = root / "settings.json"
            write_image(image, np.full((20, 40), 7, np.uint8))
            labels.write_text("old labels")
            config.write_text("old config")
            before = {p: p.read_bytes() for p in (image, labels, config)}
            with patch(
                "utils_new.import_export.atomic.os.replace",
                side_effect=OSError("disk error"),
            ):
                with self.assertRaises(OSError):
                    write_image(image, np.zeros((20, 40), np.uint8))
                with self.assertRaises(OSError):
                    write_yolo_labels(labels, [], ImageSize(40, 20))
                with self.assertRaises(OSError):
                    write_json_atomic(config, {"new": 1})
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(set(root.iterdir()), set(before))
            with self.assertRaises(ValueError):
                write_json_atomic(config, {"nan": float("nan")})
            self.assertEqual(config.read_bytes(), before[config])

    def test_image_record_policies_preserve_model_rounding_and_source_geometry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            write_image(path, np.zeros((301, 1000), np.uint16))
            normal = load_one_image_item(path, max_side=640)
            model = load_segmentation_image(path, max_side=640)
            self.assertEqual(normal["image"].shape, (193, 640, 3))
            self.assertEqual(model["image"].shape, (192, 640, 3))
            self.assertEqual(normal["source_size"], (1000, 301))
            self.assertEqual(model["source_size"], (1000, 301))
            self.assertNotEqual(
                normal["normalization_policy"], model["normalization_policy"]
            )

    def test_cache_coalesces_loads_and_does_not_repopulate_after_shutdown(self):
        started, release, waiting = Event(), Event(), Event()
        count = [0]
        lock = Lock()

        def load(path):
            with lock:
                count[0] += 1
            started.set()
            release.wait(3)
            return {"path": path}

        cache = LazyImageCache(["image"], load)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(cache.get, 0)
                self.assertTrue(started.wait(1))

                def second_get():
                    waiting.set()
                    return cache.get(0)

                second = pool.submit(second_get)
                self.assertTrue(waiting.wait(1))
                cache.shutdown()
                self.assertIsNone(second.result(timeout=1))
                release.set()
                self.assertIsNone(first.result(timeout=1))
            self.assertEqual(count[0], 1)
            self.assertFalse(cache.is_loaded(0))
            self.assertIsNone(cache.get(0))
            cache.prefetch([0])  # Closed cache never submits new work.
        finally:
            release.set()
            cache.shutdown()
        broken = LazyImageCache(
            ["broken"], lambda p: (_ for _ in ()).throw(ValueError("decode failed"))
        )
        try:
            self.assertIsNone(broken.get(0))
            self.assertIsInstance(broken.get_error(0), ValueError)
        finally:
            broken.shutdown()
        with self.assertRaises(ValueError):
            LazyImageCache([], lambda p: p, max_resident=0)
