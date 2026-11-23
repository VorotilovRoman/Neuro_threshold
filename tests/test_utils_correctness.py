"""Cross-domain utility regressions: exact sources, class IDs and no lost labels."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import yaml

from utils_new.dataset.pairs import (
    scan_detection_pairs,
    scan_image_only_pairs,
    scan_segmentation_pairs,
)
from utils_new.import_export.catalog import load_unet_catalog
from utils_new.import_export.coco import load_coco, write_yolo_txt_for_coco_items
from utils_new.import_export.dataset_output import write_classes_and_yaml
from utils_new.import_export.images import write_image
from utils_new.import_export.mask_yaml import (
    is_unet_yaml,
    load_unet_yaml,
    load_training_sources,
    write_yolo_txt_for_unet_pairs,
)
from utils_new.import_export.yolo_labels import read_yolo_labels
from utils_new.import_export.yolo_yaml import find_mask_for_image
from utils_new.annotations import ImageSize
from utils_new.segmentation.model_output import decode_output


def image(path, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((12, 16), np.uint8)
    values[2:10, 3:13] = 1 if mask else 200
    write_image(path, values)
    return str(path)


class FilesystemUtilities(unittest.TestCase):
    def test_mask_paths_are_sufficient_to_recognize_mask_yaml(self):
        with TemporaryDirectory() as directory:
            cfg = Path(directory) / "data.yaml"
            cfg.write_text("train: images/train\nmasks: {train: mask_sources}\n")
            self.assertTrue(is_unet_yaml(cfg))

    def test_mask_catalog_and_training_agree_on_nested_lists_custom_masks_and_val_alias(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            a = image(root / "images/train/nested/a.jpg")
            image(root / "images/train/extra.jpg")
            b = image(root / "images/valid/b.jpg")
            image(root / "custom_train/nested/a.PNG", True)
            image(root / "custom_val/b.png", True)
            (root / "nested.txt").write_text("./images/train/nested/a.jpg\n")
            (root / "train.txt").write_text("./nested.txt\n")
            cfg = root / "data.yaml"
            cfg.write_text(
                yaml.safe_dump(
                    dict(
                        train="train.txt",
                        valid="images/valid",
                        masks=dict(train="custom_train", valid="custom_val"),
                        class_names=["background", "object"],
                        num_classes=2,
                    )
                )
            )
            catalog = load_unet_yaml(cfg)
            training = load_training_sources(cfg)
            self.assertEqual({p[0] for p in catalog["pairs"]}, {a, b})
            self.assertEqual(training["splits"]["train"], [catalog["pairs"][0]])
            self.assertEqual(training["splits"]["val"], [catalog["pairs"][1]])
            self.assertEqual(catalog["class_names"], {0: "object"})

    def test_missing_masks_are_reported_in_catalog_and_rejected_for_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "images/train/good.jpg")
            image(root / "images/train/missing.jpg")
            image(root / "masks/train/good.png", True)
            cfg = root / "data.yaml"
            cfg.write_text("train: images/train\n")
            messages = []
            pairs = load_unet_yaml(cfg, messages.append)["pairs"]
            self.assertEqual(len(pairs), 2)
            self.assertEqual(sum(aux is None for _, aux in pairs), 1)
            self.assertTrue(any("missing" in message for message in messages))
            with self.assertRaises(FileNotFoundError):
                load_training_sources(cfg)

    def test_ambiguous_masks_are_rejected_by_folder_catalog_and_training_readers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "images/train/a.jpg")
            image(root / "masks/train/a.png", True)
            image(root / "masks/train/a.tif", True)
            cfg = root / "data.yaml"
            cfg.write_text("train: images/train\n")
            for operation in (
                lambda: load_unet_yaml(cfg),
                lambda: load_training_sources(cfg),
                lambda: scan_segmentation_pairs(
                    root / "images/train", root / "masks/train"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Неоднозначная"):
                    operation()

    def test_mask_offset_roundtrips_pixel_ids_names_and_training_class_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "images/train/a.png")
            path = root / "masks/train/a.png"
            path.parent.mkdir(parents=True)
            write_image(path, np.full((12, 16), 4, np.uint8))
            for split in ("val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "masks" / split).mkdir(parents=True)
            write_classes_and_yaml(
                root,
                {0: "object"},
                out_yolo_txt=False,
                out_unet_yaml=True,
                out_masks_offset=4,
            )
            cfg = root / "masks_data.yaml"
            document = yaml.safe_load(cfg.read_text())
            self.assertEqual(document["num_classes"], 5)
            self.assertEqual(document["class_names"][4], "object")
            self.assertEqual(load_unet_catalog(cfg).class_names, {4: "object"})
            self.assertEqual(load_training_sources(cfg)["num_classes"], 5)

    def test_one_logit_binary_model_can_name_background_and_foreground(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "images/train/a.png")
            image(root / "masks/train/a.png", True)
            cfg = root / "data.yaml"
            cfg.write_text(
                "train: images/train\nnum_classes: 1\nclass_names: [background, object]\n"
            )
            self.assertEqual(load_training_sources(cfg)["num_classes"], 1)

    def test_sparse_export_class_ids_are_rejected_before_writing_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "dense"):
                write_classes_and_yaml(root, {7: "object"})
            self.assertEqual(list(root.iterdir()), [])

    def test_folder_scanning_keeps_all_images_with_same_stem(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {image(root / "images/a.jpg"), image(root / "images/a.png")}
            labels = root / "labels"
            labels.mkdir()
            (labels / "a.txt").write_text("0 .5 .5 .5 .5\n")
            self.assertEqual(
                {p[0] for p in scan_image_only_pairs(root / "images")}, paths
            )
            pairs, _ = scan_detection_pairs(root / "images", labels)
            self.assertEqual({p[0] for p in pairs}, paths)
            image(root / "masks/a.PNG", True)
            self.assertEqual(
                {
                    p[0]
                    for p in scan_segmentation_pairs(root / "images", root / "masks")
                },
                paths,
            )

    def test_viewer_prefers_canonical_mask_and_does_not_treat_jpeg_as_index_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "images/train/a.jpg")
            image(root / "images/train/a.png")
            expected = image(root / "masks/train/a.PNG", True)
            self.assertEqual(find_mask_for_image(source, root), expected)
            image(root / "masks/train/a.TIF", True)
            with self.assertRaisesRegex(ValueError, "Неоднозначная"):
                find_mask_for_image(source, root)
            other = image(root / "images/train/other.png")
            image(root / "masks/train/other.jpg")
            self.assertIsNone(find_mask_for_image(other, root))

    def test_coco_sidecars_never_overwrite_same_stem_or_existing_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [image(root / "x/Same.jpg"), image(root / "y/same.jpg")]
            output = root / "labels"
            output.mkdir()
            existing = output / "SAME.TXT"
            existing.write_text("original\n")
            items = [
                dict(
                    img_path=path,
                    width=16,
                    height=12,
                    annotations=[("detect", cid, 0.5, 0.5, 0.5, 0.5)],
                )
                for cid, path in enumerate(paths)
            ]
            pairs = write_yolo_txt_for_coco_items(items, output)
            self.assertEqual(len({p[1].casefold() for p in pairs}), 2)
            self.assertEqual(existing.read_text(), "original\n")
            for cid, (_, label) in enumerate(pairs):
                self.assertEqual(
                    read_yolo_labels(label, ImageSize(16, 12))[0].class_id, cid
                )

    def test_mask_conversion_reserves_names_across_calls(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "a.jpg")
            mask = image(root / "mask.png", True)
            output = root / "labels"
            first = write_yolo_txt_for_unet_pairs(
                [(source, mask)], output, class_names={0: "object"}
            )
            first_content = Path(first[0][1]).read_bytes()
            second = write_yolo_txt_for_unet_pairs(
                [(source, mask)], output, class_names={0: "object"}
            )
            self.assertNotEqual(first[0][1], second[0][1])
            self.assertEqual(Path(first[0][1]).read_bytes(), first_content)

    def test_non_crowd_rle_is_an_explicit_bbox_conversion_instead_of_lost_object(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "a.png")
            payload = dict(
                images=[dict(id=1, file_name=source, width=16, height=12)],
                categories=[dict(id=3, name="object")],
                annotations=[
                    dict(
                        id=10,
                        image_id=1,
                        category_id=3,
                        segmentation=dict(size=[12, 16], counts="compressed"),
                        bbox=[3, 2, 10, 8],
                        iscrowd=0,
                    )
                ],
            )
            path = root / "coco.json"
            path.write_text(json.dumps(payload))
            logs = []
            items, _, _ = load_coco(path, logs.append, preserve_category_ids=True)
            self.assertEqual(items[0]["annotations"][0][:2], ("detect", 3))
            self.assertTrue(any("lossy" in message for message in logs))
            del payload["annotations"][0]["bbox"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "RLE"):
                load_coco(path)


class NumericalUtilities(unittest.TestCase):
    def test_canonical_torch_cosine_scheduler_name_is_accepted(self):
        import torch
        from utils_new.training.optimization import get_scheduler

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = get_scheduler(optimizer, "CosineAnnealingLR", 2, eta_min=0.01)
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_zero_weight_class_batches_and_disabled_ce_do_not_poison_losses(self):
        import torch
        from utils_new.training.losses import get_loss_function

        target = torch.ones((2, 4, 5), dtype=torch.long)
        for name, weights in [("CE", None), ("CE+Dice", {"ce": 0.0, "dice": 1.0})]:
            prediction = torch.randn(2, 2, 4, 5, requires_grad=True)
            criterion = get_loss_function(
                name, num_classes=2, class_weights=[1.0, 0.0], loss_weights=weights
            )
            result = criterion(prediction, target)
            self.assertTrue(torch.isfinite(result))
            result.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            if name == "CE":
                self.assertEqual(result.item(), 0.0)
                self.assertEqual(prediction.grad.abs().sum().item(), 0.0)
        prediction = torch.randn(2, 2, 4, 5, requires_grad=True)
        target[0, 0, 0] = -100
        weights = torch.tensor([2.0e-20, 3.0e-20])
        criterion = get_loss_function("CE", 2, ignore_index=-100, class_weights=weights)
        expected = torch.nn.functional.cross_entropy(
            prediction, target, weight=weights, ignore_index=-100
        )
        torch.testing.assert_close(criterion(prediction, target), expected)

    def test_confusion_metrics_validate_counts_and_preserve_empty_foreground_semantics(
        self,
    ):
        from utils_new.segmentation.model_metrics import metrics_from_confusion

        for matrix in ([[1, 2]], [[1, -1], [0, 1]], [[1, float("nan")], [0, 1]]):
            with self.assertRaises(ValueError):
                metrics_from_confusion(matrix, 2)
        metrics = metrics_from_confusion([[4, 1], [2, 3]], 2)
        self.assertAlmostEqual(metrics["IoU"], 0.5)
        self.assertAlmostEqual(metrics["Accuracy"], 0.7)
        self.assertEqual(metrics_from_confusion([[4, 0], [0, 0]], 2)["Dice"], 1.0)

    def test_large_finite_logits_are_not_downcast_to_infinity_and_empty_prediction(
        self,
    ):
        logits = np.array([[[1.0e300, -1.0e300]], [[-1.0e300, 1.0e300]]])
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(decode_output(logits), [[0, 255]])


if __name__ == "__main__":
    unittest.main()
