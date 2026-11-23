"""Round-trip, geometry, unlabelled-data and rollback contracts for dataset exports."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from utils_new.annotations import ImageSize
from utils_new.dataset.preparation import allocate_split_counts
from utils_new.import_export.coco import (
    decode_rle,
    encode_rle,
    export_coco_yolo,
    load_coco,
    write_yolo_txt_for_coco_items,
)
from utils_new.import_export.dataset_export import export_subset
from utils_new.import_export.dataset_generator import DatasetGeneratorCoreNew
from utils_new.import_export.dataset_output import make_index_mask
from utils_new.import_export.dataset_sources import (
    defaults,
    import_save_targets,
    preview,
    write_manifest,
)
from utils_new.import_export.images import read_index_mask, write_image
from utils_new.import_export.mask_yaml import load_unet_yaml
from utils_new.import_export.voc import build_voc_from_yolo, read_voc_pairs
from utils_new.import_export.yolo_labels import read_yolo_labels
from utils_new.import_export.yolo_yaml import load_yolo_yaml


def image(root, name="one"):
    path = root / "images/train" / (name + ".png")
    write_image(path, np.full((24, 40, 3), 137, np.uint8))
    return path


def label(root, name="one", text="0 .5 .5 .5 .5\n"):
    path = root / "labels/train" / (name + ".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def generator(root, pairs, **kwargs):
    options = dict(
        images_folder=str(root / "images"),
        labels_folder=str(root / "labels"),
        output_folder=str(root / "out"),
        pairs=pairs,
        train_pct=1,
        val_pct=0,
        test_pct=0,
        target_size=None,
        augmentations={},
        has_augmentations=False,
        aug_multiplier=1,
        class_names={0: "объект's: A"},
        class_mapping={0: 0},
        bg_color=0,
        input_type=0,
        masks_folder=None,
        num_classes=1,
        mask_class_remap={},
        resize_mode="keep",
        out_masks=True,
        out_unet_yaml=True,
        out_coco=True,
        random_seed=42,
        strict_complete=True,
        include_empty_annotations=True,
    )
    options.update(kwargs)
    return DatasetGeneratorCoreNew(**options)


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


class DatasetExportContracts(unittest.TestCase):
    def test_missing_and_explicit_empty_annotations_survive_every_format(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b, c = [image(root, n) for n in ("one", "empty", "unknown")]
            pairs = [
                (str(a), str(label(root)), None),
                (str(b), str(label(root, "empty", "")), None),
                (str(c), None, None),
            ]
            ok, message = generator(root, pairs).execute()
            self.assertTrue(ok, message)
            output = root / "out"
            self.assertEqual(len(list((output / "images/train").iterdir())), 3)
            self.assertTrue((output / "labels/train/empty.txt").is_file())
            self.assertEqual((output / "labels/train/empty.txt").read_text(), "")
            self.assertFalse((output / "labels/train/unknown.txt").exists())
            self.assertFalse((output / "masks/train/unknown.png").exists())
            self.assertEqual(read_index_mask(output / "masks/train/empty.png").max(), 0)
            records = json.loads((output / "annotations_train.json").read_text())[
                "images"
            ]
            statuses = {
                Path(i["file_name"]).stem: i["annotation_status"] for i in records
            }
            self.assertEqual(
                statuses, dict(one="labelled", empty="empty", unknown="unlabelled")
            )
            items, _, _ = load_coco(output / "annotations_train.json")
            converted = write_yolo_txt_for_coco_items(items, root / "converted")
            self.assertIsNone(dict((Path(i).stem, a) for i, a in converted)["unknown"])
            unet = load_unet_yaml(output / "masks_data.yaml")
            self.assertEqual(len(unet["pairs"]), 3)
            self.assertIsNone(
                dict((Path(i).stem, m) for i, m in unet["pairs"])["unknown"]
            )

    def test_mask_rle_preserves_holes_single_pixels_and_sparse_class_remapping(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = image(root)
            mask = np.zeros((24, 40), np.uint16)
            mask[2:18, 4:30] = 1000
            mask[6:13, 9:20] = 0
            mask[21, 37] = 256
            mask_file = root / "source_mask.png"
            write_image(mask_file, mask)
            ok, message = generator(
                root,
                [(str(source), None, str(mask_file))],
                input_type=1,
                mask_class_remap={1000: 0, 256: 1},
                class_names={0: "ring", 1: "dot"},
                out_yolo_txt=False,
                num_classes=2,
            ).execute()
            self.assertTrue(ok, message)
            payload = json.loads((root / "out/annotations_train.json").read_text())
            self.assertEqual(len(payload["annotations"]), 2)
            expected = np.zeros_like(mask, dtype=np.uint8)
            expected[mask == 1000] = 1
            expected[mask == 256] = 2
            recovered = np.zeros_like(expected)
            for ann in payload["annotations"]:
                binary = decode_rle(ann["segmentation"])
                self.assertEqual(ann["area"], int(binary.sum()))
                ys, xs = np.nonzero(binary)
                self.assertEqual(
                    ann["bbox"],
                    [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ],
                )
                recovered[binary > 0] = ann["category_id"]
            np.testing.assert_array_equal(recovered, expected)
            np.testing.assert_array_equal(
                read_index_mask(root / "out/masks/train/one.png"), expected
            )
            self.assertEqual(payload["annotations"][1]["area"], 1)

    def test_yolo_subtype_does_not_change_other_export_geometry(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = image(root)
            txt = label(root, text="0 .1 .1 .9 .1 .2 .9\n")
            ok, message = generator(
                root, [(str(source), str(txt), None)], out_yolo_subtype="detect"
            ).execute()
            self.assertTrue(ok, message)
            size = ImageSize(40, 24)
            original = read_yolo_labels(txt, size, strict=True)
            np.testing.assert_array_equal(
                read_index_mask(root / "out/masks/train/one.png"),
                make_index_mask(original, size),
            )
            payload = json.loads((root / "out/annotations_train.json").read_text())
            self.assertEqual(len(payload["annotations"][0]["segmentation"][0]), 6)
            self.assertEqual(
                read_yolo_labels(root / "out/labels/train/one.txt", size)[0].kind,
                "detect",
            )

    def test_export_is_portable_and_declared_empty_splits_exist(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = image(root)
            ok, message = generator(
                root, [(str(source), str(label(root)), None)]
            ).execute()
            self.assertTrue(ok, message)
            moved = root / "перенесённый dataset"
            (root / "out").rename(moved)
            for name in ("data.yaml", "masks_data.yaml"):
                self.assertNotIn("path", yaml.safe_load((moved / name).read_text()))
            self.assertEqual(len(load_yolo_yaml(moved / "data.yaml")[0]), 1)
            self.assertEqual(len(load_unet_yaml(moved / "masks_data.yaml")["pairs"]), 1)
            self.assertEqual(len(load_coco(moved / "annotations_train.json")[0]), 1)
            for folder in ("images", "labels", "masks"):
                for split in ("train", "val", "test"):
                    self.assertTrue((moved / folder / split).is_dir())

    def test_repeat_export_requires_policy_and_compatible_merge_preserves_both_runs(
        self,
    ):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = [(str(image(root)), str(label(root)), None)]
            self.assertTrue(generator(root, first).execute()[0])
            old = snapshot(root / "out")
            self.assertFalse(generator(root, first).execute()[0])
            self.assertEqual(old, snapshot(root / "out"))
            self.assertFalse(
                generator(
                    root,
                    first,
                    existing_output_policy="merge",
                    class_names={0: "different"},
                ).execute()[0]
            )
            self.assertEqual(old, snapshot(root / "out"))
            self.assertTrue(
                generator(root, first, existing_output_policy="merge").execute()[0]
            )
            merged = json.loads((root / "out/annotations_train.json").read_text())
            self.assertEqual(len(merged["images"]), 2)
            self.assertEqual(len({i["id"] for i in merged["images"]}), 2)
            self.assertEqual(len({a["id"] for a in merged["annotations"]}), 2)
            self.assertEqual(
                len({i["file_name"].casefold() for i in merged["images"]}), 2
            )
            self.assertEqual(len(list((root / "out/images/train").iterdir())), 2)
            manifest = json.loads((root / "out/generation_manifest.json").read_text())
            self.assertEqual(manifest["variants_created"], 2)
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertTrue(
                generator(root, first, existing_output_policy="replace").execute()[0]
            )
            self.assertEqual(len(list((root / "out/images/train").iterdir())), 1)

    def test_yolo_channels_are_transactional_and_merge_rejects_a_different_mode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [(str(image(root)), str(label(root)), None)]
            ok, message = generator(root, pairs, yolo_channels=1).execute()
            self.assertTrue(ok, message)
            self.assertEqual(
                yaml.safe_load((root / "out/data.yaml").read_text())["channels"], 1
            )
            old = snapshot(root / "out")
            ok, message = generator(
                root, pairs, yolo_channels=3, existing_output_policy="merge"
            ).execute()
            self.assertFalse(ok)
            self.assertEqual(snapshot(root / "out"), old)
            ok, message = generator(
                root, pairs, yolo_channels=4, output_folder=str(root / "bad")
            ).execute()
            self.assertFalse(ok)
            self.assertIn("channels", message)
            self.assertFalse((root / "bad").exists())

    def test_publish_failure_rolls_back_replacement(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [(str(image(root)), str(label(root)), None)]
            self.assertTrue(generator(root, pairs).execute()[0])
            before = snapshot(root / "out")
            original = os.replace
            failed = False

            def fail_once(source, target):
                nonlocal failed
                if not failed and Path(target) == root / "out/data.yaml":
                    failed = True
                    raise OSError("injected publish failure")
                return original(source, target)

            with patch(
                "utils_new.import_export.dataset_generator.os.replace",
                side_effect=fail_once,
            ):
                ok, message = generator(
                    root, pairs, existing_output_policy="replace"
                ).execute()
            self.assertFalse(ok)
            self.assertIn("injected", message)
            self.assertEqual(snapshot(root / "out"), before)

    def test_zero_share_split_and_nonfinite_ratios(self):
        for count in range(1, 20):
            allocated = allocate_split_counts(count, (0.8, 0.2, 0))
            self.assertEqual(sum(allocated), count)
            self.assertEqual(allocated[2], 0)
        for ratios in ((float("nan"), 0.5, 0.5), (0.8, 0.21, 0), (-1, 1, 1)):
            with self.assertRaises(ValueError):
                allocate_split_counts(5, ratios)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [(str(image(root)), str(label(root)), None)]
            ok, message = generator(
                root, pairs, split_by_source=False, train_pct=0.8, val_pct=0.2
            ).execute()
            self.assertTrue(ok, message)
            self.assertEqual(list((root / "out/images/test").iterdir()), [])

    def test_images_only_yaml_has_no_false_class_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ok, message = generator(
                root, [(str(image(root)), None, None)], input_type=2
            ).execute()
            self.assertTrue(ok, message)
            config = yaml.safe_load((root / "out/data.yaml").read_text())
            self.assertEqual(config["dataset_type"], "images_only")
            self.assertNotIn("names", config)
            self.assertFalse((root / "out/classes.txt").exists())

    def test_subset_rejects_traversal_and_windows_name_collisions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = image(root, "same"), image(root, "SAME")
            with self.assertRaises(ValueError):
                export_subset(
                    [(str(a), None)],
                    root / "out",
                    {},
                    dataset_type=2,
                    split_by_image={str(a): ["../../escaped"]},
                )
            with self.assertRaises(ValueError):
                export_subset(
                    [(str(a), None), (str(b), None)], root / "out", {}, dataset_type=2
                )
            self.assertFalse((root / "out").exists())

    def test_voc_honors_reviewed_source_and_preserves_unlabelled_images(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = image(root), image(root, "unknown")
            for img in (a, b):
                img.with_suffix(".txt").write_text("1 .5 .5 .8 .8\n")
            label(root)
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(dict(train="images/train", names=["selected", "stale"]))
            )
            approved = write_manifest(
                preview(defaults([source], "yolo"), "labels"), root / "approved"
            )[0]
            target = Path(build_voc_from_yolo(approved, root / "export"))
            self.assertIn("selected", (target / "Annotations/one.xml").read_text())
            self.assertFalse((target / "Annotations/unknown.xml").exists())
            self.assertEqual(
                (target / "JPEGImages/one.png").read_bytes(), a.read_bytes()
            )
            self.assertEqual(
                len((target / "ImageSets/All/train.txt").read_text().splitlines()), 2
            )
            self.assertEqual(
                len((target / "ImageSets/Main/train.txt").read_text().splitlines()), 1
            )
            self.assertTrue((target / "voc.yaml").is_file())

    def test_voc_mask_metadata_survives_relocation_and_rejects_wrong_size(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image(root)
            mask = root / "masks/train/one.png"
            write_image(mask, np.full((24, 40), 2, np.uint8))
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(
                    dict(
                        train_images="images/train",
                        train_masks="masks/train",
                        num_classes=3,
                        class_names=["background", "A", "B"],
                        mask_class_offset=1,
                    )
                )
            )
            approved = write_manifest(
                preview(defaults([source], "unet"), "declared"), root / "approved"
            )[0]
            target = Path(build_voc_from_yolo(approved, root / "export"))
            moved = root / "moved"
            target.rename(moved)
            cfg = yaml.safe_load((moved / "voc.yaml").read_text())
            self.assertEqual(cfg["num_classes"], 3)
            self.assertEqual(cfg["class_names"], ["background", "A", "B"])
            self.assertEqual(
                len(
                    read_voc_pairs(
                        moved / cfg["train_txt"],
                        moved / cfg["jpeg_dir"],
                        moved / cfg["seg_dir"],
                    )
                ),
                1,
            )
            write_image(mask, np.zeros((2, 3), np.uint8))
            with self.assertRaises(ValueError):
                build_voc_from_yolo(approved, root / "broken")
            self.assertFalse((root / "broken/VOC").exists())

    def test_editor_save_destinations_are_durable_and_update_reads(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            img = image(root, "nested/one")
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(dict(train="images/train", names=["object"]))
            )
            approved = write_manifest(
                preview(defaults([source], "yolo"), "none"), root / "temporary"
            )[0]
            targets = import_save_targets([approved], [str(img)])
            shutil.rmtree(root / "temporary")
            self.assertEqual(
                Path(targets[str(img)]), root / "labels/train/nested/one.txt"
            )
            if os.environ.get("NEURO_RUN_GUI_TESTS") != "1":
                return
            from tabs.layout_dataset import Labeler
            from tabs.viewing_dataset import ViewingDataset

            for cls in (Labeler, ViewingDataset):
                for kind in ("yolo", "unet", "coco"):
                    stub = SimpleNamespace(_save_source_kind=kind, _yaml_label_map={})
                    cls._after_label_saved(stub, str(img), targets[str(img)])
                    self.assertEqual(stub._yaml_label_map[str(img)], targets[str(img)])

    def test_yolo_detect_obb_and_segment_export_roundtrips(self):
        from utils_new.annotations import polygon_area

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            img = image(root)
            txt = label(root)
            for subtype, obb_format in (
                ("detect", "corners"),
                ("obb", "corners"),
                ("obb", "xywhr"),
                ("segment", "corners"),
            ):
                with self.subTest(subtype=subtype, obb_format=obb_format):
                    output = root / (subtype + obb_format)
                    ok, message = generator(
                        root,
                        [(str(img), str(txt), None)],
                        output_folder=str(output),
                        out_yolo_subtype=subtype,
                        out_yolo_obb_format=obb_format,
                    ).execute()
                    self.assertTrue(ok, message)
                    exported = read_yolo_labels(
                        output / "labels/train/one.txt", ImageSize(40, 24), strict=True
                    )
                    self.assertEqual(exported[0].kind, subtype)
                    self.assertAlmostEqual(
                        polygon_area(exported[0].points_px), 240, places=3
                    )

    def test_generated_variants_and_later_merges_stay_with_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [
                (str(image(root, n)), str(label(root, n, "0 .3 .4 .2 .3\n")), None)
                for n in ("one", "two")
            ]
            options = dict(
                has_augmentations=True,
                aug_multiplier=2,
                augmentations={"horizontal_flip": True},
                train_pct=0.5,
                val_pct=0.5,
            )
            ok, message = generator(root, pairs, **options).execute()
            self.assertTrue(ok, message)
            before = json.loads((root / "out/generation_manifest.json").read_text())
            memberships = {
                row["source"]: split
                for split, rows in before["splits"].items()
                for row in rows
            }
            ok, message = generator(
                root, pairs, existing_output_policy="merge", random_seed=900, **options
            ).execute()
            self.assertTrue(ok, message)
            manifest = json.loads((root / "out/generation_manifest.json").read_text())
            for split, rows in manifest["splits"].items():
                for row in rows:
                    self.assertEqual(memberships[row["source"]], split)
            self.assertEqual(manifest["variants_created"], 8)

    def test_rle_known_column_order_and_invalid_data(self):
        mask = np.array([[0, 1, 0], [1, 1, 0]], np.uint8)
        self.assertEqual(encode_rle(mask)["counts"], [1, 3, 2])
        np.testing.assert_array_equal(decode_rle(encode_rle(mask)), mask)
        for bad in (
            {"size": [2, 3], "counts": [4]},
            {"size": [0, 2], "counts": [0]},
            {"size": [2, 3], "counts": [-1, 7]},
            {"size": [2, 3], "counts": "~"},
        ):
            with self.assertRaises(ValueError):
                decode_rle(bad)

    def test_coco_yolo_preserves_all_polygons_and_homogeneous_task_with_dense_ids(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            img = image(root, "nested/one")
            path = root / "data.json"
            path.write_text(
                json.dumps(
                    dict(
                        images=[
                            dict(
                                id=4,
                                file_name=img.relative_to(root).as_posix(),
                                width=40,
                                height=24,
                            )
                        ],
                        categories=[dict(id=9, name="part's: A")],
                        annotations=[
                            dict(
                                id=1,
                                image_id=4,
                                category_id=9,
                                segmentation=[
                                    [2, 2, 10, 2, 10, 10],
                                    [20, 10, 30, 10, 30, 20],
                                ],
                            ),
                            dict(id=2, image_id=4, category_id=9, bbox=[1, 1, 4, 4]),
                        ],
                    )
                )
            )
            names = export_coco_yolo(path, root / "export", subtype="segment")
            self.assertEqual(names, ["part's: A"])
            values = read_yolo_labels(
                root / "export/labels/train/nested/one.txt",
                ImageSize(40, 24),
                strict=True,
            )
            self.assertEqual(len(values), 3)
            self.assertEqual({v.kind for v in values}, {"segment"})
            self.assertEqual({v.class_id for v in values}, {0})
            with self.assertRaises(ValueError):
                export_coco_yolo(path, root / "other", class_names=["wrong"])


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "Qt optional")
class SahiExportContracts(unittest.TestCase):
    def test_sahi_uses_canonical_obb_coordinates_and_polygon_area(self):
        from tabs.slice_dataset_sahi import yolo_labels_to_coco_annotations

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            txt = label(root, text="0 .5 .5 .4 .2 .7\n")
            records = yolo_labels_to_coco_annotations(txt, 100, 30, "obb_xywhr")
            self.assertAlmostEqual(records[0]["area"], 240, places=3)
            self.assertEqual(len(records[0]["segmentation"][0]), 8)
            txt.write_text("0 .1 .1 .9 .1 .2 .9\n")
            records = yolo_labels_to_coco_annotations(txt, 100, 30, "segment")
            self.assertAlmostEqual(records[0]["area"], 960, places=3)
            self.assertNotEqual(
                records[0]["area"], records[0]["bbox"][2] * records[0]["bbox"][3]
            )

    def test_negative_zero_split_zero_and_no_external_deletion(self):
        from tabs.slice_dataset_sahi import (
            subsample_negative_tiles,
            split_and_move,
            write_yolo_yaml,
        )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiles = root / "tiles"
            tiles.mkdir()
            (tiles / "one.png").write_bytes(b"tile")
            payload = dict(
                images=[dict(id=1, file_name="one.png")], annotations=[], categories=[]
            )
            original = deepcopy(payload)
            self.assertEqual(subsample_negative_tiles(payload, tiles, 0)["images"], [])
            self.assertEqual(payload, original)
            self.assertFalse((tiles / "one.png").exists())
            (tiles / "one.png").write_bytes(b"tile")
            result, _ = split_and_move(payload, tiles, train_ratio=0)
            self.assertEqual(result["images"][0]["file_name"], "images/val/one.png")
            self.assertTrue((tiles / "images/val/one.png").exists())
            external = root / "external.png"
            external.write_bytes(b"original")
            payload["images"][0]["file_name"] = str(external)
            with self.assertRaises(ValueError):
                subsample_negative_tiles(payload, tiles, 0)
            with self.assertRaises(ValueError):
                split_and_move(payload, tiles)
            self.assertEqual(external.read_bytes(), b"original")
            write_yolo_yaml(tiles, ["part's: вещь"])
            cfg = yaml.safe_load((tiles / "data.yaml").read_text())
            self.assertEqual(cfg["names"], ["part's: вещь"])
            self.assertNotIn("path", cfg)

    def test_sahi_unknown_tiles_are_not_background_and_source_groups_do_not_leak(self):
        from tabs.slice_dataset_sahi import subsample_negative_tiles, split_and_move

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            for index in range(6):
                name = f"tile_{index}.png"
                (root / name).write_bytes(b"tile")
                images.append(
                    dict(
                        id=index,
                        file_name=name,
                        source_image=str(index // 3),
                        annotation_status="unlabelled" if index < 3 else "labelled",
                    )
                )
            data = dict(
                images=images, annotations=[dict(id=1, image_id=3)], categories=[]
            )
            filtered = subsample_negative_tiles(data, root, 0)
            self.assertEqual({i["id"] for i in filtered["images"]}, {0, 1, 2, 3})
            final, _ = split_and_move(filtered, root, train_ratio=0.5)
            memberships = {}
            for img in final["images"]:
                memberships.setdefault(img["source_image"], set()).add(
                    Path(img["file_name"]).parts[1]
                )
            self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))

    def test_real_preparation_generate_action_publishes_selected_formats(self):
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from tabs.preparing_dataset_yaml import DatasetPreparationWindow

        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image(root)
            label(root)
            source = root / "data.yaml"
            source.write_text(
                yaml.safe_dump(dict(train="images/train", names=["object"]))
            )
            window = DatasetPreparationWindow()
            try:
                window._load_yolo_yaml_from_path(str(source))
                window.output_folder = str(root / "generated")
                window.resize_keep.setChecked(True)
                window.train_slider.setValue(100)
                window.channels_combo.setCurrentIndex(1)
                window.out_masks.setChecked(True)
                window.out_unet_yaml.setChecked(True)
                window.out_coco.setChecked(True)
                with (
                    patch.object(QMessageBox, "information"),
                    patch.object(QMessageBox, "critical") as errors,
                    patch.object(QMessageBox, "warning") as warnings,
                ):
                    window.on_generate()
                    self.assertIsNotNone(window.generator_thread)
                    self.assertTrue(window.generator_thread.wait(30000))
                    app.processEvents()
                    errors.assert_not_called()
                    warnings.assert_not_called()
                output = root / "generated"
                for relative in (
                    "data.yaml",
                    "masks_data.yaml",
                    "annotations_train.json",
                    "labels/train/one.txt",
                    "masks/train/one.png",
                ):
                    self.assertTrue((output / relative).is_file(), relative)
                self.assertEqual(
                    yaml.safe_load((output / "data.yaml").read_text())["channels"], 1
                )
            finally:
                if (
                    window.generator_thread is not None
                    and window.generator_thread.isRunning()
                ):
                    window.generator_thread.cancel()
                    window.generator_thread.wait()
                window.close()
                app.processEvents()

    def test_export_ui_defaults_and_split_slider_total(self):
        from PyQt5.QtWidgets import QApplication
        from tabs.preparing_dataset_yaml import DatasetPreparationWindow

        app = QApplication.instance() or QApplication([])
        window = DatasetPreparationWindow()
        try:
            self.assertEqual(window.existing_output_policy.currentData(), "error")
            self.assertTrue(window.group_augmentations_by_source.isChecked())
            for slider, value in (
                (window.train_slider, 100),
                (window.val_slider, 0),
                (window.test_slider, 0),
                (window.train_slider, 80),
            ):
                slider.setValue(value)
            self.assertEqual(
                sum(
                    s.value()
                    for s in (
                        window.train_slider,
                        window.val_slider,
                        window.test_slider,
                    )
                ),
                100,
            )
            window.pairs = [("one", None)]
            window.update_split_counts()
            self.assertEqual(window.test_count_label.text(), "0")
        finally:
            window.close()
            app.processEvents()

    def test_failed_sahi_pipeline_does_not_publish_partial_files(self):
        from tabs.slice_dataset_sahi import SliceWorker

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result"
            output.mkdir()
            worker = SliceWorker(dict(out=str(output)))

            def fail():
                (Path(worker.params["out"]) / "partial.txt").write_text("partial")
                raise ValueError("injected")

            with patch.object(worker, "_pipeline_staged", side_effect=fail):
                with self.assertRaisesRegex(ValueError, "injected"):
                    worker._pipeline()
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(worker.params["out"], str(output))
