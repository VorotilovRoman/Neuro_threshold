"""Training migration: real gradients, raw mask IDs, isolated workers and safe stop."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from utils_new.configuration.training import semantic_config
from utils_new.dataset.segmentation import ImageMaskDataset
from utils_new.import_export.images import read_index_mask, write_image
from utils_new.import_export.mask_yaml import load_training_sources
from utils_new.import_export.training import read_model_meta, save_weights
from utils_new.import_export.voc import build_voc_from_yolo, read_voc_pairs
from utils_new.segmentation.model_metrics import segmentation_metrics
from utils_new.training.losses import get_loss_function
from utils_new.training.optimization import (
    layerwise_optimizer,
    encoder_stages,
    predict_batch_multiscale,
)
from utils_new.training.protocol import LineDecoder, StopRequest
from utils_new.training.semantic import optimize_epoch, evaluate, run_training

ROOT = Path(__file__).resolve().parents[1]


def make_data(root, train=3, val=1, test=1):
    for split, count in [("train", train), ("val", val), ("test", test)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            write_image(
                root / "images" / split / (str(index) + ".png"),
                np.random.default_rng(index).integers(
                    0, 256, (23, 45, 3), dtype=np.uint8
                ),
            )
            mask = np.zeros((23, 45), np.uint8)
            mask[4:19, 8:35] = 1
            write_image(root / "masks" / split / (str(index) + ".png"), mask)


def config(data, output, **extra):
    return semantic_config(
        dict(
            dataset_path=str(data),
            project_path=str(output),
            encoder="resnet18",
            backbone="resnet18",
            pretrained=False,
            input_size=32,
            num_classes=2,
            batch_size=2,
            epochs=1,
            num_workers=0,
            device="cpu",
            decoder_channels=[16, 8, 4, 2, 1],
            use_ema=False,
            use_amp=False,
            plots=False,
            export_onnx=False,
            **extra,
        ),
        "unet",
    )


class Tiny(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, 1)
        self.decoder = nn.Conv2d(4, channels, 1)

    def forward(self, x):
        return self.decoder(torch.tanh(self.encoder(x)))


class TrainingContracts(unittest.TestCase):
    def test_palette_mask_ids_and_void_are_not_converted_to_rgb_or_foreground(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            mask = root / "mask.png"
            write_image(image, np.zeros((8, 12, 3), np.uint8))
            values = np.zeros((8, 12), np.uint8)
            values[:, 3:6] = 7
            values[:, 6:] = 255
            palette = Image.fromarray(values).convert("P")
            palette.putpalette([i % 256 for i in range(768)])
            palette.save(mask)
            np.testing.assert_array_equal(read_index_mask(mask), values)
            ds = ImageMaskDataset([(str(image), str(mask))], 8, 8, 255, is_train=False)
            _, result = ds[0]
            self.assertEqual(set(result.unique().tolist()), {0, 7, 255})
            ds = ImageMaskDataset([(str(image), str(mask))], 8, 1, 255, is_train=False)
            self.assertEqual(set(ds[0][1].unique().tolist()), {0, 1, 255})
            signed = root / "signed.tif"
            Image.fromarray(np.array([[0, -100], [1, 0]], np.int32)).save(signed)
            with self.assertRaises(ValueError):
                read_index_mask(signed)
            np.testing.assert_array_equal(
                read_index_mask(signed, ignore_index=-100), [[0, -100], [1, 0]]
            )

    def test_train_and_inference_preprocessing_match_and_validation_is_deterministic(
        self,
    ):
        from utils_new.segmentation.backends.deep_learning import UNetSegmentor

        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root)
            image = root / "images/train/0.png"
            mask = root / "masks/train/0.png"
            import cv2

            for mode in ("hard", "keep_aspect_pad"):
                ds = ImageMaskDataset(
                    [(str(image), str(mask))],
                    32,
                    2,
                    resize_mode=mode,
                    normalize=True,
                    is_train=False,
                )
                adapter = UNetSegmentor()
                adapter.metadata = dict(normalize=True, resize_mode=mode)
                tensor, *_ = adapter._preprocess(cv2.imread(str(image)), 32)
                torch.testing.assert_close(ds[0][0], tensor[0], rtol=0, atol=0)
            ds = ImageMaskDataset(
                [(str(image), str(mask))],
                32,
                2,
                augmentations=dict(rotate=True, noise=True),
                resize_mode="random_crop",
                is_train=False,
            )
            first = ds[0]
            np.random.seed(1)
            second = ds[0]
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])

    def test_yaml_keeps_nested_list_membership_and_rejects_missing_masks_or_split_leakage(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root, train=2)
            (root / "lists").mkdir()
            (root / "lists/train.txt").write_text("../images/train/0.png\n")
            path = root / "masks.yaml"
            path.write_text(
                yaml.safe_dump(
                    dict(
                        train_images="lists/train.txt",
                        train_masks="masks/train",
                        val_images="images/val",
                        val_masks="masks/val",
                        num_classes=2,
                    )
                )
            )
            result = load_training_sources(path)
            self.assertEqual(len(result["splits"]["train"]), 1)
            self.assertTrue(result["splits"]["train"][0][0].endswith("train/0.png"))
            data = yaml.safe_load(path.read_text())
            data.update(val_images="images/train", val_masks="masks/train")
            path.write_text(yaml.safe_dump(data))
            with self.assertRaisesRegex(ValueError, "нескольких split"):
                load_training_sources(path)
            path.write_text(
                yaml.safe_dump(dict(train_images="images/train", train_masks="missing"))
            )
            with self.assertRaises(FileNotFoundError):
                load_training_sources(path)

    def test_all_losses_handle_void_and_device_class_weights_without_nan(self):
        for classes in (1, 3):
            logits = torch.randn(1, classes, 4, 5, requires_grad=True)
            target = torch.full((1, 4, 5), 255, dtype=torch.long)
            for loss in (
                "CrossEntropy",
                "Dice",
                "Focal",
                "Tversky",
                "CrossEntropy+Dice",
                "CrossEntropy+Focal",
                "Dice+Focal",
            ):
                criterion = get_loss_function(loss, classes, ignore_index=255)
                result = criterion(logits, target)
                self.assertEqual(float(result.detach()), 0.0)
                result.backward(retain_graph=True)
            self.assertEqual(float(logits.grad.abs().sum()), 0.0)
        logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
        labels = torch.tensor([[[0, 1, 255]]])
        loss = get_loss_function("BCE", 1, 255, class_weights=[1, 2])(logits, labels)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[0, 0, 0, 1]), -0.5)
        self.assertEqual(float(logits.grad[0, 0, 0, 2]), 0.0)
        with self.assertRaises(ValueError):
            get_loss_function("typo", 2)

    def test_accumulation_matches_real_batch_including_short_final_group(self):
        torch.manual_seed(3)
        model = Tiny()
        initial = deepcopy(model.state_dict())
        images = torch.rand(5, 3, 4, 4)
        masks = torch.randint(0, 2, (5, 4, 4))
        cfg = semantic_config(
            dict(
                input_size=32,
                num_classes=1,
                epochs=1,
                accum_steps=2,
                grad_clip=1000,
                pretrained=False,
            ),
            "unet",
        )
        cfg["_frozen"] = False
        criterion = get_loss_function("BCE", 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        singles = [(images[i : i + 1], masks[i : i + 1]) for i in range(5)]
        optimize_epoch(
            model,
            singles,
            criterion,
            optimizer,
            torch.device("cpu"),
            cfg,
            StopRequest(),
        )
        reference = Tiny()
        reference.load_state_dict(initial)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for start in (0, 2, 4):
            optimizer.zero_grad()
            criterion(
                reference(images[start : start + 2]), masks[start : start + 2]
            ).backward()
            optimizer.step()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(
                value, reference.state_dict()[key], rtol=1e-5, atol=1e-7
            )

    def test_layerwise_groups_cover_each_trainable_parameter_once_for_dict_stages(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(3, 4, 1)
                self.last = nn.Conv2d(4, 4, 1)

            def get_stages(self):
                return {4: [self.stem], 8: [self.last]}

        model = Tiny()
        model.encoder = Encoder()
        optimizer = layerwise_optimizer(model, "AdamW", 0.01)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(encoder_stages(model.encoder)), 2)
        self.assertLess(
            optimizer.param_groups[0]["lr"], optimizer.param_groups[-1]["lr"]
        )

    def test_single_scale_is_applied_and_streaming_metrics_equal_full_tensor_metrics(
        self,
    ):
        seen = []

        class Network(Tiny):
            def forward(self, x):
                seen.append(tuple(x.shape[-2:]))
                return super().forward(x)

        model = Network()
        image = torch.rand(2, 3, 64, 64)
        result = predict_batch_multiscale(
            model, image, torch.device("cpu"), [0.5], False
        )
        self.assertEqual(seen, [(32, 32)])
        self.assertEqual(result.shape, (2, 1, 64, 64))
        cfg = semantic_config(dict(input_size=32, num_classes=1, use_amp=False), "unet")
        masks = torch.randint(0, 2, (2, 64, 64))
        model.eval()
        with torch.no_grad():
            expected = segmentation_metrics(model(image), masks, 1)
        _, metrics, _, retained = evaluate(
            model,
            [(image[:1], masks[:1]), (image[1:], masks[1:])],
            get_loss_function("BCE", 1),
            torch.device("cpu"),
            cfg,
            StopRequest(),
        )
        self.assertEqual(metrics, expected)
        self.assertIsNone(retained)

    def test_real_worker_training_keeps_per_checkpoint_scores_and_loads_for_inference(
        self,
    ):
        from utils_new.segmentation.backends.deep_learning import UNetSegmentor

        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root / "data")
            parameters = config(root / "data", root / "run")
            self.assertEqual(run_training(parameters, "unet"), 0)
            for name in ("best_loss", "best_iou", "last"):
                path = root / "run/weights" / (name + ".pth")
                self.assertTrue(path.is_file())
                metadata = read_model_meta(path)
                self.assertEqual(metadata["epoch"], 1)
                self.assertIsNotNone(metadata["validation_metrics"])
                self.assertEqual("test_metrics" in metadata, name == "best_iou")
            adapter = UNetSegmentor()
            adapter.load(weights_path=str(root / "run/weights/best_iou.pth"))
            self.assertEqual(
                adapter.predict(np.zeros((21, 39, 3), np.uint8)).shape, (21, 39)
            )

    def test_two_phase_ema_and_cooperative_stop_keep_last_complete_epoch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root / "data")
            stop = StopRequest()
            lines = []

            def logger(value):
                lines.append(value)
                if value.startswith("PROGRESS:") and ":2:2" in value:
                    stop.set()

            params = config(
                root / "data",
                root / "run",
                two_phase_training=True,
                freeze_epochs=1,
                unfreeze_epochs=1,
                use_layerwise_lr=True,
            )
            params["use_ema"] = True
            params["scheduler"] = "Poly"
            with patch("utils_new.training.semantic.build_model", return_value=Tiny(2)):
                self.assertEqual(
                    run_training(params, "unet", stop=stop, log=logger), 130
                )
            self.assertTrue(any(line.startswith("[PHASE]") for line in lines))
            last = root / "run/weights/last.pth"
            self.assertEqual(read_model_meta(last)["epoch"], 2)
            self.assertTrue(read_model_meta(last)["epoch_complete"])
            self.assertEqual(read_model_meta(last)["weights_source"], "ema")
            self.assertTrue((root / "run/weights/interrupted_epoch_2.pth").exists())

    def test_atomic_weights_failure_preserves_previous_weights_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "last.pth"
            model = Tiny()
            save_weights(path, model, dict(epoch=1))
            before = path.read_bytes()
            metadata = read_model_meta(path)
            with patch(
                "utils_new.import_export.atomic.os.replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    save_weights(path, model, dict(epoch=2))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_model_meta(path), metadata)

    def test_voc_keeps_index_one_and_sparse_class_names_and_xml_uses_real_image_name(
        self,
    ):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root / "data", train=1, val=0, test=0)
            label = root / "data/labels/train"
            label.mkdir(parents=True)
            # Explicit Segment: XML must use enclosing bounds, not misread the first four numbers as xywh.
            (label / "0.txt").write_text("5 .1 .1 .9 .1 .8 .8\n")
            source = root / "data/data.yaml"
            source.write_text(
                yaml.safe_dump(
                    dict(
                        train="images/train",
                        train_masks="masks/train",
                        names={5: "object"},
                    )
                )
            )
            destination = Path(build_voc_from_yolo(source, root / "export"))
            np.testing.assert_array_equal(
                read_index_mask(destination / "SegmentationClass/0.png"),
                read_index_mask(root / "data/masks/train/0.png"),
            )
            doc = ET.parse(destination / "Annotations/0.xml").getroot()
            self.assertEqual(doc.findtext("filename"), "0.png")
            self.assertEqual(doc.findtext("object/name"), "object")
            self.assertGreaterEqual(int(doc.findtext("object/bndbox/xmin")), 1)
            self.assertEqual(
                len(
                    read_voc_pairs(
                        destination / "ImageSets/Segmentation/train.txt",
                        destination / "JPEGImages",
                        destination / "SegmentationClass",
                    )
                ),
                1,
            )
            with self.assertRaises(FileExistsError):
                build_voc_from_yolo(source, root / "export")

    def test_experiment_runner_separates_test_metrics_and_streams_grid(self):
        import io
        from utils_new.training.experiments import run_experiments

        class Child:
            returncode = 0

            def __init__(self, *args, **kwargs):
                self.stdout = io.StringIO(
                    'TEST_METRICS: {"IoU": 0.1}\nMETRICS: {"IoU": 0.8}\nRESULT_DIR:/tmp/trial\n'
                )

            def wait(self, *args, **kwargs):
                return 0

        with TemporaryDirectory() as directory:
            with patch(
                "utils_new.training.experiments.subprocess.Popen", side_effect=Child
            ) as launch:
                rows = run_experiments(
                    "unet",
                    dict(lr=[0.001, 0.01], epochs=[1]),
                    Path(directory) / "runs",
                    log=lambda text: None,
                )
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["metrics"], {"IoU": 0.8})
                self.assertEqual(row["test_metrics"], {"IoU": 0.1})
                self.assertEqual(row["status"], "completed")
            self.assertIn("utils_new.training.unet", launch.call_args.args[0])
            self.assertIn("--params_file", launch.call_args.args[0])

    def test_moved_ae_profile_path_and_signal_handlers_are_preserved(self):
        import signal
        from utils_new.training.autoencoder import main

        before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        with (
            TemporaryDirectory() as directory,
            patch("utils_new.autoencoders.training.run_training") as trainer,
        ):
            trainer.return_value = {"calibration": {"metric": "mae"}}
            self.assertEqual(
                main(
                    [
                        "--profile",
                        "conv_bn_baseline",
                        "--data",
                        directory,
                        "--output",
                        directory,
                        "--device",
                        "cpu",
                    ]
                ),
                0,
            )
            self.assertEqual(trainer.call_args.args[0].model["arch"], "conv")
        self.assertEqual(
            before, (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        )

    def test_workers_have_no_import_side_effects_and_fragmented_protocol_roundtrips(
        self,
    ):
        code = """import signal,sys
before=(signal.getsignal(signal.SIGINT),signal.getsignal(signal.SIGTERM),sys.stdout)
import utils_new.training.unet,utils_new.training.deeplab,utils_new.training.yolo,utils_new.training.autoencoder,utils_new.training.experiments
assert before==(signal.getsignal(signal.SIGINT),signal.getsignal(signal.SIGTERM),sys.stdout)
assert 'torch' not in sys.modules and 'PyQt5' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        decoder = LineDecoder()
        data = "PROGRESS:50:1:2\nRESULT_DIR:снимки\nготово".encode()
        lines = []
        for value in data:
            lines += decoder.feed(bytes([value]))
        lines += decoder.feed(final=True)
        self.assertEqual(lines, ["PROGRESS:50:1:2", "RESULT_DIR:снимки", "готово"])
        self.assertFalse((ROOT / "utils_training").exists())

    def test_cli_output_is_utf8_even_when_parent_forces_ascii(self):
        modules = [
            "utils_new.training." + kind
            for kind in ("unet", "deeplab", "yolo", "autoencoder", "experiments")
        ] + ["utils_new.import_export.voc"]
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT,
                    env=dict(os.environ, PYTHONIOENCODING="ascii"),
                    capture_output=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
                self.assertIn("--help", result.stdout.decode("utf-8"))


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in GUI checks")
class TrainingGui(unittest.TestCase):
    def gui(self, code):
        code = ("from unittest.mock import patch as _source_patch\n"
                "_source_patch('ui.dataset_source.confirm_dataset_import', side_effect=lambda owner, paths, kind: paths).start()\n" + code)
        with TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                MPLCONFIGDIR=directory,
                XDG_CACHE_HOME=directory,
                XDG_CONFIG_HOME=directory,
                YOLO_CONFIG_DIR=directory,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stop_file_and_fragmented_semantic_protocol_keep_ui_responsive(self):
        self.gui("""
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QProcess,QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.deep_learning_training import YoloTrainWindow
app=QApplication([]);window=YoloTrainWindow();window.show()
temporary=tempfile.TemporaryDirectory();stop=Path(temporary.name)/'stop'
window._worker_artifacts['unet']=(temporary,stop)
process=QProcess();window.unet_process=process;process.setProcessChannelMode(QProcess.MergedChannels)
process.readyReadStandardOutput.connect(window._on_unet_output);process.finished.connect(window._on_unet_finished)
child="import pathlib,sys,time;print('PROGRESS:50:1:2',flush=True);print('RESULT_DIR:'+sys.argv[2],flush=True);print('готово',flush=True)\\nwhile not pathlib.Path(sys.argv[1]).exists():time.sleep(.01)\\nsys.exit(130)"
with patch.object(QMessageBox,'information') as info,patch.object(QMessageBox,'critical') as error:
 process.start(sys.executable,['-u','-c',child,str(stop),temporary.name]);assert process.waitForStarted(3000)
 timer=QElapsedTimer();timer.start()
 while 'готово' not in window.log_text.toPlainText() and timer.elapsed()<5000:QTest.qWait(10)
 assert window.unet_train_widget.progress_bar.value()==50
 window.on_stop_unet()
 while window.unet_process is not None and timer.elapsed()<10000:QTest.qWait(10)
 assert window.unet_process is None
 assert 'остановлено пользователем' in window.log_text.toPlainText()
 info.assert_not_called();error.assert_not_called()
window.close();app.processEvents()
""")

    def test_actual_unet_process_launch_from_training_tab(self):
        self.gui("""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.deep_learning_training import YoloTrainWindow
app=QApplication([]);window=YoloTrainWindow();window.show()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)/'данные'
 for split in ('train','val'):
  (root/'images'/split).mkdir(parents=True);(root/'masks'/split).mkdir(parents=True)
  for index in range(2):
   Image.fromarray(np.zeros((24,40,3),np.uint8)).save(root/'images'/split/(str(index)+'.png'))
   mask=np.zeros((24,40),np.uint8);mask[4:20,8:30]=1
   Image.fromarray(mask).save(root/'masks'/split/(str(index)+'.png'))
 params=dict(dataset_path=str(root),project_path=directory,name='actual_run',encoder='resnet18',input_size=32,num_classes=2,pretrained=False,decoder_channels=[16,8,4,2,1],batch_size=2,epochs=1,num_workers=0,use_ema=False,use_amp=False,export_onnx=False,plots=False,loss='CE+Dice',device='cpu')
 with patch.object(QMessageBox,'critical') as error,patch.object(QMessageBox,'information') as info:
  window._run_unet_training_from_widget(params);timer=QElapsedTimer();timer.start()
  while window.unet_process is not None and timer.elapsed()<30000:QTest.qWait(20)
  assert window.unet_process is None,window.log_text.toPlainText()
  error.assert_not_called();info.assert_called_once()
  folder=Path(window.unet_train_widget._last_run_dir)
  assert (folder/'weights/best_iou.pth').is_file(),window.log_text.toPlainText()
  assert not window._worker_artifacts
window.close();app.processEvents()
""")

    def test_main_close_waits_for_training_subprocess(self):
        self.gui("""
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QProcess,QElapsedTimer
from PyQt5.QtTest import QTest
from main import ImagePlayer
app=QApplication([]);main=ImagePlayer([('yolo_train','Training')]);main.show();window=main._tab_widgets[0]
temporary=tempfile.TemporaryDirectory();stop=Path(temporary.name)/'stop';window._worker_artifacts['unet']=(temporary,stop)
process=QProcess();window.unet_process=process;process.finished.connect(window._on_unet_finished)
child="import pathlib,sys,time\\nwhile not pathlib.Path(sys.argv[1]).exists():time.sleep(.01)\\nsys.exit(130)"
process.start(sys.executable,['-c',child,str(stop)]);assert process.waitForStarted(3000)
with patch.object(QMessageBox,'critical') as error,patch.object(QMessageBox,'information') as info:
 main.close();timer=QElapsedTimer();timer.start()
 while main.isVisible() and timer.elapsed()<10000:QTest.qWait(10)
 assert not main.isVisible() and window.unet_process is None
 info.assert_not_called();error.assert_not_called()
app.processEvents()
""")
