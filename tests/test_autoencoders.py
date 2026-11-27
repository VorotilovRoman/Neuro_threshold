"""AE migration contracts, real CPU optimization, resume and anomaly calibration."""

import ast
from concurrent.futures import CancelledError
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils_new.autoencoders.dataset import (
    AutoencoderDataset,
    collect_training_images,
    split_images,
)
from utils_new.autoencoders.inference import infer_autoencoder
from utils_new.autoencoders.losses import ReconstructionLoss, kl_loss
from utils_new.autoencoders.models import PatchDiscriminator, build_ae_model_from_cfg
from utils_new.autoencoders.preprocessing import (
    compute_edge_map_torch,
    preprocess_image_for_ae,
)
from utils_new.autoencoders.training import calibrate, run_training, train_epoch
from utils_new.configuration.autoencoder import TrainConfig
from utils_new.import_export.autoencoder import (
    load_autoencoder,
    read_checkpoint,
    save_checkpoint,
)
from utils_new.import_export.images import write_image

ROOT = Path(__file__).resolve().parents[1]


def small_config(**overrides):
    raw = json.loads((ROOT / "settings/autoencoders/conv_bn.json").read_text())
    raw.update(
        epochs=2,
        batch_size=2,
        augmentation=True,
        noise_std=0.02,
        use_ssim=False,
        grad_loss_w=0,
        use_gan=False,
        calibration_pixels=2000,
    )
    raw["model"].update(model_type="ae", img_h=32, img_w=48, latent_dim=4)
    raw.update(overrides)
    return TrainConfig.from_dict(raw)


def write_dataset(root, count=5):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        write_image(
            root / (str(index) + ".png"),
            np.random.default_rng(index).integers(0, 256, (20, 50, 3), dtype=np.uint8),
        )
    return sorted(map(str, root.glob("*.png")))


class AutoencoderContracts(unittest.TestCase):
    def test_reconstruction_display_preserves_color_geometry_and_raw_error(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x[:, :channels], None

        for channels in (1, 3):
            for version in ('legacy-opencv-v1', 'ae-pil-v1'):
                for aspect in (False, True):
                    with self.subTest(channels=channels, version=version, aspect=aspect):
                        image = np.full((12, 70, 3), 73, np.uint8)
                        if channels == 3:
                            image[:] = (21, 87, 149)  # BGR, not RGB
                        cfg = dict(in_channels=channels, out_channels=channels,
                                   img_h=32, img_w=48, keep_aspect=aspect,
                                   preprocessing_version=version, pad_value=255,
                                   use_edge_channel=True, model_in_channels=channels+1)
                        result = infer_autoencoder(image, Identity(), cfg, 'cpu')
                        self.assertEqual(result['reconstruction_bgr'].dtype, np.uint8)
                        np.testing.assert_array_equal(result['reconstruction_bgr'], image)
                        self.assertEqual(result['image_score'], 0)

        class Unbounded(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, 2)

        result = infer_autoencoder(np.zeros((12, 70, 3), np.uint8), Unbounded(),
                                   dict(in_channels=3, img_h=32, img_w=48), 'cpu')
        self.assertTrue(np.all(result['reconstruction_bgr'] == 255))
        self.assertEqual(result['image_score'], 2)  # display clipping cannot alter MAE

    def test_all_model_state_schemas_match_original_training_scripts(self):
        fixtures = json.loads(
            (ROOT / "tests/fixtures/autoencoder_architectures.json").read_text()
        )
        for entry in fixtures:
            cfg = entry["config"]
            with self.subTest(
                arch=cfg["arch"], kind=cfg["model_type"], channels=cfg["in_channels"]
            ):
                from utils_new.configuration.autoencoder import normalize_model_config
                from utils_new.autoencoders.models import detect_arch

                self.assertEqual(detect_arch(normalize_model_config(cfg)), cfg["arch"])
                model = build_ae_model_from_cfg(cfg)
                descriptors = [
                    (k, list(v.shape)) for k, v in model.state_dict().items()
                ]
                self.assertEqual(
                    hashlib.sha256(json.dumps(descriptors).encode()).hexdigest(),
                    entry["state_signature"],
                )

    def test_profiles_validate_without_torch_and_bad_configs_are_rejected(self):
        for path in (ROOT / "settings/autoencoders").glob("*.json"):
            TrainConfig.from_dict(json.loads(path.read_text()))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys;from utils_new.configuration.autoencoder import TrainConfig;assert 'torch' not in sys.modules",
            ],
            cwd=ROOT,
            check=True,
        )
        raw = small_config().to_dict()
        for update in (
            {"batch_size": 0},
            {"lr": float("nan")},
            {"val_split": 1},
            {"unknown": 1},
        ):
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    TrainConfig.from_dict(dict(raw, **update))
        for update in (
            {"img_w": 31},
            {"latent_dim": 1.5},
            {"in_channels": True},
            {"model_in_channels": 2},
            {"preprocessing_version": "unknown"},
            {"median_kernel": 2},
        ):
            invalid = deepcopy(raw)
            invalid["model"].update(update)
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    TrainConfig.from_dict(invalid)

    def test_shared_preprocessing_clean_targets_derived_edges_and_paired_geometry(self):
        with TemporaryDirectory() as directory:
            paths = write_dataset(Path(directory), 2)
            cfg = small_config()
            cfg.model.update(
                arch="unet_v2",
                latent_channels=4,
                use_edge_channel=True,
                model_in_channels=2,
            )
            dataset = AutoencoderDataset(paths, cfg)
            item = dataset[0]
            import cv2

            expected, region = preprocess_image_for_ae(cv2.imread(paths[0]), cfg.model)
            torch.testing.assert_close(item["input"], expected, rtol=0, atol=0)
            torch.testing.assert_close(item["target"], expected[:1], rtol=0, atol=0)
            self.assertEqual(
                int(item["valid"].sum()),
                (region[2] - region[0]) * (region[3] - region[1]),
            )
            noisy = AutoencoderDataset(paths, cfg, training=True)
            seen = []

            def geometry(*args, **kwargs):
                seen.append(kwargs["img_size"])
                return 0.0, (0, 0), 1.0, (0.0, 0.0)

            with patch(
                "utils_new.autoencoders.dataset.transforms.RandomAffine.get_params",
                side_effect=geometry,
            ):
                value = noisy[0]
            self.assertEqual(seen, [[48, 32]])
            self.assertFalse(torch.equal(value["input"][:1], value["target"]))
            torch.testing.assert_close(
                value["input"][1:], compute_edge_map_torch(value["input"][:1])
            )
            cfg.noise_std = 0
            cfg.augmentation = False
            clean = AutoencoderDataset(paths, cfg, training=True)[0]
            torch.testing.assert_close(
                clean["input"][:1], clean["target"], rtol=0, atol=0
            )

    def test_cache_separates_same_names_and_preserves_float32_on_reloads(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for folder, index in [("a", 0), ("b", 1)]:
                (root / folder).mkdir()
                p = root / folder / "same.bmp"
                write_image(
                    p,
                    np.random.default_rng(index).integers(
                        0, 256, (32, 48, 3), dtype=np.uint8
                    ),
                )
                os.utime(p, ns=(1000000000, 1000000000))
                paths.append(str(p))
            cfg = small_config()
            cfg.model.update(
                arch="unet_v2",
                latent_channels=4,
                use_edge_channel=True,
                model_in_channels=2,
            )
            dataset = AutoencoderDataset(paths, cfg, edge_cache=root / "cache")
            first = dataset[0]["input"]
            second = dataset[1]["input"]
            self.assertEqual(len(list((root / "cache").glob("*.npy"))), 2)
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(first, dataset[0]["input"], rtol=0, atol=0)
            for path in (root / "cache").glob("*.npy"):
                path.write_bytes(b"broken")
            torch.testing.assert_close(first, dataset[0]["input"], rtol=0, atol=0)

    def test_split_groups_identical_content_and_excludes_run_artifacts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_dataset(root / "data", 3)
            duplicate = root / "data/duplicate.png"
            duplicate.write_bytes(Path(paths[0]).read_bytes())
            paths.append(str(duplicate))
            train, val = split_images(paths, 0.5, 42)
            self.assertFalse(set(train) & set(val))
            self.assertEqual(paths[0] in train, str(duplicate) in train)
            self.assertEqual(set(train + val), set(paths))
            output = root / "data/run"
            output.mkdir()
            write_image(output / "example.png", np.zeros((20, 20), np.uint8))
            self.assertEqual(
                set(collect_training_images(root / "data", exclude=[output])),
                set(paths),
            )
            with self.assertRaises(ValueError):
                split_images([paths[0], str(duplicate)], 0.2, 42)

    def test_pixel_and_image_calibration_exclude_padding_and_match_inference(self):
        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x), None

        with TemporaryDirectory() as directory:
            image = np.full((8, 40, 3), 64, np.uint8)
            path = Path(directory) / "image.png"
            write_image(path, image)
            cfg = small_config()
            cfg.model["pad_value"] = 255
            ds = AutoencoderDataset([str(path)], cfg)
            model = Zero()
            value = calibrate(model, torch.utils.data.DataLoader(ds), cfg, "cpu")
            result = infer_autoencoder(image, model, cfg.model, "cpu")
            self.assertAlmostEqual(value["image_threshold"], 64 / 255, places=6)
            self.assertAlmostEqual(value["pixel_threshold"], 64 / 255, places=6)
            self.assertAlmostEqual(
                result["image_score"], value["image_threshold"], places=6
            )
            self.assertLess(value["valid_pixels"], 32 * 48)
            cfg.calibration_pixels = 3
            sampled = calibrate(model, torch.utils.data.DataLoader(ds), cfg, "cpu")
            self.assertEqual(sampled["sampled_pixels"], 3)
            self.assertEqual(sampled["pixel_sampling"], "uniform_priority")

    def test_kl_reductions_and_reconstruction_gradients(self):
        mu = torch.ones((2, 3, 4, 5))
        variance = torch.zeros_like(mu)
        self.assertEqual(float(kl_loss(mu, variance, reduction="mean")), 0.5)
        self.assertEqual(float(kl_loss(mu, variance, reduction="sum_per_sample")), 30.0)
        cfg = small_config(use_ssim=True, grad_loss_w=0.5)
        criterion = ReconstructionLoss(cfg)
        image = torch.rand((1, 1, 32, 48), requires_grad=True)
        target = torch.rand_like(image)
        loss = criterion(image, target, torch.ones_like(image))
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)

    def test_real_gan_step_updates_generator_and_discriminator(self):
        cfg = small_config(use_gan=True, gan_start_epoch=0, discriminator_base=8)
        model = build_ae_model_from_cfg(cfg.model)
        discriminator = PatchDiscriminator(1, 8)
        criterion = ReconstructionLoss(cfg)
        before = model.fc_encode.weight.detach().clone()
        before_d = next(discriminator.parameters()).detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=0.0001)
        batch = dict(
            input=torch.rand(2, 1, 32, 48),
            target=torch.rand(2, 1, 32, 48),
            valid=torch.ones(2, 1, 32, 48),
        )
        stats = train_epoch(
            model,
            [batch],
            optimizer,
            criterion,
            "cpu",
            0,
            discriminator=discriminator,
            opt_d=opt_d,
            use_gan=True,
        )
        self.assertFalse(torch.equal(before, model.fc_encode.weight))
        self.assertFalse(torch.equal(before_d, next(discriminator.parameters())))
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))

    def test_checkpoint_atomicity_versions_and_legacy_threshold_is_not_pixel_threshold(
        self,
    ):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            cfg = small_config()
            model = build_ae_model_from_cfg(cfg.model)
            save_checkpoint(path, model, cfg.model)
            original = path.read_bytes()
            meta_path = path.with_name('best_meta.json')
            original_meta = meta_path.read_bytes()
            with patch(
                "utils_new.import_export.atomic.os.replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    save_checkpoint(path, model, cfg.model)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(set(Path(directory).iterdir()), {path, meta_path})
            self.assertEqual(meta_path.read_bytes(), original_meta)
            legacy = Path(directory) / "legacy.pt"
            config = dict(cfg.model)
            config.pop("preprocessing_version")
            torch.save(
                dict(model=model.state_dict(), config=config, threshold_value=0.001),
                legacy,
            )
            loaded = load_autoencoder(legacy)
            self.assertTrue(loaded.legacy)
            self.assertIsNone(loaded.calibration)
            self.assertEqual(loaded.config["preprocessing_version"], "legacy-opencv-v1")
            payload = torch.load(path, weights_only=True)
            payload["schema_version"] = 999
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_cpu_rng_snapshot_does_not_initialize_cuda_and_contract_covers_dilations(
        self,
    ):
        from utils_new.autoencoders.training import capture_rng
        from utils_new.import_export.autoencoder import (
            inference_contract,
            validate_calibration,
        )
        from utils_new.configuration.autoencoder import normalize_model_config

        generator = torch.Generator()
        with patch(
            "torch.cuda.get_rng_state_all",
            side_effect=AssertionError("CPU touched CUDA"),
        ):
            self.assertEqual(
                capture_rng(generator, generator, device="cpu")["cuda"], []
            )
        config = normalize_model_config(
            dict(
                arch="unet_v2",
                model_type="ae",
                in_channels=1,
                img_h=32,
                img_w=48,
                latent_channels=4,
            )
        )
        calibration = dict(
            version=1,
            metric="mae",
            contract=inference_contract(config),
            image_threshold=0.2,
            pixel_threshold=0.5,
            image_percentile=99.0,
            pixel_percentile=99.0,
            images=2,
            image_aggregation="mean_valid_pixels",
            pixel_aggregation="valid_pixels",
        )
        # Contracts survive the JSON artifact round-trip.
        validate_calibration(json.loads(json.dumps(calibration)), config)
        changed = dict(config, bottleneck_dilations=(1, 2))
        with self.assertRaisesRegex(ValueError, "Калибровка"):
            validate_calibration(calibration, changed)

    def test_real_training_cancel_resume_is_exact_and_changed_dataset_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_dataset(data)
            cfg = small_config()
            complete = run_training(cfg, data, root / "complete", plots=False)
            stop = Event()

            def progress(event):
                if event["event"] == "epoch":
                    stop.set()

            with self.assertRaises(CancelledError):
                run_training(
                    cfg,
                    data,
                    root / "resumed",
                    cancel=stop,
                    progress=progress,
                    plots=False,
                )
            last = root / "resumed/weights/last.pt"
            self.assertEqual(read_checkpoint(last)["epoch"], 1)
            resumed = run_training(
                cfg, data, root / "resumed", resume=last, plots=False
            )
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["calibration"], resumed["calibration"])
            first = read_checkpoint(complete["checkpoint"])
            second = read_checkpoint(resumed["checkpoint"])
            for key in first["model"]:
                torch.testing.assert_close(
                    first["model"][key], second["model"][key], rtol=0, atol=0
                )
            write_image(data / "0.png", np.zeros((20, 50, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "Датасет изменился"):
                run_training(cfg, data, root / "resumed", resume=last, plots=False)

    def test_workers_are_lazy_and_old_modules_have_no_consumers(self):
        self.assertFalse((ROOT / "utils_aeb").exists())
        self.assertFalse((ROOT / "utils_new/segmentation/autoencoder.py").exists())
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys;import utils_new.training.autoencoder;import utils_new.autoencoders;assert 'torch' not in sys.modules",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "utils_new/training/autoencoder.py"),
                "--help",
            ],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--resume", result.stdout)
        for path in ROOT.rglob("*.py"):
            if ".git" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(
                        node.module, "utils_new.segmentation.autoencoder", str(path)
                    )
                    self.assertFalse(
                        (node.module or "").startswith("utils_aeb."), str(path)
                    )


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in Qt tests")
class AutoencoderGui(unittest.TestCase):
    def test_loading_new_and_old_checkpoints_sets_correct_pixel_threshold_and_image_score(
        self,
    ):
        code = """
import tempfile,sys
from pathlib import Path
import cv2,numpy as np
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.yolo_test_aeb import YoloInspectWindow
app=QApplication([]);window=YoloInspectWindow();window.show()
assert 'torch' not in sys.modules
window.clear_ae_model();assert 'torch' not in sys.modules
import torch
from utils_new.autoencoders.models import build_ae_model_from_cfg
from utils_new.configuration.autoencoder import normalize_model_config
from utils_new.import_export.autoencoder import inference_contract,save_checkpoint
cfg=normalize_model_config(dict(arch='conv',model_type='ae',in_channels=1,img_h=32,img_w=48,latent_dim=4,preprocessing_version='ae-pil-v1',vae_use_mean=True))
model=build_ae_model_from_cfg(cfg)
calibration=dict(version=1,metric='mae',contract=inference_contract(cfg),image_aggregation='mean_valid_pixels',pixel_aggregation='valid_pixels',image_threshold=.2,pixel_threshold=.6,image_percentile=99.,pixel_percentile=99.5,images=2)
with tempfile.TemporaryDirectory() as directory,patch.object(QMessageBox,'critical',side_effect=AssertionError('model load failed')):
 root=Path(directory);weights=root/'new.pt';save_checkpoint(weights,model,cfg,calibration=calibration)
 assert window._load_ae_model_from_path(str(weights))
 assert window.chk_ae_use_ckpt_thr.isEnabled() and window._ae_ckpt_threshold==.6
 image=root/'image.png';cv2.imwrite(str(image),np.zeros((20,60,3),np.uint8));window._load_images([str(image)]);window.update_timer.stop()
 window.btn_apply_ae.click();timer=QElapsedTimer();timer.start()
 while window._ae_task is not None and timer.elapsed()<10000:QTest.qWait(10)
 assert window._ae_task is None,window.processing_status.text()
 assert 'MAE' in window.ae_score_label.text() and '0.200000' in window.ae_score_label.text()
 assert window._ae_threshold==.6
 original_to=torch.nn.Module.to
 def limited_to(network,device,*args,**kwargs):
  if str(device).startswith('cuda'):raise RuntimeError('simulated CUDA allocation failure')
  return original_to(network,device,*args,**kwargs)
 with patch.object(window,'_resolve_torch_device',return_value=torch.device('cuda:0')),patch.object(torch.nn.Module,'to',new=limited_to):
  assert window._load_ae_model_from_path(str(weights))
  window.btn_apply_ae.click();timer.restart()
  while window._ae_task is not None and timer.elapsed()<10000:QTest.qWait(10)
  assert window._ae_task is None and window.ae_device.type=='cpu',window.processing_status.text()
  assert window._ae_err_hw is not None,window.processing_status.text()
 legacy=root/'old.pt';oldcfg=dict(cfg);oldcfg.pop('preprocessing_version')
 torch.save(dict(model=model.state_dict(),config=oldcfg,threshold_value=.001),legacy)
 assert window._load_ae_model_from_path(str(legacy))
 assert not window.chk_ae_use_ckpt_thr.isEnabled() and window._ae_ckpt_threshold is None
 window.clear_ae_model();assert window.ae_score_label.text()=='Ошибка снимка: —'
window.close();app.processEvents()
"""
        with TemporaryDirectory() as directory:
            environment = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CACHE_HOME=directory,
                XDG_CONFIG_HOME=directory,
                MPLCONFIGDIR=directory,
                YOLO_CONFIG_DIR=directory,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
