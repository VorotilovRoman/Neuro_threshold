"""Modern spatial backbones: latent topology, true VAE loss and native lifecycle."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import torch

from utils_new.autoencoders.models import build_ae_model_from_cfg, GlobalResponseNorm, detect_arch
from utils_new.autoencoders.losses import ReconstructionLoss, training_objective
from utils_new.autoencoders.training import run_training, train_epoch
from utils_new.configuration.autoencoder import TrainConfig
from utils_new.import_export.autoencoder import load_autoencoder, inference_contract
from utils_new.import_export.autoencoder import read_checkpoint
from utils_new.autoencoders.inference import infer_autoencoder
from test_autoencoder_training_ui import dataset

ROOT = Path(__file__).resolve().parents[1]


def config(arch='residual', kind='ae'):
    raw = json.loads((ROOT / f'settings/autoencoders/{arch}_{kind}.json').read_text())
    raw.update(epochs=1, batch_size=2, augmentation=False, noise_std=0, use_ssim=False, grad_loss_w=0,
               calibration_pixels=1000, num_workers=0)
    raw['model'].update(img_h=32, img_w=48, base=4, blocks_per_stage=1, latent_channels=3)
    return TrainConfig.from_dict(raw)


class ModernArchitectures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_spatial_bottleneck_shapes_gradients_and_parameter_count(self):
        for arch in ('residual', 'convnext_v2'):
            for kind in ('ae', 'vae'):
                with self.subTest(arch=arch, kind=kind):
                    cfg = config(arch, kind)
                    raw = dict(cfg.model)
                    raw.pop('preprocessing_version')
                    self.assertEqual(detect_arch(raw), arch)
                    model = build_ae_model_from_cfg(cfg.model)
                    inputs = torch.rand(2, 3, 32, 48)
                    output, extra = model(inputs)
                    self.assertEqual(output.shape, inputs.shape)
                    loss, _, kl = training_objective(output, inputs, torch.ones(2,1,32,48), extra,
                                                     ReconstructionLoss(cfg), cfg.beta_vae_target)
                    loss.backward()
                    head = model.fc_mu if kind == 'vae' else model.compress
                    self.assertGreater(float(head.weight.grad.abs().sum()), 0)
                    if kind == 'vae':self.assertGreater(float(kl.detach()), 0)
                    model.eval()
                    with torch.inference_mode():
                        z, _ = model.encode(inputs)
                        self.assertEqual(z.shape, (2,3,4,6))
                        self.assertTrue(torch.equal(model(inputs)[0], model(inputs)[0]))
                        self.assertFalse(torch.allclose(model.decode(z), model.decode(z+1)))
                    larger = dict(cfg.model, img_h=64, img_w=96)
                    self.assertEqual(sum(p.numel() for p in model.parameters()),
                                     sum(p.numel() for p in build_ae_model_from_cfg(larger).parameters()))
        grn = GlobalResponseNorm(4)
        value = torch.randn(1, 2, 3, 4)
        self.assertTrue(torch.equal(grn(value), value))

    def test_cli_default_is_spatial_ae_without_adversarial_training(self):
        from utils_new.training.autoencoder import main
        with TemporaryDirectory() as directory, \
                patch('utils_new.autoencoders.training.run_training', return_value={'calibration': {}}) as trainer:
            self.assertEqual(main(['--data',directory,'--output',directory,'--device','cpu']),0)
            cfg = trainer.call_args.args[0]
            self.assertEqual(cfg.model['arch'],'residual')
            self.assertFalse(cfg.use_gan)

    def test_native_training_checkpoint_calibration_and_inference_all_variants(self):
        with TemporaryDirectory() as directory:
            root = Path(directory);source = dataset(root/'data')
            for arch in ('residual', 'convnext_v2'):
                for kind in ('ae', 'vae'):
                    with self.subTest(arch=arch, kind=kind):
                        cfg = config(arch, kind)
                        result = run_training(cfg, source, root/f'{arch}_{kind}', plots=False)
                        loaded = load_autoencoder(result['checkpoint'])
                        self.assertEqual(loaded.arch, arch)
                        self.assertEqual(loaded.calibration['source_split'], 'calibration')
                        value = infer_autoencoder(np.zeros((21,39,3), np.uint8),loaded.model,loaded.config,'cpu')
                        self.assertTrue(np.isfinite(value['error']).all())
                        changed = dict(loaded.config, blocks_per_stage=2)
                        self.assertNotEqual(inference_contract(changed), inference_contract(loaded.config))

    def test_unet_skip_mode_is_in_contract_and_singleton_batchnorm_trains(self):
        raw = config().to_dict()
        raw['model'].update(arch='unet', img_h=8, img_w=8, base=32, skip_mode='none')
        cfg = TrainConfig.from_dict(raw)
        model = build_ae_model_from_cfg(cfg.model)
        seen = []
        hook = model.decoder.register_forward_pre_hook(
            lambda module, inputs: seen.append(all(not torch.count_nonzero(skip) for skip in inputs[1:])))
        output, extra = model(torch.rand(2,3,8,8))
        loss = output.sum();loss.backward();hook.remove()
        self.assertEqual(seen, [True])
        self.assertGreater(float(model.compress.weight.grad.abs().sum()), 0)
        self.assertNotEqual(inference_contract(cfg.model), inference_contract(dict(cfg.model,skip_mode='full')))
        loader = [dict(input=torch.rand(1,3,8,8),target=torch.rand(1,3,8,8),valid=torch.ones(1,1,8,8))]
        stats = train_epoch(model,loader,torch.optim.AdamW(model.parameters()),ReconstructionLoss(cfg),'cpu',0)
        self.assertTrue(np.isfinite(stats['loss']))

    def test_modern_vae_resume_restores_stochastic_training_exactly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory);source = dataset(root/'data')
            cfg = config('convnext_v2','vae');cfg.epochs = 2
            run_training(cfg,source,root/'full',plots=False)
            stop = Event()
            def progress(event):
                if event['event'] == 'epoch':stop.set()
            with self.assertRaises(CancelledError):
                run_training(cfg,source,root/'resumed',plots=False,cancel=stop,progress=progress)
            run_training(cfg,source,root/'resumed',resume=root/'resumed/weights/last.pt',plots=False)
            expected = read_checkpoint(root/'full/weights/last.pt')
            actual = read_checkpoint(root/'resumed/weights/last.pt')
            for key in expected['model']:
                self.assertTrue(torch.equal(expected['model'][key],actual['model'][key]),key)


@unittest.skipUnless(__import__('os').environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
class ModernApplicationUI(unittest.TestCase):
    def test_new_models_load_in_existing_inference_tab(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-c', r'''
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QMessageBox
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_autoencoder_architectures import config
from utils_new.autoencoders.models import build_ae_model_from_cfg
from utils_new.import_export.autoencoder import save_checkpoint
from tabs.yolo_test_aeb import YoloInspectWindow
app=QApplication([]);errors=[]
QMessageBox.critical=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
window=YoloInspectWindow()
with TemporaryDirectory() as directory:
 for arch in ('residual','convnext_v2'):
  for kind in ('ae','vae'):
   cfg=config(arch,kind);path=Path(directory)/(arch+'_'+kind+'.pt')
   save_checkpoint(path,build_ae_model_from_cfg(cfg.model),cfg.model)
   assert window._load_ae_model_from_path(str(path)),errors
   assert window.ae_arch==arch
 assert not errors,errors
window.close();app.processEvents()
'''], cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__':unittest.main()
