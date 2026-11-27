"""Blind masked pretraining contracts, native lifecycle and exact resume."""
from concurrent.futures import CancelledError
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import numpy as np
import torch

from utils_new.autoencoders.models import build_ae_model_from_cfg
from utils_new.autoencoders.masked import patchify,unpatchify
from utils_new.autoencoders.losses import ReconstructionLoss,training_objective
from utils_new.autoencoders.training import run_training
from utils_new.autoencoders.inference import infer_autoencoder
from utils_new.configuration.autoencoder import TrainConfig
from utils_new.import_export.autoencoder import load_autoencoder,read_checkpoint,inference_contract
from utils_new.import_export.yolo_yaml import load_image_splits
from test_autoencoder_training_ui import dataset

ROOT=Path(__file__).resolve().parents[1]


def config(arch):
    profile='mae_vit_compact' if arch=='mae' else 'fcmae_convnextv2'
    raw=json.loads((ROOT/f'settings/autoencoders/{profile}.json').read_text())
    raw.update(epochs=1,batch_size=2,augmentation=False,noise_std=0,calibration_pixels=1000)
    raw['model'].update(img_h=64,img_w=96,base=2,blocks_per_stage=1,encoder_dim=24,encoder_depth=1,
                        encoder_heads=4,decoder_dim=16,decoder_depth=1,decoder_heads=4)
    return TrainConfig.from_dict(raw)


class MaskedPretraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_masks_patch_order_no_hidden_pixel_leak_and_masked_only_gradient(self):
        for arch in ('mae','fcmae'):
            with self.subTest(arch=arch):
                cfg=config(arch);model=build_ae_model_from_cfg(cfg.model)
                x=torch.rand(2,3,64,96,requires_grad=True)
                self.assertTrue(torch.equal(x,unpatchify(patchify(x,model.patch),model.patch,64,96,3)))
                mask=model.validation_mask(x);pixels=model.mask_pixels(mask)
                output,extra=model.forward_masked(x,mask)
                other=model.forward_masked(x+pixels*3,mask)[0]
                self.assertTrue(torch.allclose(output,other,atol=1e-6,rtol=1e-6))
                loss,_,_=training_objective(output,torch.zeros_like(x),torch.ones(2,1,64,96),extra,ReconstructionLoss(cfg),0)
                loss.backward()
                self.assertTrue(torch.equal(x.grad*pixels,torch.zeros_like(x)))
                self.assertGreater(float((x.grad*(~pixels)).abs().sum()),0)
                predicted=torch.ones_like(x,requires_grad=True)
                criterion=ReconstructionLoss(cfg)
                value,_,_=training_objective(predicted,torch.zeros_like(x),torch.ones(2,1,64,96),extra,criterion,0)
                value.backward()
                self.assertFalse((predicted.grad*(~pixels)).any())
                self.assertTrue((predicted.grad[pixels.expand_as(x)]!=0).all())
                if arch=='mae':
                    encoded,_=model.encode_visible(x.detach(),mask)
                    self.assertEqual(encoded.shape[1],model.tokens-model.hidden+1)
                features=model.encode_features(x.detach())
                self.assertEqual(features.shape,(2,cfg.model['latent_channels'],*model.grid))

    def test_deterministic_inference_covers_all_hidden_patches_and_ignores_global_rng(self):
        for arch in ('mae','fcmae'):
            cfg=config(arch);model=build_ae_model_from_cfg(cfg.model).eval();x=torch.rand(1,3,64,96)
            masks=[];native=model.reconstruct
            def capture(inputs,mask):
                masks.append(mask.clone());return native(inputs,mask)
            model.reconstruct=capture
            rng=torch.get_rng_state()
            with torch.inference_mode():a=model(x)[0];b=model(x)[0]
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.stack(masks).any(0).all())
            valid=torch.zeros(1,1,64,96);valid[:,:,10:11,10:11]=1
            fixed=model.validation_mask(x,valid)
            self.assertTrue((valid*model.mask_pixels(fixed)).any())
            for _ in range(3):self.assertTrue((valid*model.mask_pixels(model.random_mask(x,valid))).any())

    def test_training_yaml_with_masks_no_thresholds_reload_and_resume(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root/'data')
            for arch in ('mae','fcmae'):
                cfg=config(arch);cfg.epochs=2
                full=root/arch/'full';resumed=root/arch/'resumed'
                result=run_training(cfg,source,full,plots=False)
                self.assertIsNone(result['calibration']);self.assertFalse((full/'calibration.json').exists())
                self.assertFalse(json.loads((full/'test_metrics.json').read_text())['thresholds_applied'])
                loaded=load_autoencoder(result['checkpoint']);self.assertEqual(loaded.arch,arch)
                values=infer_autoencoder(np.zeros((23,45,3),np.uint8),loaded.model,loaded.config,'cpu')
                self.assertTrue(np.isfinite(values['error']).all())
                self.assertNotEqual(inference_contract(cfg.model),inference_contract(dict(cfg.model,mask_eval_seed=1)))
                stop=Event()
                def progress(event):
                    if event['event']=='epoch':stop.set()
                with self.assertRaises(CancelledError):run_training(cfg,source,resumed,plots=False,cancel=stop,progress=progress)
                run_training(cfg,source,resumed,plots=False,resume=resumed/'weights/last.pt')
                before=read_checkpoint(full/'weights/last.pt');after=read_checkpoint(resumed/'weights/last.pt')
                for key in before['model']:self.assertTrue(torch.equal(before['model'][key],after['model'][key]),key)

    def test_masked_configuration_rejects_incompatible_loss_and_shapes(self):
        raw=config('mae').to_dict()
        for change in ({'use_gan':True},{'use_ssim':True},{'loss_kind':'l1'}):
            with self.assertRaises(ValueError):TrainConfig.from_dict(dict(raw,**change))
        for change in ({'mask_ratio':1},{'encoder_dim':25},{'model_type':'vae'},{'use_edge_channel':True,'model_in_channels':4}):
            with self.assertRaises(ValueError):TrainConfig.from_dict(dict(raw,model=dict(raw['model'],**change)))
        raw=config('fcmae').to_dict();raw['model']['patch_size']=16
        with self.assertRaises(ValueError):TrainConfig.from_dict(raw)

    def test_mask_yaml_folder_discovery_plots_normal_calibration_and_larger_warmstart(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root/'data')
            source.rename(root/'data/masks_data.yaml')
            # A real mask image must never enter the selected image splits.
            from utils_new.import_export.images import write_image
            write_image(root/'data/masks/train/fake.png',np.zeros((10,10),np.uint8))
            sources=load_image_splits(root/'data')
            self.assertEqual(len(sources['splits']['train']),2)
            for arch in ('mae','fcmae'):
                cfg=config(arch);cfg.calibrate_thresholds=True
                trained=run_training(cfg,root/'data',root/arch,plots=True)
                self.assertEqual(trained['calibration']['source_split'],'calibration')
                self.assertTrue((root/arch/'masked_examples.png').is_file())
                if arch=='mae':
                    cfg.model['img_h']=96
                    warm=run_training(cfg,root/'data',root/'larger',initial_weights=trained['checkpoint'],plots=False)
                    self.assertEqual(read_checkpoint(warm['checkpoint'])['config']['img_h'],96)
            for path in (root/'data/images/test').glob('*.png'):path.unlink()
            self.assertEqual(load_image_splits(root/'data')['splits']['test'],[])


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS')=='1','opt-in Qt')
class MaskedTrainingUI(unittest.TestCase):
    def test_native_worker_from_ui_and_application_of_both_masked_models(self):
        code=r'''
import sys,time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.deep_learning_training import YoloTrainWindow
from tabs.yolo_test_aeb import YoloInspectWindow
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_masked_autoencoders import config
from test_autoencoder_training_ui import dataset
app=QApplication([]);errors=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
QMessageBox.critical=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
with TemporaryDirectory() as directory:
 root=Path(directory);source=dataset(root/'data')
 owner=YoloTrainWindow();widget=owner.autoencoder_train_widget
 for arch in ('mae','fcmae'):
  cfg=config(arch);widget.apply_config(cfg.to_dict())
  assert widget.config()==cfg.to_dict()
  assert widget.pages.widget(2).isEnabled()
  assert not widget.train_controls['use_gan'].isEnabled()
  assert not widget.train_controls['calibrate_thresholds'].isChecked()
  widget.source.setText(str(source));widget.project_path.setText(str(root/'runs'))
  widget.experiment_name.setText(arch);widget.device_combo.setCurrentText('cpu');widget.plots.setChecked(False)
  widget.run_btn.click();assert owner.autoencoder_process is not None
  end=time.monotonic()+70
  while owner.autoencoder_process is not None and time.monotonic()<end:
   app.processEvents();time.sleep(.01)
  assert owner.autoencoder_process is None,owner.log_text.toPlainText()
  assert not errors,(errors,owner.log_text.toPlainText())
  best=root/'runs'/arch/'weights/best.pt'
  viewer=YoloInspectWindow();assert viewer._load_ae_model_from_path(str(best)),errors
  assert viewer.ae_arch==arch and viewer._ae_calibration is None
  viewer.close();app.processEvents()
 owner.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
