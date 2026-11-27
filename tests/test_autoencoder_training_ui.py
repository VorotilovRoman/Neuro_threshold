"""Unlabelled YAML sources, AE training artifacts and real Qt worker integration."""
from concurrent.futures import CancelledError
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from utils_new.autoencoders.dataset import training_sources
from utils_new.autoencoders.training import run_training
from utils_new.configuration.autoencoder import TrainConfig
from utils_new.import_export.autoencoder import load_autoencoder, read_checkpoint
from utils_new.import_export.images import write_image
from utils_new.import_export.yolo_yaml import load_image_splits

ROOT=Path(__file__).resolve().parents[1]


def config(epochs=1):
    raw=json.loads((ROOT/'settings/autoencoders/conv_bn_baseline.json').read_text())
    raw.update(epochs=epochs,batch_size=2,augmentation=False,noise_std=0,use_ssim=False,
               use_gan=False,patience=epochs+2,calibration_pixels=2000)
    raw['model'].update(model_type='ae',img_h=32,img_w=48,latent_dim=4)
    return TrainConfig.from_dict(raw)


def dataset(root):
    for index,split in enumerate(('train','val','calibration','test')):
        folder=root/'images'/split;folder.mkdir(parents=True)
        for number in range(2):
            image=np.random.default_rng(index*10+number).integers(0,256,(24,40,3),dtype=np.uint8)
            write_image(folder/f'{number}.png',image)
    path=root/'data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(root),train='images/train',val='images/val',
        calibration='images/calibration',test='images/test',train_masks='does-not-exist',masks='ignored')))
    return path


class UnlabelledImageSources(unittest.TestCase):
    def test_yolo_and_masks_yaml_read_images_without_annotations(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root)
            with patch('utils_new.import_export.images.read_index_mask',side_effect=AssertionError('mask read')), \
                 patch('utils_new.import_export.yolo_labels.read_yolo_labels',side_effect=AssertionError('TXT read')):
                value=training_sources(source,.2,42)
            self.assertEqual({key:len(paths) for key,paths in value['splits'].items()},
                             dict(train=2,val=2,calibration=2,test=2))
            self.assertFalse(value['auto_split'])
            self.assertEqual(len(value['manifest']['test']),2)

    def test_txt_membership_alias_and_auto_val_do_not_widen_to_entire_folder(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root)
            (root/'selected.txt').write_text('images/train/0.png\nimages/train/1.png\n')
            source.write_text(yaml.safe_dump(dict(path=str(root),train='selected.txt',valid='images/val')))
            value=load_image_splits(source)
            self.assertEqual(len(value['splits']['train']),2)
            self.assertEqual(len(value['splits']['val']),2)
            source.write_text(yaml.safe_dump(dict(path=str(root),train='selected.txt')))
            value=training_sources(source,.5,42)
            self.assertTrue(value['auto_split'])
            self.assertEqual([len(value['splits'][name]) for name in ('train','val')],[1,1])

    def test_duplicate_contents_missing_sources_and_cancel_are_errors(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root)
            shutil.copyfile(root/'images/train/0.png',root/'images/test/duplicate.png')
            with self.assertRaisesRegex(ValueError,'Одинаковое содержимое'):
                training_sources(source,.2,42)
            source.write_text('train: missing\n')
            with self.assertRaisesRegex(ValueError,'источник'):
                load_image_splits(source)
            stop=Event();stop.set()
            with self.assertRaises(CancelledError):load_image_splits(root,cancel=stop)
        with self.assertRaises(ValueError):load_image_splits('')


class AETrainingArtifacts(unittest.TestCase):
    def test_yaml_training_calibration_test_warmstart_and_metadata_reload(self):
        import torch
        from utils_new.autoencoders.inference import infer_autoencoder

        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            root=Path(directory);source=dataset(root/'data');run=root/'run'
            result=run_training(config(),source,run,plots=False)
            best=run/'weights/best.pt';last=run/'weights/last.pt'
            self.assertEqual(result['checkpoint'],str(best))
            for path in (best,last,best.with_name('best_meta.json'),last.with_name('last_meta.json'),
                         run/'args.json',run/'split.json',run/'metrics.json',run/'test_metrics.json',run/'calibration.json'):
                self.assertTrue(path.is_file(),path)
            calibration=result['calibration']
            self.assertEqual(calibration['images'],2)
            self.assertEqual(calibration['source_split'],'calibration')
            self.assertTrue(calibration['independent_of_selection'])
            test=json.loads((run/'test_metrics.json').read_text())
            self.assertEqual(test['images'],2)
            self.assertFalse(test['used_for_selection'])
            loaded=load_autoencoder(best)
            image=np.zeros((24,40,3),np.uint8)
            before=infer_autoencoder(image,loaded.model,loaded.config,'cpu')
            best.with_name('best_meta.json').unlink()
            loaded=load_autoencoder(best)
            after=infer_autoencoder(image,loaded.model,loaded.config,'cpu')
            np.testing.assert_array_equal(before['error'],after['error'])
            self.assertEqual(loaded.metadata['epoch'],1)
            cfg=config();cfg.lr/=2
            warm=run_training(cfg,source,root/'warm',initial_weights=best,plots=False)
            self.assertEqual(read_checkpoint(warm['checkpoint'])['epoch'],1)
            source.write_text(yaml.safe_dump(dict(path=str(root/'data'),train='images/val',val='images/train')))
            with self.assertRaisesRegex(ValueError,'Датасет изменился'):
                run_training(config(),source,run,resume=last,plots=False)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS')=='1','opt-in Qt')
class AETrainingGui(unittest.TestCase):
    def gui(self, code, timeout=120):
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=timeout,
                              env=dict(os.environ,QT_QPA_PLATFORM='offscreen'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_all_profiles_variants_and_checkpoint_controls(self):
        self.gui(r'''
import sys,json,time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication
from ui.autoencoder_training import AutoencoderTrainWidget
from utils_new.configuration.autoencoder import AE_PROFILES,TrainConfig
from utils_new.autoencoders.models import build_ae_model_from_cfg
from utils_new.import_export.autoencoder import save_checkpoint
app=QApplication([]);widget=AutoencoderTrainWidget()
for name in AE_PROFILES:
 raw=json.loads((Path('settings/autoencoders')/(name+'.json')).read_text())
 widget.apply_config(raw)
 assert widget.config()==TrainConfig.from_dict(raw).to_dict(),name
for arch in ('conv','conv_gn','unet','unet_v2','residual','convnext_v2'):
 widget.model_controls['arch'].setCurrentIndex(widget.model_controls['arch'].findData(arch))
 for kind in ('ae','vae'):
  widget.model_controls['model_type'].setCurrentIndex(widget.model_controls['model_type'].findData(kind))
  assert widget.config()['model']['model_type']==kind
  assert widget.model_controls['latent_dim'].isEnabled()==(arch in ('conv','conv_gn'))
  assert widget.train_controls['beta_vae_target'].isEnabled()==(kind=='vae')
state=widget.config()
try:
 widget.apply_config(dict(state,epochs=100001))
 raise AssertionError('oversized setting was silently clamped')
except ValueError:pass
assert widget.config()==state
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_autoencoder_training_ui import config
with TemporaryDirectory() as directory:
 path=Path(directory)/'weights'/'last.pt';path.parent.mkdir()
 cfg=config(2)
 model=build_ae_model_from_cfg(cfg.model)
 save_checkpoint(path,model,cfg.model,epoch=1,training_config=cfg.to_dict(),
  training_state=dict(epoch_complete=True,config=cfg.to_dict(),device='cpu'))
 widget._inspect_checkpoint(str(path),True)
 end=time.monotonic()+30
 while widget._checkpoint_job is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert widget._checkpoint_job is None
 assert widget.config()==cfg.to_dict()
 assert not widget.pages.widget(1).isEnabled()
 widget.resume.clear();assert widget.pages.widget(1).isEnabled()
 widget.close_jobs()
from unittest.mock import patch
with patch('utils_new.import_export.yolo_yaml.load_image_splits',side_effect=ValueError('stale source')):
 widget.source.setText('missing.yaml');widget._check_source()
 job=widget._source_job;assert job.wait(3000)
 widget.close_jobs();before=widget.source_info.text()
 app.processEvents();assert widget.source_info.text()==before
''')

    def test_real_gui_worker_trains_and_saved_weights_load_for_inference(self):
        self.gui(r'''
import sys,time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.deep_learning_training import YoloTrainWindow
from tabs.yolo_test_aeb import YoloInspectWindow
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_autoencoder_training_ui import config,dataset
app=QApplication([]);errors=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.critical=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
with TemporaryDirectory() as directory:
 root=Path(directory);source=dataset(root/'данные')
 owner=YoloTrainWindow();widget=owner.autoencoder_train_widget
 assert owner.tab_widget.tabText(5)=='Экспорт модели'
 assert owner.tab_widget.tabText(6)=='Обучение VAE/AE'
 widget.apply_config(config().to_dict());widget.source.setText(str(source))
 widget.project_path.setText(str(root/'runs'));widget.experiment_name.setText('first')
 widget.device_combo.setCurrentText('cpu');widget.plots.setChecked(False)
 widget._on_run()
 assert owner.autoencoder_process is not None
 end=time.monotonic()+75
 while owner.autoencoder_process is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert owner.autoencoder_process is None,owner.log_text.toPlainText()
 assert not errors,(errors,owner.log_text.toPlainText())
 best=root/'runs/first/weights/best.pt'
 assert best.exists(),owner.log_text.toPlainText()
 assert widget._last_run_dir==str(root/'runs/first')
 assert widget.run_btn.isEnabled() and not widget.stop_btn.isEnabled()
 viewer=YoloInspectWindow();assert viewer._load_ae_model_from_path(str(best))
 assert viewer._ae_calibration['source_split']=='calibration'
 assert viewer.chk_ae_use_ckpt_thr.isChecked()
 viewer.close();owner.close();app.processEvents()
''')

    def test_native_stop_preserves_complete_checkpoint_and_cleans_worker(self):
        self.gui(r'''
import sys,time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.deep_learning_training import YoloTrainWindow
from utils_new.import_export.autoencoder import read_checkpoint
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_autoencoder_training_ui import config,dataset
app=QApplication([])
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.critical=lambda *a,**k:QMessageBox.Ok
with TemporaryDirectory() as directory:
 root=Path(directory);source=dataset(root/'data')
 owner=YoloTrainWindow();widget=owner.autoencoder_train_widget
 widget.apply_config(config(1000).to_dict());widget.source.setText(str(source))
 widget.project_path.setText(str(root/'runs'));widget.experiment_name.setText('cancelled')
 widget.device_combo.setCurrentText('cpu');widget.plots.setChecked(False)
 widget._on_run();end=time.monotonic()+60
 while '"event": "epoch"' not in owner.log_text.toPlainText() and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert owner.autoencoder_process is not None,owner.log_text.toPlainText()
 widget.stop_btn.click();end=time.monotonic()+30
 while owner.autoencoder_process is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert owner.autoencoder_process is None,owner.log_text.toPlainText()
 last=root/'runs/cancelled/weights/last.pt'
 assert read_checkpoint(last)['training_state']['epoch_complete'] is True
 assert widget.run_btn.isEnabled()
 assert 'autoencoder' not in owner._worker_artifacts
 owner.close();app.processEvents()
''')


if __name__=='__main__':unittest.main()
