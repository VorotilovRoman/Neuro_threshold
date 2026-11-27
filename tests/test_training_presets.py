"""Complete training recipes: cross-page state, persistence and backend payloads."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
class TrainingPresetUI(unittest.TestCase):
    def test_all_recipes_roundtrip_and_reach_workers(self):
        code = r'''
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication,QMessageBox
from ui.DL_train_settings_base import YOLOTrainWidget,UNetTrainWidget
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget,SegFormerTrainWidget
from ui.sam.training import SAMTrainWidget
from utils_new.configuration.presets_store import read_presets
from utils_new.configuration.training import build_train_args,semantic_config
app=QApplication([]);errors=[]
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
with TemporaryDirectory() as directory:
 root=Path(directory);source=root/'data.yaml'
 source.write_text('path: .\ntrain: images/train\nval: images/val\nnames: [object]\nkpt_shape: [3, 3]\nflip_idx: [0, 1, 2]\n')
 (root/'train/part').mkdir(parents=True);(root/'val/part').mkdir(parents=True)
 weights=root/'local.pt';weights.write_bytes(b'UI only')
 count=0
 for cls in (YOLOTrainWidget,UNetTrainWidget,DeepLabV3TrainWidget,SAMTrainWidget,SegFormerTrainWidget):
  widget=cls();bar=widget.training_preset_bar
  if hasattr(widget,'dataset_folder'):widget.dataset_folder.setText(str(root))
  elif hasattr(widget,'dataset_path'):widget.dataset_path.setText(str(root))
  else:widget.source.setText(str(source))
  widget.project_path.setText(str(root/'runs'));widget.experiment_name.setText('my-run')
  paths={key:value for key,value in bar.snapshot(True)['settings'].items() if key in ('dataset_folder','dataset_path','source','project_path','experiment_name')}
  recipes=[bar.selector.itemData(i) for i in range(1,bar.selector.count())]
  for preset in recipes+recipes[:1]:
   bar.apply_preset(preset)
   assert bar.snapshot()['settings']==preset['settings'],(cls.__name__,preset['title'])
   assert {key:bar.snapshot(True)['settings'][key] for key in paths}==paths
   if cls==SAMTrainWidget:widget.weights.setText(str(weights))
   params=widget.get_params();assert params is not None,(cls.__name__,errors)
   if cls==YOLOTrainWidget:
    cfg=build_train_args(params)
    for key,value in preset['worker_parameters'].items():assert cfg[key]==value,(key,cfg[key],value)
   elif cls in (UNetTrainWidget,DeepLabV3TrainWidget,SegFormerTrainWidget):
    cfg=semantic_config(params,bar.family)
    for key,value in preset['worker_parameters'].items():assert cfg[key]==value,(key,cfg[key],value)
   else:
    for key,value in preset['worker_parameters'].items():assert params[key]==value
   # Saving paths is optional; the same form snapshot must roundtrip exactly.
   for include in (False,True):
    if cls==SAMTrainWidget:widget.weights.setText(str(weights))
    path=root/(bar.family+str(include)+'.json');before=bar.snapshot(include)
    bar.save_to(path,include);saved=read_presets(path)
    assert saved==before,(cls.__name__,before,saved)
    widget.epochs.setValue(widget.epochs.value()+1)
    if cls==SAMTrainWidget:
     widget.model.setCurrentIndex((widget.model.currentIndex()+1)%widget.model.count())
    bar.apply_preset(saved)
    assert bar.snapshot(include)==saved,cls.__name__
   count+=1
  if cls==SAMTrainWidget:
   import json
   widget.weights.setText(str(weights))
   resume=root/'last.resume.pt';resume.write_bytes(b'UI checkpoint')
   resume.with_name('last.resume_meta.json').write_text(json.dumps(dict(kind='sam',complete_epoch=True,config=widget.get_params())))
   widget.resume.setText(str(resume));saved=bar.snapshot(True)
   widget.model.setCurrentIndex((widget.model.currentIndex()+1)%widget.model.count())
   bar.apply_preset(saved)
   assert widget.resume.text()==str(resume) and widget.weights.text()==str(weights)
   assert widget.get_params()['resume']==str(resume)
   bar.apply_preset(recipes[0]);assert not widget.resume.text()
   widget.weights.setText(str(weights))
  # A malformed recipe must leave every control and hidden worker field intact.
  before=bar.snapshot(True)
  for mutation in ('family','missing','unknown','range','worker'):
   bad=deepcopy(recipes[0])
   if mutation=='family':bad['family']='other'
   if mutation=='missing':bad['settings'].pop('epochs')
   if mutation=='unknown':bad['settings']['bogus']=1
   if mutation=='range':bad['settings']['epochs']=-1
   if mutation=='worker':bad['worker_parameters']['bogus']=True
   try:bar.apply_preset(bad)
   except ValueError:pass
   else:raise AssertionError(mutation)
   assert bar.snapshot(True)==before,mutation
  widget.set_running(True);assert not bar.isEnabled()
  widget.set_running(False);assert bar.isEnabled()
  widget.close()
 assert count==23,count
 assert not errors,errors
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_semantic_recipes_build_real_architectures_without_download(self):
        code = r'''
import gc,json
from pathlib import Path
import torch
from PyQt5.QtWidgets import QApplication
from ui.DL_train_settings_base import UNetTrainWidget
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget
from utils_new.configuration.training import semantic_config
from utils_new.training.semantic import build_model
app=QApplication([]);torch.set_num_threads(2)
for cls in (UNetTrainWidget,DeepLabV3TrainWidget):
 widget=cls();widget.dataset_path.setText(str(Path('.').resolve()))
 bar=widget.training_preset_bar
 for index in range(1,bar.selector.count()):
  bar.apply_preset(bar.selector.itemData(index));params=widget.get_params()
  assert params is not None
  cfg=semantic_config(dict(params,pretrained=False,input_size=64),bar.family)
  model=build_model(cfg);model.train()
  output=model(torch.rand(2,3,64,64))
  assert output.shape==(2,2,64,64),(cfg['architecture'],output.shape)
  output.square().mean().backward()
  assert any(p.grad is not None for p in model.parameters())
  del output,model;gc.collect()
 widget.close()
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TrainingPresetBackend(unittest.TestCase):
    def test_yolo_recipe_augmentation_disabled_explicitly(self):
        from utils_new.configuration.training import build_train_args,validate_yolo_options
        from utils_new.configuration.presets_store import TRAINING_WORKER_DEFAULTS
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'data.yaml';source.write_text('names: [object]')
            params=dict(TRAINING_WORKER_DEFAULTS['yolo'],model='yolo11n.pt',data=str(source),disable_aug=True)
            cfg=build_train_args(params)
            self.assertEqual(cfg['fliplr'],0)
            self.assertEqual(cfg['hsv_s'],0)
            self.assertEqual(cfg['mosaic'],0)
            self.assertEqual(cfg['workers'],0)
            self.assertEqual(cfg['seed'],42)
        for value in ({'amp':1},{'workers':-1},{'mixup':2},{'lr0':.001}):
            with self.assertRaises(ValueError):validate_yolo_options(value)

    def test_sam31_not_offered_as_trainable(self):
        from utils_new.configuration.sam_training import sam_training_config
        with self.assertRaisesRegex(ValueError,'Обучение поддерживает'):
            sam_training_config({'model_key':'sam3.1'})


if __name__ == '__main__':unittest.main()
