"""Stage navigation must preserve worker payloads and live widget behavior."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS')=='1','opt-in Qt')
class TrainingStageUI(unittest.TestCase):
    def test_payloads_footer_and_model_source_switches_survive_grouping(self):
        code=r'''
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
from ui.DL_train_settings_base import BaseTrainWidget,YOLOTrainWidget,UNetTrainWidget
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget,SegFormerTrainWidget
from ui.sam.training import SAMTrainWidget
app=QApplication([]);errors=[]
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
with TemporaryDirectory() as directory:
 root=Path(directory);source=root/'data.yaml'
 source.write_text('path: .\ntrain: images/train\nval: images/val\nnames: [object]\n')
 weights=root/'local.pt';weights.write_bytes(b'not loaded by UI parameter collection')
 def prepare(widget):
  widget.project_path.setText(str(root/'runs'))
  if hasattr(widget,'dataset_folder'):widget.dataset_folder.setText(str(root))
  elif hasattr(widget,'dataset_path'):widget.dataset_path.setText(str(root))
  else:widget.source.setText(str(source));widget.weights.setText(str(weights))
 for cls in (YOLOTrainWidget,UNetTrainWidget,DeepLabV3TrainWidget,SegFormerTrainWidget,SAMTrainWidget):
  with patch.object(BaseTrainWidget,'organize_training_stages'):
   before=cls();prepare(before);expected=before.get_params()
  widget=cls();prepare(widget);actual=widget.get_params()
  assert actual is not None and actual==expected,(cls.__name__,errors,actual,expected)
  widget.resize(1000,750);widget.show();app.processEvents()
  for index in range(widget.settings_tabs.count()):
   widget.settings_tabs.setCurrentIndex(index);app.processEvents()
   assert widget.run_btn.isVisible() and widget.progress_bar.isVisible(),cls.__name__
   assert not widget.settings_tabs.isAncestorOf(widget.stop_btn)
  widget.settings_tabs.setCurrentIndex(list(widget.stage_widgets).index('model'));app.processEvents()
  if hasattr(widget,'model_from_file'):
   widget.model_from_file.setChecked(True);app.processEvents()
   assert widget.file_widget.isVisible() and widget.preset_widget.isHidden(),cls.__name__
   widget.custom_model_path.setText(str(weights))
   assert widget.get_params() is not None,errors
   widget.model_from_preset.setChecked(True)
   assert not widget.preset_widget.isHidden()
  else:
   widget.settings_tabs.setCurrentIndex(list(widget.stage_widgets).index('prompts'));app.processEvents()
   assert widget.positive_points.isVisible() and widget.negative_points.isVisible()
   assert widget.stage_widgets['data'].isAncestorOf(widget.ignore_index)
  rate=widget.lr0 if hasattr(widget,'lr0') else widget.lr
  rate.setValue(rate.value()*2)
  assert widget.get_params()!=actual,cls.__name__
  received=[];widget.run_training.connect(received.append)
  widget.run_btn.click();assert received and received[0]==widget.get_params(),errors
  stops=[];widget.stop_training.connect(lambda:stops.append(True))
  widget.set_running(True);assert not widget.settings_tabs.isEnabled() and widget.stop_btn.isEnabled()
  widget.stop_btn.click();assert stops
  widget.set_running(False);assert widget.settings_tabs.isEnabled() and widget.run_btn.isEnabled()
  before.close();widget.close();app.processEvents()
 assert not errors,errors
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_deeplab_preset_and_two_phase_connections_work_across_stages(self):
        code=r'''
from PyQt5.QtWidgets import QApplication
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget
app=QApplication([]);widget=DeepLabV3TrainWidget()
widget.two_phase_check.setChecked(False);widget.two_phase_check.setChecked(True)
widget.freeze_epochs.setValue(3);widget.unfreeze_epochs.setValue(8)
assert widget.epochs.value()==11
widget.two_phase_check.setChecked(False);assert widget.epochs.isEnabled()
widget.class_weights_mode.setCurrentText('From file');assert widget.class_weights_file.isEnabled()
widget.class_weights_mode.setCurrentText('None');assert not widget.class_weights_file.isEnabled()
widget._apply_preset_dict(widget.HIGH_RES_PRESET)
assert widget.num_classes.value()>0
for name,stage in [('scheduler','training'),('num_classes','model'),('use_amp','advanced'),('threshold','validation'),('aug_hflip','augmentation'),('loss','loss')]:
 assert widget.stage_widgets[stage].isAncestorOf(getattr(widget,name)),name
widget.close()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
