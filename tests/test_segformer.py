"""Native MiT training/checkpoints, semantic decoding, GUI process boundaries."""
from concurrent.futures import CancelledError
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import numpy as np
import torch

from utils_new.configuration.training import semantic_config
from utils_new.import_export.training import read_semantic_checkpoint, read_weights, save_weights
from utils_new.segmentation.backends.deep_learning import create_segmentor
from utils_new.training.semantic import build_model, run_training

ROOT = Path(__file__).resolve().parents[1]


def make_data(root):
    from utils_new.import_export.images import write_image
    import yaml
    for split in ('train','val','test'):
        for index in range(2):
            image = np.random.default_rng(index+len(split)).integers(0,256,(40,70,3),dtype=np.uint8)
            mask = np.zeros((40,70),np.uint8);mask[8:33,12:59]=1
            write_image(root/'images'/split/f'{index}.png', image)
            write_image(root/'masks'/split/f'{index}.png', mask)
    path=root/'masks_data.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(root),num_classes=2,
        train_images='images/train',train_masks='masks/train',
        val_images='images/val',val_masks='masks/val',test_images='images/test',test_masks='masks/test')))
    return path


def config(source, output, **kwargs):
    return semantic_config(dict(dataset_path=str(source), project_path=str(output),
        architecture='segformer', encoder='mit_b0', input_size=64, num_classes=2,
        decoder_segmentation_channels=32, pretrained=False, batch_size=2, epochs=1,
        num_workers=0,use_amp=False,plots=False,device='cpu',loss='CrossEntropy+Dice',**kwargs), 'segformer')


class SegFormerContracts(unittest.TestCase):
    def test_all_mit_variants_and_configuration_errors(self):
        torch.set_num_threads(2)
        for index in range(6):
            cfg=semantic_config(dict(encoder=f'mit_b{index}',pretrained=False,
                decoder_segmentation_channels=768 if index == 2 else 32,input_size=64), 'segformer')
            self.assertEqual(cfg["backbone"],cfg["encoder"])
            model=build_model(cfg).eval()
            with torch.inference_mode():output=model(torch.rand(1,3,64,64))
            self.assertEqual(tuple(output.shape),(1,2,64,64))
            self.assertTrue(torch.isfinite(output).all())
            del model,output;gc.collect()
        for values in (dict(encoder='resnet50'),dict(decoder_segmentation_channels=0),dict(architecture='unet')):
            with self.assertRaises(ValueError):semantic_config(values,'segformer')
        result=subprocess.run([sys.executable,'-c',
            "import sys;import utils_new.training.segformer;assert 'torch' not in sys.modules;assert 'PyQt5' not in sys.modules"],
            cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_real_training_metadata_reload_warmstart_metrics_and_stop(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root=Path(directory);source=make_data(root/'data')
            cfg=config(source,root/'run');torch.manual_seed(cfg['seed'])
            initial=build_model(cfg).encoder.patch_embed1.proj.weight.detach().clone()
            self.assertEqual(run_training(cfg,'segformer',log=lambda _:None),0)
            weights=root/'run/weights';state,meta=read_semantic_checkpoint(weights/'best_iou.pth')
            self.assertFalse(torch.equal(initial,state['encoder.patch_embed1.proj.weight']))
            self.assertEqual(meta['encoder'],'mit_b0');self.assertEqual(meta['decoder_segmentation_channels'],32)
            self.assertEqual(meta['preprocessing_version'],'semantic-opencv-v1')
            self.assertEqual(meta['epoch'],1);self.assertTrue(meta['epoch_complete'])
            self.assertTrue(np.isfinite(meta['validation_metrics']['IoU']))
            self.assertTrue((root/'run/test_metrics.json').is_file())
            self.assertTrue((root/'run/results.csv').is_file())
            self.assertTrue((root/'run/args.json').is_file())
            # An embedded checkpoint remains usable after losing its JSON sidecar.
            local=root/'local.pth'
            moved_meta=dict(meta, unet_yaml=str(root/'deleted_source.yaml'),
                            deeplab_yaml=str(root/'deleted_deeplab.yaml'))
            save_weights(local,state,moved_meta)
            segmentor=create_segmentor('SegFormer',device='cpu');segmentor.load(weights_path=str(local))
            image=np.full((45,91,3),80,np.uint8)
            result=segmentor.predict_result(image,threshold=0)
            self.assertEqual(result.shape,image.shape[:2]);self.assertTrue(result.objects)
            for kind in ('detect','obb','segment'):
                annotations=result.annotations(image,kind)
                self.assertTrue(annotations,kind)
                self.assertEqual(annotations[0][1],0)
            for resize in ('hard','keep_aspect_pad','random_crop'):
                segmentor.metadata['resize_mode']=resize
                mask=segmentor.predict(image)
                self.assertEqual(mask.shape,image.shape[:2]);self.assertTrue(np.isin(mask,[0,255]).all())
            # Legacy raw state_dict needs its explicit metadata; never guess.
            raw=root/'raw.pth';torch.save(state,raw)
            with self.assertRaisesRegex(ValueError,'raw state_dict'):segmentor.load(str(raw))
            raw.with_name('raw_meta.json').write_text(json.dumps(meta))
            segmentor.load(str(raw))
            # Conflicting head dimensions are rejected before constructing a model.
            local.with_name('local_meta.json').write_text(json.dumps(dict(meta,decoder_segmentation_channels=64)))
            with self.assertRaisesRegex(ValueError,'JSON'):segmentor.load(str(local))
            # Full-model warmstart restores architecture from weights, not stale UI.
            cfg2=config(source,root/'warm');cfg2.update(model_path=str(weights/'last.pth'),encoder='mit_b2',decoder_segmentation_channels=64)
            self.assertEqual(run_training(cfg2,'segformer',log=lambda _:None),0)
            self.assertEqual(read_semantic_checkpoint(root/'warm/weights/last.pth')[1]['decoder_segmentation_channels'],32)
            stop=Event();cfg3=config(source,root/'stop');cfg3['epochs']=3
            def log(line):
                if line.startswith('PROGRESS:'):stop.set()
            self.assertEqual(run_training(cfg3,'segformer',stop=stop,log=log),130)
            self.assertEqual(read_semantic_checkpoint(root/'stop/weights/last.pth')[1]['epoch'],1)
            stopped=list((root/'stop/weights').glob('interrupted*pth'));self.assertEqual(len(stopped),1)
            self.assertFalse(read_semantic_checkpoint(stopped[0])[1]['epoch_complete'])

    def test_multiclass_labels_keep_annotation_class_ids(self):
        cfg=semantic_config(dict(pretrained=False,encoder='mit_b0',num_classes=3,
            decoder_segmentation_channels=32,input_size=64), 'segformer')
        model=build_model(cfg)
        with torch.no_grad():
            model.segmentation_head[0].weight.zero_();model.segmentation_head[0].bias.fill_(-10)
            model.segmentation_head[0].bias[2]=10
        with TemporaryDirectory() as directory:
            path=Path(directory)/'classes.pth'
            save_weights(path,model,dict(cfg,annotation_class_offset=1))
            segmentor=create_segmentor('SegFormer');segmentor.load(str(path))
            result=segmentor.predict_result(np.zeros((37,55,3),np.uint8))
            self.assertTrue(np.all(result.label_map==2));self.assertEqual(result.objects[0].class_id,1)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS')=='1','opt-in Qt')
class SegFormerGUI(unittest.TestCase):
    def test_native_process_from_form_then_load_and_segment_in_application(self):
        code=r'''
import tempfile,sys
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QElapsedTimer
from PyQt5.QtTest import QTest
from tabs.deep_learning_training import YoloTrainWindow
from tabs.deep_learning_methods import DeepLearningWindow
app=QApplication([]);window=YoloTrainWindow();dl=DeepLearningWindow()
assert 'torch' not in sys.modules,'UI must stay lazy'
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_segformer import make_data
from utils_new.import_export.images import read_image
window.show();dl.show()
def wait(condition):
 timer=QElapsedTimer();timer.start()
 while not condition() and timer.elapsed()<60000:QTest.qWait(10)
 assert condition(),window.log_text.toPlainText()+'\n'+dl.log_text.toPlainText()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);source=make_data(root/'data');form=window.segformer_train_widget
 form.unet_yaml.setText(str(source));form.project_path.setText(str(root/'run'))
 form.pretrained_checkbox.setChecked(False);form.input_size.addItem('64');form.input_size.setCurrentText('64')
 form.epochs.setValue(1);form.batch.setValue(2);form.auto_batch.setChecked(False)
 form.decoder_segmentation_channels.setValue(32);form.device_combo.setCurrentText('cpu')
 form.training_preset_bar.worker_parameters=dict(plots=False,num_workers=0)
 with patch.object(QMessageBox,'critical') as error,patch.object(QMessageBox,'information') as info:
  form.run_btn.click();assert window.segformer_process is not None
  assert not form.settings_tabs.isEnabled() and form.stop_btn.isEnabled()
  wait(lambda:window.segformer_process is None)
  error.assert_not_called();assert info.call_count==1
 assert form.run_btn.isEnabled() and not window._worker_artifacts
 checkpoint=Path(form._last_run_dir)/'weights/best_iou.pth';assert checkpoint.is_file()
 item=dl.model_combo.model().item(dl.model_combo.findText('SegFormer'));assert item.isEnabled()
 dl.model_combo.setCurrentText('SegFormer')
 dl.segformer_settings._model_path.setText(str(checkpoint));dl.load_model_from_widget('SegFormer')
 wait(lambda:not dl.jobs.busy);assert dl.segmentor is not None
 assert dl.segformer_settings._encoder.currentText()=='mit_b0'
 assert dl.segformer_settings._input_size.currentText()=='64'
 assert not dl.segformer_settings._normalize_override.isEnabled()
 dl.segformer_settings.apply_metadata(dict(dl.segmentor.metadata,num_classes=151))
 assert dl.segformer_settings._num_classes_override.value()==151
 dl.segformer_settings.apply_metadata(dl.segmentor.metadata)
 image=root/'data/images/train/0.png';dl._load_images([str(image)]);wait(lambda:dl.original_image is not None and not dl.jobs.busy);dl.update_timer.stop()
 dl.run_segmentation();wait(lambda:not dl.jobs.busy)
 assert dl.prediction_mask is not None,dl.log_text.toPlainText()
 assert dl.prediction_mask.shape==dl.original_image.shape[:2]
from PyQt5.QtCore import QProcess
stop_dir=tempfile.TemporaryDirectory();stop_path=Path(stop_dir.name)/'stop'
window._worker_artifacts['segformer']=(stop_dir,stop_path)
process=QProcess();window.segformer_process=process
process.finished.connect(lambda code,status:window._semantic_finished('segformer',code,status))
child="import pathlib,sys,time\nwhile not pathlib.Path(sys.argv[1]).exists():time.sleep(.01)\nsys.exit(130)"
process.start(sys.executable,['-c',child,str(stop_path)]);assert process.waitForStarted(3000)
window.close();wait(lambda:not window.isVisible() and window.segformer_process is None)
assert not window._worker_artifacts
dl.close();app.processEvents()
'''
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=100)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
