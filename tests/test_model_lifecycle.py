"""Model metadata, native geometry, class identities and compiled export options."""
from importlib.util import find_spec
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import subprocess
import sys

import numpy as np

from utils_new.import_export.model_export import validate_export_options
from utils_new.segmentation.result import InferenceObject, ModelPrediction


class PredictionContracts(unittest.TestCase):
    def test_touching_classes_and_morphology_do_not_merge_identities(self):
        from utils_new.images.morphology import MorphologyStep

        labels = np.zeros((80, 120), np.int32)
        labels[10:60, 10:50] = 1
        labels[10:60, 50:90] = 2
        prediction = ModelPrediction.semantic(labels, dict(num_classes=3))
        np.testing.assert_array_equal(prediction.label_map, labels)
        processed = prediction.postprocessed([MorphologyStep('dilate', 3)], 'Rectangle')
        self.assertEqual([value.class_id for value in processed.objects], [0, 1])
        self.assertTrue((processed.objects[0].mask & processed.objects[1].mask).any())
        for mode in ('detect', 'obb', 'segment'):
            shapes = processed.annotations(np.zeros((80, 120, 3), np.uint8), mode)
            self.assertEqual({item[1] for item in shapes}, {0, 1})
            self.assertEqual({item[0] for item in shapes}, {mode})
        same_ids = ModelPrediction.semantic(labels, dict(num_classes=3, annotation_class_offset=0))
        self.assertEqual([value.class_id for value in same_ids.objects], [1, 2])

    def test_native_obb_keeps_subpixel_corners_until_mask_changes(self):
        from utils_new.annotations.model import Annotation, ImageSize
        from utils_new.annotations.masks import annotation_binary_mask
        from utils_new.images.morphology import MorphologyStep

        annotation = Annotation('obb', 7, [[20.25,10.5],[70.25,20.5],[60.25,60.5],[10.25,50.5]])
        mask = annotation_binary_mask(annotation, ImageSize(100, 80)) * 255
        prediction = ModelPrediction((80, 100), (InferenceObject(7, mask, annotation),), mask_kind='rasterized_geometry')
        original = prediction.postprocessed([], 'Rectangle')
        # An already convex OBB must keep subpixel geometry with hull enabled.
        shapes = original.annotations(np.zeros((80, 100, 3), np.uint8), 'obb', use_hull=True)
        self.assertEqual(shapes[0][1], 7)
        points = np.asarray(shapes[0][2]).reshape(-1, 2) * [100, 80]
        np.testing.assert_allclose(points, annotation.points_px)
        changed = prediction.postprocessed([MorphologyStep('erode', 3)], 'Rectangle')
        self.assertIsNone(changed.objects[0].native)
        self.assertEqual(changed.objects[0].class_id, 7)

    def test_yolo_structured_results_keep_masks_and_classes(self):
        from utils_new.segmentation.backends.deep_learning import YOLOSegmentor

        class Masks:
            data = np.zeros((2, 80, 120), np.float32)
            data[0, 10:40, 10:40] = 1
            data[1, 20:60, 70:100] = 1
            xy = [np.array([[10,10],[40,10],[40,40],[10,40]]),
                  np.array([[70,20],[100,20],[100,60],[70,60]])]
            def __len__(self): return len(self.data)

        boxes = SimpleNamespace(cls=np.array([3,7]), conf=np.array([.8,.9]))
        result = SimpleNamespace(obb=None, masks=Masks(), boxes=boxes)
        model = YOLOSegmentor()
        model.model = SimpleNamespace(predict=lambda *args, **kw: [result])
        prediction = model.predict_result(np.zeros((80,120,3), np.uint8))
        self.assertEqual([item.class_id for item in prediction.objects], [3,7])
        self.assertEqual(prediction.mask_kind, 'segmentation')
        self.assertEqual(prediction.combined_mask().shape, (80,120))

    def test_export_options_reject_unsupported_combinations(self):
        for options in ({'nms': True}, {'threshold': float('nan')}, {'max_det': 0}, {'opset': 15}):
            with self.assertRaises(ValueError):
                validate_export_options('U-Net', 'NCNN', options)
        self.assertFalse(validate_export_options('U-Net', 'NCNN', {'dynamic': True})['dynamic'])


@unittest.skipUnless(find_spec('torch') and find_spec('segmentation_models_pytorch'), 'training extra')
class CheckpointMetadata(unittest.TestCase):
    def test_checkpoint_loads_without_json_and_rejects_a_foreign_sidecar(self):
        import torch
        import segmentation_models_pytorch as smp
        from utils_new.import_export.training import save_weights, read_semantic_checkpoint
        from utils_new.segmentation.backends.deep_learning import UNetSegmentor

        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'trained.pth'
            meta = dict(architecture='unet', encoder='resnet18', num_classes=3, input_size=32,
                        decoder_channels=[16,8,4,2,1], normalize=True, resize_mode='hard', epoch=5,
                        annotation_class_offset=0, class_names=['background','a','b'])
            model = smp.Unet(encoder_name='resnet18',encoder_weights=None,classes=3,decoder_channels=(16,8,4,2,1))
            save_weights(path, model, meta)
            sidecar = path.with_name('trained_meta.json')
            sidecar.unlink()
            loaded = UNetSegmentor();loaded.load(str(path))
            self.assertEqual(loaded.metadata['epoch'], 5)
            self.assertEqual(loaded.metadata['annotation_class_offset'], 0)
            self.assertEqual(loaded.predict(np.zeros((25,45,3),np.uint8)).shape,(25,45))
            sidecar.write_text(json.dumps(dict(meta, num_classes=4)))
            with self.assertRaisesRegex(ValueError, 'JSON'):
                read_semantic_checkpoint(path)
            sidecar.unlink()
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, 'raw state_dict'):
                read_semantic_checkpoint(path)
            self.assertEqual(read_semantic_checkpoint(path, require_metadata=False)[1], {})


@unittest.skipUnless(all(find_spec(name) for name in ('torch', 'onnx', 'onnxruntime')), 'export extra')
class PortableMetadata(unittest.TestCase):
    def test_semantic_onnx_embeds_preprocessing_and_threshold(self):
        import torch
        from utils_new.import_export.model_export import export_semantic
        from utils_new.segmentation.backends.deep_learning import ONNXSegmentor

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.onnx'
            meta = dict(input_size=32, num_classes=1, normalize=False, resize_mode='hard')
            export_semantic(torch.nn.Conv2d(3,1,1), meta, 32, path, 'ONNX',
                            options=dict(opset=16, dynamic=False, threshold=.8))
            path.with_name('model_meta.json').unlink()
            model=ONNXSegmentor();model.load(str(path))
            self.assertEqual(model.metadata['threshold'], .8)
            self.assertFalse(model.metadata['export_parameters']['dynamic'])
            path.with_name('model_meta.json').write_text(json.dumps(dict(meta,normalize=True)))
            with self.assertRaisesRegex(ValueError, 'JSON'):
                model.load(str(path))

    def test_yolo_onnx_nms_recipe_survives_without_sidecar(self):
        from runtime_policy import configure_ultralytics
        configure_ultralytics()
        from ultralytics import YOLO
        from utils_new.import_export.model_export import export_model
        from utils_new.import_export.model_load import load_yolo

        with TemporaryDirectory() as directory:
            root=Path(directory);weights=root/'source.pt'
            YOLO('yolo11n.yaml').save(weights)
            bundle=Path(export_model('YOLO', weights, 'ONNX', root, size=64,
                                    options=dict(nms=True,conf=1e-6,iou=.45,max_det=20,opset=16)))
            path=bundle/'model.onnx'
            (bundle/'model_meta.json').unlink()
            model=load_yolo(path)
            self.assertTrue(model.neuro_metadata['export_parameters']['nms'])
            output=model.predict(np.zeros((47,61,3),np.uint8),imgsz=64,device='cpu',verbose=False)
            self.assertEqual(output[0].orig_shape, (47,61))
            self.assertGreater(len(output[0].boxes),0)
            with self.assertRaisesRegex(ValueError,'conf'):
                model.predict(np.zeros((47,61,3),np.uint8),conf=0)
            with self.assertRaisesRegex(ValueError,'IoU'):
                model.predict(np.zeros((47,61,3),np.uint8),iou=.7)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
class PredictionGui(unittest.TestCase):
    def test_native_classes_and_geometry_reach_the_saved_annotation(self):
        code = r'''
from pathlib import Path
from tempfile import TemporaryDirectory
import sys,time
import numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from tabs.deep_learning_methods import DeepLearningWindow
from utils_new.import_export.images import write_image
from utils_new.segmentation.result import ModelPrediction,InferenceObject
from utils_new.annotations.model import Annotation,ImageSize
from utils_new.annotations.masks import annotation_binary_mask
sys.path.insert(0,str(Path.cwd()/'tests'))
from dataset_test_support import wait_dataset
app=QApplication([])
class Model:
 metadata={}
 def predict(self,*a,**kw):raise AssertionError('structured prediction must be used')
 def predict_result(self,image,**kw):
  annotation=Annotation('obb',7,[[20.25,10.5],[70.25,20.5],[60.25,60.5],[10.25,50.5]])
  mask=annotation_binary_mask(annotation,ImageSize(image.shape[1],image.shape[0]))*255
  return ModelPrediction(image.shape[:2],(InferenceObject(7,mask,annotation),),mask_kind='rasterized_geometry')
with TemporaryDirectory() as directory:
 path=Path(directory)/'image.png';write_image(path,np.zeros((80,100,3),np.uint8))
 owner=DeepLearningWindow();owner.segmentor=Model();owner.hull_checkbox.setChecked(False)
 owner.draw_combo.setCurrentIndex(owner.draw_combo.findData('obb'))
 owner._load_images([str(path)]);wait_dataset(owner)
 deadline=time.monotonic()+30
 while (owner.jobs.busy or owner.prediction_mask is None) and time.monotonic()<deadline:
  app.processEvents();time.sleep(.01)
 owner.display_current_image();app.processEvents()
 assert owner.current_objects_full,owner.result_status_label.text()
 assert {item[1] for item in owner.current_objects_full}=={7},owner.current_objects_full
 assert owner.current_objects_full[0][0]=='obb'
 QMessageBox.question=lambda *a,**kw:QMessageBox.Yes
 QMessageBox.information=lambda *a,**kw:QMessageBox.Ok
 owner.save_current_annotations()
 text=path.with_suffix('.txt').read_text()
 assert text.startswith('7 '),text
 assert len(text.strip().split())==9,text
 owner.close();app.processEvents()
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], timeout=90,
                                env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
