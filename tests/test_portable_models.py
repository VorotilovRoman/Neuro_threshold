"""Executable conversion contracts, publishing, shared loading and Qt lifecycle."""
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from utils_new.import_export.model_export import export_capabilities, export_model
from utils_new.import_export.model_load import load_yolo, ncnn_param


class PortableContracts(unittest.TestCase):
    def test_unsupported_formats_are_explicit_and_do_not_touch_weights(self):
        with TemporaryDirectory() as directory:
            weights = Path(directory) / 'model.pt'
            weights.write_bytes(b'original')
            for kind in ('YOLO', 'U-Net', 'DeepLabV3+', 'SAM', 'ONNX / NCNN'):
                with self.assertRaisesRegex(ValueError, 'GGUF'):
                    export_model(kind, weights, 'GGUF', directory)
            self.assertEqual(weights.read_bytes(), b'original')
            self.assertEqual(list(Path(directory).iterdir()), [weights])
        self.assertTrue(export_capabilities('SAM', 'sam.pt')['ONNX'])
        self.assertTrue(export_capabilities('YOLO', 'model.onnx')['NCNN'])
        self.assertTrue(export_capabilities('ONNX / NCNN', 'model.param')['ONNX'])

    def test_ncnn_requires_graph_weight_pair_and_unambiguous_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            param = root / 'model.param'
            param.write_text('7767517\n0 0\n')
            with self.assertRaisesRegex(ValueError, '.bin'):
                ncnn_param(param)
            param.with_suffix('.bin').touch()
            self.assertEqual(ncnn_param(root), param)
            (root / 'other.param').write_text('7767517\n0 0\n')
            with self.assertRaisesRegex(ValueError, 'ровно один'):
                ncnn_param(root)
            param.write_text('not a graph')
            with self.assertRaisesRegex(ValueError, 'заголовок'):
                ncnn_param(param)

    def test_gguf_selection_reports_missing_runtime(self):
        with self.assertRaisesRegex(ValueError, 'backend'):
            load_yolo('local.gguf')

    def test_ncnn_unaligned_size_is_rejected_before_native_conversion(self):
        with self.assertRaisesRegex(ValueError, 'кратный 32'):
            export_model('DeepLabV3+', 'unused.pth', 'NCNN', '.', size=65)

    def test_failed_validation_publishes_nothing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / 'source.pth'
            weights.write_bytes(b'original')
            with patch('utils_new.import_export.model_export.load_export_model',
                       return_value=(type('Adapter', (), {'model': object()})(), {})), \
                 patch('importlib.util.find_spec', return_value=object()), \
                 patch('utils_new.import_export.model_export.export_semantic', side_effect=ValueError('parity')):
                with self.assertRaisesRegex(ValueError, 'parity'):
                    export_model('U-Net', weights, 'NCNN', root, size=64)
            self.assertEqual(list(root.iterdir()), [weights])


@unittest.skipUnless(all(find_spec(name) for name in ('torch', 'onnx', 'onnxruntime', 'pnnx', 'ncnn')),
                     'optional export profile')
class NativePortableModels(unittest.TestCase):
    def test_semantic_metadata_and_native_input_contracts_are_strict(self):
        import json
        from utils_new.import_export.model_load import NCNNRuntime
        from utils_new.segmentation.backends.deep_learning import ONNXSegmentor

        runtime = NCNNRuntime.__new__(NCNNRuntime)
        for invalid in (np.ones((1,3,4,4), np.float64), np.ones((1,1,4,4), np.float32),
                        np.full((1,3,4,4), np.nan, np.float32), np.ones((1,3,0,4), np.float32)):
            with self.assertRaisesRegex(ValueError, 'float32'):
                runtime.run(invalid)  # Must fail before accessing the native net.
        model = ONNXSegmentor()
        meta = dict(input_size=32, num_classes=2, normalize=False, resize_mode='hard', output_kind='logits')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.onnx'
            sidecar = path.with_name('graph_meta.json')
            for invalid in (dict(meta,input_size=True), dict(meta,num_classes=0), dict(meta,resize_mode='typo')):
                sidecar.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    model._load_metadata(str(path))
            for field, value in (('threshold', .8), ('inference_resize_mode', 'hard'), ('class_names', ['a','b'])):
                embedded = dict(meta,threshold=.5,inference_resize_mode='keep_aspect_pad',class_names=['bg','object'])
                sidecar.write_text(json.dumps(dict(embedded, **{field:value})))
                with self.assertRaisesRegex(ValueError, 'JSON'):
                    model._load_metadata(str(path), embedded)
            sidecar.write_text(json.dumps(meta));model._load_metadata(str(path))
            model._input_shape = lambda: [1,3,32,32]
            for output in (np.zeros((1,84,100), np.float32), np.zeros((1,3,32,32), np.float32)):
                model._run_tensor = lambda tensor, result=output: result
                with self.assertRaises(ValueError):
                    model.predict(np.zeros((11,19,3), np.uint8))

    def test_ncnn_unicode_paths_use_stream_trace_relative_converter_and_windows_weight_staging(self):
        import torch
        import subprocess
        from utils_new.import_export.model_export import export_semantic
        from utils_new.import_export.model_load import NCNNRuntime

        torch.set_num_threads(1)
        model = torch.nn.Conv2d(3, 2, 1).eval()
        metadata = dict(input_size=32, normalize=False, resize_mode='hard',
                        output_kind='logits', model_type='U-Net', num_classes=2)
        with TemporaryDirectory() as directory:
            root = Path(directory) / 'Северная заря (модели)'
            root.mkdir()
            target = root / 'model_ncnn_model'
            before = Path.cwd()
            with patch('torch.jit.save', wraps=torch.jit.save) as save, \
                    patch('subprocess.run', wraps=subprocess.run) as convert:
                export_semantic(model, metadata, 32, target, 'NCNN')
            self.assertTrue(hasattr(save.call_args.args[1], 'write'))
            self.assertEqual(convert.call_args.args[0][1], 'trace.pt')
            self.assertEqual(convert.call_args.kwargs['cwd'], target)
            self.assertEqual(Path.cwd(), before)
            with patch('sys.platform', 'win32'):
                runtime = NCNNRuntime(target / 'model.ncnn.param')
            data = np.ones((1, 3, 32, 32), np.float32)
            with torch.inference_mode():
                expected = model(torch.from_numpy(data)).numpy()
            np.testing.assert_allclose(runtime.run(data)[0], expected, rtol=2e-3, atol=2e-4)
            self.assertFalse(list(target.glob('trace*')))

    def test_ncnn_converter_failure_preserves_diagnostics(self):
        import torch
        import subprocess
        from utils_new.import_export.model_export import export_semantic
        with TemporaryDirectory() as directory, \
                patch('subprocess.run', return_value=subprocess.CompletedProcess([], 9, b'', b'unsupported layer')):
            with self.assertRaisesRegex(RuntimeError, 'unsupported layer'):
                export_semantic(torch.nn.Conv2d(3, 2, 1), dict(normalize=False, resize_mode='hard'),
                                32, Path(directory) / 'ncnn', 'NCNN')

    def test_semantic_logits_and_masks_roundtrip_for_both_formats(self):
        import torch
        from utils_new.import_export.model_export import export_semantic
        from utils_new.segmentation.backends.deep_learning import create_segmentor

        torch.set_num_threads(1)
        # Non-square source + normalization + centered padding exercise geometry
        # and RGB/BGR/channel decoding independently of converter numerics.
        model = torch.nn.Conv2d(3, 2, 1).eval()
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.weight[1, 0] = 1
        metadata = dict(input_size=32, normalize=True, resize_mode='keep_aspect_pad',
                        output_kind='logits', model_type='U-Net', num_classes=2)
        image = np.zeros((15, 43, 3), np.uint8)
        image[:, :20, 2] = 255
        masks = []
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for format in ('ONNX', 'NCNN'):
                target = root / ('model.onnx' if format == 'ONNX' else 'model_ncnn_model')
                export_semantic(model, metadata, 32, target, format)
                path = target if format == 'ONNX' else target / 'model.ncnn.param'
                segmentor = create_segmentor('ONNX / NCNN')
                segmentor.load(str(path))
                mask = segmentor.predict(image)
                self.assertEqual(mask.shape, image.shape[:2])
                self.assertTrue((mask[:, :15] == 255).all())
                self.assertTrue((mask[:, 27:] == 0).all())
                masks.append(mask)
            np.testing.assert_array_equal(*masks)

    def test_yolo_task_metadata_and_predictions_roundtrip(self):
        from runtime_policy import configure_ultralytics
        configure_ultralytics()
        from ultralytics import YOLO

        with TemporaryDirectory() as directory:
            root = Path(directory) / 'Северная заря (модели YOLO)'
            root.mkdir()
            path = root / 'source.pt'
            model = YOLO('yolo11n-seg.yaml')  # packaged architecture, no downloads
            model.save(str(path))
            before = path.read_bytes()
            image = np.zeros((47, 61, 3), np.uint8)
            for format in ('ONNX', 'NCNN'):
                bundle = Path(export_model('YOLO', path, format, root, size=64))
                artifact = next(bundle.glob('*.onnx' if format == 'ONNX' else '*.param'))
                with patch('utils_new.import_export.model_load._native_path_unsafe',
                           side_effect=lambda path: 'Северная' in str(path)):
                    loaded = load_yolo(artifact)
                self.assertEqual(loaded.task, 'segment')
                result = loaded.predict(image, imgsz=64, device='cpu', verbose=False)
                self.assertEqual(result[0].orig_shape, image.shape[:2])
                if format == 'NCNN':
                    self.assertTrue((bundle / 'metadata.yaml').is_file())
                    self.assertTrue(Path(loaded._neuro_runtime_directory.name).is_dir())
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / 'source.onnx').exists())


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt')
class ExportGuiTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('segmentation_models_pytorch') and find_spec('onnx'), 'optional training/export')
    def test_successful_load_export_and_source_change(self):
        code = r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication
import segmentation_models_pytorch as smp
from utils_new.import_export.training import save_weights
from ui.model_export import ModelExportWidget
app=QApplication([])
def wait(widget):
 end=time.monotonic()+60
 while widget.process is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert widget.process is None,'worker timeout'
with TemporaryDirectory() as directory:
 root=Path(directory);path=root/'unet.pth'
 model=smp.Unet(encoder_name='resnet18',encoder_weights=None,classes=1,decoder_channels=(16,8,4,2,1))
 save_weights(path,model,dict(model_type='U-Net',encoder='resnet18',architecture='unet',input_size=32,
   num_classes=1,decoder_channels=[16,8,4,2,1],normalize=False,resize_mode='hard'))
 widget=ModelExportWidget();widget.kind.setCurrentText('U-Net');widget.path.setText(str(path))
 widget.inspect();wait(widget)
 assert widget.loaded is not None,widget.status.text()
 assert widget.load.styleSheet()
 assert widget.buttons['ONNX'].isEnabled()
 widget.output.setText(str(root/'exports'));widget.start_export('ONNX');wait(widget)
 assert widget.status.text().startswith('Сохранено:'),widget.status.text()
 assert list((root/'exports').rglob('*.onnx'))
 assert list((root/'exports').rglob('*_meta.json'))
 assert not list((root/'exports').glob('.neuro*'))
 widget.size.setValue(33)
 assert not widget.buttons['NCNN'].isEnabled()
 widget.path.setText(str(root/'different.pth'))
 assert widget.loaded is None
 assert not widget.buttons['ONNX'].isEnabled()
 widget.close()
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=150,
                                env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_export_tab_loading_failure_and_cancellation(self):
        code = r'''
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QProcess
from tabs.deep_learning_training import YoloTrainWindow
app=QApplication([])
owner=YoloTrainWindow()
widget=owner.model_export_widget
assert owner.tab_widget.tabText(4)=='SAM'
assert owner.tab_widget.tabText(5)=='Экспорт модели'
assert not widget.buttons['GGUF'].isEnabled()
assert 'GGUF' in widget.buttons['GGUF'].toolTip()
with TemporaryDirectory() as directory:
 path=Path(directory)/'bad.pt';path.write_bytes(b'broken')
 widget.path.setText(str(path));widget.inspect()
 assert widget.process is not None
 assert not widget.load.isEnabled()
 end=time.monotonic()+60
 while widget.process is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert widget.process is None, 'worker did not finish'
 assert widget.loaded is None
 assert widget.status.text().startswith('Ошибка:'),widget.status.text()
 assert not widget.buttons['ONNX'].isEnabled()
 widget.inspect();widget.cancel()
 end=time.monotonic()+20
 while widget.process is not None and time.monotonic()<end:
  app.processEvents();time.sleep(.01)
 assert widget.process is None
 assert widget.status.text()=='Операция отменена.'
 assert widget.workspace is None
owner.close();app.processEvents()
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=100,
                                env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
