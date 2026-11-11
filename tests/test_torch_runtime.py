"""Native-loader failure regressions; no Windows DLL needed for fault injection."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import runtime_torch

ROOT = Path(__file__).resolve().parents[1]


class TorchRuntimeTests(unittest.TestCase):
    def test_dll_failure_is_cached_and_not_reported_as_working_cpu(self):
        with patch.object(runtime_torch, '_runtime', None), \
                patch.object(runtime_torch.importlib, 'import_module',
                             side_effect=OSError('[WinError 1114] c10.dll init failed')) as loader:
            first = runtime_torch.get_torch_runtime()
            self.assertIs(runtime_torch.get_torch_runtime(), first)
            self.assertIsNone(first.module)
            self.assertIn('1114', first.error)
            with self.assertRaisesRegex(RuntimeError, '1114'):
                runtime_torch.require_torch()
            loader.assert_called_once_with('torch')

    def test_successful_cpu_import_is_reused(self):
        module = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(runtime_torch, '_runtime', None), \
                patch.object(runtime_torch.importlib, 'import_module', return_value=module) as loader:
            self.assertIs(runtime_torch.require_torch(), module)
            self.assertEqual(runtime_torch.get_torch_runtime().device, 'cpu')
            self.assertEqual(runtime_torch.get_torch_runtime().error, '')
            loader.assert_called_once()

    def test_cuda_probe_failure_preserves_loaded_cpu_module_and_error(self):
        module = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(side_effect=OSError('driver error'))))
        with patch.object(runtime_torch, '_runtime', None), \
                patch.object(runtime_torch.importlib, 'import_module', return_value=module):
            status = runtime_torch.get_torch_runtime()
            self.assertIs(status.module, module)
            self.assertEqual(status.device, 'cpu')
            self.assertIn('driver error', status.error)

    def test_only_windows_preloads_torch(self):
        with patch.object(runtime_torch, 'get_torch_runtime') as probe:
            with patch.object(runtime_torch.sys, 'platform', 'linux'):
                self.assertIsNone(runtime_torch.prepare_windows_torch())
                probe.assert_not_called()
            with patch.object(runtime_torch.sys, 'platform', 'win32'):
                self.assertIs(runtime_torch.prepare_windows_torch(), probe.return_value)
                probe.assert_called_once()

    @unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in GUI/runtime suite')
    def test_analysis_window_and_model_worker_survive_dll_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', NO_ALBUMENTATIONS_UPDATE='1',
                       XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory, MPLCONFIGDIR=directory)
            code = '''
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication, QEvent
from unittest.mock import patch
import runtime_torch
from tabs.dataset_analysis import DatasetAnalysisWindow, YoloValThread
app = QApplication([])
with patch.object(runtime_torch, '_runtime', None), patch.object(
        runtime_torch.importlib, 'import_module', side_effect=OSError('[WinError 1114] c10.dll')) as loader:
    windows = [DatasetAnalysisWindow(), DatasetAnalysisWindow()]
    assert runtime_torch.get_torch_runtime().module is None
    worker = YoloValThread('not-loaded.pt', 'not-loaded.yaml', device='cpu')
    results = []
    worker.finished_signal.connect(results.append)
    worker.run()
    assert results and '1114' in results[0]['error']
    loader.assert_called_once_with('torch')
    for window in windows:
        assert window.validator is not None and window.analyzer is not None
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
'''
            result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
