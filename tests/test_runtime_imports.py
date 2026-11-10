"""Opt-in GUI/process integration checks: NEURO_RUN_GUI_TESTS=1 python -m unittest discover -s tests."""
import concurrent.futures
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABS = sorted(p.stem for p in (ROOT / 'tabs').glob('*.py'))


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in GUI/runtime suite')
class RuntimeImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = dict(os.environ, QT_QPA_PLATFORM='offscreen',
                        XDG_CONFIG_HOME=self.temp.name, XDG_CACHE_HOME=self.temp.name,
                        MPLCONFIGDIR=self.temp.name, YOLO_CONFIG_DIR=self.temp.name,
                        NO_ALBUMENTATIONS_UPDATE='1')
        self.env.pop('PYTHONPATH', None)

    def run_python(self, args, cwd=ROOT):
        result = subprocess.run([sys.executable, *args], cwd=cwd, env=self.env,
                                capture_output=True, text=True, timeout=90)
        return result.returncode, result.stdout + result.stderr

    def test_each_tab_without_main_and_both_orders(self):
        def check(name):
            code = f'import importlib; importlib.import_module("tabs.{name}")'
            return name, self.run_python(['-c', code])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for name, (status, output) in pool.map(check, TABS):
                with self.subTest(tab=name):
                    self.assertEqual(status, 0, output)
        for order in (TABS, TABS[::-1]):
            status, output = self.run_python(['-c', f'''import importlib, sys
for name in {order!r}: importlib.import_module('tabs.' + name)
assert 'main' not in sys.modules
'''])
            self.assertEqual(status, 0, output)

    def test_all_tabs_together(self):
        status, output = self.run_python(['-c', '''
from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication([])
def fail(*args, **kwargs): raise AssertionError(str(args[1:]))
QMessageBox.warning = fail
QMessageBox.critical = fail
from main import ImagePlayer
from main_ui import ALL_TABS
window = ImagePlayer(ALL_TABS)
window.show()
app.processEvents()
assert window.tab_widget.count() == len(ALL_TABS)
from PyQt5.QtCore import QTimer
from ui.application import exec_window
QTimer.singleShot(100, window.close)
assert exec_window(window, app) == 0
'''])
        self.assertEqual(status, 0, output)

    def test_direct_and_module_entrypoints(self):
        def check(case):
            name, mode = case
            # A real Qt event loop exits through a timer; no product test flag.
            code = '''import runpy, sys
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
original = QApplication.exec_
def timed_exec(self):
    QTimer.singleShot(100, self.quit)
    return original()
QApplication.exec_ = timed_exec
def fail(*args, **kwargs): raise AssertionError(str(args[1:]))
QMessageBox.warning = fail
QMessageBox.critical = fail
'''
            if mode == 'direct':
                code += f'runpy.run_path({str(ROOT / "tabs" / (name + ".py"))!r}, run_name="__main__")'
                cwd = self.temp.name
            else:
                code += f'runpy.run_module("tabs.{name}", run_name="__main__")'
                cwd = ROOT
            return case, self.run_python(['-c', code], cwd)
        cases = [(name, mode) for name in TABS for mode in ('direct', 'module')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for case, (status, output) in pool.map(check, cases):
                with self.subTest(case=case):
                    self.assertEqual(status, 0, output)

    def test_worker_help_without_qt_or_training(self):
        for name in ('train_yolo_worker.py', 'train_unet_worker.py',
                     'train_unet_worker_рабочий.py', 'train_DeepLabV3_worker.py'):
            with self.subTest(worker=name):
                path = ROOT / 'utils_training' / name
                status, output = self.run_python([str(path), '--help'], self.temp.name)
                self.assertEqual(status, 0, output)
                self.assertIn('--params', output)


if __name__ == '__main__':
    unittest.main()
