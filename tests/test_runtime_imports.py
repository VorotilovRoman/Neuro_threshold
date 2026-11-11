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

    def test_module_cache_and_no_ultralytics_autoinstall(self):
        status, output = self.run_python(['-c', """
import importlib, cv2, torch
original = cv2
# Also cover a caller who imported Ultralytics before application policy.
import ultralytics
from runtime_policy import configure_ultralytics
configure_ultralytics()
from ultralytics.utils import checks
from unittest.mock import patch
assert not checks.AUTOINSTALL
with patch.object(checks.subprocess, 'check_output') as installer:
    checks.check_requirements('neuro_architecture_missing_backend_12345==0')
    installer.assert_not_called()
for _ in range(10):
    assert importlib.import_module('cv2') is original
    assert importlib.import_module('torch') is torch
assert ultralytics.utils.patches.cv2 is original
"""])
        self.assertEqual(status, 0, output)

    def test_embedded_editor_viewer_buttons_load_and_toggle(self):
        status, output = self.run_python(['-c', """
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2, numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from main import ImagePlayer
from ui.application import exec_window
app = QApplication([])
window = ImagePlayer([('labeler','Editor'), ('test','Viewer')])
window.show()
with tempfile.TemporaryDirectory() as directory:
    image = Path(directory)/'sample.png'
    cv2.imwrite(str(image), np.zeros((24,32,3), dtype=np.uint8))
    for index, editor in enumerate(window._tab_widgets):
        window.tab_widget.setCurrentIndex(index)
        app.processEvents()
        with patch.object(QFileDialog, 'getOpenFileNames', return_value=([str(image)], '')) as dialog:
            QTest.mouseClick(editor.nav_widget._load_images_btn, Qt.LeftButton)
            app.processEvents()
            dialog.assert_called_once()
        assert editor.image_paths == [str(image)]
        assert (editor.save_button if index == 0 else editor.btn_save).isEnabled()
        view = editor.image_view if index == 0 else editor.viewer
        assert view.display_style.show_frames
        QTest.mouseClick(editor.btn_hide_frames, Qt.LeftButton)
        assert not view.display_style.show_frames
        QTest.mouseClick(editor.btn_hide_frames, Qt.LeftButton)
        assert view.display_style.show_frames
        if editor._lazy_cache is not None: editor._lazy_cache.shutdown()
    QTimer.singleShot(10,window.close)
    assert exec_window(window,app) == 0
"""])
        self.assertEqual(status, 0, output)

    def test_method_tabs_keep_visible_window_and_dialog_parent(self):
        status, output = self.run_python(['-c', """
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtTest import QTest
from main import ImagePlayer
from ui.application import exec_window
app = QApplication([])
keys = ['threshold', 'gradient', 'interactive', 'traditional_ml', 'deep_learning']
window = ImagePlayer([(key,key) for key in keys])
window.show()
with tempfile.TemporaryDirectory() as directory:
    image = Path(directory)/'methods.png'
    cv2.imwrite(str(image), np.tile(np.arange(64, dtype=np.uint8)*4, (64,1)))
    for index, method in enumerate(window._tab_widgets):
        window.tab_widget.setCurrentIndex(index)
        app.processEvents()
        assert method.isVisible(), keys[index] + ': controller window is hidden'
        assert not method.isWindow(), keys[index] + ': detached top-level window'
        assert method.centralWidget() is not None
        button = method.nav_widget._load_images_btn
        point = button.mapToGlobal(button.rect().center())
        assert app.widgetAt(point) is button, keys[index] + ': another widget covers the button'
        def select(parent, *args, **kwargs):
            assert parent.isVisible(), keys[index] + ': file dialog owner is hidden'
            return [str(image)], ''
        with patch.object(QFileDialog, 'getOpenFileNames', side_effect=select) as dialog:
            QTest.mouseClick(window.windowHandle(), Qt.LeftButton,
                             pos=window.mapFromGlobal(point))
            dialog.assert_called_once()
        QTest.qWait(150)
        gray = method.gray_images[0] if index == 3 else method.current_gray
        assert gray is not None and gray.shape == (64,64), keys[index] + ': image not loaded'
        if index == 0:
            method.method_combo.setCurrentIndex(0)
            method.threshold_slider.setValue(40)
            QTest.qWait(150)
            assert method.current_threshold == 40
            first = method.current_base_image.copy()
            method.threshold_slider.setValue(200)
            QTest.qWait(150)
            assert method.current_threshold == 200
            assert not np.array_equal(first, method.current_base_image)
        cache = getattr(method, '_lazy_cache', None)
        if cache is not None: cache.shutdown()
    QTimer.singleShot(10,window.close)
    assert exec_window(window,app) == 0
"""])
        self.assertEqual(status, 0, output)

    def test_worker_help_without_qt_or_training(self):
        for name in ('train_yolo_worker.py', 'train_unet_worker.py',
                     'train_unet_worker_рабочий.py', 'train_DeepLabV3_worker.py', 'yolo2voc.py'):
            with self.subTest(worker=name):
                path = ROOT / 'utils_training' / name
                status, output = self.run_python([str(path), '--help'], self.temp.name)
                self.assertEqual(status, 0, output)
                self.assertIn('--yaml' if name == 'yolo2voc.py' else '--params', output)


if __name__ == '__main__':
    unittest.main()
