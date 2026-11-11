"""Regression coverage for master 5043de7's shared label-save workflow."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils_new.dataset_yaml_new import build_yolo_save_target_map
from utils_new.dataset_yolo_labels_new import (
    build_label_save_plan,
    format_label_save_confirmation,
)

ROOT = Path(__file__).resolve().parents[1]


class LabelSaveMergeTests(unittest.TestCase):
    def test_yaml_save_targets_are_separate_and_preserve_subpaths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / 'images/train/nested/a.png'
            valid = root / 'images/valid/b.png'
            for image in (train, valid):
                image.parent.mkdir(parents=True, exist_ok=True)
                image.touch()
            config = root / 'data.yaml'
            config.write_text('train: images/train\nvalidation: images/valid\n')
            targets = build_yolo_save_target_map(config, [str(train), str(valid)])
            self.assertEqual(Path(targets[str(train)]), root / 'labels/train/nested/a.txt')
            self.assertEqual(Path(targets[str(valid)]), root / 'labels/val/b.txt')
            self.assertFalse((root / 'labels').exists())

    def test_colliding_image_stems_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images/train'
            images.mkdir(parents=True)
            config = root / 'data.yaml'
            config.write_text('train: images/train\n')
            paths = [images / 'same.png', images / 'same.jpg']
            for image in paths:
                image.touch()
            with self.assertRaises(ValueError):
                build_yolo_save_target_map(config, list(map(str, paths)))
            self.assertFalse((root / 'labels').exists())

    def test_plan_counts_actual_destinations_and_warns_for_unet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing, new = root / 'existing.txt', root / 'new.txt'
            existing.write_text('old')
            plan = build_label_save_plan([('a.png', existing), ('b.png', new)])
            self.assertEqual((plan.overwrite_count, plan.create_count), (1, 1))
            text = format_label_save_confirmation(plan, operation='Save All',
                                                  source_kind='unet', unloaded_count=3)
            self.assertIn('PNG', text)
            self.assertIn('3', text)
            self.assertFalse(new.exists())
            with self.assertRaises(ValueError):
                build_label_save_plan([('a.png', existing), ('b.png', existing)])

    @unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in GUI/runtime suite')
    def test_editor_and_viewer_confirm_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', NO_ALBUMENTATIONS_UPDATE='1',
                       XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory, MPLCONFIGDIR=directory)
            code = '''
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QCoreApplication, QEvent
from tabs.layout_dataset import Labeler
from tabs.viewing_dataset import ViewingDataset
app = QApplication([])
for cls, save_method, storage in [(Labeler, 'save_labels', 'all_annotations'),
                                  (ViewingDataset, 'save_current_annotations', 'annotations')]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image, original = root/'a.png', root/'a.txt'
        target = root/'labels/train/a.txt'
        cv2.imwrite(str(image), np.zeros((24,32,3), dtype=np.uint8))
        original.write_text('0 0.5 0.5 0.5 0.5\\n')
        window = cls()
        window._load_images([str(image)], label_map={str(image):str(original)})
        window._save_label_map = {str(image):str(target)}
        window._save_source_kind = 'yolo'
        before = original.read_bytes()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.No) as ask:
            getattr(window, save_method)()
            assert ask.call_args.args[-1] == QMessageBox.No
        assert not target.exists() and not target.parent.exists()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes), \
             patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'critical') as error:
            getattr(window, save_method)()
            error.assert_not_called()
        assert target.exists() and original.read_bytes() == before
        assert window._yaml_label_map[str(image)] == str(target)
        # Save All must not write the lazy, unvisited second image.
        other = root/'unvisited.png'
        window.image_paths.append(str(other))
        getattr(window, storage).append(None)
        window._save_label_map[str(other)] = str(root/'labels/train/unvisited.txt')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes), \
             patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'critical') as error:
            window.save_all_annotations()
            error.assert_not_called()
        assert not (root/'labels/train/unvisited.txt').exists()
        if window._lazy_cache is not None: window._lazy_cache.shutdown()
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
'''
            result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
