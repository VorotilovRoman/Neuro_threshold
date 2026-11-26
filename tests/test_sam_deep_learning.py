"""Shared SAM workflow on the DL tab: instances, lifecycle, presets and export."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from utils_new.segmentation.backends.sam import SamResult


class SamResultProcessingTests(unittest.TestCase):
    def test_instance_morphology_preserves_input_masks_and_class_geometry(self):
        masks = np.zeros((2, 50, 80), bool)
        masks[0, 5:25, 10:40] = True
        masks[1, 15:35, 30:60] = True
        result = SamResult(masks, np.array([.9, .8]))
        original = masks.copy()
        processed = result.postprocessed([{'operation': 'dilate', 'kernel_px': 5}])
        self.assertEqual(processed.masks.shape, masks.shape)
        self.assertGreater(int(processed.masks.sum()), int(masks.sum()))
        self.assertEqual(len(processed.annotations(class_id=7)), 2)
        self.assertTrue(all(ann[1] == 7 for ann in processed.annotations(class_id=7)))
        np.testing.assert_array_equal(masks, original)
        np.testing.assert_array_equal(processed.combined_mask(), processed.masks.any(axis=0) * 255)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt tests')
class SamDeepLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tabs.deep_learning_methods import DeepLearningWindow
        self.owner = owner = DeepLearningWindow()
        self.directory = TemporaryDirectory()
        from utils_new.import_export.images import write_image
        owner._eager_records = []
        owner.image_paths = []
        for index in range(2):
            image = np.full((60, 100, 3), 40 + index, np.uint8)
            path = Path(self.directory.name) / f'{index}.png'
            write_image(path, image)
            owner.image_paths.append(str(path))
            owner._eager_records.append(dict(image=image, gray=image[:, :, 0]))
        owner.model_combo.setCurrentIndex(owner.model_combo.findText('SAM'))
        owner.update_timer.stop()
        owner.display_current_image()
        self.controller, self.panel = owner.sam_controller, owner.sam_panel

    def tearDown(self):
        self.owner.close()
        self.app.processEvents()
        self.directory.cleanup()

    def until(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(condition(), 'Background task did not settle')

    def result(self):
        masks = np.zeros((2, 60, 100), bool)
        masks[0, 10:35, 10:45] = True
        masks[1, 20:45, 30:65] = True
        return SamResult(masks, np.array([.91, .87]))

    def fake_backend(self, predict=None):
        backend = SimpleNamespace(model_key=self.panel.model.currentData(), device='cpu',
                                  predict=predict or Mock(return_value=self.result()))
        self.controller.backend = backend
        self.controller.loaded_key = self.controller.settings_key()
        self.owner.original_view.prompt_points = [(20, 20)]
        self.owner.original_view.prompt_labels = [1]
        self.owner.original_view.prompts_changed.emit()
        return backend

    def test_shared_download_button_populates_dl_paths_without_running_inference(self):
        from PyQt5.QtWidgets import QMessageBox
        from test_sam_weights import Opener, Response, WEIGHTS

        root = Path(self.directory.name) / 'segment_anything'
        with patch('ui.sam.downloads.DEFAULT_OUTPUT', root), \
                patch('ui.sam.downloads.QMessageBox.question', return_value=QMessageBox.Yes), \
                patch('utils_new.import_export.sam_weights.build_opener', return_value=Opener(Response(WEIGHTS))):
            self.panel.model.setCurrentIndex(self.panel.model.findData('sam-vit-b'))
            self.panel.download.click()
            self.assertFalse(self.owner.model_combo.isEnabled())
            self.until(lambda: self.controller.job is None)
            self.assertEqual(self.panel.weights.text(), str(root / 'sam/sam_vit_b_01ec64.pth'))
            self.assertTrue(self.owner.model_combo.isEnabled())
            self.assertTrue(self.panel.controls.isEnabled())
            self.assertIsNone(self.owner.prediction_mask)
            self.assertIsNone(self.controller.backend)

    def apply(self):
        self.fake_backend()
        self.panel.class_id.setValue(7)
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.controller.accept()

    def test_both_tabs_use_identical_widgets_controller_and_backend_factory(self):
        from tabs.layout_dataset import Labeler
        from ui.sam.panel import SamPanel
        from ui.sam.controller import SamController
        from ui.segmentation.prompt_view import PromptInputView
        editor = Labeler()
        try:
            for owner in (editor, self.owner):
                self.assertIsInstance(owner.sam_panel, SamPanel)
                self.assertIsInstance(owner.sam_controller, SamController)
            self.assertIsInstance(self.owner.original_view, PromptInputView)
            self.assertTrue(self.owner.model_combo.model().item(self.owner.model_combo.findText('SAM')).isEnabled())
            self.assertTrue(self.owner.segment_button.isHidden())
        finally:
            editor.close()

    def test_preview_accept_mask_and_overlapping_instances(self):
        self.fake_backend()
        self.panel.class_id.setValue(7)
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.assertIsNone(self.owner.prediction_mask)
        self.assertEqual(len(self.owner.original_view.prompt_preview), 2)
        self.controller.accept()
        np.testing.assert_array_equal(self.owner.prediction_mask, self.result().combined_mask())
        self.assertEqual(len(self.owner.current_objects_full), 2)
        self.assertTrue(all(ann[1] == 7 for ann in self.owner.current_objects_full))
        self.assertIsNotNone(self.owner.dl_view._current_pixmap)
        self.assertIsNotNone(self.owner.morph_view._current_pixmap)
        self.assertTrue(self.owner.save_button.isEnabled())

    def test_merge_accepted_sam_instances_preserves_mask_and_skips_inference(self):
        backend = self.fake_backend()
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.controller.accept()
        before = self.owner.prediction_mask.copy()
        self.panel.merge.setChecked(True)
        self.assertEqual(len(self.owner.current_objects_full), 1)
        np.testing.assert_array_equal(self.owner.prediction_mask, before)
        self.panel.merge.setChecked(False)
        self.assertEqual(len(self.owner.current_objects_full), 2)
        backend.predict.assert_called_once()

    def test_empty_result_can_be_applied_and_inverted_as_a_binary_mask(self):
        self.fake_backend(Mock(return_value=SamResult(np.empty((0, 60, 100), bool), np.empty(0))))
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.assertTrue(self.panel.accept.isEnabled())
        self.controller.accept()
        self.assertFalse(self.owner.final_mask.any())
        self.assertEqual(self.owner.current_objects_full, [])
        self.owner.invert_checkbox.setChecked(True)
        self.owner.display_current_image()
        self.assertTrue((self.owner.final_mask == 255).all())

    def test_morphology_inversion_and_annotation_mode_without_reinference(self):
        backend = self.fake_backend()
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.controller.accept()
        state = self.owner.morphology_widget.get_settings()
        for operation in state['operations']:
            if operation['operation'] == 'dilate':
                operation['kernel_px'] = 5
        self.owner.morphology_widget.set_settings(state)
        self.owner.display_current_image()
        self.assertEqual(len(self.owner.current_objects_full), 2)
        normal = self.owner.final_mask.copy()
        self.owner.invert_checkbox.setChecked(True)
        self.owner.display_current_image()
        np.testing.assert_array_equal(self.owner.final_mask, 255 - normal)
        self.owner.draw_combo.setCurrentIndex(self.owner.draw_combo.findData('detect'))
        self.assertTrue(all(ann[0] == 'detect' for ann in self.owner.current_objects_full))
        self.assertEqual(backend.predict.call_count, 1)

    def test_frame_change_clears_prompt_and_result_without_automatic_prediction(self):
        backend = self.fake_backend()
        self.owner.run_segmentation()
        self.until(lambda: self.controller.job is None)
        self.controller.accept()
        self.owner.next_image()
        self.owner.update_timer.stop()
        self.assertIsNone(self.owner.prediction_mask)
        self.assertEqual(self.owner.current_objects_full, [])
        self.assertEqual(self.owner.original_view.prompt_points, [])
        self.assertEqual(backend.predict.call_count, 1)

    def test_background_stale_frame_cancel_and_close(self):
        entered, release = Event(), Event()
        def predict(*args, **kwargs):
            entered.set(); release.wait(3)
            return self.result()
        self.fake_backend(predict)
        self.owner.run_segmentation()
        try:
            self.until(entered.is_set)
            self.assertTrue(self.owner.nav_widget.isEnabled())
            self.owner.next_image()
            self.owner.update_timer.stop()
        finally:
            release.set()
        self.until(lambda: self.controller.job is None)
        self.assertIsNone(self.controller.preview_result)
        self.assertIsNone(self.owner.prediction_mask)
        self.owner.close()
        self.assertTrue(self.controller.closed)

    def test_load_all_families_uses_the_same_shared_factory(self):
        for key in ('sam-vit-b', 'sam2.1-tiny', 'sam3'):
            self.panel.model.setCurrentIndex(self.panel.model.findData(key))
            self.panel.weights.setText(key + '.pt')
            backend = SimpleNamespace(model_key=key, device='cpu')
            with patch('ui.sam.controller.load_sam_backend', return_value=backend) as load:
                self.owner.load_model_from_widget('SAM')
                self.until(lambda: self.controller.job is None)
            self.assertIs(self.controller.backend, backend)
            self.assertEqual(load.call_args.args, (key, key + '.pt'))

    def test_preset_roundtrip_restores_sam_settings_but_never_loads_weights_implicitly(self):
        self.panel.model.setCurrentIndex(self.panel.model.findData('sam3'))
        self.panel.mode.setCurrentIndex(self.panel.mode.findData('text'))
        self.panel.weights.setText('local sam3.pt')
        self.panel.text.setText('metal screw')
        manager = self.owner.preset_manager
        state = manager.get_current_settings()
        manager.presets['test-sam'] = state
        self.panel.text.setText('changed')
        with patch('ui.sam.controller.load_sam_backend') as factory:
            self.assertTrue(manager.apply_preset('test-sam'))
        factory.assert_not_called()
        self.assertEqual(self.panel.text.text(), 'metal screw')
        self.assertEqual(self.panel.model.currentData(), 'sam3')
        self.assertEqual(self.panel.mode.currentData(), 'text')

    def test_sam_annotations_export_through_existing_dl_save_flow(self):
        from PyQt5.QtWidgets import QMessageBox
        from utils_new.import_export.yolo_labels import load_annotations
        self.apply()
        self.owner.current_selected_indices = list(range(len(self.owner.current_objects_full)))
        with patch('ui.segmentation.export.QMessageBox.question', return_value=QMessageBox.Yes), \
                patch('ui.segmentation.export.QMessageBox.information'):
            self.owner.save_current_annotations()
        target = Path(self.owner.image_paths[0]).with_suffix('.txt')
        saved = load_annotations(target, 100, 60)
        self.assertEqual(len(saved), 2)
        self.assertTrue(all(ann[:2] == ('segment', 7) for ann in saved))


if __name__ == '__main__':
    unittest.main()
