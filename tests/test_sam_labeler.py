"""Offscreen editor workflow; enable with NEURO_RUN_GUI_TESTS=1."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI checks are opt-in")
class SamLabelerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tabs.layout_dataset import Labeler
        from ui.image_view import numpy_to_qpixmap

        self.owner = owner = Labeler()
        self.image = np.zeros((60, 100, 3), np.uint8)
        self.item = {"image": self.image, "gray": np.zeros((60, 100), np.uint8)}

        class Cache:
            def get(inner, index):
                return self.item

            def is_loaded(inner, index):
                return True

            def shutdown(inner):
                pass

        owner._lazy_cache = Cache()
        owner.image_paths = ["one.png", "two.png"]
        owner.all_annotations = [[], []]
        owner.img_w, owner.img_h = 100, 60
        owner._displayed_frame = (owner._lazy_cache, 0)
        owner.image_view.set_pixmap(numpy_to_qpixmap(self.image))
        owner.image_view.set_annotations([], 100, 60)
        owner.auto_backend_combo.setCurrentIndex(1)
        self.controller = owner.sam_controller
        self.controller.frame_changed()
        self.panel, self.view = owner.sam_panel, owner.image_view

    def tearDown(self):
        self.owner.close()
        self.app.processEvents()

    def wait_until(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(condition(), "Background task did not settle")

    def fake_backend(self, predict=None):
        from utils_new.segmentation.backends.sam import SamResult
        masks = np.zeros((1, 60, 100), bool)
        masks[:, 10:45, 20:80] = True
        backend = SimpleNamespace(model_key="sam2.1-tiny", device="cpu",
                                  predict=predict or (lambda *args, **kwargs: SamResult(masks, np.array([0.95]))))
        self.controller.backend = backend
        self.controller.loaded_key = self.controller.settings_key()
        return backend

    def prompt(self):
        self.view.sam_points, self.view.sam_labels = [(30, 20)], [1]
        self.view.prompts_changed.emit()

    def test_provider_switch_keeps_yolo_controls_and_sam_capabilities(self):
        self.assertIs(self.owner.auto_stack.currentWidget(), self.panel)
        self.assertFalse(self.panel.text.isEnabled())
        self.panel.model.setCurrentIndex(self.panel.model.findData("sam3"))
        self.panel.mode.setCurrentIndex(1)
        self.assertTrue(self.panel.text.isEnabled())
        self.assertFalse(self.panel.positive.isEnabled())
        self.owner.auto_backend_combo.setCurrentIndex(0)
        self.assertIs(self.owner.auto_stack.currentWidget(), self.owner.auto_group)
        self.assertEqual(self.owner.auto_conf_spin.value(), 0.25)

    def test_browse_separately_downloaded_uppercase_sam3_detects_model_and_bpe(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "SAM3.pt"
            weights.write_bytes(b"checkpoint")
            bpe = root / "sam3" / "bpe_simple_vocab_16e6.txt.gz"
            bpe.parent.mkdir()
            bpe.write_bytes(b"vocabulary")
            with patch("ui.sam.controller.source_dialogs.get_open_file", return_value=(str(weights), "")):
                self.controller.browse_weights()
            self.assertEqual(self.panel.model.currentData(), "sam3")
            self.assertEqual(self.panel.tokenizer.text(), str(bpe))

    def test_grounding_dino_text_mode_loads_and_previews_without_automatic_annotation_changes(self):
        from utils_new.segmentation.backends.sam import SamResult
        from unittest.mock import Mock

        masks = np.zeros((1, 60, 100), bool)
        masks[:, 10:40, 20:70] = True
        backend = SimpleNamespace(model_key="sam2.1-tiny", device="cpu", supports_text=True,
                                  predict=Mock(return_value=SamResult(masks, np.array([.8]), np.array([.91]), ("screw",))))
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("grounded"))
        self.assertTrue(self.panel.text.isEnabled())
        self.assertFalse(self.panel.positive.isEnabled())
        self.assertFalse(self.panel.grounding_group.isHidden())
        self.panel.weights.setText("sam2.pt")
        self.panel.dino_directory.setText("local detector")
        with patch("utils_new.segmentation.backends.grounding_dino.GroundedSAM2.load", return_value=backend) as loader:
            self.controller.load()
            self.wait_until(lambda: self.controller.job is None)
        self.assertEqual(loader.call_args.args, ("sam2.1-tiny", "sam2.pt", "local detector"))
        self.assertEqual(loader.call_args.kwargs["detector_device"], "cpu")
        self.panel.text.setText("screw")
        self.panel.dino_box_score.setValue(.4)
        self.panel.dino_text_score.setValue(.3)
        self.panel.dino_max_objects.setValue(12)
        self.controller.run()
        self.wait_until(lambda: self.controller.job is None)
        arguments = backend.predict.call_args.kwargs
        self.assertEqual(arguments["box_threshold"], .4)
        self.assertEqual(arguments["text_threshold"], .3)
        self.assertEqual(arguments["max_objects"], 12)
        self.assertEqual(self.owner.current_annotations, [])
        self.assertEqual(len(self.view.sam_preview), 1)
        self.assertIn("IoU SAM2", self.panel.status.text())
        self.assertIn("screw: 0.910", self.panel.status.text())

    def test_grounding_mode_is_reset_for_sam1_and_sam3(self):
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("grounded"))
        self.assertEqual(self.panel.mode.currentData(), "grounded")
        self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
        self.assertEqual(self.panel.mode.currentData(), "points")
        self.assertTrue(self.panel.grounding_group.isHidden())
        self.panel.model.setCurrentIndex(self.panel.model.findData("sam3"))
        self.panel.mode.setCurrentIndex(self.panel.mode.findData("text"))
        self.assertTrue(self.panel.text.isEnabled())
        self.assertTrue(self.panel.grounding_group.isHidden())

    def test_preview_accept_updates_actual_editor_storage_and_can_save_yolo(self):
        from utils_new.import_export.yolo_labels import save_annotations_for_image, load_annotations

        self.fake_backend()
        self.prompt()
        self.panel.class_id.setValue(4)
        self.controller.run()
        self.wait_until(lambda: self.controller.job is None)
        self.assertEqual(self.owner.current_annotations, [])
        self.assertEqual(len(self.view.sam_preview), 1)
        self.assertTrue(self.panel.accept.isEnabled())
        self.controller.accept()
        self.assertEqual(len(self.owner.current_annotations), 1)
        self.assertEqual(self.owner.current_annotations[0][:2], ("segment", 4))
        self.assertEqual(self.owner.all_annotations[0], self.owner.current_annotations)
        self.assertEqual(self.view.sam_preview, [])
        # Exercise the same save path as the editor, then read the YOLO file back.
        with TemporaryDirectory() as directory:
            target = Path(directory) / "one.txt"
            image_path = Path(directory) / "one.png"
            from utils_new.import_export.images import write_image
            write_image(image_path, self.image)
            save_annotations_for_image(self.owner.current_annotations, target, image_path)
            self.assertTrue(target.read_text().startswith("4 "))
            self.assertEqual(load_annotations(target, 100, 60)[0][:2], ("segment", 4))

    def test_frame_change_discards_pending_result_and_clears_prompts(self):
        entered, release = Event(), Event()
        from utils_new.segmentation.backends.sam import SamResult

        def predict(*args, **kwargs):
            entered.set()
            release.wait(3)
            return SamResult(np.ones((1, 60, 100), bool), np.array([0.9]))

        self.fake_backend(predict)
        self.prompt()
        self.controller.run()
        try:
            self.wait_until(entered.is_set)
            self.owner.current_index = 1
            self.controller.frame_changed()
        finally:
            release.set()
        self.wait_until(lambda: self.controller.job is None)
        self.assertEqual(self.view.sam_preview, [])
        self.assertEqual(self.view.sam_points, [])
        self.assertEqual(self.owner.all_annotations, [[], []])
        self.assertFalse(self.panel.accept.isEnabled())

    def test_cancel_discards_result_without_disabling_navigation(self):
        entered, release = Event(), Event()
        from utils_new.segmentation.backends.sam import SamResult

        def predict(*args, **kwargs):
            entered.set()
            release.wait(3)
            return SamResult(np.ones((1, 60, 100), bool), np.array([0.9]))

        self.fake_backend(predict)
        self.prompt()
        self.controller.run()
        try:
            self.wait_until(entered.is_set)
            self.assertTrue(self.owner.nav_widget.isEnabled())
            self.assertFalse(self.owner.auto_group.isEnabled())
            self.controller.cancel()
        finally:
            release.set()
        self.wait_until(lambda: self.controller.job is None)
        self.assertEqual(self.view.sam_preview, [])
        self.assertTrue(self.owner.auto_group.isEnabled())

    def test_selected_annotation_supplies_pixel_box_and_dataset_class(self):
        self.owner.current_annotations = [("detect", 8, .5, .5, .4, .6)]
        self.view.selected_index = 0
        self.controller.selected_box()
        np.testing.assert_allclose(self.view.sam_box, [30, 12, 70, 48])
        self.assertEqual(self.panel.class_id.value(), 8)

    def test_prompt_coordinates_follow_zoom_and_do_not_create_annotations(self):
        from PyQt5.QtCore import QPointF, Qt
        from PyQt5.QtTest import QTest

        self.view.resize(600, 400)
        self.view.show()
        self.app.processEvents()
        self.view.scale(1.5, 1.5)
        self.controller.tool("positive")
        position = self.view.mapFromScene(QPointF(30, 20))
        expected = self.view.mapToScene(position)
        QTest.mouseClick(self.view.viewport(), Qt.LeftButton, pos=position)
        QTest.mouseClick(self.view.viewport(), Qt.RightButton, pos=position)
        self.assertEqual(self.view.sam_labels, [1, 0])
        np.testing.assert_allclose(self.view.sam_points[0], [expected.x(), expected.y()])
        self.assertEqual(self.owner.current_annotations, [])
        self.owner.on_tool_selected(self.owner.btn_tool_rect)
        self.assertIsNone(self.view.sam_tool)

    def test_load_is_background_and_mode_change_requires_reload(self):
        backend = SimpleNamespace(model_key="sam3", device="cpu")
        self.panel.model.setCurrentIndex(self.panel.model.findData("sam3"))
        self.panel.weights.setText("local sam3.pt")
        with patch("utils_new.segmentation.backends.sam.LocalSAM.load", return_value=backend) as loader:
            self.controller.load()
            self.wait_until(lambda: self.controller.job is None)
        self.assertIs(self.controller.backend, backend)
        self.assertEqual(loader.call_args.kwargs["prompt_mode"], "points")
        old_key = self.controller.loaded_key
        self.panel.mode.setCurrentIndex(1)
        self.assertNotEqual(old_key, self.controller.settings_key())
        self.controller.run()
        self.assertIsNone(self.controller.job)

    def test_image_navigation_does_not_cancel_model_loading(self):
        entered, release = Event(), Event()
        backend = SimpleNamespace(model_key="sam2.1-tiny", device="cpu")

        def load(*args, **kwargs):
            entered.set()
            release.wait(3)
            return backend

        with patch("utils_new.segmentation.backends.sam.LocalSAM.load", side_effect=load):
            self.controller.load()
            try:
                self.wait_until(entered.is_set)
                self.owner.current_index = 1
                self.controller.frame_changed()
                self.assertFalse(self.controller.job.cancelled)
            finally:
                release.set()
            self.wait_until(lambda: self.controller.job is None)
        self.assertIs(self.controller.backend, backend)


if __name__ == "__main__":
    unittest.main()
