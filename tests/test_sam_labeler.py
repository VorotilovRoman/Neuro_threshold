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
        self.view.prompt_points, self.view.prompt_labels = [(30, 20)], [1]
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

    def test_selecting_missing_architecture_offers_download_but_presets_do_not(self):
        from PyQt5.QtWidgets import QMessageBox

        with TemporaryDirectory() as directory, \
                patch("ui.sam.downloads.DEFAULT_OUTPUT", Path(directory)), \
                patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.No) as ask, \
                patch("utils_new.import_export.sam_weights.build_opener") as opener:
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-l"))
            self.assertFalse(ask.called)
            settings = self.panel.get_settings()
            settings.update(model="SAM ViT-H", weights="not-downloaded.pth")
            self.panel.set_settings(settings)
            self.assertFalse(ask.called)
            self.panel.model.activated.emit(self.panel.model.currentIndex())
            ask.assert_called_once()
            self.assertIn("sam_vit_h_4b8939.pth", ask.call_args.args[2])
            self.assertFalse(opener.called)
            self.assertIsNone(self.controller.job)

    def test_selecting_architecture_finds_local_weights_instead_of_using_previous_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "segment_anything"
            output.mkdir()
            previous = root / "sam_vit_b_01ec64.pth"
            selected = output / "sam_vit_l_0b3195.pth"
            previous.write_bytes(b"old")
            selected.write_bytes(b"new")
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
            self.panel.weights.setText(str(previous))
            with patch("ui.sam.downloads.DEFAULT_OUTPUT", output), \
                    patch("ui.sam.downloads.QMessageBox.question") as ask, \
                    patch("utils_new.import_export.sam_weights.build_opener") as opener:
                self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-l"))
                self.panel.model.activated.emit(self.panel.model.currentIndex())
                self.assertEqual(self.panel.weights.text(), str(selected))
                self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
                self.panel.model.activated.emit(self.panel.model.currentIndex())
                self.assertEqual(self.panel.weights.text(), str(previous))
                self.assertFalse(ask.called)
                self.assertFalse(opener.called)

    def test_download_runs_in_background_sets_paths_and_does_not_load_backend(self):
        from PyQt5.QtWidgets import QMessageBox
        from test_sam_weights import Opener, Response, WEIGHTS

        with TemporaryDirectory() as directory, \
                patch("ui.sam.downloads.DEFAULT_OUTPUT", Path(directory)), \
                patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.Yes), \
                patch("utils_new.import_export.sam_weights.build_opener", return_value=Opener(Response(WEIGHTS))), \
                patch("ui.sam.controller.load_sam_backend") as loader:
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
            self.panel.model.activated.emit(self.panel.model.currentIndex())
            self.assertIsNotNone(self.controller.job)
            self.assertFalse(self.panel.controls.isEnabled())
            self.assertTrue(self.panel.cancel.isEnabled())
            self.wait_until(lambda: self.controller.job is None)
            target = Path(directory) / "sam/sam_vit_b_01ec64.pth"
            self.assertEqual(target.read_bytes(), WEIGHTS)
            self.assertEqual(self.panel.weights.text(), str(target))
            self.assertTrue(self.panel.controls.isEnabled())
            self.assertTrue(self.panel.progress.isHidden())
            self.assertIsNone(self.controller.backend)
            self.assertFalse(loader.called)
            self.assertIn("Файлы готовы", self.panel.status.text())

    def test_custom_weights_from_preset_stay_associated_with_their_architecture(self):
        from PyQt5.QtWidgets import QMessageBox

        with TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / "custom-name.pth"
            custom.write_bytes(b"checkpoint")
            settings = self.panel.get_settings()
            settings.update(model="SAM ViT-H", weights=str(custom))
            self.panel.set_settings(settings)
            with patch("ui.sam.downloads.DEFAULT_OUTPUT", root / "segment_anything"), \
                    patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.No) as ask:
                self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
                self.panel.model.activated.emit(self.panel.model.currentIndex())
                self.assertEqual(self.panel.weights.text(), "")
                self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-h"))
                self.panel.model.activated.emit(self.panel.model.currentIndex())
                self.assertEqual(self.panel.weights.text(), str(custom))
                ask.assert_called_once()

    def test_download_failure_restores_controls_and_sam3_token_is_not_saved(self):
        from PyQt5.QtWidgets import QLineEdit, QMessageBox
        from test_sam_weights import Opener
        from urllib.error import HTTPError
        import io

        with TemporaryDirectory() as directory, \
                patch("ui.sam.downloads.DEFAULT_OUTPUT", Path(directory)), \
                patch.dict(os.environ, {"HF_TOKEN": ""}), \
                patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.Yes), \
                patch("ui.sam.downloads.QInputDialog.getText", return_value=("hf_SECRET", True)) as ask_token, \
                patch("ui.sam.controller.QMessageBox.warning") as warning, \
                patch("utils_new.import_export.sam_weights.build_opener", return_value=Opener(
                    HTTPError("https://cdn.example/SIGNED_SECRET", 403, "hf_SECRET", {}, io.BytesIO()))):
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam3"))
            self.panel.download.click()
            self.wait_until(lambda: self.controller.job is None)
            self.assertEqual(ask_token.call_args.args[-1], QLineEdit.Password)
            warning.assert_called_once()
            self.assertIn("HTTP 403", self.panel.status.text())
            self.assertNotIn("SECRET", self.panel.status.text())
            self.assertTrue(self.panel.controls.isEnabled())
            self.assertTrue(self.panel.cancel.isHidden())
            self.assertFalse(list(Path(directory).rglob("*.part")))
            self.assertFalse((Path(directory) / "download_manifest.json").exists())

    def test_download_cancel_keeps_controls_alive_and_discards_partial_file(self):
        from PyQt5.QtWidgets import QMessageBox
        from test_sam_weights import Opener, Response, WEIGHTS

        entered, release = Event(), Event()

        class SlowResponse(Response):
            def read(self, size):
                entered.set()
                release.wait(3)
                return super().read(size)

        with TemporaryDirectory() as directory, \
                patch("ui.sam.downloads.DEFAULT_OUTPUT", Path(directory)), \
                patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.Yes), \
                patch("utils_new.import_export.sam_weights.build_opener", return_value=Opener(SlowResponse(WEIGHTS))):
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam-vit-b"))
            self.panel.download.click()
            self.wait_until(entered.is_set)
            self.panel.cancel.click()
            release.set()
            self.wait_until(lambda: self.controller.job is None)
            self.assertFalse(list(Path(directory).rglob("*.part")))
            self.assertFalse((Path(directory) / "sam/sam_vit_b_01ec64.pth").exists())
            self.assertTrue(self.panel.controls.isEnabled())
            self.assertIn("отменена", self.panel.status.text())

    def test_sam3_with_existing_weights_downloads_tokenizer_without_asking_for_token(self):
        from PyQt5.QtWidgets import QMessageBox
        from test_sam_weights import Opener, Response, TOKENIZER, WEIGHTS

        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "SAM3.pt"
            weights.write_bytes(WEIGHTS)
            self.panel.model.setCurrentIndex(self.panel.model.findData("sam3"))
            self.panel.weights.setText(str(weights))
            opener = Opener(Response(TOKENIZER))
            with patch("ui.sam.downloads.DEFAULT_OUTPUT", root / "segment_anything"), \
                    patch("ui.sam.downloads.QMessageBox.question", return_value=QMessageBox.Yes), \
                    patch("ui.sam.downloads.QInputDialog.getText") as ask_token, \
                    patch("utils_new.import_export.sam_weights.build_opener", return_value=opener):
                self.panel.download.click()
                self.wait_until(lambda: self.controller.job is None)
                self.assertFalse(ask_token.called)
            self.assertEqual(len(opener.requests), 1)
            self.assertEqual(self.panel.weights.text(), str(weights))
            self.assertEqual(self.panel.tokenizer.text(), str(root / "bpe_simple_vocab_16e6.txt.gz"))

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
        self.assertEqual(len(self.view.prompt_preview), 1)
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
        self.assertEqual(len(self.view.prompt_preview), 1)
        self.assertTrue(self.panel.accept.isEnabled())
        self.controller.accept()
        self.assertEqual(len(self.owner.current_annotations), 1)
        self.assertEqual(self.owner.current_annotations[0][:2], ("segment", 4))
        self.assertEqual(self.owner.all_annotations[0], self.owner.current_annotations)
        self.assertEqual(self.view.prompt_preview, [])
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
        self.assertEqual(self.view.prompt_preview, [])
        self.assertEqual(self.view.prompt_points, [])
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
        self.assertEqual(self.view.prompt_preview, [])
        self.assertTrue(self.owner.auto_group.isEnabled())

    def test_selected_annotation_supplies_pixel_box_and_dataset_class(self):
        self.owner.current_annotations = [("detect", 8, .5, .5, .4, .6)]
        self.view.selected_index = 0
        self.controller.selected_box()
        np.testing.assert_allclose(self.view.prompt_box, [30, 12, 70, 48])
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
        self.assertEqual(self.view.prompt_labels, [0])  # Latest opposite mark replaces the old point.
        np.testing.assert_allclose(self.view.prompt_points[0], [expected.x(), expected.y()])
        self.assertEqual(self.owner.current_annotations, [])
        self.owner.on_tool_selected(self.owner.btn_tool_rect)
        self.assertIsNone(self.view.prompt_tool)

    def test_default_full_frame_and_merge_preview_do_not_repeat_inference(self):
        from unittest.mock import Mock
        from utils_new.segmentation.backends.sam import SamResult

        masks = np.zeros((2, 60, 100), bool)
        masks[0, 5:20, 10:25] = True
        masks[1, 30:50, 60:85] = True
        predict = Mock(return_value=SamResult(masks, np.array([.9, .8])))
        self.fake_backend(predict)
        self.controller.run()
        self.wait_until(lambda: self.controller.job is None)
        self.assertEqual(predict.call_args.args[1].box, (0, 0, 100, 60))
        self.assertEqual(len(self.view.prompt_preview), 2)
        self.panel.merge.setChecked(True)
        self.assertEqual(len(self.view.prompt_preview), 1)
        self.panel.merge.setChecked(False)
        self.assertEqual(len(self.view.prompt_preview), 2)
        self.panel.merge.setChecked(True)
        self.controller.accept()
        self.assertEqual(len(self.owner.current_annotations), 1)
        predict.assert_called_once()

    def test_loaded_button_tracks_weights_device_mode_and_failed_reload(self):
        backend = SimpleNamespace(model_key='sam2.1-tiny', device='cpu')
        self.panel.weights.setText('local.pt')
        self.assertFalse(self.panel.load.property('modelLoaded'))
        with patch('ui.sam.controller.load_sam_backend', return_value=backend):
            self.controller.load()
            self.wait_until(lambda: self.controller.job is None)
        self.assertTrue(self.panel.load.property('modelLoaded'))
        self.assertIn('#2e7d32', self.panel.load.styleSheet())
        region = self.panel.prompt_tools.region
        region.setCurrentIndex(region.findData(60))
        self.panel.score.setValue(.6)
        self.panel.text.setText('changed prompt')
        self.assertTrue(self.panel.load.property('modelLoaded'))
        self.panel.weights.setText('different.pt')
        self.assertFalse(self.panel.load.property('modelLoaded'))
        self.panel.weights.setText('local.pt')
        self.assertTrue(self.panel.load.property('modelLoaded'))
        self.panel.device.setCurrentText('cpu')
        self.assertFalse(self.panel.load.property('modelLoaded'))
        self.panel.device.setCurrentText('auto')
        self.assertTrue(self.panel.load.property('modelLoaded'))
        self.panel.mode.setCurrentIndex(self.panel.mode.findData('grounded'))
        self.assertFalse(self.panel.load.property('modelLoaded'))
        with patch('ui.sam.controller.load_sam_backend', side_effect=ValueError('test failure')), \
                patch('ui.sam.controller.QMessageBox.warning'):
            self.controller.load()
            self.wait_until(lambda: self.controller.job is None)
        self.assertFalse(self.panel.load.property('modelLoaded'))
        self.assertEqual(self.panel.load.styleSheet(), '')

    def test_selected_center_region_is_passed_to_sam(self):
        from unittest.mock import Mock
        from utils_new.segmentation.backends.sam import SamResult
        predict = Mock(return_value=SamResult(np.zeros((0, 60, 100), bool), np.zeros(0)))
        self.fake_backend(predict)
        region = self.panel.prompt_tools.region
        region.setCurrentIndex(region.findData(70))
        self.controller.run()
        self.wait_until(lambda: self.controller.job is None)
        np.testing.assert_allclose(predict.call_args.args[1].box, (15, 9, 85, 51))

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
