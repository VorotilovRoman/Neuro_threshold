"""Shared prompt gestures, full-frame initialization and enclosing object output."""

import os
import time
import unittest

import numpy as np

from utils_new.annotations.adapters import from_legacy_many
from utils_new.annotations.model import ImageSize
from utils_new.segmentation.backends.sam import SamResult
from utils_new.segmentation.postprocessing import extract_objects


class MergedOutputTests(unittest.TestCase):
    def test_tiny_fragments_are_grouped_before_area_filtering(self):
        mask = np.zeros((128, 256), np.uint8)
        mask[10:12, 20:22] = 255
        mask[30:32, 60:62] = 255
        image = np.zeros((128, 256, 3), np.uint8)
        self.assertEqual(extract_objects(image, mask, 'segment')[1], [])
        self.assertEqual(len(extract_objects(image, mask, 'segment', merge_all=True)[1]), 1)
        result = SamResult.from_binary_mask(mask)
        self.assertEqual(result.annotations(min_area=8), [])
        self.assertEqual(len(result.annotations(min_area=8, merge_all=True)), 1)
        self.assertEqual(result.annotations(min_area=9, merge_all=True), [])
        mask[50, 100:110] = 255  # A zero-area contour cannot become a polygon.
        self.assertEqual(len(extract_objects(image, mask, 'segment', merge_all=True)[1]), 1)

    def test_disconnected_components_enclose_one_object_without_changing_mask(self):
        masks = np.zeros((2, 60, 100), bool)
        masks[0, 5:20, 10:25] = True
        masks[1, 30:50, 60:85] = True
        result = SamResult(masks, np.array([.9, .8]))
        binary = result.combined_mask()
        before = binary.copy()
        image = np.zeros((60, 100, 3), np.uint8)
        for kind in ('detect', 'obb', 'segment'):
            with self.subTest(kind=kind):
                self.assertEqual(len(result.annotations(kind=kind)), 2)
                sam = result.annotations(kind=kind, class_id=7, merge_all=True)
                classical = extract_objects(image, binary, kind, merge_all=True)[1]
                self.assertEqual(len(sam), 1)
                self.assertEqual(len(classical), 1)
                ann = from_legacy_many(sam, ImageSize(100, 60))[0]
                self.assertEqual(ann.class_id, 7)
                self.assertEqual(ann.kind, kind)
                self.assertLessEqual(ann.points_px[:, 0].min(), 10.01)
                self.assertGreaterEqual(ann.points_px[:, 0].max(), 83.99)
        np.testing.assert_array_equal(binary, before)
        np.testing.assert_array_equal(result.combined_mask(), before)
        empty = SamResult(np.zeros((0, 60, 100), bool), np.zeros(0))
        self.assertEqual(empty.annotations(merge_all=True), [])


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in Qt tests')
class InteractivePromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from tabs.interactive_methods import InteractiveMethodsWindow
        self.owner = owner = InteractiveMethodsWindow()
        image = np.full((60, 100, 3), 220, np.uint8)
        image[15:45, 30:70] = 25
        owner.image_paths = ['one.png', 'two.png']
        owner._eager_records = [dict(image=image, gray=image[:, :, 0]),
                                dict(image=image.copy(), gray=image[:, :, 0].copy())]
        owner.display_current_image()
        self.view = owner.image_view
        self.view.resize(500, 400)
        self.view.show()
        self.app.processEvents()

    def tearDown(self):
        self.owner.close()
        self.app.processEvents()

    def point(self, x, y):
        from PyQt5.QtCore import QPointF
        return self.view.mapFromScene(QPointF(x, y))

    def drag(self, x1, y1, x2, y2, button=None):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        button = button or Qt.LeftButton
        QTest.mousePress(self.view.viewport(), button, pos=self.point(x1, y1))
        QTest.mouseMove(self.view.viewport(), self.point(x2, y2))
        QTest.mouseRelease(self.view.viewport(), button, pos=self.point(x2, y2))

    def until(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(condition())

    def test_shared_view_strokes_are_continuous_and_opposite_marks_replace_foreground(self):
        from PyQt5.QtCore import Qt
        from ui.segmentation.prompt_view import PromptInputView
        self.assertIsInstance(self.view, PromptInputView)
        self.owner.activate_fg_tool()
        self.drag(30, 25, 65, 25)
        self.assertTrue(np.all(self.owner.fg_scribbles[25, 30:66]))
        self.drag(45, 20, 45, 30, Qt.RightButton)
        self.assertTrue(self.owner.bg_scribbles[25, 45])
        self.assertFalse(self.owner.fg_scribbles[25, 45])
        self.assertFalse(np.any((self.owner.fg_scribbles > 0) & (self.owner.bg_scribbles > 0)))
        self.assertTrue(any(label == 0 for label in self.view.prompt_labels))
        self.owner.reset_mask()
        self.assertEqual(self.view.prompt_strokes, [])
        self.assertEqual(self.owner.prompt_tools.region.currentData(), 100)

    def test_full_frame_can_run_grabcut_without_drawing_a_box(self):
        self.assertIsNone(self.owner.rect)
        self.assertEqual(self.owner.effective_roi(), (0, 0, 100, 60))
        self.assertTrue(self.owner.run_seg_btn.isEnabled())
        self.owner.run_segmentation()
        self.until(lambda: not self.owner.jobs.busy)
        self.assertIsNotNone(self.owner.mask)
        self.assertEqual(self.owner.mask[30, 50], 255)
        self.assertEqual(self.owner.mask[0, 0], 0)

    def test_center_percentages_scale_with_frame_and_preserve_seed_marks(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        self.owner.activate_fg_tool()
        QTest.mouseClick(self.view.viewport(), Qt.LeftButton, pos=self.point(40, 30))
        original_seeds = self.owner.fg_scribbles.copy()
        region = self.owner.prompt_tools.region
        for percent in (50, 60, 70, 80, 100, 90):
            region.setCurrentIndex(region.findData(percent))
            box = self.view.effective_box()
            np.testing.assert_allclose(box, ((100 - percent) / 2, (100 - percent) * .3,
                                            (100 + percent) / 2, (100 + percent) * .3))
            self.assertIsNone(self.view.prompt_tool)
            np.testing.assert_array_equal(self.owner.fg_scribbles, original_seeds)
        self.view.set_image_data(200, 80)
        self.view.clear_prompts()
        self.assertEqual(region.currentData(), 90)
        np.testing.assert_allclose(self.view.effective_box(), (10, 4, 190, 76))
        self.view.set_prompt_box((20, 10, 80, 60))
        self.assertIsNone(region.currentData())
        self.view.clear_prompts()
        self.assertEqual(region.currentData(), 100)

    def test_box_gesture_follows_zoom_and_escape_keeps_roi(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        self.view.scale(1.8, 1.8)
        self.owner.activate_rect_tool()
        self.drag(20, 10, 80, 50)
        np.testing.assert_allclose(self.owner.effective_roi(), (20, 10, 80, 50), atol=1)
        self.assertIsNone(self.owner.prompt_tools.region.currentData())
        self.owner.activate_fg_tool()
        QTest.keyClick(self.view, Qt.Key_Escape)
        self.assertIsNone(self.view.prompt_tool)
        self.assertTrue(self.owner.pan_zoom_btn.isChecked())
        self.owner.prompt_tools.region.setCurrentIndex(self.owner.prompt_tools.region.findData(100))
        self.assertEqual(self.owner.effective_roi(), (0, 0, 100, 60))

    def test_result_merge_is_geometry_only_and_active_object_supplies_roi(self):
        from ui.segmentation import results
        mask = np.zeros((60, 100), np.uint8)
        mask[10:25, 10:25] = 255
        mask[30:50, 60:85] = 255
        self.owner.mask = mask
        self.owner.draw_combo.setCurrentIndex(self.owner.draw_combo.findData('detect'))
        self.owner.display_current_image()
        self.assertEqual(len(self.owner.current_objects_full), 2)
        self.owner.merge_objects_checkbox.setChecked(True)
        self.assertEqual(len(self.owner.current_objects_full), 1)
        np.testing.assert_array_equal(self.owner.final_mask, mask)
        results.focus_row(self.owner, 0)
        self.owner.prompt_tools.selected_box.click()
        np.testing.assert_allclose(self.owner.effective_roi(), (10, 10, 85, 50), atol=1)
        self.assertIsNone(self.owner.prompt_tools.region.currentData())
        state = self.owner.preset_manager.get_current_settings()
        self.assertTrue(state['merge_objects'])

