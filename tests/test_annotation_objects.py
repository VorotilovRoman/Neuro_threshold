"""Inspector geometry, stable identity through sorting/filtering, and real UI clicks."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_new.annotations import ImageSize, from_legacy
from utils_new.annotations.object_info import ObjectInfo
from utils_new.dataset_yolo_labels_new import serialize_yolo_labels, write_yolo_labels

ROOT = Path(__file__).resolve().parents[1]


class ObjectInfoTests(unittest.TestCase):
    def test_source_and_working_geometry_have_explicit_scales(self):
        info = ObjectInfo.from_tuple(7, ("detect", 2, 0.5, 0.4, 0.2, 0.3))
        source, working = info.geometry((2000, 1000)), info.geometry((640, 320))
        np.testing.assert_allclose(source.bounds, [800, 250, 1200, 550])
        np.testing.assert_allclose(source.center, [1000, 400])
        np.testing.assert_allclose(
            (source.width, source.height, source.area), (400, 300, 120000)
        )
        np.testing.assert_allclose(working.points, source.points * 0.32)
        self.assertAlmostEqual(info.area_fraction, 0.06)
        self.assertEqual(info.index, 7)
        self.assertEqual(info.normalized.class_id, 2)
        obb = ObjectInfo.from_tuple(
            0, ("obb", 0, [0.2, 0.1, 0.4, 0.2, 0.3, 0.4, 0.1, 0.3])
        )
        polygon = ObjectInfo.from_tuple(
            1, ("segment", 3, [0.1, 0.1, 0.8, 0.1, 0.1, 0.7])
        )
        self.assertAlmostEqual(obb.geometry((100, 200)).area, 1000)
        self.assertAlmostEqual(polygon.area_fraction, 0.21)

    def test_clipboard_serializer_is_exact_writer_including_four_point_segment(self):
        size = ImageSize(2000, 1000)
        annotations = [
            from_legacy(value, size)
            for value in [
                ("detect", 2, 0.5, 0.4, 0.2, 0.3),
                ("obb", 0, [0.2, 0.1, 0.4, 0.2, 0.3, 0.4, 0.1, 0.3]),
                ("segment", 3, [0.1, 0.1, 0.8, 0.1, 0.8, 0.7, 0.1, 0.7]),
            ]
        ]
        text = serialize_yolo_labels(annotations, size)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            write_yolo_labels(path, annotations, size)
            self.assertEqual(text, path.read_text())
        self.assertEqual(len(text.splitlines()[2].split()), 11)
        self.assertEqual(serialize_yolo_labels([], size), "")


@unittest.skipUnless(
    os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "GUI tests are opt-in"
)
class ObjectInspectorGuiTests(unittest.TestCase):
    def run_gui(self, code):
        with tempfile.TemporaryDirectory() as config:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CONFIG_HOME=config,
                XDG_CACHE_HOME=config,
                MPLCONFIGDIR=config,
                YOLO_CONFIG_DIR=config,
                NO_ALBUMENTATIONS_UPDATE="1",
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_sort_filter_focus_bulk_checks_and_clipboard(self):
        self.run_gui("""
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy,QTest
from ui.annotation_objects import AnnotationObjectsWidget
app=QApplication([]);w=AnnotationObjectsWidget();w.resize(520,780);w.show()
values=[('detect',0,.2,.3,.1,.2),('detect',1,.6,.5,.4,.6),('segment',0,[.1,.1,.2,.1,.2,.3,.1,.3])]
w.set_objects(values,(2000,1000),(640,320));app.processEvents()
signals=QSignalSpy(w.checked_changed)
w.table.sortByColumn(5,Qt.DescendingOrder);app.processEvents()
assert w.proxy.index(0,0).data(Qt.UserRole)==1
QTest.mouseClick(w.table.viewport(),Qt.LeftButton,pos=w.table.visualRect(w.proxy.index(0,2)).center())
assert w.focused_index==1
w.model.setData(w.model.index(0,0),Qt.Unchecked,Qt.CheckStateRole)
assert w.checked_indices==[1,2] and w.focused_index==1
w.search.setText('Segment');assert w.proxy.rowCount()==1 and w.focused_index==1
assert w.checked_indices==[1,2]
w.focus_object(1,reveal=True);assert not w.search.text() and w.focused_index==1
w.details_toggle.click();assert w.details.isVisible() and w.vertices.rowCount()==4
w.units.setCurrentIndex(2);assert w.vertices.index(0,1).data()=='0.400000'
w.units.setCurrentIndex(0);assert w.vertices.index(0,1).data()=='800.00'
w.copy_focused.click();assert QApplication.clipboard().text()==w.yolo_text([1])
w.copy_checked.click();assert QApplication.clipboard().text()==w.yolo_text([1,2])
w.search.setText('BBox');before=len(signals);w.none_button.click()
assert w.checked_indices==[] and len(signals)==before+1
assert not w.copy_checked.isEnabled() and not w.zoom_button.isEnabled()
w.invert_button.click();assert w.checked_indices==[0,1,2]
w.clear();assert w.model.rowCount()==0 and w.vertices.rowCount()==0
assert not w.copy_focused.isEnabled() and not w.copy_checked.isEnabled()
w.close()
""")

    def test_all_five_tabs_map_sorted_rows_and_scene_to_full_objects(self):
        self.run_gui("""
from pathlib import Path
import tempfile,cv2,numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.segmentation import results
from utils_new.annotations import ImageSize,from_legacy
from utils_new.dataset_yolo_labels_new import serialize_yolo_labels
from main import ImagePlayer
app=QApplication([]);w=ImagePlayer([(k,k) for k in ['threshold','gradient','interactive','traditional_ml','deep_learning']]);w.show()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'large.png';cv2.imwrite(str(path),np.zeros((1000,2000,3),np.uint8))
    for owner in w._tab_widgets:
        owner._load_images([str(path)]);owner.update_timer.stop()
        assert not hasattr(owner,'coord_text') and not hasattr(owner,'object_list')
        owner.current_base_image=owner.original_image.copy()
        values=[('detect',0,.2,.3,.1,.2),('detect',1,.6,.5,.4,.6),('segment',0,[.1,.1,.2,.1,.2,.3,.1,.3])]
        owner.current_objects_full=values;owner.update_object_list()
        panel=owner.objects_widget
        assert panel.source_size==(2000,1000)
        panel.table.sortByColumn(5,Qt.DescendingOrder);app.processEvents()
        panel.set_checked_indices([1,2]);panel.focus_object(2)
        assert owner._focused_object_index==2 and owner.annotated_view.selected_index==1
        assert owner.annotated_view._annotation_label(1,0)=='№2 · класс 1'
        assert owner.annotated_view._annotation_label(0,1)=='№3 · класс 0'
        panel.search.setText('Segment');results.focus_annotation(owner,0)
        assert panel.focused_index==1 and owner.annotated_view.selected_index==0 and not panel.search.text()
        assert owner.current_selected_indices==[1,2]
        panel.copy_checked.click()
        expected=serialize_yolo_labels([from_legacy(values[i],ImageSize(2000,1000)) for i in [1,2]],ImageSize(2000,1000))
        assert QApplication.clipboard().text()==expected
        panel.zoom_button.click();assert owner.annotated_view.transform().m11()>0
        owner.clear_images();assert panel.model.rowCount()==0 and panel.vertices.rowCount()==0
w.close();app.processEvents()
""")

    def test_large_contour_uses_virtual_vertex_table_and_keeps_geometry_read_only(self):
        self.run_gui("""
import numpy as np
from PyQt5.QtWidgets import QApplication
from ui.annotation_objects import AnnotationObjectsWidget
app=QApplication([]);w=AnnotationObjectsWidget()
a=np.linspace(0,2*np.pi,10000,endpoint=False)
points=np.column_stack((.5+.3*np.cos(a),.5+.3*np.sin(a))).ravel().tolist();original=list(points)
w.set_objects([('segment',0,points)],(2048,1024),(640,320));w.focus_object(0)
assert w.vertices.rowCount()==10000
w.units.setCurrentIndex(2);w.units.setCurrentIndex(1);w.set_checked_indices([])
assert points==original and w.focused_index==0 and w.vertices.rowCount()==10000
assert not w.zoom_button.isEnabled()
w.close()
""")
