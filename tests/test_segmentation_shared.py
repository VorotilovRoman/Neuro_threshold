"""Common-service migration contracts; unique auto research is explicitly frozen."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from utils_new.dataset_yolo_labels_new import load_annotations
from utils_new.presets_store import read_presets, write_presets
from utils_new.segmentation import image_session, postprocessing
from utils_new.segmentation.export import write_annotations_atomic

ROOT = Path(__file__).resolve().parents[1]


class SegmentationCommonTests(unittest.TestCase):
    def test_auto_research_sources_and_special_interfaces_unchanged(self):
        frozen = json.loads((ROOT/'tests/fixtures/segmentation_auto_freeze.json').read_text())
        for name, digest in frozen['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest, name)
        for name, methods in frozen['methods'].items():
            nodes = {n.name:n for n in ast.walk(ast.parse((ROOT/name).read_text()))
                     if isinstance(n, ast.FunctionDef)}
            for method, digest in methods.items():
                self.assertEqual(hashlib.sha256(ast.dump(nodes[method], include_attributes=False).encode()).hexdigest(),
                                 digest, name+':'+method)

    def test_legacy_normalization_and_nonzero_floor_resize(self):
        np.testing.assert_array_equal(image_session.legacy_uint8(np.array([[0.,2.,5.,10.]],np.float32)),
                                      [[0,2,5,10]])
        np.testing.assert_array_equal(image_session.legacy_uint8(np.array([[0,256,65535]],np.uint16)),
                                      [[0,1,255]])
        self.assertEqual(image_session.resize_legacy(np.zeros((301,1000,3),np.uint8),640).shape,(192,640,3))
        self.assertEqual(image_session.resize_legacy(np.zeros((1,10000,3),np.uint8),640).shape,(1,640,3))

    def test_image_records_keep_source_geometry_and_unicode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)/'снимок.png'
            cv2.imwrite(str(image), np.zeros((301,1000,3),np.uint8))
            record = image_session.load_one_image_item(str(image),max_side=640)
            self.assertEqual(record['source_size'],(1000,301))
            self.assertEqual(record['working_size'],(640,192))
            self.assertEqual(record['gray'].shape,(192,640))
            self.assertEqual(record['annotations'],[])
            with patch.object(image_session,'read_image_with_fallback',return_value=np.zeros((10,20,1),np.uint16)):
                self.assertEqual(image_session.load_one_image_item('missing-source.png')['image'].shape,(10,20,3))
            with patch.object(image_session,'read_image_with_fallback',return_value=np.zeros((10,20,2),np.uint8)):
                self.assertIsNone(image_session.load_one_image_item('missing-source.png'))
            cache = image_session.create_cache([str(image)],max_side=640)
            try:
                self.assertEqual(cache.get(0)['working_size'],(640,192))
            finally:
                cache.shutdown()

    def test_atomic_export_preserves_segment_type_and_old_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'labels.txt'
            polygon = ('segment',2,[.1,.2,.8,.2,.8,.7,.1,.7])
            write_annotations_atomic([polygon],path,1000,200)
            self.assertEqual(load_annotations(path,1000,200)[0][0],'segment')
            self.assertEqual(len(path.read_text().split()),11)
            before = path.read_bytes()
            def fail(annotations,temporary,*args):
                Path(temporary).write_text('partial')
                return False
            with patch('utils_new.segmentation.export.save_annotations',side_effect=fail):
                with self.assertRaises(ValueError):write_annotations_atomic([polygon],path,1000,200)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])
            self.assertEqual(list(Path(directory).glob('*.txt')),[path])
            write_annotations_atomic([],path,1000,200)
            self.assertEqual(path.read_text(),'')

    def test_extraction_matches_previous_geometry(self):
        from utils_ops.segmentation_ops import (
            segment_contours,
            segment_min_area_rect,
            segment_projections,
        )
        mask = np.zeros((80,120),np.uint8)
        mask[10:40,20:70]=255
        mask[23:25,35:38]=0
        image = np.zeros((80,120,3),np.uint8)
        for mode,fn in [('Segmentation (Polygon)',segment_contours),('Bounding Box (Detect)',segment_projections),
                        ('OBB (Oriented Box)',segment_min_area_rect)]:
            old = fn(image,mask,draw=False,**({'use_hull':False} if fn!=segment_projections else {}))[1]
            self.assertEqual(postprocessing.extract_objects(image,mask,mode)[1],old)
        self.assertTrue(postprocessing.extract_objects(image,mask,'Contours (simple)')[1])

    def test_preset_store_writes_atomically_without_widget_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'presets.json'
            presets = {'default':{'morphology':{'units':'px','shape':'Ellipse','operations':[]}}}
            write_presets(path,presets)
            self.assertEqual(read_presets(path),presets)
            before = path.read_bytes()
            with self.assertRaises(TypeError):write_presets(path,{'bad':object()})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_current_views_and_settings_have_single_canonical_owners(self):
        import ui.annotation_review_view
        import ui.image_view
        import ui.settings_service
        from utils.image_io import LazyImageCache
        from utils_new.dataset_files_new import LazyImageCache as canonical
        self.assertIs(LazyImageCache, canonical)
        self.assertEqual(ui.image_view.SmartGraphicsViewNew.__module__, 'ui.image_view')
        self.assertEqual(ui.annotation_review_view.DatasetReviewGraphicsView.__module__, 'ui.annotation_review_view')
        for module in ('utils/settings.py','utils/presets.py','utils_new/smart_view_new.py','utils_new/dataset_review_view.py'):
            self.assertFalse((ROOT/module).exists(), module)

    def test_five_tabs_and_their_ui_no_longer_import_legacy_utils(self):
        code = """import importlib,sys
for name in ('threshold','gradient','interactive','traditional_ml','deep_learning'):
    importlib.import_module('tabs.'+name+'_methods')
    importlib.import_module('ui_tabs.'+name+'_methods_ui')
legacy = [name for name in sys.modules if name.startswith('utils.')]
assert not legacy, legacy
"""
        result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=60,
                                env=dict(os.environ,NO_ALBUMENTATIONS_UPDATE='1'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_new_computational_services_do_not_import_qt_or_ml(self):
        code = '''import sys
import utils_new.segmentation.image_session
import utils_new.segmentation.postprocessing
import utils_new.segmentation.export
import utils_new.presets_store
import utils_new.segmentation.presets
import utils_new.settings_store
assert not any(n.startswith(('PyQt5','torch','ultralytics','matplotlib')) for n in sys.modules)
'''
        result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)




@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS') == '1', 'opt-in GUI/runtime suite')
class SegmentationSharedGuiTests(unittest.TestCase):
    def run_gui(self, code):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', NO_ALBUMENTATIONS_UPDATE='1',
                       XDG_CONFIG_HOME=directory, XDG_CACHE_HOME=directory,
                       MPLCONFIGDIR=directory, YOLO_CONFIG_DIR=directory)
            result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,
                                    capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_panels_presets_and_vector_selection_in_all_five_tabs(self):
        self.run_gui('''
from pathlib import Path
import tempfile,json
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QFileDialog,QInputDialog
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtTest import QSignalSpy
from main import ImagePlayer
from ui.application import exec_window
app=QApplication([])
keys=['threshold','gradient','interactive','traditional_ml','deep_learning']
window=ImagePlayer([(key,key) for key in keys]);window.show()
with tempfile.TemporaryDirectory() as directory:
    image=Path(directory)/'frame.png'
    cv2.imwrite(str(image),np.zeros((64,128,3),np.uint8))
    for index,owner in enumerate(window._tab_widgets):
        window.tab_widget.setCurrentIndex(index);app.processEvents()
        with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(image)],'')):
            owner.load_images_from_dialog()
        owner.current_base_image=np.full((64,128,3),127,np.uint8)
        before=owner.current_base_image.copy()
        owner.current_objects_full=[('detect',0,.3,.5,.2,.3),('detect',1,.7,.5,.2,.3)]
        owner.update_object_list()
        assert owner.current_selected_indices==[0,1]
        owner.object_list.item(0).setCheckState(Qt.Unchecked)
        assert owner.current_selected_indices==[1]
        assert len(owner.annotated_view.annotations)==1
        assert np.array_equal(owner.current_base_image,before)
        panel=owner.morphology_widget
        events=QSignalSpy(panel.parameters_changed)
        panel.set_operation_size('close',13)
        assert len(events)==1
        assert panel.get_steps()[0].kernel_px==13
        owner.preset_manager.presets_file=str(Path(directory)/(keys[index]+'.json'))
        with patch.object(QInputDialog,'getText',return_value=('shared_refactor',True)):
            owner.save_as_preset()
        stored=json.loads(Path(owner.preset_manager.presets_file).read_text())
        assert 'shared_refactor' in stored
        assert stored['shared_refactor']['morphology']['operations'][0]['kernel_px']==13
        assert stored['shared_refactor']['morphology']['units']=='px'
        assert 'close_factor' not in stored['shared_refactor']
        owner.preset_manager.apply_preset('shared_refactor')
        cache=getattr(owner,'_lazy_cache',None)
        if cache is not None:cache.shutdown()
    QTimer.singleShot(10,window.close);exec_window(window,app)
''')

    def test_common_export_cancels_without_write_and_roundtrips_in_all_tabs(self):
        self.run_gui('''
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
from PyQt5.QtCore import QTimer
from main import ImagePlayer
from ui.application import exec_window
from utils_new.dataset_yolo_labels_new import load_annotations
app=QApplication([])
keys=['threshold','gradient','interactive','traditional_ml','deep_learning']
window=ImagePlayer([(key,key) for key in keys]);window.show()
with tempfile.TemporaryDirectory() as directory:
    for index,owner in enumerate(window._tab_widgets):
        window.tab_widget.setCurrentIndex(index);app.processEvents()
        image=Path(directory)/(keys[index]+'.png');label=image.with_suffix('.txt')
        cv2.imwrite(str(image),np.zeros((80,160,3),np.uint8))
        label.write_text('0 0.5 0.5 0.1 0.1\\n');before=label.read_bytes()
        with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(image)],'')):
            owner.load_images_from_dialog()
        owner.current_objects_full=[('segment',2,[.1,.2,.8,.2,.8,.7,.1,.7])]
        owner.current_selected_indices=[0]
        with patch.object(QMessageBox,'question',return_value=QMessageBox.No) as question:
            owner.save_current_annotations()
            assert question.call_args.args[-1]==QMessageBox.No
            assert question.call_args.args[0].isVisible()
        assert label.read_bytes()==before
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes),patch.object(QMessageBox,'information'),patch.object(QMessageBox,'critical') as error:
            owner.save_current_annotations();error.assert_not_called()
        assert load_annotations(label,160,80)[0][0]=='segment'
        assert len(label.read_text().split())==11
        cache=getattr(owner,'_lazy_cache',None)
        if cache is not None:cache.shutdown()
    QTimer.singleShot(10,window.close);exec_window(window,app)
''')

    def test_new_interactive_view_keeps_roi_and_fg_bg_after_zoom(self):
        self.run_gui('''
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import Qt,QPointF,QTimer
from PyQt5.QtTest import QTest
from main import ImagePlayer
from ui.application import exec_window
app=QApplication([])
host=ImagePlayer([('interactive','Interactive')]);host.show()
owner=host._tab_widgets[0]
with tempfile.TemporaryDirectory() as directory:
    image=Path(directory)/'roi.png'
    pixels=np.zeros((64,128,3),np.uint8);pixels[16:50,25:90]=220
    cv2.imwrite(str(image),pixels)
    owner._load_images([str(image)]);app.processEvents()
    owner.params_widget.grabcut_mode.setCurrentText('Both')
    view=owner.image_view;view.scale(1.4,1.4);app.processEvents()
    def point(x,y):return view.mapFromScene(QPointF(x,y))
    owner.activate_rect_tool()
    QTest.mousePress(view.viewport(),Qt.LeftButton,pos=point(15,10))
    QTest.mouseMove(view.viewport(),point(105,55),30)
    QTest.mouseRelease(view.viewport(),Qt.LeftButton,pos=point(105,55))
    assert owner.rect is not None
    owner.activate_fg_tool()
    QTest.mouseClick(view.viewport(),Qt.LeftButton,pos=point(50,30))
    owner.activate_bg_tool()
    QTest.mouseClick(view.viewport(),Qt.LeftButton,pos=point(5,5))
    assert owner.fg_scribbles[30,50]>0 and owner.bg_scribbles[5,5]>0
    assert owner.fg_scribbles.shape==(64,128)
    with patch.object(QMessageBox,'warning') as warning,patch.object(QMessageBox,'critical') as error:
        owner.run_segmentation();warning.assert_not_called();error.assert_not_called()
    assert owner.mask is not None and owner.mask.shape==(64,128) and owner.mask.max()>0
    owner._lazy_cache.shutdown()
    QTimer.singleShot(10,host.close);exec_window(host,app)
''')
    def test_navigation_resets_unique_result_state_and_updates_view_geometry(self):
        self.run_gui('''
from pathlib import Path
import tempfile
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QFileDialog
from PyQt5.QtCore import QTimer
from main import ImagePlayer
from ui.application import exec_window
app=QApplication([])
keys=['threshold','gradient','interactive','traditional_ml','deep_learning']
host=ImagePlayer([(key,key) for key in keys]);host.show()
with tempfile.TemporaryDirectory() as directory:
    first=Path(directory)/'first.png';second=Path(directory)/'second.png'
    cv2.imwrite(str(first),np.zeros((64,128,3),np.uint8))
    cv2.imwrite(str(second),np.full((96,80,3),120,np.uint8))
    for index,owner in enumerate(host._tab_widgets):
        host.tab_widget.setCurrentIndex(index);app.processEvents()
        with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(first),str(second)],'')):
            owner.load_images_from_dialog()
        if index==2:
            owner.rect=(10,10,40,40)
            owner.fg_scribbles=np.ones((64,128),np.uint8)*255
            owner.goto_image(1)
            assert owner.rect==(10,10,40,40)
        if index in (3,4): owner.prediction_mask=np.ones((64,128),np.uint8)*255
        owner.next_image()
        assert owner.current_index==1
        for name in owner.segmentation_view_names:
            view=getattr(owner,name)
            assert (view.img_width,view.img_height)==(80,96), (keys[index],name)
        if index==2:
            assert owner.rect is None and (owner.fg_scribbles is None or not owner.fg_scribbles.any())
        if index in (3,4):assert owner.prediction_mask is None
        owner.goto_image(1)
        assert owner.current_index==0
        owner.clear_images()
        assert not owner.image_paths
        for name in owner.segmentation_view_names:
            view=getattr(owner,name)
            assert not view.annotations and (view.img_width,view.img_height)==(0,0)
    QTimer.singleShot(10,host.close);exec_window(host,app)
''')


if __name__ == '__main__':
    unittest.main()
