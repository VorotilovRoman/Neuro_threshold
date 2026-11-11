"""Contracts of common annotation kinds, masks, presentation and task lifecycle."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils_new.annotations import ImageSize, from_legacy
from utils_new.annotations.modes import MODE_LABELS, supports_hull
from utils_new.segmentation.postprocessing import extract_objects
from utils_new.segmentation.result import binary_mask, masked_preview

ROOT=Path(__file__).resolve().parents[1]


class UnifiedCoreTests(unittest.TestCase):
    def test_modes_are_dataset_annotation_kinds_and_same_extractor(self):
        image=np.zeros((80,160,3),np.uint8);mask=np.zeros((80,160),np.uint8);mask[20:60,40:120]=255
        self.assertEqual(list(MODE_LABELS.values()),['Нет','BBox','OBB','Segment'])
        for mode in ('detect','obb','segment'):
            objects=extract_objects(image,mask,mode,True)[1]
            self.assertEqual(len(objects),1)
            annotation=from_legacy(objects[0],ImageSize(160,80))
            self.assertEqual(annotation.kind,mode)
        self.assertEqual(extract_objects(image,mask,'none')[1],[])
        self.assertFalse(supports_hull('detect'))
        self.assertTrue(supports_hull('obb'))
        with self.assertRaises(ValueError):extract_objects(image,mask,'Contours (simple)')

    def test_mask_contract_distinguishes_missing_empty_labels_probabilities_and_errors(self):
        image=np.full((20,30,3),17,np.uint8)
        np.testing.assert_array_equal(masked_preview(image,None),image)
        np.testing.assert_array_equal(masked_preview(image,np.zeros((20,30),np.uint8)),255)
        labels=np.zeros((20,30),np.int16);labels[3:10,4:20]=2
        expected=(labels>0).astype(np.uint8)*255
        np.testing.assert_array_equal(binary_mask(labels,image.shape),expected)
        np.testing.assert_array_equal(binary_mask((labels>0)*.75,image.shape),expected)
        for invalid in (np.zeros((30,20)),np.full((20,30),np.nan),np.full((20,30),-1),np.full((20,30),.0)+5.5):
            with self.assertRaises(ValueError):binary_mask(invalid,image.shape)
        np.testing.assert_array_equal(image,17)


@unittest.skipUnless(os.environ.get('NEURO_RUN_GUI_TESTS')=='1','opt-in GUI/runtime suite')
class UnifiedGuiTests(unittest.TestCase):
    def run_gui(self,code):
        with tempfile.TemporaryDirectory() as directory:
            env=dict(os.environ,QT_QPA_PLATFORM='offscreen',NO_ALBUMENTATIONS_UPDATE='1',
                     XDG_CONFIG_HOME=directory,XDG_CACHE_HOME=directory,MPLCONFIGDIR=directory,YOLO_CONFIG_DIR=directory)
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_modes_list_focus_and_visual_settings_are_identical_in_all_tabs(self):
        self.run_gui('''
from unittest.mock import patch
import tempfile
from pathlib import Path
import cv2,numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import QApplication,QAbstractItemView
from main import ImagePlayer
from ui.application import exec_window
app=QApplication([]);w=ImagePlayer([(k,k) for k in ['threshold','gradient','interactive','traditional_ml','deep_learning']]);w.show()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'frame.png';cv2.imwrite(str(path),np.zeros((80,160,3),np.uint8))
    for owner in w._tab_widgets:
        owner._load_images([str(path)])
        assert [owner.draw_combo.itemText(i) for i in range(4)]==['Нет','BBox','OBB','Segment']
        assert [owner.draw_combo.itemData(i) for i in range(4)]==['none','detect','obb','segment']
        assert owner.object_list.selectionMode()==QAbstractItemView.SingleSelection
        owner.current_base_image=np.zeros((80,160,3),np.uint8)
        owner.current_objects_full=[('detect',0,.3,.5,.2,.3),('detect',1,.7,.5,.2,.3)]
        owner.update_object_list();owner.object_list.item(0).setCheckState(Qt.Unchecked)
        owner.object_list.setCurrentRow(1)
        assert owner.current_selected_indices==[1] and owner.annotated_view.selected_index==0
        text=owner.coord_text.toPlainText();base=owner.current_base_image
        with patch.object(owner,'display_current_image') as predict,patch.object(owner,'update_object_list') as rebuild:
            owner.on_global_settings_changed({})
            predict.assert_not_called();rebuild.assert_not_called()
        assert owner.current_selected_indices==[1] and owner.annotated_view.selected_index==0
        assert owner.coord_text.toPlainText()==text and owner.current_base_image is base
        if getattr(owner,'_lazy_cache',None):owner._lazy_cache.shutdown()
QTimer.singleShot(10,w.close);assert exec_window(w,app)==0
''')
