"""Source capabilities, shared import dialogs and real cross-tab input actions."""

import ast
import textwrap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from utils_new.dataset_files_new import collect_image_paths

ROOT = Path(__file__).resolve().parents[1]


class SourceContracts(unittest.TestCase):
    def test_image_enumeration_preserves_selection_order_and_explicit_recursion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            first = root / "a.PNG"
            second = root / "b.jpg"
            deep = root / "nested" / "c.tif"
            for path in (first, second, deep):
                path.touch()
            self.assertEqual(collect_image_paths(root), [str(first), str(second)])
            self.assertEqual(
                collect_image_paths(root, recursive=True),
                [str(first), str(second), str(deep)],
            )
            self.assertEqual(
                collect_image_paths([second, first, second]), [str(second), str(first)]
            )
            self.assertEqual(collect_image_paths(first), [str(first)])
            self.assertEqual(collect_image_paths(None), [])

    def test_descriptions_cover_all_tabs_and_open_dialogs_have_one_owner(self):
        from ui.dataset_source import TAB_DESCRIPTIONS

        tree = ast.parse((ROOT / "main_ui.py").read_text())
        tabs = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ALL_TABS" for t in n.targets)
        )
        self.assertEqual(set(TAB_DESCRIPTIONS), {key for key, _ in tabs})
        self.assertTrue(all(len(value) > 35 for value in TAB_DESCRIPTIONS.values()))
        for folder in ("tabs", "ui", "ui_tabs", "utils_new"):
            for path in (ROOT / folder).rglob("*.py"):
                if path.name == "dataset_source.py":
                    continue
                for n in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                    if (
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("getOpenFileName", "getOpenFileNames")
                    ):
                        self.fail(f"Duplicated input dialog in {path}:{n.lineno}")


@unittest.skipUnless(os.environ.get("NEURO_RUN_GUI_TESTS") == "1", "opt-in Qt tests")
class SourceGuiTests(unittest.TestCase):
    def gui(self, code):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                XDG_CONFIG_HOME=directory,
                XDG_CACHE_HOME=directory,
                MPLCONFIGDIR=directory,
                YOLO_CONFIG_DIR=directory,
                NO_ALBUMENTATIONS_UPDATE="1",
                PYTHONFAULTHANDLER="1",
            )
            cleanup = "\nfrom PyQt5.QtWidgets import QApplication\nfrom PyQt5.QtCore import QCoreApplication,QEvent\napp=QApplication.instance()\nif app:\n for widget in app.topLevelWidgets():\n  widget.close();widget.deleteLater()\n QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)\n"
            script = (
                "try:\n"
                + textwrap.indent(code, " ")
                + "\nfinally:\n"
                + textwrap.indent(cleanup, " ")
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=70,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_tabs_use_shared_controls_with_truthful_capabilities(self):
        self.gui("""
from PyQt5.QtWidgets import QApplication
from main import ImagePlayer
from main_ui import ALL_TABS,StartupDialog
from ui.dataset_source import DatasetSourceWidget,ACTIONS,TAB_DESCRIPTIONS
app=QApplication([]);window=ImagePlayer(ALL_TABS);window.show();app.processEvents()
window.resize(1280,900);app.processEvents()
assert window.width()<=1280 and window.height()<=900,(window.width(),window.height())
chooser=StartupDialog()
for index,(key,title) in enumerate(ALL_TABS):
 window.tab_widget.setCurrentIndex(index);app.processEvents()
 owner=window._tab_widgets[index]
 assert window.tab_widget.tabToolTip(index)==TAB_DESCRIPTIONS[key]
 assert chooser.checkboxes[key].toolTip()==TAB_DESCRIPTIONS[key]
 widgets=owner.findChildren(DatasetSourceWidget)
 if key=='settings':assert not widgets
 else:assert widgets,key
 for widget in widgets:
  for action,button in widget.buttons.items():
   assert button.text()==ACTIONS[action][0],(key,action,button.text())
   assert button.toolTip(),(key,action)
 if hasattr(owner,'nav_widget'):
  assert owner.nav_widget.height()<=40,(key,owner.nav_widget.height())
 for widget in widgets:
  if widget.isVisible():assert widget._bar.height()<=36,(key,widget._bar.height())
 if key=='sahi':
  assert owner.dataset_source.type_combo.count()==1
  assert 'unet_yaml' not in owner.dataset_source.buttons and 'masks' not in owner.dataset_source.buttons
  from unittest.mock import patch
  from PyQt5.QtWidgets import QMessageBox
  with patch.object(QMessageBox,'warning') as warning:
   owner._load_unet_from_path('not-loaded.yaml');assert warning.called and owner.dataset_type==0
  from PyQt5.QtWidgets import QFileDialog
  with patch.object(QFileDialog,'getOpenFileName',return_value=('masked.yaml','')),patch('utils_new.dataset_loader_new.is_unet_yaml',return_value=True),patch.object(QMessageBox,'warning') as warning,patch.object(QMessageBox,'question') as question:
   owner.btn_load_yaml.click();assert warning.called;question.assert_not_called()
 if key in ('dataset','dataset_analysis'):
  panel=owner.dataset_source;assert panel.isAncestorOf(owner.btn_load_yaml)
  panel.type_combo.setCurrentIndex(panel.type_combo.findData(1))
  assert panel.button('labels').text()==ACTIONS['masks'][0]
  assert not panel.format_combo.isEnabled()
  panel.type_combo.setCurrentIndex(panel.type_combo.findData(2));assert not panel.button('labels').isEnabled()
  panel.type_combo.setCurrentIndex(0)
chooser.close();window.close();app.processEvents()
""")

    def test_real_image_and_folder_buttons_across_all_image_tabs(self):
        self.gui("""
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QFileDialog
from main import ImagePlayer
from ui.dataset_source import DatasetSourceWidget
keys=['threshold','gradient','interactive','traditional_ml','deep_learning','labeler','test','yolo_demo','yolo_aeb']
app=QApplication([]);window=ImagePlayer([(key,key) for key in keys]);window.show();app.processEvents()
window._tab_widgets[7].auto_run_yolo=False
with tempfile.TemporaryDirectory() as directory:
 image=Path(directory)/'снимок.png';cv2.imwrite(str(image),np.full((40,80,3),100,np.uint8))
 for key,owner in zip(keys,window._tab_widgets):
  panel=owner.nav_widget.source_widget
  with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(image)],'')) as picker:
   panel.button('images').click();assert picker.call_count==1,key
  assert owner.image_paths==[str(image)],key
  assert str(640 if key=='traditional_ml' else 1024) in owner.nav_widget._resize_cb.text()
  with patch.object(QFileDialog,'getOpenFileNames',return_value=([],'')):
   panel.button('images').click()
  assert owner.image_paths==[str(image)],key
  with patch.object(QFileDialog,'getExistingDirectory',return_value=directory) as picker:
   panel.button('folder').click();assert picker.call_count==1,key
  assert owner.image_paths==[str(image)],key
 window.close();app.processEvents()
""")

    def test_cancelled_auxiliary_folder_keeps_yaml_dataset_and_mask_ids(self):
        self.gui("""
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np,yaml
from PyQt5.QtWidgets import QApplication,QFileDialog
from tabs.dataset_analysis import DatasetAnalysisWindow
from utils_new.dataset_files_new import read_index_mask
app=QApplication([]);owner=DatasetAnalysisWindow();owner.show()
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);(root/'images').mkdir();(root/'labels').mkdir()
 image=root/'images'/'one.png';cv2.imwrite(str(image),np.zeros((40,80),np.uint8));(root/'labels'/'one.txt').write_text('0 .5 .5 .4 .4\\n')
 data=root/'data.yaml';data.write_text(yaml.safe_dump({'path':str(root),'train':'images','names':{0:'Object'}}))
 with patch.object(QFileDialog,'getOpenFileName',return_value=(str(data),'')):
  owner.btn_load_yaml.click()
 assert owner._yaml_mode
 pairs=list(owner.pairs);source=owner._source_yaml_path
 with patch.object(QFileDialog,'getExistingDirectory',return_value=''),patch.object(owner.validator,'clear_cache') as clear:
  owner.btn_labels.click();clear.assert_not_called()
 assert owner._yaml_mode and owner._source_yaml_path==source and owner.pairs==pairs
 owner.val_data_edit.setText('previous.yaml')
 with patch.object(QFileDialog,'getOpenFileName',return_value=('','')),patch.object(QFileDialog,'getExistingDirectory') as directory_picker:
  owner.btn_browse_val_data.click();directory_picker.assert_not_called()
 assert owner.val_data_edit.text()=='previous.yaml'
 mask=root/'index.png';arr=np.zeros((40,80),np.uint16);arr[10:30,20:50]=1000;cv2.imwrite(str(mask),arr)
 assert read_index_mask(mask).dtype==np.uint16 and read_index_mask(mask).max()==1000
 owner.dataset_type_combo.setCurrentIndex(1)
 (root/'masks').mkdir();indexed=root/'masks'/'one.png';cv2.imwrite(str(indexed),arr)
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root/'images')):
  owner.btn_folder.click()
 with patch.object(QFileDialog,'getExistingDirectory',return_value=str(root/'masks')):
  owner.btn_labels.click()
 assert owner.pairs==[(str(image),str(indexed))],owner.pairs
 assert not owner._yaml_mode and owner._source_yaml_path is None
 assert read_index_mask(owner.pairs[0][1]).max()==1000
 owner.close();app.processEvents()
""")

    def test_coco_files_and_folder_share_real_chooser_and_cancellation(self):
        self.gui("""
import tempfile,json
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication,QFileDialog,QMessageBox
from ui.dataset_source import choose_coco_sources
app=QApplication([])
def choose(caption):
 def click():
  box=app.activeModalWidget();assert isinstance(box,QMessageBox)
  next(button for button in box.buttons() if button.text()==caption).click()
 QTimer.singleShot(0,click)
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'annotations.json';path.write_text(json.dumps({'images':[],'annotations':[],'categories':[]}))
 choose('Файлы JSON…')
 with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(path)],'')):
  assert choose_coco_sources(None)==[str(path)]
 choose('Папка JSON…')
 with patch.object(QFileDialog,'getExistingDirectory',return_value=directory):
  assert choose_coco_sources(None)==[str(path)]
 choose('Отмена')
 with patch.object(QFileDialog,'getOpenFileNames') as picker:
  assert choose_coco_sources(None)==[];picker.assert_not_called()
""")

    def test_compact_sources_keep_hidden_actions_available_with_correct_state(self):
        self.gui("""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from ui.dataset_source import DatasetSourceWidget,ACTIONS
app=QApplication([])
panel=DatasetSourceWidget(('images','folder','labels','masks','yolo_yaml','coco','unet_yaml'),types=(0,1,2),formats=True,profile='test')
panel.resize(500,32);panel.show();app.processEvents()
assert panel.height()<=36,panel.height()
assert panel.overflow.isVisible()
visible=[button for button in panel.buttons.values() if button.isVisible()]
assert len(visible)<len(panel.buttons)
assert len({button.geometry().center().y() for button in visible})<=1
seen=[];panel.requested.connect(seen.append)
def choose_hidden():
 menu=panel.overflow.menu()
 next(action for action in menu.actions() if action.text()==ACTIONS['unet_yaml'][0]).trigger()
 menu.close()
QTimer.singleShot(0,choose_hidden);panel.overflow.click()
assert seen==['unet_yaml'],seen
panel.type_combo.setCurrentIndex(panel.type_combo.findData(1));panel._refresh_menu()
assert panel.button('labels').text()==ACTIONS['masks'][0]
panel.type_combo.setCurrentIndex(panel.type_combo.findData(2));panel._refresh_menu()
assert not panel.button('labels').isEnabled() and not panel.format_combo.isEnabled()
assert any(action.text()==ACTIONS['labels'][0] and not action.isEnabled() for action in panel.overflow.menu().actions())
panel.resize(1600,32);app.processEvents()
assert not panel.overflow.isVisible() and all(button.isVisible() for button in panel.buttons.values())
assert panel.help_button.toolTip() and panel.help_button.isVisible()
panel.close()
""")

    def test_training_sources_use_correct_dialogs_and_keep_state_on_cancel(self):
        self.gui(r"""
import tempfile
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QFileDialog
from ui.DL_train_settings_base import YOLOTrainWidget,UNetTrainWidget
from ui.DL_train_settings_new_model import DeepLabV3TrainWidget
from ui.dataset_source import ACTIONS
app=QApplication([])
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);yolo=root/'data.yaml';yolo.write_text('names: [object]\ntrain: images/train\n')
 masks=root/'masks_data.yaml';masks.write_text('path: '+str(root)+'\nnum_classes: 3\ntrain_images: images/train\ntrain_masks: masks/train\n')
 for cls,field,yaml_field,role,path in [(YOLOTrainWidget,'dataset_folder','dataset_yaml','yolo_yaml',yolo),(UNetTrainWidget,'dataset_path','unet_yaml','unet_yaml',masks),(DeepLabV3TrainWidget,'dataset_path','deeplab_yaml','unet_yaml',masks)]:
  widget=cls();widget.show();app.processEvents();panel=widget.dataset_source
  with patch.object(QFileDialog,'getExistingDirectory',return_value=directory) as picker:
   panel.button('dataset_root').click();assert picker.call_args.args[1]==ACTIONS['dataset_root'][1]
  assert getattr(widget,field).text()==directory
  with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')) as picker:
   panel.button(role).click();assert picker.call_args.args[3]==ACTIONS[role][2]
  before=(getattr(widget,field).text(),getattr(widget,yaml_field).text())
  assert before[1]==str(path)
  with patch.object(QFileDialog,'getOpenFileName',return_value=('','')):
   panel.button(role).click()
  with patch.object(QFileDialog,'getExistingDirectory',return_value=''):
   panel.button('dataset_root').click()
  assert before==(getattr(widget,field).text(),getattr(widget,yaml_field).text())
  widget.close();widget.deleteLater();app.processEvents()
""")

    def test_model_selection_uses_common_supported_filters(self):
        self.gui("""
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication,QFileDialog
from ui.DL_predict_settings import YOLOInferenceSettings
from ui.dataset_source import ACTIONS
app=QApplication([]);widget=YOLOInferenceSettings();widget.show()
with patch.object(QFileDialog,'getOpenFileName',return_value=('model.onnx','')) as picker:
 widget._browse_file()
 assert widget._model_path.text()=='model.onnx'
 assert '*.onnx' in picker.call_args.args[3] and '*.engine' in picker.call_args.args[3]
widget.close();app.processEvents()
""")
