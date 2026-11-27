"""Common complete training presets above all per-stage forms."""
import json
import math
from copy import deepcopy
from pathlib import Path

from PyQt5.QtWidgets import (QWidget,QHBoxLayout,QVBoxLayout,QComboBox,QCheckBox,QRadioButton,
    QSpinBox,QDoubleSpinBox,QLineEdit,QPushButton,QLabel,QFileDialog,QMessageBox,QInputDialog,QSizePolicy)

from path_setup import get_project_root
from ui import dataset_source as sources
from utils_new.configuration.presets_store import (read_presets,write_presets,validate_training_preset,
                                                    TRAINING_WORKER_DEFAULTS,training_worker_defaults)

RESOURCE_FIELDS={'dataset_path','dataset_folder','dataset_yaml','unet_yaml','deeplab_yaml',
                 'source','val_source','custom_model_path','weights','resume','project_path','experiment_name','class_weights_file'}
SOURCE_FIELDS={'dataset_yaml','unet_yaml','deeplab_yaml','source','val_source'}
IGNORED_FIELDS={'preset_path','preview_model_name','output_activation','advanced_button'}


def form_controls(owner):
    return {name:control for name,control in vars(owner).items()
            if name not in IGNORED_FIELDS and not name.startswith('_')
            and isinstance(control,(QComboBox,QCheckBox,QRadioButton,QSpinBox,QDoubleSpinBox,QLineEdit))
            and not (isinstance(control,QLineEdit) and control.isReadOnly())}


def control_value(name,control):
    if isinstance(control,(QCheckBox,QRadioButton)):return control.isChecked()
    if isinstance(control,QComboBox):return control.currentData() if control.currentData() is not None else control.currentText()
    if isinstance(control,QLineEdit):return sources.dataset_path(control) if name in SOURCE_FIELDS else control.text()
    return control.value()


class TrainingPresetBar(QWidget):
    def __init__(self,owner,family):
        super().__init__(owner)
        self.owner,self.family=owner,family
        self.controls=form_controls(owner)
        self.worker_parameters={}
        self.required=set(self.controls)-RESOURCE_FIELDS
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        row=QHBoxLayout();layout.addLayout(row)
        self.selector=QComboBox();self.selector.addItem('Текущие параметры',None)
        self.selector.setMinimumContentsLength(18)
        self.selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.selector.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        catalog=Path(get_project_root())/'settings/training'/f'{family}.json'
        if catalog.is_file():
            for preset in read_presets(catalog)['presets']:
                self.validate(preset)
                self.selector.addItem(preset['title'],preset)
                self.selector.setItemData(self.selector.count()-1,preset.get('description',''),3)
        self.apply_button=QPushButton('Применить')
        self.import_button=QPushButton('JSON…');self.save_button=QPushButton('Сохранить…')
        self.import_button.setToolTip('Загрузить полный пресет из JSON')
        self.save_button.setToolTip('Сохранить полную конфигурацию в JSON')
        self.extra_button=QPushButton('Доп. JSON…')
        self.extra_button.setEnabled(bool(TRAINING_WORKER_DEFAULTS[family]))
        self.paths=QCheckBox('С путями')
        self.paths.setToolTip('При сохранении включить датасет, веса, папку результата и имя запуска. Одобренные источники сохраняются рядом с JSON.')
        row.addWidget(self.selector,1)
        for control in (self.apply_button,self.import_button,self.save_button,self.extra_button,self.paths):row.addWidget(control)
        self.note=QLabel('Пресет задаёт параметры всех страниц. Датасет и папка результата сохраняются. Нажмите «Применить».')
        self.note.setWordWrap(True);layout.addWidget(self.note)
        self.apply_button.clicked.connect(self.apply_selected)
        self.import_button.clicked.connect(self.import_json)
        self.save_button.clicked.connect(self.save_json)
        self.extra_button.clicked.connect(self.edit_extra)

    def snapshot(self,include_paths=False):
        worker=training_worker_defaults(self.family)
        if self.family=='sam' and self.owner.resume.text().strip():
            cfg=getattr(self.owner,'_resume_defaults',{})
            worker.update({key:cfg[key] for key in worker if key in cfg})
        worker.update(deepcopy(self.worker_parameters))
        return dict(schema_version=1,family=self.family,title='Мой пресет',
                    settings={name:control_value(name,control) for name,control in self.controls.items()
                              if include_paths or name not in RESOURCE_FIELDS},worker_parameters=worker)

    def validate(self,preset):
        validate_training_preset(preset,self.family,self.controls,self.required)
        for name,value in preset['settings'].items():
            control=self.controls[name]
            if isinstance(control,(QCheckBox,QRadioButton)):
                if type(value) is not bool:raise ValueError(f'{name}: требуется bool.')
            elif isinstance(control,QComboBox):
                if control.findData(value)<0 and control.findText(str(value))<0:raise ValueError(f'{name}: неизвестное значение {value}.')
            elif isinstance(control,QLineEdit):
                if not isinstance(value,str):raise ValueError(f'{name}: требуется строка.')
            else:
                if type(value) not in (int,float) or not math.isfinite(value) or not control.minimum()<=value<=control.maximum():
                    raise ValueError(f'{name}: значение вне диапазона формы.')
                if isinstance(control,QSpinBox) and type(value) is not int:raise ValueError(f'{name}: требуется int.')
                if isinstance(control,QDoubleSpinBox) and not math.isclose(value,round(value,control.decimals()),abs_tol=1e-12,rel_tol=0):
                    raise ValueError(f'{name}: форма поддерживает {control.decimals()} знаков после запятой.')
        settings=preset['settings']
        if 'model_from_preset' in settings and settings['model_from_preset']==settings['model_from_file']:
            raise ValueError('Выберите один источник модели.')
        if settings.get('two_phase_check') and settings['epochs']!=settings['freeze_epochs']+settings['unfreeze_epochs']:
            raise ValueError('Число эпох не равно сумме фаз.')
        if self.family=='sam' and (settings['batch']!=1 or settings['auto_batch']):
            raise ValueError('SAM: batch=1, AutoBatch выключен.')
        self.validate_worker(preset.get('worker_parameters',{}))
        return preset

    def validate_worker(self,worker):
        defaults=TRAINING_WORKER_DEFAULTS[self.family]
        if not isinstance(worker,dict) or set(worker)-set(defaults):raise ValueError('Неизвестные дополнительные параметры воркера.')
        # Reuse pure backend validation without requiring a selected dataset.
        if self.family in ('unet','deeplab','segformer'):
            from utils_new.configuration.training import semantic_config
            semantic_config(worker,self.family)
        elif self.family=='yolo':
            from utils_new.configuration.training import validate_yolo_options
            validate_yolo_options(worker)
        else:
            for name,value in worker.items():
                if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError(f'{name}: требуется конечное неотрицательное число.')
            if self.family=='sam':
                if not 0<worker.get('validation_fraction',.2)<=.5 or not 0<=worker.get('box_jitter',.05)<=.25 or worker.get('grad_clip',1)<=0:
                    raise ValueError('Неверная validation_fraction/box_jitter/grad_clip.')

    def apply_preset(self,preset):
        self.validate(preset)  # Reject the entire recipe before changing the form.
        values=preset['settings']
        previous_worker=deepcopy(self.worker_parameters)
        resume_defaults=None
        if self.family=='sam' and values.get('resume'):
            from utils_new.import_export.training import read_model_meta
            meta=read_model_meta(values['resume'])
            if meta.get('kind')!='sam' or not meta.get('complete_epoch'):
                raise ValueError('Для resume SAM нужен checkpoint полной эпохи и его метаданные.')
            resume_defaults=meta.get('config')
            if not isinstance(resume_defaults,dict) or resume_defaults.get('model_key')!=values.get('model'):
                raise ValueError('Архитектура пресета не соответствует checkpoint SAM.')
        try:
            # Dependencies: flags first, source paths then architecture/remaining
            # fields, phase durations before the final total epochs. SAM model
            # changes reset weights/resume, so restore these two paths last.
            order=sorted(values,key=lambda name:(0 if isinstance(self.controls[name],(QCheckBox,QRadioButton)) else
                                                3 if self.family=='sam' and name in ('weights','resume') else
                                                1 if name in RESOURCE_FIELDS else 4 if name=='epochs' else 2))
            for name in order:
                control,value=self.controls[name],values[name]
                if isinstance(control,(QCheckBox,QRadioButton)):control.setChecked(value)
                elif isinstance(control,QComboBox):
                    index=control.findData(value)
                    control.setCurrentIndex(index if index>=0 else control.findText(str(value)))
                elif isinstance(control,QLineEdit):
                    if name in SOURCE_FIELDS:sources.set_dataset_path(control,value)
                    else:control.setText(value)
                else:control.setValue(value)
            if self.family=='sam':
                if 'resume' not in values:self.owner.resume.clear()
                if not self.owner.resume.text():self.owner._resume_defaults={}
                elif resume_defaults is not None:self.owner._resume_defaults=resume_defaults
                if 'weights' not in values:self.owner._find_weights()
            self.worker_parameters=deepcopy(preset['worker_parameters'])
            self.note.setText(preset.get('description') or 'Применены все параметры пресета; проверьте источник данных и веса.')
        except Exception:
            # Values/types were prevalidated. Resource signal handlers retain
            # their own visible diagnostics; do not silently continue a failure.
            self.worker_parameters=previous_worker
            raise

    def merge(self,params):
        overlap=set(params)&set(self.worker_parameters)
        if overlap and self.family!='sam':raise ValueError(f'Дополнительные параметры перекрывают поля формы: {sorted(overlap)}')
        self.validate_worker(self.worker_parameters)
        return dict(params,**self.worker_parameters)

    def apply_selected(self):
        value=self.selector.currentData()
        if value:
            try:self.apply_preset(value)
            except (ValueError,TypeError,OSError) as error:QMessageBox.warning(self,'Пресет обучения',str(error))

    def import_json(self):
        path,_=sources.get_open_file(self,'training_preset')
        if path:
            try:self.apply_preset(read_presets(path))
            except (ValueError,TypeError,OSError) as error:QMessageBox.warning(self,'Пресет обучения',str(error))

    def save_to(self,path,include_paths=False):
        value=self.snapshot(include_paths)
        if include_paths:
            from utils_new.import_export.dataset_sources import persist_source
            import hashlib
            target=Path(path).resolve()
            for name in SOURCE_FIELDS & value['settings'].keys():
                source=value['settings'][name]
                if source and Path(source).is_file():
                    fingerprint=hashlib.sha256(Path(source).read_bytes()).hexdigest()[:12]
                    value['settings'][name]=persist_source(source,target.parent/(target.stem+'.dataset_sources')/fingerprint)
        self.validate(value);write_presets(path,value)

    def save_json(self):
        path,_=QFileDialog.getSaveFileName(self,'Сохранить все параметры обучения','training.json','JSON (*.json)')
        if path:
            try:self.save_to(path,self.paths.isChecked())
            except (ValueError,TypeError,OSError) as error:QMessageBox.warning(self,'Пресет обучения',str(error))

    def edit_extra(self):
        defaults=TRAINING_WORKER_DEFAULTS[self.family]
        value,accepted=QInputDialog.getMultiLineText(self,'Дополнительные параметры воркера',
            'JSON. Поля формы не перекрываются; допустимые ключи:\n'+', '.join(defaults),json.dumps(self.worker_parameters or defaults,ensure_ascii=False,indent=2))
        if accepted:
            try:
                parsed=json.loads(value);self.validate_worker(parsed);self.worker_parameters=parsed
                self.note.setText('Дополнительные параметры воркера обновлены: '+', '.join(parsed))
            except (ValueError,TypeError) as error:QMessageBox.warning(self,'Параметры воркера',str(error))
