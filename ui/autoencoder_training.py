"""All AE/VAE variants in one parameter form; training stays in a worker."""
from pathlib import Path

from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                            QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QTabWidget,
                            QVBoxLayout, QWidget)

from path_setup import get_project_root
from ui import dataset_source as sources
from ui.DL_train_settings_base import BaseTrainWidget
from ui.segmentation.jobs import BackgroundTask
from utils_new.configuration.autoencoder import AE_PROFILES, ARCHITECTURES, TrainConfig, training_request
from utils_new.import_export.json_objects import read_json_object, write_json_atomic


class AutoencoderTrainWidget(BaseTrainWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = self._applying = False
        self._source_job = self._checkpoint_job = None
        self._checkpoint_error = ''
        self.options = QWidget()
        layout = QVBoxLayout(self.options)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.profile = QComboBox()
        self.profile.addItems(AE_PROFILES)
        self.profile.setCurrentText('residual_ae')
        self.apply_profile = QPushButton('Применить профиль')
        self.import_config = QPushButton('JSON…')
        self.export_config = QPushButton('Сохранить JSON…')
        row.addWidget(self.profile, 1)
        for button in (self.apply_profile, self.import_config, self.export_config): row.addWidget(button)
        layout.addLayout(row)
        self.pages = QTabWidget()
        layout.addWidget(self.pages)
        forms = {}
        for key, label in (('data', 'Данные'), ('model', 'Модель'), ('training', 'Обучение'),
                           ('augmentation', 'Аугментации'), ('loss', 'Потери'), ('calibration', 'Калибровка')):
            page = QWidget()
            forms[key] = QFormLayout(page)
            forms[key].setSpacing(6)
            self.pages.addTab(page, label)
        self.forms = forms
        self.form_layout.addRow(self.options)
        self.model_controls, self.train_controls = {}, {}

        self.source_widget = sources.DatasetSourceWidget(('folder', 'ae_images_yaml'))
        self.source = QLineEdit()
        self.source_widget.add_path_field('Снимки:', self.source)
        self.source_widget.button('folder').clicked.connect(self._choose_folder)
        self.source_widget.button('ae_images_yaml').clicked.connect(self._choose_yaml)
        forms['data'].addRow(self.source_widget)
        self.check_source = QPushButton('Проверить источники снимков')
        forms['data'].addRow(self.check_source)
        self.source_info = self._hint(forms['data'],
            'Все выбранные снимки считаются нормальными. TXT/маски не читаются. '
            'YAML: train, val, необязательные calibration и test. Без val выделяется доля train по seed.')
        self.initial_weights = self._path(forms['data'], 'Начальные веса:', lambda: self._choose_checkpoint(False))
        self.resume = self._path(forms['data'], 'Продолжить last.pt:', lambda: self._choose_checkpoint(True))
        self.checkpoint_info = self._hint(forms['data'], 'Начальные веса: новый запуск, новые оптимизатор и пороги. '
                                         'Resume: та же конфигурация/данные/устройство, последняя полная эпоха.')
        self.project_path = self._path(forms['data'], 'Папка проекта:', self._choose_project)
        self.project_path.setText(str(Path(get_project_root())/'runs'/'autoencoder'))
        self.experiment_name = QLineEdit('ae')
        forms['data'].addRow('Название запуска:', self.experiment_name)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['auto', 'cuda:0', 'cpu'])
        forms['data'].addRow('Устройство:', self.device_combo)

        model_spec = [
            ('arch', 'Архитектура', list(ARCHITECTURES)),
            ('model_type', 'Режим', ['ae','vae']), ('in_channels', 'Снимок', [('Grayscale',1),('RGB',3)]),
            ('img_h','Высота H, px',(8,4096)), ('img_w','Ширина W, px',(8,4096)),
            ('latent_dim','Латентный вектор (Conv)',(1,8192)),
            ('latent_channels','Латентные каналы (spatial)',(1,1024)),
            ('base','Ширина base (V2 / spatial)',(1,256)),
            ('blocks_per_stage','Блоков на stage (spatial)',(1,16)),
            ('skip_mode','U-Net skip-связи',[('Включены (реконструкция)', 'full'), ('Отключены (через z)', 'none')]), ('norm_type','Нормализация residual/V2/GAN',['gn','bn']),
            ('bottleneck_dilations','Dilations V2, через запятую','text'),
            ('keep_aspect','Сохранять пропорции',bool), ('pad_value','Padding 0–255',(0,255)),
            ('use_edge_channel','Дополнительный канал границ (V2 / spatial)',bool),
            ('edge_kernel','Оператор границ',['scharr','sobel']), ('median_kernel','Медианный фильтр, px',(1,31)),
            ('edge_lo','Нижний порог границ',(0.,10.)), ('edge_hi','Верхний порог границ',(0.,10.)),
            ('logvar_min','Нижняя граница logvar',(-80.,80.)), ('logvar_max','Верхняя граница logvar',(-80.,80.))]
        for key,label,spec in model_spec:
            self.model_controls[key] = self._field(forms['model'], label, spec)
        self.io_info = self._hint(forms['model'], '')
        self._hint(forms['model'], 'Conv: H/W кратны 16; U-Net/residual/ConvNeXt: кратны 8. '
                   'ConvNeXt V2: LayerNorm/GRN; residual: GN/BN. Spatial-модели не имеют skip-обхода z. '
                   'Выход восстанавливает только каналы снимка; edge заново строится из входа.')

        specs = {
          'training': [('epochs','Эпохи',(1,100000)),('batch_size','Batch (FP32, без AutoBatch)',(1,1024)),
                       ('lr','AdamW learning rate',(1e-8,1.)),('min_lr','Минимальный LR',(0.,1.)),
                       ('weight_decay','Weight decay',(0.,1.)),('patience','Early stopping',(1,100000)),
                       ('grad_clip','Ограничение градиента',(0.000001,1000.)),('val_split','Доля val, если нет в YAML',(.001,.9)),
                       ('seed','Seed',(0,2147483647)),('num_workers','DataLoader workers',(0,64))],
          'augmentation': [('augmentation','Аугментации',bool),('translate','Максимальный сдвиг (доля стороны)',(0.,1.)),
                           ('degrees','Поворот ± градусов',(0.,180.)),('brightness','Изменение яркости',(0.,1.)),
                           ('contrast','Изменение контраста',(0.,1.)),('hflip','Случайное горизонтальное отражение',bool),
                           ('vflip','Случайное вертикальное отражение',bool),('noise_std','Шум во входе, std [0,1]',(0.,1.))],
          'loss': [('loss_kind','Пиксельная ошибка',['l1','mse']),('l1_weight','Вес пиксельной ошибки',(0.,100.)),
                   ('use_ssim','SSIM',bool),('ssim_weight','Вес SSIM',(0.,100.)),('grad_loss_w','Вес градиентов Sobel',(0.,100.)),
                   ('edge_weighted_l1','Усилить L1 на границах',bool),('edge_l1_extra_w','Доп. вес границ L1',(0.,100.)),
                   ('beta_vae_target','Целевой β KL (VAE)',(0.,100.)),('beta_vae_warmup_epochs','Разогрев β, эпох',(0,100000)),
                   ('kl_reduction','Нормировка KL',['mean','sum_per_sample']),('use_gan','PatchGAN',bool),
                   ('gan_start_epoch','Начало GAN, эпоха',(0,100000)),('gan_weight','Вес GAN',(0.,100.)),
                   ('gan_lr','LR дискриминатора',(1e-8,1.)),('discriminator_base','Ширина дискриминатора',(1,256))],
          'calibration': [('image_percentile','Перцентиль MAE снимков',(0.,100.)),
                          ('pixel_percentile','Перцентиль MAE пикселей',(0.,100.)),
                          ('calibration_pixels','Макс. пикселей для калибровки',(1,100000000))]}
        for page, entries in specs.items():
            for key,label,spec in entries: self.train_controls[key] = self._field(forms[page], label, spec)
        self.plots = QCheckBox('График потерь и примеры реконструкций')
        self.plots.setChecked(True)
        forms['calibration'].addRow(self.plots)
        self._hint(forms['augmentation'], 'Геометрия согласована у входа и эталона. Шум добавляется только во вход. '
                   'Границы вычисляются после аугментаций; padding исключается из MAE и калибровки.')
        self._hint(forms['calibration'], 'Пороги считаются по нормальным снимкам calibration; если split отсутствует — по val. '
                   'Test используется только для итоговой оценки. Для VAE оценка детерминирована через mu.')
        self._hint(forms['loss'], 'Val сравнивается при фиксированном целевом β, без GAN-составляющей. '
                   'Низкая ошибка реконструкции сама по себе не доказывает обнаружение дефектов.')
        self.add_buttons()
        self.add_progress_bar()
        self.apply_profile.clicked.connect(self._load_profile)
        self.import_config.clicked.connect(self._import_config)
        self.export_config.clicked.connect(self._export_config)
        self.check_source.clicked.connect(self._check_source)
        self.source.textChanged.connect(self._source_changed)
        self.resume.textChanged.connect(self._resume_state)
        for resume, field in ((True,self.resume),(False,self.initial_weights)):
            field.editingFinished.connect(lambda mode=resume:self._entered_checkpoint(mode))
        for widget in list(self.model_controls.values()) + list(self.train_controls.values()):
            signal = (widget.currentIndexChanged if isinstance(widget,QComboBox) else
                      widget.toggled if isinstance(widget,QCheckBox) else
                      widget.textChanged if isinstance(widget,QLineEdit) else widget.valueChanged)
            signal.connect(self._capabilities)
        self._load_profile()

    def _hint(self, form, text):
        widget=QLabel(text);widget.setWordWrap(True);form.addRow(widget);return widget

    def _path(self, form, label, callback):
        field=QLineEdit();button=QPushButton('…');button.clicked.connect(callback)
        row=QHBoxLayout();row.addWidget(field,1);row.addWidget(button);form.addRow(label,row);return field

    def _field(self, form, label, spec):
        if spec is bool: widget=QCheckBox()
        elif spec == 'text': widget=QLineEdit()
        elif isinstance(spec,list):
            widget=QComboBox()
            for value in spec:
                title,data=value if isinstance(value,tuple) else (value,value)
                widget.addItem(title,data)
        elif all(isinstance(value,int) for value in spec):
            widget=QSpinBox();widget.setRange(*spec)
        else:
            widget=QDoubleSpinBox();widget.setDecimals(8);widget.setRange(*spec);widget.setSingleStep(.01)
        form.addRow(label+':',widget);return widget

    @staticmethod
    def _value(widget):
        if isinstance(widget,QCheckBox): return widget.isChecked()
        if isinstance(widget,QComboBox): return widget.currentData()
        if isinstance(widget,QLineEdit): return widget.text().strip()
        return widget.value()

    def _set(self, widget, value):
        if isinstance(widget,QCheckBox): widget.setChecked(value)
        elif isinstance(widget,QComboBox): widget.setCurrentIndex(widget.findData(value))
        elif isinstance(widget,QLineEdit): widget.setText(', '.join(map(str,value)) if isinstance(value,(tuple,list)) else str(value))
        else:
            if not widget.minimum() <= value <= widget.maximum():
                raise ValueError(f'Значение {value} вне диапазона формы [{widget.minimum()}, {widget.maximum()}]. Используйте CLI или измените JSON.')
            widget.setValue(value)

    def apply_config(self, raw):
        cfg=TrainConfig.from_dict(raw).to_dict()
        # Validate every numeric control before changing any value; Qt spinboxes
        # otherwise silently clamp imported metadata/configurations.
        defaults={'latent_dim':128,'latent_channels':8,'blocks_per_stage':2,'skip_mode':'full'}
        assignments=[(widget,cfg['model'].get(key,defaults.get(key))) for key,widget in self.model_controls.items()]
        assignments += [(widget,cfg[key]) for key,widget in self.train_controls.items()]
        for widget,value in assignments:
            if isinstance(widget,(QSpinBox,QDoubleSpinBox)) and not widget.minimum() <= value <= widget.maximum():
                raise ValueError(f'Значение {value} вне диапазона формы [{widget.minimum()}, {widget.maximum()}]. Используйте CLI или измените JSON.')
        self._applying=True
        try:
            for widget,value in assignments:self._set(widget,value)
        finally: self._applying=False
        self._capabilities()

    def config(self):
        model={key:self._value(widget) for key,widget in self.model_controls.items()}
        try: model['bottleneck_dilations']=[int(value.strip()) for value in model['bottleneck_dilations'].split(',') if value.strip()]
        except ValueError as error: raise ValueError('Dilations: целые числа через запятую.') from error
        model.update(preprocessing_version='ae-pil-v1',vae_use_mean=True,
                     model_in_channels=model['in_channels']+int(model['use_edge_channel']),out_channels=model['in_channels'])
        if model['arch'] not in ('residual','convnext_v2'):model.pop('blocks_per_stage',None)
        if model['arch'] not in ('unet','unet_v2'):model.pop('skip_mode',None)
        model.pop('latent_channels' if model['arch'] in ('conv','conv_gn') else 'latent_dim',None)
        return TrainConfig.from_dict(dict(model=model,**{key:self._value(widget) for key,widget in self.train_controls.items()})).to_dict()

    def _capabilities(self, *_):
        if self._applying: return
        controls=self.model_controls;train=self.train_controls
        arch=controls['arch'].currentData();vae=controls['model_type'].currentData()=='vae'
        v2=arch=='unet_v2';conv=arch in ('conv','conv_gn');spatial=arch in ('residual','convnext_v2')
        if not (v2 or spatial) and controls['use_edge_channel'].isChecked(): controls['use_edge_channel'].setChecked(False)
        controls['base'].setEnabled(v2 or spatial)
        controls['blocks_per_stage'].setEnabled(spatial)
        controls['skip_mode'].setEnabled(arch in ('unet','unet_v2'))
        controls['bottleneck_dilations'].setEnabled(v2)
        controls['use_edge_channel'].setEnabled(v2 or spatial)
        controls['norm_type'].setEnabled(v2 or arch=='residual' or train['use_gan'].isChecked())
        controls['latent_dim'].setEnabled(conv);controls['latent_channels'].setEnabled(not conv)
        edge=(v2 or spatial) and controls['use_edge_channel'].isChecked()
        for key in ('edge_kernel','median_kernel','edge_lo','edge_hi'): controls[key].setEnabled(edge)
        for key in ('logvar_min','logvar_max'): controls[key].setEnabled(vae)
        for key in ('beta_vae_target','beta_vae_warmup_epochs','kl_reduction'): train[key].setEnabled(vae)
        for key in ('gan_start_epoch','gan_weight','gan_lr','discriminator_base'): train[key].setEnabled(train['use_gan'].isChecked())
        for key in ('translate','degrees','brightness','contrast','hflip','vflip'): train[key].setEnabled(train['augmentation'].isChecked())
        train['ssim_weight'].setEnabled(train['use_ssim'].isChecked())
        l1=train['loss_kind'].currentData()=='l1'
        if not l1 and train['edge_weighted_l1'].isChecked(): train['edge_weighted_l1'].setChecked(False)
        train['edge_weighted_l1'].setEnabled(l1);train['edge_l1_extra_w'].setEnabled(l1 and train['edge_weighted_l1'].isChecked())
        h,w=controls['img_h'].value(),controls['img_w'].value();channels=controls['in_channels'].currentData()
        latent=str(controls['latent_dim'].value()) if conv else f"{controls['latent_channels'].value()}×{h//8}×{w//8}"
        self.io_info.setText(f'Вход: N×{channels+int(edge)}×{h}×{w}; выход: N×{channels}×{h}×{w}; z: {latent}. '
                             + (f" Параметров FC bottleneck ≈ {(3 if vae else 2)*(128 if arch=='conv' else 64)*(h//16)*(w//16)*controls['latent_dim'].value():,}." if conv else '')
                             + ('KL отключен: β=0.' if vae and train['beta_vae_target'].value()==0 else ''))

    def _load_profile(self):
        try: self.apply_config(read_json_object(Path(get_project_root())/'settings/autoencoders'/(self.profile.currentText()+'.json')))
        except (OSError,ValueError) as error: QMessageBox.warning(self,'Профиль AE/VAE',str(error))

    def _import_config(self):
        path,_=sources.get_open_file(self,'ae_training_config')
        if path:
            try: self.apply_config(read_json_object(path))
            except (OSError,ValueError) as error: QMessageBox.warning(self,'Конфигурация',str(error))

    def _export_config(self):
        from PyQt5.QtWidgets import QFileDialog
        try:
            config=self.config()
            path,_=QFileDialog.getSaveFileName(self,'Сохранить параметры AE/VAE','autoencoder.json','JSON (*.json)')
            if path: write_json_atomic(path,config)
        except (OSError,ValueError) as error: QMessageBox.warning(self,'Конфигурация',str(error))

    def _choose_folder(self):
        path=sources.get_directory(self,'folder')
        if path: self.source.setText(path);self._check_source()

    def _choose_yaml(self):
        path,_=sources.get_open_file(self,'ae_images_yaml')
        if path: self.source.setText(path);self._check_source()

    def _choose_project(self):
        path=sources.get_directory(self,'model_export_dir')
        if path: self.project_path.setText(path)

    def _source_changed(self):
        if self._source_job: self._source_job.close();self._source_job=None
        self.source_info.setText('Нажмите «Проверить источники снимков». Разметка не используется; все снимки считаются нормальными.')

    def _check_source(self):
        if self._source_job: self._source_job.close()
        path=self.source.text().strip()
        self.source_info.setText('Поиск снимков…')
        def compute(cancel):
            from utils_new.import_export.yolo_yaml import load_image_splits
            return load_image_splits(path,cancel=cancel)
        job=self._source_job=BackgroundTask(compute)
        def accepted(data):
            if self._source_job is not job or self.source.text().strip()!=path:return
            counts='; '.join(f'{key}: {len(value)}' for key,value in data['splits'].items())
            self.source_info.setText(counts+('; val будет выделен из train' if data['auto_split'] else '')+
                                     '. Все снимки считаются нормальными. TXT и маски не читаются.')
        job.result_ready.connect(accepted)
        def failed(text):
            if self._source_job is job:self.source_info.setText('Ошибка источника: '+text)
        job.failed.connect(failed)
        job.start()

    def _choose_checkpoint(self, resume):
        path,_=sources.get_open_file(self,'model_torch')
        if not path:return
        self._inspect_checkpoint(path,resume)

    def _entered_checkpoint(self, resume):
        path=(self.resume if resume else self.initial_weights).text().strip()
        if path and (resume,path)!=getattr(self,'_checkpoint_key',None):
            self._inspect_checkpoint(path,resume)

    def _resume_state(self):
        locked=bool(self.resume.text().strip())
        for index in range(1,self.pages.count()):self.pages.widget(index).setEnabled(not locked)
        for widget in (self.profile,self.apply_profile,self.import_config):widget.setEnabled(not locked)
        if not locked:self._checkpoint_error=''

    def _inspect_checkpoint(self, path, resume):
        self._checkpoint_key=(resume,path)
        self.resume.setText(path if resume else '')
        self.initial_weights.setText('' if resume else path)
        if self._checkpoint_job:self._checkpoint_job.close()
        self._checkpoint_error=''
        self.checkpoint_info.setText('Чтение метаданных весов…')
        self.options.setEnabled(False);self.run_btn.setEnabled(False)
        def compute(cancel):
            from runtime_torch import require_torch
            require_torch()
            from utils_new.import_export.autoencoder import read_checkpoint
            value=read_checkpoint(path)
            state=value.get('training_state') or {}
            if resume and not state.get('epoch_complete'):
                raise ValueError('Для продолжения нужен last.pt с полной эпохой.')
            return dict(model=value['config'],training=value.get('training_config') or state.get('config'),
                        epoch=value['epoch'],val_loss=value.get('val_loss'),calibration=value.get('calibration'),
                        device=state.get('device'),source=(value.get('data_info') or {}).get('source'))
        job=self._checkpoint_job=BackgroundTask(compute)
        def accepted(meta):
            if self._checkpoint_job is not job:return
            raw=meta['training'] if meta['training'] else self.config()
            raw=dict(raw,model=dict(meta['model']))
            if not resume:raw['model'].update(preprocessing_version='ae-pil-v1',vae_use_mean=True)
            try:self.apply_config(raw)
            except (TypeError,ValueError) as error:
                failed(str(error));return
            if resume:
                folder=Path(path).parent
                if folder.name=='weights':folder=folder.parent
                self.project_path.setText(str(folder.parent));self.experiment_name.setText(folder.name)
                if meta['device'] in ('cpu','cuda:0'):self.device_combo.setCurrentText(meta['device'])
                if not self.source.text().strip() and meta['source']:
                    self.source.setText(meta['source']);self._check_source()
            self.checkpoint_info.setText(f"Эпоха: {meta['epoch']}; val loss: {meta['val_loss']}; "
                                         + ('продолжение той же конфигурации.' if resume else 'новый запуск из начальных весов; пороги будут рассчитаны заново.'))
        def failed(text):
            if self._checkpoint_job is not job:return
            self._checkpoint_error=text;self.checkpoint_info.setText('Ошибка весов: '+text)
        job.result_ready.connect(accepted);job.failed.connect(failed)
        job.settled.connect(lambda:self._checkpoint_settled(job))
        job.start()

    def _checkpoint_settled(self, job):
        if self._checkpoint_job is job:
            self._checkpoint_job=None;self.options.setEnabled(not self._running);self.run_btn.setEnabled(not self._running)

    def get_params(self):
        try:
            if self._checkpoint_error and (self.resume.text() or self.initial_weights.text()):raise ValueError(self._checkpoint_error)
            return training_request(dict(config=self.config(),source=self.source.text().strip(),
                project_path=self.project_path.text().strip(),name=self.experiment_name.text().strip(),
                device=self.device_combo.currentText(),resume=self.resume.text().strip(),
                initial_weights=self.initial_weights.text().strip(),plots=self.plots.isChecked()))
        except (OSError,ValueError) as error:
            QMessageBox.warning(self,'Параметры AE/VAE',str(error));return None

    def set_running(self, running):
        self._running=running
        super().set_running(running)
        self.options.setEnabled(not running)

    def close_jobs(self):
        for name in ('_source_job','_checkpoint_job'):
            job=getattr(self,name,None)
            if job:job.close();setattr(self,name,None)
