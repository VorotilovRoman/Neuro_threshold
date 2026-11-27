"""Preset collection persistence; method schemas belong to segmentation.presets."""

from utils_new.import_export.json_objects import read_json_object, write_json_atomic


def read_presets(path):
    return read_json_object(path)


def write_presets(path, presets):
    if not isinstance(presets, dict):
        raise TypeError("Preset collection must be a JSON object")
    write_json_atomic(path, presets)


TRAINING_WORKER_DEFAULTS = {
    'yolo': dict(amp=True, plots=True, pretrained=True, workers=0, seed=42, patience=50,
                 weight_decay=.0005, momentum=.937, warmup_epochs=3., fliplr=.5, flipud=0.,
                 degrees=0., translate=.1, scale=.5, shear=0., perspective=0.,
                 hsv_h=.015, hsv_s=.7, hsv_v=.4, mixup=0., copy_paste=0.),
    'unet': dict(num_workers=0, seed=42, patience=25, grad_clip=5., plots=True,
                 pretrained=True, encoder_weights='imagenet', weight_decay=.0001,
                 ignore_index=None, decoder_channels=[256,128,64,32,16],use_amp=True),
    'deeplab': dict(num_workers=0, seed=42, patience=25, grad_clip=5., plots=True,
                    pretrained=True, encoder_weights='imagenet',pr_pixels=200000),
    'sam': dict(weight_decay=.0001,box_jitter=.05,validation_fraction=.2,grad_clip=1.),
    'segformer': dict(num_workers=0, seed=42, patience=25, grad_clip=1., plots=True,
                      encoder_weights='imagenet', weight_decay=.01, momentum=.9,
                      use_ema=False, ema_decay=.999, pr_pixels=200000, poly_power=.9,
                      use_layerwise_lr=False, layerwise_decay=.1),
}


def training_worker_defaults(family):
    """Actual backend defaults for saving an unmodified form, not suggested recipes."""
    from copy import deepcopy
    keys=TRAINING_WORKER_DEFAULTS[family]
    if family in ('unet','deeplab','segformer'):
        from utils_new.configuration.training import semantic_config
        actual=semantic_config({},family)
        return {key:deepcopy(actual[key]) for key in keys}
    if family=='yolo':
        from importlib.util import find_spec
        from pathlib import Path
        import yaml
        spec=find_spec('ultralytics')
        if spec is None:raise ValueError('Для сохранения defaults YOLO установите ultralytics.')
        actual=yaml.safe_load((Path(spec.origin).parent/'cfg/default.yaml').read_text(encoding='utf-8'))
        return {key:deepcopy(actual[key]) for key in keys}
    return deepcopy(keys)


def validate_training_preset(value, family, fields, required):
    """Complete form snapshots; resource paths are optional, not partial recipes."""
    if not isinstance(value,dict) or value.get('schema_version')!=1 or value.get('family')!=family:
        raise ValueError('Неверная версия или семейство пресета обучения.')
    if set(value)-{'schema_version','family','id','title','description','settings','worker_parameters'}:
        raise ValueError('Неизвестные поля пресета обучения.')
    settings=value.get('settings')
    if not isinstance(settings,dict) or set(settings)-set(fields) or set(required)-set(settings):
        raise ValueError('Пресет должен содержать все поля конфигурации своей формы; пути необязательны.')
    worker=value.get('worker_parameters',{})
    if not isinstance(worker,dict) or set(worker)!=set(TRAINING_WORKER_DEFAULTS[family]):
        raise ValueError('Пресет должен содержать полный набор поддерживаемых дополнительных параметров воркера.')
    return value
