"""Validated semantic-model training settings, independent of Torch and Qt."""

from copy import deepcopy
from math import isfinite
from pathlib import Path


def _int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}: требуется целое число ≥ {minimum}.")
    return value


def _float(value, name, minimum=0.0, maximum=None):
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name}: недопустимое значение {value!r}.")
    return value


def semantic_config(params, kind):
    if kind not in ("unet", "deeplab", "segformer") or not isinstance(params, dict):
        raise ValueError("Нужны параметры U-Net, DeepLab или SegFormer.")
    defaults = dict(
        dataset_path=None,
        model_path=None,
        project_path=None,
        name=None,
        architecture={"unet":"unet", "deeplab":"deeplabv3plus", "segformer":"segformer"}[kind],
        encoder="mit_b0" if kind == "segformer" else "resnet50",
        backbone="resnet50",
        input_size=512,
        num_classes=2,
        optimizer="AdamW" if kind == "segformer" else "Adam" if kind == "unet" else "SGD",
        loss="BCE+Dice" if kind == "unet" else "CrossEntropy",
        lr=0.00006 if kind == "segformer" else 0.0001 if kind == "unet" else 0.007,
        batch_size=8,
        auto_batch=False,
        epochs=100,
        dropout=0.0,
        scheduler="None" if kind == "unet" else "Poly",
        augmentations={},
        device="auto",
        threshold=0.5,
        num_workers=0,
        normalize=(kind == "segformer"),
        resize_mode="hard",
        ignore_index=None,
        decoder_channels=[256, 128, 64, 32, 16],
        pretrained=True,
        encoder_weights="imagenet",
        weight_decay=0.0001,
        momentum=0.9,
        unet_yaml=None,
        deeplab_yaml=None,
        output_stride=16,
        atrous_rates=None,
        class_weights=None,
        loss_weights={"ce": 1.0, "dice": 1.0, "focal": 1.0},
        two_phase_training=False,
        freeze_epochs=0,
        unfreeze_epochs=0,
        unfreeze_last_n_stages=None,
        freeze_bn_second_phase=True,
        use_amp=(kind != "unet"),
        accum_steps=1,
        use_ema=(kind == "deeplab"),
        ema_decay=0.999,
        use_layerwise_lr=False,
        layerwise_decay=0.1,
        poly_power=0.9,
        multi_scale_infer=False,
        infer_scales=[0.75, 1.0, 1.25],
        seed=42,
        patience=5 if kind == "unet" else 15,
        grad_clip=5.0,
        plots=True,
        pr_pixels=200000,
        output_activation="argmax",
    )
    if kind == "segformer":
        defaults["decoder_segmentation_channels"] = 256
    cfg = deepcopy(defaults)
    cfg.update(deepcopy(params))
    cfg.pop('export_onnx', None)
    cfg["kind"] = kind
    aliases = {"unet++": "unetplusplus", "deeplabv3+": "deeplabv3plus"}
    cfg["architecture"] = aliases.get(
        str(cfg["architecture"]).lower(), str(cfg["architecture"]).lower()
    )
    if cfg["architecture"] not in (
        {"unet": ("unet", "unetplusplus"), "deeplab": ("deeplabv3", "deeplabv3plus"),
         "segformer": ("segformer",)}[kind]
    ):
        raise ValueError("Архитектура не соответствует воркеру.")
    if kind == "segformer":
        if cfg["encoder"] not in tuple(f"mit_b{i}" for i in range(6)):
            raise ValueError("SegFormer: выберите encoder mit_b0…mit_b5.")
        _int(cfg["decoder_segmentation_channels"], "decoder_segmentation_channels")
        cfg["backbone"] = cfg["encoder"]
    for key in (
        "input_size",
        "num_classes",
        "epochs",
        "accum_steps",
        "patience",
        "pr_pixels",
    ):
        _int(cfg[key], key)
    if cfg["input_size"] < 16:
        raise ValueError("input_size должен быть не менее 16.")
    for key in ("num_workers", "seed", "freeze_epochs", "unfreeze_epochs"):
        _int(cfg[key], key, 0)
    if cfg["batch_size"] != -1:
        _int(cfg["batch_size"], "batch_size")
    if type(cfg["batch_size"]) is not int:
        raise ValueError("batch_size должен быть целым.")
    for key in (
        "lr",
        "weight_decay",
        "momentum",
        "dropout",
        "threshold",
        "ema_decay",
        "layerwise_decay",
        "poly_power",
        "grad_clip",
    ):
        _float(cfg[key], key)
    if (
        cfg["lr"] <= 0
        or cfg["dropout"] >= 1
        or cfg["threshold"] > 1
        or cfg["ema_decay"] >= 1
        or not 0 < cfg["layerwise_decay"] <= 1
        or cfg["grad_clip"] <= 0
    ):
        raise ValueError(
            "Некорректные LR, dropout, threshold, EMA, layerwise_decay или grad_clip."
        )
    for key in (
        "auto_batch",
        "normalize",
        "pretrained",
        "two_phase_training",
        "freeze_bn_second_phase",
        "use_amp",
        "use_ema",
        "use_layerwise_lr",
        "multi_scale_infer",
        "plots",
    ):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} должен быть bool.")
    if cfg["ignore_index"] is not None and type(cfg["ignore_index"]) is not int:
        raise ValueError("ignore_index должен быть целым или null.")
    if cfg["resize_mode"] not in ("hard", "keep_aspect_pad", "random_crop"):
        raise ValueError("Неизвестный resize_mode.")
    if cfg["two_phase_training"]:
        if cfg["freeze_epochs"] + cfg["unfreeze_epochs"] < 1:
            raise ValueError("Длительность фаз должна быть положительной.")
        cfg["epochs"] = cfg["freeze_epochs"] + cfg["unfreeze_epochs"]
    if cfg["unfreeze_last_n_stages"] is not None:
        _int(cfg["unfreeze_last_n_stages"], "unfreeze_last_n_stages", 0)
    if cfg["output_stride"] not in (8, 16):
        raise ValueError("SMP DeepLab поддерживает output_stride 8 или 16.")
    if cfg["atrous_rates"] is None:
        cfg["atrous_rates"] = {8: [12, 24, 36], 16: [6, 12, 18], 32: [3, 6, 9]}[
            cfg["output_stride"]
        ]
    if (
        not isinstance(cfg["atrous_rates"], (list, tuple))
        or len(cfg["atrous_rates"]) != 3
    ):
        raise ValueError("Нужны три atrous_rates.")
    cfg["atrous_rates"] = [_int(v, "atrous_rate") for v in cfg["atrous_rates"]]
    if (
        not isinstance(cfg["decoder_channels"], (list, tuple))
        or len(cfg["decoder_channels"]) != 5
    ):
        raise ValueError("Нужны пять decoder_channels.")
    cfg["decoder_channels"] = [
        _int(v, "decoder_channels") for v in cfg["decoder_channels"]
    ]
    scales = cfg["infer_scales"]
    if isinstance(scales, str):
        scales = [float(s.strip()) for s in scales.split(",") if s.strip()]
    if not isinstance(scales, (list, tuple)) or not scales:
        raise ValueError("infer_scales не должен быть пустым.")
    cfg["infer_scales"] = [_float(s, "infer_scale", 0.01, 10.0) for s in scales]
    if not isinstance(cfg["augmentations"], dict) or any(
        k not in ("hflip", "vflip", "rotate", "scale", "noise") or type(v) is not bool
        for k, v in cfg["augmentations"].items()
    ):
        raise ValueError("Некорректные аугментации.")
    if not isinstance(cfg["loss_weights"], dict) or any(
        k not in ("ce", "dice", "focal") for k in cfg["loss_weights"]
    ):
        raise ValueError("Некорректные веса loss.")
    for key, value in cfg["loss_weights"].items():
        _float(value, key)
    for key in ("unet_yaml", "deeplab_yaml", "model_path"):
        if cfg[key] and not Path(cfg[key]).is_file():
            raise FileNotFoundError(f"{key}: файл не найден: {cfg[key]}")
    return cfg


import math
import re


def is_standard_yolo_model(model_path):
    """Allow downloads only for a known bare model name, never a missing user path."""
    return (
        re.fullmatch(
            r"(?:yolov8|yolo11|yolo26)[nslmx](?:-seg|-obb|-pose|-cls)?\.pt",
            str(model_path),
        )
        is not None
    )


YOLO_EXTRA_ARGUMENTS = {'seed','patience','weight_decay','momentum','warmup_epochs','fliplr','flipud',
                        'degrees','translate','scale','shear','perspective','hsv_h','hsv_s','hsv_v',
                        'mixup','copy_paste','amp','plots','pretrained','workers'}


def validate_yolo_options(params):
    from math import isfinite
    for key,value in params.items():
        if key not in YOLO_EXTRA_ARGUMENTS:raise ValueError(f'Неизвестный параметр YOLO: {key}')
        if key in ('amp','plots','pretrained'):
            if type(value) is not bool:raise ValueError(f'{key}: требуется bool.')
        elif type(value) not in (int,float) or not isfinite(value) or value<0:
            raise ValueError(f'{key}: требуется конечное неотрицательное число.')
        elif key in ('seed','patience','workers') and type(value) is not int:
            raise ValueError(f'{key}: требуется int.')
        elif key in ('fliplr','flipud','translate','scale','perspective','hsv_h','hsv_s','hsv_v','mixup','copy_paste','momentum') and value>1:
            raise ValueError(f'{key}: требуется значение 0..1.')
    return params


def build_train_args(params):
    """Validate before loading CUDA. Do not estimate or replace the user's batch."""
    if not isinstance(params, dict):
        raise TypeError("Параметры должны быть JSON-объектом.")
    for name in ("model", "data"):
        if not isinstance(params.get(name), str) or not params[name].strip():
            raise ValueError(f"Не указан строковый параметр '{name}'.")
    data = Path(params["data"])
    task = params.get('task') or ('classify' if str(params['model']).endswith('-cls.pt') else
                                'pose' if str(params['model']).endswith('-pose.pt') else None)
    if task is not None and task not in ('detect','segment','obb','pose','classify'):
        raise ValueError('Неизвестная задача YOLO.')
    if task == 'classify':
        if not data.is_dir() or not (data/'train').is_dir() or not (data/'val').is_dir():
            raise ValueError('Classification требует корень с train/ и val/, внутри — папки классов.')
        classes = sorted(p.name for p in (data/'train').iterdir() if p.is_dir())
        if not classes or sorted(p.name for p in (data/'val').iterdir() if p.is_dir()) != classes:
            raise ValueError('Classification: папки классов train/val должны совпадать.')
    else:
        if data.is_dir():data /= "data.yaml"
        if not data.is_file():raise ValueError(f"Файл датасета не найден: {data}")
    if task == 'pose':
        import yaml
        from utils_new.import_export.task_datasets import schema_from_yaml
        schema = schema_from_yaml(yaml.safe_load(data.read_text(encoding='utf-8-sig')))
        if schema.flip_idx is None and not params.get('disable_aug',False) and params.get('fliplr',.5)>0:
            raise ValueError('Pose: задайте flip_idx в YAML либо fliplr=0 в дополнительных параметрах.')
    if not Path(params["model"]).is_file() and not is_standard_yolo_model(
        params["model"]
    ):
        raise ValueError(f"Файл модели не найден: {params['model']}")
    args = {
        "data": str(data),
        "epochs": params.get("epochs", 100),
        "imgsz": params.get("imgsz", 640),
        "batch": params.get("batch", 16),
        "optimizer": params.get("optimizer", "auto"),
        "lr0": params.get("lr0", 0.01),
        "lrf": params.get("lrf", 0.01),
        "verbose": True,
    }
    for name in ("epochs", "imgsz"):
        if type(args[name]) is not int or args[name] < 1:
            raise ValueError(f"{name} должен быть положительным целым числом.")
    batch = args["batch"]
    if not (
        type(batch) is int
        and (batch == -1 or batch > 0)
        or type(batch) is float
        and math.isfinite(batch)
        and 0 < batch < 1
    ):
        raise ValueError(
            "batch: -1 для AutoBatch, положительное целое или доля памяти 0 < batch < 1."
        )
    for name in ("lr0", "lrf"):
        value = args[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} должен быть конечным неотрицательным числом.")
    for name in ("amp", "plots", "pretrained", "disable_aug"):
        if name in params and type(params[name]) is not bool:
            raise ValueError(f"{name} должен быть bool.")
    for name in ("close_mosaic", "workers"):
        if name in params and params[name] is not None:
            if type(params[name]) is not int or params[name] < 0:
                raise ValueError(f"{name} должен быть неотрицательным целым числом.")
            args[name] = params[name]
    if params.get("mosaic") is not None:
        value = params["mosaic"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("mosaic должен быть в диапазоне [0, 1].")
        args["mosaic"] = value
    for name in ("amp", "plots", "pretrained"):
        if name in params:
            args[name] = params[name]
    for name in ("project", "name"):
        if params.get(name):
            args[name] = params[name]
    extra={name:params[name] for name in YOLO_EXTRA_ARGUMENTS if name in params}
    args.update(validate_yolo_options(extra))
    if params.get('task') is not None:args['task']=task
    if params.get("device") not in (None, "", "auto"):
        args["device"] = params["device"]
    if params.get("disable_aug", False):
        args.update(
            fliplr=0.0,
            flipud=0.0,
            scale=0.0,
            translate=0.0,
            shear=0.0,
            perspective=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            copy_paste=0.0,
            mixup=0.0,
            erasing=0.0,
            auto_augment=None,
            mosaic=0.0,
            degrees=0.0,
            cutmix=0.0,
            close_mosaic=0,
        )
    return args
