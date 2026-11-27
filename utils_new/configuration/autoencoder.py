"""Validated AE model and training configurations, independent of Torch and Qt."""

from dataclasses import asdict, dataclass, field, fields
from math import isfinite

PREPROCESSING_VERSIONS = ("ae-pil-v1", "legacy-opencv-v1")
ARCHITECTURES = ("conv", "conv_gn", "unet", "unet_v2", "residual", "convnext_v2")
AE_PROFILES = ('conv_bn_baseline', 'conv_bn', 'conv_gn', 'unet', 'unet_v2_edge_gan',
               'residual_ae', 'residual_vae', 'convnext_v2_ae', 'convnext_v2_vae')


def _integer(value, name, low=1):
    if type(value) is not int or value < low:
        raise ValueError(f"{name}: требуется целое число ≥ {low}.")
    return value


def _number(value, name, low=0, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not isfinite(value)
        or value < low
        or (high is not None and value > high)
    ):
        raise ValueError(f"{name}: недопустимое значение {value!r}.")
    return value


def normalize_model_config(values, *, arch=None):
    if not isinstance(values, dict):
        raise ValueError("Config модели должен быть объектом.")
    cfg = dict(values)
    cfg["arch"] = arch or cfg.get("arch", "conv")
    if cfg["arch"] not in ARCHITECTURES or cfg.get("model_type") not in ("ae", "vae"):
        raise ValueError("Неизвестная архитектура или model_type AE/VAE.")
    if cfg['arch'] in ('residual', 'convnext_v2'):
        cfg.setdefault('blocks_per_stage', 2)
        _integer(cfg['blocks_per_stage'], 'blocks_per_stage')
    if cfg['arch'] in ('unet', 'unet_v2'):
        cfg.setdefault('skip_mode', 'full')
        if cfg['skip_mode'] not in ('full', 'none'):
            raise ValueError('skip_mode: full или none.')
    stride = 16 if cfg["arch"] in ("conv", "conv_gn") else 8
    for key in ("img_h", "img_w"):
        if _integer(cfg.get(key), key, stride) % stride:
            raise ValueError(f"{key} должен быть кратным {stride}.")
    if type(cfg.get("in_channels")) is not int or cfg["in_channels"] not in (1, 3):
        raise ValueError("in_channels должен быть 1 или 3.")
    defaults = dict(
        keep_aspect=True,
        pad_value=0,
        model_in_channels=cfg["in_channels"] + int(cfg.get("use_edge_channel", False)),
        out_channels=cfg["in_channels"],
        use_edge_channel=cfg.get("model_in_channels", cfg["in_channels"])
        > cfg["in_channels"],
        preprocessing_version="legacy-opencv-v1",
        vae_use_mean=False,
        norm_type="gn",
        base=32,
        bottleneck_dilations=(2, 4),
        edge_kernel="scharr",
        median_kernel=3,
        edge_lo=0.10,
        edge_hi=0.35,
        logvar_min=-10.0,
        logvar_max=10.0,
    )
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    if cfg["preprocessing_version"] not in PREPROCESSING_VERSIONS:
        raise ValueError("Неизвестная версия предобработки AE.")
    for key in ("keep_aspect", "use_edge_channel", "vae_use_mean"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} должен быть bool.")
    _integer(cfg["pad_value"], "pad_value", 0)
    if cfg["pad_value"] > 255:
        raise ValueError("pad_value должен быть 0..255.")
    _integer(cfg["base"], "base")
    latent = "latent_dim" if cfg["arch"] in ("conv", "conv_gn") else "latent_channels"
    _integer(cfg.get(latent), latent)
    _integer(cfg["model_in_channels"], "model_in_channels")
    _integer(cfg["out_channels"], "out_channels")
    edge_channels = cfg["model_in_channels"] - cfg["in_channels"]
    if edge_channels not in (0, 1) or bool(edge_channels) != cfg["use_edge_channel"]:
        raise ValueError("Число входных каналов не согласовано с use_edge_channel.")
    if (cfg["use_edge_channel"] and cfg["arch"] not in ("unet_v2", "residual", "convnext_v2")) or cfg[
        "out_channels"
    ] != cfg["in_channels"]:
        raise ValueError(
            "Edge поддерживают unet_v2/residual/convnext_v2; выход содержит только каналы снимка."
        )
    if cfg["norm_type"] not in ("gn", "bn") or cfg["edge_kernel"] not in (
        "scharr",
        "sobel",
    ):
        raise ValueError("Неизвестная нормализация или edge_kernel.")
    _integer(cfg["median_kernel"], "median_kernel")
    if cfg["median_kernel"] % 2 != 1 or cfg["median_kernel"] // 2 >= min(
        cfg["img_h"], cfg["img_w"]
    ):
        raise ValueError("median_kernel должен быть нечётным и меньше размера снимка.")
    _number(cfg["edge_lo"], "edge_lo")
    _number(cfg["edge_hi"], "edge_hi")
    if cfg["edge_lo"] >= cfg["edge_hi"]:
        raise ValueError("Нужно edge_lo < edge_hi.")
    for key in ("logvar_min", "logvar_max"):
        _number(cfg[key], key, -80, 80)
    if cfg["logvar_min"] >= cfg["logvar_max"]:
        raise ValueError("Нужно logvar_min < logvar_max.")
    if (
        not isinstance(cfg["bottleneck_dilations"], (list, tuple))
        or not cfg["bottleneck_dilations"]
    ):
        raise ValueError("Нужен непустой список bottleneck_dilations.")
    cfg["bottleneck_dilations"] = tuple(
        _integer(d, "dilation") for d in cfg["bottleneck_dilations"]
    )
    return cfg


@dataclass
class TrainConfig:
    model: dict = field(default_factory=dict)
    epochs: int = 120
    batch_size: int = 16
    lr: float = 0.0003
    weight_decay: float = 0.00001
    min_lr: float = 0.00001
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 0
    patience: int = 25
    noise_std: float = 0.05
    augmentation: bool = True
    translate: float = 0.03
    degrees: float = 2.0
    brightness: float = 0.15
    contrast: float = 0.15
    hflip: bool = True
    vflip: bool = False
    loss_kind: str = "l1"
    l1_weight: float = 0.7
    use_ssim: bool = True
    ssim_weight: float = 0.35
    grad_loss_w: float = 2.0
    edge_weighted_l1: bool = True
    edge_l1_extra_w: float = 3.0
    beta_vae_target: float = 0.0
    beta_vae_warmup_epochs: int = 60
    kl_reduction: str = "mean"
    use_gan: bool = True
    gan_start_epoch: int = 60
    gan_weight: float = 0.05
    gan_lr: float = 0.0001
    discriminator_base: int = 32
    grad_clip: float = 5.0
    image_percentile: float = 99.0
    pixel_percentile: float = 99.5
    calibration_pixels: int = 1000000

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Конфигурация обучения должна быть объектом.")
        unknown = values.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Неизвестные параметры обучения: {sorted(unknown)}")
        config = cls(**values)
        config.model = normalize_model_config(config.model)
        if config.model["preprocessing_version"] != "ae-pil-v1":
            raise ValueError("Новое обучение использует общую предобработку ae-pil-v1.")
        if config.model["model_type"] == "vae" and not config.model["vae_use_mean"]:
            raise ValueError(
                "Для воспроизводимой калибровки VAE нужен vae_use_mean=True."
            )
        for key in (
            "epochs",
            "batch_size",
            "patience",
            "discriminator_base",
            "calibration_pixels",
        ):
            _integer(getattr(config, key), key)
        for key in ("seed", "num_workers", "gan_start_epoch", "beta_vae_warmup_epochs"):
            _integer(getattr(config, key), key, 0)
        for key in (
            "lr",
            "weight_decay",
            "min_lr",
            "noise_std",
            "translate",
            "degrees",
            "brightness",
            "contrast",
            "l1_weight",
            "ssim_weight",
            "grad_loss_w",
            "edge_l1_extra_w",
            "beta_vae_target",
            "gan_weight",
            "gan_lr",
            "grad_clip",
        ):
            _number(getattr(config, key), key)
        if (
            config.lr <= 0
            or config.gan_lr <= 0
            or config.min_lr > config.lr
            or config.grad_clip <= 0
        ):
            raise ValueError("Недопустимая скорость обучения или grad_clip.")
        _number(config.val_split, "val_split", 0, 1)
        if not 0 < config.val_split < 1:
            raise ValueError("val_split должен быть между 0 и 1.")
        if config.translate > 1:
            raise ValueError("translate должен быть 0..1.")
        for key in ("image_percentile", "pixel_percentile"):
            _number(getattr(config, key), key, 0, 100)
        for key in (
            "augmentation",
            "hflip",
            "vflip",
            "use_ssim",
            "edge_weighted_l1",
            "use_gan",
        ):
            if type(getattr(config, key)) is not bool:
                raise ValueError(f"{key} должен быть bool.")
        if config.loss_kind not in ("l1", "mse") or config.kl_reduction not in (
            "mean",
            "sum_per_sample",
        ):
            raise ValueError("Неизвестная функция потерь или нормировка KL.")
        if config.edge_weighted_l1 and config.loss_kind != "l1":
            raise ValueError("edge_weighted_l1 применяется только к L1.")
        if config.use_gan and min(config.model["img_h"], config.model["img_w"]) < 32:
            raise ValueError("PatchGAN требует минимум 32×32.")
        if (
            config.l1_weight == 0
            and config.grad_loss_w == 0
            and (not config.use_ssim or config.ssim_weight == 0)
        ):
            raise ValueError("Нужна ненулевая функция реконструкции.")
        return config

    def to_dict(self):
        return asdict(self)


def training_request(values):
    """GUI/worker envelope; the same TrainConfig remains usable from the CLI."""
    from pathlib import Path

    allowed = {'config', 'source', 'project_path', 'name', 'device', 'resume', 'initial_weights', 'plots'}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError('Неизвестные параметры запуска AE/VAE.')
    config = TrainConfig.from_dict(values.get('config', {})).to_dict()
    source = Path(values.get('source') or '').expanduser().resolve()
    if not values.get('source') or not source.exists():
        raise ValueError('Выберите папку нормальных снимков или YAML.')
    if not source.is_dir() and source.suffix.lower() not in ('.yaml', '.yml'):
        raise ValueError('Для AE/VAE поддерживается папка или YOLO/MASKS YAML.')
    name = values.get('name') or 'ae'
    if name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Название запуска должно быть именем одной подпапки.')
    device = values.get('device', 'auto')
    if device not in ('auto', 'cpu', 'cuda:0'):
        raise ValueError('Выберите auto, cpu или cuda:0.')
    result = dict(config=config, source=str(source), name=name, device=device,
                  project_path=str(Path(values.get('project_path') or 'runs/autoencoder').expanduser().resolve()),
                  plots=values.get('plots', True))
    if type(result['plots']) is not bool:
        raise ValueError('plots должен быть bool.')
    for key in ('resume', 'initial_weights'):
        path = values.get(key) or ''
        if path and not Path(path).is_file():
            raise ValueError(f'{key}: файл не найден: {path}')
        result[key] = str(Path(path).resolve()) if path else ''
    if result['resume'] and result['initial_weights']:
        raise ValueError('Выберите либо продолжение last.pt, либо начальные веса нового запуска.')
    return result
