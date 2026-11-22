"""Versioned AE checkpoints and run artifacts; safe replacement, no Qt."""

from dataclasses import dataclass
from pathlib import Path
import csv
import math

from utils_new.configuration.autoencoder import normalize_model_config
from utils_new.import_export.atomic import atomic_path
from utils_new.import_export.json_objects import write_json_atomic

SCHEMA_VERSION = 2


def inference_contract(config):
    keys = (
        "arch",
        "model_type",
        "in_channels",
        "model_in_channels",
        "out_channels",
        "img_h",
        "img_w",
        "preprocessing_version",
        "keep_aspect",
        "pad_value",
        "use_edge_channel",
        "edge_kernel",
        "median_kernel",
        "edge_lo",
        "edge_hi",
        "vae_use_mean",
        "logvar_min",
        "logvar_max",
    )
    keys += (
        ("latent_dim",)
        if config["arch"] in ("conv", "conv_gn")
        else ("latent_channels",)
    )
    if config["arch"] == "unet_v2":
        keys += ("base", "norm_type", "bottleneck_dilations")
    return {
        key: list(config[key]) if isinstance(config[key], tuple) else config[key]
        for key in keys
    }


def validate_calibration(value, config):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or value.get("metric") != "mae"
    ):
        raise ValueError("Неизвестный формат калибровки AE.")
    if value.get("contract") != inference_contract(config):
        raise ValueError("Калибровка не соответствует модели, предобработке или режиму VAE.")
    if (
        value.get("image_aggregation") != "mean_valid_pixels"
        or value.get("pixel_aggregation") != "valid_pixels"
    ):
        raise ValueError("Несовместимый способ расчёта порогов AE.")
    if config["model_type"] == "vae" and not config["vae_use_mean"]:
        raise ValueError("Калибровка требует детерминированный VAE.")
    for key in ("image_threshold", "pixel_threshold"):
        threshold = value.get(key)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (float, int))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ValueError(f"Некорректный {key}.")
    for key in ("image_percentile", "pixel_percentile"):
        percentile = value.get(key)
        if (
            not isinstance(percentile, (int, float))
            or not math.isfinite(percentile)
            or not 0 <= percentile <= 100
        ):
            raise ValueError(f"Некорректный {key}.")
    if type(value.get("images")) is not int or value["images"] < 1:
        raise ValueError("Калибровочная выборка пуста.")
    return value


def save_checkpoint(
    path,
    model,
    config,
    *,
    epoch=0,
    calibration=None,
    training_state=None,
    val_loss=None,
):
    import torch

    config = normalize_model_config(config, arch=config.get("arch"))
    validate_calibration(calibration, config)
    payload = dict(
        schema_version=SCHEMA_VERSION,
        config=config,
        model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        epoch=int(epoch),
        calibration=calibration,
        val_loss=val_loss,
    )
    if training_state is not None:
        payload["training_state"] = training_state
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def read_checkpoint(path):
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Torch versions predating weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("model"), dict)
        or not isinstance(checkpoint.get("config"), dict)
    ):
        raise ValueError("В checkpoint нужны поля 'model' и 'config'.")
    schema = checkpoint.get("schema_version")
    if schema not in (None, SCHEMA_VERSION):
        raise ValueError(f"Неизвестная версия AE checkpoint: {schema}")
    from utils_new.autoencoders.models import detect_arch

    config = checkpoint["config"]
    arch = detect_arch(config, state_dict=checkpoint["model"])
    if schema == SCHEMA_VERSION and config.get("arch") != arch:
        raise ValueError("Архитектура config не совпадает с весами.")
    config = normalize_model_config(config, arch=arch)
    if schema == SCHEMA_VERSION:
        calibration = validate_calibration(checkpoint.get("calibration"), config)
    else:
        # Earlier scripts calibrated a scalar per image (MSE or L1). It cannot
        # be compared to individual pixels in the current MAE error map.
        calibration = None
    return dict(
        checkpoint, config=config, calibration=calibration, legacy=(schema is None)
    )


@dataclass
class LoadedAutoencoder:
    model: object
    config: dict
    arch: str
    calibration: dict | None
    legacy: bool


def load_autoencoder(path, device="cpu"):
    from utils_new.autoencoders.models import build_ae_model_from_cfg

    checkpoint = read_checkpoint(path)
    config = checkpoint["config"]
    model = build_ae_model_from_cfg(config, arch=config["arch"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return LoadedAutoencoder(
        model, config, config["arch"], checkpoint["calibration"], checkpoint["legacy"]
    )


def save_history(directory, history, *, plot=True):
    directory = Path(directory)
    write_json_atomic(directory / "history.json", dict(epochs=history))
    with atomic_path(directory / "results.csv") as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            keys = list(history[0]) if history else ["epoch"]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
    if plot and history:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        figure = Figure(figsize=(10, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [r["epoch"] for r in history],
            [r["train_loss"] for r in history],
            label="train",
        )
        axis.plot(
            [r["epoch"] for r in history], [r["val_loss"] for r in history], label="val"
        )
        axis.legend()
        axis.grid(True)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        with atomic_path(directory / "loss.png") as temporary:
            figure.savefig(temporary, format="png", dpi=120)


def save_reconstruction_examples(path, model, dataset, device):
    import numpy as np
    import torch
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    count = min(6, len(dataset))
    figure = Figure(figsize=(2.2 * count, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, count, squeeze=False)
    model.eval()
    with torch.inference_mode():
        for column in range(count):
            item = dataset[column]
            output = model(item["input"][None].to(device))[0][0].cpu()
            target = item["target"]
            error = (output - target).abs().mean(0) * item["valid"][0]
            for row, image in enumerate((target, output, error)):
                array = image.numpy()
                if array.ndim == 3:
                    array = (
                        array[0] if array.shape[0] == 1 else np.moveaxis(array, 0, -1)
                    )
                axes[row, column].imshow(
                    array, cmap="hot" if row == 2 else "gray", vmin=0, vmax=1
                )
                axes[row, column].axis("off")
    with atomic_path(path) as temporary:
        figure.savefig(temporary, format="png", dpi=120)
