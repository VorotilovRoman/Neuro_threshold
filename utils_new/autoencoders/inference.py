"""Validated reconstruction, pixel anomaly maps and image scores; no Qt."""

import cv2
import numpy as np
import torch
from utils_new.autoencoders.preprocessing import preprocess_image_for_ae


def infer_autoencoder(image, model, cfg, device, cancel=None):
    """Validated reconstruction/error on the model grid; no widgets or Qt."""
    from concurrent.futures import CancelledError

    def checkpoint():
        if cancel is not None and cancel.is_set():
            raise CancelledError()

    checkpoint()
    model.to(device)
    model.eval()
    model.deterministic_eval = bool(cfg.get("vae_use_mean", False))
    x, valid_region = preprocess_image_for_ae(image, cfg)
    with torch.inference_mode():
        output = model(x.unsqueeze(0).to(device))
        reconstruction = output[0] if isinstance(output, (tuple, list)) else output
        if (
            not isinstance(reconstruction, torch.Tensor)
            or reconstruction.ndim != 4
            or reconstruction.shape[0] != 1
        ):
            raise ValueError(
                "AE/VAE должен вернуть Tensor [1,C,H,W] либо tuple с таким первым элементом."
            )
        in_channels = int(cfg.get("in_channels", 3))
        expected = int(cfg.get("out_channels", in_channels))
        if (
            expected not in (1, 3)
            or reconstruction.shape[1] != expected
            or expected > in_channels
            or reconstruction.shape[2:] != x.shape[1:]
        ):
            raise ValueError(
                f"Выход AE/VAE {tuple(reconstruction.shape)} не соответствует image-каналам входа {tuple(x.shape)} и out_channels={expected}."
            )
        reconstruction = reconstruction.float().cpu()
        if not torch.isfinite(reconstruction).all():
            raise ValueError("Реконструкция содержит NaN/Inf.")
        error = (
            (x[:expected].unsqueeze(0).cpu() - reconstruction)
            .abs()
            .mean(dim=1)[0]
            .numpy()
        )
    checkpoint()
    if not np.isfinite(error).all():
        raise ValueError("Карта ошибки содержит NaN/Inf.")
    x0, y0, x1, y1 = valid_region
    # Display the image channels only, without letterbox padding. Keep the
    # unmodified float output above for MAE; clipping is display-only.
    reconstructed = reconstruction[0, :, y0:y1, x0:x1].permute(1, 2, 0).numpy()
    reconstructed = np.rint(np.clip(reconstructed, 0, 1) * 255).astype(np.uint8)
    reconstructed = cv2.cvtColor(
        reconstructed,
        cv2.COLOR_GRAY2BGR if expected == 1 else cv2.COLOR_RGB2BGR,
    )
    reconstructed = cv2.resize(
        reconstructed, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    checkpoint()
    return {
        "reconstruction_bgr": reconstructed,
        "image_score": float(error[y0:y1, x0:x1].mean()),
        "error_metric": "mae",
        "error": error,
        "valid_region": valid_region,
        "input_shape": tuple(x.shape),
        "output_shape": tuple(reconstruction.shape),
        "device": str(device),
    }
