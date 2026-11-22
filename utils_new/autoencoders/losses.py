"""AE reconstruction objectives with explicit KL reduction and valid-pixel masks."""

import torch
from torch import nn
from torch.nn import functional as F

from utils_new.autoencoders.preprocessing import compute_edge_map_torch


def masked_mean(values, valid):
    selected = valid.expand_as(values)
    return (values * selected).sum() / selected.sum().clamp_min(1)


class SSIMLoss(nn.Module):
    def __init__(self, channels, window_size=11):
        super().__init__()
        axis = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-(axis**2) / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        window = (gaussian[:, None] * gaussian[None, :])[None, None]
        self.register_buffer(
            "window", window.expand(channels, 1, window_size, window_size).contiguous()
        )
        self.padding = window_size // 2

    def forward(self, first, second, valid):
        def smooth(value):
            return F.conv2d(
                value, self.window, padding=self.padding, groups=first.shape[1]
            )

        mu1, mu2 = smooth(first), smooth(second)
        sigma1 = smooth(first * first) - mu1**2
        sigma2 = smooth(second * second) - mu2**2
        sigma12 = smooth(first * second) - mu1 * mu2
        similarity = ((2 * mu1 * mu2 + 0.01**2) * (2 * sigma12 + 0.03**2)) / (
            (mu1**2 + mu2**2 + 0.01**2) * (sigma1 + sigma2 + 0.03**2)
        )
        return masked_mean(1 - similarity, valid)


def kl_loss(mu, logvar, *, reduction="mean", logvar_min=-10.0, logvar_max=10.0):
    logvar = logvar.clamp(logvar_min, logvar_max)
    values = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    if reduction == "mean":
        return values.mean()
    if reduction == "sum_per_sample":
        return values.flatten(1).sum(1).mean()
    raise ValueError("Неизвестная нормировка KL.")


def beta_at(epoch, warmup, target):
    return target if warmup <= 0 else target * min(1.0, max(0.0, epoch / warmup))


class ReconstructionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ssim = SSIMLoss(config.model["out_channels"])
        # Preserve the earlier GradientLoss's Sobel convention.
        sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
            None, None
        ]
        self.register_buffer("sobel_x", sobel)
        self.register_buffer("sobel_y", sobel.transpose(2, 3).contiguous())

    def forward(self, reconstruction, target, valid):
        cfg = self.config
        difference = reconstruction - target
        values = difference.square() if cfg.loss_kind == "mse" else difference.abs()
        if cfg.edge_weighted_l1:
            with torch.no_grad():
                weights = 1 + cfg.edge_l1_extra_w * compute_edge_map_torch(target)
            values = values * weights
        result = cfg.l1_weight * masked_mean(values, valid)
        if cfg.grad_loss_w:
            channels = target.shape[1]
            # Each channel is compared; padding is excluded from contributing centers.
            gradient = target.new_zeros(())
            for kernel in (self.sobel_x, self.sobel_y):
                first = F.conv2d(
                    reconstruction,
                    kernel.expand(channels, 1, 3, 3),
                    padding=1,
                    groups=channels,
                )
                second = F.conv2d(
                    target, kernel.expand(channels, 1, 3, 3), padding=1, groups=channels
                )
                gradient = gradient + masked_mean((first - second).abs(), valid)
            result = result + cfg.grad_loss_w * gradient
        if cfg.use_ssim:
            result = result + cfg.ssim_weight * self.ssim(reconstruction, target, valid)
        return result


def training_objective(reconstruction, target, valid, extra, criterion, beta):
    recon = criterion(reconstruction, target, valid)
    kl = recon.new_zeros(())
    cfg = criterion.config
    if cfg.model["model_type"] == "vae":
        if extra is None:
            raise ValueError("VAE не вернул параметры распределения.")
        kl = kl_loss(
            *extra,
            reduction=cfg.kl_reduction,
            logvar_min=cfg.model["logvar_min"],
            logvar_max=cfg.model["logvar_max"],
        )
    return recon + beta * kl, recon, kl


def discriminator_loss(real, fake):
    return 0.5 * (F.relu(1 - real).mean() + F.relu(1 + fake).mean())
