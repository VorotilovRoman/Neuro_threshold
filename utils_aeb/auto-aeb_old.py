# auto-aeb.py — legacy conv+BN
import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ================== КОНФИГ ==================
DATA_DIR     = r"C:\Users\prodis\Pictures\ДАТАСЕТЫ ДЛЯ ВКРМ\bga_ball\Good_ball"
OUT_DIR      = r"C:\Users\prodis\Pictures\ДАТАСЕТЫ ДЛЯ ВКРМ\bga_ball\Good_ball\runs_ae_def"

IMG_H        = 128
IMG_W        = 192
KEEP_ASPECT  = True
PAD_VALUE    = 0

COLOR_MODE   = "grayscale"
IN_CHANNELS  = 3 if COLOR_MODE == "color" else 1

MODEL_TYPE   = "vae"

# FIX: β через target+warmup. warmup=0 → β=target с 1-й эпохи (legacy-поведение).
BETA_VAE_TARGET        = 1.0
BETA_VAE_WARMUP_EPOCHS = 0

USE_SSIM     = True
SSIM_WEIGHT  = 0.5

USE_NOISE_AUG = True
NOISE_STD     = 0.05

USE_AUGMENTATION = True
AUG_TRANSLATE    = 0.03
AUG_DEGREES      = 2.0
AUG_BRIGHTNESS   = 0.15
AUG_CONTRAST     = 0.15

LATENT_DIM   = 128
BATCH_SIZE   = 16
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.1
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
PATIENCE     = 30

THRESHOLD_PERCENTILE = 99.0

# FIX: клиппинг logvar.
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0
# ============================================

assert IMG_H % 16 == 0 and IMG_W % 16 == 0
assert COLOR_MODE in ("color", "grayscale")
assert MODEL_TYPE in ("ae", "vae")

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------ SSIM Loss ------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=None):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        if channel is not None:
            self.register_buffer("window", self._create_window(window_size, channel))

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return _2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        if self.channel is None or not hasattr(self, "window"):
            self.channel = img1.size(1)
            self.window = self._create_window(self.window_size, self.channel).to(img1.device)
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


# ------------------ Датасет ------------------
class ImageFolderDataset(Dataset):
    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, paths, img_h, img_w, color_mode, augment,
                 keep_aspect=True, pad_value=0, noise_std=0.0):
        self.paths = paths
        self.img_h, self.img_w = img_h, img_w
        self.keep_aspect = keep_aspect
        self.pad_value = pad_value
        self.noise_std = noise_std
        self.mode = "RGB" if color_mode == "color" else "L"

        aug_ops = []
        if augment:
            if AUG_TRANSLATE > 0 or AUG_DEGREES > 0:
                aug_ops.append(transforms.RandomAffine(
                    degrees=AUG_DEGREES,
                    translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                ))
            if AUG_BRIGHTNESS > 0 or AUG_CONTRAST > 0:
                aug_ops.append(transforms.ColorJitter(
                    brightness=AUG_BRIGHTNESS,
                    contrast=AUG_CONTRAST,
                ))
        aug_ops.append(transforms.ToTensor())
        self.aug_tf = transforms.Compose(aug_ops)

    def _fit(self, img):
        w, h = img.size
        scale = min(self.img_w / w, self.img_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new(img.mode, (self.img_w, self.img_h), color=self.pad_value)
        off_x = (self.img_w - new_w) // 2
        off_y = (self.img_h - new_h) // 2
        canvas.paste(img, (off_x, off_y))
        return canvas

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert(self.mode)
        if self.keep_aspect:
            img = self._fit(img)
        else:
            img = img.resize((self.img_w, self.img_h), Image.BILINEAR)
        x = self.aug_tf(img)
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x


def collect_images(root, exclude_names=("loss.png", "recon_examples.png", "best.pt")):
    root = Path(root)
    files = []
    for ext in ImageFolderDataset.EXTS:
        files += glob.glob(str(root / "**" / f"*{ext}"), recursive=True)
        files += glob.glob(str(root / "**" / f"*{ext.upper()}"), recursive=True)
    excl = {n.lower() for n in exclude_names}
    files = [f for f in files if Path(f).name.lower() not in excl]
    return sorted(set(files))


# ------------------ Модели (conv + BatchNorm, enc_c=128) ------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 128
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.encoder = ConvEncoder(in_channels)
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoder(in_channels)

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        z = self.fc_encode(z)
        z = self.fc_decode(z)
        z = z.view(z.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(z), None


class ConvVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 128
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoder(in_channels)

    def reparameterize(self, mu, logvar):
        # FIX: клиппинг logvar.
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z, logvar_c = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(h_dec), (mu, logvar_c)


def build_model(model_type, in_channels, latent_dim, img_h, img_w):
    if model_type == "vae":
        return ConvVAE(in_channels, latent_dim, img_h, img_w)
    return ConvAutoencoder(in_channels, latent_dim, img_h, img_w)


def vae_kl_loss(mu, logvar):
    # FIX: клиппинг logvar.
    logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def get_beta(epoch, warmup_epochs, target_beta):
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, float(epoch) / float(warmup_epochs))


# ------------------ Обучение ------------------
def train_one_epoch(model, loader, optimizer, criterion_mse, criterion_ssim,
                    model_type, device, beta):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n = 0
    n_skipped = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        recon, extra = model(x)

        loss_mse = criterion_mse(recon, x)
        if USE_SSIM:
            recon_loss = loss_mse + SSIM_WEIGHT * criterion_ssim(recon, x)
        else:
            recon_loss = loss_mse

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)
            loss = recon_loss + beta * kl
        else:
            loss = recon_loss

        # FIX: NaN-guard.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs
        n += bs

    if n_skipped:
        print(f"  [WARN] пропущено батчей: {n_skipped}")

    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, criterion_mse, criterion_ssim, model_type, device, beta):
    model.eval()
    total_loss = 0.0
    n = 0
    all_errors = []

    for x in loader:
        x = x.to(device, non_blocking=True)
        recon, extra = model(x)

        loss_mse = criterion_mse(recon, x)
        if USE_SSIM:
            recon_loss = loss_mse + SSIM_WEIGHT * criterion_ssim(recon, x)
        else:
            recon_loss = loss_mse

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)
            loss = recon_loss + beta * kl
        else:
            loss = recon_loss

        if not torch.isfinite(loss):
            continue

        per_sample = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
        all_errors.append(per_sample.cpu())

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    if n == 0:
        return float("nan"), np.array([])
    all_errors = torch.cat(all_errors).numpy()
    return total_loss / n, all_errors


def main():
    print(f"{'='*60}")
    print(f"Конфиг:")
    print(f"  MODEL_TYPE     = {MODEL_TYPE}"
          + (f"  (β_target={BETA_VAE_TARGET}, warmup={BETA_VAE_WARMUP_EPOCHS})"
             if MODEL_TYPE == "vae" else ""))
    print(f"  ARCH           = conv (BatchNorm, enc_c=128)")
    print(f"  COLOR_MODE     = {COLOR_MODE} (in_channels={IN_CHANNELS})")
    print(f"  IMG            = ({IMG_H}, {IMG_W})  KEEP_ASPECT={KEEP_ASPECT}")
    print(f"  LR             = {LR}")
    print(f"  DEVICE         = {DEVICE}")
    print(f"{'='*60}")

    all_paths = collect_images(DATA_DIR)
    if not all_paths:
        raise SystemExit(f"Не найдено изображений в {DATA_DIR}")
    print(f"Найдено изображений: {len(all_paths)}")

    if len(all_paths) < 20:
        print(f"[WARN] Всего {len(all_paths)} изображений — модель переобучится.")

    random.shuffle(all_paths)

    if len(all_paths) < 5:
        train_paths, val_paths = all_paths, all_paths
        print(f"[WARN] train и val совпадают.")
    else:
        n_val = max(1, int(len(all_paths) * VAL_SPLIT))
        n_val = min(n_val, len(all_paths) - 1)
        val_paths = all_paths[:n_val]
        train_paths = all_paths[n_val:]

    n_train, n_val = len(train_paths), len(val_paths)
    print(f"[data] train={n_train}, val={n_val}")

    noise_std = NOISE_STD if USE_NOISE_AUG else 0.0
    train_ds = ImageFolderDataset(train_paths, IMG_H, IMG_W, COLOR_MODE,
                                  augment=USE_AUGMENTATION, keep_aspect=KEEP_ASPECT,
                                  pad_value=PAD_VALUE, noise_std=noise_std)
    val_ds = ImageFolderDataset(val_paths, IMG_H, IMG_W, COLOR_MODE,
                                augment=False, keep_aspect=KEEP_ASPECT,
                                pad_value=PAD_VALUE, noise_std=0.0)

    if n_train == 0:
        raise SystemExit("Обучающая выборка пуста.")

    eff_batch = min(BATCH_SIZE, max(1, n_train // 4))
    drop_last = (n_train // eff_batch) >= 2
    print(f"[data] batch_size={eff_batch}, drop_last={drop_last}")

    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)

    model = build_model(MODEL_TYPE, IN_CHANNELS, LATENT_DIM, IMG_H, IMG_W).to(DEVICE)
    print(model)
    if hasattr(model, "flat_dim"):
        print(f"bottleneck: {model.enc_c}x{model.enc_h}x{model.enc_w} "
              f"= {model.flat_dim} → latent {LATENT_DIM}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # FIX: CosineAnnealingLR вместо CosineAnnealingWarmRestarts.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    criterion_mse = nn.MSELoss()
    criterion_ssim = SSIMLoss(channel=IN_CHANNELS).to(DEVICE)

    best_val = float("inf")
    hist = {"train": [], "val": [], "train_recon": [], "train_kl": [], "beta": []}
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch, BETA_VAE_WARMUP_EPOCHS, BETA_VAE_TARGET)

        tr_loss, tr_recon, tr_kl = train_one_epoch(
            model, train_loader, optimizer, criterion_mse,
            criterion_ssim, MODEL_TYPE, DEVICE, beta,
        )
        va_loss, va_errors = evaluate(
            model, val_loader, criterion_mse, criterion_ssim,
            MODEL_TYPE, DEVICE, beta,
        )
        scheduler.step()

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["train_recon"].append(tr_recon)
        hist["train_kl"].append(tr_kl)
        hist["beta"].append(beta)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{EPOCHS}] "
              f"train={tr_loss:.6f} (recon={tr_recon:.6f}, kl={tr_kl:.6f}, β={beta:.4f})  "
              f"val={va_loss:.6f}  lr={lr_now:.2e}")

        # FIX: явная обработка NaN в val.
        if not np.isfinite(va_loss):
            print(f"  [WARN] val loss = {va_loss} — пропускаем сохранение.")
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Ранняя остановка (NaN).")
                break
            continue

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            threshold = float(np.percentile(va_errors, THRESHOLD_PERCENTILE)) \
                if va_errors.size > 0 else float("nan")
            torch.save({
                "model": model.state_dict(),
                "config": {
                    # FIX: явная метка архитектуры для инференса.
                    "arch": "conv",
                    "model_type": MODEL_TYPE,
                    "beta_vae_target": BETA_VAE_TARGET,
                    "beta_vae_warmup_epochs": BETA_VAE_WARMUP_EPOCHS,
                    "in_channels": IN_CHANNELS,
                    "color_mode": COLOR_MODE,
                    "latent_dim": LATENT_DIM,
                    "img_h": IMG_H, "img_w": IMG_W,
                    "keep_aspect": KEEP_ASPECT,
                    "pad_value": PAD_VALUE,
                    "use_ssim": USE_SSIM,
                    "ssim_weight": SSIM_WEIGHT,
                    "logvar_min": LOGVAR_MIN,
                    "logvar_max": LOGVAR_MAX,
                },
                "epoch": epoch,
                "val_loss": best_val,
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "threshold_value": threshold,
            }, os.path.join(OUT_DIR, "best.pt"))
            print(f"  ↳ сохранено (best val={best_val:.6f}, "
                  f"threshold={threshold:.6f} @ p{THRESHOLD_PERCENTILE})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Ранняя остановка: нет улучшений {PATIENCE} эпох.")
                break

    # --- Графики ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist["train"], label="train")
    axes[0].plot(hist["val"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(True); axes[0].set_title("Total Loss")

    axes[1].plot(hist["train_recon"], label="recon")
    if MODEL_TYPE == "vae":
        axes[1].plot(hist["train_kl"], label="kl")
        axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[1].grid(True); axes[1].set_title("Components")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=120)

    # --- Примеры реконструкций ---
    ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        x = next(iter(val_loader)).to(DEVICE)[:8]
        recon, _ = model(x)

    n = x.size(0)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 6))
    if n == 1:
        axes = axes.reshape(3, 1)
    for i in range(n):
        xi, ri = x[i].cpu(), recon[i].cpu()
        err = (xi - ri).abs().mean(dim=0)
        if IN_CHANNELS == 1:
            axes[0, i].imshow(xi[0], cmap="gray", vmin=0, vmax=1)
            axes[1, i].imshow(ri[0], cmap="gray", vmin=0, vmax=1)
        else:
            axes[0, i].imshow(xi.permute(1, 2, 0))
            axes[1, i].imshow(ri.permute(1, 2, 0).clamp(0, 1))
        axes[2, i].imshow(err, cmap="hot")
        axes[0, i].set_title("orig", fontsize=8)
        axes[1, i].set_title("recon", fontsize=8)
        axes[2, i].set_title("|err|", fontsize=8)
        for ax in axes[:, i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recon_examples.png"), dpi=120)

    print(f"\nГотово. Артефакты в: {OUT_DIR}")


if __name__ == "__main__":
    main()