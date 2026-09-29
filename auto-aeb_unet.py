# auto-aeb.py
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

# --- Геометрия ---
# ВАЖНО: вход должен быть кратен 8 (U-Net делает 3 пула, а не 4).
IMG_H        = 128
IMG_W        = 192
KEEP_ASPECT  = True
PAD_VALUE    = 0

# --- Цвет ---
COLOR_MODE   = "grayscale"              # "color" | "grayscale"
IN_CHANNELS  = 3 if COLOR_MODE == "color" else 1

# --- Тип автоэнкодера ---
MODEL_TYPE   = "vae"                # "ae" | "vae"
BETA_VAE     = 1.0

# --- Функция потерь ---
USE_SSIM       = True
SSIM_WEIGHT    = 0.25                # уменьшил: L1+grad уже дают резкость
GRAD_LOSS_W    = 0.5                 # вес градиентной компоненты
L1_WEIGHT      = 1.0

# --- Шумовая аугментация ---
USE_NOISE_AUG = True
NOISE_STD     = 0.05

# --- Аугментация ---
USE_AUGMENTATION = True
AUG_TRANSLATE    = 0.03
AUG_DEGREES      = 2.0
AUG_BRIGHTNESS   = 0.15
AUG_CONTRAST     = 0.15
AUG_HFLIP        = True              # зеркала по осям (безопасно для большинства задач)
AUG_VFLIP        = False

# --- Обучение ---
# Пространственный латент: это ЧИСЛО КАНАЛОВ в бутылочном горлышке,
# а не размер flat-вектора. Для 128×192 и шага 8 латент = C×16×24.
LATENT_CHANNELS = 8
BATCH_SIZE   = 16                   # будет автоматически уменьшен, если выборка мала
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.1
NUM_WORKERS  = 0                    # 0 безопаснее на Windows
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
PATIENCE     = 30

THRESHOLD_PERCENTILE = 99.0
# ============================================

assert IMG_H % 8 == 0 and IMG_W % 8 == 0, \
    "U-Net делает 3 пула → H и W должны быть кратны 8."
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


# ------------------ Gradient Loss ------------------
class GradientLoss(nn.Module):
    """
    L1 между пространственными градиентами pred и target.
    Заставляет декодер воспроизводить контуры/края, что напрямую
    повышает чувствительность карты ошибок к мелким локальным дефектам.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        C = pred.size(1)
        kx = self.kx.expand(C, 1, 3, 3).to(pred.device, pred.dtype)
        ky = self.ky.expand(C, 1, 3, 3).to(pred.device, pred.dtype)
        gxp = F.conv2d(pred, kx, padding=1, groups=C)
        gyp = F.conv2d(pred, ky, padding=1, groups=C)
        gxt = F.conv2d(target, kx, padding=1, groups=C)
        gyt = F.conv2d(target, ky, padding=1, groups=C)
        return F.l1_loss(gxp, gxt) + F.l1_loss(gyp, gyt)


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
            if AUG_HFLIP:
                aug_ops.append(transforms.RandomHorizontalFlip(p=0.5))
            if AUG_VFLIP:
                aug_ops.append(transforms.RandomVerticalFlip(p=0.5))
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


# ============================================================
# U-Net AE / VAE
# ============================================================
class UNetEncoder(nn.Module):
    """
    3 пула → шаг 8. Бутылочное горлышко для 128×192: 16×24.
    Skip-выходы f0, f1, f2 идут в декодер напрямую.
    """
    def __init__(self, in_channels, base=32):
        super().__init__()
        b = base
        self.enc0 = nn.Sequential(                     # 1/1
            nn.Conv2d(in_channels, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(                     # 1/2
            nn.Conv2d(b, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(                     # 1/4
            nn.Conv2d(b * 2, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(                     # 1/8 (bottleneck)
            nn.Conv2d(b * 4, b * 8, 3, padding=1),
            nn.BatchNorm2d(b * 8), nn.ReLU(inplace=True),
            nn.Conv2d(b * 8, b * 8, 3, padding=1),
            nn.BatchNorm2d(b * 8), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(self.pool(f0))
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        return f0, f1, f2, f3


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, base=32):
        super().__init__()
        b = base
        self.up2 = nn.Sequential(                      # 1/8 → 1/4
            nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(b * 4 + b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(                      # 1/4 → 1/2
            nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(b * 2 + b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.up0 = nn.Sequential(                      # 1/2 → 1/1
            nn.ConvTranspose2d(b * 2, b, 2, stride=2),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(b + b, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h, f0, f1, f2):
        d = self.up2(h);  d = torch.cat([d, f2], dim=1); d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, f1], dim=1); d = self.dec1(d)
        d = self.up0(d);  d = torch.cat([d, f0], dim=1); d = self.dec0(d)
        return d


class UNetAE(nn.Module):
    """
    Полностью свёрточный AE. Латент — пространственная карта
    (B, latent_channels, H/8, W/8), а не плоский вектор.
    """
    def __init__(self, in_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.encoder = UNetEncoder(in_channels, base)
        self.compress = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(in_channels, base)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        z = self.compress(f3)
        h = self.expand(z)
        recon = self.decoder(h, f0, f1, f2)
        return recon, None


class UNetVAE(nn.Module):
    """Тот же U-Net, но с VAE-головой на пространственной карте."""
    def __init__(self, in_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.latent_channels = latent_channels
        self.latent_h = img_h // 8
        self.latent_w = img_w // 8
        self.base = base
        self.encoder = UNetEncoder(in_channels, base)
        self.fc_mu = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(in_channels, base)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        mu = self.fc_mu(f3)
        logvar = self.fc_logvar(f3)
        z = self.reparameterize(mu, logvar)
        h = self.expand(z)
        recon = self.decoder(h, f0, f1, f2)
        return recon, (mu, logvar)


def build_model(model_type, in_channels, latent_channels, img_h, img_w):
    if model_type == "vae":
        return UNetVAE(in_channels, latent_channels, img_h, img_w)
    return UNetAE(in_channels, latent_channels, img_h, img_w)


def vae_kl_loss(mu, logvar):
    """
    Нормированный KL для пространственного латента.
    Делим на (C*H*W), чтобы значение было сопоставимо с пиксельным лоссом
    (иначе β_Vae физически "не работает" — KL доминирует над реконструкцией).
    """
    n = mu.size(1) * mu.size(2) * mu.size(3)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (mu.size(0) * n)


# ------------------ Обучение ------------------
def _compute_recon_loss(recon, x, criterion_l1, criterion_grad, criterion_ssim):
    loss = L1_WEIGHT * criterion_l1(recon, x)
    loss = loss + GRAD_LOSS_W * criterion_grad(recon, x)
    if USE_SSIM:
        loss = loss + SSIM_WEIGHT * criterion_ssim(recon, x)
    return loss


def train_one_epoch(model, loader, optimizer, criterion_l1, criterion_grad,
                    criterion_ssim, model_type, device):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    n = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        recon, extra = model(x)
        recon_loss = _compute_recon_loss(recon, x, criterion_l1, criterion_grad, criterion_ssim)

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)
            loss = recon_loss + BETA_VAE * kl
        else:
            loss = recon_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs
        n += bs

    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim,
             model_type, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_errors = []

    for x in loader:
        x = x.to(device, non_blocking=True)
        recon, extra = model(x)

        recon_loss = _compute_recon_loss(recon, x, criterion_l1, criterion_grad, criterion_ssim)

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)
            loss = recon_loss + BETA_VAE * kl
        else:
            loss = recon_loss

        per_sample = F.l1_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
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
    print(f"  MODEL_TYPE     = {MODEL_TYPE}" + (f"  (β={BETA_VAE})" if MODEL_TYPE == "vae" else ""))
    print(f"  ARCH           = U-Net (3 пула, skip-connections)")
    print(f"  COLOR_MODE     = {COLOR_MODE} (in_channels={IN_CHANNELS})")
    print(f"  IMG            = ({IMG_H}, {IMG_W})  KEEP_ASPECT={KEEP_ASPECT}")
    print(f"  LATENT_CHANNELS= {LATENT_CHANNELS}  →  {LATENT_CHANNELS}×{IMG_H//8}×{IMG_W//8}"
          f" = {LATENT_CHANNELS * (IMG_H//8) * (IMG_W//8)}")
    print(f"  LOSS           = L1 + {GRAD_LOSS_W}·grad" +
          (f" + {SSIM_WEIGHT}·ssim" if USE_SSIM else ""))
    print(f"  USE_NOISE_AUG  = {USE_NOISE_AUG}" + (f"  (σ={NOISE_STD})" if USE_NOISE_AUG else ""))
    print(f"  AUGMENTATION   = {USE_AUGMENTATION}"
          + (f"  (hflip={AUG_HFLIP}, vflip={AUG_VFLIP})" if USE_AUGMENTATION else ""))
    print(f"  DEVICE         = {DEVICE}")
    print(f"{'='*60}")

    all_paths = collect_images(DATA_DIR)
    if not all_paths:
        raise SystemExit(f"Не найдено изображений в {DATA_DIR}")
    print(f"Найдено изображений: {len(all_paths)}")

    if len(all_paths) < 20:
        print(f"[WARN] Всего {len(all_paths)} изображений — модель почти наверняка переобучится.")
        print(f"[WARN] Рекомендуется ≥100 снимков, либо включите USE_AUGMENTATION=True.")
        print(f"[WARN] Сейчас USE_AUGMENTATION = {USE_AUGMENTATION}")

    random.shuffle(all_paths)

    # --- Сплит train/val с защитой от пустой валидации ---
    if len(all_paths) < 5:
        train_paths, val_paths = all_paths, all_paths
        print(f"[WARN] Слишком мало изображений — train и val совпадают.")
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
    print(f"[data] batch_size={eff_batch} (задан {BATCH_SIZE}), drop_last={drop_last}")

    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)

    model = build_model(MODEL_TYPE, IN_CHANNELS, LATENT_CHANNELS, IMG_H, IMG_W).to(DEVICE)
    print(model)
    print(f"bottleneck: {LATENT_CHANNELS}×{IMG_H//8}×{IMG_W//8} "
          f"= {LATENT_CHANNELS * (IMG_H//8) * (IMG_W//8)} значений латента")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-5
    )

    criterion_l1 = nn.L1Loss()
    criterion_grad = GradientLoss().to(DEVICE)
    criterion_ssim = SSIMLoss(channel=IN_CHANNELS).to(DEVICE)

    best_val = float("inf")
    hist = {"train": [], "val": [], "train_recon": [], "train_kl": []}
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_recon, tr_kl = train_one_epoch(
            model, train_loader, optimizer, criterion_l1, criterion_grad,
            criterion_ssim, MODEL_TYPE, DEVICE,
        )
        va_loss, va_errors = evaluate(
            model, val_loader, criterion_l1, criterion_grad,
            criterion_ssim, MODEL_TYPE, DEVICE,
        )
        scheduler.step()

        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["train_recon"].append(tr_recon)
        hist["train_kl"].append(tr_kl)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{EPOCHS}] "
              f"train={tr_loss:.6f} (recon={tr_recon:.6f}, kl={tr_kl:.6f})  "
              f"val={va_loss:.6f}  lr={lr_now:.2e}")

        if np.isfinite(va_loss) and va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            threshold = float(np.percentile(va_errors, THRESHOLD_PERCENTILE)) \
                if va_errors.size > 0 else float("nan")
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "arch": "unet",
                    "model_type": MODEL_TYPE,
                    "beta_vae": BETA_VAE,
                    "in_channels": IN_CHANNELS,
                    "color_mode": COLOR_MODE,
                    "latent_channels": LATENT_CHANNELS,
                    # legacy-ключ: некоторые внешние скрипты могут ожидать "latent_dim"
                    "latent_dim": LATENT_CHANNELS,
                    "img_h": IMG_H, "img_w": IMG_W,
                    "keep_aspect": KEEP_ASPECT,
                    "pad_value": PAD_VALUE,
                    "use_ssim": USE_SSIM,
                    "ssim_weight": SSIM_WEIGHT,
                    "grad_loss_w": GRAD_LOSS_W,
                    "l1_weight": L1_WEIGHT,
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
        im = axes[2, i].imshow(err, cmap="hot")
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