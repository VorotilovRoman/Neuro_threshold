# auto-aeb_unet.py — U-Net AE/VAE + edge channel (disk cache) + soft arch + PatchGAN
import os
import glob
import time
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import matplotlib.pyplot as plt
# ================== КОНФИГ ==================
DATA_DIR     = r"D:\ДАТАСЕТЫ НЕЙРО\фильтры керамика\new\new"
OUT_DIR      = r"D:\ДАТАСЕТЫ НЕЙРО\фильтры керамика\new\new\runs_ae_def_new"
EDGE_CACHE_DIR = r"D:\ДАТАСЕТЫ НЕЙРО\фильтры керамика\new\new\_edge_cache"
PREWARM_EDGE_CACHE = True

IMG_H        = 256
IMG_W        = 256
KEEP_ASPECT  = True
PAD_VALUE    = 0

COLOR_MODE   = "color"
IN_CHANNELS  = 3 if COLOR_MODE == "color" else 1

# ---- Модель ----
# CHANGED: "ae" вместо "vae" — KL всё равно сколлапсировал (kl≈1.8e-5),
# вклад регуляризатора ≈0. Для задачи «реконструкция → diff → дефекты»
# AE честнее и убирает лишний шум от reparameterize.
MODEL_TYPE   = "ae"

# CHANGED: 24 вместо 16 — даём больше ёмкости для тонких дорожек
# и мелких структур платы. При 256×256 латент станет 24×32×32.
LATENT_CHANNELS = 24
NORM_TYPE       = "gn"
BOTTLENECK_DILATIONS = (2, 4)

# ---- VAE (не используется при MODEL_TYPE="ae", но пусть будет) ----
BETA_VAE_TARGET        = 0.0    # CHANGED: 0.3 → 0.0
BETA_VAE_WARMUP_EPOCHS = 60
LOGVAR_MIN, LOGVAR_MAX = -10.0, 10.0

# ---- Рёбра ----
USE_EDGE_CHANNEL = True
EDGE_KERNEL      = "scharr"
MEDIAN_KERNEL    = 3
EDGE_LO          = 0.10
EDGE_HI          = 0.35

# ---- Лоссы ----
USE_SSIM       = True
SSIM_WEIGHT    = 0.35
GRAD_LOSS_W    = 2.0
L1_WEIGHT      = 0.7
USE_EDGE_WEIGHTED_L1 = True
EDGE_L1_EXTRA_W      = 3.0

# ---- Аугментации ----
USE_NOISE_AUG = True
NOISE_STD     = 0.05

USE_AUGMENTATION = True
AUG_TRANSLATE    = 0.03
AUG_DEGREES      = 2.0
AUG_BRIGHTNESS   = 0.15
AUG_CONTRAST     = 0.15
AUG_HFLIP        = True
AUG_VFLIP        = False

# ---- GAN ----
USE_GAN          = True
GAN_START_EPOCH  = 60           # CHANGED: 40 → 60 (дать реконструкции устояться)
GAN_WEIGHT       = 0.05         # CHANGED: 0.08 → 0.05 (мягче, чтобы не тянуть в текстуры)
GAN_LR           = 1e-4
D_BASE           = 32

# ---- Обучение ----
BATCH_SIZE   = 16
EPOCHS       = 120              # CHANGED: 200 → 120 (модель на плато с ep 60-120)
LR           = 3e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT    = 0.2              # CHANGED: 0.1 → 0.2 (при 27 картинках: 5 в val вместо 2)
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
PATIENCE     = 25               # CHANGED: 50 → 25 (короче цикл, всё равно на плато)

THRESHOLD_PERCENTILE = 99.0
# ============================================

MODEL_IN_CHANNELS = IN_CHANNELS + (1 if USE_EDGE_CHANNEL else 0)

assert IMG_H % 8 == 0 and IMG_W % 8 == 0
assert COLOR_MODE in ("color", "grayscale")
assert MODEL_TYPE in ("ae", "vae")
assert NORM_TYPE in ("gn", "bn")
assert EDGE_KERNEL in ("scharr", "sobel")

if USE_GAN and MODEL_TYPE == "vae" and BETA_VAE_TARGET > 0.5:
    print(f"[WARN] USE_GAN=True и BETA_VAE_TARGET={BETA_VAE_TARGET} > 0.5 — "
          f"GAN и KL могут конфликтовать. Рекомендуется β ≤ 0.3 или MODEL_TYPE='ae'.")

os.makedirs(OUT_DIR, exist_ok=True)
if USE_EDGE_CHANNEL:
    os.makedirs(EDGE_CACHE_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================== РЁБРА / МЕДИАНА ==================
def _to_gray(x):
    if x.size(1) == 3:
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    return x


def median_blur(x, k=3):
    """x: [B,C,H,W]. Медиана через unfold."""
    if k <= 1:
        return x
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)
    patches = patches.contiguous().view(*x.shape, k * k)
    return patches.median(dim=-1).values


_SCHARR = torch.tensor([[ 3.,  0., -3.],
                        [10.,  0.,-10.],
                        [ 3.,  0., -3.]], dtype=torch.float32)
_SOBEL  = torch.tensor([[-1.,  0.,  1.],
                        [-2.,  0.,  2.],
                        [-1.,  0.,  1.]], dtype=torch.float32)


def _edge_kernels(kind):
    kx = _SCHARR if kind == "scharr" else _SOBEL
    kx = kx.view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    scale = 16.0 if kind == "scharr" else 4.0
    return kx / scale, ky / scale


def edge_map(x, lo=EDGE_LO, hi=EDGE_HI, kind=EDGE_KERNEL, med=MEDIAN_KERNEL):
    """x: [B,C,H,W] в [0,1] → [B,1,H,W] мягкая карта рёбер."""
    g = _to_gray(x)
    if med > 1:
        g = median_blur(g, med)
    kx, ky = _edge_kernels(kind)
    kx = kx.to(x.device, x.dtype)
    ky = ky.to(x.device, x.dtype)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    mid = 0.5 * (lo + hi)
    scale = 6.0 / max(hi - lo, 1e-6)
    return torch.sigmoid(scale * (mag - mid))


# ================== ЛОССЫ ==================
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


class GradientLoss(nn.Module):
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


def d_hinge_loss(d_real, d_fake):
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_hinge_loss(d_fake):
    return -d_fake.mean()


# ================== ДАТАСЕТ С КЭШЕМ РЁБЕР ==================
class ImageFolderDataset(Dataset):
    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, paths, img_h, img_w, color_mode, augment,
                 keep_aspect=True, pad_value=0, noise_std=0.0,
                 use_edge=False, edge_cache_dir=None):
        self.paths = paths
        self.img_h, self.img_w = img_h, img_w
        self.keep_aspect = keep_aspect
        self.pad_value = pad_value
        self.noise_std = noise_std
        self.use_edge = use_edge
        self.mode = "RGB" if color_mode == "color" else "L"
        self.augment = augment

        self.cache_dir = None
        if use_edge and edge_cache_dir:
            self.cache_dir = Path(edge_cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.jitter = None
        if augment and (AUG_BRIGHTNESS > 0 or AUG_CONTRAST > 0):
            self.jitter = transforms.ColorJitter(
                brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST,
            )

    # ---------- image utils ----------
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

    def _load_resized_tensor(self, path):
        img = Image.open(path).convert(self.mode)
        img = self._fit(img) if self.keep_aspect \
              else img.resize((self.img_w, self.img_h), Image.BILINEAR)
        return TF.to_tensor(img)  # [C,H,W] float32 in [0,1]

    # ---------- edge cache ----------
    def _edge_cache_path(self, img_path):
        src = Path(img_path)
        try:
            st = src.stat()
            src_sig = f"{st.st_mtime_ns}_{st.st_size}"
        except OSError:
            src_sig = "nostat"
        key = "|".join([
            src.name, src_sig,
            f"{self.img_h}x{self.img_w}",
            f"ka{int(self.keep_aspect)}",
            f"pad{self.pad_value}",
            f"mode{self.mode}",
            f"med{MEDIAN_KERNEL}",
            f"k{EDGE_KERNEL}",
            f"lo{EDGE_LO:.4f}_hi{EDGE_HI:.4f}",
        ])
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{src.stem}_{h}.npy"

    def _load_or_build_edge(self, img_path, img_t):
        """Возвращает [1,H,W] float32 — карта рёбер чистого (без аугментации/шума) изображения."""
        if self.cache_dir is None:
            with torch.no_grad():
                return edge_map(img_t.unsqueeze(0))[0]

        cp = self._edge_cache_path(img_path)
        if cp.exists():
            try:
                arr = np.load(cp)
                return torch.from_numpy(arr.astype(np.float32))
            except Exception:
                pass  # повреждён — пересчитаем

        with torch.no_grad():
            e = edge_map(img_t.unsqueeze(0))[0]  # [1,H,W] float32

        try:
            np.save(cp, e.cpu().numpy().astype(np.float16))
        except OSError:
            pass
        return e

    # ---------- joint augmentation ----------
    def _augment_geom(self, img_t, edge_t):
        """Одинаковые геом. преобразования для image и edge (или edge_t is None)."""
        if AUG_TRANSLATE > 0 or AUG_DEGREES > 0:
            # FIX: get_params требует пару (min, max), а не float.
            degrees = (-float(AUG_DEGREES), float(AUG_DEGREES))
            translate = (float(AUG_TRANSLATE), float(AUG_TRANSLATE))
            params = transforms.RandomAffine.get_params(
                degrees=degrees,
                translate=translate,
                scale_ranges=None,
                shears=None,
                img_size=list(img_t.shape[-2:]),
            )
            img_t = TF.affine(img_t, *params,
                              interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            if edge_t is not None:
                edge_t = TF.affine(edge_t, *params,
                                   interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)

        if AUG_HFLIP and random.random() < 0.5:
            img_t = torch.flip(img_t, dims=[-1])
            if edge_t is not None:
                edge_t = torch.flip(edge_t, dims=[-1])

        if AUG_VFLIP and random.random() < 0.5:
            img_t = torch.flip(img_t, dims=[-2])
            if edge_t is not None:
                edge_t = torch.flip(edge_t, dims=[-2])

        return img_t, edge_t

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img_t = self._load_resized_tensor(path)                   # [C,H,W]

        edge_t = None
        if self.use_edge:
            edge_t = self._load_or_build_edge(path, img_t)        # [1,H,W]

        if self.augment:
            img_t, edge_t = self._augment_geom(img_t, edge_t)

        if self.jitter is not None:
            img_t = self.jitter(img_t)                            # только image

        target = img_t.clone()

        if self.noise_std > 0:
            noise = torch.randn_like(img_t) * self.noise_std
            img_t = torch.clamp(img_t + noise, 0.0, 1.0)

        if self.use_edge:
            img_t = torch.cat([img_t, edge_t], dim=0)             # [C+1,H,W]

        return img_t, target


def collect_images(root, exclude_names=("loss.png", "recon_examples.png", "best.pt")):
    root = Path(root)
    files = []
    for ext in ImageFolderDataset.EXTS:
        files += glob.glob(str(root / "**" / f"*{ext}"), recursive=True)
        files += glob.glob(str(root / "**" / f"*{ext.upper()}"), recursive=True)
    excl = {n.lower() for n in exclude_names}
    files = [f for f in files if Path(f).name.lower() not in excl]
    return sorted(set(files))


@torch.no_grad()
def prewarm_edge_cache(dataset):
    """Считает и сохраняет edge-карты для всех сэмплов датасета (только новые)."""
    if dataset.cache_dir is None or not dataset.use_edge:
        return
    print(f"[edge-cache] прогрев {len(dataset)} изображений → {dataset.cache_dir}")
    t0 = time.time()
    n_new = n_hit = n_fail = 0
    for i in range(len(dataset)):
        cp = dataset._edge_cache_path(dataset.paths[i])
        if cp.exists():
            n_hit += 1
            continue
        try:
            img_t = dataset._load_resized_tensor(dataset.paths[i])
            e = edge_map(img_t.unsqueeze(0))[0]
            np.save(cp, e.cpu().numpy().astype(np.float16))
            n_new += 1
            if n_new % 100 == 0:
                print(f"  ...{n_new} новых за {time.time() - t0:.1f}s")
        except Exception as ex:
            n_fail += 1
            print(f"  [WARN] {dataset.paths[i]}: {ex}")
    print(f"[edge-cache] готово: new={n_new}, hit={n_hit}, fail={n_fail}, "
          f"время={time.time() - t0:.1f}s")


# ================== АРХИТЕКТУРА ==================
def _norm(c, kind=NORM_TYPE):
    """GroupNorm с безопасным числом групп (c % g == 0)."""
    if kind == "gn":
        g = min(8, c)
        while g > 1 and c % g != 0:
            g -= 1
        return nn.GroupNorm(g, c)
    return nn.BatchNorm2d(c)


def _act():
    return nn.LeakyReLU(0.1, inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            s = stride if i == 0 else 1
            layers += [
                nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, stride=s, padding=1),
                _norm(out_c),
                _act(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DilatedResidual(nn.Module):
    def __init__(self, c, dilations=(2, 4)):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=d, dilation=d),
                _norm(c), _act(),
                nn.Conv2d(c, c, 3, padding=d, dilation=d),
                _norm(c),
            ) for d in dilations
        ])
        self.act = _act()

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return self.act(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        b = base
        self.enc0 = ConvBlock(in_channels, b,     stride=1, n_convs=2)
        self.enc1 = ConvBlock(b,          b * 2,  stride=2, n_convs=2)
        self.enc2 = ConvBlock(b * 2,      b * 4,  stride=2, n_convs=2)
        self.enc3 = ConvBlock(b * 4,      b * 8,  stride=2, n_convs=2)
        self.bottleneck = DilatedResidual(b * 8, BOTTLENECK_DILATIONS)

    def forward(self, x):
        f0 = self.enc0(x)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f3 = self.bottleneck(f3)
        return f0, f1, f2, f3


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.norm_up = _norm(out_c)
        self.conv1 = nn.Conv2d(out_c + skip_c, out_c, 3, padding=1)
        self.n1 = _norm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.n2 = _norm(out_c)
        self.act = _act()

    def forward(self, x, skip):
        x = self.act(self.norm_up(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, base=32):
        super().__init__()
        b = base
        self.dec2 = DecoderBlock(b * 8, b * 4, b * 4)
        self.dec1 = DecoderBlock(b * 4, b * 2, b * 2)
        self.dec0 = DecoderBlock(b * 2, b,     b)
        self.head = nn.Sequential(
            nn.Conv2d(b, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h, f0, f1, f2):
        d = self.dec2(h, f2)
        d = self.dec1(d, f1)
        d = self.dec0(d, f0)
        return self.head(d)


class UNetAE(nn.Module):
    # FIX: отдельные in/out channels (edge-канал только на входе).
    def __init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.encoder = UNetEncoder(in_channels, base)
        self.compress = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(out_channels, base)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        z = self.compress(f3)
        h = self.expand(z)
        recon = self.decoder(h, f0, f1, f2)
        return recon, None


class UNetVAE(nn.Module):
    # FIX: отдельные in/out channels.
    def __init__(self, in_channels, out_channels, latent_channels, img_h, img_w, base=32):
        super().__init__()
        assert img_h % 8 == 0 and img_w % 8 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.encoder = UNetEncoder(in_channels, base)
        self.fc_mu = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(base * 8, latent_channels, kernel_size=1)
        self.expand = nn.Conv2d(latent_channels, base * 8, kernel_size=1)
        self.decoder = UNetDecoder(out_channels, base)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        mu = self.fc_mu(f3)
        logvar = self.fc_logvar(f3)
        z, logvar_c = self.reparameterize(mu, logvar)
        h = self.expand(z)
        recon = self.decoder(h, f0, f1, f2)
        return recon, (mu, logvar_c)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch, base=D_BASE):
        super().__init__()
        def blk(i, o, s, norm=True):
            L = [nn.Conv2d(i, o, 4, s, 1)]
            if norm:
                L.append(_norm(o))
            L.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*L)
        self.net = nn.Sequential(
            blk(in_ch,   base,     2, norm=False),
            blk(base,    base * 2, 2),
            blk(base * 2, base * 4, 2),
            blk(base * 4, base * 4, 1),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_model(model_type, in_channels, out_channels, latent_channels, img_h, img_w):
    # FIX: принимает и in, и out.
    if model_type == "vae":
        return UNetVAE(in_channels, out_channels, latent_channels, img_h, img_w)
    return UNetAE(in_channels, out_channels, latent_channels, img_h, img_w)


def vae_kl_loss(mu, logvar):
    logvar = torch.clamp(logvar, LOGVAR_MIN, LOGVAR_MAX)
    n = mu.size(1) * mu.size(2) * mu.size(3)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (mu.size(0) * n)


def get_beta(epoch, warmup_epochs, target_beta):
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, float(epoch) / float(warmup_epochs))


# ================== СБОРКА ЛОССА ==================
def _edge_weighted_l1(pred, target, w=EDGE_L1_EXTRA_W):
    base = F.l1_loss(pred, target, reduction="none")
    with torch.no_grad():
        e = edge_map(target)
    weight = 1.0 + w * e
    return (base * weight).mean()


def _compute_recon_loss(recon, target, criterion_l1, criterion_grad, criterion_ssim):
    if USE_EDGE_WEIGHTED_L1:
        loss = L1_WEIGHT * _edge_weighted_l1(recon, target)
    else:
        loss = L1_WEIGHT * criterion_l1(recon, target)
    loss = loss + GRAD_LOSS_W * criterion_grad(recon, target)
    if USE_SSIM:
        loss = loss + SSIM_WEIGHT * criterion_ssim(recon, target)
    return loss


# ================== ОБУЧЕНИЕ ==================
def train_one_epoch(model, loader, optimizer, D, opt_d,
                    criterion_l1, criterion_grad, criterion_ssim,
                    model_type, device, beta, use_gan, gan_weight):
    model.train()
    if D is not None:
        D.train()

    total_loss = total_recon = total_kl = total_adv = total_d = 0.0
    n = n_d = n_skipped = 0

    for x_in, target in loader:
        x_in = x_in.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ---- Generator ----
        optimizer.zero_grad(set_to_none=True)
        recon, extra = model(x_in)
        recon_loss = _compute_recon_loss(recon, target,
                                         criterion_l1, criterion_grad, criterion_ssim)

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)

        g_adv = torch.tensor(0.0, device=device)
        if use_gan:
            g_adv = g_hinge_loss(D(recon))

        loss = recon_loss + beta * kl + (gan_weight * g_adv if use_gan else 0.0)

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # ---- Discriminator ----
        if use_gan:
            opt_d.zero_grad(set_to_none=True)
            d_real = D(target.detach())
            d_fake = D(recon.detach())
            d_loss = d_hinge_loss(d_real, d_fake)
            if torch.isfinite(d_loss):
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
                opt_d.step()
                total_d += d_loss.item() * x_in.size(0)
                n_d += x_in.size(0)

        bs = x_in.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs
        if use_gan:
            total_adv += g_adv.item() * bs
        n += bs

    if n_skipped:
        print(f"  [WARN] пропущено батчей: {n_skipped}")

    if n == 0:
        return dict(loss=float("nan"), recon=float("nan"), kl=float("nan"),
                    adv=float("nan"), d=float("nan"))
    return dict(
        loss=total_loss / n,
        recon=total_recon / n,
        kl=total_kl / n,
        adv=total_adv / n if use_gan else 0.0,
        d=total_d / n_d if n_d > 0 else 0.0,
    )


@torch.no_grad()
def evaluate(model, loader, criterion_l1, criterion_grad, criterion_ssim,
             model_type, device, beta):
    model.eval()
    total_loss = 0.0
    n = 0
    all_errors = []

    for x_in, target in loader:
        x_in = x_in.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        recon, extra = model(x_in)
        recon_loss = _compute_recon_loss(recon, target,
                                         criterion_l1, criterion_grad, criterion_ssim)

        kl = torch.tensor(0.0, device=device)
        if model_type == "vae" and extra is not None:
            mu, logvar = extra
            kl = vae_kl_loss(mu, logvar)

        loss = recon_loss + beta * kl
        if not torch.isfinite(loss):
            continue

        per_sample = F.l1_loss(recon, target, reduction="none").flatten(1).mean(dim=1)
        all_errors.append(per_sample.cpu())

        bs = x_in.size(0)
        total_loss += loss.item() * bs
        n += bs

    if n == 0:
        return float("nan"), np.array([])
    all_errors = torch.cat(all_errors).numpy()
    return total_loss / n, all_errors


# ================== MAIN ==================
def main():
    print("=" * 60)
    print("Конфиг:")
    print(f"  MODEL_TYPE        = {MODEL_TYPE}"
          + (f"  (β_target={BETA_VAE_TARGET}, warmup={BETA_VAE_WARMUP_EPOCHS})"
             if MODEL_TYPE == "vae" else ""))
    print(f"  ARCH              = U-Net (3 пула, skip, GN, dilated bottleneck)")
    print(f"  COLOR_MODE        = {COLOR_MODE} (IN={IN_CHANNELS}, MODEL_IN={MODEL_IN_CHANNELS})")
    print(f"  EDGE_CHANNEL      = {USE_EDGE_CHANNEL}  (kernel={EDGE_KERNEL}, median={MEDIAN_KERNEL})")
    if USE_EDGE_CHANNEL:
        print(f"  EDGE_CACHE_DIR    = {EDGE_CACHE_DIR}")
    print(f"  IMG               = ({IMG_H}, {IMG_W})")
    print(f"  LATENT_CHANNELS   = {LATENT_CHANNELS}  →  {LATENT_CHANNELS}×{IMG_H // 8}×{IMG_W // 8}")
    print(f"  BOTTLENECK_DILATIONS = {BOTTLENECK_DILATIONS}")
    print(f"  LR                = {LR}")
    print(f"  USE_GAN           = {USE_GAN}"
          + (f"  (start_epoch={GAN_START_EPOCH}, w={GAN_WEIGHT})" if USE_GAN else ""))
    print(f"  DEVICE            = {DEVICE}")
    print("=" * 60)

    all_paths = collect_images(DATA_DIR)
    if not all_paths:
        raise SystemExit(f"Не найдено изображений в {DATA_DIR}")
    print(f"Найдено изображений: {len(all_paths)}")

    if len(all_paths) < 20:
        print(f"[WARN] Всего {len(all_paths)} изображений — модель переобучится.")

    random.shuffle(all_paths)

    if len(all_paths) < 5:
        train_paths, val_paths = all_paths, all_paths
        print("[WARN] train и val совпадают.")
    else:
        n_val = max(1, int(len(all_paths) * VAL_SPLIT))
        n_val = min(n_val, len(all_paths) - 1)
        val_paths = all_paths[:n_val]
        train_paths = all_paths[n_val:]

    n_train, n_val = len(train_paths), len(val_paths)
    print(f"[data] train={n_train}, val={n_val}")

    noise_std = NOISE_STD if USE_NOISE_AUG else 0.0
    edge_cache = EDGE_CACHE_DIR if USE_EDGE_CHANNEL else None

    train_ds = ImageFolderDataset(train_paths, IMG_H, IMG_W, COLOR_MODE,
                                  augment=USE_AUGMENTATION, keep_aspect=KEEP_ASPECT,
                                  pad_value=PAD_VALUE, noise_std=noise_std,
                                  use_edge=USE_EDGE_CHANNEL,
                                  edge_cache_dir=edge_cache)
    val_ds = ImageFolderDataset(val_paths, IMG_H, IMG_W, COLOR_MODE,
                                augment=False, keep_aspect=KEEP_ASPECT,
                                pad_value=PAD_VALUE, noise_std=0.0,
                                use_edge=USE_EDGE_CHANNEL,
                                edge_cache_dir=edge_cache)

    if n_train == 0:
        raise SystemExit("Обучающая выборка пуста.")

    # ---- Прогрев кэша рёбер ----
    if USE_EDGE_CHANNEL and PREWARM_EDGE_CACHE:
        prewarm_edge_cache(train_ds)
        prewarm_edge_cache(val_ds)

    eff_batch = min(BATCH_SIZE, max(1, n_train // 4))
    drop_last = (n_train // eff_batch) >= 2
    print(f"[data] batch_size={eff_batch}, drop_last={drop_last}")

    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)

    # FIX: раздельные in/out каналы.
    model = build_model(MODEL_TYPE, MODEL_IN_CHANNELS, IN_CHANNELS,
                        LATENT_CHANNELS, IMG_H, IMG_W).to(DEVICE)
    print(model)
    print(f"bottleneck: {LATENT_CHANNELS}×{IMG_H // 8}×{IMG_W // 8}")

    # ---------- sanity-check ----------
    x_dbg, t_dbg = next(iter(train_loader))
    print(f"[sanity] x_in={tuple(x_dbg.shape)}, target={tuple(t_dbg.shape)}")
    with torch.no_grad():
        r_dbg, _ = model(x_dbg.to(DEVICE))
    print(f"[sanity] recon={tuple(r_dbg.shape)} "
          f"(ожидаем [B, {IN_CHANNELS}, {IMG_H}, {IMG_W}])")
    assert r_dbg.shape[1] == IN_CHANNELS, \
        f"out_channels mismatch: recon has {r_dbg.shape[1]}, expected {IN_CHANNELS}"
    del x_dbg, t_dbg, r_dbg

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    D = None
    opt_d = None
    if USE_GAN:
        D = PatchDiscriminator(IN_CHANNELS, base=D_BASE).to(DEVICE)
        opt_d = torch.optim.AdamW(D.parameters(), lr=GAN_LR, betas=(0.5, 0.999))

    criterion_l1 = nn.L1Loss()
    criterion_grad = GradientLoss().to(DEVICE)
    criterion_ssim = SSIMLoss(channel=IN_CHANNELS).to(DEVICE)

    best_val = float("inf")
    epochs_no_improve = 0
    hist = {"train": [], "val": [], "train_recon": [], "train_kl": [],
            "train_adv": [], "d": [], "beta": [], "gan_on": []}

    for epoch in range(1, EPOCHS + 1):
        beta = get_beta(epoch, BETA_VAE_WARMUP_EPOCHS, BETA_VAE_TARGET)
        use_gan_now = USE_GAN and (epoch >= GAN_START_EPOCH)

        stats = train_one_epoch(
            model, train_loader, optimizer, D, opt_d,
            criterion_l1, criterion_grad, criterion_ssim,
            MODEL_TYPE, DEVICE, beta, use_gan_now, GAN_WEIGHT,
        )
        va_loss, va_errors = evaluate(
            model, val_loader, criterion_l1, criterion_grad, criterion_ssim,
            MODEL_TYPE, DEVICE, beta,
        )
        scheduler.step()

        hist["train"].append(stats["loss"])
        hist["val"].append(va_loss)
        hist["train_recon"].append(stats["recon"])
        hist["train_kl"].append(stats["kl"])
        hist["train_adv"].append(stats["adv"])
        hist["d"].append(stats["d"])
        hist["beta"].append(beta)
        hist["gan_on"].append(int(use_gan_now))

        lr_now = optimizer.param_groups[0]["lr"]
        msg = (f"[{epoch:03d}/{EPOCHS}] "
               f"train={stats['loss']:.6f} "
               f"(recon={stats['recon']:.6f}, kl={stats['kl']:.6f}, β={beta:.3f})  "
               f"val={va_loss:.6f}  lr={lr_now:.2e}")
        if use_gan_now:
            msg += f"  adv={stats['adv']:.4f}  D={stats['d']:.4f}"
        print(msg)

        if not np.isfinite(va_loss):
            print(f"  [WARN] val loss = {va_loss} — пропускаем сохранение.")
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("  Ранняя остановка (NaN).")
                break
            continue

        if va_loss < best_val:
            best_val = va_loss
            epochs_no_improve = 0
            threshold = (float(np.percentile(va_errors, THRESHOLD_PERCENTILE))
                         if va_errors.size > 0 else float("nan"))
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "arch": "unet_v2",
                    "model_type": MODEL_TYPE,
                    "beta_vae_target": BETA_VAE_TARGET,
                    "beta_vae_warmup_epochs": BETA_VAE_WARMUP_EPOCHS,
                    "in_channels": IN_CHANNELS,
                    "out_channels": IN_CHANNELS,   # NEW
                    "model_in_channels": MODEL_IN_CHANNELS,
                    "color_mode": COLOR_MODE,
                    "latent_channels": LATENT_CHANNELS,
                    "img_h": IMG_H, "img_w": IMG_W,
                    "keep_aspect": KEEP_ASPECT,
                    "pad_value": PAD_VALUE,
                    "use_ssim": USE_SSIM,
                    "ssim_weight": SSIM_WEIGHT,
                    "grad_loss_w": GRAD_LOSS_W,
                    "l1_weight": L1_WEIGHT,
                    "logvar_min": LOGVAR_MIN,
                    "logvar_max": LOGVAR_MAX,
                    "use_edge_channel": USE_EDGE_CHANNEL,
                    "edge_kernel": EDGE_KERNEL,
                    "median_kernel": MEDIAN_KERNEL,
                    "edge_lo": EDGE_LO, "edge_hi": EDGE_HI,
                    "norm_type": NORM_TYPE,
                    "bottleneck_dilations": BOTTLENECK_DILATIONS,
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

    # ---------- Графики ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(hist["train"], label="train")
    axes[0].plot(hist["val"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(True); axes[0].set_title("Total Loss")

    axes[1].plot(hist["train_recon"], label="recon")
    if MODEL_TYPE == "vae":
        axes[1].plot(hist["train_kl"], label="kl")
        axes[1].set_yscale("log")
    if USE_GAN:
        axes[1].plot(hist["train_adv"], label="adv")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[1].grid(True); axes[1].set_title("Components")

    if MODEL_TYPE == "vae":
        axes[2].plot(hist["beta"], label="β")
    if USE_GAN:
        axes[2].plot(hist["d"], label="D loss")
    axes[2].set_xlabel("epoch"); axes[2].legend(); axes[2].grid(True)
    axes[2].set_title("β / D")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss.png"), dpi=120)

    # ---------- Примеры реконструкций ----------
    ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        x_batch, t_batch = next(iter(val_loader))
        x_batch = x_batch.to(DEVICE)
        t_batch = t_batch.to(DEVICE)
        recon, _ = model(x_batch)

    n = min(8, x_batch.size(0))
    rows = 4 if USE_EDGE_CHANNEL else 3
    fig, axes = plt.subplots(rows, n, figsize=(2.2 * n, 1.8 * rows))
    if n == 1:
        axes = axes.reshape(rows, 1)

    for i in range(n):
        xi = x_batch[i].cpu()
        ti = t_batch[i].cpu()
        ri = recon[i].cpu()
        err = (ti - ri).abs().mean(dim=0)

        def show(ax, img):
            if IN_CHANNELS == 1:
                ax.imshow(img[0] if img.dim() == 3 else img, cmap="gray",
                          vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1) if img.dim() == 3 else img)

        show(axes[0, i], ti); axes[0, i].set_title("target", fontsize=8)
        show(axes[1, i], ri); axes[1, i].set_title("recon", fontsize=8)
        axes[2, i].imshow(err, cmap="hot"); axes[2, i].set_title("|err|", fontsize=8)
        if USE_EDGE_CHANNEL:
            e = xi[IN_CHANNELS].numpy()
            axes[3, i].imshow(e, cmap="gray", vmin=0, vmax=1)
            axes[3, i].set_title("edge", fontsize=8)

        for r in range(rows):
            axes[r, i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recon_examples.png"), dpi=120)

    print(f"\nГотово. Артефакты в: {OUT_DIR}")
    if USE_EDGE_CHANNEL:
        print(f"Кэш рёбер: {EDGE_CACHE_DIR}")


if __name__ == "__main__":
    main()