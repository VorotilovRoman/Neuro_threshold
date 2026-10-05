# utils_aeb/yolo_aeb_utils.py
"""
Утилиты для вкладки YOLO + AE/VAE.

Содержит:
  • модели AE / VAE в трёх архитектурах:
        - conv        : старая conv-сеть, BatchNorm, enc_c=128, шаг 16;
        - conv_gn     : новая conv-сеть, GroupNorm, enc_c=64, шаг 16;
        - unet        : U-Net со skip-связями, BatchNorm, шаг 8;
  • фабрику моделей по config чекпойнта (detect_arch / build_ae_model_from_cfg),
    понимающую явное поле "arch" и умеющую угадывать архитектуру по
    структуре state_dict (если поля нет — старые чекпойнты);
  • препроцессинг изображения под вход AE/VAE;
  • вспомогательные функции для работы с детекциями:
      - маска объекта,
      - морфология внутри маски,
      - кроп / доворот по объекту,
      - заливка фона за пределами объекта,
      - безопасный суффикс имени файла по классу,
      - ограничение размера изображения.

Ничего из этого модуля не зависит от QWidget-ов и не трогает UI.
"""
from import_libs_internal import *


# ============================================================
# AE / VAE — старая архитектура (Conv + BatchNorm, шаг 16)
# ============================================================
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
        # FIX: клиппинг logvar — страховка от exp() → inf на инференсе.
        logvar = torch.clamp(logvar, -10.0, 10.0)
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


# ============================================================
# AE / VAE — новая conv-архитектура (GroupNorm, bottleneck enc_c=64)
# ============================================================
def _gn(channels, max_groups=8):
    """GroupNorm с автоматическим подбором числа групп — как в обучении."""
    g = max_groups
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class ConvEncoderGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            _gn(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            _gn(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            _gn(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=1),
            _gn(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class ConvDecoderGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=2),
            _gn(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            _gn(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            _gn(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class ConvAutoencoderGN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 64
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.encoder = ConvEncoderGN(in_channels)
        self.fc_encode = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoderGN(in_channels)

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        z = self.fc_encode(z)
        z = self.fc_decode(z)
        z = z.view(z.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(z), None


class ConvVAEGN(nn.Module):
    def __init__(self, in_channels, latent_dim, img_h, img_w):
        super().__init__()
        self.enc_c = 64
        self.enc_h, self.enc_w = img_h // 16, img_w // 16
        self.flat_dim = self.enc_c * self.enc_h * self.enc_w
        self.latent_dim = latent_dim
        self.encoder = ConvEncoderGN(in_channels)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = ConvDecoderGN(in_channels)

    def reparameterize(self, mu, logvar):
        # Та же защита, что и в обучении — клиппинг logvar.
        logvar = torch.clamp(logvar, -10.0, 10.0)
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


# ============================================================
# AE / VAE — U-Net (шаг 8, skip-связи, BatchNorm)
# ============================================================
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        b = base
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),
            nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(b, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(b * 2, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
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
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(b * 4 + b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
            nn.Conv2d(b * 4, b * 4, 3, padding=1),
            nn.BatchNorm2d(b * 4), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(b * 2 + b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            nn.Conv2d(b * 2, b * 2, 3, padding=1),
            nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
        )
        self.up0 = nn.Sequential(
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
        # FIX: тот же клиппинг, что и в conv-моделях.
        logvar = torch.clamp(logvar, -10.0, 10.0)
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


# ============================================================
# Фабрика моделей по config чекпойнта
# ============================================================
def detect_arch_from_state_dict(state_dict):
    """
    Определяет архитектуру по ключам state_dict.
    Приоритет: U-Net → старая conv+BN → новая conv+GN.
    """
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.enc0.") for k in keys):
        return "unet"
    # BatchNorm оставляет running_mean/running_var; GroupNorm — нет.
    if any(k.endswith("running_mean") for k in keys):
        return "conv"          # старая архитектура (BatchNorm, enc_c=128)
    return "conv_gn"           # новая (GroupNorm, enc_c=64)


def detect_arch(cfg, state_dict=None):
    """
    Определяет архитектуру AE/VAE.
    1) Явный 'arch' в config — приоритет.
    2) Наличие 'latent_channels' → unet.
    3) Фолбэк по state_dict.
    4) Фолбэк по умолчанию — старая conv.
    """
    arch = cfg.get("arch")
    if arch in ("unet", "conv", "conv_gn"):
        return arch
    if "latent_channels" in cfg:
        return "unet"
    if state_dict is not None:
        return detect_arch_from_state_dict(state_dict)
    return "conv"


def build_ae_model_from_cfg(cfg, arch=None):
    """
    Собирает модель AE/VAE по config чекпойнта.
    arch можно передать явно (обычно — результат detect_arch), иначе
    определяется автоматически по config.
    """
    if arch is None:
        arch = detect_arch(cfg)

    model_type = cfg["model_type"]
    in_channels = int(cfg["in_channels"])
    img_h = int(cfg["img_h"])
    img_w = int(cfg["img_w"])

    if arch == "unet":
        latent_channels = int(cfg["latent_channels"])
        if model_type == "vae":
            return UNetVAE(in_channels, latent_channels, img_h, img_w)
        return UNetAE(in_channels, latent_channels, img_h, img_w)

    latent_dim = int(cfg["latent_dim"])

    if arch == "conv_gn":
        if model_type == "vae":
            return ConvVAEGN(in_channels, latent_dim, img_h, img_w)
        return ConvAutoencoderGN(in_channels, latent_dim, img_h, img_w)

    # arch == "conv" — старая архитектура (BatchNorm, enc_c=128)
    if model_type == "vae":
        return ConvVAE(in_channels, latent_dim, img_h, img_w)
    return ConvAutoencoder(in_channels, latent_dim, img_h, img_w)


# ============================================================
# Препроцессинг изображения под вход AE/VAE
# ============================================================
def preprocess_image_for_ae(img_bgr, cfg):
    """
    Приводит произвольный BGR-снимок к тензору [C, H, W] float32 [0..1],
    совместимому с моделью AE/VAE. Возвращает (tensor, valid_region),
    где valid_region = (x1, y1, x2, y2) — область исходного изображения
    внутри padded-кадра.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Пустое изображение для AE/VAE")

    H = int(cfg["img_h"]); W = int(cfg["img_w"])
    keep_aspect = bool(cfg.get("keep_aspect", True))
    pad_value = int(cfg.get("pad_value", 0))
    in_channels = int(cfg.get("in_channels", 3))

    img = img_bgr
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Неподдерживаемая форма изображения: {img.shape}")

    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            f = img.astype(np.float32)
            mx = float(f.max()) if f.size else 0.0
            img = np.zeros(img.shape, dtype=np.uint8) if mx <= 0 else \
                np.clip(f * (255.0 / mx), 0, 255).astype(np.uint8)
        else:
            f = img.astype(np.float32)
            mn = float(f.min()); mx = float(f.max())
            img = np.zeros(img.shape, dtype=np.uint8) if mx - mn < 1e-9 else \
                np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif in_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Неподдерживаемое число входных каналов: {in_channels}")

    h, w = img.shape[:2]
    if keep_aspect:
        scale = min(W / w, H / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if in_channels == 1:
            canvas = np.full((H, W), pad_value, dtype=np.uint8)
        else:
            canvas = np.full((H, W, 3), pad_value, dtype=np.uint8)
        ox = (W - nw) // 2
        oy = (H - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        padded = canvas
        valid_region = (ox, oy, ox + nw, oy + nh)
    else:
        padded = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        valid_region = (0, 0, W, H)

    x = padded.astype(np.float32) / 255.0
    if in_channels == 1:
        x = x[None, :, :]
    else:
        x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    return torch.from_numpy(x), valid_region


# ============================================================
# Геометрия: угол по длинной стороне четырёхугольника
# ============================================================
def longest_edge_angle_deg(corners):
    n = len(corners)
    best_len = -1.0
    best_angle = 0.0
    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L > best_len:
            best_len = L
            best_angle = math.degrees(math.atan2(dy, dx))
    return best_angle


# ============================================================
# Работа с маской выбранного объекта
# ============================================================
def build_detection_mask(det, W, H):
    """
    Бинарная маска (uint8, 0/255) для детекции.
    Для seg используется mask_xy (точный контур), иначе — corners.
    Возвращает None, если контур некорректен.
    """
    if det is None:
        return None
    if det.get("kind") == "seg" and det.get("mask_xy"):
        poly = np.asarray(det["mask_xy"], dtype=np.float32).reshape(-1, 2)
    else:
        poly = np.asarray(det["corners"], dtype=np.float32).reshape(-1, 2)

    if poly.shape[0] < 3:
        return None

    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def morph_image_in_mask(img, mask, erosion_px, dilation_px,
                        shape_name="Rectangle", contour_only=True):
    """
    Применяет эрозию и/или дилатацию к изображению, сохраняя результат
    только внутри маски (вне маски — исходный img).

    Если contour_only=True — результат сохраняется не по всей площади
    маски, а только в граничной полосе (ring), толщина полосы привязана
    к размеру ядра операции.
    """
    if shape_name == "Rectangle":
        shape = cv2.MORPH_RECT
    elif shape_name == "Ellipse":
        shape = cv2.MORPH_ELLIPSE
    else:
        shape = cv2.MORPH_CROSS

    # -------- выбор эффективной маски: полная площадь либо контур --------
    effective_mask = mask
    if contour_only:
        ring_px = max(int(erosion_px), int(dilation_px), 3)
        if ring_px % 2 == 0:
            ring_px += 1
        ring_px = max(3, ring_px)
        ring_kernel = cv2.getStructuringElement(shape, (ring_px, ring_px))
        eroded_mask = cv2.erode(mask, ring_kernel, iterations=1)
        ring_mask = cv2.subtract(mask, eroded_mask)
        if ring_mask.any():
            effective_mask = ring_mask
        # если маска слишком тонкая и кольцо пустое — работаем по всей маске

    if img.ndim == 3:
        mask_bool = effective_mask[..., None] > 0
    else:
        mask_bool = effective_mask > 0

    if erosion_px > 1:
        k = erosion_px if erosion_px % 2 == 1 else erosion_px + 1
        k = max(3, k)
        kernel = cv2.getStructuringElement(shape, (k, k))
        eroded = cv2.erode(img, kernel, iterations=1)
        img = np.where(mask_bool, eroded, img)

    if dilation_px > 1:
        k = dilation_px if dilation_px % 2 == 1 else dilation_px + 1
        k = max(3, k)
        kernel = cv2.getStructuringElement(shape, (k, k))
        dilated = cv2.dilate(img, kernel, iterations=1)
        img = np.where(mask_bool, dilated, img)

    return img


def apply_object_morphology(img, det, erosion_px, dilation_px,
                            shape_name="Rectangle", contour_only=True):
    """
    Полный сценарий морфологии по выбранной детекции.
    Возвращает (img_out, info) где info — словарь с деталями для лога
    либо None, если операция не применялась.
    """
    if erosion_px <= 1 and dilation_px <= 1:
        return img, None
    if det is None:
        return img, None

    H, W = img.shape[:2]
    mask = build_detection_mask(det, W, H)
    if mask is None or not mask.any():
        return img, None

    img = morph_image_in_mask(img, mask, erosion_px, dilation_px,
                              shape_name=shape_name, contour_only=contour_only)

    info = {
        "kind": det.get("kind"),
        "cls_name": det.get("cls_name"),
        "contour_mode": "contour" if contour_only else "full",
        "erosion": int(erosion_px),
        "dilation": int(dilation_px),
        "mask_px": int((mask > 0).sum()),
    }
    return img, info


# ============================================================
# Заливка фона за пределами объекта
# ============================================================
def fill_outside_object(img, det, fill_mode,
                        rotation_matrix=None, crop_offset=(0, 0)):
    """
    Заливает пиксели вне маски объекта (fill_mode: 'none' / 'black' / 'white').
    rotation_matrix и crop_offset позволяют применить маску к уже
    повёрнутому/обрезанному изображению.
    """
    if fill_mode == "none":
        return img
    color_val = 0 if fill_mode == "black" else 255

    H, W = img.shape[:2]

    if det.get("kind") == "seg" and det.get("mask_xy"):
        poly = np.asarray(det["mask_xy"], dtype=np.float32).reshape(-1, 2)
    else:
        poly = np.asarray(det["corners"], dtype=np.float32).reshape(-1, 2)

    if poly.shape[0] < 3:
        return img

    if rotation_matrix is not None:
        ones = np.ones((poly.shape[0], 1), dtype=np.float32)
        poly_h = np.hstack([poly, ones])
        poly = (rotation_matrix @ poly_h.T).T

    ox, oy = crop_offset
    poly = poly.copy()
    poly[:, 0] -= float(ox)
    poly[:, 1] -= float(oy)

    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)

    outside = (mask == 0)
    if img.ndim == 2:
        img[outside] = color_val
    else:
        img[outside] = (color_val,) * img.shape[2]
    return img


# ============================================================
# Кроп / доворот по выбранной детекции
# ============================================================
def crop_to_detection(img, det, padding, off_x, off_y, fill_mode):
    """Обрезает изображение по xyxy объекта с учётом padding и смещений."""
    x1, y1, x2, y2 = det["xyxy"]
    H, W = img.shape[:2]
    xi1 = max(0, int(round(x1)) - padding + off_x)
    yi1 = max(0, int(round(y1)) - padding + off_y)
    xi2 = min(W, int(round(x2)) + padding + off_x)
    yi2 = min(H, int(round(y2)) + padding + off_y)
    if xi2 <= xi1 or yi2 <= yi1:
        return img
    cropped = img[yi1:yi2, xi1:xi2].copy()
    cropped = fill_outside_object(
        cropped, det, fill_mode, rotation_matrix=None, crop_offset=(xi1, yi1)
    )
    return cropped


def rotate_with_optional_crop(img, det, auto_crop, padding, off_x, off_y, fill_mode):
    """
    Доворачивает изображение так, чтобы длинная сторона объекта стала
    горизонтальной. Если auto_crop=True — обрезает по границам объекта
    с учётом padding и смещений.
    """
    angle_deg = det["angle_deg"]
    cx, cy = det["center"]
    H, W = img.shape[:2]

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
    new_w = int(H * sin_a + W * cos_a)
    new_h = int(H * cos_a + W * sin_a)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    if auto_crop:
        corners = np.array(det["corners"], dtype=np.float32)
        ones = np.ones((corners.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners, ones])
        new_corners = (M @ corners_h.T).T
        x1n = int(np.floor(new_corners[:, 0].min())) - padding + off_x
        y1n = int(np.floor(new_corners[:, 1].min())) - padding + off_y
        x2n = int(np.ceil(new_corners[:, 0].max())) + padding + off_x
        y2n = int(np.ceil(new_corners[:, 1].max())) + padding + off_y
        xi1 = max(0, x1n); yi1 = max(0, y1n)
        xi2 = min(rotated.shape[1], x2n)
        yi2 = min(rotated.shape[0], y2n)
        if xi2 > xi1 and yi2 > yi1:
            rotated = rotated[yi1:yi2, xi1:xi2].copy()
            rotated = fill_outside_object(
                rotated, det, fill_mode, rotation_matrix=M, crop_offset=(xi1, yi1)
            )
    return rotated


# ============================================================
# Имя файла и ограничение размера
# ============================================================
def cls_suffix_from_detection(det, index=None):
    """
    Безопасный суффикс имени файла по классу объекта и его номеру в списке.

    index — 1-based позиция объекта в текущем списке детекций (может быть None).
    Итоговый вид: "ClassName_N" или просто "ClassName", если index не задан
    либо <= 0. Пустая строка, если объект/имя класса не заданы.
    """
    if det is None:
        return ""
    cls_name = det.get("cls_name")
    if not cls_name:
        return ""
    safe = "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in str(cls_name)
    ).strip("_")
    if not safe:
        return ""
    if index is not None and int(index) > 0:
        return f"{safe}_{int(index)}"
    return safe


def limit_image_size(img, max_side):
    """Пропорционально уменьшает изображение, если длинная сторона > max_side."""
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = float(max_side) / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)