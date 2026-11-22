"""Versioned image preparation, shared exactly by training and inference."""

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

_EDGE_SCHARR = torch.tensor(
    [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]], dtype=torch.float32
)


_EDGE_SOBEL = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
)


def _median_blur_torch(x, k=3):
    """x: [B,C,H,W]. Медианный фильтр через unfold (совместим с обучением)."""
    if k <= 1:
        return x
    p = k // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)
    patches = patches.contiguous().view(*x.shape, k * k)
    return patches.median(dim=-1).values


def compute_edge_map_torch(
    img_tensor, edge_kernel="scharr", median_kernel=3, edge_lo=0.10, edge_hi=0.35
):
    """
    img_tensor: [C,H,W] или [B,C,H,W], float32 в [0,1].
    Возвращает [1,H,W] или [B,1,H,W] — сохраняет batch-мерность.
    """
    if (
        edge_kernel not in ("scharr", "sobel")
        or type(median_kernel) is not int
        or median_kernel < 1
        or median_kernel % 2 != 1
    ):
        raise ValueError("Некорректный edge_kernel или median_kernel.")
    if not np.isfinite([edge_lo, edge_hi]).all() or edge_lo < 0 or edge_hi <= edge_lo:
        raise ValueError("Нужно 0 ≤ edge_lo < edge_hi.")
    if (
        img_tensor.ndim not in (3, 4)
        or img_tensor.shape[-3] not in (1, 3)
        or not torch.isfinite(img_tensor).all()
    ):
        raise ValueError("Edge ожидает конечный CHW/BCHW снимок с 1 или 3 каналами.")
    if median_kernel // 2 >= min(img_tensor.shape[-2:]):
        raise ValueError("Медианное окно больше изображения.")
    squeeze = False
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze = True

    if img_tensor.size(1) == 3:
        g = (
            0.299 * img_tensor[:, 0:1]
            + 0.587 * img_tensor[:, 1:2]
            + 0.114 * img_tensor[:, 2:3]
        )
    else:
        g = img_tensor

    if median_kernel > 1:
        g = _median_blur_torch(g, int(median_kernel))

    kx = _EDGE_SCHARR if edge_kernel == "scharr" else _EDGE_SOBEL
    kx = kx.view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    scale = 16.0 if edge_kernel == "scharr" else 4.0
    kx = (kx / scale).to(img_tensor.device, img_tensor.dtype)
    ky = (ky / scale).to(img_tensor.device, img_tensor.dtype)

    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    mid = 0.5 * (float(edge_lo) + float(edge_hi))
    s = 6.0 / max(float(edge_hi) - float(edge_lo), 1e-6)
    edge = torch.sigmoid(s * (mag - mid))

    return edge[0] if squeeze else edge


def _legacy_preprocess(img_bgr, cfg):
    """
    Приводит произвольный BGR-снимок к тензору [C, H, W] float32 [0..1],
    совместимому с моделью AE/VAE.

    Если cfg.use_edge_channel=True (или model_in_channels > in_channels) —
    автоматически добавляет второй канал: карту рёбер (Scharr + мягкий порог),
    посчитанную от того же (ресайзнутого, с паддингом) изображения.
    Параметры edge-канала берутся из cfg (те же, что сохраняются при обучении).

    Возвращает (tensor, valid_region),
    где valid_region = (x1, y1, x2, y2) — область исходного изображения
    внутри padded-кадра.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Пустое изображение для AE/VAE")

    H = int(cfg["img_h"])
    W = int(cfg["img_w"])
    keep_aspect = bool(cfg.get("keep_aspect", True))
    pad_value = int(cfg.get("pad_value", 0))
    in_channels = int(cfg.get("in_channels", 3))

    img = _source_uint8(img_bgr)

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
        canvas[oy : oy + nh, ox : ox + nw] = resized
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
    x_t = torch.from_numpy(x)

    return append_edge(x_t, cfg), valid_region


def _source_uint8(img_bgr):
    if (
        not isinstance(img_bgr, np.ndarray)
        or not img_bgr.size
        or not np.isfinite(img_bgr).all()
    ):
        raise ValueError("Снимок пуст или содержит NaN/Inf.")
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
            img = (
                np.zeros(img.shape, dtype=np.uint8)
                if mx <= 0
                else np.clip(f * (255.0 / mx), 0, 255).astype(np.uint8)
            )
        else:
            f = img.astype(np.float32)
            mn = float(f.min())
            mx = float(f.max())
            img = (
                np.zeros(img.shape, dtype=np.uint8)
                if mx - mn < 1e-9
                else np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
            )

    return img


def append_edge(image, cfg):
    channels = int(cfg.get("in_channels", image.shape[-3]))
    expected = int(
        cfg.get(
            "model_in_channels", channels + bool(cfg.get("use_edge_channel", False))
        )
    )
    if expected not in (channels, channels + 1):
        raise ValueError("Некорректное число каналов AE.")
    if expected == channels + 1:
        edge = compute_edge_map_torch(
            image,
            edge_kernel=cfg.get("edge_kernel", "scharr"),
            median_kernel=cfg.get("median_kernel", 3),
            edge_lo=cfg.get("edge_lo", 0.1),
            edge_hi=cfg.get("edge_hi", 0.35),
        )
        return torch.cat((image, edge), dim=-3)
    return image


def preprocess_image_for_ae(image, cfg):
    """Return CHW image(+edge) and valid pixel bounds, with an explicit version.

    New training and inference both use PIL bilinear resampling. The earlier
    OpenCV transform remains explicit for checkpoints created before schema 2.
    """
    for key in ("img_h", "img_w"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key}: нужен положительный целый размер.")
    if cfg.get("in_channels", 3) not in (1, 3):
        raise ValueError("in_channels должен быть 1 или 3.")
    version = cfg.get("preprocessing_version", "legacy-opencv-v1")
    if version == "legacy-opencv-v1":
        return _legacy_preprocess(image, cfg)
    if version != "ae-pil-v1":
        raise ValueError("Неизвестная версия предобработки AE.")
    bgr = _source_uint8(image)
    mode = "L" if cfg["in_channels"] == 1 else "RGB"
    source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert(mode)
    height, width = cfg["img_h"], cfg["img_w"]
    if cfg.get("keep_aspect", True):
        scale = min(width / source.width, height / source.height)
        nw, nh = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        resized = source.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new(mode, (width, height), color=int(cfg.get("pad_value", 0)))
        x, y = (width - nw) // 2, (height - nh) // 2
        canvas.paste(resized, (x, y))
        region = (x, y, x + nw, y + nh)
    else:
        canvas = source.resize((width, height), Image.Resampling.BILINEAR)
        region = (0, 0, width, height)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array[None] if mode == "L" else array.transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    return append_edge(tensor, cfg), region
