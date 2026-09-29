# yolo_test.py
"""
Логика приложения YOLO Inspect + AE/VAE.

Содержит:
  • архитектуры AE/VAE (Conv — старая, U-Net — новая) и фабрики;
  • препроцессинг снимка под вход AE/VAE;
  • утилиту longest_edge_angle_deg;
  • подкласс YoloInspectWindow с вычислительной логикой:
    – инференс YOLO,
    – применение AE/VAE и рендер карты ошибок,
    – авто- и ручные трансформации снимка,
    – выбор объектов и работа с устройствами.

UI-слой (разметка окна, загрузка моделей и изображений, сохранение,
зум/панорама) — в yolo_test_ui.py.
"""
import sys
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt5.QtCore import QTimer

from yolo_test_ui import (
    YoloInspectWindow as _YoloInspectWindow,
    ZoomableImageView,          # реэкспорт для удобства
    cv_to_qpixmap,
)


# ============================================================
# Модели AE / VAE — старая архитектура (Conv, шаг 16)
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(h_dec.size(0), self.enc_c, self.enc_h, self.enc_w)
        return self.decoder(h_dec), (mu, logvar)


# ============================================================
# Модели AE / VAE — новая архитектура (U-Net, шаг 8, skip)
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


# ============================================================
# Фабрика: выбор архитектуры по cfg
# ============================================================
def detect_arch(cfg):
    arch = cfg.get("arch")
    if arch in ("unet", "conv"):
        return arch
    if "latent_channels" in cfg:
        return "unet"
    return "conv"


def build_ae_model_from_cfg(cfg):
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
    if model_type == "vae":
        return ConvVAE(in_channels, latent_dim, img_h, img_w)
    return ConvAutoencoder(in_channels, latent_dim, img_h, img_w)


def build_ae_model(model_type, in_channels, latent, img_h, img_w):
    if model_type == "vae":
        return ConvVAE(in_channels, latent, img_h, img_w)
    return ConvAutoencoder(in_channels, latent, img_h, img_w)


# ============================================================
# Препроцессинг
# ============================================================
def preprocess_image_for_ae(img_bgr, cfg):
    """
    Приводит произвольный снимок к тензору, совместимому с моделью AE/VAE.
    Гарантирует число каналов == cfg["in_channels"] и dtype float32 [0..1].

    Поддерживает:
      • 8-битные BGR / RGB / grayscale;
      • 16-битные TIFF (10/12/14/16 бит) — нормируются к 8 битам;
      • 32/64-битные float-изображения — нормируются по min/max;
      • 4-канальные BGRA / RGBA — альфа отбрасывается.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("Пустое изображение для AE/VAE")

    H = int(cfg["img_h"]); W = int(cfg["img_w"])
    keep_aspect = bool(cfg.get("keep_aspect", True))
    pad_value = int(cfg.get("pad_value", 0))
    in_channels = int(cfg.get("in_channels", 3))

    # --- 1. Приводим к 3-канальному BGR uint8 -------------------------
    img = img_bgr
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Неподдерживаемая форма изображения: {img.shape}")

    # --- 2. Приводим dtype к uint8 (TIFF часто 16-битный) -------------
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            f = img.astype(np.float32)
            mx = float(f.max()) if f.size else 0.0
            if mx <= 0.0:
                img = np.zeros(img.shape, dtype=np.uint8)
            else:
                img = np.clip(f * (255.0 / mx), 0, 255).astype(np.uint8)
        elif img.dtype in (np.float32, np.float64):
            f = img.astype(np.float32)
            mn = float(f.min()); mx = float(f.max())
            if mx - mn < 1e-9:
                img = np.zeros(img.shape, dtype=np.uint8)
            else:
                img = np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
        else:
            f = img.astype(np.float32)
            mn = float(f.min()); mx = float(f.max())
            if mx - mn < 1e-9:
                img = np.zeros(img.shape, dtype=np.uint8)
            else:
                img = np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

    # --- 3. Каналы — строго по in_channels модели ---------------------
    if in_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif in_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(
            f"Неподдерживаемое число входных каналов модели: {in_channels} "
            f"(ожидается 1 или 3)"
        )

    # --- 4. Resize / letterbox ----------------------------------------
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

    # --- 5. В тензор [C, H, W] float32 [0..1] -------------------------
    x = padded.astype(np.float32) / 255.0
    if in_channels == 1:
        x = x[None, :, :]
    else:
        x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    return torch.from_numpy(x), valid_region


# ============================================================
# Утилита: угол самой длинной стороны четырёхугольника
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
# Логика окна: подкласс базового UI-класса
# ============================================================
class YoloInspectWindow(_YoloInspectWindow):

    # --------------------------------------------------------
    # Инициализация/переопределения поверх UI-базы
    # --------------------------------------------------------
    def __init__(self):
        # Поля, которых нет в UI-базе, — до super().__init__(),
        # т.к. UI-конструктор вызывает _build_ui и _update_actions_enabled.
        self.annotated_image = None
        self._manual_ops = []
        self._yolo_reinfer_timer = None
        super().__init__()

        # Debounce для перезапуска инференса при изменении
        # conf / iou / imgsz / device: спиннеры шлют valueChanged
        # на каждый тик, а инференс недешёвый.
        self._yolo_reinfer_timer = QTimer(self)
        self._yolo_reinfer_timer.setSingleShot(True)
        self._yolo_reinfer_timer.setInterval(300)  # ms
        self._yolo_reinfer_timer.timeout.connect(self._do_reinfer)

    def _show_current(self):
        # Сбрасываем накопленный стек при переключении изображения.
        self.annotated_image = None
        self._manual_ops = []
        # Отменяем отложенный реинференс, если он был запланирован:
        # новый снимок и так прогонится через _auto_apply_all().
        if self._yolo_reinfer_timer is not None:
            self._yolo_reinfer_timer.stop()
        super()._show_current()

    # --------------------------------------------------------
    # Устройства
    # --------------------------------------------------------
    def _resolve_device(self):
        d = self.device_combo.currentText()
        if d == "auto":
            return "0" if self._cuda_available() else "cpu"
        return d

    def _resolve_torch_device(self):
        d = self.device_combo.currentText()
        if d == "auto":
            if self._cuda_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if d == "cpu":
            return torch.device("cpu")
        if d == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        try:
            idx = int(d)
            if torch.cuda.is_available() and idx < torch.cuda.device_count():
                return torch.device(f"cuda:{idx}")
            return torch.device("cpu")
        except ValueError:
            return torch.device("cpu")

    @staticmethod
    def _cuda_available():
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    # --------------------------------------------------------
    # Единая сборка displayed_image
    # --------------------------------------------------------
    def _show_boxes_enabled(self) -> bool:
        """Текущее состояние чекбокса «Показывать боксы» (если он есть)."""
        chk = getattr(self, "chk_show_boxes", None)
        if chk is None:
            return True
        try:
            return bool(chk.isChecked())
        except Exception:
            return True

    def _rebuild_displayed(self):
        """
        displayed_image = base (annotated или original)
                          → авто(обрезка/доворот)
                          → ручные ops.
        Никогда не трогает current_detections и original_image.
        """
        if self.original_image is None:
            return

        # Если пользователь отключил отображение разметки YOLO —
        # используем оригинал как базис, но детекции сохраняем
        # (авто-обрезка/доворот продолжают работать).
        if self.annotated_image is not None and self._show_boxes_enabled():
            base = self.annotated_image
            mode = "annotated"
        else:
            base = self.original_image
            mode = "original"

        img = base.copy()

        do_rotate = self.chk_auto_rotate.isChecked()
        do_crop = self.chk_auto_crop.isChecked()

        if (do_rotate or do_crop) and self.current_detections:
            det = self._get_selected_detection()
            if det is not None:
                if do_rotate and det.get("angle_deg") is not None:
                    img = self._rotate_with_optional_crop(img, det,
                                                          auto_crop=do_crop)
                    mode = "rotated"
                elif do_crop:
                    img = self._crop_to_detection(img, det)
                    mode = "cropped"

        if self._manual_ops:
            for op in self._manual_ops:
                if op == "rot_cw":
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif op == "rot_ccw":
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif op == "flip_v":
                    img = cv2.flip(img, 0)
                elif op == "flip_h":
                    img = cv2.flip(img, 1)
            mode = "manual"

        self.displayed_image = img
        self.display_mode = mode
        self._set_image(self.displayed_image)
        self._update_actions_enabled()

    # --------------------------------------------------------
    # Автозапуск YOLO + AE
    # --------------------------------------------------------
    def _auto_apply_ae(self):
        if self.ae_model is None or self.displayed_image is None:
            return
        try:
            self.apply_ae(silent=True)
        except Exception as e:
            self._log(f"[AE/VAE auto] ошибка: {e}")

    def _auto_apply_all(self):
        """
        Полный цикл для текущего изображения:
        YOLO (если модель загружена) → пересборка вида → AE/VAE.
        """
        if self.original_image is None:
            return

        if self.model is not None:
            # _run_inference_on_current сам заполнит annotated_image,
            # current_detections, обновит список и вызовет _rebuild_displayed.
            self._run_inference_on_current()
        else:
            self.annotated_image = None
            self.current_detections = []
            self._rebuild_displayed()
            self._update_objects_list()

        self._auto_apply_ae()

    # --------------------------------------------------------
    # Реакция на изменение YOLO-параметров (conf / iou / imgsz / device)
    # --------------------------------------------------------
    def _on_yolo_params_changed(self, *_):
        """
        Вызывается спиннерами/комбобоксом параметров YOLO.
        Не запускает инференс сразу — только перезапускает debounce-таймер,
        чтобы подряд идущие тики (стрелками, колесом) не гоняли модель
        несколько раз.
        """
        if self.model is None or self.original_image is None:
            return
        if self._yolo_reinfer_timer is None:
            return
        self._yolo_reinfer_timer.start()

    def _do_reinfer(self):
        """
        Фактический перезапуск: YOLO → пересборка вида → AE/VAE.
        Делается через _auto_apply_all(), чтобы не дублировать логику
        и корректно учесть отсутствие модели.
        """
        if self.model is None or self.original_image is None:
            return

        self.statusBar().showMessage("Обновление результатов…")
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        self._auto_apply_all()

    # --------------------------------------------------------
    # Применение AE/VAE
    # --------------------------------------------------------
    def apply_ae(self, silent=False):
        if self.ae_model is None:
            if not silent:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Нет модели AE/VAE",
                                    "Сначала загрузите чекпойнт AE/VAE.")
            return
        if self.displayed_image is None:
            if not silent:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Нет изображения",
                                    "Сначала откройте изображение.")
            return

        target = self._resolve_torch_device()
        if target != self.ae_device:
            try:
                self.ae_model.to(target)
                self.ae_device = target
                self._log(f"[AE/VAE] Модель переведена на {target}")
            except Exception as e:
                self._log(f"[AE/VAE] Не удалось перевести на {target}: {e}.")

        device = self.ae_device

        if not silent:
            self.statusBar().showMessage("Инференс AE/VAE…")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()

        try:
            x, valid_region = preprocess_image_for_ae(self.displayed_image, self.ae_cfg)
            x_t = x.unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _ = self.ae_model(x_t)
            recon = recon.cpu()
            x_cpu = x.unsqueeze(0).cpu()
        except Exception as e:
            if not silent:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Ошибка инференса AE/VAE", str(e))
            else:
                self._log(f"[AE/VAE auto] ошибка инференса: {e}")
            self.statusBar().showMessage("Ошибка AE/VAE")
            return

        err = (x_cpu - recon).abs().mean(dim=1)[0].numpy()

        self._ae_err_hw = err
        self._ae_valid_region = valid_region
        self._ae_base_bgr = self.displayed_image.copy()
        self._ae_threshold = self._compute_threshold()

        self._render_ae_view()

        x1, y1, x2, y2 = valid_region
        roi = err[y1:y2, x1:x2]
        if roi.size > 0:
            mean_err = float(roi.mean())
            max_err = float(roi.max())
            thr = self._ae_threshold
            frac_above = float((roi > thr).mean()) * 100 if np.isfinite(thr) else 0.0
            self.statusBar().showMessage(
                f"AE/VAE: mean err={mean_err:.4f}, max={max_err:.4f}, "
                f"above-thr={frac_above:.2f}%"
            )
            if not silent:
                self._log("-" * 60)
                self._log(f"[AE/VAE] arch={self.ae_arch}  "
                          f"тензор входа: {tuple(x.shape)}  "
                          f"valid ROI: x[{x1}:{x2}] y[{y1}:{y2}]  "
                          f"устройство: {device}")
                self._log(f"[AE/VAE] error (valid ROI): mean={mean_err:.6f} "
                          f"max={max_err:.6f}")
                if np.isfinite(thr):
                    src = "ckpt" if (self.chk_ae_use_ckpt_thr.isChecked()
                                     and self._ae_ckpt_threshold is not None) else "percentile"
                    self._log(f"[AE/VAE] threshold = {thr:.6f} (src={src}); "
                              f"доля пикселей выше порога: {frac_above:.3f}%")
                self._log("-" * 60)
        else:
            self.statusBar().showMessage("AE/VAE: пустая валидная область")

    # --------------------------------------------------------
    # Порог и рендер карты
    # --------------------------------------------------------
    def _compute_threshold(self):
        if self._ae_err_hw is None or self._ae_valid_region is None:
            return float("nan")
        if self.chk_ae_use_ckpt_thr.isChecked() and \
                self._ae_ckpt_threshold is not None and \
                np.isfinite(self._ae_ckpt_threshold):
            return float(self._ae_ckpt_threshold)
        x1, y1, x2, y2 = self._ae_valid_region
        roi = self._ae_err_hw[y1:y2, x1:x2]
        if roi.size == 0:
            return float("nan")
        return float(np.percentile(roi, float(self.ae_perc_spin.value())))

    def _on_ae_threshold_params_changed(self, *_):
        if self._ae_err_hw is None:
            return
        self._ae_threshold = self._compute_threshold()
        self._render_ae_view()

    def _on_ae_render_params_changed(self, *_):
        if self._ae_err_hw is None:
            return
        self._render_ae_view()

    def _on_ae_alpha_changed(self, v):
        self.ae_alpha_label.setText(str(v))
        self._on_ae_render_params_changed()

    def _render_ae_view(self):
        if self._ae_err_hw is None or self._ae_base_bgr is None:
            return

        err = self._ae_err_hw
        base = self._ae_base_bgr
        H, W = base.shape[:2]

        if self._ae_valid_region is not None:
            x1, y1, x2, y2 = self._ae_valid_region
            err_crop = err[y1:y2, x1:x2]
            if err_crop.size == 0:
                err_crop = err
        else:
            err_crop = err

        err_full = cv2.resize(err_crop, (W, H), interpolation=cv2.INTER_LINEAR)

        lo = float(np.percentile(err_full, 1))
        hi = float(np.percentile(err_full, 99))
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        norm = np.clip((err_full - lo) / (hi - lo), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

        overlay_on = self.chk_ae_overlay.isChecked()
        show_mask = self.chk_ae_show_mask.isChecked()
        alpha_heat = float(self.ae_alpha_slider.value()) / 100.0

        if overlay_on:
            err_img = cv2.addWeighted(base, 1.0 - alpha_heat,
                                      heat, alpha_heat, 0.0)
        else:
            err_img = heat

        if show_mask and np.isfinite(self._ae_threshold):
            mask_small = (err_crop > self._ae_threshold).astype(np.uint8)
            mask_full = cv2.resize(mask_small, (W, H),
                                   interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(err_img, contours, -1, (255, 255, 255), 2)

        self._error_bgr = err_img
        self._error_pixmap = cv_to_qpixmap(err_img)

        self.error_label.set_image_pixmap(self._error_pixmap, reset_view=False)
        self.error_label.setVisible(True)

        self._update_actions_enabled()

    # --------------------------------------------------------
    # Авто-трансформации (единая точка входа)
    # --------------------------------------------------------
    def _crop_to_detection(self, img, det):
        x1, y1, x2, y2 = det["xyxy"]
        padding = self.crop_padding_spin.value()
        H, W = img.shape[:2]
        xi1 = max(0, int(round(x1)) - padding)
        yi1 = max(0, int(round(y1)) - padding)
        xi2 = min(W, int(round(x2)) + padding)
        yi2 = min(H, int(round(y2)) + padding)
        if xi2 <= xi1 or yi2 <= yi1:
            return img
        return img[yi1:yi2, xi1:xi2].copy()

    def _rotate_with_optional_crop(self, img, det, auto_crop):
        angle_deg = det["angle_deg"]
        cx, cy = det["center"]
        H, W = img.shape[:2]
        padding = self.crop_padding_spin.value()

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
            x1n = int(np.floor(new_corners[:, 0].min())) - padding
            y1n = int(np.floor(new_corners[:, 1].min())) - padding
            x2n = int(np.ceil(new_corners[:, 0].max())) + padding
            y2n = int(np.ceil(new_corners[:, 1].max())) + padding
            xi1 = max(0, x1n); yi1 = max(0, y1n)
            xi2 = min(rotated.shape[1], x2n)
            yi2 = min(rotated.shape[0], y2n)
            if xi2 > xi1 and yi2 > yi1:
                rotated = rotated[yi1:yi2, xi1:xi2].copy()
        return rotated

    def _apply_auto_transforms(self):
        # Совместимость с прежним API: просто пересобрать вид.
        self._rebuild_displayed()

    def _refresh_current_view(self):
        self._rebuild_displayed()
        self._auto_apply_ae()

    def _on_auto_transform_changed(self, *_):
        if self.original_image is None:
            return
        self._rebuild_displayed()
        self._auto_apply_ae()

    def _on_padding_changed(self, _v):
        if self.original_image is None:
            return
        if not (self.chk_auto_crop.isChecked() or self.chk_auto_rotate.isChecked()):
            return
        self._rebuild_displayed()
        self._auto_apply_ae()

    def _on_object_selection_changed(self, _row):
        if self._updating_objects_list:
            return
        if self.original_image is None:
            return
        if not (self.chk_auto_crop.isChecked() or self.chk_auto_rotate.isChecked()):
            return
        self._rebuild_displayed()
        self._auto_apply_ae()

    # --------------------------------------------------------
    # Тумблер отображения разметки YOLO
    # --------------------------------------------------------
    def _on_show_boxes_changed(self, *_):
        """
        Реакция на переключение чекбокса «Показывать боксы».
        Пересобирает отображаемое изображение (и, если загружена AE/VAE,
        повторно считает карту ошибок от нового базиса).
        """
        if self.original_image is None:
            return
        self._rebuild_displayed()
        self._auto_apply_ae()

    # --------------------------------------------------------
    # Инференс YOLO
    # --------------------------------------------------------
    def _run_inference_on_current(self):
        if self.model is None or self.original_image is None:
            return

        conf = float(self.conf_spin.value())
        iou = float(self.iou_spin.value())
        imgsz = int(self.imgsz_spin.value())
        device = self._resolve_device()

        self.statusBar().showMessage(
            f"Инференс… conf={conf} iou={iou} imgsz={imgsz} device={device}"
        )
        from PyQt5.QtWidgets import QApplication, QMessageBox
        QApplication.processEvents()

        try:
            results = self.model(
                self.original_image,
                conf=conf, iou=iou, imgsz=imgsz, device=device,
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка инференса", str(e))
            self.statusBar().showMessage("Ошибка инференса")
            return

        r = results[0]
        names = self.model.names

        self.current_detections = []
        kind = "detect"

        obb = getattr(r, "obb", None)
        if obb is not None and getattr(obb, "xyxyxyxy", None) is not None and len(obb) > 0:
            kind = "obb"
            corners_all = obb.xyxyxyxy.cpu().numpy()
            cls_all = obb.cls.cpu().numpy().astype(int)
            conf_all = obb.conf.cpu().numpy()
            for i in range(len(cls_all)):
                corners = corners_all[i].tolist()
                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                cx = sum(xs) / 4.0
                cy = sum(ys) / 4.0
                cls_id = int(cls_all[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                self.current_detections.append({
                    "kind": "obb",
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": float(conf_all[i]),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "corners": corners,
                    "angle_deg": longest_edge_angle_deg(corners),
                    "center": (float(cx), float(cy)),
                })
        elif getattr(r, "masks", None) is not None and getattr(r.masks, "xy", None) is not None \
                and len(r.masks.xy) > 0:
            kind = "seg"
            boxes = r.boxes
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()
            for i, poly in enumerate(r.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                pts = poly.astype(np.float32).reshape(-1, 1, 2)
                rect = cv2.minAreaRect(pts)
                box_pts = cv2.boxPoints(rect).tolist()
                angle_deg = longest_edge_angle_deg(box_pts)
                cx, cy = float(rect[0][0]), float(rect[0][1])
                cls_id = int(cls_all[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                xyxy = [float(v) for v in xyxy_all[i].tolist()]
                self.current_detections.append({
                    "kind": "seg",
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "conf": float(conf_all[i]),
                    "xyxy": xyxy,
                    "corners": box_pts,
                    "angle_deg": angle_deg,
                    "center": (cx, cy),
                })
        else:
            kind = "detect"
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_all = boxes.xyxy.cpu().numpy()
                cls_all = boxes.cls.cpu().numpy().astype(int)
                conf_all = boxes.conf.cpu().numpy()
                for i in range(len(cls_all)):
                    x1, y1, x2, y2 = [float(v) for v in xyxy_all[i].tolist()]
                    w = x2 - x1
                    h = y2 - y1
                    angle_deg = 0.0 if w >= h else 90.0
                    corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    cls_id = int(cls_all[i])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    self.current_detections.append({
                        "kind": "detect",
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": float(conf_all[i]),
                        "xyxy": [x1, y1, x2, y2],
                        "corners": corners,
                        "angle_deg": angle_deg,
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    })

        self.current_detections.sort(key=lambda d: -d["conf"])

        try:
            annotated = r.plot()
        except Exception:
            annotated = self.original_image.copy()

        # Оверлей YOLO — базис для авто- и ручных трансформаций.
        self.annotated_image = annotated

        # Список объектов (guard _updating_objects_list не даст зациклиться).
        self._update_objects_list()

        # Пересобираем вид с учётом текущих авто/ручных трансформаций.
        self._rebuild_displayed()

        self.log.clear()
        self._log(f"[ИНФЕРЕНС YOLO] conf={conf} iou={iou} imgsz={imgsz} device={device}")
        self._log(f"[ТИП ВЫВОДА] {kind}")
        self._log(f"[ФАЙЛ]     {self.image_paths[self.current_index]}")
        self._log("-" * 60)
        for i, d in enumerate(self.current_detections):
            x1, y1, x2, y2 = d["xyxy"]
            ang = d.get("angle_deg")
            ang_s = f" angle={ang:6.1f}°" if ang is not None else ""
            self._log(
                f"{i+1:3d}. {d['cls_name']}  conf={d['conf']:.3f}  "
                f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}){ang_s}"
            )
        self._log("-" * 60)
        self._log(f"Всего объектов: {len(self.current_detections)}")

        self.statusBar().showMessage(
            f"Готово. Объектов: {len(self.current_detections)} ({kind})"
        )

    def _update_objects_list(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        self._updating_objects_list = True
        try:
            self.objects_list.clear()
            for i, d in enumerate(self.current_detections):
                x1, y1, x2, y2 = d["xyxy"]
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                ang = d.get("angle_deg")
                ang_s = f"  ∠{ang:5.1f}°" if ang is not None else ""
                text = (f"{i+1}. {d['cls_name']}  conf={d['conf']:.3f}  "
                        f"[{int(w)}×{int(h)}]{ang_s}  ({d['kind']})")
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, i)
                self.objects_list.addItem(item)
            if self.current_detections:
                self.objects_list.setCurrentRow(0)
        finally:
            self._updating_objects_list = False

    def _get_selected_detection(self):
        if not self.current_detections:
            return None
        cur = self.objects_list.currentRow()
        if 0 <= cur < len(self.current_detections):
            return self.current_detections[cur]
        return self.current_detections[0]

    # --------------------------------------------------------
    # Ручные трансформации
    # --------------------------------------------------------
    def manual_transform(self, op: str):
        if self.displayed_image is None:
            return
        if op not in ("rot_cw", "rot_ccw", "flip_v", "flip_h"):
            return

        # Детекции НЕ трогаем: они живут в координатах original_image
        # и нужны для авто-обрезки/доворота.
        self._manual_ops.append(op)

        self._rebuild_displayed()
        self._clear_ae_view()

        names = {
            "rot_cw": "Поворот 90° по часовой",
            "rot_ccw": "Поворот 90° против часовой",
            "flip_v": "Отражение по вертикали",
            "flip_h": "Отражение по горизонтали",
        }
        h, w = self.displayed_image.shape[:2]
        self._log(f"{names[op]}: новое изображение {w}×{h} px")
        self.statusBar().showMessage(f"{names[op]} → {w}×{h}")
        self._auto_apply_ae()

    # --------------------------------------------------------
    # Сброс
    # --------------------------------------------------------
    def reset_view(self):
        if self.original_image is None:
            return
        self._manual_ops = []
        self._clear_ae_view()
        # Никакого повторного инференса: только пересобрать вид.
        self._rebuild_displayed()
        self._auto_apply_ae()


# ------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------
def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = YoloInspectWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()