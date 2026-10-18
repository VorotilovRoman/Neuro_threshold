#!/usr/bin/env python
# train_unet_worker.py
# Поддерживается бинарная (num_classes=1) и многоклассовая (num_classes>1) сегментация
# Сохранение результатов организовано по аналогии с YOLO
#
# Сохраняются три модели:
#   weights/best_loss.pth — лучшая по val_loss
#   weights/best_iou.pth  — лучшая по IoU (Dice)
#   weights/last.pth      — последняя эпоха
#
# Пути сплитов (train_images / val_images / test_images и т.д.) читаются
# из masks_data.yaml, если он указан; иначе — дефолтная структура
# dataset/images/{train,val,test} + dataset/masks/{train,val,test}.
#
# Режим ресайза (resize_mode):
#   "hard"            — жёсткий ресайз до (input_size, input_size);
#   "keep_aspect_pad" — масштаб с сохранением AR + padding до квадрата;
#   "random_crop"     — обучение: случайный кроп; валидация/тест: центральный кроп.
#                       Для инференса в приложении пользователь отдельно выбирает
#                       hard / keep_aspect_pad (в metadata: inference_resize_mode).

import sys
import io
import json
import argparse
import os
import signal
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image

# Принудительная буферизация вывода
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------- Проверка наличия дополнительных библиотек ----------
HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_YAML = False

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    print("Предупреждение: matplotlib не установлен. Графики и визуализации сохранены не будут.")

if HAS_MATPLOTLIB:
    try:
        import seaborn as sns

        HAS_SEABORN = True
    except ImportError:
        print("Предупреждение: seaborn не установлен. Матрицы ошибок будут отображены без тепловой карты.")

try:
    import yaml

    HAS_YAML = True
except ImportError:
    print("Предупреждение: PyYAML не установлен. Параметры будут сохранены в JSON.")

# Попытка импорта segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Ошибка: не установлена библиотека 'segmentation-models-pytorch'")
    print("Установите: pip install segmentation-models-pytorch")
    sys.exit(1)

# ========== Глобальные конфигурации ==========
DECODER_CHANNELS = (256, 128, 64, 32, 16)  # каналы декодера U-Net
NUM_WORKERS_DEFAULT = 4  # количество workers для DataLoader
AUTO_BATCH_MEM_FACTOR = 0.5  # использовать не более 50% общей памяти
AUTO_BATCH_MIN = 1  # минимальный batch
AUTO_BATCH_MAX = 64  # максимальный batch
NUM_TRAIN_BATCHES_TO_SAVE = 3  # сколько тренировочных батчей сохранить как примеры
NUM_VAL_SAMPLES_TO_SAVE = 8  # сколько валидационных изображений для предсказаний

# Early stopping — ослабленные значения
EARLY_STOPPING_PATIENCE = 5
OVERFITTING_GAP_THRESHOLD = 0.5
OVERFITTING_MIN_PATIENCE = 3

THRESHOLD_FACTOR = 0.7  # порог бинаризации для метрик по умолчанию

# Поддерживаемые расширения
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
MASK_EXTS = ('.png', '.tif', '.tiff', '.bmp')  # без .jpg — плохо для масок

# Допустимые значения resize_mode
VALID_RESIZE_MODES = ("hard", "keep_aspect_pad", "random_crop")

# Глобальные переменные для graceful shutdown
model = None
optimizer = None
scheduler = None
best_model_path = None  # алиас на best_loss (для обратной совместимости)
best_loss_model_path = None
best_iou_model_path = None
final_model_path = None
current_epoch = 0
best_val_loss = float('inf')
best_iou = -float('inf')
stop_requested = False

history = {  # для сбора метрик по эпохам
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'iou': [],
    'dice': [],
    'f1': [],
    'precision': [],
    'recall': [],
    'accuracy': [],
    'map50': [],
    'map50_95': []
}


def log(msg):
    print(msg)
    sys.stdout.flush()


def signal_handler(signum, frame):
    global stop_requested
    log(f"Получен сигнал {signum}. Запрос остановки обучения...")
    stop_requested = True
    if model is not None and final_model_path is not None:
        checkpoint_path = final_model_path.replace(".pth", f"_checkpoint_epoch_{current_epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        log(f"Модель сохранена как {checkpoint_path}")
        # Бэкапим лучшие модели, если они уже есть
        for src_path, suffix in ((best_loss_model_path, "_interrupted"),
                                 (best_iou_model_path, "_interrupted")):
            if src_path and os.path.exists(src_path):
                backup = src_path.replace(".pth", f"{suffix}.pth")
                torch.save(torch.load(src_path), backup)
                log(f"Лучшая модель скопирована в {backup}")


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ---------- Dataset (поддержка бинарной и многоклассовой сегментации) ----------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, input_size=512, num_classes=1,
                 augmentations=None, normalize=False,
                 resize_mode="hard", is_train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.augmentations = augmentations or {}
        self.normalize = normalize
        if resize_mode not in VALID_RESIZE_MODES:
            log(f"⚠️ Неизвестный resize_mode={resize_mode!r}, использую 'hard'")
            resize_mode = "hard"
        self.resize_mode = resize_mode
        self.is_train = bool(is_train)
        self.image_paths = []
        self.mask_paths = []
        skipped = 0

        # Собираем список изображений, для каждого ищем маску с любым расширением
        for f in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            base = os.path.splitext(f)[0]
            mask_path = None
            for mext in MASK_EXTS:
                cand = os.path.join(masks_dir, base + mext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                log(f"⚠️ Пропуск: для {f} не найдена маска в {masks_dir}")
                skipped += 1
                continue
            self.image_paths.append(os.path.join(images_dir, f))
            self.mask_paths.append(mask_path)

        if not self.image_paths:
            raise RuntimeError(f"Не найдено ни одной пары изображение-маска в {images_dir} и {masks_dir}")
        log(f"✅ Загружено {len(self.image_paths)} пар "
            f"(пропущено {skipped}, режим: {resize_mode}, is_train={is_train})")

    def __len__(self):
        return len(self.image_paths)

    def _resize_pair(self, image, mask):
        """
        Приведение пары (image, mask) к (input_size, input_size).
        Маска всегда интерполируется NEAREST — без размытия границ классов.
        """
        S = self.input_size
        mode = self.resize_mode

        if mode == "keep_aspect_pad":
            w, h = image.size  # PIL: (w, h)
            scale = S / max(h, w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

            pad_h = S - new_h
            pad_w = S - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        elif mode == "random_crop":
            w, h = image.size
            # Масштабируем так, чтобы обе стороны были >= S
            scale = max(S / h, S / w)
            new_h = max(S, int(round(h * scale)))
            new_w = max(S, int(round(w * scale)))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

            if self.is_train:
                x = int(np.random.randint(0, max(1, new_w - S + 1)))
                y = int(np.random.randint(0, max(1, new_h - S + 1)))
            else:
                # Валидация/тест — детерминированный центр
                x = (new_w - S) // 2
                y = (new_h - S) // 2
            image = TF.crop(image, y, x, S, S)
            mask = TF.crop(mask, y, x, S, S)

        else:  # "hard"
            image = transforms.Resize(
                (S, S),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )(image)
            mask = transforms.Resize(
                (S, S),
                interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)

        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])

        # Базовый ресайз (hard / keep_aspect_pad / random_crop)
        image, mask = self._resize_pair(image, mask)

        # --- Аугментации ---
        if self.augmentations.get('hflip', False) and np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.augmentations.get('vflip', False) and np.random.rand() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if self.augmentations.get('rotate', False):
            angle = float(np.random.uniform(-10, 10))
            image = TF.rotate(image, angle)
            # NEAREST обязателен — иначе маска «поплывёт»
            mask = TF.rotate(
                mask, angle,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0,
            )
        if self.augmentations.get('scale', False):
            scale = float(np.random.uniform(0.9, 1.1))
            new_size = (int(self.input_size * scale), int(self.input_size * scale))
            image = transforms.Resize(
                new_size, interpolation=transforms.InterpolationMode.BILINEAR,
            )(image)
            mask = transforms.Resize(
                new_size, interpolation=transforms.InterpolationMode.NEAREST,
            )(mask)
            image = transforms.CenterCrop(self.input_size)(image)
            mask = transforms.CenterCrop(self.input_size)(mask)
        if self.augmentations.get('noise', False):
            img_np = np.array(image) / 255.0
            noise = np.random.normal(0, 0.05, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        image = transforms.ToTensor()(image)
        if self.normalize:
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )(image)

        # --- Преобразование маски в тензор ---
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask_np = mask_np.squeeze()

        if self.num_classes == 1:
            # Бинарная сегментация.
            if int(mask_np.max()) > 1:
                binary = (mask_np > 127).astype(np.float32)
            else:
                binary = (mask_np > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(binary).unsqueeze(0)  # [1, H, W]
        else:
            # Многоклассовая сегментация.
            mask_tensor = torch.from_numpy(mask_np.astype(np.int64))
            max_val = int(mask_tensor.max())
            if max_val >= self.num_classes:
                log(f"⚠️ ВНИМАНИЕ: в маске {os.path.basename(self.mask_paths[idx])} "
                    f"найдено значение пикселя {max_val}, что >= num_classes={self.num_classes}. "
                    f"Проверьте поле «Количество классов (вкл. фон)». "
                    f"Некорректные пиксели будут заменены на 0 (фон).")
                mask_tensor = mask_tensor.clone()
                mask_tensor[mask_tensor >= self.num_classes] = 0
        return image, mask_tensor


# ---------- Функции потерь (поддержка бинарной и многоклассовой) ----------
def get_loss_function(loss_name, num_classes=1):
    loss_name = loss_name.lower().replace(' ', '')
    if num_classes > 1:
        # Многоклассовый режим
        if loss_name in ("ce", "crossentropy"):
            return nn.CrossEntropyLoss()
        elif loss_name == "dice":
            return smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=num_classes)
        elif loss_name == "focal":
            return smp.losses.FocalLoss(mode='multiclass', alpha=0.25, gamma=2, from_logits=True)
        elif loss_name == "tversky":
            return smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, from_logits=True)
        elif loss_name in ("bce+dice", "ce+dice", "crossentropy+dice"):
            ce = nn.CrossEntropyLoss()
            dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=num_classes)
            return lambda pred, target: ce(pred, target) + dice(pred, target)
        else:
            return nn.CrossEntropyLoss()
    else:
        # Бинарный режим
        if loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "dice":
            return smp.losses.DiceLoss(mode='binary', from_logits=True)
        elif loss_name in ("bce+dice", "ce+dice", "crossentropy+dice"):
            bce = nn.BCEWithLogitsLoss()
            dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
            return lambda pred, target: bce(pred, target) + dice(pred, target)
        elif loss_name == "focal":
            return smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2, from_logits=True)
        elif loss_name == "tversky":
            return smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, from_logits=True)
        else:
            return nn.BCEWithLogitsLoss()


# ---------- Оптимизатор ----------
def get_optimizer(model, opt_name, lr):
    opt_name = opt_name.lower()
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr)
    else:
        return optim.Adam(model.parameters(), lr=lr)


# ---------- Scheduler ----------
def get_scheduler(optimizer, scheduler_name, epochs):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    elif scheduler_name == "cosineannealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        return None


# ---------- Метрики ----------
def compute_metrics(pred_logits, true_masks, num_classes=1, threshold=0.5):
    if num_classes == 1:
        pred_probs = torch.sigmoid(pred_logits)
        pred = (pred_probs > threshold).float()
        if true_masks.dim() == 3:
            true = true_masks.unsqueeze(1).float()
        else:
            true = true_masks.float()
        intersection = (pred * true).sum(dim=(2, 3))
        pred_sum = pred.sum(dim=(2, 3))
        true_sum = true.sum(dim=(2, 3))
        union = pred_sum + true_sum - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred_sum + true_sum + 1e-6)
        tp = intersection
        fp = pred_sum - intersection
        fn = true_sum - intersection
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        correct = (pred == true).float().sum(dim=(2, 3))
        total = pred.shape[2] * pred.shape[3]
        accuracy = correct / total
        map50 = (iou > 0.5).float().mean().item()
        map50_95 = iou.mean().item()
        return {
            'IoU': iou.mean().item(),
            'Dice': dice.mean().item(),
            'Precision': precision.mean().item(),
            'Recall': recall.mean().item(),
            'F1': f1.mean().item(),
            'Accuracy': accuracy.mean().item(),
            'mAP50': map50,
            'mAP50-95': map50_95,
        }
    else:
        pred = torch.argmax(pred_logits, dim=1)
        true = true_masks.long()
        iou_per_class, dice_per_class = [], []
        precision_per_class, recall_per_class = [], []
        for c in range(num_classes):
            pred_c = (pred == c)
            true_c = (true == c)
            intersection = (pred_c & true_c).sum().float()
            union = (pred_c | true_c).sum().float()
            iou = (intersection + 1e-6) / (union + 1e-6)
            iou_per_class.append(iou)
            tp = intersection
            fp = (pred_c & ~true_c).sum().float()
            fn = (~pred_c & true_c).sum().float()
            dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
            dice_per_class.append(dice)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            precision_per_class.append(prec)
            recall_per_class.append(rec)
        miou = torch.mean(torch.tensor(iou_per_class)).item()
        mdice = torch.mean(torch.tensor(dice_per_class)).item()
        precision = torch.mean(torch.tensor(precision_per_class)).item()
        recall = torch.mean(torch.tensor(recall_per_class)).item()
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (pred == true).float().mean().item()
        return {
            'IoU': miou,
            'Dice': mdice,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
            'mAP50': miou,
            'mAP50-95': miou,
        }


# ---------- Автоподбор batch_size (УЛУЧШЕННЫЙ) ----------
def auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN,
                    max_batch=AUTO_BATCH_MAX):
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            # Получаем общую и свободную память
            total_mem = torch.cuda.get_device_properties(device).total_memory
            free_mem, _ = torch.cuda.mem_get_info(device)
            total_mem_gb = total_mem / (1024 ** 3)
            free_mem_gb = free_mem / (1024 ** 3)

            # Ограничиваем использование 50% от общей памяти (оставляем на пики)
            max_allowed_mem_gb = min(total_mem_gb * mem_factor, free_mem_gb)

            # Оценка памяти на 1 изображение (в ГБ) для U-Net
            # Для 1024x1024 эмпирически нужно ~2.5 ГБ (прямой + обратный проход + активации)
            # Зависимость квадратичная от размера входа
            mem_per_img_gb = 2.5 * (input_size / 1024.0) ** 2

            suggested = int(max_allowed_mem_gb / mem_per_img_gb)
            suggested = max(min_batch, min(suggested, max_batch))

            log(f"Auto batch (GPU): total={total_mem_gb:.1f}GB, free={free_mem_gb:.1f}GB, "
                f"limit={mem_factor * 100:.0f}% ({max_allowed_mem_gb:.1f}GB), "
                f"mem/img≈{mem_per_img_gb:.2f}GB -> batch={suggested}")
            return suggested
        except Exception as e:
            log(f"Ошибка при определении памяти GPU: {e}, переключаемся на RAM")
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / (1024 ** 3)
        # Для RAM также ограничиваем 50% и учитываем размер
        max_allowed_ram = free_ram * mem_factor
        mem_per_img_gb = 2.5 * (input_size / 1024.0) ** 2
        suggested = int(max_allowed_ram / mem_per_img_gb)
        suggested = max(min_batch, min(suggested, max_batch))
        log(f"Auto batch (RAM): free={free_ram:.1f}GB, limit={mem_factor * 100:.0f}% ({max_allowed_ram:.1f}GB) -> batch={suggested}")
        return suggested
    except ImportError:
        log("⚠️ psutil не установлен, auto batch использует значение по умолчанию 8")
        return 8


# ---------- Сохранение аргументов ----------
def save_args_yaml(args_dict, save_path):
    if HAS_YAML:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(args_dict, f, default_flow_style=False, allow_unicode=True)
        log(f"Аргументы сохранены в YAML: {save_path}")
    else:
        json_path = save_path.replace('.yaml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        log(f"Аргументы сохранены в JSON (YAML недоступен): {json_path}")


def save_model_meta(meta_dict, model_path):
    meta_path = os.path.splitext(model_path)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    log(f"Метаданные модели сохранены в {meta_path}")


# ---------- Графики ----------
def save_training_plots(history_dict, save_dir):
    if not HAS_MATPLOTLIB:
        return
    epochs = history_dict['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(epochs, history_dict['train_loss'], label='Train Loss')
    if history_dict['val_loss']:
        axes[0, 0].plot(epochs, history_dict['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch');
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend();
    axes[0, 0].set_title('Loss')
    axes[0, 1].plot(epochs, history_dict['iou'], label='IoU')
    axes[0, 1].plot(epochs, history_dict['dice'], label='Dice')
    axes[0, 1].set_xlabel('Epoch');
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend();
    axes[0, 1].set_title('IoU & Dice')
    axes[1, 0].plot(epochs, history_dict['precision'], label='Precision')
    axes[1, 0].plot(epochs, history_dict['recall'], label='Recall')
    axes[1, 0].plot(epochs, history_dict['f1'], label='F1')
    axes[1, 0].set_xlabel('Epoch');
    axes[1, 0].legend()
    axes[1, 0].set_title('Precision/Recall/F1')
    axes[1, 1].plot(epochs, history_dict['map50'], label='mAP50')
    axes[1, 1].plot(epochs, history_dict['map50_95'], label='mAP50-95')
    axes[1, 1].set_xlabel('Epoch');
    axes[1, 1].legend()
    axes[1, 1].set_title('mAP')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results.png'), dpi=150)
    plt.close()
    log(f"График обучения сохранён: {os.path.join(save_dir, 'results.png')}")


def save_csv_results(history_dict, save_path):
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=history_dict.keys())
        writer.writeheader()
        rows = [dict(zip(history_dict, t)) for t in zip(*history_dict.values())]
        writer.writerows(rows)
    log(f"CSV история сохранена: {save_path}")


def save_pr_curve(preds, masks, save_dir, num_classes=1):
    if not HAS_MATPLOTLIB:
        return
    if num_classes == 1:
        preds_np = preds.cpu().numpy().flatten()
        masks_np = masks.cpu().numpy().flatten()
        thresholds = np.linspace(0, 1, 51)
        precisions, recalls, f1_scores = [], [], []
        for thresh in thresholds:
            pred_bin = (preds_np > thresh).astype(np.float32)
            tp = np.sum(pred_bin * masks_np)
            fp = np.sum(pred_bin * (1 - masks_np))
            fn = np.sum((1 - pred_bin) * masks_np)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            precisions.append(prec);
            recalls.append(rec);
            f1_scores.append(f1)
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label='PR curve')
        plt.xlabel('Recall');
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve');
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskPR_curve.png'));
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1')
        plt.xlabel('Threshold');
        plt.ylabel('Score')
        plt.title('Precision, Recall, F1 vs Threshold');
        plt.legend();
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskF1_curve.png'));
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions, label='Precision')
        plt.xlabel('Threshold');
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold');
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskP_curve.png'));
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, recalls, label='Recall')
        plt.xlabel('Threshold');
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold');
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskR_curve.png'));
        plt.close()
        log(f"Кривые PR, F1, P, R сохранены в {save_dir}")
    else:
        log("PR-кривые не генерируются для многоклассовой сегментации")


def save_confusion_matrix(preds, true_masks, save_dir, num_classes=1, threshold=0.5):
    if not HAS_MATPLOTLIB:
        return
    if num_classes == 1:
        pred_bin = (preds > threshold).float()
        true_bin = true_masks.float()
        tp = ((pred_bin == 1) & (true_bin == 1)).sum().item()
        fp = ((pred_bin == 1) & (true_bin == 0)).sum().item()
        fn = ((pred_bin == 0) & (true_bin == 1)).sum().item()
        tn = ((pred_bin == 0) & (true_bin == 0)).sum().item()
        conf_matrix = np.array([[tn, fp], [fn, tp]])
        tick_labels = ['Background', 'Object']
    else:
        pred = torch.argmax(preds, dim=1)
        true = true_masks.long()
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t in range(num_classes):
            for p in range(num_classes):
                conf_matrix[t, p] = ((true == t) & (pred == p)).sum().item()
        conf_matrix = conf_matrix.numpy()
        tick_labels = [f'Class {i}' for i in range(num_classes)]

    plt.figure(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
    if HAS_SEABORN:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=tick_labels, yticklabels=tick_labels)
    else:
        plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')
        plt.xticks(range(len(tick_labels)), tick_labels)
        plt.yticks(range(len(tick_labels)), tick_labels)
    plt.xlabel('Predicted');
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'));
    plt.close()

    conf_norm = conf_matrix.astype(np.float32) / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)
    plt.figure(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
    if HAS_SEABORN:
        sns.heatmap(conf_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=tick_labels, yticklabels=tick_labels)
    else:
        plt.imshow(conf_norm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        for i in range(conf_norm.shape[0]):
            for j in range(conf_norm.shape[1]):
                plt.text(j, i, f"{conf_norm[i, j]:.2f}", ha='center', va='center')
        plt.xticks(range(len(tick_labels)), tick_labels)
        plt.yticks(range(len(tick_labels)), tick_labels)
    plt.xlabel('Predicted');
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'));
    plt.close()
    log(f"Матрицы ошибок сохранены в {save_dir}")


def save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1,
                              normalize=False):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, (images, masks) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        n_images = min(images.size(0), samples_per_batch)
        images = images[:n_images];
        masks = masks[:n_images]
        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 8))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_images):
            # Денормализация для корректной отрисовки
            img = images[i].cpu().permute(1, 2, 0).numpy()
            if normalize:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)

            axes[0, i].imshow((img * 255).astype(np.uint8));
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Image {i + 1}')
            mask = masks[i].cpu().numpy().squeeze()
            if num_classes == 1:
                mask_vis = (mask > 0.5).astype(np.uint8) * 255
                axes[1, i].imshow(mask_vis, cmap='gray')
            else:
                mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for c in range(num_classes):
                    mask_vis[mask == c] = [c * 255 // max(1, num_classes - 1), 128, 255 - c * 128]
                axes[1, i].imshow(mask_vis)
            axes[1, i].axis('off');
            axes[1, i].set_title(f'Mask {i + 1}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'train_batch{batch_idx}.jpg'))
        plt.close()
    log(f"Сохранено {num_batches} тренировочных батчей в {save_dir}")


def save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1, normalize=False):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_list, true_masks_list, pred_masks_list = [], [], []
    collected = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device);
            masks = masks.to(device)
            outputs = model(images)
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1).float()
            for i in range(images.size(0)):
                if collected >= num_samples:
                    break
                # Денормализация
                img = images[i].cpu().permute(1, 2, 0).numpy()
                if normalize:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = np.clip(img, 0, 1)

                images_list.append((img * 255).astype(np.uint8))
                true_masks_list.append(masks[i].cpu().numpy().squeeze())
                pred_masks_list.append(preds[i].cpu().numpy().squeeze())
                collected += 1
            if collected >= num_samples:
                break
    if not images_list:
        log("Нет примеров для сохранения валидации");
        return
    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i in range(n):
        axes[i, 0].imshow(images_list[i]);
        axes[i, 0].axis('off');
        axes[i, 0].set_title('Image')
        if num_classes == 1:
            tv = (true_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i, 1].imshow(tv, cmap='gray')
        else:
            tv = np.zeros((true_masks_list[i].shape[0], true_masks_list[i].shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                tv[true_masks_list[i] == c] = [c * 255 // max(1, num_classes - 1), 128, 255 - c * 128]
            axes[i, 1].imshow(tv)
        axes[i, 1].axis('off');
        axes[i, 1].set_title('True Mask')
        if num_classes == 1:
            pv = (pred_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i, 2].imshow(pv, cmap='gray')
        else:
            pv = np.zeros((pred_masks_list[i].shape[0], pred_masks_list[i].shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                pv[pred_masks_list[i] == c] = [c * 255 // max(1, num_classes - 1), 128, 255 - c * 128]
            axes[i, 2].imshow(pv)
        axes[i, 2].axis('off');
        axes[i, 2].set_title('Pred Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_examples.jpg'))
    plt.close()
    log(f"Сохранено {collected} валидационных примеров в {save_dir}/val_examples.jpg")


# ---------- Проверка масок ----------
def check_dataset_masks_consistency(dataset, num_classes, max_check=20):
    warnings_out = []
    global_max = 0
    n = min(len(dataset), max_check)

    for i in range(n):
        mask_path = None
        try:
            mask_path = dataset.mask_paths[i]
            with Image.open(mask_path) as im:
                m = np.array(im)

            if m.size == 0:
                continue

            if m.ndim == 3:
                m = m[:, :, 0]
            m = m.squeeze()

            try:
                m_int = m.astype(np.int64)
            except Exception:
                continue

            mx = int(m_int.max())
            if mx > global_max:
                global_max = mx

            uniq = np.unique(m_int)
            uniq_preview = list(uniq[:10])
            preview_str = ", ".join(str(int(v)) for v in uniq_preview)
            if len(uniq) > 10:
                preview_str += ", ..."

            if num_classes > 1 and mx >= num_classes:
                warnings_out.append(
                    f"  • {os.path.basename(mask_path)}: max={mx}, "
                    f"уникальные=[{preview_str}]"
                )
        except Exception as e:
            warnings_out.append(
                f"  • {os.path.basename(mask_path or '?')}: "
                f"ошибка чтения ({type(e).__name__}: {e})"
            )
            continue

    return global_max, warnings_out


# ---------- Основная функция ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help='JSON-строка с параметрами')
    parser.add_argument('--params_file', type=str, help='Путь к JSON-файлу с параметрами')
    args = parser.parse_args()

    log("WORKER STARTED")

    if args.params_file:
        with open(args.params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
    elif args.params:
        params = json.loads(args.params)
    else:
        log("Ошибка: укажите --params или --params_file")
        sys.exit(1)

    dataset_path = params.get('dataset_path')
    model_source = params.get('model_path')
    project_path = params.get('project_path')
    name = params.get('name')
    encoder = params.get('encoder', 'resnet50')
    input_size = params.get('input_size', 512)
    num_classes = params.get('num_classes', 2)
    optimizer_name = params.get('optimizer', 'Adam')
    loss_name = params.get('loss', 'BCE+Dice')
    lr = params.get('lr', 0.0001)
    batch_size = params.get('batch_size', 8)
    auto_batch_flag = params.get('auto_batch', False) or (batch_size == -1)
    epochs = params.get('epochs', 100)
    dropout = params.get('dropout', 0.0)
    scheduler_name = params.get('scheduler', 'None')
    augmentations = params.get('augmentations', {})
    device_pref = params.get('device', 'auto')
    threshold = params.get('threshold', THRESHOLD_FACTOR)
    num_workers = params.get('num_workers', NUM_WORKERS_DEFAULT)

    # НОВЫЙ ПАРАМЕТР: Архитектура
    architecture = params.get('architecture', 'unet').lower()

    # --- Препроцессинг ---
    normalize = bool(params.get('normalize', False))

    resize_mode = params.get('resize_mode')
    if resize_mode not in VALID_RESIZE_MODES:
        if 'keep_aspect_ratio' in params:
            resize_mode = "keep_aspect_pad" if bool(params['keep_aspect_ratio']) else "hard"
            log(f"ℹ️ resize_mode не задан, использую keep_aspect_ratio={params['keep_aspect_ratio']} → {resize_mode}")
        else:
            resize_mode = "hard"
            log(f"ℹ️ resize_mode не задан, использую дефолт 'hard'")

    keep_aspect_ratio = (resize_mode == "keep_aspect_pad")

    unet_yaml = params.get('unet_yaml')

    split_paths = {
        'train_images': 'images/train',
        'train_masks': 'masks/train',
        'val_images': 'images/val',
        'val_masks': 'masks/val',
        'test_images': 'images/test',
        'test_masks': 'masks/test',
    }

    if unet_yaml and os.path.isfile(unet_yaml):
        try:
            with open(unet_yaml, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}

            yaml_root = cfg.get('path', '')
            if yaml_root:
                if not os.path.isabs(yaml_root):
                    yaml_root = os.path.normpath(
                        os.path.join(os.path.dirname(unet_yaml), yaml_root)
                    )
                if dataset_path != yaml_root:
                    log(f"📄 dataset_path переопределён из YAML: "
                        f"{dataset_path!r} → {yaml_root!r}")
                dataset_path = yaml_root
            else:
                yaml_root = os.path.dirname(os.path.abspath(unet_yaml))
                log(f"📄 В YAML нет поля 'path', использую папку YAML: {yaml_root}")
                dataset_path = yaml_root

            for key in split_paths.keys():
                val = cfg.get(key)
                if isinstance(val, str) and val.strip():
                    if not os.path.isabs(val):
                        val = os.path.normpath(os.path.join(dataset_path, val))
                    split_paths[key] = val
                    log(f"📄 {key} из YAML: {val}")

            nc_yaml = cfg.get('num_classes')
            if nc_yaml is not None:
                try:
                    nc_yaml_int = int(nc_yaml)
                except (ValueError, TypeError):
                    nc_yaml_int = None
                if nc_yaml_int is not None and nc_yaml_int >= 2:
                    if nc_yaml_int != num_classes:
                        log(f"📄 num_classes переопределён из YAML: "
                            f"{num_classes} → {nc_yaml_int}")
                    num_classes = nc_yaml_int
                elif nc_yaml_int is not None:
                    log(f"⚠️ num_classes в YAML = {nc_yaml_int}, ожидалось ≥ 2. "
                        f"Оставляю значение из UI: {num_classes}")

            cn_list = cfg.get('class_names', [])
            if cn_list:
                log(f"📄 Классы из YAML ({len(cn_list)}): {cn_list}")

            mco = cfg.get('mask_class_offset')
            if mco is not None:
                log(f"📄 mask_class_offset из YAML: {mco}")
        except Exception as e:
            log(f"⚠️ Не удалось прочитать {unet_yaml}: {e}. "
                f"Использую параметры из UI.")
    elif unet_yaml:
        log(f"⚠️ U-Net YAML указан, но файл не найден: {unet_yaml}. "
            f"Использую параметры из UI.")

    # Формирование пути сохранения
    if name and project_path:
        save_root = os.path.join(project_path, name)
    elif name:
        save_root = name
    elif project_path:
        save_root = project_path
    else:
        save_root = "runs/unet"
    os.makedirs(save_root, exist_ok=True)
    weights_dir = os.path.join(save_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    log("=== Запуск обучения ===")
    log(f"Архитектура: {architecture.upper()}")
    log(f"Датасет: {dataset_path}")
    log(f"Модель: {model_source if model_source else 'предустановленная'}")
    log(f"Энкодер: {encoder}")
    log(f"Размер входа: {input_size}")
    log(f"Классов (включая фон): {num_classes}")
    log(f"Оптимизатор: {optimizer_name}, LR={lr}")
    log(f"Loss: {loss_name}")
    log(f"Batch size: {batch_size}, Epochs: {epochs}")
    log(f"Scheduler: {scheduler_name}, Dropout: {dropout}")
    log(f"Аугментации: {augmentations}")
    log(f"Устройство: {device_pref}")
    log(f"Num workers: {num_workers}")
    log(f"Порог (threshold): {threshold}")
    log(f"Нормализация входа (ImageNet): {normalize}")
    log(f"Режим ресайза (resize_mode): {resize_mode}")
    if resize_mode == "random_crop":
        log("⚠️ random_crop: обучение идёт на случайных кропах. "
            "Валидация/тест используют центральный кроп. "
            "Для инференса пользователь выберет hard / keep_aspect_pad отдельно.")
    log(f"Результаты будут сохранены в: {save_root}")

    if not dataset_path or not os.path.exists(dataset_path):
        log(f"Ошибка: папка датасета не существует: {dataset_path}")
        sys.exit(1)

    for key, val in split_paths.items():
        if not os.path.isabs(val):
            split_paths[key] = os.path.normpath(os.path.join(dataset_path, val))

    train_img_dir = split_paths['train_images']
    train_mask_dir = split_paths['train_masks']
    val_img_dir = split_paths['val_images']
    val_mask_dir = split_paths['val_masks']
    test_img_dir = split_paths['test_images']
    test_mask_dir = split_paths['test_masks']

    if not os.path.exists(train_img_dir) or not os.path.exists(train_mask_dir):
        log(f"Ошибка: не найдены папки:\n{train_img_dir}\n{train_mask_dir}")
        sys.exit(1)

    val_exists = os.path.exists(val_img_dir) and os.path.exists(val_mask_dir)

    # Auto batch
    if auto_batch_flag:
        if device_pref == 'auto':
            temp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_pref.isdigit():
            temp_device = torch.device(f'cuda:{device_pref}') if torch.cuda.is_available() else torch.device('cpu')
        elif device_pref.lower() in ('cuda', 'gpu'):
            temp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            temp_device = torch.device(device_pref)
        # Передаем input_size для корректного расчета
        batch_size = auto_batch_size(temp_device, input_size)
        log(f"Auto batch: выбран batch_size = {batch_size}")
    elif batch_size <= 0:
        log("Ошибка: batch_size должен быть положительным числом")
        sys.exit(1)

    # Метаданные пользовательской модели
    decoder_channels = DECODER_CHANNELS
    if model_source is not None and os.path.exists(model_source):
        base, _ = os.path.splitext(model_source)
        meta_path = base + '_meta.json'
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    model_meta = json.load(f)
                log(f"Загружены метаданные модели из {meta_path}")
                encoder = model_meta.get('encoder', encoder)
                decoder_channels = tuple(model_meta.get('decoder_channels', decoder_channels))
                dropout = model_meta.get('dropout', dropout)
                input_size = model_meta.get('input_size', input_size)
                num_classes = model_meta.get('num_classes', num_classes)
                # Восстанавливаем архитектуру, если она была сохранена
                architecture = model_meta.get('architecture', architecture)
                log(f"Архитектура восстановлена: type={architecture}, encoder={encoder}, "
                    f"decoder_channels={decoder_channels}, dropout={dropout}, "
                    f"input_size={input_size}, num_classes={num_classes}")
            except Exception as e:
                log(f"Предупреждение: не удалось загрузить метафайл {meta_path}: {e}. "
                    f"Используются параметры из JSON.")
        else:
            log(f"⚠️ Метаданные модели не найдены: {meta_path}. "
                f"Используются параметры из JSON. Возможна несовместимость архитектуры.")
    elif model_source is not None and not os.path.exists(model_source):
        log(f"Ошибка: файл модели {model_source} не найден")
        sys.exit(1)

    # ---------- Датасеты ----------
    train_dataset = SegmentationDataset(
        train_img_dir, train_mask_dir,
        input_size, num_classes, augmentations,
        normalize=normalize, resize_mode=resize_mode, is_train=True,
    )

    val_dataset = None
    val_loader = None
    if val_exists:
        try:
            val_dataset = SegmentationDataset(
                val_img_dir, val_mask_dir,
                input_size, num_classes, {},
                normalize=normalize, resize_mode=resize_mode, is_train=False,
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
            log(f"Загружено {len(train_dataset)} обучающих, "
                f"{len(val_dataset)} валидационных образцов")
        except RuntimeError as e:
            log(f"⚠️ Не удалось загрузить валидационный датасет: {e}")
            log("⚠️ Обучение продолжится БЕЗ валидации.")
            val_dataset = None
            val_loader = None
    else:
        log(f"⚠️ Валидационные папки не найдены ({val_img_dir}, {val_mask_dir}), "
            f"валидация будет пропущена")

    test_dataset = None
    test_loader = None
    if os.path.exists(test_img_dir) and os.path.exists(test_mask_dir):
        try:
            test_dataset = SegmentationDataset(
                test_img_dir, test_mask_dir,
                input_size, num_classes, {},
                normalize=normalize, resize_mode=resize_mode, is_train=False,
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
            log(f"Загружено {len(test_dataset)} тестовых образцов "
                f"({test_img_dir})")
        except RuntimeError as e:
            log(f"⚠️ Не удалось загрузить тестовый датасет: {e}")
            test_dataset = None
            test_loader = None
    else:
        log(f"ℹ️ Тестовые папки не найдены ({test_img_dir}, {test_mask_dir}), "
            f"тестовая оценка будет пропущена")

    # --- Проверка масок на согласованность с num_classes ---
    log("--- Проверка масок на согласованность с num_classes ---")
    max_pixel, warnings_list = check_dataset_masks_consistency(train_dataset, num_classes, max_check=20)
    log(f"Максимальное значение пикселя среди первых 20 train-масок: {max_pixel}")

    if num_classes == 1:
        if max_pixel > 0:
            log(f"✅ Бинарный режим (num_classes=1). Маски содержат объекты "
                f"(max pixel = {max_pixel}).")
        else:
            log("⚠️ ВНИМАНИЕ: бинарный режим (num_classes=1), но все маски пусты!")
    else:
        if max_pixel >= num_classes:
            log("=" * 60)
            log(f"❌ ОШИБКА КОНФИГУРАЦИИ: num_classes={num_classes}, "
                f"но в масках встречаются пиксели со значением {max_pixel}.")
            log(f"Значение пикселя должно быть < num_classes.")
            log(f"Похоже, в масках {max_pixel + 1} классов (включая фон), "
                f"а в UI указано {num_classes}.")
            log("Примеры некорректных масок:")
            for w in warnings_list[:5]:
                log(w)
            log(f"Решение: в UI укажите «Количество классов» = {max_pixel + 1}.")
            log("=" * 60)
        else:
            expected_nc = max_pixel + 1
            if expected_nc < num_classes:
                log(f"⚠️ num_classes={num_classes}, но в масках максимум {max_pixel}. "
                    f"Возможно, у вас лишние «пустые» классы (это допустимо, но модель может "
                    f"плохо учиться из-за пустых каналов).")
            else:
                log(f"✅ Согласовано: num_classes={num_classes}, max pixel={max_pixel}.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    # Модель
    global model, best_model_path, final_model_path
    global best_loss_model_path, best_iou_model_path, best_iou
    load_pretrained = (model_source is None) or (not os.path.exists(model_source))

    # Выбор архитектуры
    if architecture in ('unet++', 'unetplusplus'):
        log("Инициализация архитектуры: UNet++")
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet' if load_pretrained else None,
            in_channels=3,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_channels=decoder_channels,
            decoder_dropout=dropout,
        )
    else:
        log("Инициализация архитектуры: U-Net (vanilla)")
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet' if load_pretrained else None,
            in_channels=3,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_channels=decoder_channels,
            decoder_dropout=dropout,
        )

    if model_source is not None and os.path.exists(model_source):
        state_dict = torch.load(model_source, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log(f"⚠️ Отсутствуют ключи: {missing[:5]}..." if len(missing) > 5 else f"⚠️ Отсутствуют ключи: {missing}")
        if unexpected:
            log(f"⚠️ Лишние ключи: {unexpected[:5]}..." if len(unexpected) > 5 else f"⚠️ Лишние ключи: {unexpected}")
        log(f"Загружены веса из {model_source}")

    # Устройство
    if device_pref == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.isdigit():
        if torch.cuda.is_available() and int(device_pref) < torch.cuda.device_count():
            device = torch.device(f'cuda:{device_pref}')
        else:
            log(f"Предупреждение: CUDA device {device_pref} недоступен, используется CPU")
            device = torch.device('cpu')
    elif device_pref.lower() in ('cuda', 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.lower() == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    log(f"Используется устройство: {device}")

    # Loss, оптимизатор, scheduler
    criterion = get_loss_function(loss_name, num_classes)
    optimizer = get_optimizer(model, optimizer_name, lr)
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)

    # --- Пути сохранения моделей ---
    best_loss_model_path = os.path.join(weights_dir, 'best_loss.pth')
    best_iou_model_path = os.path.join(weights_dir, 'best_iou.pth')
    final_model_path = os.path.join(weights_dir, 'last.pth')
    best_model_path = best_loss_model_path

    args_dict = {
        'architecture': architecture,
        'encoder': encoder,
        'input_size': input_size,
        'num_classes': num_classes,
        'encoder_weights': 'imagenet' if load_pretrained else None,
        'optimizer': optimizer_name,
        'lr': lr,
        'loss': loss_name,
        'batch': batch_size,
        'epochs': epochs,
        'dropout': dropout,
        'scheduler': scheduler_name,
        'augmentations': augmentations,
        'device': device_pref,
        'num_workers': num_workers,
        'threshold': threshold,
        'project': project_path,
        'name': name,
        'date': datetime.now().isoformat(),
        'decoder_channels': decoder_channels,
        'normalize': normalize,
        'resize_mode': resize_mode,
        'keep_aspect_ratio': keep_aspect_ratio,
    }
    save_args_yaml(args_dict, os.path.join(save_root, 'args.yaml'))

    meta_info_initial = {
        'architecture': architecture,
        'encoder': encoder,
        'input_size': input_size,
        'num_classes': num_classes,
        'loss': loss_name,
        'optimizer': optimizer_name,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'scheduler': scheduler_name,
        'dropout': dropout,
        'augmentations': augmentations,
        'device_used': str(device),
        'experiment_name': name,
        'decoder_channels': decoder_channels,
        'num_workers': num_workers,
        'threshold': threshold,
        'best_val_loss': None,
        'best_iou': None,
        'normalize': normalize,
        'resize_mode': resize_mode,
        'keep_aspect_ratio': keep_aspect_ratio,
        'date': datetime.now().isoformat()
    }
    save_model_meta(meta_info_initial, best_loss_model_path)
    save_model_meta(meta_info_initial, final_model_path)
    log("Метаданные модели сохранены (начальная версия).")

    patience_counter = 0
    best_val_loss = float('inf')
    best_iou = -float('inf')

    global current_epoch, stop_requested
    log("=== Начало обучения ===")

    for epoch in range(1, epochs + 1):
        current_epoch = epoch

        if stop_requested:
            log("Обучение остановлено пользователем. Завершаем...")
            break

        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)

        # Валидация
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_masks = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    if num_classes == 1:
                        all_preds.append(torch.sigmoid(outputs).cpu())
                        all_masks.append(masks.cpu())
                    else:
                        all_preds.append(outputs.cpu())
                        all_masks.append(masks.cpu())
                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_dataset)
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            metrics = compute_metrics(all_preds, all_masks, num_classes, threshold=threshold)
        else:
            val_loss = None
            metrics = None

        # Сохранение лучших моделей и early stopping
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_loss_model_path)
                patience_counter = 0
                log(f"✅ Новая лучшая модель по val_loss сохранена "
                    f"(val_loss={best_val_loss:.4f}) → {os.path.basename(best_loss_model_path)}")
            else:
                patience_counter += 1
                if patience_counter >= OVERFITTING_MIN_PATIENCE:
                    loss_gap = (val_loss - train_loss) / (train_loss + 1e-8)
                    if loss_gap > OVERFITTING_GAP_THRESHOLD:
                        log(f"⚠️ Переобучение: разрыв val_loss ({val_loss:.4f}) и "
                            f"train_loss ({train_loss:.4f}) слишком велик "
                            f"({loss_gap * 100:.1f}% > "
                            f"{OVERFITTING_GAP_THRESHOLD * 100:.0f}%). Остановка обучения.")
                        break
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"⚠️ Early stopping: val_loss не улучшается "
                        f"{EARLY_STOPPING_PATIENCE} эпох. Остановка обучения.")
                    break

            if metrics is not None:
                current_iou = metrics['IoU']
                if current_iou > best_iou:
                    best_iou = current_iou
                    torch.save(model.state_dict(), best_iou_model_path)
                    log(f"✅ Новая лучшая модель по IoU сохранена "
                        f"(IoU={best_iou:.4f}) → {os.path.basename(best_iou_model_path)}")
        else:
            if best_val_loss == float('inf'):
                best_val_loss = train_loss
                torch.save(model.state_dict(), best_loss_model_path)
                log("✅ Модель сохранена (нет валидации, best_loss≈train_loss)")

        # История
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        if metrics:
            history['iou'].append(metrics['IoU'])
            history['dice'].append(metrics['Dice'])
            history['f1'].append(metrics['F1'])
            history['precision'].append(metrics['Precision'])
            history['recall'].append(metrics['Recall'])
            history['accuracy'].append(metrics['Accuracy'])
            history['map50'].append(metrics['mAP50'])
            history['map50_95'].append(metrics['mAP50-95'])
        else:
            for k in ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'map50', 'map50_95']:
                history[k].append(float('nan'))

        # Scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    scheduler.step(train_loss)
            else:
                scheduler.step()

        percent = int(epoch / epochs * 100)
        log(f"PROGRESS:{percent}:{epoch}:{epochs}")
        if val_loss is not None:
            prob_mean = all_preds.mean().item()
            prob_min = all_preds.min().item()
            prob_max = all_preds.max().item()
            log(f"Эпоха {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"IoU: {metrics['IoU']:.4f} | Dice: {metrics['Dice']:.4f} | F1: {metrics['F1']:.4f} | "
                f"Prob: mean={prob_mean:.4f} min={prob_min:.4f} max={prob_max:.4f}")
        else:
            log(f"Эпоха {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Валидация пропущена")

    # Финальная модель
    torch.save(model.state_dict(), final_model_path)
    log(f"Финальная модель сохранена: {final_model_path}")

    meta_info_final = {
        'architecture': architecture,
        'encoder': encoder,
        'input_size': input_size,
        'num_classes': num_classes,
        'loss': loss_name,
        'optimizer': optimizer_name,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'scheduler': scheduler_name,
        'dropout': dropout,
        'augmentations': augmentations,
        'device_used': str(device),
        'experiment_name': name,
        'decoder_channels': decoder_channels,
        'threshold': threshold,
        'num_workers': num_workers,
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'best_iou': best_iou if best_iou > -float('inf') else None,
        'normalize': normalize,
        'resize_mode': resize_mode,
        'keep_aspect_ratio': keep_aspect_ratio,
        'date': datetime.now().isoformat()
    }
    save_model_meta(meta_info_final, best_loss_model_path)
    if os.path.exists(best_iou_model_path):
        save_model_meta(meta_info_final, best_iou_model_path)
    save_model_meta(meta_info_final, final_model_path)
    log("Метаданные модели обновлены (финальная версия).")

    # ---------- Пост-обработка ----------
    if val_loader is not None:
        log("=== Генерация итоговых отчётов и графиков ===")
        if os.path.exists(best_iou_model_path):
            eval_model_path = best_iou_model_path
            log(f"Для отчётов использую модель по IoU: {os.path.basename(eval_model_path)}")
        else:
            eval_model_path = best_loss_model_path
            log(f"Модель по IoU не найдена, использую: {os.path.basename(eval_model_path)}")

        best_state = torch.load(eval_model_path, map_location='cpu')
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()

        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                if num_classes == 1:
                    all_preds.append(torch.sigmoid(outputs).cpu())
                    all_masks.append(masks.cpu())
                else:
                    all_preds.append(outputs.cpu())
                    all_masks.append(masks.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        save_csv_results(history, os.path.join(save_root, 'results.csv'))
        if HAS_MATPLOTLIB:
            save_training_plots(history, save_root)
            save_pr_curve(all_preds, all_masks, save_root, num_classes)
            save_confusion_matrix(all_preds, all_masks, save_root, num_classes)

        # Передаем normalize
        save_train_batch_examples(train_loader, save_root,
                                  num_batches=NUM_TRAIN_BATCHES_TO_SAVE,
                                  samples_per_batch=min(4, batch_size),
                                  num_classes=num_classes,
                                  normalize=normalize)

        # Передаем normalize
        save_val_examples(val_loader, save_root, num_samples=NUM_VAL_SAMPLES_TO_SAVE,
                          device=device, num_classes=num_classes,
                          normalize=normalize)

        final_metrics = compute_metrics(all_preds, all_masks, num_classes, threshold=threshold)
        log(f"METRICS: {json.dumps(final_metrics)}")
    else:
        log("METRICS: null")
        log("⚠️ Валидация отсутствует – дополнительные графики и кривые не сгенерированы")
        save_csv_results(history, os.path.join(save_root, 'results.csv'))

    # ---------- Финальная оценка на тестовом наборе ----------
    if test_loader is not None:
        log("=== Оценка на тестовом наборе ===")
        if os.path.exists(best_iou_model_path):
            eval_model_path = best_iou_model_path
        else:
            eval_model_path = best_loss_model_path
        model.load_state_dict(torch.load(eval_model_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        test_preds, test_masks = [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                if num_classes == 1:
                    test_preds.append(torch.sigmoid(outputs).cpu())
                else:
                    test_preds.append(outputs.cpu())
                test_masks.append(masks.cpu())
        test_preds = torch.cat(test_preds, dim=0)
        test_masks = torch.cat(test_masks, dim=0)
        test_metrics = compute_metrics(test_preds, test_masks,
                                       num_classes, threshold=threshold)
        log(f"TEST_METRICS: {json.dumps(test_metrics)}")
        with open(os.path.join(save_root, 'test_metrics.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)

        meta_info_final['test_metrics'] = test_metrics
        save_model_meta(meta_info_final, best_loss_model_path)
        if os.path.exists(best_iou_model_path):
            save_model_meta(meta_info_final, best_iou_model_path)
        save_model_meta(meta_info_final, final_model_path)
    else:
        log("ℹ️ Тестовая оценка пропущена (нет test_loader)")

    log("=== Обучение завершено ===")
    if os.path.exists(best_loss_model_path):
        log(f"Best loss model: {best_loss_model_path} (val_loss={best_val_loss:.4f})")
    if os.path.exists(best_iou_model_path):
        log(f"Best IoU model:  {best_iou_model_path} (IoU={best_iou:.4f})")
    log(f"Last model:      {final_model_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()