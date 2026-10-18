#!/usr/bin/env python
# train_DeepLabV3_worker.py
# Обучение DeepLabV3 / DeepLabV3+ (segmentation_models_pytorch).
#
# Возможности:
#   - Архитектуры: DeepLabV3+ и DeepLabV3 (без плюса)
#   - Бинарная и многоклассовая сегментация, ignore_index, class_weights
#   - Двухфазное обучение (freeze/unfreeze backbone, опц. только последних N стадий)
#   - Безопасный auto_batch_size (50% VRAM + учёт input_size)
#   - Mixed Precision (AMP) + Gradient Accumulation
#   - EMA (Exponential Moving Average) весов
#   - Layer-wise LR для энкодера
#   - Poly LR по итерациям (а не по эпохам)
#   - Multi-scale inference на валидации/тестах
#   - Три точки сохранения: best_loss, best_iou, last (все — EMA-веса)
#   - Денормализация при визуализации батчей
# ------------------------------------------------------------------------------

import sys
import io
import json
import argparse
import os
import signal
import csv
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------- Библиотеки ----------
HAS_MATPLOTLIB = False
HAS_SEABORN = False
HAS_YAML = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Предупреждение: matplotlib не установлен. Графики не сохранятся.")

if HAS_MATPLOTLIB:
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        print("Предупреждение: seaborn не установлен. Confusion matrix без тепловой карты.")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    print("Предупреждение: PyYAML не установлен. Параметры будут сохранены в JSON.")

try:
    import segmentation_models_pytorch as smp
except ImportError:
    print("Ошибка: не установлена библиотека 'segmentation-models-pytorch'")
    print("Установите: pip install segmentation-models-pytorch")
    sys.exit(1)

# ========== Глобальные конфигурации ==========
NUM_WORKERS_DEFAULT = 4
AUTO_BATCH_MEM_FACTOR = 0.5
AUTO_BATCH_MIN = 1
AUTO_BATCH_MAX = 64
NUM_TRAIN_BATCHES_TO_SAVE = 3
NUM_VAL_SAMPLES_TO_SAVE = 8

EARLY_STOPPING_PATIENCE = 5
OVERFITTING_GAP_THRESHOLD = 0.5
OVERFITTING_MIN_PATIENCE = 3

# Глобальные переменные для graceful shutdown
model = None
optimizer = None
scheduler = None
best_loss_model_path = None
best_iou_model_path = None
final_model_path = None
current_epoch = 0
best_val_loss = float('inf')
best_iou = -float('inf')
stop_requested = False

history = {
    'epoch': [], 'train_loss': [], 'val_loss': [],
    'iou': [], 'dice': [], 'f1': [],
    'precision': [], 'recall': [], 'accuracy': [],
    'map50': [], 'map50_95': []
}

def log(msg):
    print(msg)
    sys.stdout.flush()

def signal_handler(signum, frame):
    global stop_requested
    log(f"Получен сигнал {signum}. Запрос остановки...")
    stop_requested = True
    if model is not None and final_model_path is not None:
        ckpt = final_model_path.replace(".pth", f"_checkpoint_epoch_{current_epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        log(f"Checkpoint сохранён: {ckpt}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# =============================================================================
# VOC / U-Net-style Dataset
# =============================================================================
def find_image_file(jpeg_dir, basename):
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        path = os.path.join(jpeg_dir, basename + ext)
        if os.path.exists(path):
            return path
    return None

def read_voc_file_list(txt_path, root_dir):
    if not os.path.exists(txt_path):
        return []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
    except UnicodeDecodeError:
        with open(txt_path, 'r', encoding='cp1251') as f:
            lines = [l.strip() for l in f if l.strip()]
    jpeg_dir = os.path.join(root_dir, 'JPEGImages')
    seg_dir = os.path.join(root_dir, 'SegmentationClass')
    pairs = []
    for name in lines:
        img = find_image_file(jpeg_dir, name)
        mask = os.path.join(seg_dir, name + '.png')
        if img is not None and os.path.exists(mask):
            pairs.append((img, mask))
        else:
            log(f"⚠️ Пропуск: {name} — нет изображения или маски")
    return pairs

def create_voc_loaders(root, input_size, num_classes, augmentations,
                       ignore_index, batch_size, num_workers,
                       normalize=False, resize_mode="hard"):
    seg_root = os.path.join(root, 'ImageSets', 'Segmentation')
    train_txt = os.path.join(seg_root, 'train.txt')
    val_txt = os.path.join(seg_root, 'val.txt')
    if not os.path.exists(train_txt):
        raise RuntimeError(f"Не найден {train_txt}")
    train_pairs = read_voc_file_list(train_txt, root)
    val_pairs = read_voc_file_list(val_txt, root) if os.path.exists(val_txt) else []
    if not train_pairs:
        raise RuntimeError("Нет обучающих пар в VOC-датасете.")
    train_ds = VOCSegmentationPairs(train_pairs, input_size, num_classes, ignore_index,
                                    augmentations, normalize=normalize, resize_mode=resize_mode)
    val_ds = (VOCSegmentationPairs(val_pairs, input_size, num_classes, ignore_index,
                                   {}, normalize=normalize, resize_mode=resize_mode)
              if val_pairs else None)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers) if val_ds else None)
    return train_loader, val_loader, len(train_ds), (len(val_ds) if val_ds else 0)


class VOCSegmentationPairs(Dataset):
    def __init__(self, image_mask_pairs, input_size, num_classes, ignore_index,
                 augmentations, normalize=False, resize_mode="hard"):
        self.pairs = image_mask_pairs
        self.input_size = input_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augmentations = augmentations
        self.normalize = normalize
        self.resize_mode = resize_mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask_np = np.array(Image.open(mask_path))

        if self.num_classes == 2 and np.max(mask_np) == 255:
            mask_np = (mask_np > 127).astype(np.int64)

        if self.resize_mode == "keep_aspect_pad":
            S = self.input_size
            w, h = image.size
            scale = S / max(h, w)
            nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            image = image.resize((nw, nh), Image.BILINEAR)
            mask_img = Image.fromarray(mask_np.astype(np.uint8)).resize((nw, nh), Image.NEAREST)
            pad_h, pad_w = S - nh, S - nw
            pt, pl = pad_h // 2, pad_w // 2
            pb, pr = pad_h - pt, pad_w - pl
            image = TF.pad(image, (pl, pt, pr, pb), fill=0)
            fill_val = self.ignore_index if self.ignore_index is not None else 0
            mask_img = TF.pad(mask_img, (pl, pt, pr, pb), fill=fill_val)
            mask_np = np.array(mask_img)
        else:
            resize_op = transforms.Resize((self.input_size, self.input_size),
                                          interpolation=transforms.InterpolationMode.NEAREST)
            image = resize_op(image)
            mask_np = np.array(resize_op(Image.fromarray(mask_np.astype(np.uint8))))

        aug = self.augmentations
        if aug.get('hflip', False) and np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask_np = np.fliplr(mask_np).copy()
        if aug.get('vflip', False) and np.random.rand() < 0.5:
            image = TF.vflip(image)
            mask_np = np.flipud(mask_np).copy()
        if aug.get('rotate', False):
            angle = np.random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask_np = np.array(TF.rotate(Image.fromarray(mask_np.astype(np.uint8)), angle,
                                         interpolation=TF.InterpolationMode.NEAREST))
        if aug.get('scale', False):
            scale = np.random.uniform(0.9, 1.1)
            new_size = (int(self.input_size * scale), int(self.input_size * scale))
            image = transforms.Resize(new_size)(image)
            mask_img = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(
                Image.fromarray(mask_np.astype(np.uint8)))
            image = transforms.CenterCrop(self.input_size)(image)
            mask_img = transforms.CenterCrop(self.input_size)(mask_img)
            mask_np = np.array(mask_img)
        if aug.get('noise', False):
            img_np = np.array(image) / 255.0
            img_np = np.clip(img_np + np.random.normal(0, 0.05, img_np.shape), 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        image = transforms.ToTensor()(image)
        if self.normalize:
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image)
        mask = torch.from_numpy(mask_np).long()
        return image, mask


# =============================================================================
# Loss / Optimizer / Scheduler / EMA / Layer-wise / Multi-scale
# =============================================================================
def get_loss_function(loss_name, num_classes, ignore_index=None,
                      class_weights=None, loss_weights=None):
    loss_weights = loss_weights or {'ce': 1.0, 'dice': 1.0, 'focal': 0.0}

    weights_tensor = None
    if class_weights is not None and class_weights != 'auto':
        try:
            if isinstance(class_weights, str) and os.path.exists(class_weights):
                with open(class_weights, 'r') as f:
                    class_weights = json.load(f)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        except Exception as e:
            log(f"⚠️ Ошибка загрузки class_weights: {e}")

    mode = 'binary' if num_classes == 1 else 'multiclass'

    if mode == 'binary':
        ce_loss = nn.BCEWithLogitsLoss(pos_weight=weights_tensor) if weights_tensor is not None \
                  else nn.BCEWithLogitsLoss()
        dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)
        focal_loss = smp.losses.FocalLoss(mode=mode, alpha=0.25, gamma=2, ignore_index=ignore_index)
    else:
        ign = ignore_index if ignore_index is not None else -100
        ce_loss = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ign)
        dice_loss = smp.losses.DiceLoss(mode=mode, classes=num_classes, ignore_index=ignore_index)
        focal_loss = smp.losses.FocalLoss(mode=mode, alpha=0.25, gamma=2, ignore_index=ignore_index)

    ln = loss_name.lower()
    if ln == 'crossentropy':
        return ce_loss
    elif ln == 'dice':
        return dice_loss
    elif ln == 'focal':
        return focal_loss
    elif ln == 'crossentropy+dice':
        ce_w, dice_w = loss_weights.get('ce', 1.0), loss_weights.get('dice', 1.0)
        return lambda p, t: ce_w * ce_loss(p, t) + dice_w * dice_loss(p, t)
    elif ln == 'crossentropy+focal':
        ce_w, focal_w = loss_weights.get('ce', 1.0), loss_weights.get('focal', 1.0)
        return lambda p, t: ce_w * ce_loss(p, t) + focal_w * focal_loss(p, t)
    elif ln == 'dice+focal':
        dice_w, focal_w = loss_weights.get('dice', 1.0), loss_weights.get('focal', 1.0)
        return lambda p, t: dice_w * dice_loss(p, t) + focal_w * focal_loss(p, t)
    log(f"⚠️ Неизвестный loss '{loss_name}', использую CrossEntropy")
    return ce_loss


def get_optimizer(model, opt_name, lr, momentum=0.9, weight_decay=0.0001):
    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_layerwise_optimizer(model, opt_name, base_lr, momentum=0.9,
                             weight_decay=0.0001, decay_factor=0.1):
    """Layer-wise LR: глубокие стадии энкодера учатся быстрее мелких."""
    try:
        stages = model.encoder.get_stages()
    except AttributeError:
        log("⚠️ encoder.get_stages() недоступен — использую единый LR")
        return get_optimizer(model, opt_name, base_lr, momentum, weight_decay)

    n = len(stages)
    param_groups = []
    for i, stage in enumerate(stages):
        stage_lr = base_lr * (decay_factor ** (n - 1 - i))
        params = [p for p in stage.parameters() if p.requires_grad]
        if params:
            param_groups.append({'params': params, 'lr': stage_lr, 'name': f'enc_stage_{i}'})
            log(f"   encoder_stage_{i}: lr={stage_lr:.2e}")

    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if getattr(model, 'segmentation_head', None) is not None:
        dec_params += [p for p in model.segmentation_head.parameters() if p.requires_grad]
    if dec_params:
        param_groups.append({'params': dec_params, 'lr': base_lr, 'name': 'decoder_head'})
        log(f"   decoder_head: lr={base_lr:.2e}")

    opt_name = opt_name.lower()
    if opt_name == 'sgd':
        return optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(param_groups, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(param_groups, weight_decay=weight_decay)
    else:
        return optim.Adam(param_groups, weight_decay=weight_decay)


def get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-6):
    """Scheduler'ы, кроме Poly, шагают по эпохам. Poly — по итерациям (PolyLRSchedulerIter)."""
    sn = scheduler_name.lower()
    if sn == 'poly':
        # Возвращаем None — Poly будет создан отдельно как PolyLRSchedulerIter
        return None
    elif sn == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif sn == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif sn == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    return None


class PolyLRSchedulerIter:
    """Poly LR по итерациям: lr = base_lr * (1 - iter/max_iter)^power."""
    def __init__(self, optimizer, max_iter, power=0.9, min_lr=0.0):
        self.optimizer = optimizer
        self.max_iter = max(1, int(max_iter))
        self.power = float(power)
        self.min_lr = float(min_lr)
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_iter = 0

    def step(self):
        self.current_iter += 1
        factor = max(0.0, (1.0 - self.current_iter / self.max_iter)) ** self.power
        for i, g in enumerate(self.optimizer.param_groups):
            g['lr'] = max(self.min_lr, self.base_lrs[i] * factor)

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


class ModelEMA:
    """Exponential Moving Average весов модели."""
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k not in msd:
                continue
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)

    def state_dict(self):
        return self.ema.state_dict()


def predict_batch_multiscale(model, images, device, scales, use_amp):
    """Усредняет логиты модели по нескольким масштабам входа."""
    H, W = images.shape[2], images.shape[3]
    if not scales or len(scales) == 1:
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
            return model(images)

    avg = None
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            scaled = images
        else:
            nh = max(16, int(round(H * s / 16)) * 16)
            nw = max(16, int(round(W * s / 16)) * 16)
            scaled = torch.nn.functional.interpolate(
                images, size=(nh, nw), mode='bilinear', align_corners=False)
        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
            out = model(scaled)
        if out.shape[2:] != (H, W):
            out = torch.nn.functional.interpolate(
                out.float(), size=(H, W), mode='bilinear', align_corners=False)
        avg = out if avg is None else avg + out
    return avg / len(scales)


# =============================================================================
# Метрики / Auto batch / Сохранение / Визуализация
# =============================================================================
def compute_metrics(pred_logits, true_masks, num_classes, ignore_index=None,
                    output_activation='argmax', threshold=0.5):
    if num_classes == 1:
        probs = torch.sigmoid(pred_logits)
        pred = (probs > threshold).float()
        true = true_masks.float()
        if true.dim() == 3:
            true = true.unsqueeze(1)
        if ignore_index is not None:
            m = (true == ignore_index)
            pred = pred.masked_fill(m, 0)
            true = true.masked_fill(m, 0)
        inter = (pred * true).sum(dim=(2, 3))
        ps, ts = pred.sum(dim=(2, 3)), true.sum(dim=(2, 3))
        union = ps + ts - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        dice = (2 * inter + 1e-6) / (ps + ts + 1e-6)
        tp = inter
        prec = tp / (tp + (ps - inter) + 1e-6)
        rec = tp / (tp + (ts - inter) + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        acc = (pred == true).float().sum(dim=(2, 3)) / (pred.shape[2] * pred.shape[3])
        map50 = (iou > 0.5).float().mean().item()
        return {
            'IoU': iou.mean().item(), 'Dice': dice.mean().item(),
            'Precision': prec.mean().item(), 'Recall': rec.mean().item(),
            'F1': f1.mean().item(), 'Accuracy': acc.mean().item(),
            'mAP50': map50, 'mAP50-95': iou.mean().item(),
        }
    else:
        if output_activation == 'softmax':
            pred = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
        else:
            pred = torch.argmax(pred_logits, dim=1)
        true = true_masks.long()
        if ignore_index is not None:
            valid = (true != ignore_index)
            pred, true = pred[valid], true[valid]
        iou_l, dice_l, prec_l, rec_l = [], [], [], []
        for c in range(num_classes):
            pc = (pred == c); tc = (true == c)
            inter = (pc & tc).sum().float()
            union = (pc | tc).sum().float()
            iou_l.append((inter + 1e-6) / (union + 1e-6))
            tp = inter
            fp = (pc & ~tc).sum().float()
            fn = (~pc & tc).sum().float()
            dice_l.append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
            prec_l.append(tp / (tp + fp + 1e-6))
            rec_l.append(tp / (tp + fn + 1e-6))
        miou = torch.mean(torch.tensor(iou_l)).item()
        mdice = torch.mean(torch.tensor(dice_l)).item()
        prec = torch.mean(torch.tensor(prec_l)).item()
        rec = torch.mean(torch.tensor(rec_l)).item()
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        acc = (pred == true).float().mean().item()
        return {
            'IoU': miou, 'Dice': mdice, 'Precision': prec, 'Recall': rec,
            'F1': f1, 'Accuracy': acc, 'mAP50': miou, 'mAP50-95': miou,
        }


def auto_batch_size(device, input_size, mem_factor=AUTO_BATCH_MEM_FACTOR,
                    min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX):
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            free, _ = torch.cuda.mem_get_info(device)
            free_gb = free / (1024 ** 3)
            allowed = min(total * mem_factor, free_gb)
            mem_per_img = 2.8 * (input_size / 1024.0) ** 2
            suggested = max(min_batch, min(int(allowed / mem_per_img), max_batch))
            log(f"Auto batch (GPU): total={total:.1f}GB, free={free_gb:.1f}GB, "
                f"limit={mem_factor*100:.0f}% ({allowed:.1f}GB), "
                f"mem/img≈{mem_per_img:.2f}GB -> batch={suggested}")
            return suggested
        except Exception as e:
            log(f"⚠️ Ошибка определения GPU-памяти: {e}")
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / (1024 ** 3)
        allowed = free_ram * mem_factor
        mem_per_img = 2.8 * (input_size / 1024.0) ** 2
        suggested = max(min_batch, min(int(allowed / mem_per_img), max_batch))
        log(f"Auto batch (RAM): free={free_ram:.1f}GB, limit={mem_factor*100:.0f}% -> batch={suggested}")
        return suggested
    except ImportError:
        log("⚠️ psutil не установлен, auto batch = 8")
        return 8


def save_args_yaml(args_dict, save_path):
    if HAS_YAML:
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(args_dict, f, default_flow_style=False, allow_unicode=True)
        log(f"Аргументы сохранены в YAML: {save_path}")
    else:
        p = save_path.replace('.yaml', '.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        log(f"Аргументы сохранены в JSON: {p}")


def save_model_meta(meta_dict, model_path):
    p = os.path.splitext(model_path)[0] + '_meta.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    log(f"Метаданные сохранены в {p}")


def save_training_plots(history_dict, save_dir):
    if not HAS_MATPLOTLIB:
        return
    ep = history_dict['epoch']
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax[0, 0].plot(ep, history_dict['train_loss'], label='Train')
    if history_dict['val_loss']:
        ax[0, 0].plot(ep, history_dict['val_loss'], label='Val')
    ax[0, 0].set_xlabel('Epoch'); ax[0, 0].set_ylabel('Loss'); ax[0, 0].legend(); ax[0, 0].set_title('Loss')
    ax[0, 1].plot(ep, history_dict['iou'], label='IoU'); ax[0, 1].plot(ep, history_dict['dice'], label='Dice')
    ax[0, 1].set_xlabel('Epoch'); ax[0, 1].legend(); ax[0, 1].set_title('IoU & Dice')
    ax[1, 0].plot(ep, history_dict['precision'], label='Precision')
    ax[1, 0].plot(ep, history_dict['recall'], label='Recall')
    ax[1, 0].plot(ep, history_dict['f1'], label='F1')
    ax[1, 0].set_xlabel('Epoch'); ax[1, 0].legend(); ax[1, 0].set_title('P/R/F1')
    ax[1, 1].plot(ep, history_dict['map50'], label='mAP50')
    ax[1, 1].plot(ep, history_dict['map50_95'], label='mAP50-95')
    ax[1, 1].set_xlabel('Epoch'); ax[1, 1].legend(); ax[1, 1].set_title('mAP')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'results.png'), dpi=150); plt.close()
    log(f"График обучения: {os.path.join(save_dir, 'results.png')}")


def save_csv_results(history_dict, save_path):
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=history_dict.keys()); w.writeheader()
        for t in zip(*history_dict.values()):
            w.writerow(dict(zip(history_dict, t)))
    log(f"CSV история: {save_path}")


def save_pr_curve(preds, masks, save_dir, threshold=0.5):
    if not HAS_MATPLOTLIB:
        return
    probs = torch.sigmoid(preds).cpu().numpy().flatten()
    masks = masks.cpu().numpy().flatten()
    ths = np.linspace(0, 1, 51)
    P, R, F = [], [], []
    for th in ths:
        pb = (probs > th).astype(np.float32)
        tp = np.sum(pb * masks); fp = np.sum(pb * (1 - masks)); fn = np.sum((1 - pb) * masks)
        p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8); f = 2 * p * r / (p + r + 1e-8)
        P.append(p); R.append(r); F.append(f)
    plt.figure(figsize=(8, 6)); plt.plot(R, P, label='PR'); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('PR'); plt.grid(True); plt.savefig(os.path.join(save_dir, 'MaskPR_curve.png')); plt.close()
    plt.figure(figsize=(10, 6)); plt.plot(ths, P, label='Precision'); plt.plot(ths, R, label='Recall'); plt.plot(ths, F, label='F1')
    plt.xlabel('Threshold'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'MaskF1_curve.png')); plt.close()


def save_confusion_matrix(preds, true_masks, save_dir, num_classes, ignore_index=None, threshold=0.5):
    if not HAS_MATPLOTLIB:
        return
    if num_classes == 1:
        pb = (torch.sigmoid(preds) > threshold).long()
        tb = (true_masks > 0.5).long() if true_masks.dim() == 3 else true_masks.long().unsqueeze(1)
        if ignore_index is not None:
            m = (true_masks == ignore_index)
            pb, tb = pb.masked_fill(m, 0), tb.masked_fill(m, 0)
        tp = ((pb == 1) & (tb == 1)).sum().item()
        fp = ((pb == 1) & (tb == 0)).sum().item()
        fn = ((pb == 0) & (tb == 1)).sum().item()
        tn = ((pb == 0) & (tb == 0)).sum().item()
        cm = np.array([[tn, fp], [fn, tp]]); labels = ['BG', 'FG']
    else:
        pred = torch.argmax(preds, dim=1); true = true_masks.long()
        if ignore_index is not None:
            v = (true != ignore_index); pred, true = pred[v], true[v]
        cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for t in range(num_classes):
            for p in range(num_classes):
                cm[t, p] = ((true == t) & (pred == p)).sum().item()
        cm = cm.numpy(); labels = [f'C{i}' for i in range(num_classes)]
    for arr, name, fmt in [(cm, 'confusion_matrix', 'd'),
                           (cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8),
                            'confusion_matrix_normalized', '.2f')]:
        plt.figure(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
        if HAS_SEABORN:
            sns.heatmap(arr, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels)
        else:
            plt.imshow(arr, cmap='Blues'); plt.colorbar()
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    plt.text(j, i, format(arr[i, j], fmt), ha='center', va='center')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, f'{name}.png')); plt.close()


def save_train_batch_examples(loader, save_dir, num_batches=3, samples_per_batch=4,
                               num_classes=1, normalize=False):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    for b_idx, (images, masks) in enumerate(loader):
        if b_idx >= num_batches:
            break
        n = min(images.size(0), samples_per_batch)
        images, masks = images[:n], masks[:n]
        fig, ax = plt.subplots(2, n, figsize=(n * 4, 8))
        if n == 1:
            ax = ax.reshape(2, 1)
        for i in range(n):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            if normalize:
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
            ax[0, i].imshow((img * 255).astype(np.uint8)); ax[0, i].axis('off'); ax[0, i].set_title(f'Img {i+1}')
            m = masks[i].cpu().numpy()
            if num_classes == 1:
                ax[1, i].imshow((m > 0.5).astype(np.uint8) * 255, cmap='gray')
            else:
                vis = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
                for c in range(num_classes):
                    vis[m == c] = [c * 255 // max(1, num_classes - 1), 128, 255 - c * 128]
                ax[1, i].imshow(vis)
            ax[1, i].axis('off'); ax[1, i].set_title(f'Mask {i+1}')
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'train_batch{b_idx}.jpg')); plt.close()
    log(f"Сохранено {num_batches} батчей в {save_dir}")


def save_val_examples(loader, model, device, save_dir, num_samples=8,
                      num_classes=1, output_activation='argmax',
                      normalize=False, use_amp=False,
                      multi_scale=False, scales=None):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    imgs, tm, pm = [], [], []
    collected = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            if multi_scale:
                outputs = predict_batch_multiscale(model, images, device, scales, use_amp)
            else:
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                    outputs = model(images)
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                preds = (torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                         if output_activation == 'softmax'
                         else torch.argmax(outputs, dim=1)).float()
            for i in range(images.size(0)):
                if collected >= num_samples:
                    break
                img = images[i].cpu().permute(1, 2, 0).numpy()
                if normalize:
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                imgs.append((img * 255).astype(np.uint8))
                tm.append(masks[i].cpu().numpy())
                pm.append(preds[i].cpu().numpy())
                collected += 1
            if collected >= num_samples:
                break
    if not imgs:
        log("Нет примеров для сохранения валидации"); return
    n = len(imgs)
    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        ax = ax.reshape(1, 3)
    for i in range(n):
        ax[i, 0].imshow(imgs[i]); ax[i, 0].axis('off'); ax[i, 0].set_title('Image')
        for col, mask_arr, title in [(1, tm[i], 'True'), (2, pm[i], 'Pred')]:
            if num_classes == 1:
                ax[i, col].imshow((mask_arr > 0.5).astype(np.uint8) * 255, cmap='gray')
            else:
                vis = np.zeros_like(imgs[i])
                for c in range(num_classes):
                    vis[mask_arr == c] = [c * 255 // max(1, num_classes - 1), 128, 255 - c * 128]
                ax[i, col].imshow(vis)
            ax[i, col].axis('off'); ax[i, col].set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'val_examples.jpg')); plt.close()
    log(f"Сохранено {collected} val-примеров в {save_dir}/val_examples.jpg")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str)
    parser.add_argument('--params_file', type=str)
    args = parser.parse_args()

    log("WORKER STARTED (DeepLabV3 / DeepLabV3+)")

    if args.params_file:
        with open(args.params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
    elif args.params:
        params = json.loads(args.params)
    else:
        log("Ошибка: укажите --params или --params_file"); sys.exit(1)

    # ---------- Извлечение параметров ----------
    dataset_path = params.get('dataset_path')
    model_path = params.get('model_path')
    project_path = params.get('project_path')
    name = params.get('name')
    backbone = params.get('backbone', 'resnet50')
    output_stride = params.get('output_stride', 16)
    output_activation = params.get('output_activation', 'argmax')
    num_classes = params.get('num_classes', 21)
    ignore_index = params.get('ignore_index')
    class_weights = params.get('class_weights')
    loss_name = params.get('loss', 'CrossEntropy')
    loss_weights = params.get('loss_weights', {'ce': 1.0, 'dice': 1.0, 'focal': 0.0})
    optimizer_name = params.get('optimizer', 'SGD')
    lr = params.get('lr', 0.007)
    momentum = params.get('momentum', 0.9)
    weight_decay = params.get('weight_decay', 0.0001)
    scheduler_name = params.get('scheduler', 'Poly')
    epochs = params.get('epochs', 100)
    batch_size = params.get('batch_size', 8)
    two_phase = params.get('two_phase_training', False)
    freeze_epochs = params.get('freeze_epochs', 0)
    unfreeze_epochs = params.get('unfreeze_epochs', 0)
    unfreeze_last_n_stages = params.get('unfreeze_last_n_stages', None)  # None = весь энкодер
    augmentations = params.get('augmentations', {})
    device_pref = params.get('device', 'auto')
    threshold = params.get('threshold', 0.5)
    num_workers = params.get('num_workers', NUM_WORKERS_DEFAULT)
    input_size = params.get('input_size', 512)
    normalize = bool(params.get('normalize', False))
    resize_mode = params.get('resize_mode', 'hard')

    # Архитектура
    architecture = params.get('architecture', 'deeplabv3plus').lower()
    if architecture in ('deeplabv3plus', 'deeplabv3+'):
        architecture = 'deeplabv3plus'
    elif architecture == 'deeplabv3':
        architecture = 'deeplabv3'
    else:
        log(f"⚠️ Неизвестная архитектура '{architecture}', использую deeplabv3plus")
        architecture = 'deeplabv3plus'

    # Обучение: улучшения
    use_amp = bool(params.get('use_amp', True))
    accum_steps = max(1, int(params.get('accum_steps', 1)))
    use_ema = bool(params.get('use_ema', True))
    ema_decay = float(params.get('ema_decay', 0.999))
    use_layerwise_lr = bool(params.get('use_layerwise_lr', False))
    layerwise_decay = float(params.get('layerwise_decay', 0.1))
    poly_power = float(params.get('poly_power', 0.9))
    multi_scale_infer = bool(params.get('multi_scale_infer', False))
    infer_scales = params.get('infer_scales', [0.75, 1.0, 1.25])
    if isinstance(infer_scales, str):
        infer_scales = [float(s) for s in infer_scales.replace(' ', '').split(',') if s]

    # ---------- Устройство для auto-batch ----------
    if device_pref == 'auto':
        tmp_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.isdigit():
        tmp_dev = torch.device(f'cuda:{device_pref}') if (torch.cuda.is_available()
                            and int(device_pref) < torch.cuda.device_count()) else torch.device('cpu')
    elif device_pref.lower() in ('cuda', 'gpu'):
        tmp_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.lower() == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        tmp_dev = torch.device('mps')
    else:
        tmp_dev = torch.device('cpu')

    if batch_size == -1:
        batch_size = auto_batch_size(tmp_dev, input_size)
        log(f"Auto batch: выбран batch_size = {batch_size}")
    elif batch_size <= 0:
        log(f"Ошибка: batch_size={batch_size} должен быть положительным"); sys.exit(1)

    # ---------- Путь сохранения ----------
    if name and project_path:
        save_root = os.path.join(project_path, name)
    elif name:
        save_root = name
    elif project_path:
        save_root = project_path
    else:
        save_root = "runs/deeplabv3plus"
    os.makedirs(save_root, exist_ok=True)
    weights_dir = os.path.join(save_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # ---------- Лог старта ----------
    log("=== Запуск обучения DeepLabV3 / DeepLabV3+ ===")
    log(f"Архитектура: {architecture}")
    log(f"Датасет: {dataset_path}")
    log(f"Backbone: {backbone}, output_stride={output_stride}")
    log(f"Выходная активация: {output_activation}")
    log(f"Классов: {num_classes}, ignore_index={ignore_index}")
    log(f"Веса классов: {class_weights}")
    log(f"Loss: {loss_name}, веса: {loss_weights}")
    log(f"Оптимизатор: {optimizer_name}, LR={lr}, momentum={momentum}, WD={weight_decay}")
    log(f"Scheduler: {scheduler_name}")
    log(f"Эпохи: {epochs}, two_phase={two_phase} (freeze={freeze_epochs}, unfreeze={unfreeze_epochs})")
    if unfreeze_last_n_stages is not None:
        log(f"Размораживать только последние {unfreeze_last_n_stages} стадий энкодера")
    log(f"Batch size: {batch_size}, device: {device_pref}")
    log(f"Input size: {input_size}, normalize={normalize}, resize_mode={resize_mode}")
    log(f"AMP: {use_amp}, Accum: {accum_steps}")
    log(f"EMA: {use_ema} (decay={ema_decay})")
    log(f"Layer-wise LR: {use_layerwise_lr} (decay={layerwise_decay})")
    log(f"Poly power: {poly_power}")
    log(f"Multi-scale infer: {multi_scale_infer} (scales={infer_scales})")
    log(f"Аугментации: {augmentations}")
    log(f"Результаты: {save_root}")

    if not dataset_path or not os.path.exists(dataset_path):
        log(f"Ошибка: папка датасета не существует: {dataset_path}"); sys.exit(1)

    # ---------- Создание датасетов ----------
    try:
        if os.path.exists(os.path.join(dataset_path, 'images', 'train')) and \
           os.path.exists(os.path.join(dataset_path, 'masks', 'train')):
            log("Обнаружена U-Net-style структура (images/, masks/)")
            from glob import glob
            tr_imgs = sorted(glob(os.path.join(dataset_path, 'images', 'train', '*.*')))
            va_imgs = sorted(glob(os.path.join(dataset_path, 'images', 'val', '*.*')))
            def mk_pairs(imgs, mask_dir):
                return [(img, os.path.join(mask_dir, os.path.splitext(os.path.basename(img))[0] + '.png'))
                        for img in imgs
                        if os.path.exists(os.path.join(mask_dir,
                            os.path.splitext(os.path.basename(img))[0] + '.png'))]
            train_pairs = mk_pairs(tr_imgs, os.path.join(dataset_path, 'masks', 'train'))
            val_pairs = mk_pairs(va_imgs, os.path.join(dataset_path, 'masks', 'val'))
            train_ds = VOCSegmentationPairs(train_pairs, input_size, num_classes, ignore_index,
                                            augmentations, normalize=normalize, resize_mode=resize_mode)
            val_ds = (VOCSegmentationPairs(val_pairs, input_size, num_classes, ignore_index,
                                           {}, normalize=normalize, resize_mode=resize_mode)
                      if val_pairs else None)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                          if val_ds else None)
            train_len = len(train_ds)
            val_len = len(val_ds) if val_ds else 0
        else:
            log("Обнаружена PASCAL VOC структура (ImageSets/Segmentation)")
            train_loader, val_loader, train_len, val_len = create_voc_loaders(
                dataset_path, input_size, num_classes, augmentations, ignore_index,
                batch_size, num_workers, normalize=normalize, resize_mode=resize_mode)
    except Exception as e:
        log(f"Ошибка создания датасета: {e}"); sys.exit(1)

    log(f"Загружено {train_len} train, {val_len} val")

    # ---------- Модель ----------
    global model, best_loss_model_path, best_iou_model_path, final_model_path
    global best_val_loss, best_iou, current_epoch, stop_requested
    load_pretrained = (model_path is None) or (not os.path.exists(model_path))

    if architecture == 'deeplabv3plus':
        log("Инициализация: DeepLabV3+")
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights='imagenet' if load_pretrained else None,
            in_channels=3, classes=num_classes,
            encoder_output_stride=output_stride,
            decoder_use_batchnorm=True,
        )
    else:
        log("Инициализация: DeepLabV3 (без плюса)")
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights='imagenet' if load_pretrained else None,
            in_channels=3, classes=num_classes,
            encoder_output_stride=output_stride,
        )

    if model_path is not None and os.path.exists(model_path):
        sd = torch.load(model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing: log(f"⚠️ Отсутствуют ключи: {missing[:5]}...")
        if unexpected: log(f"⚠️ Лишние ключи: {unexpected[:5]}...")
        log(f"Загружены веса: {model_path}")

    # ---------- Устройство ----------
    if device_pref == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.isdigit():
        device = (torch.device(f'cuda:{device_pref}')
                  if torch.cuda.is_available() and int(device_pref) < torch.cuda.device_count()
                  else torch.device('cpu'))
    elif device_pref.lower() in ('cuda', 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.lower() == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    log(f"Device: {device}")

    # ---------- Auto class weights ----------
    if class_weights == 'auto':
        log("Автоматический расчёт весов классов...")
        counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        for _, mask in train_loader:
            m = mask.to(device)
            if ignore_index is not None:
                m = m[m != ignore_index]
            if m.numel() == 0:
                continue
            uniq, cnt = torch.unique(m, return_counts=True)
            for c, n_c in zip(uniq, cnt):
                if c < num_classes:
                    counts[c] += n_c
        total = counts.sum()
        class_weights = (total / (num_classes * (counts.float() + 1e-6))).tolist()
        log(f"Авто-веса: {class_weights}")

    criterion = get_loss_function(loss_name, num_classes, ignore_index,
                                  class_weights, loss_weights)

    # ---------- Двухфазный режим: подготовка ----------
    def freeze_backbone():
        for p in model.encoder.parameters():
            p.requires_grad = False
        log("Backbone заморожен полностью.")

    def unfreeze_backbone():
        if unfreeze_last_n_stages is None:
            for p in model.encoder.parameters():
                p.requires_grad = True
            log("Backbone разморожен полностью.")
        else:
            try:
                stages = model.encoder.get_stages()
                n = len(stages)
                k = min(unfreeze_last_n_stages, n)
                for i, stage in enumerate(stages):
                    if i >= n - k:
                        for p in stage.parameters():
                            p.requires_grad = True
                log(f"Разморожены последние {k} из {n} стадий энкодера.")
            except AttributeError:
                log("⚠️ get_stages() недоступен — размораживаю весь энкодер.")
                for p in model.encoder.parameters():
                    p.requires_grad = True

    if two_phase:
        total_epochs = freeze_epochs + unfreeze_epochs
        if total_epochs != epochs:
            log(f"⚠️ Сумма фаз={total_epochs} != epochs={epochs}. Использую {total_epochs}.")
            epochs = total_epochs
        freeze_backbone()
    else:
        freeze_epochs = 0

    # ---------- Оптимизатор и Scheduler ----------
    if use_layerwise_lr:
        log("Создаю layer-wise оптимизатор")
        optimizer = get_layerwise_optimizer(model, optimizer_name, lr,
                                             momentum, weight_decay, layerwise_decay)
    else:
        optimizer = get_optimizer(model, optimizer_name, lr, momentum, weight_decay)

    def make_scheduler(opt, n_epochs):
        if scheduler_name.lower() == 'poly':
            n_iters = max(1, (n_epochs * len(train_loader)) // accum_steps)
            log(f"Poly LR: max_iter={n_iters}, power={poly_power}")
            return PolyLRSchedulerIter(opt, max_iter=n_iters, power=poly_power)
        return get_scheduler(opt, scheduler_name, n_epochs)

    current_phase_epochs = epochs
    scheduler = make_scheduler(optimizer, current_phase_epochs)

    # ---------- EMA ----------
    ema = ModelEMA(model, decay=ema_decay, device=device) if use_ema else None
    if use_ema:
        log(f"EMA инициализирован (decay={ema_decay})")

    # ---------- Скаляр AMP ----------
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))

    # ---------- Точки сохранения ----------
    best_loss_model_path = os.path.join(weights_dir, 'best_loss.pth')
    best_iou_model_path = os.path.join(weights_dir, 'best_iou.pth')
    final_model_path = os.path.join(weights_dir, 'last.pth')

    # ---------- Сохранение args / meta ----------
    args_dict = {
        'architecture': architecture, 'backbone': backbone,
        'output_stride': output_stride, 'output_activation': output_activation,
        'num_classes': num_classes, 'ignore_index': ignore_index,
        'class_weights': class_weights, 'loss': loss_name, 'loss_weights': loss_weights,
        'optimizer': optimizer_name, 'lr': lr, 'momentum': momentum,
        'weight_decay': weight_decay, 'scheduler': scheduler_name,
        'epochs': epochs, 'batch_size': batch_size,
        'two_phase_training': two_phase, 'freeze_epochs': freeze_epochs,
        'unfreeze_epochs': unfreeze_epochs, 'unfreeze_last_n_stages': unfreeze_last_n_stages,
        'augmentations': augmentations, 'device': device_pref,
        'threshold': threshold, 'num_workers': num_workers,
        'input_size': input_size, 'normalize': normalize, 'resize_mode': resize_mode,
        'use_amp': use_amp, 'accum_steps': accum_steps,
        'use_ema': use_ema, 'ema_decay': ema_decay,
        'use_layerwise_lr': use_layerwise_lr, 'layerwise_decay': layerwise_decay,
        'poly_power': poly_power,
        'multi_scale_infer': multi_scale_infer, 'infer_scales': infer_scales,
        'project': project_path, 'name': name,
        'date': datetime.now().isoformat(),
    }
    save_args_yaml(args_dict, os.path.join(save_root, 'args.yaml'))

    meta_info = {
        'architecture': architecture, 'backbone': backbone,
        'output_stride': output_stride, 'output_activation': output_activation,
        'num_classes': num_classes, 'ignore_index': ignore_index,
        'loss': loss_name, 'optimizer': optimizer_name, 'lr': lr,
        'batch_size': batch_size, 'epochs': epochs,
        'device_used': str(device), 'experiment_name': name,
        'threshold': threshold, 'input_size': input_size,
        'normalize': normalize, 'resize_mode': resize_mode,
        'use_ema': use_ema, 'ema_decay': ema_decay,
        'use_layerwise_lr': use_layerwise_lr, 'layerwise_decay': layerwise_decay,
        'poly_power': poly_power,
        'multi_scale_infer': multi_scale_infer, 'infer_scales': infer_scales,
        'best_val_loss': None, 'best_iou': None,
        'date': datetime.now().isoformat(),
    }
    save_model_meta(meta_info, best_loss_model_path)
    save_model_meta(meta_info, final_model_path)
    log("Метаданные сохранены (начальная версия).")

    # ---------- Цикл обучения ----------
    patience_counter = 0
    best_val_loss = float('inf')
    best_iou = -float('inf')
    log("=== Начало обучения ===")

    for epoch in range(1, epochs + 1):
        current_epoch = epoch
        if stop_requested:
            log("Остановка по запросу пользователя."); break

        # --- Переключение фазы ---
        if two_phase and epoch == freeze_epochs + 1:
            log("=" * 60)
            log(f"Переключение во вторую фазу (эпоха {epoch}).")
            unfreeze_backbone()

            # Пересоздаём optimizer (layer-wise или обычный)
            if use_layerwise_lr:
                optimizer = get_layerwise_optimizer(
                    model, optimizer_name, lr, momentum, weight_decay, layerwise_decay)
            else:
                optimizer = get_optimizer(model, optimizer_name, lr, momentum, weight_decay)

            # Scheduler для второй фазы
            current_phase_epochs = epochs - freeze_epochs
            scheduler = make_scheduler(optimizer, current_phase_epochs)

            # Сбрасываем EMA — начинаем усреднять размороженную модель заново
            if use_ema:
                ema = ModelEMA(model, decay=ema_decay, device=device)
                log(f"EMA сброшен (decay={ema_decay})")

            # Scaler не зависит от optimizer, оставляем
            log("=" * 60)

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for b_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            target = masks.unsqueeze(1).float() if num_classes == 1 else masks.long()

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, target) / accum_steps

            scaler.scale(loss).backward()

            if (b_idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                if isinstance(scheduler, PolyLRSchedulerIter):
                    scheduler.step()

            train_loss += loss.item() * accum_steps * images.size(0)

        # Остаток накопленных градиентов
        if len(train_loader) % accum_steps != 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            if isinstance(scheduler, PolyLRSchedulerIter):
                scheduler.step()

        train_loss /= train_len

        # ---------- Валидация (с EMA и multi-scale) ----------
        if val_loader is not None:
            eval_model = ema.ema if ema is not None else model
            eval_model.eval()
            val_loss = 0.0
            all_preds, all_masks = [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)

                    if multi_scale_infer:
                        outputs = predict_batch_multiscale(
                            eval_model, images, device, infer_scales, use_amp)
                    else:
                        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                            outputs = eval_model(images)

                    target = masks.unsqueeze(1).float() if num_classes == 1 else masks.long()
                    vloss = criterion(outputs, target)
                    all_preds.append(outputs.float().cpu())
                    all_masks.append(masks.cpu())
                    val_loss += vloss.item() * images.size(0)
            val_loss /= val_len
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            metrics = compute_metrics(all_preds, all_masks, num_classes,
                                      ignore_index=ignore_index,
                                      output_activation=output_activation,
                                      threshold=threshold)
        else:
            val_loss, metrics = None, None

        # ---------- Сохранение лучших ----------
        state_to_save = ema.state_dict() if ema is not None else model.state_dict()
        ema_note = " (EMA)" if ema is not None else ""

        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state_to_save, best_loss_model_path)
                patience_counter = 0
                log(f"✅ best_loss{ema_note}: val_loss={best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= OVERFITTING_MIN_PATIENCE:
                    gap = (val_loss - train_loss) / (train_loss + 1e-8)
                    if gap > OVERFITTING_GAP_THRESHOLD:
                        log(f"⚠️ Переобучение: gap={gap*100:.1f}% > {OVERFITTING_GAP_THRESHOLD*100:.0f}%. Стоп.")
                        break
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"⚠️ Early stopping: {EARLY_STOPPING_PATIENCE} эпох без улучшений.")
                    break

            if metrics is not None and metrics['IoU'] > best_iou:
                best_iou = metrics['IoU']
                torch.save(state_to_save, best_iou_model_path)
                log(f"✅ best_iou{ema_note}: IoU={best_iou:.4f}")
        else:
            if best_val_loss == float('inf'):
                best_val_loss = train_loss
                torch.save(state_to_save, best_loss_model_path)
                log(f"✅ Модель сохранена (нет валидации){ema_note}")

        # ---------- История ----------
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        for k in ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'map50', 'map50_95']:
            if metrics:
                # ключи в metrics — с заглавной буквы
                key_map = {'iou': 'IoU', 'dice': 'Dice', 'f1': 'F1',
                           'precision': 'Precision', 'recall': 'Recall',
                           'accuracy': 'Accuracy', 'map50': 'mAP50', 'map50_95': 'mAP50-95'}
                history[k].append(metrics[key_map[k]])
            else:
                history[k].append(float('nan'))

        # ---------- Scheduler step (для НЕ-Poly) ----------
        if scheduler is not None and not isinstance(scheduler, PolyLRSchedulerIter):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

        # ---------- Лог эпохи ----------
        pct = int(epoch / epochs * 100)
        log(f"PROGRESS:{pct}:{epoch}:{epochs}")
        cur_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        lr_str = f"LR={cur_lrs[0]}" if len(cur_lrs) == 1 else f"LR=[{','.join(cur_lrs[:3])}...]"
        if val_loss is not None:
            log(f"Эпоха {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"IoU: {metrics['IoU']:.4f} | Dice: {metrics['Dice']:.4f} | F1: {metrics['F1']:.4f} | {lr_str}")
        else:
            log(f"Эпоха {epoch}/{epochs} | Train: {train_loss:.4f} | (нет val) | {lr_str}")

    # ---------- Финал ----------
    final_state = ema.state_dict() if ema is not None else model.state_dict()
    torch.save(final_state, final_model_path)
    log(f"Финальная модель{ema_note}: {final_model_path}")

    meta_info['best_val_loss'] = best_val_loss if best_val_loss != float('inf') else None
    meta_info['best_iou'] = best_iou if best_iou > -float('inf') else None
    meta_info['date'] = datetime.now().isoformat()
    save_model_meta(meta_info, best_loss_model_path)
    if os.path.exists(best_iou_model_path):
        save_model_meta(meta_info, best_iou_model_path)
    save_model_meta(meta_info, final_model_path)

    # ---------- Пост-обработка ----------
    if val_loader is not None:
        log("=== Итоговые отчёты ===")
        eval_path = best_iou_model_path if os.path.exists(best_iou_model_path) else best_loss_model_path
        log(f"Использую: {os.path.basename(eval_path)}")
        model.load_state_dict(torch.load(eval_path, map_location='cpu'))
        model = model.to(device)
        model.eval()

        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                if multi_scale_infer:
                    outputs = predict_batch_multiscale(model, images, device, infer_scales, use_amp)
                else:
                    with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                        outputs = model(images)
                all_preds.append(outputs.float().cpu())
                all_masks.append(masks.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        save_csv_results(history, os.path.join(save_root, 'results.csv'))
        if HAS_MATPLOTLIB:
            save_training_plots(history, save_root)
            if num_classes == 1:
                save_pr_curve(all_preds, all_masks, save_root, threshold)
            save_confusion_matrix(all_preds, all_masks, save_root, num_classes, ignore_index, threshold)
        try:
            save_train_batch_examples(train_loader, save_root, NUM_TRAIN_BATCHES_TO_SAVE,
                                       min(4, batch_size), num_classes, normalize)
        except Exception as e:
            log(f"⚠️ Ошибка визуализации train-батчей: {e}")
        try:
            save_val_examples(val_loader, model, device, save_root,
                              NUM_VAL_SAMPLES_TO_SAVE, num_classes, output_activation,
                              normalize, use_amp, multi_scale_infer, infer_scales)
        except Exception as e:
            log(f"⚠️ Ошибка визуализации val-примеров: {e}")

        final_metrics = compute_metrics(all_preds, all_masks, num_classes, ignore_index,
                                        output_activation, threshold)
        log(f"METRICS: {json.dumps(final_metrics)}")
    else:
        log("METRICS: null (нет валидации)")
        save_csv_results(history, os.path.join(save_root, 'results.csv'))

    log("=== Обучение завершено ===")
    if os.path.exists(best_loss_model_path):
        log(f"best_loss: {best_loss_model_path}")
    if os.path.exists(best_iou_model_path):
        log(f"best_iou:  {best_iou_model_path}")
    log(f"last:      {final_model_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()