#!/usr/bin/env python
# train_DeepLabV3_worker.py
# =============================================================================
# Обучение DeepLabV3 / DeepLabV3+ (segmentation_models_pytorch)
#
# ПРЕСЕТЫ ОБУЧЕНИЯ (скопируйте нужный в UI/JSON):
#
# 1) DeepLabV3+ PASCAL VOC Standard (OS=16, 513x513)
#    architecture: deeplabv3plus
#    backbone: resnet101
#    output_stride: 16
#    input_size: 513
#    lr: 0.007
#    optimizer: SGD (momentum=0.9, wd=1e-4)
#    scheduler: Poly (power=0.9)
#    epochs: 60
#    batch_size: 16 (или -1 для авто)
#    loss: CrossEntropy
#    use_ema: True, ema_decay: 0.999
#    two_phase_training: True, freeze_epochs: 30, unfreeze_epochs: 30
#    unfreeze_last_n_stages: 2
#    freeze_bn_second_phase: True
#    use_layerwise_lr: False
#    multi_scale_infer: True, infer_scales: [0.75, 1.0, 1.25]
#
# 2) DeepLabV3+ High-Resolution (OS=8, 1025x1025)
#    architecture: deeplabv3plus
#    backbone: resnet101
#    output_stride: 8
#    input_size: 1025
#    lr: 0.001
#    optimizer: AdamW (wd=1e-4)
#    scheduler: Poly (power=0.9)
#    epochs: 40
#    batch_size: 4-8 (OS=8 требует много памяти)
#    loss: CrossEntropy
#    use_ema: True, ema_decay: 0.999
#    two_phase_training: False
#    use_layerwise_lr: True, layerwise_decay: 0.1
#    multi_scale_infer: False (память)
#
# 3) DeepLabV3 Memory-Efficient (OS=32, 513x513)
#    architecture: deeplabv3
#    backbone: resnet50
#    output_stride: 32
#    input_size: 513
#    lr: 0.01
#    optimizer: SGD (momentum=0.9, wd=1e-4)
#    scheduler: Poly (power=0.9)
#    epochs: 80
#    batch_size: -1 (авто)
#    loss: CrossEntropy
#    use_ema: True, ema_decay: 0.999
#    two_phase_training: False
#    multi_scale_infer: True, infer_scales: [0.75, 1.0]
#
# 4) DeepLabV3+ Fine-Tuning on Small Dataset (OS=16, 513x513)
#    architecture: deeplabv3plus
#    backbone: resnet50
#    output_stride: 16
#    input_size: 513
#    lr: 0.0001
#    optimizer: AdamW (wd=1e-4)
#    scheduler: Poly (power=0.9)
#    epochs: 30
#    batch_size: 8
#    loss: CrossEntropy+Dice (loss_weights: ce=1.0, dice=1.0)
#    class_weights: auto (для дисбаланса)
#    use_ema: True, ema_decay: 0.99
#    two_phase_training: True, freeze_epochs: 10, unfreeze_epochs: 20
#    unfreeze_last_n_stages: 2
#    freeze_bn_second_phase: True
#    use_layerwise_lr: True, layerwise_decay: 0.1
#    multi_scale_infer: True, infer_scales: [0.75, 1.0, 1.25]
#
# =============================================================================
# Ключевые возможности:
#   - Архитектуры: DeepLabV3+ и DeepLabV3
#   - Бинарная и многоклассовая сегментация, ignore_index, class_weights
#   - Двухфазное обучение (freeze/unfreeze backbone, опц. только последних N стадий)
#   - Заморозка BatchNorm на первой и второй фазах
#   - Безопасный auto_batch_size (50% VRAM + учёт input_size)
#   - Mixed Precision (AMP): bf16 → без GradScaler; fp16 → с GradScaler
#   - Gradient Accumulation
#   - EMA (Exponential Moving Average) с адаптацией decay под размер датасета
#   - Layer-wise LR для энкодера
#   - Poly LR по итерациям (а не по эпохам)
#   - Multi-scale inference на валидации/тестах
#   - Три точки сохранения: best_loss, best_iou, last
#   - Авто-паддинг входа до кратного output_stride (513 → 528, 1025 → 1032)
#   - Проверка crop_size по формуле output_stride * k + 1
#   - LR ×0.1 на первой фазе (frozen backbone)
#   - isfinite(loss) guard — пропуск nan/inf батчей
#   - Диагностика class balance на train
#   - class_weights автоматически на нужном device
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

# Патч 4: увеличено с 5 до 15 (малые датасеты требуют больше эпох без улучшений)
EARLY_STOPPING_PATIENCE = 15
OVERFITTING_GAP_THRESHOLD = 0.5
OVERFITTING_MIN_PATIENCE = 3

ATROUS_RATES_MAP = {
    32: (3, 6, 9),
    16: (6, 12, 18),
    8: (12, 24, 36),
}

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

def read_voc_pairs(txt_path, jpeg_dir, seg_dir):
    if not txt_path or not os.path.exists(txt_path):
        return []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
    except UnicodeDecodeError:
        with open(txt_path, 'r', encoding='cp1251') as f:
            lines = [l.strip() for l in f if l.strip()]
    pairs = []
    for name in lines:
        img = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            cand = os.path.join(jpeg_dir, name + ext)
            if os.path.exists(cand):
                img = cand
                break
        mask = os.path.join(seg_dir, name + '.png')
        if img is not None and os.path.exists(mask):
            pairs.append((img, mask))
        else:
            log(f"⚠️ Пропуск: {name} — нет изображения или маски")
    return pairs


def read_voc_file_list(txt_path, root_dir):
    jpeg_dir = os.path.join(root_dir, 'JPEGImages')
    seg_dir = os.path.join(root_dir, 'SegmentationClass')
    return read_voc_pairs(txt_path, jpeg_dir, seg_dir)

def _resolve_path(root, p):
    if not p:
        return None
    p = str(p).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(root, p))


def create_loaders_from_yaml(yaml_path, default_input_size, default_num_classes,
                              augmentations, ignore_index, batch_size,
                              num_workers, normalize, resize_mode):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    root_raw = str(cfg.get('path', '') or '').strip()
    if root_raw:
        if os.path.isabs(root_raw):
            root = root_raw
        else:
            root = os.path.normpath(os.path.join(os.path.dirname(yaml_path), root_raw))
    else:
        root = os.path.dirname(os.path.abspath(yaml_path))
    log(f"📄 YAML dataset root: {root}")

    nc_raw = cfg.get('num_classes')
    try:
        nc = int(nc_raw) if nc_raw is not None else default_num_classes
    except (TypeError, ValueError):
        nc = default_num_classes
    if nc < 1:
        nc = default_num_classes
    log(f"📄 YAML num_classes: {nc}")

    class_names = cfg.get('class_names')
    if class_names:
        if isinstance(class_names, (list, tuple)):
            log(f"📄 YAML class_names ({len(class_names)}): {class_names}")
        else:
            log(f"📄 YAML class_names: {class_names}")

    train_pairs = []
    val_pairs = []

    if cfg.get('train_images'):
        tr_imgs = _resolve_path(root, cfg.get('train_images'))
        tr_masks = _resolve_path(root, cfg.get('train_masks', ''))
        if not tr_imgs or not tr_masks:
            raise RuntimeError("YAML: указан 'train_images', но нет 'train_masks'")

        def mk_pairs(imgs_dir, masks_dir):
            if not imgs_dir or not masks_dir:
                return []
            if not os.path.isdir(imgs_dir) or not os.path.isdir(masks_dir):
                log(f"⚠️ Папка не существует: imgs={imgs_dir}, masks={masks_dir}")
                return []
            from glob import glob
            pairs = []
            for img in sorted(glob(os.path.join(imgs_dir, '*.*'))):
                stem = os.path.splitext(os.path.basename(img))[0]
                for ext in ('.png', '.tif', '.tiff', '.bmp'):
                    cand = os.path.join(masks_dir, stem + ext)
                    if os.path.exists(cand):
                        pairs.append((img, cand))
                        break
            return pairs

        train_pairs = mk_pairs(tr_imgs, tr_masks)
        va_imgs = _resolve_path(root, cfg.get('val_images', '') or '')
        va_masks = _resolve_path(root, cfg.get('val_masks', '') or '')
        if va_imgs and va_masks:
            val_pairs = mk_pairs(va_imgs, va_masks)
        log(f"📄 U-Net-style: train={len(train_pairs)}, val={len(val_pairs)}")

    elif cfg.get('train_txt'):
        jpeg_dir = _resolve_path(root, cfg.get('jpeg_dir', 'JPEGImages'))
        seg_dir = _resolve_path(root, cfg.get('seg_dir', 'SegmentationClass'))
        train_txt = _resolve_path(root, cfg.get('train_txt'))
        val_txt = _resolve_path(root, cfg['val_txt']) if cfg.get('val_txt') else None

        if not jpeg_dir or not seg_dir or not train_txt:
            raise RuntimeError("YAML: для VOC нужны 'jpeg_dir', 'seg_dir', 'train_txt'")

        train_pairs = read_voc_pairs(train_txt, jpeg_dir, seg_dir)
        if val_txt:
            val_pairs = read_voc_pairs(val_txt, jpeg_dir, seg_dir)
        log(f"📄 VOC-style: train={len(train_pairs)}, val={len(val_pairs)}")

    else:
        raise RuntimeError(
            "YAML должен содержать 'train_images' (U-Net-style) "
            "или 'train_txt' (VOC-style)."
        )

    if not train_pairs:
        raise RuntimeError("YAML: не удалось собрать ни одной обучающей пары.")

    train_ds = VOCSegmentationPairs(
        train_pairs, default_input_size, nc, ignore_index,
        augmentations, normalize=normalize, resize_mode=resize_mode
    )
    val_ds = (
        VOCSegmentationPairs(
            val_pairs, default_input_size, nc, ignore_index,
            {}, normalize=normalize, resize_mode=resize_mode
        ) if val_pairs else None
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers) if val_ds else None
    )
    return train_loader, val_loader, len(train_ds), (len(val_ds) if val_ds else 0), nc


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
# Loss / Optimizer / Scheduler
# =============================================================================
def get_loss_function(loss_name, num_classes, ignore_index=None,
                      class_weights=None, loss_weights=None, device='cpu'):
    """
    Патч 1: weights_tensor создаётся на нужном устройстве (device).
    """
    loss_weights = loss_weights or {'ce': 1.0, 'dice': 1.0, 'focal': 0.0}

    weights_tensor = None
    if class_weights is not None and class_weights != 'auto':
        try:
            if isinstance(class_weights, str) and os.path.exists(class_weights):
                with open(class_weights, 'r') as f:
                    class_weights = json.load(f)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
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
    """
    Layer-wise LR. Устойчив к тому, что get_stages() может возвращать
    смешанный список (модули + int / другие объекты).
    """
    try:
        raw_stages = model.encoder.get_stages()
    except AttributeError:
        log("⚠️ encoder.get_stages() недоступен — использую единый LR")
        return get_optimizer(model, opt_name, base_lr, momentum, weight_decay)

    stages = []
    skipped = 0
    for s in raw_stages:
        if hasattr(s, 'parameters'):
            stages.append(s)
        else:
            skipped += 1
    if skipped:
        log(f"⚠️ get_stages() вернул {skipped} не-модульных элементов — пропущены.")
    if not stages:
        log("⚠️ В get_stages() нет модулей — использую единый LR")
        return get_optimizer(model, opt_name, base_lr, momentum, weight_decay)

    n = len(stages)
    param_groups = []
    for i, stage in enumerate(stages):
        stage_lr = base_lr * (decay_factor ** (n - 1 - i))
        params = [p for p in stage.parameters() if p.requires_grad]
        if params:
            param_groups.append({'params': params, 'lr': stage_lr,
                                 'name': f'enc_stage_{i}'})
            log(f"   encoder_stage_{i}: lr={stage_lr:.2e}, params={len(params)}")

    dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if getattr(model, 'segmentation_head', None) is not None:
        dec_params += [p for p in model.segmentation_head.parameters()
                       if p.requires_grad]
    if dec_params:
        param_groups.append({'params': dec_params, 'lr': base_lr,
                             'name': 'decoder_head'})
        log(f"   decoder_head: lr={base_lr:.2e}, params={len(dec_params)}")

    if not param_groups:
        log("⚠️ Не удалось собрать param_groups — использую единый LR")
        return get_optimizer(model, opt_name, base_lr, momentum, weight_decay)

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
    sn = scheduler_name.lower()
    if sn == 'poly':
        return None
    elif sn == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif sn == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    elif sn == 'reducelronplateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    return None


class PolyLRSchedulerIter:
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


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def _pad_to_multiple(x, multiple=16):
    h, w = x.shape[-2], x.shape[-1]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return x, (h, w)
    x = torch.nn.functional.pad(x, (0, nw - w, 0, nh - h), mode='constant', value=0)
    return x, (h, w)


def _crop_to_hw(x, hw):
    h, w = hw
    return x[..., :h, :w]


def _forward_safe(model, x, use_amp, device, output_stride=None):
    """
    Патч 3: убран nan_to_num(nan=0.0) — он маскировал проблему,
    превращая nan-логиты в 0 → softmax([0,0]) = [0.5,0.5] → argmax = «всё фон».
    Теперь nan передаётся наружу, его ловит isfinite(loss) guard в цикле.
    """
    if output_stride is None:
        output_stride = getattr(model.encoder, 'output_stride', 16)
    try:
        output_stride = int(output_stride)
    except (TypeError, ValueError):
        output_stride = 16

    padded, hw = _pad_to_multiple(x, output_stride)

    if use_amp and device.type == 'cuda':
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(padded)
    else:
        out = model(padded)

    out = _crop_to_hw(out, hw)
    return out.float()


def predict_batch_multiscale(model, images, device, scales, use_amp):
    H, W = images.shape[2], images.shape[3]
    if not scales or len(scales) == 1:
        return _forward_safe(model, images, use_amp, device)

    avg = None
    for s in scales:
        if abs(s - 1.0) < 1e-6:
            scaled = images
        else:
            nh = max(16, int(round(H * s / 16)) * 16)
            nw = max(16, int(round(W * s / 16)) * 16)
            scaled = torch.nn.functional.interpolate(
                images, size=(nh, nw), mode='bilinear', align_corners=False)
        out = _forward_safe(model, scaled, use_amp, device)
        if out.shape[2:] != (H, W):
            out = torch.nn.functional.interpolate(
                out, size=(H, W), mode='bilinear', align_corners=False)
        avg = out if avg is None else avg + out
    return (avg / len(scales)).float()


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
                outputs = _forward_safe(model, images, use_amp, device)
            if num_classes == 1:
                preds = (torch.sigmoid(outputs.float()) > 0.5).float()
            else:
                preds = (torch.argmax(torch.softmax(outputs.float(), dim=1), dim=1)
                         if output_activation == 'softmax'
                         else torch.argmax(outputs.float(), dim=1)).float()
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
    unfreeze_last_n_stages = params.get('unfreeze_last_n_stages', None)
    freeze_bn_second_phase = bool(params.get('freeze_bn_second_phase', True))
    augmentations = params.get('augmentations', {})
    device_pref = params.get('device', 'auto')
    threshold = params.get('threshold', 0.5)
    num_workers = params.get('num_workers', NUM_WORKERS_DEFAULT)
    input_size = params.get('input_size', 512)
    normalize = bool(params.get('normalize', False))
    resize_mode = params.get('resize_mode', 'hard')

    architecture = params.get('architecture', 'deeplabv3plus').lower()
    if architecture in ('deeplabv3plus', 'deeplabv3+'):
        architecture = 'deeplabv3plus'
    elif architecture == 'deeplabv3':
        architecture = 'deeplabv3'
    else:
        log(f"⚠️ Неизвестная архитектура '{architecture}', использую deeplabv3plus")
        architecture = 'deeplabv3plus'

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

    # Патч 7: предупреждение про мало эпох
    if two_phase and epochs < 30:
        log(f"⚠️ Эпох {epochs} при early_stopping_patience={EARLY_STOPPING_PATIENCE}. "
            f"Модель может не успеть сойтись.")

    atrous_rates = params.get('atrous_rates')
    if atrous_rates is None:
        atrous_rates = ATROUS_RATES_MAP.get(output_stride, (6, 12, 18))
        log(f"atrous_rates не заданы явно, выбираю автоматически для OS={output_stride}: {atrous_rates}")
    else:
        atrous_rates = tuple(int(r) for r in atrous_rates)
        log(f"atrous_rates заданы явно: {atrous_rates}")

    expected_k = (input_size - 1) / output_stride if output_stride > 0 else None
    if expected_k is None or abs(expected_k - round(expected_k)) > 1e-6:
        log(f"⚠️ input_size={input_size} не соответствует формуле crop_size = output_stride * k + 1 "
            f"для output_stride={output_stride}. Рекомендуется 513 (OS=16) или 1025 (OS=8/16).")
    else:
        log(f"✅ input_size={input_size} соответствует формуле crop_size = {output_stride}*{int(round(expected_k))}+1")
        log(f"ℹ️ smp.DeepLabV3/V3+ требует H,W % {output_stride} == 0 — "
            f"к модели будет применён авто-паддинг до {((input_size + output_stride - 1) // output_stride) * output_stride}")

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

    log("=== Запуск обучения DeepLabV3 / DeepLabV3+ ===")
    log(f"Архитектура: {architecture}")
    log(f"Датасет (папка): {dataset_path}")
    log(f"Датасет (YAML):  {params.get('deeplab_yaml')}")
    log(f"Backbone: {backbone}, output_stride={output_stride}")
    log(f"Atrous rates: {atrous_rates}")
    log(f"Выходная активация: {output_activation}")
    log(f"Классов: {num_classes}, ignore_index={ignore_index}")
    log(f"Веса классов: {class_weights}")
    log(f"Loss: {loss_name}, веса: {loss_weights}")
    log(f"Оптимизатор: {optimizer_name}, LR={lr}, momentum={momentum}, WD={weight_decay}")
    log(f"Scheduler: {scheduler_name} (power={poly_power})")
    log(f"Эпохи: {epochs}, two_phase={two_phase} (freeze={freeze_epochs}, unfreeze={unfreeze_epochs})")
    if unfreeze_last_n_stages is not None:
        log(f"Размораживать только последние {unfreeze_last_n_stages} стадий энкодера")
    log(f"Заморозка BatchNorm на 2-й фазе: {freeze_bn_second_phase}")
    log(f"Batch size: {batch_size}, device: {device_pref}")
    log(f"Input size: {input_size}, normalize={normalize}, resize_mode={resize_mode}")
    log(f"AMP: {use_amp}, Accum: {accum_steps}")
    log(f"EMA: {use_ema} (decay={ema_decay})")
    log(f"Layer-wise LR: {use_layerwise_lr} (decay={layerwise_decay})")
    log(f"Multi-scale infer: {multi_scale_infer} (scales={infer_scales})")
    log(f"Аугментации: {augmentations}")
    log(f"Результаты: {save_root}")

    deeplab_yaml = params.get('deeplab_yaml')

    if deeplab_yaml:
        if not os.path.isfile(deeplab_yaml):
            log(f"Ошибка: DeepLab YAML не найден: {deeplab_yaml}")
            sys.exit(1)
        log(f"📄 DeepLab YAML задан: {deeplab_yaml}")
        log("📄 path, num_classes и пути сплитов будут прочитаны из YAML "
            "(приоритет над папкой датасета).")
        try:
            train_loader, val_loader, train_len, val_len, num_classes_yaml = \
                create_loaders_from_yaml(
                    deeplab_yaml,
                    default_input_size=input_size,
                    default_num_classes=num_classes,
                    augmentations=augmentations,
                    ignore_index=ignore_index,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    normalize=normalize,
                    resize_mode=resize_mode,
                )
            if num_classes_yaml != num_classes:
                log(f"📄 num_classes переопределён из YAML: {num_classes} → {num_classes_yaml}")
                num_classes = num_classes_yaml
        except Exception as e:
            log(f"Ошибка создания датасета из YAML: {e}")
            sys.exit(1)
    else:
        if not dataset_path or not os.path.exists(dataset_path):
            log(f"Ошибка: папка датасета не существует: {dataset_path}")
            sys.exit(1)
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
            log(f"Ошибка создания датасета: {e}")
            sys.exit(1)

    log(f"Загружено {train_len} train, {val_len} val")

    # Диагностика масок (первые 3)
    try:
        _pairs = train_loader.dataset.pairs[:3] if hasattr(train_loader.dataset, 'pairs') else []
        for _ip, _mp in _pairs:
            _m = np.array(Image.open(_mp))
            _u = np.unique(_m)
            log(f"  mask {os.path.basename(_mp)}: dtype={_m.dtype}, "
                f"shape={_m.shape}, unique={_u[:10].tolist()}, max={int(_m.max())}")
    except Exception as _e:
        log(f"  ⚠️ Диагностика масок: {_e}")

    # Патч 6: статистика class balance
    try:
        fg_pixels = 0
        total_pixels = 0
        for _ip, _mp in train_loader.dataset.pairs:
            _m = np.array(Image.open(_mp))
            if num_classes == 2 and _m.max() == 255:
                _m = (_m > 127).astype(np.int64)
            fg_pixels += int((_m > 0).sum())
            total_pixels += int(_m.size)
        fg_ratio = fg_pixels / max(total_pixels, 1)
        log(f"📊 Class balance (train): foreground={fg_ratio*100:.2f}%, "
            f"background={100 - fg_ratio*100:.2f}%")
        if fg_ratio < 0.05:
            log("⚠️ Сильный дисбаланс классов (< 5% foreground). "
                "Рекомендуется class_weights='auto' или loss='CrossEntropy+Dice'.")
    except Exception as _e:
        log(f"⚠️ Диагностика class balance: {_e}")

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
            decoder_atrous_rates=atrous_rates,
            decoder_use_batchnorm=True,
        )
    else:
        log("Инициализация: DeepLabV3 (без плюса)")
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights='imagenet' if load_pretrained else None,
            in_channels=3, classes=num_classes,
            encoder_output_stride=output_stride,
            decoder_atrous_rates=atrous_rates,
        )

    if model_path is not None and os.path.exists(model_path):
        sd = torch.load(model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing: log(f"⚠️ Отсутствуют ключи: {missing[:5]}...")
        if unexpected: log(f"⚠️ Лишние ключи: {unexpected[:5]}...")
        log(f"Загружены веса: {model_path}")

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

    # Патч 1: device передаётся в criterion
    criterion = get_loss_function(loss_name, num_classes, ignore_index,
                                  class_weights, loss_weights, device=device)

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
                        if hasattr(stage, 'parameters'):
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
    # LR ×0.1 на первой фазе (frozen backbone)
    if two_phase and freeze_epochs > 0:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.1
        log(f"ℹ️ Первая фаза (frozen backbone): LR уменьшен в 10×")
    scheduler = make_scheduler(optimizer, current_phase_epochs)

    # Патч 5: EMA decay адаптивно под размер датасета
    total_iters = epochs * len(train_loader)
    if use_ema and total_iters < 500 and ema_decay >= 0.99:
        ema_decay_eff = max(0.9, 1.0 - 10.0 / max(total_iters, 1))
        log(f"ℹ️ EMA decay адаптирован под {total_iters} итераций: "
            f"{ema_decay} → {ema_decay_eff:.4f}")
        ema_decay = ema_decay_eff
    ema = ModelEMA(model, decay=ema_decay, device=device) if use_ema else None
    if use_ema:
        log(f"EMA инициализирован (decay={ema_decay})")

    # Патч 2: GradScaler только для fp16
    try:
        bf16_ok = (device.type == 'cuda') and torch.cuda.is_bf16_supported()
    except Exception:
        bf16_ok = False
    use_fp16_amp = bool(use_amp and device.type == 'cuda' and not bf16_ok)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16_amp)
    log(f"AMP: bf16={bf16_ok}, fp16={use_fp16_amp}, scaler_enabled={use_fp16_amp}")

    best_loss_model_path = os.path.join(weights_dir, 'best_loss.pth')
    best_iou_model_path = os.path.join(weights_dir, 'best_iou.pth')
    final_model_path = os.path.join(weights_dir, 'last.pth')

    args_dict = {
        'architecture': architecture, 'backbone': backbone,
        'output_stride': output_stride, 'atrous_rates': list(atrous_rates),
        'output_activation': output_activation,
        'num_classes': num_classes, 'ignore_index': ignore_index,
        'class_weights': class_weights, 'loss': loss_name, 'loss_weights': loss_weights,
        'optimizer': optimizer_name, 'lr': lr, 'momentum': momentum,
        'weight_decay': weight_decay, 'scheduler': scheduler_name,
        'epochs': epochs, 'batch_size': batch_size,
        'two_phase_training': two_phase, 'freeze_epochs': freeze_epochs,
        'unfreeze_epochs': unfreeze_epochs, 'unfreeze_last_n_stages': unfreeze_last_n_stages,
        'freeze_bn_second_phase': freeze_bn_second_phase,
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
        'dataset_path': dataset_path,
        'deeplab_yaml': deeplab_yaml,
    }
    save_args_yaml(args_dict, os.path.join(save_root, 'args.yaml'))

    meta_info = {
        'architecture': architecture, 'backbone': backbone,
        'output_stride': output_stride, 'atrous_rates': list(atrous_rates),
        'output_activation': output_activation,
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
        'dataset_path': dataset_path,
        'deeplab_yaml': deeplab_yaml,
    }
    save_model_meta(meta_info, best_loss_model_path)
    save_model_meta(meta_info, final_model_path)
    log("Метаданные сохранены (начальная версия).")

    patience_counter = 0
    best_val_loss = float('inf')
    best_iou = -float('inf')
    log("=== Начало обучения ===")

    for epoch in range(1, epochs + 1):
        current_epoch = epoch
        if stop_requested:
            log("Остановка по запросу пользователя."); break

        if two_phase and epoch == freeze_epochs + 1:
            log("=" * 60)
            log(f"Переключение во вторую фазу (эпоха {epoch}).")
            unfreeze_backbone()

            if freeze_bn_second_phase:
                set_bn_eval(model.encoder)
                log("BatchNorm энкодера переведён в eval() (running-статистики заморожены).")

            if use_layerwise_lr:
                optimizer = get_layerwise_optimizer(
                    model, optimizer_name, lr, momentum, weight_decay, layerwise_decay)
            else:
                optimizer = get_optimizer(model, optimizer_name, lr, momentum, weight_decay)

            current_phase_epochs = epochs - freeze_epochs
            scheduler = make_scheduler(optimizer, current_phase_epochs)

            if use_ema:
                ema = ModelEMA(model, decay=ema_decay, device=device)
                log(f"EMA сброшен (decay={ema_decay})")

            log("=" * 60)

        model.train()
        # BN энкодера в eval, пока backbone frozen (1-я фаза)
        if two_phase and epoch <= freeze_epochs:
            set_bn_eval(model.encoder)
        # И на 2-й фазе, если пользователь выбрал freeze_bn_second_phase
        elif freeze_bn_second_phase and two_phase and epoch > freeze_epochs:
            set_bn_eval(model.encoder)

        train_loss = 0.0
        optimizer.zero_grad()

        for b_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            target = masks.unsqueeze(1).float() if num_classes == 1 else masks.long()

            outputs = _forward_safe(model, images, use_amp, device)
            loss = criterion(outputs, target) / accum_steps

            if not torch.isfinite(loss):
                log(f"⚠️ Skipping non-finite loss (batch {b_idx}): {loss.item()}")
                optimizer.zero_grad()
                continue

            # Патч 2: fp16 → scaler, bf16/no-amp → обычный backward
            if use_fp16_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (b_idx + 1) % accum_steps == 0:
                if use_fp16_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
                if isinstance(scheduler, PolyLRSchedulerIter):
                    scheduler.step()

            train_loss += loss.item() * accum_steps * images.size(0)

        if len(train_loader) % accum_steps != 0:
            if use_fp16_amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            if isinstance(scheduler, PolyLRSchedulerIter):
                scheduler.step()

        train_loss /= train_len

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
                        outputs = _forward_safe(eval_model, images, use_amp, device)

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

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        for k in ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'map50', 'map50_95']:
            if metrics:
                key_map = {'iou': 'IoU', 'dice': 'Dice', 'f1': 'F1',
                           'precision': 'Precision', 'recall': 'Recall',
                           'accuracy': 'Accuracy', 'map50': 'mAP50', 'map50_95': 'mAP50-95'}
                history[k].append(metrics[key_map[k]])
            else:
                history[k].append(float('nan'))

        if scheduler is not None and not isinstance(scheduler, PolyLRSchedulerIter):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler.step()

        pct = int(epoch / epochs * 100)
        log(f"PROGRESS:{pct}:{epoch}:{epochs}")
        cur_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        lr_str = f"LR={cur_lrs[0]}" if len(cur_lrs) == 1 else f"LR=[{','.join(cur_lrs[:3])}...]"
        if val_loss is not None:
            log(f"Эпоха {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"IoU: {metrics['IoU']:.4f} | Dice: {metrics['Dice']:.4f} | F1: {metrics['F1']:.4f} | {lr_str}")
        else:
            log(f"Эпоха {epoch}/{epochs} | Train: {train_loss:.4f} | (нет val) | {lr_str}")

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
                    outputs = _forward_safe(model, images, use_amp, device)
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

    if os.path.exists(best_loss_model_path):
        log(f"best_loss: {best_loss_model_path}")
    if os.path.exists(best_iou_model_path):
        log(f"best_iou:  {best_iou_model_path}")
    log(f"last:      {final_model_path}")


    # =========================================================================
    # ЭКСПОРТ В ONNX
    # =========================================================================
    log("=== Экспорт финальной модели в ONNX ===")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        log("⚠️ Пропуск экспорта в ONNX: не установлены 'onnx' или 'onnxruntime'.")
        log("   Установите: pip install onnx onnxruntime")
    else:
        onnx_path = os.path.join(weights_dir, 'model.onnx')
        onnx_meta_path = os.path.join(weights_dir, 'model_onnx_meta.json')

        # Загружаем лучшие веса в модель (лучше всего экспортировать best_iou или last)
        best_weights = best_iou_model_path if os.path.exists(best_iou_model_path) else final_model_path
        log(f"Загрузка весов для экспорта: {os.path.basename(best_weights)}")
        model.load_state_dict(torch.load(best_weights, map_location='cpu'))
        model.eval()
        model.to('cpu')  # Экспорт стабильнее на CPU

        # Создаем dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size, device='cpu')

        # Динамические оси: разрешаем менять batch size и H/W
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }

        log(f"Запуск torch.onnx.export (opset=17, dynamic_axes=True)...")
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
            log(f"✅ ONNX-модель сохранена: {onnx_path}")

            # Проверка валидности ONNX-графа
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            log("✅ ONNX-граф прошел проверку checker.")

            # Быстрый прогон через ONNX Runtime для проверки
            ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            log(f"✅ Тестовый инференс ONNX Runtime успешен. Выход: shape={ort_outs[0].shape}")

            # Метаданные для ONNX-модели
            onnx_meta = {
                'format': 'ONNX',
                'source_weights': os.path.basename(best_weights),
                'onnx_opset': 18,
                'architecture': architecture,
                'backbone': backbone, # ИСПРАВЛЕНО: здесь был encoder
                'output_stride': output_stride,
                'num_classes': num_classes,
                'input_size': input_size,
                'input_names': ['input'],
                'output_names': ['output'],
                'dynamic_axes': dynamic_axes,
                'normalize': normalize,
                'resize_mode': resize_mode,
                'threshold': threshold,
                'export_date': datetime.now().isoformat(),
                'description': 'Модель DeepLabV3/V3+, экспортированная из PyTorch. '
                               'Вход: [B, 3, H, W] RGB-изображение (нормализованное при необходимости). '
                               'Выход: логиты [B, num_classes, H, W]. '
                               'Для получения маски примените argmax (multiclass) или sigmoid+threshold (binary).'
            }
            with open(onnx_meta_path, 'w', encoding='utf-8') as f:
                json.dump(onnx_meta, f, indent=2, ensure_ascii=False)
            log(f"✅ Метаданные ONNX сохранены: {onnx_meta_path}")

        except Exception as e:
            log(f"❌ Ошибка при экспорте в ONNX: {type(e).__name__}: {e}")
            import traceback
            log(traceback.format_exc())
    # =========================================================================

    log("=== Обучение завершено ===")





if __name__ == "__main__":
    main()