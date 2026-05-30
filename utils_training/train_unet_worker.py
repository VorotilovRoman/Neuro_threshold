#!/usr/bin/env python
# train_unet_worker.py
# Поддерживается бинарная (num_classes=1) и многоклассовая (num_classes>1) сегментация
# Сохранение результатов организовано по аналогии с YOLO

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

# ========== Глобальные конфигурации (можно менять вручную) ==========
DECODER_CHANNELS = (256, 128, 64, 32, 16)   # каналы декодера U-Net
NUM_WORKERS_DEFAULT = 4                     # количество workers для DataLoader
AUTO_BATCH_MEM_FACTOR = 0.7                 # использовать 70% доступной памяти
AUTO_BATCH_MIN = 1                          # минимальный batch
AUTO_BATCH_MAX = 64                         # максимальный batch
NUM_TRAIN_BATCHES_TO_SAVE = 3               # сколько тренировочных батчей сохранить как примеры
NUM_VAL_SAMPLES_TO_SAVE = 8                 # сколько валидационных изображений для предсказаний
EARLY_STOPPING_PATIENCE = 3                 # количество эпох без улучшения val_loss
EARLY_STOPPING_TOLERANCE = 0.05             # допустимый относительный рост val_loss (5%)
OVERFITTING_GAP_THRESHOLD = 0.2             # допустимый разрыв val_loss - train_loss (20% от train_loss)
THRESHOLD_FACTOR = 0.7                      # порог бинаризации для метрик

# Глобальные переменные для graceful shutdown
model = None
optimizer = None
scheduler = None
best_model_path = None
final_model_path = None
current_epoch = 0
best_val_loss = float('inf')
stop_requested = False                       # флаг ручной остановки

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
    # Сохраняем текущую модель, если нужно
    if model is not None and final_model_path is not None:
        checkpoint_path = final_model_path.replace(".pth", f"_checkpoint_epoch_{current_epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        log(f"Модель сохранена как {checkpoint_path}")
        if best_model_path and os.path.exists(best_model_path):
            backup_best = best_model_path.replace(".pth", f"_interrupted.pth")
            torch.save(torch.load(best_model_path), backup_best)
            log(f"Лучшая модель скопирована в {backup_best}")
    # Не вызываем sys.exit(0) – дадим циклу завершиться корректно

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- Dataset (поддержка бинарной и многоклассовой сегментации) ----------
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, input_size=512, num_classes=1, augmentations=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.input_size = input_size
        self.num_classes = num_classes
        self.augmentations = augmentations or {}
        self.image_paths = []
        self.mask_paths = []
        skipped = 0
        for f in os.listdir(images_dir):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            base = os.path.splitext(f)[0]
            mask_path = os.path.join(masks_dir, base + '.png')
            if not os.path.exists(mask_path):
                log(f"⚠️ Пропуск: для {f} не найдена маска {mask_path}")
                skipped += 1
                continue
            self.image_paths.append(os.path.join(images_dir, f))
            self.mask_paths.append(mask_path)
        if not self.image_paths:
            raise RuntimeError(f"Не найдено ни одной пары изображение-маска в {images_dir} и {masks_dir}")
        log(f"✅ Загружено {len(self.image_paths)} пар (пропущено {skipped})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        transform_resize = transforms.Resize((self.input_size, self.input_size))
        image = transform_resize(image)
        mask = transform_resize(mask)

        # Аугментации
        if self.augmentations.get('hflip', False) and np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.augmentations.get('vflip', False) and np.random.rand() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if self.augmentations.get('rotate', False):
            angle = np.random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        if self.augmentations.get('scale', False):
            scale = np.random.uniform(0.9, 1.1)
            new_size = (int(self.input_size * scale), int(self.input_size * scale))
            image = transforms.Resize(new_size)(image)
            mask = transforms.Resize(new_size)(mask)
            image = transforms.CenterCrop(self.input_size)(image)
            mask = transforms.CenterCrop(self.input_size)(mask)
        if self.augmentations.get('noise', False):
            img_np = np.array(image) / 255.0
            noise = np.random.normal(0, 0.05, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        image = transforms.ToTensor()(image)
        # Преобразование маски в тензор
        mask_np = np.array(mask)
        if self.num_classes == 1:
            # Бинарная: преобразуем в float и порог
            mask_tensor = torch.from_numpy(mask_np).float() / 255.0
            mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0)  # [1, H, W]
        else:
            # Многоклассовая: целочисленные метки от 0 до C-1
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]   # берём первый канал
            mask_tensor = torch.from_numpy(mask_np).long()
            if mask_tensor.max() >= self.num_classes:
                mask_tensor = (mask_tensor > 0).long()
                if self.num_classes > 2:
                    log("⚠️ Внимание: маска содержит значения >1, преобразована в бинарную.")
        return image, mask_tensor

# ---------- Функции потерь (поддержка бинарной и многоклассовой) ----------
def get_loss_function(loss_name, num_classes=1):
    loss_name = loss_name.lower()
    if num_classes > 1:
        # Многоклассовый режим
        if loss_name == "ce" or loss_name == "crossentropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "dice":
            return smp.losses.DiceLoss(mode='multiclass', from_logits=True, classes=num_classes)
        elif loss_name == "focal":
            return smp.losses.FocalLoss(mode='multiclass', alpha=0.25, gamma=2, from_logits=True)
        elif loss_name == "tversky":
            return smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, from_logits=True)
        elif loss_name == "bce+dice":
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
        elif loss_name == "bce+dice":
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
    """
    pred_logits: тензор [B, C, H, W] (логиты)
    true_masks: тензор [B, 1, H, W] для бинарного; [B, H, W] для мультикласса
    """
    if num_classes == 1:
        # Бинарный случай
        pred_probs = torch.sigmoid(pred_logits)          # [B, 1, H, W]
        pred = (pred_probs > threshold).float()          # [B, 1, H, W]
        # Приводим true_masks к [B, 1, H, W] если нужно
        if true_masks.dim() == 3:
            true = true_masks.unsqueeze(1).float()
        else:
            true = true_masks.float()
        # Суммируем по H и W (размерности 2 и 3)
        intersection = (pred * true).sum(dim=(2,3))
        pred_sum = pred.sum(dim=(2,3))
        true_sum = true.sum(dim=(2,3))
        union = pred_sum + true_sum - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2 * intersection + 1e-6) / (pred_sum + true_sum + 1e-6)
        tp = intersection
        fp = pred_sum - intersection
        fn = true_sum - intersection
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        # Accuracy (попиксельная)
        correct = (pred == true).float().sum(dim=(2,3))
        total = pred.shape[2] * pred.shape[3]
        accuracy = correct / total
        map50 = (iou > 0.5).float().mean().item()
        map50_95 = iou.mean().item()
        log(f"[DEBUG] intersection.mean()={intersection.mean().item():.4f}, union.mean()={union.mean().item():.4f}, pred_sum.mean()={pred_sum.mean().item():.4f}, true_sum.mean()={true_sum.mean().item():.4f}, threshold = {threshold}")
        return {
            'IoU': iou.mean().item(),
            'Dice': dice.mean().item(),
            'Precision': precision.mean().item(),
            'Recall': recall.mean().item(),
            'F1': f1.mean().item(),
            'Accuracy': accuracy.mean().item(),
            'mAP50': map50,
            'mAP50-95': map50_95
        }
    else:
        # Многоклассовый случай
        pred = torch.argmax(pred_logits, dim=1)   # [B, H, W]
        true = true_masks.long()                 # [B, H, W]
        iou_per_class = []
        dice_per_class = []
        precision_per_class = []
        recall_per_class = []
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
        map50 = miou
        map50_95 = miou
        return {
            'IoU': miou,
            'Dice': mdice,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
            'mAP50': map50,
            'mAP50-95': map50_95
        }

# ---------- Автоматический подбор batch_size ----------
def auto_batch_size(device, mem_factor=AUTO_BATCH_MEM_FACTOR, min_batch=AUTO_BATCH_MIN, max_batch=AUTO_BATCH_MAX):
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            free_mem_gb = free_mem / (1024**3)
            suggested = int((free_mem_gb * mem_factor) * 4)
            suggested = max(min_batch, min(suggested, max_batch))
            log(f"Auto batch (GPU): free={free_mem_gb:.1f}GB -> batch={suggested}")
            return suggested
        except Exception as e:
            log(f"Ошибка при определении памяти GPU: {e}, переключаемся на RAM")
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / (1024**3)
        suggested = int((free_ram * mem_factor) * 2)
        suggested = max(min_batch, min(suggested, max_batch))
        log(f"Auto batch (RAM): free={free_ram:.1f}GB -> batch={suggested}")
        return suggested
    except ImportError:
        log("⚠️ psutil не установлен, auto batch использует значение по умолчанию 8")
        return 8

# ---------- Сохранение аргументов в YAML / JSON ----------
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

# ---------- Сохранение метаданных модели рядом с весами ----------
def save_model_meta(meta_dict, model_path):
    meta_path = os.path.splitext(model_path)[0] + '_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    log(f"Метаданные модели сохранены в {meta_path}")

# ---------- Построение и сохранение графиков обучения ----------
def save_training_plots(history_dict, save_dir):
    if not HAS_MATPLOTLIB:
        return
    epochs = history_dict['epoch']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Loss
    axes[0,0].plot(epochs, history_dict['train_loss'], label='Train Loss')
    if history_dict['val_loss']:
        axes[0,0].plot(epochs, history_dict['val_loss'], label='Val Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].set_title('Loss')
    # IoU / Dice
    axes[0,1].plot(epochs, history_dict['iou'], label='IoU')
    axes[0,1].plot(epochs, history_dict['dice'], label='Dice')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Score')
    axes[0,1].legend()
    axes[0,1].set_title('IoU & Dice')
    # Precision/Recall
    axes[1,0].plot(epochs, history_dict['precision'], label='Precision')
    axes[1,0].plot(epochs, history_dict['recall'], label='Recall')
    axes[1,0].plot(epochs, history_dict['f1'], label='F1')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].legend()
    axes[1,0].set_title('Precision/Recall/F1')
    # mAP
    axes[1,1].plot(epochs, history_dict['map50'], label='mAP50')
    axes[1,1].plot(epochs, history_dict['map50_95'], label='mAP50-95')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].legend()
    axes[1,1].set_title('mAP')
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

# ---------- Построение precision-recall кривой по порогу (только для бинарного) ----------
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
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
        # PR-кривая
        plt.figure(figsize=(8,6))
        plt.plot(recalls, precisions, marker='.', label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskPR_curve.png'))
        plt.close()
        # P, R, F1 по порогу
        plt.figure(figsize=(10,6))
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, F1 vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskF1_curve.png'))
        plt.close()
        # Отдельно P и R кривые
        plt.figure(figsize=(8,6))
        plt.plot(thresholds, precisions, label='Precision')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskP_curve.png'))
        plt.close()
        plt.figure(figsize=(8,6))
        plt.plot(thresholds, recalls, label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'MaskR_curve.png'))
        plt.close()
        log(f"Кривые PR, F1, P, R сохранены в {save_dir}")
    else:
        log("PR-кривые не генерируются для многоклассовой сегментации")

# ---------- Построение confusion matrix ----------
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
                conf_matrix[t,p] = ((true == t) & (pred == p)).sum().item()
        conf_matrix = conf_matrix.numpy()
        tick_labels = [f'Class {i}' for i in range(num_classes)]

    plt.figure(figsize=(max(6, num_classes*0.6), max(5, num_classes*0.5)))
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
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    conf_norm = conf_matrix.astype(np.float32) / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)
    plt.figure(figsize=(max(6, num_classes*0.6), max(5, num_classes*0.5)))
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
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'))
    plt.close()
    log(f"Матрицы ошибок сохранены в {save_dir}")

# ---------- Сохранение примеров тренировочных батчей (один файл на батч) ----------
def save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        n_images = min(images.size(0), samples_per_batch)
        images = images[:n_images]
        masks = masks[:n_images]
        # Создаём сетку 2 строки (изображения и маски) × n_images столбцов
        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 8))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_images):
            img = images[i].cpu().permute(1,2,0).numpy() * 255
            img = img.astype(np.uint8)
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Image {i+1}')
            # Маска: убираем лишнюю ось канала
            mask = masks[i].cpu().numpy().squeeze()  # -> (H,W) или (H,W,3) для мультикласса
            if num_classes == 1:
                mask_vis = (mask > 0.5).astype(np.uint8) * 255
                axes[1, i].imshow(mask_vis, cmap='gray')
            else:
                mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for c in range(num_classes):
                    mask_vis[mask == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
                axes[1, i].imshow(mask_vis)
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Mask {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'train_batch{batch_idx}.jpg'))
        plt.close()
    log(f"Сохранено {num_batches} тренировочных батчей (показано до {samples_per_batch} изображений на батч) в {save_dir}")

# ---------- Сохранение примеров валидации (один файл-коллаж) ----------
def save_val_examples(val_loader, save_dir, num_samples=8, device='cpu', num_classes=1):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    images_list = []
    true_masks_list = []
    pred_masks_list = []
    collected = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            else:
                preds = torch.argmax(outputs, dim=1).float()
            for i in range(images.size(0)):
                if collected >= num_samples:
                    break
                img = images[i].cpu().permute(1,2,0).numpy() * 255
                img = img.astype(np.uint8)
                true_mask = masks[i].cpu().numpy().squeeze()   # убираем ось канала
                pred_mask = preds[i].cpu().numpy().squeeze()   # убираем ось канала
                images_list.append(img)
                true_masks_list.append(true_mask)
                pred_masks_list.append(pred_mask)
                collected += 1
            if collected >= num_samples:
                break

    if not images_list:
        log("Нет примеров для сохранения валидации")
        return

    # Создаём сетку: n строк, 3 столбца
    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i in range(n):
        # Изображение
        axes[i, 0].imshow(images_list[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_title('Image')
        # Истинная маска
        if num_classes == 1:
            true_viz = (true_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i, 1].imshow(true_viz, cmap='gray')
        else:
            true_viz = np.zeros((true_masks_list[i].shape[0], true_masks_list[i].shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                true_viz[true_masks_list[i] == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
            axes[i, 1].imshow(true_viz)
        axes[i, 1].axis('off')
        axes[i, 1].set_title('True Mask')
        # Предсказанная маска
        if num_classes == 1:
            pred_viz = (pred_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i, 2].imshow(pred_viz, cmap='gray')
        else:
            pred_viz = np.zeros((pred_masks_list[i].shape[0], pred_masks_list[i].shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                pred_viz[pred_masks_list[i] == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
            axes[i, 2].imshow(pred_viz)
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Pred Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'val_examples.jpg'))
    plt.close()
    log(f"Сохранено {collected} валидационных примеров в {save_dir}/val_examples.jpg")

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

    # Извлечение параметров
    dataset_path = params.get('dataset_path')
    model_source = params.get('model_path')
    project_path = params.get('project_path')
    name = params.get('name')
    encoder = params.get('encoder', 'resnet50')
    input_size = params.get('input_size', 512)
    num_classes = params.get('num_classes', 1)
    optimizer_name = params.get('optimizer', 'Adam')
    loss_name = params.get('loss', 'BCE')
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

    log("=== Запуск обучения U‑Net ===")
    log(f"Датасет: {dataset_path}")
    log(f"Модель: {model_source if model_source else 'предустановленная'}")
    log(f"Энкодер: {encoder}")
    log(f"Размер входа: {input_size}")
    log(f"Классов: {num_classes}")
    log(f"Оптимизатор: {optimizer_name}, LR={lr}")
    log(f"Loss: {loss_name}")
    log(f"Batch size: {batch_size}, Epochs: {epochs}")
    log(f"Scheduler: {scheduler_name}, Dropout: {dropout}")
    log(f"Аугментации: {augmentations}")
    log(f"Устройство: {device_pref}")
    log(f"Num workers: {num_workers}")
    log(f"Порог (threshold): {threshold}")
    log(f"Результаты будут сохранены в: {save_root}")

    # Проверка датасета
    if not dataset_path or not os.path.exists(dataset_path):
        log(f"Ошибка: папка датасета не существует: {dataset_path}")
        sys.exit(1)
    train_img_dir = os.path.join(dataset_path, 'images', 'train')
    train_mask_dir = os.path.join(dataset_path, 'masks', 'train')
    val_img_dir = os.path.join(dataset_path, 'images', 'val')
    val_mask_dir = os.path.join(dataset_path, 'masks', 'val')

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
        batch_size = auto_batch_size(temp_device)
        log(f"Auto batch: выбран batch_size = {batch_size}")
    elif batch_size <= 0:
        log("Ошибка: batch_size должен быть положительным числом")
        sys.exit(1)

    # Загрузка метаданных пользовательской модели
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
                log(f"Архитектура восстановлена: encoder={encoder}, decoder_channels={decoder_channels}, dropout={dropout}, input_size={input_size}, num_classes={num_classes}")
            except Exception as e:
                log(f"Предупреждение: не удалось загрузить метафайл {meta_path}: {e}. Используются параметры из JSON.")
        else:
            log(f"⚠️ Метаданные модели не найдены: {meta_path}. Используются параметры из JSON. Возможна несовместимость архитектуры.")
    elif model_source is not None and not os.path.exists(model_source):
        log(f"Ошибка: файл модели {model_source} не найден")
        sys.exit(1)

    # Датасеты
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, input_size, num_classes, augmentations)
    if val_exists:
        val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, input_size, num_classes, {})
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        log(f"Загружено {len(train_dataset)} обучающих, {len(val_dataset)} валидационных образцов")
    else:
        log("⚠️ Валидационные папки не найдены, валидация будет пропущена")
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Модель
    global model, best_model_path, final_model_path
    load_pretrained = (model_source is None) or (not os.path.exists(model_source))
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
            log(f"⚠️ Отсутствуют ключи: {missing[:5]}..." if len(missing)>5 else f"⚠️ Отсутствуют ключи: {missing}")
        if unexpected:
            log(f"⚠️ Лишние ключи: {unexpected[:5]}..." if len(unexpected)>5 else f"⚠️ Лишние ключи: {unexpected}")
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

    best_model_path = os.path.join(weights_dir, 'best.pth')
    final_model_path = os.path.join(weights_dir, 'last.pth')

    # Сохраняем аргументы обучения в YAML
    args_dict = {
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
    }
    save_args_yaml(args_dict, os.path.join(save_root, 'args.yaml'))

    # Сохраняем начальные метаданные
    meta_info_initial = {
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
        'date': datetime.now().isoformat()
    }
    save_model_meta(meta_info_initial, best_model_path)
    save_model_meta(meta_info_initial, final_model_path)
    log("Метаданные модели сохранены (начальная версия).")

    # Early stopping счётчики
    patience_counter = 0
    best_val_loss = float('inf')

    # Тренировочный цикл
    global current_epoch, stop_requested
    log("=== Начало обучения ===")

    for epoch in range(1, epochs + 1):
        current_epoch = epoch

        # Проверка ручной остановки
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
            if num_classes == 1:
                # masks: [B, 1, H, W]
                loss = criterion(outputs, masks)
            else:
                # masks: [B, H, W]
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
                    if num_classes == 1:
                        loss = criterion(outputs, masks)
                        preds_sigmoid = torch.sigmoid(outputs)
                        all_preds.append(preds_sigmoid.cpu())
                        all_masks.append(masks.cpu())   # masks already [B,1,H,W]
                    else:
                        loss = criterion(outputs, masks)
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

        # Сохранение лучшей модели и early stopping
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                log(f"✅ Новая лучшая модель сохранена (val_loss={best_val_loss:.4f})")
            else:
                # Проверка на резкий рост val_loss (переобучение)
                rel_increase = (val_loss - best_val_loss) / (best_val_loss + 1e-8)
                if rel_increase > EARLY_STOPPING_TOLERANCE:
                    log(f"⚠️ Переобучение: val_loss вырос на {rel_increase*100:.1f}% > {EARLY_STOPPING_TOLERANCE*100:.0f}%. Остановка обучения.")
                    break

                # Проверка на большой разрыв train-val loss
                loss_gap = (val_loss - train_loss) / (train_loss + 1e-8)
                if loss_gap > OVERFITTING_GAP_THRESHOLD and patience_counter >= 1:
                    log(f"⚠️ Переобучение: разрыв val_loss ({val_loss:.4f}) и train_loss ({train_loss:.4f}) слишком велик ({loss_gap*100:.1f}%). Остановка обучения.")
                    break

                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"⚠️ Early stopping: val_loss не улучшается {EARLY_STOPPING_PATIENCE} эпох. Остановка обучения.")
                    break
        else:
            # Если валидации нет, сохраняем текущую модель один раз
            if best_val_loss == float('inf'):
                best_val_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
                log(f"✅ Модель сохранена (нет валидации)")

        # Обновление истории
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
            for k in ['iou','dice','f1','precision','recall','accuracy','map50','map50_95']:
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

    # Сохранение финальной модели
    torch.save(model.state_dict(), final_model_path)
    log(f"Финальная модель сохранена: {final_model_path}")

    # Обновляем метаданные
    meta_info_final = {
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
        'date': datetime.now().isoformat()
    }
    save_model_meta(meta_info_final, best_model_path)
    save_model_meta(meta_info_final, final_model_path)
    log("Метаданные модели обновлены (финальная версия).")

    # Пост-обработка и визуализации
    if val_loader is not None:
        log("=== Генерация итоговых отчётов и графиков ===")
        best_state = torch.load(best_model_path, map_location='cpu')
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
        save_train_batch_examples(train_loader, save_root, num_batches=NUM_TRAIN_BATCHES_TO_SAVE,
                                  samples_per_batch=min(4, batch_size), num_classes=num_classes)
        save_val_examples(val_loader, save_root, num_samples=NUM_VAL_SAMPLES_TO_SAVE,
                          device=device, num_classes=num_classes)

        final_metrics = compute_metrics(all_preds, all_masks, num_classes, threshold=threshold)
        log(f"METRICS: {json.dumps(final_metrics)}")
    else:
        log("METRICS: null")
        log("⚠️ Валидация отсутствует – дополнительные графики и кривые не сгенерированы")
        save_csv_results(history, os.path.join(save_root, 'results.csv'))

    log("=== Обучение завершено ===")
    if best_model_path and os.path.exists(best_model_path):
        log(f"Лучшая модель сохранена: {best_model_path} (val_loss={best_val_loss:.4f})")
    log("SUCCESS")

if __name__ == "__main__":
    main()