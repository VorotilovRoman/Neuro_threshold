#!/usr/bin/env python
# train_deeplabv3_worker.py
# Обучение DeepLabV3+ с поддержкой бинарной и многоклассовой сегментации,
# двухфазного обучения, кастомных весов классов, игнорирования индекса и т.д.
# Формат датасета: PASCAL VOC (JPEGImages, SegmentationClass, ImageSets/Segmentation)
# Альтернативно: структура images/train, masks/train (как в U‑Net)
# ------------------------------------------------------------------------------

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
NUM_WORKERS_DEFAULT = 4
AUTO_BATCH_MEM_FACTOR = 0.7
AUTO_BATCH_MIN = 1
AUTO_BATCH_MAX = 64
NUM_TRAIN_BATCHES_TO_SAVE = 3
NUM_VAL_SAMPLES_TO_SAVE = 8
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_TOLERANCE = 0.05
OVERFITTING_GAP_THRESHOLD = 0.2

# Глобальные переменные для graceful shutdown
model = None
optimizer = None
scheduler = None
best_model_path = None
final_model_path = None
current_epoch = 0
best_val_loss = float('inf')
stop_requested = False

history = {
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
        if best_model_path and os.path.exists(best_model_path):
            backup_best = best_model_path.replace(".pth", f"_interrupted.pth")
            torch.save(torch.load(best_model_path), backup_best)
            log(f"Лучшая модель скопирована в {backup_best}")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- Вспомогательные функции для чтения VOC-датасета ----------
def find_image_file(jpeg_dir, basename):
    """Ищет файл изображения в JPEGImages по базовому имени с разными расширениями."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for ext in extensions:
        path = os.path.join(jpeg_dir, basename + ext)
        if os.path.exists(path):
            return path
    return None

def read_voc_file_list(txt_path, root_dir):
    """
    Читает список относительных путей (без расширения) из файла train.txt / val.txt.
    Возвращает список кортежей (путь_к_изображению, путь_к_маске).
    """
    if not os.path.exists(txt_path):
        return []
    # Пробуем utf-8, если не получилось – cp1251
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        with open(txt_path, 'r', encoding='cp1251') as f:
            lines = [line.strip() for line in f if line.strip()]

    jpeg_dir = os.path.join(root_dir, 'JPEGImages')
    seg_dir = os.path.join(root_dir, 'SegmentationClass')
    pairs = []
    for name in lines:
        img_path = find_image_file(jpeg_dir, name)
        mask_path = os.path.join(seg_dir, name + '.png')
        if img_path is not None and os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            log(f"Предупреждение: для {name} не найдено изображение или маска")
    if len(pairs) != len(lines):
        log(f"Предупреждение: из {len(lines)} пар в {txt_path} доступно только {len(pairs)}.")
    return pairs



def create_voc_loaders(dataset_root, input_size, num_classes, augmentations, ignore_index,
                       batch_size, num_workers):
    """Создаёт train_loader и val_loader для VOC-структуры."""
    seg_root = os.path.join(dataset_root, 'ImageSets', 'Segmentation')
    train_txt = os.path.join(seg_root, 'train.txt')
    val_txt = os.path.join(seg_root, 'val.txt')
    if not os.path.exists(train_txt):
        raise RuntimeError(f"Не найден {train_txt}. Убедитесь в структуре VOC.")
    train_pairs = read_voc_file_list(train_txt, dataset_root)
    val_pairs = read_voc_file_list(val_txt, dataset_root) if os.path.exists(val_txt) else []

    if not train_pairs:
        raise RuntimeError("Нет обучающих пар в VOC-датасете.")

    train_dataset = VOCSegmentationPairs(train_pairs, input_size, num_classes, ignore_index, augmentations)
    val_dataset = VOCSegmentationPairs(val_pairs, input_size, num_classes, ignore_index, {}) if val_pairs else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers) if val_dataset else None
    return train_loader, val_loader, len(train_dataset), len(val_dataset) if val_dataset else 0

# ---------- Остальные функции (потери, оптимизаторы, метрики, визуализации) без изменений ----------
# ... (они уже были в исходном файле, здесь не повторяются для краткости, но в итоговом файле остаются)


class VOCSegmentationPairs(Dataset):
    def __init__(self, image_mask_pairs, input_size, num_classes, ignore_index, augmentations):
        self.pairs = image_mask_pairs
        self.input_size = input_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augmentations = augmentations

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        # Маска VOC - палитра, нужно преобразовать в индексный тензор.
        # PIL.Image с режимом 'P' хранит индексы. Конвертируем в numpy.
        mask_np = np.array(mask)   # для VOC-масок это матрица индексов (0..N-1, 255 для игнора)
        # Если маска бинарная (0 и 255) и мы ожидаем классы 0..num_classes-1, преобразуем 255 в 1
        if self.num_classes == 2 and np.max(mask_np) == 255:
            mask_np = (mask_np > 127).astype(np.int64)
        # Изменяем размер
        transform_resize = transforms.Resize((self.input_size, self.input_size),
                                             interpolation=transforms.InterpolationMode.NEAREST)
        image = transform_resize(image)
        mask_img = Image.fromarray(mask_np.astype(np.uint8))
        mask_img = transform_resize(mask_img)
        mask_np = np.array(mask_img)

        # Аугментации (только для изображения, маску синхронно)
        if self.augmentations.get('hflip', False) and np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask_np = np.fliplr(mask_np).copy()
        if self.augmentations.get('vflip', False) and np.random.rand() < 0.5:
            image = TF.vflip(image)
            mask_np = np.flipud(mask_np).copy()
        if self.augmentations.get('rotate', False):
            angle = np.random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask_np = TF.rotate(Image.fromarray(mask_np.astype(np.uint8)), angle,
                                interpolation=TF.InterpolationMode.NEAREST)
            mask_np = np.array(mask_np).astype(np.int64)
        if self.augmentations.get('scale', False):
            scale = np.random.uniform(0.9, 1.1)
            new_size = (int(self.input_size * scale), int(self.input_size * scale))
            image = transforms.Resize(new_size)(image)
            mask_img = Image.fromarray(mask_np.astype(np.uint8))
            mask_img = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(mask_img)
            image = transforms.CenterCrop(self.input_size)(image)
            mask_img = transforms.CenterCrop(self.input_size)(mask_img)
            mask_np = np.array(mask_img)
        if self.augmentations.get('noise', False):
            img_np = np.array(image) / 255.0
            noise = np.random.normal(0, 0.05, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        # Преобразование в тензор
        image = transforms.ToTensor()(image)   # [C, H, W] в [0,1]
        # Маска: [H, W] с индексами классов, ignore_index сохраняется
        mask = torch.from_numpy(mask_np).long()
        return image, mask

# ---------- Функции потерь (с поддержкой ignore_index и взвешиванием) ----------
def get_loss_function(loss_name, num_classes, ignore_index=None, class_weights=None, loss_weights=None):
    loss_weights = loss_weights or {'ce': 1.0, 'dice': 1.0, 'focal': 0.0}

    # Веса классов
    weights_tensor = None
    if class_weights is not None and class_weights != 'auto':
        try:
            if isinstance(class_weights, str) and os.path.exists(class_weights):
                with open(class_weights, 'r') as f:
                    weights = json.load(f)
                weights_tensor = torch.tensor(weights, dtype=torch.float)
            else:
                weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        except Exception as e:
            log(f"Ошибка загрузки class_weights: {e}. Использую None.")

    mode = 'binary' if num_classes == 1 else 'multiclass'

    # CE loss
    if mode == 'binary':
        ce_loss = nn.BCEWithLogitsLoss(
            pos_weight=weights_tensor) if weights_tensor is not None else nn.BCEWithLogitsLoss()
    else:
        ign_idx = ignore_index if ignore_index is not None else -100
        ce_loss = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=ign_idx)

    # Dice и Focal (без from_logits, с правильным режимом)
    if num_classes == 1:
        dice_loss = smp.losses.DiceLoss(mode=mode, ignore_index=ignore_index)
        focal_loss = smp.losses.FocalLoss(mode=mode, alpha=0.25, gamma=2, ignore_index=ignore_index)
    else:
        dice_loss = smp.losses.DiceLoss(mode=mode, classes=num_classes, ignore_index=ignore_index)
        focal_loss = smp.losses.FocalLoss(mode=mode, alpha=0.25, gamma=2, ignore_index=ignore_index)

    # Комбинации
    loss_name_lower = loss_name.lower()
    if loss_name_lower == 'crossentropy':
        return ce_loss
    elif loss_name_lower == 'dice':
        return dice_loss
    elif loss_name_lower == 'focal':
        return focal_loss
    elif loss_name_lower == 'crossentropy+dice':
        ce_w = loss_weights.get('ce', 1.0)
        dice_w = loss_weights.get('dice', 1.0)
        return lambda pred, target: ce_w * ce_loss(pred, target) + dice_w * dice_loss(pred, target)
    elif loss_name_lower == 'crossentropy+focal':
        ce_w = loss_weights.get('ce', 1.0)
        focal_w = loss_weights.get('focal', 1.0)
        return lambda pred, target: ce_w * ce_loss(pred, target) + focal_w * focal_loss(pred, target)
    elif loss_name_lower == 'dice+focal':
        dice_w = loss_weights.get('dice', 1.0)
        focal_w = loss_weights.get('focal', 1.0)
        return lambda pred, target: dice_w * dice_loss(pred, target) + focal_w * focal_loss(pred, target)
    else:
        log(f"Неизвестная функция потерь {loss_name}, использую CrossEntropy")
        return ce_loss



# ---------- Оптимизатор и Scheduler ----------
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
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, scheduler_name, epochs, eta_min=1e-6):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'poly':
        # Полиномиальное затухание: lr = base_lr * (1 - iter/max_iter)^power
        # Реализуем через LambdaLR
        def lambda_poly(epoch):
            return (1 - epoch / max(epochs, 1)) ** 0.9
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        return None

# ---------- Метрики (с учётом ignore_index и выбранной активации) ----------
def compute_metrics(pred_logits, true_masks, num_classes, ignore_index=None, output_activation='argmax', threshold=0.5):
    """
    pred_logits: тензор [B, C, H, W] (логиты)
    true_masks: тензор [B, H, W] (индексы классов) для мультикласса, или [B,1,H,W] для бинарного?
    Функция приводится к единому формату.
    """
    if num_classes == 1:
        # Бинарный режим
        probs = torch.sigmoid(pred_logits)             # [B,1,H,W]
        pred = (probs > threshold).float()             # [B,1,H,W]
        true = true_masks.float()                      # [B,1,H,W] или [B,H,W] -> добавим измерение
        if true.dim() == 3:
            true = true.unsqueeze(1)
        # Игнорирование: если ignore_index не None и присутствует в true, то пропустить
        if ignore_index is not None:
            ignore_mask = (true == ignore_index)
            pred = pred.masked_fill(ignore_mask, 0)
            true = true.masked_fill(ignore_mask, 0)
        # Бинарные метрики
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
        correct = (pred == true).float().sum(dim=(2,3))
        total = pred.shape[2] * pred.shape[3]
        accuracy = correct / total
        map50 = (iou > 0.5).float().mean().item()
        map50_95 = iou.mean().item()
        return {
            'iou': iou.mean().item(),
            'dice': dice.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'accuracy': accuracy.mean().item(),
            'map50': map50,
            'map50_95': map50_95
        }
    else:
        # Многоклассовый режим
        if output_activation == 'argmax':
            pred = torch.argmax(pred_logits, dim=1)   # [B, H, W]
        elif output_activation == 'softmax':
            probs = torch.softmax(pred_logits, dim=1)
            pred = torch.argmax(probs, dim=1)
        else:
            # fallback
            pred = torch.argmax(pred_logits, dim=1)
        true = true_masks.long()                     # [B, H, W]

        # Игнорирование
        if ignore_index is not None:
            valid_mask = (true != ignore_index)
            pred = pred[valid_mask]
            true = true[valid_mask]
        # Метрики по каждому классу
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
            'iou': miou,
            'dice': mdice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'map50': map50,
            'map50_95': map50_95
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

# ---------- Визуализации: PR-кривая (только бинар), матрица ошибок, примеры ----------
def save_pr_curve(preds, masks, save_dir, num_classes=1, threshold=0.5):
    if not HAS_MATPLOTLIB or num_classes != 1:
        return
    # preds: [B,1,H,W] логиты или вероятности? будем использовать сигмоид
    if isinstance(preds, torch.Tensor):
        probs = torch.sigmoid(preds).cpu().numpy().flatten()
        masks = masks.cpu().numpy().flatten()
    else:
        probs = preds.flatten()
        masks = masks.flatten()
    thresholds = np.linspace(0, 1, 51)
    precisions, recalls, f1_scores = [], [], []
    for th in thresholds:
        pred_bin = (probs > th).astype(np.float32)
        tp = np.sum(pred_bin * masks)
        fp = np.sum(pred_bin * (1 - masks))
        fn = np.sum((1 - pred_bin) * masks)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, marker='.', label='PR curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'MaskPR_curve.png'))
    plt.close()

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
    log(f"Кривые PR, F1 сохранены в {save_dir}")

def save_confusion_matrix(preds, true_masks, save_dir, num_classes, ignore_index=None, threshold=0.5):
    if not HAS_MATPLOTLIB:
        return
    # Приводим к предсказанным классам
    if num_classes == 1:
        prob = torch.sigmoid(preds)
        pred_bin = (prob > threshold).long()
        true_bin = (true_masks > 0.5).long() if true_masks.dim() == 3 else true_masks.long().unsqueeze(1)
        # Игнорируем ignore_index
        if ignore_index is not None:
            ignore_mask = (true_masks == ignore_index)
            pred_bin = pred_bin.masked_fill(ignore_mask, 0)
            true_bin = true_bin.masked_fill(ignore_mask, 0)
        tp = ((pred_bin == 1) & (true_bin == 1)).sum().item()
        fp = ((pred_bin == 1) & (true_bin == 0)).sum().item()
        fn = ((pred_bin == 0) & (true_bin == 1)).sum().item()
        tn = ((pred_bin == 0) & (true_bin == 0)).sum().item()
        conf_matrix = np.array([[tn, fp], [fn, tp]])
        tick_labels = ['Background', 'Foreground']
    else:
        pred = torch.argmax(preds, dim=1)
        true = true_masks.long()
        if ignore_index is not None:
            valid = (true != ignore_index)
            pred = pred[valid]
            true = true[valid]
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

    # Нормализованная
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

def save_train_batch_examples(train_loader, save_dir, num_batches=3, samples_per_batch=4, num_classes=1, ignore_index=None):
    if not HAS_MATPLOTLIB:
        return
    os.makedirs(save_dir, exist_ok=True)
    batch_idx = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
        n_images = min(images.size(0), samples_per_batch)
        images = images[:n_images]
        masks = masks[:n_images]  # [B, H, W]
        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 4, 8))
        if n_images == 1:
            axes = axes.reshape(2, 1)
        for i in range(n_images):
            img = images[i].cpu().permute(1,2,0).numpy() * 255
            img = img.astype(np.uint8)
            axes[0, i].imshow(img)
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Image {i+1}')
            # Маска
            mask = masks[i].cpu().numpy()
            if num_classes == 1:
                # Бинарная
                mask_viz = (mask > 0.5).astype(np.uint8) * 255
                axes[1, i].imshow(mask_viz, cmap='gray')
            else:
                # Многоклассовая: раскраска
                mask_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for c in range(num_classes):
                    mask_viz[mask == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
                axes[1, i].imshow(mask_viz)
            axes[1, i].axis('off')
            axes[1, i].set_title(f'Mask {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'train_batch{batch_idx}.jpg'))
        plt.close()
    log(f"Сохранено {num_batches} тренировочных батчей (показано до {samples_per_batch} изображений) в {save_dir}")

def save_val_examples(val_loader, model, device, save_dir, num_samples=8, num_classes=1,
                      output_activation='argmax', ignore_index=None):
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
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
            else:
                if output_activation == 'argmax':
                    preds = torch.argmax(outputs, dim=1).float()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).float()
            for i in range(images.size(0)):
                if collected >= num_samples:
                    break
                img = images[i].cpu().permute(1,2,0).numpy() * 255
                img = img.astype(np.uint8)
                true_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()
                images_list.append(img)
                true_masks_list.append(true_mask)
                pred_masks_list.append(pred_mask)
                collected += 1
            if collected >= num_samples:
                break

    if not images_list:
        log("Нет примеров для сохранения валидации")
        return

    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for i in range(n):
        axes[i,0].imshow(images_list[i])
        axes[i,0].axis('off')
        axes[i,0].set_title('Image')
        # Истинная маска
        if num_classes == 1:
            true_viz = (true_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i,1].imshow(true_viz, cmap='gray')
        else:
            true_viz = np.zeros_like(images_list[i])
            for c in range(num_classes):
                true_viz[true_masks_list[i] == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
            axes[i,1].imshow(true_viz)
        axes[i,1].axis('off')
        axes[i,1].set_title('True Mask')
        # Предсказанная
        if num_classes == 1:
            pred_viz = (pred_masks_list[i] > 0.5).astype(np.uint8) * 255
            axes[i,2].imshow(pred_viz, cmap='gray')
        else:
            pred_viz = np.zeros_like(images_list[i])
            for c in range(num_classes):
                pred_viz[pred_masks_list[i] == c] = [c * 255 // max(1, num_classes-1), 128, 255 - c*128]
            axes[i,2].imshow(pred_viz)
        axes[i,2].axis('off')
        axes[i,2].set_title('Pred Mask')
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

    log("WORKER STARTED (DeepLabV3+)")

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
    model_path = params.get('model_path')   # пользовательский .pth или None
    project_path = params.get('project_path')
    name = params.get('name')
    backbone = params.get('backbone', 'resnet50')
    output_stride = params.get('output_stride', 16)
    output_activation = params.get('output_activation', 'argmax')
    num_classes = params.get('num_classes', 21)
    ignore_index = params.get('ignore_index')  # целое или None
    class_weights = params.get('class_weights')  # None, 'auto' или путь к файлу
    loss_name = params.get('loss', 'CrossEntropy')
    loss_weights = params.get('loss_weights', {'ce':1.0, 'dice':1.0, 'focal':0.0})
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
    augmentations = params.get('augmentations', {})
    device_pref = params.get('device', 'auto')
    threshold = params.get('threshold', 0.5)
    num_workers = params.get('num_workers', NUM_WORKERS_DEFAULT)


    # Определяем устройство для auto batch (требуется для auto_batch_size)
    if device_pref == 'auto':
        temp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.isdigit():
        if torch.cuda.is_available() and int(device_pref) < torch.cuda.device_count():
            temp_device = torch.device(f'cuda:{device_pref}')
        else:
            temp_device = torch.device('cpu')
    elif device_pref.lower() in ('cuda', 'gpu'):
        temp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_pref.lower() == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        temp_device = torch.device('mps')
    else:
        temp_device = torch.device('cpu')

    # Автоматический подбор batch_size
    if batch_size == -1:
        batch_size = auto_batch_size(temp_device)
        log(f"Auto batch: выбран batch_size = {batch_size}")
    elif batch_size <= 0:
        log(f"Ошибка: batch_size должен быть положительным, получено {batch_size}")
        sys.exit(1)


    # Путь сохранения
    if name and project_path:
        save_root = os.path.join(project_path, name)
    elif name:
        save_root = name
    elif project_path:
        save_root = project_path
    else:
        save_root = "runs/deeplabv3"
    os.makedirs(save_root, exist_ok=True)
    weights_dir = os.path.join(save_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    log("=== Запуск обучения DeepLabV3+ ===")
    log(f"Датасет: {dataset_path}")
    log(f"Backbone: {backbone}, output_stride={output_stride}")
    log(f"Выходная активация: {output_activation}")
    log(f"Классов: {num_classes}, ignore_index={ignore_index}")
    log(f"Веса классов: {class_weights}")
    log(f"Loss: {loss_name}, веса loss: {loss_weights}")
    log(f"Оптимизатор: {optimizer_name}, LR={lr}, momentum={momentum}, WD={weight_decay}")
    log(f"Scheduler: {scheduler_name}")
    log(f"Эпохи: {epochs}, two_phase={two_phase} (freeze={freeze_epochs}, unfreeze={unfreeze_epochs})")
    log(f"Batch size: {batch_size}, device: {device_pref}")
    log(f"Аугментации: {augmentations}")
    log(f"Порог (threshold): {threshold}")
    log(f"Результаты будут сохранены в: {save_root}")

    # Проверка датасета
    if not dataset_path or not os.path.exists(dataset_path):
        log(f"Ошибка: папка датасета не существует: {dataset_path}")
        sys.exit(1)

    # Определяем, какой формат датасета использовать
    # Если есть подпапки images/train и masks/train – используем U‑Net стиль
    try:
        if os.path.exists(os.path.join(dataset_path, 'images', 'train')) and \
           os.path.exists(os.path.join(dataset_path, 'masks', 'train')):
            # U‑Net стиль
            train_img = os.path.join(dataset_path, 'images', 'train')
            train_mask = os.path.join(dataset_path, 'masks', 'train')
            val_img = os.path.join(dataset_path, 'images', 'val')
            val_mask = os.path.join(dataset_path, 'masks', 'val')
            train_dataset = VOCSegmentationDataset(train_img, train_mask, input_size=512,
                                                   num_classes=num_classes, ignore_index=ignore_index,
                                                   augmentations=augmentations)
            val_dataset = VOCSegmentationDataset(val_img, val_mask, input_size=512,
                                                 num_classes=num_classes, ignore_index=ignore_index,
                                                 augmentations={}) if os.path.exists(val_img) else None
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_dataset else None
            train_len = len(train_dataset)
            val_len = len(val_dataset) if val_dataset else 0
        else:
            # VOC-стиль (ImageSets/Segmentation)
            train_loader, val_loader, train_len, val_len = create_voc_loaders(
                dataset_path, input_size=512, num_classes=num_classes,
                augmentations=augmentations, ignore_index=ignore_index,
                batch_size=batch_size, num_workers=num_workers
            )
    except Exception as e:
        log(f"Ошибка при создании датасета: {e}")
        sys.exit(1)

    log(f"Загружено {train_len} обучающих, {val_len} валидационных образцов")

    # --- Модель DeepLabV3+ ---
    global model, best_model_path, final_model_path
    load_pretrained = (model_path is None) or (not os.path.exists(model_path))
    # Для DeepLabV3+ используем smp.DeepLabV3Plus
    model = smp.DeepLabV3Plus(
        encoder_name=backbone,
        encoder_weights='imagenet' if load_pretrained else None,
        in_channels=3,
        classes=num_classes,
        encoder_output_stride=output_stride,
        decoder_use_batchnorm=True,
    )

    if model_path is not None and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log(f"⚠️ Отсутствуют ключи: {missing[:5]}..." if len(missing)>5 else f"⚠️ Отсутствуют ключи: {missing}")
        if unexpected:
            log(f"⚠️ Лишние ключи: {unexpected[:5]}..." if len(unexpected)>5 else f"⚠️ Лишние ключи: {unexpected}")
        log(f"Загружены веса из {model_path}")

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

    # Функция потерь (с учётом class_weights и ignore_index)
    # Для class_weights='auto' нужно вычислить веса по датасету. Пока реализуем позже.
    if class_weights == 'auto':
        log("Автоматический расчёт весов классов по обучающей выборке...")
        class_counts = torch.zeros(num_classes, dtype=torch.long, device=device)  # ← добавлен device
        for _, mask in train_loader:
            mask = mask.to(device)
            if ignore_index is not None:
                mask = mask[mask != ignore_index]
            if mask.numel() == 0:
                continue
            unique, counts = torch.unique(mask, return_counts=True)
            for c, cnt in zip(unique, counts):
                if c < num_classes:
                    class_counts[c] += cnt
    elif class_weights is not None and isinstance(class_weights, str) and os.path.exists(class_weights):
        with open(class_weights, 'r') as f:
            w = json.load(f)
        class_weights = w
    # Иначе class_weights остаётся None или списком

    criterion = get_loss_function(loss_name, num_classes, ignore_index=ignore_index,
                                  class_weights=class_weights, loss_weights=loss_weights)

    # Оптимизатор
    optimizer = get_optimizer(model, optimizer_name, lr, momentum, weight_decay)

    # Scheduler
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)

    best_model_path = os.path.join(weights_dir, 'best.pth')
    final_model_path = os.path.join(weights_dir, 'last.pth')

    # Сохраняем аргументы
    args_dict = {
        'backbone': backbone,
        'output_stride': output_stride,
        'output_activation': output_activation,
        'num_classes': num_classes,
        'ignore_index': ignore_index,
        'class_weights': class_weights,
        'loss': loss_name,
        'loss_weights': loss_weights,
        'optimizer': optimizer_name,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'scheduler': scheduler_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'two_phase_training': two_phase,
        'freeze_epochs': freeze_epochs,
        'unfreeze_epochs': unfreeze_epochs,
        'augmentations': augmentations,
        'device': device_pref,
        'threshold': threshold,
        'num_workers': num_workers,
        'project': project_path,
        'name': name,
        'date': datetime.now().isoformat()
    }
    save_args_yaml(args_dict, os.path.join(save_root, 'args.yaml'))

    meta_info = {
        'backbone': backbone,
        'output_stride': output_stride,
        'output_activation': output_activation,
        'num_classes': num_classes,
        'ignore_index': ignore_index,
        'loss': loss_name,
        'optimizer': optimizer_name,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'device': str(device),
        'experiment_name': name,
        'threshold': threshold,
        'best_val_loss': None,
        'date': datetime.now().isoformat()
    }
    save_model_meta(meta_info, best_model_path)
    save_model_meta(meta_info, final_model_path)

    # --- Two-phase training ---
    def freeze_backbone():
        for param in model.encoder.parameters():
            param.requires_grad = False
        log("Backbone заморожен.")
    def unfreeze_backbone():
        for param in model.encoder.parameters():
            param.requires_grad = True
        log("Backbone разморожен.")

    if two_phase:
        total_epochs = freeze_epochs + unfreeze_epochs
        if total_epochs != epochs:
            log(f"Предупреждение: сумма freeze_epochs+unfreeze_epochs={total_epochs} не равна epochs={epochs}. Использую {total_epochs}.")
            epochs = total_epochs
        # Первая фаза: заморозка
        freeze_backbone()
        # Пересоздаём оптимизатор? Необязательно, но для чистоты можно создать новый (со старыми параметрами)
        optimizer = get_optimizer(model, optimizer_name, lr, momentum, weight_decay)
        # Сбросим scheduler, если он завязан на эпохи (оставим как есть)
    else:
        freeze_epochs = 0
        unfreeze_epochs = epochs

    # Early stopping переменные
    patience_counter = 0
    best_val_loss = float('inf')
    global current_epoch, stop_requested

    log("=== Начало обучения ===")

    # Цикл по эпохам (с учётом двухфазного разделения)
    for epoch in range(1, epochs + 1):
        current_epoch = epoch
        if stop_requested:
            log("Обучение остановлено пользователем. Завершаем...")
            break

        # Определяем, в какой мы фазе (только если two_phase)
        if two_phase and epoch == freeze_epochs + 1:
            log("Переключение во вторую фазу: разморозка backbone.")
            unfreeze_backbone()
            # Сбросим оптимизатор (можно сохранить параметры, но лучше создать заново с меньшим lr)
            # По желанию можно уменьшить lr в 10 раз
            new_lr = lr * 0.1
            log(f"Устанавливаем новый learning rate для второй фазы: {new_lr}")
            optimizer = get_optimizer(model, optimizer_name, new_lr, momentum, weight_decay)
            # Создаём scheduler заново (полиномиальный от начала второй фазы)
            remaining_epochs = epochs - freeze_epochs
            scheduler = get_scheduler(optimizer, scheduler_name, remaining_epochs)

        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # masks: [B, H, W] (индексы) для мультикласса; для бинарного: [B,H,W] или [B,1,H,W]?
            # Наша маска всегда [B,H,W] (индексы). Для бинарной (num_classes==1) она содержит 0/1.
            # loss ожидает для BCEWithLogitsLoss: target [B,1,H,W]? В нашей get_loss_function для бинарного режима используется BCEWithLogitsLoss, который требует target [B,1,H,W].
            # Приведём маску к нужному виду.
            if num_classes == 1:
                target = masks.unsqueeze(1).float()   # [B,1,H,W]
            else:
                target = masks.long()                 # [B,H,W]
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= train_len

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
                        target = masks.unsqueeze(1).float()
                        loss = criterion(outputs, target)
                        all_preds.append(outputs.cpu())   # логиты
                    else:
                        target = masks.long()
                        loss = criterion(outputs, target)
                        all_preds.append(outputs.cpu())   # логиты
                    all_masks.append(masks.cpu())
                    val_loss += loss.item() * images.size(0)
            val_loss /= val_len
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)
            metrics = compute_metrics(all_preds, all_masks, num_classes,
                                      ignore_index=ignore_index,
                                      output_activation=output_activation,
                                      threshold=threshold)
        else:
            val_loss = None
            metrics = None

        # Сохранение лучшей модели
        if val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                log(f"✅ Новая лучшая модель сохранена (val_loss={best_val_loss:.4f})")
            else:
                rel_increase = (val_loss - best_val_loss) / (best_val_loss + 1e-8)
                if rel_increase > EARLY_STOPPING_TOLERANCE:
                    log(f"⚠️ Переобучение: val_loss вырос на {rel_increase*100:.1f}% > {EARLY_STOPPING_TOLERANCE*100:.0f}%. Остановка обучения.")
                    break
                loss_gap = (val_loss - train_loss) / (train_loss + 1e-8)
                if loss_gap > OVERFITTING_GAP_THRESHOLD and patience_counter >= 1:
                    log(f"⚠️ Переобучение: разрыв val_loss ({val_loss:.4f}) и train_loss ({train_loss:.4f}) слишком велик ({loss_gap*100:.1f}%). Остановка обучения.")
                    break
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    log(f"⚠️ Early stopping: val_loss не улучшается {EARLY_STOPPING_PATIENCE} эпох. Остановка обучения.")
                    break
        else:
            if best_val_loss == float('inf'):
                best_val_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
                log(f"✅ Модель сохранена (нет валидации)")

        # Сохраняем историю
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        if metrics:
            for k in ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'map50', 'map50_95']:
                history[k].append(metrics[k])  # теперь k точно совпадает с ключом в metrics
        else:
            for k in ['iou', 'dice', 'f1', 'precision', 'recall', 'accuracy', 'map50', 'map50_95']:
                history[k].append(float('nan'))

        # Обновляем scheduler
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
            log(f"Эпоха {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f} | F1: {metrics['f1']:.4f}")
        else:
            log(f"Эпоха {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Валидация пропущена")

    # Сохраняем финальную модель
    torch.save(model.state_dict(), final_model_path)
    log(f"Финальная модель сохранена: {final_model_path}")

    # Обновляем метаданные
    meta_info['best_val_loss'] = best_val_loss if best_val_loss != float('inf') else None
    meta_info['date'] = datetime.now().isoformat()
    save_model_meta(meta_info, best_model_path)
    save_model_meta(meta_info, final_model_path)

    # Итоговые отчёты и визуализации
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
                all_preds.append(outputs.cpu())
                all_masks.append(masks.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        save_csv_results(history, os.path.join(save_root, 'results.csv'))
        if HAS_MATPLOTLIB:
            save_training_plots(history, save_root)
            if num_classes == 1:
                save_pr_curve(all_preds, all_masks, save_root, num_classes, threshold)
            save_confusion_matrix(all_preds, all_masks, save_root, num_classes, ignore_index, threshold)
        # Сохраняем примеры тренировочных батчей (первые несколько)
        try:
            save_train_batch_examples(train_loader, save_root, num_batches=NUM_TRAIN_BATCHES_TO_SAVE,
                                      samples_per_batch=min(4, batch_size), num_classes=num_classes,
                                      ignore_index=ignore_index)
        except Exception as e:
            log(f"Ошибка сохранения тренировочных примеров: {e}")
        try:
            save_val_examples(val_loader, model, device, save_root, num_samples=NUM_VAL_SAMPLES_TO_SAVE,
                              num_classes=num_classes, output_activation=output_activation,
                              ignore_index=ignore_index)
        except Exception as e:
            log(f"Ошибка сохранения валидационных примеров: {e}")

        final_metrics = compute_metrics(all_preds, all_masks, num_classes, ignore_index, output_activation, threshold)
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