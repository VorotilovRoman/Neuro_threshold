# utils/unet_io.py
"""
Загрузка U-Net-датасета из masks_data.yaml (или data.yaml со секцией masks).

Возвращает пары (img_path, mask_path) + метаданные классов.
"""
from import_libs_external import *
import os
from pathlib import Path

import yaml


def load_unet_yaml(yaml_path, log_cb=None):
    """
    Загружает masks_data.yaml (или data.yaml с masks-полями).

    Возвращает dict:
        {
            'pairs':            [(img_path, mask_path), ...],
            'class_names':      {yolo_cls_id: name},   # плотная 0..N-1
            'mask_class_offset': int,
            'num_classes':      int,     # включая фон
            'root':             str,
        }
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"U-Net YAML not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    root = cfg.get('path', '')
    if not root:
        root = os.path.dirname(yaml_path)
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(yaml_path), root))
    root = os.path.abspath(root)

    # Ключи могут быть разными: train_images/train_masks
    # или images/train + masks/train через общий train.
    splits_cfg = []
    for split in ('train', 'val', 'test'):
        img_rel = cfg.get(f'{split}_images')
        mask_rel = cfg.get(f'{split}_masks')

        # Fallback на общий формат YOLO: images/{split} + masks/{split}
        if not img_rel:
            split_val = cfg.get(split)
            if isinstance(split_val, str):
                img_rel = split_val
            elif isinstance(split_val, list) and split_val:
                img_rel = split_val[0]
        if not mask_rel:
            mask_rel = f'masks/{split}'

        if img_rel:
            splits_cfg.append((split, img_rel, mask_rel))

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    mask_exts = ('.png', '.tif', '.tiff', '.bmp')

    pairs = []
    for split, img_rel, mask_rel in splits_cfg:
        img_dir = os.path.join(root, img_rel)
        mask_dir = os.path.join(root, mask_rel)
        if not os.path.isdir(img_dir):
            if log_cb:
                log_cb(f"[U-Net] нет папки изображений: {img_dir}")
            continue
        if not os.path.isdir(mask_dir):
            if log_cb:
                log_cb(f"[U-Net] нет папки масок: {mask_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in img_exts:
                continue
            img_path = os.path.join(img_dir, fname)
            mask_path = None
            for mext in mask_exts:
                cand = os.path.join(mask_dir, stem + mext)
                if os.path.exists(cand):
                    mask_path = cand
                    break
            if mask_path is None:
                if log_cb:
                    log_cb(f"[U-Net] нет маски для {fname}")
                continue
            pairs.append((img_path, mask_path))

    # Классы
    class_names_list = cfg.get('class_names', [])
    mask_class_offset = int(cfg.get('mask_class_offset', 1))
    num_classes = int(cfg.get('num_classes', len(class_names_list)))

    # class_names_list[0] обычно 'background'.
    # Для приложения yolo_cls_id начинается с 0.
    # Если offset=1 → yolo_cls i соответствует pixel i+1, а name — class_names_list[i+1]
    # Если offset=0 → yolo_cls i соответствует pixel i, name — class_names_list[i]
    class_names = {}
    if class_names_list:
        if mask_class_offset == 1:
            for i, name in enumerate(class_names_list):
                if i == 0:
                    continue  # background
                class_names[i - 1] = str(name)
        else:
            for i, name in enumerate(class_names_list):
                class_names[i] = str(name)

    if log_cb:
        log_cb(f"[U-Net] загружено: pairs={len(pairs)}, "
               f"classes={len(class_names)}, "
               f"offset={mask_class_offset}, root={root}")

    return {
        'pairs': pairs,
        'class_names': class_names,
        'mask_class_offset': mask_class_offset,
        'num_classes': num_classes,
        'root': root,
    }


def is_unet_yaml(yaml_path):
    """
    Эвристика: является ли YAML U-Net-структурой.
    Признак — наличие 'train_masks' / 'val_masks' / 'mask_class_offset'
    или 'class_names' со строкой 'background'.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return False

    if any(k in cfg for k in ('train_masks', 'val_masks', 'test_masks',
                              'mask_class_offset')):
        return True
    cn = cfg.get('class_names')
    if isinstance(cn, list) and cn and str(cn[0]).lower() == 'background':
        return True
    return False