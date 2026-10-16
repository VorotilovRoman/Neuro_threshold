#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_embeddings.py (с поддержкой rotation-robust TTA)

Дополнительно к базовому режиму:
  --rotation-mode tta-average
      Для каждого снимка генерируются варианты (повороты × зеркало),
      прогоняются через DINOv2, эмбеддинги усредняются и L2-нормализуются.
      Итог — один инвариантный к поворотам вектор на изображение.
"""

import os
import sys
import argparse
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================================
CONFIG = {
    "input_dir":    r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    "input_dir":    r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin_aligned",
    "input_dir":    r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    "model_name":   "dinov2_vits14",
    "batch_size":   8,                # сколько ИСХОДНЫХ изображений за один шаг
    "num_workers":  0,
    "device":       "auto",
    "image_size":   224,
    "overwrite":    False,
    "recursive":    True,

    # --- Rotation-robust ---
    # none         — обычный режим (по умолчанию)
    # tta-average  — усреднение эмбеддингов по поворотам/зеркалам
    "rotation_mode": "none",
    "angles":        [0, 90, 180, 270],
    "apply_flip":    False,
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ----------------------------------------------------------------------------
#  Утилиты
# ----------------------------------------------------------------------------
def read_image_rgb(path: str):
    img = None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img[:, :, :3]
    return img


def collect_images(folder: str, recursive: bool = True):
    out = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(folder, f))
    out.sort()
    return out


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def embedding_path_for(img_path: str) -> str:
    return os.path.splitext(img_path)[0] + "_embeding.txt"


def apply_transform(img, angle_deg, flip):
    out = img
    if angle_deg != 0:
        k = (angle_deg // 90) % 4
        if k:
            out = np.rot90(out, k).copy()
    if flip:
        out = np.fliplr(out).copy()
    return out


# ----------------------------------------------------------------------------
#  DINOv2
# ----------------------------------------------------------------------------
def load_dinov2(model_name: str, device: str):
    print(f"Загрузка DINOv2 '{model_name}' через torch.hub...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return model, transform


@torch.no_grad()
def extract_batch(model, batch_tensor, device):
    batch_tensor = batch_tensor.to(device, non_blocking=True)
    feats = model(batch_tensor)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Извлечение эмбеддингов DINOv2 (опц. rotation-robust)."
    )
    ap.add_argument("-i", "--input", default=CONFIG["input_dir"])
    ap.add_argument("--model", default=CONFIG["model_name"])
    ap.add_argument("--batch-size", type=int, default=CONFIG["batch_size"],
                    help="сколько исходных изображений за раз (в TTA-режиме к этому "
                         "числу домножается количество вариантов)")
    ap.add_argument("--device", default=CONFIG["device"])
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    ap.add_argument("--no-recursive", action="store_true", default=not CONFIG["recursive"])
    ap.add_argument("--rotation-mode", choices=["none", "tta-average"],
                    default=CONFIG["rotation_mode"])
    ap.add_argument("--angles", default=",".join(str(a) for a in CONFIG["angles"]),
                    help="углы через запятую, например 0,90,180,270")
    ap.add_argument("--apply-flip", action="store_true", default=CONFIG["apply_flip"])
    return ap.parse_args()


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    if not args.input or not os.path.isdir(args.input):
        print(f"[ERROR] Папка не найдена: {args.input}")
        sys.exit(2)

    try:
        angles = [int(x.strip()) for x in args.angles.split(",") if x.strip()]
    except ValueError as e:
        print(f"[ERROR] Не удалось распарсить --angles: {e}")
        sys.exit(2)
    if not angles:
        angles = [0]

    recursive = not args.no_recursive
    paths = collect_images(args.input, recursive=recursive)
    total = len(paths)

    print("=" * 60)
    print(f"Input         : {args.input}")
    print(f"Model         : {args.model}")
    print(f"Rotation mode : {args.rotation_mode}")
    if args.rotation_mode == "tta-average":
        print(f"Angles        : {angles}, flip={args.apply_flip}")
    print(f"Recursive     : {recursive}")
    print(f"Images        : {total}")
    print("=" * 60)

    if total == 0:
        print("Нечего обрабатывать.")
        return

    if not args.overwrite:
        pending = [p for p in paths if not os.path.exists(embedding_path_for(p))]
        skipped = total - len(pending)
        if skipped:
            print(f"Пропущено (эмбеддинг уже есть): {skipped}")
        paths = pending
        total = len(paths)
        if total == 0:
            print("Все эмбеддинги уже извлечены.")
            return

    device = resolve_device(args.device)
    print(f"Device        : {device}")

    model, transform = load_dinov2(args.model, device)

    # Список вариантов (angle, flip)
    if args.rotation_mode == "tta-average":
        variants = [(a, False) for a in angles]
        if args.apply_flip:
            variants += [(a, True) for a in angles]
    else:
        variants = [(0, False)]
    K = len(variants)
    print(f"Вариантов на изображение: {K}")

    dim = None
    ok = 0
    failed = 0
    done = 0
    batch_size = max(1, args.batch_size)

    for start in range(0, total, batch_size):
        chunk = paths[start:start + batch_size]
        tensors = []
        chunk_paths = []
        for p in chunk:
            img = read_image_rgb(p)
            if img is None:
                print(f"[SKIP] не удалось прочитать: {p}")
                failed += 1
                continue
            chunk_paths.append(p)
            for (angle, flip) in variants:
                v = apply_transform(img, angle, flip)
                tensors.append(transform(Image.fromarray(v)))

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors)          # [N*K, 3, H, W]
        try:
            embs = extract_batch(model, batch_tensor, device)   # [N*K, D]
        except Exception as e:
            print(f"[ERROR] Инференс батча: {e}")
            failed += len(chunk_paths)
            continue

        if dim is None:
            dim = embs.shape[1]
            print(f"Размер эмбеддинга: {dim}")

        for i, p in enumerate(chunk_paths):
            image_embs = embs[i * K:(i + 1) * K]     # [K, D]
            if K > 1:
                avg = image_embs.mean(axis=0)
                n = np.linalg.norm(avg)
                if n > 0:
                    avg = avg / n
            else:
                avg = image_embs[0]

            out_path = embedding_path_for(p)
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(f"{v:.6f}" for v in avg.tolist()))
                ok += 1
            except Exception as e:
                print(f"[ERROR] Запись {out_path}: {e}")
                failed += 1

        done += len(chunk_paths)
        if done % 50 == 0 or done == total:
            print(f"[{done}/{total}] сохранено: {ok}, ошибок: {failed}")

    print("=" * 60)
    print(f"Готово. Сохранено эмбеддингов: {ok}")
    print(f"Ошибок: {failed}")
    print(f"Размерность: {dim}")


if __name__ == "__main__":
    main()