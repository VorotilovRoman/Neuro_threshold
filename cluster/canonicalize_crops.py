#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonicalize_crops.py

Приводит все кропы к единой ориентации относительно эталона:
перебирает повороты (по умолчанию 0/90/180/270) и зеркало,
выбирает комбинацию, при которой изображение максимально похоже
на эталон (Pearson correlation), и сохраняет выровненный кроп.

Workflow:
    1) python canonicalize_crops.py
    2) python extract_embeddings.py        # по папке *_aligned
    3) python cluster_embeddings.py        # по эмбеддингам из *_aligned
"""

import os
import sys
import csv
import argparse
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================================
CONFIG = {
    # --- Пути ---
    # Эталон. Пусто = берётся первое изображение из input_dir.
    # Хороший эталон — типичный представитель класса в желаемой ориентации.


    "input_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin_aligned",
    "reference_path": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\ref_IN.png",

    "input_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    "reference_path": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\ref_OUT.png",

    "recursive": False,

    # --- Параметры выравнивания ---
    # Углы (против часовой, в градусах). Для кратных 90° — [0, 90, 180, 270].
    # Для произвольной ориентации — list(range(0, 360, 15)) или похожее.
    "angles": [0, 90, 180, 270],
    "apply_flip": True,      # пробовать горизонтальное отражение
    "compare_size": 224,     # размер для метрики (на сохраняемое разрешение не влияет)

    # --- Прочее ---
    "overwrite": False,
    "save_debug": True,      # сохранять монтаж выровненных кропов
    "debug_max": 60,
    "debug_cols": 8,
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ----------------------------------------------------------------------------
#  Утилиты
# ----------------------------------------------------------------------------
def read_image_bgr(path: str):
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
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def imwrite_safe(path: str, img_bgr) -> bool:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def collect_images(folder: str, recursive: bool = False):
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


def prepare_compare(img_bgr, size: int):
    """Grayscale → ресайз с сохранением пропорций + паддинг → zero-mean/unit-std."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = size / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size), dtype=np.float32)
    yo = (size - nh) // 2
    xo = (size - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized.astype(np.float32)

    m = canvas.mean()
    s = canvas.std() + 1e-6
    return (canvas - m) / s


def apply_transform(img, angle_deg: int, flip: bool):
    out = img
    if angle_deg != 0:
        k = (angle_deg // 90) % 4
        if k:
            out = np.rot90(out, k).copy()
    if flip:
        out = np.fliplr(out).copy()
    return out


def pearson(a, b) -> float:
    return float((a * b).mean())


def find_best_transform(img_bgr, ref_cmp, angles, apply_flip, compare_size):
    flips = [False, True] if apply_flip else [False]
    best_score = -np.inf
    best_angle = 0
    best_flip = False
    for angle in angles:
        for flip in flips:
            v = apply_transform(img_bgr, angle, flip)
            cmp_ = prepare_compare(v, compare_size)
            s = pearson(cmp_, ref_cmp)
            if s > best_score:
                best_score, best_angle, best_flip = s, angle, flip
    return best_angle, best_flip, best_score


def make_debug_montage(records, out_path, cols=8, max_items=60):
    """records: list of (img_bgr, angle, flip, score)."""
    n = min(len(records), max_items)
    if n == 0:
        return
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img_bgr, angle, flip, score = records[i]
            h, w = img_bgr.shape[:2]
            scale = 96 / max(h, w)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            disp = cv2.resize(img_bgr, (nw, nh))
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            ax.imshow(disp)
            ax.set_title(f"{angle}\u00b0 f={int(flip)} s={score:.2f}", fontsize=6)
    fig.suptitle("Aligned crops", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Приведение кропов к единой ориентации относительно эталона."
    )
    ap.add_argument("-i", "--input",  default=CONFIG["input_dir"])
    ap.add_argument("-o", "--output", default=CONFIG["output_dir"])
    ap.add_argument("-r", "--reference", default=CONFIG["reference_path"],
                    help="путь к эталону; пусто — первое изображение из input")
    ap.add_argument("--angles", default=",".join(str(a) for a in CONFIG["angles"]),
                    help="углы через запятую, например 0,90,180,270")
    ap.add_argument("--no-flip", action="store_true", default=not CONFIG["apply_flip"])
    ap.add_argument("--compare-size", type=int, default=CONFIG["compare_size"])
    ap.add_argument("--recursive", action="store_true", default=CONFIG["recursive"])
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    ap.add_argument("--no-debug", action="store_true", default=not CONFIG["save_debug"])
    ap.add_argument("--debug-max", type=int, default=CONFIG["debug_max"])
    ap.add_argument("--debug-cols", type=int, default=CONFIG["debug_cols"])
    return ap.parse_args()


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] Папка не найдена: {args.input}")
        sys.exit(2)

    try:
        angles = [int(x.strip()) for x in args.angles.split(",") if x.strip()]
    except ValueError as e:
        print(f"[ERROR] Не удалось распарсить --angles: {e}")
        sys.exit(2)
    if not angles:
        print("[ERROR] Список углов пуст.")
        sys.exit(2)

    apply_flip = not args.no_flip

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        print("[ERROR] input и output не должны совпадать.")
        sys.exit(2)

    os.makedirs(args.output, exist_ok=True)

    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        print(f"[ERROR] В {args.input} нет изображений.")
        sys.exit(2)

    if args.reference:
        ref_path = args.reference
        if not os.path.isfile(ref_path):
            print(f"[ERROR] Эталон не найден: {ref_path}")
            sys.exit(2)
        if ref_path not in images:
            images.append(ref_path)
    else:
        ref_path = images[0]
        print(f"Эталон (авто): {ref_path}")

    ref_bgr = read_image_bgr(ref_path)
    if ref_bgr is None:
        print(f"[ERROR] Не удалось прочитать эталон: {ref_path}")
        sys.exit(2)

    ref_cmp = prepare_compare(ref_bgr, args.compare_size)
    print(f"Эталон: shape={ref_bgr.shape}, compare_size={args.compare_size}")

    print("=" * 60)
    print(f"Input       : {args.input}")
    print(f"Output      : {args.output}")
    print(f"Reference   : {ref_path}")
    print(f"Angles      : {angles}")
    print(f"Flip        : {apply_flip}")
    print(f"Images total: {len(images)}")
    print("=" * 60)

    log_path = os.path.join(args.output, "alignment_log.csv")
    records_for_debug = []
    ok = 0
    skipped = 0
    failed = 0

    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "angle_deg", "flip", "score", "status"])

        for i, p in enumerate(images, 1):
            img = read_image_bgr(p)
            if img is None:
                w.writerow([os.path.basename(p), "", "", "", "read_failed"])
                failed += 1
                continue

            base_name = os.path.basename(p)
            out_path = os.path.join(args.output, base_name)
            if os.path.exists(out_path) and not args.overwrite:
                w.writerow([base_name, "", "", "", "skipped_exists"])
                skipped += 1
                continue

            angle, flip, score = find_best_transform(
                img, ref_cmp, angles, apply_flip, args.compare_size
            )
            aligned = apply_transform(img, angle, flip)

            if not imwrite_safe(out_path, aligned):
                w.writerow([base_name, angle, int(flip), f"{score:.4f}", "write_failed"])
                failed += 1
                continue

            w.writerow([base_name, angle, int(flip), f"{score:.4f}", "ok"])
            ok += 1
            records_for_debug.append((aligned, angle, flip, score))

            if i % 25 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] aligned={ok}, skipped={skipped}, failed={failed}")

    print("=" * 60)
    print(f"Готово. Выровнено: {ok}, пропущено: {skipped}, ошибок: {failed}")
    print(f"Лог: {log_path}")

    if not args.no_debug and records_for_debug:
        debug_path = os.path.join(args.output, "alignment_debug.png")
        make_debug_montage(records_for_debug, debug_path,
                           cols=args.debug_cols, max_items=args.debug_max)
        print(f"Debug-монтаж: {debug_path}")


if __name__ == "__main__":
    main()