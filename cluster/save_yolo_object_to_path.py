#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_crop_objects.py

Прогоняет YOLO по всем изображениям из INPUT_DIR, вырезает объекты
указанного CLASS_ID и сохраняет их в OUTPUT_DIR.

Запуск с настройками из блока CONFIG (см. ниже) — просто `python yolo_crop_objects.py`.
Либо с аргументами командной строки, например:
    python yolo_crop_objects.py -i ./images -o ./crops -m ./best.pt --class-id 0 --conf 0.9
"""

import os
import sys
import argparse

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
#  Можно править здесь. Аргументы командной строки перекрывают эти значения.
# ============================================================================
CONFIG = {
    "input_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\dataset_270\images\train",              # папка с изображениями (обязательно)
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin",              # папка для сохранения кропов (обязательно)
    "model_path": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\weights-qfn147\best.pt",              # путь к .pt / .onnx / .engine (обязательно)
    "class_id":   0,                # ID класса
    "conf":       0.7,              # порог уверенности
    "iou":        0.45,             # IoU для NMS
    "imgsz":      1280,              # размер инференса
    "device":     "auto",           # "auto" | "0" | "cpu" | "mps"
    "padding":    -10,              # расширение бокса в пикселях (может быть < 0)
    "out_ext":    None,             # None — как у исходника; либо ".png"
    "min_size":   0,                # мин. размер бокса по ширине И высоте (px); 0 — фильтр отключён
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ----------------------------------------------------------------------------
#  Утилиты
# ----------------------------------------------------------------------------
def read_image(path: str):
    """Чтение изображения с поддержкой кириллицы в пути + нормализация к BGR uint8."""
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

    # 16-бит / float → uint8
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def imwrite_safe(path: str, img) -> bool:
    """Запись с поддержкой кириллицы в пути."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def unique_path(path: str) -> str:
    """Если файл уже существует — добавляет _1, _2, ..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def collect_images(folder: str):
    """Возвращает список путей ко всем изображениям (рекурсивно)."""
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device_str == "mps":
        try:
            import torch
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return device_str


# ----------------------------------------------------------------------------
#  Основная логика
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="YOLO: вырезать объекты заданного класса и сохранить в папку."
    )
    ap.add_argument("-i", "--input",     default=CONFIG["input_dir"],
                    help="папка с исходными изображениями")
    ap.add_argument("-o", "--output",    default=CONFIG["output_dir"],
                    help="папка для сохранения кропов")
    ap.add_argument("-m", "--model",     default=CONFIG["model_path"],
                    help="путь к весам YOLO (.pt/.onnx/.engine)")
    ap.add_argument("--class-id", type=int, default=CONFIG["class_id"],
                    help="ID класса (по умолчанию 0)")
    ap.add_argument("--conf", type=float, default=CONFIG["conf"],
                    help="порог уверенности (по умолчанию 0.9)")
    ap.add_argument("--iou",  type=float, default=CONFIG["iou"])
    ap.add_argument("--imgsz", type=int,  default=CONFIG["imgsz"])
    ap.add_argument("--device", default=CONFIG["device"],
                    help="auto | 0 | cpu | mps")
    ap.add_argument("--padding", type=int, default=CONFIG["padding"],
                    help="расширение бокса в пикселях (может быть < 0)")
    ap.add_argument("--out-ext", default=CONFIG["out_ext"],
                    help="расширение выходных файлов, например .png; "
                         "по умолчанию — как у исходника")
    ap.add_argument("--min-size", type=int, default=CONFIG["min_size"],
                    help="минимальный размер бокса по ширине И высоте в пикселях; "
                         "объекты меньше — пропускаются. 0 — фильтр отключён.")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---- Валидация путей ----
    if not args.input:
        print("[ERROR] Не указана папка с изображениями (--input / CONFIG['input_dir']).")
        sys.exit(2)
    if not args.output:
        print("[ERROR] Не указана папка сохранения (--output / CONFIG['output_dir']).")
        sys.exit(2)
    if not args.model:
        print("[ERROR] Не указан путь к модели (--model / CONFIG['model_path']).")
        sys.exit(2)
    if not os.path.isdir(args.input):
        print(f"[ERROR] Папка не найдена: {args.input}")
        sys.exit(2)
    if not os.path.isfile(args.model):
        print(f"[ERROR] Файл модели не найден: {args.model}")
        sys.exit(2)

    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)

    print("=" * 60)
    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Model  : {args.model}")
    print(f"Class  : {args.class_id}")
    print(f"Conf   : {args.conf}, IoU: {args.iou}, imgsz: {args.imgsz}")
    print(f"Device : {device}, padding: {args.padding}px, min-size: {args.min_size}px")
    print("=" * 60)

    print("Загрузка модели YOLO...")
    model = YOLO(args.model)
    print("Модель загружена.")

    images = collect_images(args.input)
    total = len(images)
    print(f"Найдено изображений: {total}")
    if total == 0:
        print("Нечего обрабатывать.")
        return

    out_ext = args.out_ext
    if out_ext:
        if not out_ext.startswith("."):
            out_ext = "." + out_ext

    saved_total = 0
    processed = 0
    errors = 0
    skipped_small = 0

    for i, img_path in enumerate(images, 1):
        img = read_image(img_path)
        if img is None:
            print(f"[{i}/{total}] не удалось прочитать: {img_path}")
            errors += 1
            continue

        H, W = img.shape[:2]

        try:
            results = model(
                img,
                conf=float(args.conf),
                iou=float(args.iou),
                imgsz=int(args.imgsz),
                device=device,
                verbose=False,
            )
        except Exception as e:
            print(f"[{i}/{total}] ошибка инференса {img_path}: {e}")
            errors += 1
            continue

        boxes = getattr(results[0], "boxes", None) if results else None
        if boxes is None or len(boxes) == 0:
            processed += 1
            continue

        cls_all  = boxes.cls.cpu().numpy().astype(int)
        xyxy_all = boxes.xyxy.cpu().numpy()

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        saved_here = 0

        for j in range(len(cls_all)):
            if int(cls_all[j]) != int(args.class_id):
                continue

            x1, y1, x2, y2 = xyxy_all[j].tolist()
            pad = int(args.padding)

            # --- Фильтр по минимальному размеру бокса (по «сырому» боксу,
            #     до применения padding — чтобы padding не маскировал мелкие объекты) ---
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if args.min_size > 0 and (bw < args.min_size or bh < args.min_size):
                skipped_small += 1
                continue

            x1i = max(0, int(round(x1)) - pad)
            y1i = max(0, int(round(y1)) - pad)
            x2i = min(W, int(round(x2)) + pad)
            y2i = min(H, int(round(y2)) + pad)

            if x2i - x1i <= 1 or y2i - y1i <= 1:
                continue

            crop = img[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                continue

            ext = out_ext if out_ext else (os.path.splitext(img_path)[1] or ".png")
            out_name = f"{base_name}_crop_{saved_here}{ext}"
            out_path = unique_path(os.path.join(args.output, out_name))

            if imwrite_safe(out_path, crop):
                saved_total += 1
                saved_here += 1
            else:
                print(f"[{i}/{total}] не удалось записать: {out_path}")
                errors += 1

        processed += 1
        if i % 25 == 0 or i == total:
            print(f"[{i}/{total}] обработано, кропов сохранено: {saved_total}")

    print("=" * 60)
    print(f"Готово. Изображений обработано: {processed}/{total}")
    print(f"Вырезано объектов класса {args.class_id}: {saved_total}")
    print(f"Пропущено по min-size (< {args.min_size}px): {skipped_small}")
    print(f"Ошибок: {errors}")
    print(f"Результаты в: {args.output}")


if __name__ == "__main__":
    main()