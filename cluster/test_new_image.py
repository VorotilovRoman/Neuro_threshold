#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_new_image.py

Тест на новом снимке/папке снимков:
  • YOLO детектирует объекты указанных классов;
  • для каждого объекта — выравнивание к референсу класса (поворот+зеркало);
  • DINOv2 извлекает эмбеддинг;
  • reducer + centroids из папки кластеризации → cluster_id;
  • все результаты собираются в одну картинку-коллаж, сгруппированные
    по (class_id, cluster_id).
"""

import os
import sys
import csv
import json
import argparse
import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================================
CONFIG = {
    # --- Что тестируем ---
    "image_path":   r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\test",   # файл ИЛИ папка
    "recursive":    False,                          # обходить подпапки

    # --- YOLO ---
    "yolo_model":   r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\weights-qfn147\best.pt",
    "target_classes": [0, 2],                        # какие классы искать
    "conf":         0.8,
    "iou":          0.45,
    "imgsz":        640,
    "device":       "auto",

    # --- Параметры на каждый класс ---
    # class_id -> {reference, cluster_model}
    # reference: путь к референсному снимку (для выравнивания).
    #            Пусто → выравнивание не выполняется.
    # cluster_model: папка, где сохранены centroids.npy, reducer.joblib, model.json
    "class_configs": {
        0: {
            "reference":     r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin\(26)_SOP_1_transformed 2026.10.09 11.37.16 Shot_crop_0.png",
            "cluster_model": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_OUT_pin",
        },
        2: {
            "reference":     r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin\(55)_QFN_1_transformed 2026.10.09 11.37.16 Shot_crop_0.png",
            "cluster_model": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_IN_pin",
        },
    },

    # --- Кропы ---
    "padding":      -10,      # расширение бокса в пикселях (может быть < 0)
    "min_size":     0,        # мин. сторона бокса в px; 0 — без фильтра

    # --- DINOv2 ---
    "dinov2_model": "dinov2_vits14",

    # --- Выравнивание ---
    "alignment_angles":       [0, 90, 180, 270],
    "alignment_flip":         True,
    "alignment_compare_size": 224,

    # --- Порог уверенности кластера ---
    # Если максимальное косинусное сходство с центроидом < min_sim,
    # объект помечается как шум (cluster=-1).
    "min_sim":      0.0,

    # --- Выход ---
    "output_dir":   r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\result",
    "save_crops":   False,   # сохранять кропы отдельными файлами
    "collage_cell": 220,
    "collage_cols": 8,

    # --- Режим ---
    "save_annotated_source": True,  # сохранить исходный снимок с боксами
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ============================================================================
#  I/O
# ============================================================================
def read_image_bgr(path):
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


def imwrite_safe(path, img_bgr):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def collect_images(path, recursive=False):
    if os.path.isfile(path):
        return [path]
    out = []
    if recursive:
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(path):
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(path, f))
    out.sort()
    return out


def resolve_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return device_str


# ============================================================================
#  Выравнивание (аналог canonicalize_crops)
# ============================================================================
def prepare_compare(img_bgr, size=224):
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


def apply_transform(img, angle_deg, flip):
    out = img
    if angle_deg != 0:
        k = (angle_deg // 90) % 4
        if k:
            out = np.rot90(out, k).copy()
    if flip:
        out = np.fliplr(out).copy()
    return out


def find_best_alignment(img_bgr, ref_cmp, angles, apply_flip, compare_size):
    flips = [False, True] if apply_flip else [False]
    best = (-np.inf, 0, False)
    for angle in angles:
        for flip in flips:
            v = apply_transform(img_bgr, angle, flip)
            c = prepare_compare(v, compare_size)
            s = float((c * ref_cmp).mean())
            if s > best[0]:
                best = (s, angle, flip)
    return best[1], best[2], best[0]


# ============================================================================
#  DINOv2
# ============================================================================
def load_dinov2(model_name, device):
    print(f"Загрузка DINOv2 '{model_name}'...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    from torchvision import transforms as T
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return model, transform


@torch.no_grad()
def embed_images(model, transform, images_bgr, device, batch_size=16):
    """images_bgr: list[np.ndarray BGR]. Возвращает np.ndarray [N, D] L2-норм."""
    out = []
    n = len(images_bgr)
    for start in range(0, n, batch_size):
        chunk = images_bgr[start:start + batch_size]
        tensors = []
        for img in chunk:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = transform(Image.fromarray(rgb))
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        feats = model(batch)
        feats = F.normalize(feats, dim=-1)
        out.append(feats.cpu().numpy())
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out)


# ============================================================================
#  Загрузка состояния кластеризации
# ============================================================================
def load_cluster_state(model_dir):
    with open(os.path.join(model_dir, "model.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    centroids = np.load(os.path.join(model_dir, "centroids.npy"))
    reducer = None
    red_path = os.path.join(model_dir, "reducer.joblib")
    if os.path.exists(red_path):
        reducer = joblib.load(red_path)
    cluster_ids = meta.get("cluster_ids", list(range(len(centroids))))
    return centroids, reducer, cluster_ids, meta


def predict_cluster(emb, centroids, reducer, cluster_ids, min_sim=0.0):
    X = emb.reshape(1, -1).astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if reducer is not None:
        X = reducer.transform(X)
    if centroids.size == 0:
        return -1, 0.0
    sims = (X @ centroids.T).ravel()
    best = int(np.argmax(sims))
    sim = float(sims[best])
    if sim < min_sim:
        return -1, sim
    return int(cluster_ids[best]), sim


# ============================================================================
#  Структура результата
# ============================================================================
@dataclass
class Record:
    source_image: str
    class_id: int
    cluster_id: int = -999
    sim: float = 0.0
    align_angle: int = 0
    align_flip: bool = False
    align_score: float = 0.0
    crop_bgr: np.ndarray = field(default=None, repr=False)
    aligned_bgr: np.ndarray = field(default=None, repr=False)


# ============================================================================
#  Коллаж
# ============================================================================
def build_collage(records, out_path, cell=220, cols=8, title=""):
    """
    Строит коллаж с группировкой по (class_id, cluster_id).
    Каждая группа начинается с НОВОЙ строки.

    Сохранение: рендерим фигуру в numpy → BGR → imwrite_safe
    (cv2.imencode + buf.tofile). Полностью обходит проблемы
    matplotlib/PIL с юникод-путями и «Invalid argument» на Windows.
    """
    if not records:
        return

    records = sorted(records, key=lambda r: (
        r.class_id,
        1 if r.cluster_id == -1 else 0,
        r.cluster_id,
        os.path.basename(r.source_image),
    ))

    # --- Группировка по (class_id, cluster_id) ---
    groups = []
    current_key, current_group = None, []
    for r in records:
        key = (r.class_id, r.cluster_id)
        if key != current_key:
            if current_group:
                groups.append((current_key, current_group))
            current_key, current_group = key, [r]
        else:
            current_group.append(r)
    if current_group:
        groups.append((current_key, current_group))

    # --- Раскладка: каждый кластер с новой строки ---
    layout = []
    row = 0
    for _key, group in groups:
        col = 0
        for r in group:
            layout.append((r, row, col))
            col += 1
            if col >= cols:
                col = 0
                row += 1
        if col > 0:
            row += 1

    total_rows = max(1, row)

    # --- Рисуем ---
    dpi = 110
    fig_w = cols * cell / dpi
    fig_h = total_rows * (cell + 30) / dpi
    fig, axes = plt.subplots(total_rows, cols,
                             figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for r, rr, cc in layout:
        idx = rr * cols + cc
        if idx >= len(axes):
            continue
        ax = axes[idx]
        img = r.aligned_bgr if r.aligned_bgr is not None else r.crop_bgr
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if r.cluster_id == -1:
            t = f"C{r.class_id} / noise  s={r.sim:.2f}"
            color = "red"
        else:
            t = f"C{r.class_id} / cl{r.cluster_id}  s={r.sim:.2f}"
            color = "black"
        ax.set_title(t, fontsize=7, color=color)

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    # --- Рендер в numpy без savefig/PIL ---
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = buf.reshape(h, w, 4)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    imwrite_safe(out_path, img_bgr)


# ============================================================================
#  Основная логика
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Тест YOLO+выравнивание+DINOv2+кластер на новом снимке."
    )
    ap.add_argument("-i", "--input", default=CONFIG["image_path"])
    ap.add_argument("-m", "--model", default=CONFIG["yolo_model"])
    ap.add_argument("-o", "--output", default=CONFIG["output_dir"])
    ap.add_argument("--conf", type=float, default=CONFIG["conf"])
    ap.add_argument("--iou", type=float, default=CONFIG["iou"])
    ap.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    ap.add_argument("--device", default=CONFIG["device"])
    ap.add_argument("--min-sim", type=float, default=CONFIG["min_sim"])
    ap.add_argument("--recursive", action="store_true", default=CONFIG["recursive"])
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Не найден путь: {args.input}")
        sys.exit(2)
    if not os.path.isfile(args.model):
        print(f"[ERROR] Не найден YOLO-модель: {args.model}")
        sys.exit(2)

    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        print("[ERROR] Не найдено изображений.")
        sys.exit(2)

    target_classes = CONFIG["target_classes"]
    class_configs = CONFIG["class_configs"]

    print("=" * 60)
    print(f"Вход           : {args.input} ({len(images)} шт.)")
    print(f"YOLO модель    : {args.model}")
    print(f"Классы         : {target_classes}")
    print(f"Output         : {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    crops_dir = os.path.join(args.output, "crops")
    if CONFIG["save_crops"]:
        os.makedirs(crops_dir, exist_ok=True)

    # --- YOLO ---
    print("Загрузка YOLO...")
    yolo = YOLO(args.model)
    device = resolve_device(args.device)
    print(f"YOLO device: {device}")

    # --- DINOv2 ---
    dinov2, dinov2_transform = load_dinov2(CONFIG["dinov2_model"], device)

    # --- Кластеризация + референсы на каждый класс ---
    class_state = {}
    for cls in target_classes:
        cfg = class_configs.get(cls, {})
        ref_path = cfg.get("reference", "")
        cluster_dir = cfg.get("cluster_model", "")

        if not cluster_dir or not os.path.isdir(cluster_dir):
            print(f"[WARN] Класс {cls}: папка кластеризации не найдена "
                  f"({cluster_dir}). Класс будет пропущен.")
            continue

        centroids, reducer, cluster_ids, meta = load_cluster_state(cluster_dir)
        ref_cmp = None
        if ref_path and os.path.isfile(ref_path):
            ref_img = read_image_bgr(ref_path)
            if ref_img is not None:
                ref_cmp = prepare_compare(ref_img, CONFIG["alignment_compare_size"])
        class_state[cls] = {
            "centroids": centroids,
            "reducer": reducer,
            "cluster_ids": cluster_ids,
            "ref_cmp": ref_cmp,
            "ref_path": ref_path,
        }
        print(f"Класс {cls}: K={len(cluster_ids)} кластеров, "
              f"reducer={'да' if reducer else 'нет'}, "
              f"ref={'есть' if ref_cmp is not None else 'нет'}")

    if not class_state:
        print("[ERROR] Ни для одного класса не настроена кластеризация.")
        sys.exit(2)

    # --- Обработка снимков ---
    all_records = []
    angles = CONFIG["alignment_angles"]
    apply_flip = CONFIG["alignment_flip"]
    cmp_size = CONFIG["alignment_compare_size"]
    pad = int(CONFIG["padding"])
    min_size = int(CONFIG["min_size"])

    for img_idx, img_path in enumerate(images, 1):
        print(f"[{img_idx}/{len(images)}] {os.path.basename(img_path)}")
        img_bgr = read_image_bgr(img_path)
        if img_bgr is None:
            print("  не удалось прочитать")
            continue

        H, W = img_bgr.shape[:2]
        results = yolo(img_bgr, conf=args.conf, iou=args.iou,
                       imgsz=args.imgsz, device=device, verbose=False)
        boxes = getattr(results[0], "boxes", None) if results else None
        if boxes is None or len(boxes) == 0:
            print("  объектов не найдено")
            continue

        cls_all = boxes.cls.cpu().numpy().astype(int)
        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()

        img_records = []
        for j in range(len(cls_all)):
            cls = int(cls_all[j])
            if cls not in class_state:
                continue
            x1, y1, x2, y2 = xyxy_all[j].tolist()
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            if min_size > 0 and (bw < min_size or bh < min_size):
                continue

            x1i = max(0, int(round(x1)) - pad)
            y1i = max(0, int(round(y1)) - pad)
            x2i = min(W, int(round(x2)) + pad)
            y2i = min(H, int(round(y2)) + pad)
            if x2i - x1i <= 1 or y2i - y1i <= 1:
                continue
            crop = img_bgr[y1i:y2i, x1i:x2i].copy()
            if crop.size == 0:
                continue

            rec = Record(
                source_image=img_path,
                class_id=cls,
                crop_bgr=crop,
            )
            # Выравнивание к референсу
            st = class_state[cls]
            if st["ref_cmp"] is not None:
                angle, flip, score = find_best_alignment(
                    crop, st["ref_cmp"], angles, apply_flip, cmp_size
                )
                rec.align_angle = angle
                rec.align_flip = flip
                rec.align_score = score
                rec.aligned_bgr = apply_transform(crop, angle, flip)
            else:
                rec.aligned_bgr = crop.copy()
            img_records.append(rec)

        print(f"  детекций целевых классов: {len(img_records)}")
        all_records.extend(img_records)

        # Annotated source (нарисуем боксы)
        if CONFIG["save_annotated_source"] and img_records:
            vis = img_bgr.copy()
            for j in range(len(cls_all)):
                cls = int(cls_all[j])
                if cls not in class_state:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy_all[j].tolist()]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(vis, f"C{cls} {conf_all[j]:.2f}",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_vis = os.path.join(args.output, f"{base}_detected.jpg")
            imwrite_safe(out_vis, vis)

    if not all_records:
        print("Не найдено ни одного объекта целевых классов.")
        return

    # --- Эмбеддинги одним батчем ---
    print(f"Извлечение эмбеддингов для {len(all_records)} объектов...")
    aligned_imgs = [r.aligned_bgr for r in all_records]
    embs = embed_images(dinov2, dinov2_transform, aligned_imgs, device,
                        batch_size=16)
    print(f"Эмбеддинги: {embs.shape}")

    # --- Кластеризация ---
    for rec, emb in zip(all_records, embs):
        st = class_state[rec.class_id]
        cluster_id, sim = predict_cluster(
            emb, st["centroids"], st["reducer"], st["cluster_ids"],
            min_sim=args.min_sim,
        )
        rec.cluster_id = cluster_id
        rec.sim = sim

    # --- Сохранение кропов (опционально) ---
    if CONFIG["save_crops"]:
        for idx, r in enumerate(all_records):
            base = os.path.splitext(os.path.basename(r.source_image))[0]
            tag = f"C{r.class_id}_cl{r.cluster_id}_s{r.sim:.2f}_{idx:04d}"
            out_crop = os.path.join(crops_dir, f"{base}__{tag}.png")
            imwrite_safe(out_crop, r.aligned_bgr)

    # --- Общий CSV ---
    csv_path = os.path.join(args.output, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_image", "class_id", "cluster_id", "similarity",
                    "align_angle", "align_flip", "align_score"])
        for r in all_records:
            w.writerow([os.path.basename(r.source_image), r.class_id,
                        r.cluster_id, f"{r.sim:.4f}",
                        r.align_angle, int(r.align_flip),
                        f"{r.align_score:.4f}"])
    print(f"CSV: {csv_path}")

    # --- Один общий коллаж ---
    collage_path = os.path.join(args.output, "collage_all.png")
    build_collage(all_records, collage_path,
                  cell=CONFIG["collage_cell"],
                  cols=CONFIG["collage_cols"],
                  title=f"Все объекты — {len(all_records)} шт. "
                        f"(сорт. по class/cluster)")
    print(f"Коллаж: {collage_path}")

    # --- Отдельные коллажи по классам ---
    for cls in sorted({r.class_id for r in all_records}):
        sub = [r for r in all_records if r.class_id == cls]
        sub_path = os.path.join(args.output, f"collage_class{cls}.png")
        build_collage(sub, sub_path,
                      cell=CONFIG["collage_cell"],
                      cols=CONFIG["collage_cols"],
                      title=f"Класс {cls} — {len(sub)} объектов")
        print(f"  коллаж класса {cls}: {sub_path}")

    # --- Итоговая статистика ---
    print("=" * 60)
    print("Итоги:")
    from collections import Counter
    counter = Counter((r.class_id, r.cluster_id) for r in all_records)
    for (cls, cl), cnt in sorted(counter.items()):
        print(f"  class {cls} / cluster {cl}: {cnt}")
    print("=" * 60)


if __name__ == "__main__":
    main()