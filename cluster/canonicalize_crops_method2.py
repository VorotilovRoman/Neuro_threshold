#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canonicalize_crops.py

Приводит все кропы к единой ориентации относительно эталона.

Два способа определения ориентации:

  • "correlation" — перебор дискретных углов (0/90/180/270)
    и зеркала; оценка = Pearson correlation по grayscale.

  • "features"    — SIFT/AKAZE/ORB матчинг между кропом и
    референсом, RANSAC → similarity-transform, из которого
    извлекается угол поворота (может быть произвольным,
    не только кратным 90°). Кроп поворачивается на этот угол.

  • "auto"        — сначала features, при неудаче (мало матчей
    или RANSAC не сошёлся) переключается на correlation.

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

    # "input_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin",
    # "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    # "reference_path": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\ref_OUT.png",

    "input_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin_aligned",
    # Эталон. Пусто = берётся первое изображение из input_dir.
    "reference_path": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\ref_IN.png",
    "recursive": False,

    # --- Способ определения ориентации ---
    # "features" | "correlation" | "auto"
    "match_method": "auto",

    # --- Параметры для correlation-режима ---
    "angles":     [0, 90, 180, 270],
    "apply_flip": True,
    "compare_size": 224,

    # --- Параметры для features-режима ---
    # "sift" | "akaze" | "orb"
    "feature_detector":     "sift",
    "feature_ratio":        0.75,   # Lowe ratio
    "feature_min_good":     6,      # минимум хороших матчей для оценки
    "feature_ransac_thresh": 3.0,   # пикселей, для estimateAffinePartial2D
    "feature_max_features": 1000,   # сколько фич извлекать

    # --- Прочее ---
    "overwrite": False,
    "save_debug": True,
    "debug_max": 60,
    "debug_cols": 8,
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


# ============================================================================
#  I/O
# ============================================================================
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


# ============================================================================
#  Correlation-режим (дискретные углы + Pearson)
# ============================================================================
def prepare_compare(img_bgr, size: int = 224):
    """Grayscale → ресайз → паддинг → zero-mean/unit-std."""
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


def find_best_by_correlation(img_bgr, ref_cmp, angles, apply_flip, compare_size):
    """Перебор дискретных углов и зеркала, оценка — Pearson."""
    flips = [False, True] if apply_flip else [False]
    best_score, best_angle, best_flip = -np.inf, 0, False
    for angle in angles:
        for flip in flips:
            v = apply_transform(img_bgr, angle, flip)
            c = prepare_compare(v, compare_size)
            s = float((c * ref_cmp).mean())
            if s > best_score:
                best_score, best_angle, best_flip = s, angle, flip
    return best_angle, best_flip, best_score


# ============================================================================
#  Features-режим (SIFT/AKAZE/ORB + RANSAC → произвольный угол)
# ============================================================================
def get_detector(method: str):
    """Возвращает (detector, norm_type)."""
    m = method.lower()
    if m == "sift":
        return cv2.SIFT_create(nfeatures=CONFIG["feature_max_features"]), cv2.NORM_L2
    if m == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if m == "orb":
        return cv2.ORB_create(nfeatures=CONFIG["feature_max_features"]), cv2.NORM_HAMMING
    raise ValueError(f"Неизвестный детектор: {method}")


def extract_features(img_bgr, detector):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


def match_good(des1, des2, norm_type, ratio=0.75):
    """Lowe ratio test, возвращает список good matches."""
    if des1 is None or des2 is None:
        return []
    if len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(norm_type)
    try:
        raw = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []
    good = []
    for m_n in raw:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def estimate_rotation(img_bgr, ref_data, apply_flip,
                      ratio, min_good, ransac_thresh):
    """
    Пытается оценить угол поворота кропа относительно референса
    через features + RANSAC.

    Возвращает (angle_deg, flip, n_inliers) или (None, False, 0).
    angle_deg — угол в градусах (может быть произвольным).
    """
    detector = ref_data["detector"]
    norm_type = ref_data["norm"]
    ref_kp = ref_data["kp"]
    ref_des = ref_data["des"]

    if ref_des is None or len(ref_des) < 2:
        return None, False, 0

    flips = [False, True] if apply_flip else [False]
    best = (None, False, 0)

    for flip in flips:
        cand = np.fliplr(img_bgr).copy() if flip else img_bgr
        kp, des = extract_features(cand, detector)
        if des is None or len(des) < 2:
            continue

        good = match_good(des, ref_des, norm_type, ratio)
        if len(good) < min_good:
            if len(good) > best[2]:
                best = (None, flip, len(good))
            continue

        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh,
                maxIters=2000,
                confidence=0.995,
            )
        except cv2.error:
            continue

        if M is None:
            continue

        # M = [a  -b  tx;  b  a  ty] — similarity transform.
        # Угол поворота = atan2(b, a)
        angle_rad = float(np.arctan2(M[1, 0], M[0, 0]))
        angle_deg = float(np.degrees(angle_rad))
        n_inliers = int(inliers.sum()) if inliers is not None else len(good)

        if n_inliers > best[2]:
            best = (angle_deg, flip, n_inliers)

    return best


def rotate_image(img_bgr, angle_deg, flip):
    """
    Применяет зеркало (если надо) и поворот на произвольный угол.
    Возвращает изображение с сохранением исходного размера
    (по возможности).
    """
    out = img_bgr
    if flip:
        out = np.fliplr(out).copy()
    if abs(angle_deg) < 1e-3:
        return out

    h, w = out.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Вычисляем новые размеры, чтобы ничего не обрезалось.
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(out, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================================
#  Debug-монтаж
# ============================================================================
def save_fig_via_cv2(fig, out_path, dpi=110):
    """fig → numpy → cv2.imencode → файл. Обходит savefig/PIL на Windows."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = buf.reshape(h, w, 4)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    ext = os.path.splitext(out_path)[1] or ".png"
    ok, enc = cv2.imencode(ext, img_bgr)
    if ok:
        enc.tofile(out_path)


def make_debug_montage(records, out_path, cols=8, max_items=60):
    """records: list of (img_bgr, info_str)."""
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
            img_bgr, info = records[i]
            h, w = img_bgr.shape[:2]
            scale = 96 / max(h, w)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            disp = cv2.resize(img_bgr, (nw, nh))
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            ax.imshow(disp)
            ax.set_title(info, fontsize=6)
    fig.suptitle("Aligned crops", fontsize=10)
    plt.tight_layout()
    save_fig_via_cv2(fig, out_path, dpi=110)


# ============================================================================
#  CLI
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Приведение кропов к единой ориентации относительно эталона."
    )
    ap.add_argument("-i", "--input",  default=CONFIG["input_dir"])
    ap.add_argument("-o", "--output", default=CONFIG["output_dir"])
    ap.add_argument("-r", "--reference", default=CONFIG["reference_path"],
                    help="путь к эталону; пусто — первое изображение из input")
    ap.add_argument("--match-method", choices=["auto", "features", "correlation"],
                    default=CONFIG["match_method"])
    ap.add_argument("--feature-detector", choices=["sift", "akaze", "orb"],
                    default=CONFIG["feature_detector"])
    ap.add_argument("--angles", default=",".join(str(a) for a in CONFIG["angles"]),
                    help="углы через запятую (только для correlation)")
    ap.add_argument("--no-flip", action="store_true", default=not CONFIG["apply_flip"])
    ap.add_argument("--compare-size", type=int, default=CONFIG["compare_size"])
    ap.add_argument("--recursive", action="store_true", default=CONFIG["recursive"])
    ap.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    ap.add_argument("--no-debug", action="store_true", default=not CONFIG["save_debug"])
    ap.add_argument("--debug-max", type=int, default=CONFIG["debug_max"])
    ap.add_argument("--debug-cols", type=int, default=CONFIG["debug_cols"])
    return ap.parse_args()


# ============================================================================
#  Main
# ============================================================================
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
    if not angles and args.match_method in ("correlation", "auto"):
        angles = [0]

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

    # --- Pre-compute: correlation ref + feature ref ---
    ref_cmp = prepare_compare(ref_bgr, args.compare_size) \
        if args.match_method in ("correlation", "auto") else None

    ref_data = None
    if args.match_method in ("features", "auto"):
        detector, norm_type = get_detector(args.feature_detector)
        kp, des = extract_features(ref_bgr, detector)
        ref_data = {
            "detector": detector,
            "norm": norm_type,
            "kp": kp,
            "des": des,
        }
        n_ref_feat = 0 if kp is None else len(kp)
        print(f"Референс: features={n_ref_feat} ({args.feature_detector})")
        if n_ref_feat < CONFIG["feature_min_good"]:
            print("[WARN] Слишком мало фич на референсе — features-режим "
                  "скорее всего не сработает, будет fallback на correlation.")

    print(f"Эталон: shape={ref_bgr.shape}, compare_size={args.compare_size}")

    print("=" * 60)
    print(f"Input         : {args.input}")
    print(f"Output        : {args.output}")
    print(f"Reference     : {ref_path}")
    print(f"Match method  : {args.match_method} "
          f"(detector={args.feature_detector})")
    print(f"Flip          : {apply_flip}")
    print(f"Images total  : {len(images)}")
    print("=" * 60)

    log_path = os.path.join(args.output, "alignment_log.csv")
    records_for_debug = []
    ok = 0
    skipped = 0
    failed = 0
    used_features = 0
    used_correlation = 0

    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "method_used", "angle_deg", "flip",
                    "score", "status"])

        for i, p in enumerate(images, 1):
            img = read_image_bgr(p)
            if img is None:
                w.writerow([os.path.basename(p), "", "", "", "", "read_failed"])
                failed += 1
                continue

            base_name = os.path.basename(p)
            out_path = os.path.join(args.output, base_name)
            if os.path.exists(out_path) and not args.overwrite:
                w.writerow([base_name, "", "", "", "", "skipped_exists"])
                skipped += 1
                continue

            method_used = None
            angle_out = 0.0
            flip_out = False
            score_out = 0.0
            aligned = None

            # --- Попытка через features ---
            if args.match_method in ("features", "auto") and ref_data is not None:
                angle_deg, flip_est, n_inl = estimate_rotation(
                    img, ref_data, apply_flip,
                    args.feature_ratio if hasattr(args, "feature_ratio") else CONFIG["feature_ratio"],
                    CONFIG["feature_min_good"],
                    CONFIG["feature_ransac_thresh"],
                )
                if angle_deg is not None:
                    # Вращаем на -angle_deg, чтобы привести кроп к ориентации референса
                    aligned = rotate_image(img, -angle_deg, flip_est)
                    angle_out = angle_deg
                    flip_out = bool(flip_est)
                    score_out = float(n_inl)
                    method_used = "features"
                    used_features += 1

            # --- Fallback на correlation ---
            if aligned is None and args.match_method in ("correlation", "auto"):
                if ref_cmp is not None:
                    ang_d, flip_d, s = find_best_by_correlation(
                        img, ref_cmp, angles, apply_flip, args.compare_size
                    )
                    aligned = apply_transform(img, ang_d, flip_d)
                    angle_out = float(ang_d)
                    flip_out = bool(flip_d)
                    score_out = float(s)
                    method_used = "correlation"
                    used_correlation += 1

            if aligned is None:
                w.writerow([base_name, "", "", "", "", "no_method_succeeded"])
                failed += 1
                continue

            if not imwrite_safe(out_path, aligned):
                w.writerow([base_name, method_used, f"{angle_out:.3f}",
                            int(flip_out), f"{score_out:.4f}", "write_failed"])
                failed += 1
                continue

            w.writerow([base_name, method_used, f"{angle_out:.3f}",
                        int(flip_out), f"{score_out:.4f}", "ok"])
            ok += 1

            info = f"{method_used[:3]} {angle_out:+.1f}\u00b0 " \
                   f"f={int(flip_out)} s={score_out:.2f}"
            records_for_debug.append((aligned, info))

            if i % 25 == 0 or i == len(images):
                print(f"[{i}/{len(images)}] aligned={ok}, "
                      f"skipped={skipped}, failed={failed}")

    print("=" * 60)
    print(f"Готово. Выровнено: {ok}, пропущено: {skipped}, ошибок: {failed}")
    print(f"  через features:    {used_features}")
    print(f"  через correlation: {used_correlation}")
    print(f"Лог: {log_path}")

    if not args.no_debug and records_for_debug:
        debug_path = os.path.join(args.output, "alignment_debug.png")
        make_debug_montage(records_for_debug, debug_path,
                           cols=args.debug_cols, max_items=args.debug_max)
        print(f"Debug-монтаж: {debug_path}")


if __name__ == "__main__":
    main()