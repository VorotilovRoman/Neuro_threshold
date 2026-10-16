#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_embeddings.py — кластеризация эмбеддингов DINOv2.

Режимы:
  1) Обучение:      python cluster_embeddings.py
  2) Предсказание:  python cluster_embeddings.py --predict

Ключевая идея: перед кластеризацией эмбеддинги сжимаются (UMAP/PCA),
иначе плотностные методы в 384D работают плохо (много «шума»).
Сжатие обучается на train и сохраняется, чтобы те же преобразования
применялись к новым эмбеддингам.
"""

import os
import sys
import csv
import json
import argparse
import warnings
import joblib

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


# ============================================================================
#  НАСТРОЙКИ ПО УМОЛЧАНИЮ
# ============================================================================
CONFIG = {
    # --- Пути ---
    "input_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_OUT_pin",
    "model_dir":  r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_OUT_pin",

    "input_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_IN_pin_aligned",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_IN_pin",
    "model_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_IN_pin",

    "input_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\crop_dataset_OUT_pin_aligned",
    "output_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_OUT_pin",
    "model_dir": r"D:\РАЗБОР РЕНТГЕН ФОТО\QFN SOP полный кадра экспорт\cluster\clusters_OUT_pin",


    # --- Снижение размерности ПЕРЕД кластеризацией ---
    # none | pca | umap
    "reduce_method": "umap",
    "reduce_dim":    20,          # до скольки компонент сжимать

    # --- Параметры кластеризации (режим обучения) ---
    "method":           "auto",   # auto | hdbscan | kmeans
    "n_clusters":       0,        # K-Means: k; 0 = автоподбор по silhouette
    "k_min":            2,
    "k_max":            12,
    "min_cluster_size": 3,        # HDBSCAN: min_cluster_size (для маленьких выборок)
    "min_samples":      None,

    # --- Поиск эмбеддингов ---
    "recursive":        False,

    # --- Визуализация ---
    "montage_cols":     6,
    "montage_max":      60,

    # --- Режим работы ---
    "predict":          False,
    "make_montages":    False,
}
# ============================================================================


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
EMB_SUFFIX = "_embeding.txt"


# ============================================================================
#  Утилиты
# ============================================================================
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


def collect_embeddings(folder: str, recursive: bool = False):
    out = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(EMB_SUFFIX):
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.endswith(EMB_SUFFIX):
                out.append(os.path.join(folder, f))
    out.sort()
    return out


def load_embedding(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            values = [float(v) for v in fh.read().split()]
    except Exception as e:
        print(f"[WARN] не удалось прочитать {path}: {e}")
        return None
    if not values:
        return None
    return np.asarray(values, dtype=np.float32)


def find_image_for_embedding(emb_path: str):
    base = emb_path[:-len(EMB_SUFFIX)]
    for ext in IMAGE_EXTS:
        for e in (ext, ext.upper()):
            cand = base + e
            if os.path.exists(cand):
                return cand
    return None


def load_all_embeddings(folder: str, recursive: bool = False):
    paths = collect_embeddings(folder, recursive=recursive)
    vectors, kept = [], []
    for p in paths:
        v = load_embedding(p)
        if v is None:
            continue
        vectors.append(v)
        kept.append(p)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), []
    X = np.vstack(vectors)
    X = normalize(X, norm="l2", axis=1)
    return X, kept


# ============================================================================
#  Снижение размерности
# ============================================================================
def build_reducer(method: str, n_components: int, X: np.ndarray):
    """Обучает reducer. Для маленьких выборок корректирует n_components."""
    n_samples, n_features = X.shape
    n_components = max(2, min(int(n_components), n_samples - 1, n_features))

    if method == "none":
        return None, None
    if method == "pca":
        red = PCA(n_components=n_components, random_state=42)
        red.fit(X)
        return red, "pca"
    if method == "umap":
        try:
            import umap
        except ImportError:
            print("[WARN] umap-learn не установлен, использую PCA. "
                  "Установите: pip install umap-learn")
            red = PCA(n_components=n_components, random_state=42)
            red.fit(X)
            return red, "pca"
        n_neighbors = min(15, max(2, n_samples - 1))
        red = umap.UMAP(
            n_components=n_components,
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        )
        red.fit(X)
        return red, "umap"
    raise ValueError(f"Неизвестный метод снижения размерности: {method}")


def apply_reducer(reducer, X):
    if reducer is None:
        return X
    return reducer.transform(X)


# ============================================================================
#  Кластеризация
# ============================================================================
def fit_hdbscan(X, min_cluster_size=3, min_samples=None):
    try:
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        model = SKHDBSCAN(min_cluster_size=min_cluster_size,
                          min_samples=min_samples)
        return model.fit_predict(X), "hdbscan(sklearn)"
    except ImportError:
        pass
    try:
        import hdbscan as hdbscan_pkg
        model = hdbscan_pkg.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        return model.fit_predict(X), "hdbscan(hdbscan)"
    except ImportError:
        raise RuntimeError(
            "HDBSCAN недоступен. Установите scikit-learn>=1.3 или `pip install hdbscan`."
        )


def fit_kmeans(X, n_clusters=8, n_init=10, random_state=42):
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    return model.fit_predict(X), model


def auto_kmeans_k(X, k_min=2, k_max=12, random_state=42):
    n = X.shape[0]
    k_max = min(k_max, max(2, n - 1))
    if k_max < k_min:
        return k_min
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        try:
            labels, _ = fit_kmeans(X, n_clusters=k, random_state=random_state)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, metric="cosine")
            print(f"  k={k}: silhouette={score:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        except Exception as e:
            print(f"  k={k}: ошибка {e}")
    print(f"  лучшее k={best_k} (silhouette={best_score:.4f})")
    return best_k


# ============================================================================
#  2D-проекция и визуализация (используется только для отчёта)
# ============================================================================
def reduce_2d_for_plot(X):
    n = X.shape[0]
    try:
        import umap
        n_neighbors = min(15, max(2, n - 1))
        reducer = umap.UMAP(
            n_components=2, metric="cosine",
            n_neighbors=n_neighbors, min_dist=0.1, random_state=42,
        )
        print(f"  2D-проекция: UMAP (n_neighbors={n_neighbors})")
        return reducer.fit_transform(X), "UMAP"
    except ImportError:
        pass
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        tsne = TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                    init="pca", random_state=42)
        print(f"  2D-проекция: t-SNE (perplexity={perplexity:.1f})")
        return tsne.fit_transform(X), "t-SNE"
    except Exception as e:
        print(f"  t-SNE недоступен ({e}), использую PCA")
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X), "PCA"


def plot_scatter(X2d, labels, out_path, title):
    unique = sorted(set(int(l) for l in labels))
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(unique):
        mask = labels == c
        if c == -1:
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c="lightgray", s=18, alpha=0.6,
                       label="noise (-1)", edgecolors="none")
        else:
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=[cmap(i % 20)], s=24, alpha=0.85,
                       label=f"cluster {c} ({mask.sum()})",
                       edgecolors="none")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8, markerscale=1.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def make_montage(image_paths, title, out_path, cols=6, max_items=60):
    n = min(len(image_paths), max_items)
    if n == 0:
        return
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            img = read_image_rgb(image_paths[i])
            if img is None:
                continue
            ax.imshow(img)
            ax.set_title(os.path.basename(image_paths[i])[:28], fontsize=6)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================================
#  Сохранение / загрузка состояния
# ============================================================================
def compute_centroids(X, labels):
    """Центроиды в том пространстве, в котором выполнялась кластеризация."""
    cluster_ids = sorted(int(c) for c in set(labels) if c != -1)
    centroids = []
    for c in cluster_ids:
        mask = labels == c
        centroid = X[mask].mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
        centroids.append(centroid.astype(np.float32))
    if not centroids:
        return np.zeros((0, X.shape[1]), dtype=np.float32), []
    return np.vstack(centroids), cluster_ids


def save_state(out_dir, centroids, cluster_ids, reducer, reducer_kind,
               meta, emb_paths, labels, image_paths):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "centroids.npy"), centroids)
    if reducer is not None:
        joblib.dump(reducer, os.path.join(out_dir, "reducer.joblib"))

    meta_full = dict(meta)
    meta_full["cluster_ids"] = cluster_ids
    meta_full["centroid_dim"] = int(centroids.shape[1]) if centroids.size else 0
    meta_full["reducer"] = reducer_kind
    meta_full["n_train"] = len(emb_paths)
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as fh:
        json.dump(meta_full, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "labels.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "embedding", "cluster"])
        for ep, lp, lb in zip(emb_paths, image_paths, labels):
            w.writerow([lp or "", os.path.basename(ep), int(lb)])


def load_state(model_dir):
    with open(os.path.join(model_dir, "model.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    centroids = np.load(os.path.join(model_dir, "centroids.npy"))
    reducer = None
    red_path = os.path.join(model_dir, "reducer.joblib")
    if os.path.exists(red_path):
        reducer = joblib.load(red_path)
    return centroids, reducer, meta


# ============================================================================
#  Режим обучения
# ============================================================================
def run_train(args):
    if not os.path.isdir(args.input):
        print(f"[ERROR] Папка не найдена: {args.input}")
        sys.exit(2)

    print("=" * 60)
    print(f"Папка эмбеддингов  : {args.input}")
    print(f"Папка результата   : {args.output}")
    print(f"Метод кластеризации: {args.method}")
    print(f"Снижение размерности: {args.reduce_method} → {args.reduce_dim}D")
    print("=" * 60)

    X, emb_paths = load_all_embeddings(args.input, recursive=args.recursive)
    if X.shape[0] == 0:
        print("Не найдено ни одного файла *_embeding.txt")
        sys.exit(2)
    print(f"Загружено эмбеддингов: {X.shape[0]}, размерность: {X.shape[1]}")

    image_paths = [find_image_for_embedding(p) for p in emb_paths]

    # --- снижение размерности ---
    reducer, reducer_kind = None, "none"
    if args.reduce_method != "none":
        print(f"Обучение reducer ({args.reduce_method})...")
        reducer, reducer_kind = build_reducer(args.reduce_method,
                                              args.reduce_dim, X)
        X_red = apply_reducer(reducer, X)
        print(f"  Размерность после снижения: {X_red.shape[1]} "
              f"(reducer={reducer_kind})")
    else:
        X_red = X

    # --- кластеризация ---
    if args.method == "hdbscan":
        print(f"Кластеризация HDBSCAN "
              f"(min_cluster_size={args.min_cluster_size})...")
        labels, method_name = fit_hdbscan(
            X_red,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
        )
    elif args.method == "kmeans":
        if args.n_clusters and args.n_clusters > 0:
            k = args.n_clusters
            print(f"Кластеризация K-Means, k={k}...")
        else:
            print("Поиск оптимального k по silhouette score...")
            k = auto_kmeans_k(X_red, k_min=args.k_min, k_max=args.k_max)
        labels, _ = fit_kmeans(X_red, n_clusters=k)
        method_name = f"kmeans(k={k})"
    else:
        print("Режим auto: пробую HDBSCAN...")
        try:
            labels_h, _ = fit_hdbscan(X_red,
                                      min_cluster_size=args.min_cluster_size,
                                      min_samples=args.min_samples)
            n_clusters = len(set(int(l) for l in labels_h if l != -1))
            n_noise = int((np.asarray(labels_h) == -1).sum())
            if n_clusters >= 2 and n_noise <= 0.5 * len(labels_h):
                labels, method_name = labels_h, "hdbscan"
            else:
                raise RuntimeError(
                    f"HDBSCAN дал {n_clusters} кластеров и {n_noise} шума "
                    f"({100*n_noise/len(labels_h):.0f}%)"
                )
        except Exception as e:
            print(f"HDBSCAN не подошёл ({e}), перехожу на K-Means")
            if args.n_clusters and args.n_clusters > 0:
                k = args.n_clusters
            else:
                k = auto_kmeans_k(X_red, k_min=args.k_min, k_max=args.k_max)
            labels, _ = fit_kmeans(X_red, n_clusters=k)
            method_name = f"kmeans(k={k})"

    labels = np.asarray(labels, dtype=int)
    n_clusters = len(set(int(c) for c in labels if c != -1))
    n_noise = int((labels == -1).sum())
    print(f"Кластеров: {n_clusters}, шумовых точек: {n_noise} "
          f"({100*n_noise/max(1,len(labels)):.0f}%)")

    if n_noise > 0.3 * len(labels):
        print("[WARN] Больше 30% точек отнесены к шуму. "
              "Попробуйте: --method kmeans -k 4   или   "
              "--reduce-dim 10   или   --min-cluster-size 2")

    # --- центроиды (в сжатом пространстве) и сохранение ---
    centroids, cluster_ids = compute_centroids(X_red, labels)
    os.makedirs(args.output, exist_ok=True)

    meta = {
        "method": args.method,
        "method_name": method_name,
        "reduce_method": args.reduce_method,
        "reduce_dim": args.reduce_dim,
        "params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "k_min": args.k_min,
            "k_max": args.k_max,
        },
    }
    save_state(args.output, centroids, cluster_ids, reducer, reducer_kind,
               meta, emb_paths, labels, image_paths)
    print(f"Сохранены centroids.npy, reducer.joblib и model.json в {args.output}")

    # --- summary.csv ---
    summary_path = os.path.join(args.output, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "count", "mean_intra_cosine_sim"])
        for c in cluster_ids:
            mask = labels == c
            members = X_red[mask]     # сжатое пространство
            if len(members) > 1:
                sim = members @ members.T
                iu = np.triu_indices(len(members), k=1)
                mean_sim = float(sim[iu].mean())
            else:
                mean_sim = 1.0
            w.writerow([c, int(mask.sum()), f"{mean_sim:.4f}"])
    print(f"Сохранён {summary_path}")

    # --- визуализация ---
    print("Построение 2D-проекции...")
    X2d, proj_name = reduce_2d_for_plot(X_red)
    scatter_path = os.path.join(args.output, "embedding_scatter.png")
    plot_scatter(X2d, labels, scatter_path,
                 f"{proj_name}, метод: {method_name}, "
                 f"кластеров: {n_clusters}, шум: {n_noise}")
    print(f"Сохранён {scatter_path}")

    print("Построение монтажей по кластерам...")
    for c in cluster_ids:
        idxs = [i for i, lb in enumerate(labels) if lb == c]
        imgs = [image_paths[i] for i in idxs if image_paths[i]]
        if not imgs:
            continue
        make_montage(
            imgs,
            title=f"Кластер {c} — {len(idxs)} снимков",
            out_path=os.path.join(args.output, f"cluster_{c:02d}_montage.png"),
            cols=args.montage_cols,
            max_items=args.montage_max,
        )

    # Шум отдельно — чтобы видеть, что туда попало
    if n_noise > 0:
        noise_idx = [i for i, lb in enumerate(labels) if lb == -1]
        noise_imgs = [image_paths[i] for i in noise_idx if image_paths[i]]
        if noise_imgs:
            make_montage(
                noise_imgs,
                title=f"Шум (noise) — {len(noise_idx)} снимков",
                out_path=os.path.join(args.output, "cluster_noise_montage.png"),
                cols=args.montage_cols,
                max_items=args.montage_max,
            )

    print("Готово.")


# ============================================================================
#  Режим предсказания
# ============================================================================
def predict_centroid(X_new_red, centroids, cluster_ids):
    if centroids.size == 0:
        return -np.ones(X_new_red.shape[0], dtype=int), np.zeros(X_new_red.shape[0])
    sims = X_new_red @ centroids.T
    best = np.argmax(sims, axis=1)
    labels = np.array([cluster_ids[b] for b in best], dtype=int)
    confs = sims[np.arange(len(best)), best]
    return labels, confs


def run_predict(args):
    if not os.path.isdir(args.model):
        print(f"[ERROR] Папка модели не найдена: {args.model}")
        sys.exit(2)
    if not os.path.isdir(args.input):
        print(f"[ERROR] Папка входных эмбеддингов не найдена: {args.input}")
        sys.exit(2)

    centroids, reducer, meta = load_state(args.model)
    cluster_ids = meta.get("cluster_ids", list(range(len(centroids))))
    print("=" * 60)
    print(f"Модель         : {args.model} ({meta.get('method_name')})")
    print(f"Reducer        : {meta.get('reducer', 'none')}")
    print(f"Кластеров      : {len(cluster_ids)}")
    print(f"Входные данные : {args.input}")
    print(f"Выход          : {args.output}")
    print("=" * 60)

    X, emb_paths = load_all_embeddings(args.input, recursive=args.recursive)
    if X.shape[0] == 0:
        print("Не найдено *_embeding.txt")
        sys.exit(2)
    print(f"Загружено эмбеддингов: {X.shape[0]}")

    # Проверяем исходную размерность по reducer / центроидам
    if reducer is not None:
        expected_in = getattr(reducer, "n_features_in_", None)
        if expected_in is not None and X.shape[1] != expected_in:
            print(f"[ERROR] Размерность эмбеддингов ({X.shape[1]}) "
                  f"не совпадает с reducer ({expected_in}).")
            sys.exit(2)
    elif centroids.size and X.shape[1] != centroids.shape[1]:
        print(f"[ERROR] Размерность эмбеддингов ({X.shape[1]}) "
              f"не совпадает с моделью ({centroids.shape[1]}).")
        sys.exit(2)

    X_red = apply_reducer(reducer, X)
    if centroids.size and X_red.shape[1] != centroids.shape[1]:
        print(f"[ERROR] После reducer размерность {X_red.shape[1]} "
              f"не совпадает с центроидами {centroids.shape[1]}.")
        sys.exit(2)

    labels, confs = predict_centroid(X_red, centroids, cluster_ids)

    os.makedirs(args.output, exist_ok=True)
    out_csv = os.path.join(args.output, "predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "embedding", "cluster", "cosine_sim"])
        for ep, lb, cf in zip(emb_paths, labels, confs):
            img = find_image_for_embedding(ep)
            w.writerow([img or "", os.path.basename(ep), int(lb), f"{float(cf):.4f}"])
    print(f"Сохранён {out_csv}")

    if args.make_montages:
        print("Построение монтажей...")
        for c in sorted(set(int(l) for l in labels)):
            idxs = [i for i, lb in enumerate(labels) if lb == c]
            imgs = [find_image_for_embedding(emb_paths[i]) for i in idxs]
            imgs = [p for p in imgs if p]
            if not imgs:
                continue
            make_montage(
                imgs,
                title=f"Кластер {c} — {len(idxs)} снимков (predict)",
                out_path=os.path.join(args.output, f"cluster_{c:02d}_montage.png"),
                cols=args.montage_cols,
                max_items=args.montage_max,
            )

    print("Готово.")


# ============================================================================
#  CLI
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Кластеризация эмбеддингов DINOv2 + сохранение модели."
    )
    ap.add_argument("-i", "--input",  default=CONFIG["input_dir"])
    ap.add_argument("-o", "--output", default=CONFIG["output_dir"])
    ap.add_argument("--method", choices=["auto", "hdbscan", "kmeans"],
                    default=CONFIG["method"])
    ap.add_argument("-k", "--n-clusters", type=int, default=CONFIG["n_clusters"])
    ap.add_argument("--k-min", type=int, default=CONFIG["k_min"])
    ap.add_argument("--k-max", type=int, default=CONFIG["k_max"])
    ap.add_argument("--min-cluster-size", type=int,
                    default=CONFIG["min_cluster_size"])
    ap.add_argument("--min-samples", type=int, default=CONFIG["min_samples"])
    ap.add_argument("--recursive", action="store_true",
                    default=CONFIG["recursive"])
    ap.add_argument("--montage-cols", type=int, default=CONFIG["montage_cols"])
    ap.add_argument("--montage-max", type=int, default=CONFIG["montage_max"])

    # Снижение размерности
    ap.add_argument("--reduce-method", choices=["none", "pca", "umap"],
                    default=CONFIG["reduce_method"])
    ap.add_argument("--reduce-dim", type=int, default=CONFIG["reduce_dim"])

    # Predict
    ap.add_argument("--predict", action="store_true", default=CONFIG["predict"])
    ap.add_argument("-m", "--model", default=CONFIG["model_dir"])
    ap.add_argument("--make-montages", action="store_true",
                    default=CONFIG["make_montages"])
    return ap.parse_args()


def main():
    args = parse_args()
    if args.predict:
        if not args.model:
            print("[ERROR] Для --predict укажите -m <папка модели>.")
            sys.exit(2)
        run_predict(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()