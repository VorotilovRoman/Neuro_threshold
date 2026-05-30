# utils/prepare.py
# Обновлённая версия с использованием общих метрик из metrics_core

import pandas as pd
import numpy as np
from ast import literal_eval

# Импорт общих функций из metrics_core
from utils.metrics_core import (
    DEFAULT_IOU_THRESHOLDS,
    parse_objects,
    iou,
    compute_precision_recall_f1,
    compute_metrics as compute_metrics_core
)

# Импорты для кластеризации
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import re


# ---------- Вспомогательная функция для детального расчёта метрик (несколько порогов) ----------
def compute_metrics_detailed(row, iou_thresholds):
    """
    row: dict-like с ключами 'objects' и 'true_objects'
    iou_thresholds: список порогов, например [0.1,0.2,...,0.9]
    Возвращает:
        f1_list: список F1 для каждого порога
        tp_list, fp_list, fn_list: аналогично
        avg_iou_tp: средний IoU по всем TP (матчинг при пороге 0.5)
    """
    pred = row['objects']
    gt = row['true_objects']
    if not pred and not gt:
        # Идеальный случай: нет объектов и нет предсказаний -> F1=1 при любом пороге
        return [1.0] * len(iou_thresholds), [0] * len(iou_thresholds), [0] * len(iou_thresholds), [0] * len(
            iou_thresholds), 1.0
    if not pred:
        # Пустое предсказание -> все FN
        fn_cnt = len(gt)
        return [0.0] * len(iou_thresholds), [0] * len(iou_thresholds), [0] * len(iou_thresholds), [fn_cnt] * len(
            iou_thresholds), 0.0
    if not gt:
        # Нет истинных объектов -> все FP
        fp_cnt = len(pred)
        return [0.0] * len(iou_thresholds), [0] * len(iou_thresholds), [fp_cnt] * len(iou_thresholds), [0] * len(
            iou_thresholds), 0.0

    # Предварительно вычисляем матрицу IoU между всеми pred и gt
    iou_matrix = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou_matrix[i, j] = iou(p, g)

    f1_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    # Для avg_iou_tp используем матчинг при пороге 0.5 (классический)
    used_gt_for_avg = [False] * len(gt)
    tp_ious = []
    for i, p in enumerate(pred):
        best_iou = 0
        best_j = -1
        for j, g in enumerate(gt):
            if used_gt_for_avg[j]:
                continue
            iou_val = iou_matrix[i, j]
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou > 0.5:
            tp_ious.append(best_iou)
            used_gt_for_avg[best_j] = True
    avg_iou_tp = np.mean(tp_ious) if tp_ious else 0.0

    # Теперь для каждого порога считаем TP/FP/FN
    for thresh in iou_thresholds:
        used_gt = [False] * len(gt)
        tp = 0
        fp = 0
        for i, p in enumerate(pred):
            best_iou = 0
            best_j = -1
            for j, g in enumerate(gt):
                if used_gt[j]:
                    continue
                iou_val = iou_matrix[i, j]
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_iou > thresh:
                tp += 1
                used_gt[best_j] = True
            else:
                fp += 1
        fn = len(gt) - sum(used_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    return f1_list, tp_list, fp_list, fn_list, avg_iou_tp


def add_meanap_metrics(df, iou_thresholds=None):
    """
    Добавляет в DataFrame колонки:
        f1_50 (для совместимости),
        mean_f1 (среднее по всем порогам),
        avg_iou_tp (средний IoU для TP при пороге 0.5),
        а также для каждого порога f1_{thresh*100}.
    """
    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS
    if df is None or df.empty:
        return df, "Датасет пуст"
    required = ['objects', 'true_objects']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Колонка {col} отсутствует")
    df = df.copy()
    # Парсим объекты один раз
    df['objects_parsed'] = df['objects'].apply(parse_objects)
    df['true_objects_parsed'] = df['true_objects'].apply(parse_objects)

    f1_by_thresh = {th: [] for th in iou_thresholds}
    avg_iou_list = []

    for idx, row in df.iterrows():
        temp_row = {'objects': row['objects_parsed'], 'true_objects': row['true_objects_parsed']}
        f1_list, _, _, _, avg_iou = compute_metrics_detailed(temp_row, iou_thresholds)
        for i, th in enumerate(iou_thresholds):
            f1_by_thresh[th].append(f1_list[i])
        avg_iou_list.append(avg_iou)

    # Заполняем DataFrame
    for th in iou_thresholds:
        col_name = f'f1_{int(th * 100)}'
        df[col_name] = f1_by_thresh[th]
    # Для совместимости со старыми фильтрами, которые используют f1_50
    if 0.5 in iou_thresholds:
        df['f1_50'] = df[f'f1_50']
    else:
        # Если 0.5 нет в списке, добавим отдельно (но лучше передавать)
        pass
    # Средний F1 по всем порогам
    f1_values = np.array([f1_by_thresh[th] for th in iou_thresholds])
    df['mean_f1'] = np.mean(f1_values, axis=0)
    df['avg_iou_tp'] = avg_iou_list

    df.drop(['objects_parsed', 'true_objects_parsed'], axis=1, inplace=True)
    msg = f"Добавлены метрики meanAP (по порогам {iou_thresholds[0]}-{iou_thresholds[-1]}). Средний mean_f1 = {df['mean_f1'].mean():.4f}"
    return df, msg


# ---------- Остальные функции prepare.py (с использованием общих метрик по возможности) ----------

def compute_f1(df: pd.DataFrame, iou_threshold: float = 0.5) -> tuple[pd.DataFrame, str]:
    """Оставлена для обратной совместимости, использует старую метрику."""
    if df is None or df.empty:
        return df, "Датасет пуст, вычисление F1 невозможно."
    required = ['objects', 'true_objects']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")
    df['objects_parsed'] = df['objects'].apply(parse_objects)
    df['true_objects_parsed'] = df['true_objects'].apply(parse_objects)
    total = len(df)
    f1_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    for idx, row in df.iterrows():
        temp_row = {
            'objects': row['objects_parsed'],
            'true_objects': row['true_objects_parsed']
        }
        f1, tp, fp, fn = compute_metrics_core(temp_row, iou_threshold=iou_threshold)
        f1_list.append(f1)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
    df['f1'] = f1_list
    df['tp'] = tp_list
    df['fp'] = fp_list
    df['fn'] = fn_list
    df.drop(['objects_parsed', 'true_objects_parsed'], axis=1, inplace=True)
    msg = f"Вычислен F1 (IoU={iou_threshold}). Средний F1 = {df['f1'].mean():.4f}"
    return df, msg


def filter_by_f1_threshold(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'f1' not in df.columns:
        raise ValueError("Колонка 'f1' отсутствует. Сначала выполните compute_f1().")
    before = len(df)
    df_filtered = df[df['f1'] >= threshold].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр по F1 (порог {threshold:.2f}): удалено {before - after} строк. Осталось {after}."
    return df_filtered, msg


def sample_rows(df: pd.DataFrame, n: int, random_state: int = 42) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    total = len(df)
    if n >= total:
        return df, f"Выборка не выполнена: N={n} >= {total} строк."
    sampled = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    msg = f"Случайная выборка: {total} -> {len(sampled)} строк."
    return sampled, msg


def filter_full_image_object(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Удаляет строки, где среди предсказанных объектов есть объект, покрывающий почти всё изображение.
    Критерий: центр (0.5, 0.5) и размеры (w, h) близки к (1.0, 1.0) с допуском 0.01.
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'objects' not in df.columns:
        raise ValueError("Колонка 'objects' отсутствует.")

    def has_full_image_object(obj_str):
        try:
            obj_list = literal_eval(obj_str)
            for _, cx, cy, w, h in obj_list:
                if (abs(cx - 0.5) < 0.01 and abs(cy - 0.5) < 0.01 and
                        abs(w - 1.0) < 0.01 and abs(h - 1.0) < 0.01):
                    return True
        except:
            pass
        return False

    before = len(df)
    mask = ~df['objects'].apply(has_full_image_object)
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр 'объект на весь снимок' удалил {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_invalid_params(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    required = ['method', 'params']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")

    def has_valid_params(row):
        method = row['method']
        params_str = row['params']
        if method == 'Simple Threshold':
            try:
                int(params_str.split('=')[1])
                return True
            except:
                return False
        elif method in ['Adaptive Mean', 'Adaptive Gauss']:
            try:
                parts = params_str.split(',')
                int(parts[0].split('=')[1])
                int(parts[1].split('=')[1])
                return True
            except:
                return False
        elif method == 'Niblack':
            try:
                parts = params_str.split(',')
                int(parts[0].split('=')[1])
                float(parts[1].split('=')[1])
                return True
            except:
                return False
        elif method == 'Sauvola':
            try:
                parts = params_str.split(',')
                int(parts[0].split('=')[1])
                float(parts[1].split('=')[1])
                if len(parts) > 2:
                    int(parts[2].split('=')[1])
                return True
            except:
                return False
        elif method == 'ISODATA':
            try:
                int(params_str.split('=')[1])
                return True
            except:
                return False
        elif method == 'Background Symmetry':
            try:
                float(params_str.split('=')[1])
                return True
            except:
                return False
        elif method == 'Row Adaptive':
            try:
                parts = params_str.split(',')
                int(parts[0].split('=')[1])
                float(parts[1].split('=')[1])
                return True
            except:
                return False
        else:
            return True

    before = len(df)
    mask = df.apply(has_valid_params, axis=1)
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр некорректных параметров удалил {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_anomaly_count(df: pd.DataFrame, max_objects: int = 15) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    required = ['num_objects', 'num_true_objects']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    before = len(df)
    mask = ~((df['num_objects'] > max_objects) | (df['num_true_objects'] > max_objects))
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр аномального количества объектов (> {max_objects}) удалил {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_zero_true(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    required = ['num_objects', 'num_true_objects']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    before = len(df)
    mask = ~((df['num_objects'] == 0) & (df['num_true_objects'] >= 1))
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр 'num_objects=0 и true>=1' удалил {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_oversegmentation(df: pd.DataFrame, ratio: float = 2.0) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    required = ['num_objects', 'num_true_objects']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    before = len(df)
    mask = ~(df['num_objects'] > ratio * df['num_true_objects'])
    df_filtered = df[mask].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Фильтр 'pred > {ratio}*true' удалил {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return df, "Датасет пуст."
    before = len(df)
    df_filtered = df.drop_duplicates().reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Удаление полных дубликатов удалило {before - after} строк. Осталось {after}."
    return df_filtered, msg


def filter_no_true_objects(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Удаляет строки, в которых нет истинных объектов (num_true_objects == 0).
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'num_true_objects' not in df.columns:
        # Создаём колонку, если её нет
        if 'true_objects' in df.columns:
            df['num_true_objects'] = df['true_objects'].apply(parse_objects).apply(len)
        else:
            raise ValueError("Колонка 'true_objects' или 'num_true_objects' отсутствует")
    before = len(df)
    df_filtered = df[df['num_true_objects'] > 0].reset_index(drop=True)
    after = len(df_filtered)
    msg = f"Удалены строки без истинных объектов (num_true_objects == 0): удалено {before - after} строк. Осталось {after}."
    return df_filtered, msg


def aggregate_by_params(df: pd.DataFrame, group_by=None, score_col='mean_f1', top_k=1) -> tuple[pd.DataFrame, str]:
    """
    Агрегирует датасет, оставляя топ-K строк с наибольшим score_col в каждой группе.

    Параметры:
        df: исходный DataFrame
        group_by: список колонок для группировки. Если None, то автоматически:
                  если есть колонки 'cluster' и 'cluster_co' ->
                     ['method', 'cluster', 'cluster_co', 'invert']
                  иначе -> ['method', 'params', 'invert', 'close_factor', 'open_factor']
        score_col: колонка для сортировки (например, 'mean_f1')
        top_k: количество строк, оставляемых из каждой группы

    Возвращает:
        (new_df, сообщение)
    """
    if df is None or df.empty:
        return df, "Датасет пуст."

    if score_col not in df.columns:
        raise ValueError(f"Колонка '{score_col}' отсутствует.")

    if group_by is None:
        if 'cluster' in df.columns and 'cluster_co' in df.columns:
            group_by = ['method', 'cluster', 'cluster_co', 'invert']
        else:
            group_by = ['method', 'params', 'invert', 'close_factor', 'open_factor']

    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки для группировки: {missing}")

    before = len(df)
    df_sorted = df.sort_values(score_col, ascending=False)
    df_agg = df_sorted.groupby(group_by, group_keys=False).head(top_k).reset_index(drop=True)
    after = len(df_agg)

    msg = f"Агрегация по параметрам (группировка: {group_by}, топ-{top_k} по {score_col}): {before} -> {after} строк"
    return df_agg, msg


def save_cluster_report(df, method, cluster_ids, param_names, clusters_dir, method_name):
    """Сохраняет текстовый отчёт о кластерах для одного метода."""
    report_path = os.path.join(clusters_dir, f"clusters_{method_name.replace(' ', '_')}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Отчёт о кластерах для метода: {method}\n")
        f.write("=" * 60 + "\n\n")
        grouped = df[df['cluster'] != -1].groupby('cluster')
        for cluster_id in sorted(grouped.groups.keys()):
            cluster_df = grouped.get_group(cluster_id)
            f.write(f"Кластер {cluster_id}:\n")
            f.write(f"  Количество строк: {len(cluster_df)}\n")
            if 'mean_f1' in cluster_df.columns:
                f.write(f"  Средний mean_f1: {cluster_df['mean_f1'].mean():.4f}\n")
                f.write(f"  Медианный mean_f1: {cluster_df['mean_f1'].median():.4f}\n")
                f.write(f"  Максимальный mean_f1: {cluster_df['mean_f1'].max():.4f}\n")
                f.write(f"  Минимальный mean_f1: {cluster_df['mean_f1'].min():.4f}\n")
            if param_names:
                f.write("  Диапазоны параметров:\n")
                for param in param_names:
                    if param in cluster_df.columns:
                        min_val = cluster_df[param].min()
                        max_val = cluster_df[param].max()
                        f.write(f"    {param}: [{min_val:.2f}, {max_val:.2f}]\n")
            first_row = cluster_df.iloc[0]
            f.write(f"  Пример параметров: {first_row.get('params', 'N/A')}\n")
            f.write("\n")
        unclustered = df[df['cluster'] == -1]
        if len(unclustered) > 0:
            f.write("Строки без кластера (не удалось извлечь параметры):\n")
            f.write(f"  Количество: {len(unclustered)}\n")
            if 'mean_f1' in unclustered.columns:
                f.write(f"  Средний mean_f1: {unclustered['mean_f1'].mean():.4f}\n")
            f.write("\n")
    return report_path


def add_cluster_labels(df: pd.DataFrame, output_dir: str = None) -> tuple[pd.DataFrame, str]:
    """
    Для каждого метода выполняет кластеризацию параметров и добавляет колонку 'cluster'.
    Строки, для которых не удалось извлечь числовые параметры, получают cluster = -1.
    Визуализации кластеров и текстовые отчёты сохраняются в output_dir/clusters/.
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'method' not in df.columns or 'params' not in df.columns:
        raise ValueError("Для кластеризации необходимы колонки 'method' и 'params'")
    if output_dir is None:
        output_dir = "."
    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)
    df = df.copy()
    df['cluster'] = -1
    cluster_log = []
    grouped = df.groupby('method')
    for method, group in grouped:
        print(f"Обработка метода: {method}, строк: {len(group)}")
        param_list = []
        valid_indices = []
        param_names = set()
        for idx, row in group.iterrows():
            params_dict = extract_numeric_params(method, row['params'])
            if params_dict:
                param_list.append(params_dict)
                valid_indices.append(idx)
                param_names.update(params_dict.keys())
        if len(param_list) < 2:
            cluster_log.append(
                f"{method}: недостаточно данных для кластеризации (строк с параметрами: {len(param_list)}), все строки помечены cluster=-1")
            continue
        features_df = pd.DataFrame(param_list)
        features_df = features_df.fillna(0)
        param_names = list(param_names)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)
        n_clusters = min(10, max(2, len(param_list) // 5))
        if n_clusters > len(param_list):
            n_clusters = len(param_list)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        for i, idx in enumerate(valid_indices):
            df.loc[idx, 'cluster'] = labels[i]
        # Визуализация
        if X_scaled.shape[1] == 1:
            plt.figure(figsize=(8, 4))
            x_vals = X_scaled.flatten()
            plt.scatter(x_vals, np.zeros_like(x_vals), c=labels, cmap='tab10', alpha=0.7)
            centers_1d = scaler.inverse_transform(kmeans.cluster_centers_)
            for c in centers_1d:
                plt.axvline(x=c[0], color='red', linestyle='--', alpha=0.7)
            plt.yticks([])
            plt.title(f"{method} – кластеры (1D)")
        else:
            if X_scaled.shape[1] == 2:
                X_2d = X_scaled
                centers_2d = kmeans.cluster_centers_
            else:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X_scaled)
                centers_2d = pca.transform(kmeans.cluster_centers_)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', color='red', s=200, edgecolors='black')
            plt.colorbar(scatter, label='Cluster')
            plt.title(f"{method} – кластеры (PCA проекция)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plot_path = os.path.join(clusters_dir, f"clusters_{method.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()
        # Подготовка DataFrame для отчёта с извлечёнными параметрами
        temp_df = group.copy()
        for param in param_names:
            temp_df[param] = np.nan
        for idx, params_dict in zip(valid_indices, param_list):
            for pname, pval in params_dict.items():
                temp_df.loc[idx, pname] = pval
        temp_df['cluster'] = -1
        for i, idx in enumerate(valid_indices):
            temp_df.loc[idx, 'cluster'] = labels[i]
        report_path = save_cluster_report(temp_df, method, range(n_clusters), param_names, clusters_dir, method)
        cluster_log.append(
            f"{method}: {len(param_list)} строк → {n_clusters} кластеров. Отчёт: {os.path.basename(report_path)}")
    msg = "Кластеризация завершена.\n" + "\n".join(cluster_log)
    if cluster_log:
        msg += f"\nГрафики и отчёты сохранены в: {clusters_dir}"
    else:
        msg = "Не удалось выполнить кластеризацию ни для одного метода."
    return df, msg


def save_unique_params(df: pd.DataFrame, output_dir: str = None) -> tuple[pd.DataFrame, str]:
    """
    Для каждого метода сохраняет уникальные значения params в текстовый файл.
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'method' not in df.columns or 'params' not in df.columns:
        raise ValueError("Необходимы колонки 'method' и 'params'")
    if output_dir is None:
        output_dir = "."
    params_dir = os.path.join(output_dir, "unique_params")
    os.makedirs(params_dir, exist_ok=True)
    grouped = df.groupby('method')
    saved_files = []
    for method, group in grouped:
        unique_params = group['params'].dropna().unique()
        file_path = os.path.join(params_dir, f"{method.replace(' ', '_')}_unique_params.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Уникальные параметры для метода: {method}\n")
            f.write(f"Всего строк в методе: {len(group)}\n")
            f.write(f"Уникальных комбинаций параметров: {len(unique_params)}\n")
            f.write("=" * 60 + "\n\n")
            for i, params_str in enumerate(sorted(unique_params), 1):
                f.write(f"{i}. {params_str}\n")
        saved_files.append(file_path)
    msg = f"Сохранено {len(saved_files)} файлов с уникальными параметрами в папку: {params_dir}"
    return df, msg


def optimize_invert(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Для каждого метода оставляет только строки с лучшим значением invert (True/False)
    по среднему mean_f1. Если средние равны (с точностью 1e-6), оставляет более частый invert.
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'mean_f1' not in df.columns:
        raise ValueError("Колонка 'mean_f1' отсутствует. Сначала выполните add_meanap_metrics().")
    if 'method' not in df.columns or 'invert' not in df.columns:
        raise ValueError("Необходимы колонки 'method' и 'invert'.")

    before = len(df)
    rows_to_keep = []

    for method, group in df.groupby('method'):
        invert_values = group['invert'].unique()
        if len(invert_values) == 1:
            rows_to_keep.append(group)
            continue

        invert_stats = group.groupby('invert')['mean_f1'].agg(['mean', 'count'])
        best_invert = invert_stats['mean'].idxmax()
        # Если средние равны, выбираем более частый
        if len(invert_stats) == 2 and abs(invert_stats.iloc[0]['mean'] - invert_stats.iloc[1]['mean']) < 1e-6:
            best_invert = invert_stats['count'].idxmax()

        rows_to_keep.append(group[group['invert'] == best_invert])

    new_df = pd.concat(rows_to_keep, ignore_index=True)
    after = len(new_df)
    msg = f"Оптимизация инверсии: удалено {before - after} строк, осталось {after}."
    return new_df, msg


def cluster_open_close(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, str]:
    """
    Для каждого уникального сочетания (method, cluster) выполняет кластеризацию
    параметров close_factor и open_factor с помощью KMeans.
    Добавляет колонку 'cluster_co' с меткой кластера (0..n_clusters-1).
    Если в группе меньше n_clusters строк, каждой строке присваивается уникальный индекс (0..len-1).
    """
    if df is None or df.empty:
        return df, "Датасет пуст."
    if 'cluster' not in df.columns:
        raise ValueError("Колонка 'cluster' отсутствует. Сначала выполните add_cluster_labels().")
    required = ['method', 'close_factor', 'open_factor']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

    df = df.copy()
    df['cluster_co'] = -1  # временное значение

    grouped = df.groupby(['method', 'cluster'])
    for (method, cl), group in grouped:
        coords = group[['close_factor', 'open_factor']].values
        if len(coords) < n_clusters:
            # Каждой строке даём уникальный индекс
            for idx, original_idx in enumerate(group.index):
                df.loc[original_idx, 'cluster_co'] = idx
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)
        for original_idx, label in zip(group.index, labels):
            df.loc[original_idx, 'cluster_co'] = label

    msg = f"Кластеризация open/close завершена. Добавлена колонка 'cluster_co' (n_clusters={n_clusters} для каждой группы)."
    return df, msg


# ---------- Импорт вспомогательной функции extract_numeric_params из metrics_core ----------
from utils.metrics_core import extract_numeric_params