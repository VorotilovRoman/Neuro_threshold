# analysis.py (исправленная версия для методов с малым количеством данных)

from import_libs_external import *
from utils.metrics_core import DEFAULT_IOU_THRESHOLDS
from utils.prepare import add_meanap_metrics


def distance_between_presets(p1, p2):
    """Расстояние между двумя пресетами (Series). Учитывает метод, cluster, cluster_co."""
    method_dist = 0 if p1['method'] == p2['method'] else 1.0
    cluster_dist = 0 if p1['cluster'] == p2['cluster'] else 1.0
    cluster_co_dist = 0 if p1['cluster_co'] == p2['cluster_co'] else 1.0
    total = 0.4 * method_dist + 0.3 * cluster_dist + 0.3 * cluster_co_dist
    return total


def diverse_top_k(df, k=5, score_col='pred_f1', candidate_pool_size=None):
    """Жадный выбор топ-K разнообразных пресетов."""
    if len(df) <= k:
        return df
    if candidate_pool_size is not None and candidate_pool_size < len(df):
        candidates = df.nlargest(candidate_pool_size, score_col).copy()
    else:
        candidates = df.copy()
    selected = []
    indices = []
    while len(selected) < k and len(candidates) > 0:
        if not selected:
            best_idx = candidates.index[0]
            selected.append(candidates.loc[best_idx])
            indices.append(best_idx)
            candidates = candidates.drop(best_idx)
        else:
            min_dists = []
            for idx, row in candidates.iterrows():
                dists = [distance_between_presets(row, sel) for sel in selected]
                min_dists.append(min(dists))
            best_idx = candidates.index[np.argmax(min_dists)]
            selected.append(candidates.loc[best_idx])
            indices.append(best_idx)
            candidates = candidates.drop(best_idx)
    return pd.DataFrame(selected)


def analyze(df, output_dir=None, skip_metrics=False, iou_thresholds=None,
            use_diversity=True, use_iou_penalty=False, penalty_alpha=0.5,
            per_method=True, cluster_col='cluster', cluster_co_col='cluster_co'):
    """
    Анализ для каждого метода отдельно.
    Использует колонки cluster_col и cluster_co_col как признаки.
    Возвращает DataFrame с колонками:
        method, rank, params, invert, close_factor, open_factor, actual_f1, pred_f1, cluster, cluster_co
    """
    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS

    if not skip_metrics:
        if 'objects' not in df.columns or 'true_objects' not in df.columns:
            raise ValueError("Для пересчёта метрик необходимы колонки 'objects' и 'true_objects'")
        df, _ = add_meanap_metrics(df, iou_thresholds=iou_thresholds)
    else:
        if 'mean_f1' not in df.columns:
            raise ValueError("skip_metrics=True, но колонка 'mean_f1' отсутствует")

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    # Убедимся, что колонки кластеров есть
    if cluster_col not in df.columns:
        df[cluster_col] = 0
    if cluster_co_col not in df.columns:
        df[cluster_co_col] = 0

    # Целевая переменная
    if use_iou_penalty and 'avg_iou_tp' in df.columns:
        y = df['mean_f1'] * (df['avg_iou_tp'].clip(lower=0) ** penalty_alpha)
        target_name = f"weighted_f1 (mean_f1 * avg_iou_tp^{penalty_alpha})"
    else:
        y = df['mean_f1']
        target_name = "mean_f1"

    results = []
    methods = df['method'].unique()

    for method in methods:
        print(f"Processing method: {method}")
        df_method = df[df['method'] == method].copy()

        # Если данных мало (менее 10 строк) – просто берём топ-5 по mean_f1 без обучения модели
        if len(df_method) < 10:
            print(f"  {method}: мало данных ({len(df_method)} строк), берём топ-5 по mean_f1")
            # Группируем по уникальным комбинациям cluster и cluster_co (или по params, если кластеров нет)
            if cluster_col in df_method.columns and cluster_co_col in df_method.columns:
                group_cols = [cluster_col, cluster_co_col]
            else:
                group_cols = ['params']

            # Для каждой уникальной комбинации берём максимальный mean_f1
            unique_combos = df_method.groupby(group_cols).agg({
                'mean_f1': 'max',
                'params': 'first',
                'invert': 'first',
                'close_factor': 'first',
                'open_factor': 'first',
                cluster_col: 'first',
                cluster_co_col: 'first'
            }).reset_index(drop=True)

            unique_combos['method'] = method
            unique_combos['actual_f1'] = unique_combos['mean_f1']
            unique_combos['pred_f1'] = unique_combos['mean_f1']  # предсказание = реальное значение

            # Выбираем топ-5 (с учётом разнообразия, если включено)
            if use_diversity:
                top_method = diverse_top_k(unique_combos, k=5, score_col='mean_f1', candidate_pool_size=None)
            else:
                top_method = unique_combos.nlargest(5, 'mean_f1')

            # Формируем результат
            for rank, (_, row) in enumerate(top_method.iterrows(), 1):
                results.append({
                    'method': method,
                    'rank': rank,
                    'params': row['params'],
                    'invert': row['invert'],
                    'close_factor': row['close_factor'],
                    'open_factor': row['open_factor'],
                    'actual_f1': row['actual_f1'],
                    'pred_f1': row['pred_f1'],
                    'cluster': row[cluster_col] if cluster_col in row else -1,
                    'cluster_co': row[cluster_co_col] if cluster_co_col in row else -1
                })
            continue

        # Нормальный случай – достаточно данных для обучения
        # Признаки: cluster и cluster_co как категориальные
        X = pd.get_dummies(df_method[[cluster_col, cluster_co_col]], columns=[cluster_col, cluster_co_col])
        y_method = y.loc[df_method.index]

        # Обучаем модель
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y_method, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_test = rf.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            print(f"  {method}: MAE={mae:.4f}, R²={r2:.4f}")
        else:
            # Мало данных – используем все для обучения
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5, n_jobs=-1)
            rf.fit(X, y_method)

        # Получаем все уникальные комбинации (cluster, cluster_co) для этого метода
        unique_combos = df_method[[cluster_col, cluster_co_col]].drop_duplicates().reset_index(drop=True)
        X_all = pd.get_dummies(unique_combos, columns=[cluster_col, cluster_co_col])
        for col in X.columns:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[X.columns]

        unique_combos['pred_f1'] = rf.predict(X_all)

        # Добавим реальный средний mean_f1 для каждой комбинации
        actual = df_method.groupby([cluster_col, cluster_co_col])['mean_f1'].mean().reset_index()
        unique_combos = unique_combos.merge(actual, on=[cluster_col, cluster_co_col], how='left')
        unique_combos.rename(columns={'mean_f1': 'actual_f1'}, inplace=True)
        unique_combos['method'] = method

        # Выбираем топ-5 для метода
        if use_diversity:
            top_method = diverse_top_k(unique_combos, k=5, score_col='pred_f1', candidate_pool_size=None)
        else:
            top_method = unique_combos.nlargest(5, 'pred_f1')

        # Для каждой выбранной комбинации найдём пример реальных параметров (первую строку)
        sample_rows = []
        for _, row in top_method.iterrows():
            c_val = row[cluster_col]
            cc_val = row[cluster_co_col]
            example = df_method[(df_method[cluster_col] == c_val) & (df_method[cluster_co_col] == cc_val)].iloc[0]
            sample_rows.append({
                'method': method,
                'rank': len(sample_rows) + 1,
                'params': example['params'],
                'invert': example['invert'],
                'close_factor': example['close_factor'],
                'open_factor': example['open_factor'],
                'actual_f1': row['actual_f1'],
                'pred_f1': row['pred_f1'],
                'cluster': c_val,
                'cluster_co': cc_val
            })
        results.extend(sample_rows)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    return result_df[['method', 'rank', 'params', 'invert', 'close_factor',
                      'open_factor', 'actual_f1', 'pred_f1', 'cluster', 'cluster_co']]


def global_top_diverse(df, k=10, score_col='actual_f1', method_col='method',
                       params_col='params', invert_col='invert',
                       close_col='close_factor', open_col='open_factor'):
    """
    Жадный выбор топ-k разнообразных пресетов из глобального датасета.
    Использует distance_between_presets для оценки разнообразия.
    Требует наличия колонок 'cluster' и 'cluster_co' в df.
    """
    if len(df) <= k:
        return df.copy()

    required = ['cluster', 'cluster_co', method_col, score_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing}")

    candidates = df.sort_values(score_col, ascending=False).copy()
    selected = []

    while len(selected) < k and len(candidates) > 0:
        if not selected:
            best = candidates.iloc[[0]]
            selected.append(best.iloc[0])
            candidates = candidates.iloc[1:]
        else:
            scores = []
            for idx, row in candidates.iterrows():
                min_dist = min(distance_between_presets(row, sel) for sel in selected)
                scores.append((min_dist, row[score_col]))
            best_idx = max(range(len(scores)), key=lambda i: (scores[i][0], scores[i][1]))
            selected.append(candidates.iloc[best_idx])
            candidates = candidates.drop(candidates.index[best_idx])

    result = pd.DataFrame(selected).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)
    return result