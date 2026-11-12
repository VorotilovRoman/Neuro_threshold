# analysis.py (исправленная версия для методов с малым количеством данных)


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils_new.autoreseach_utils.prepare import add_meanap_metrics


def distance_between_presets(p1, p2):
    """Расстояние между двумя пресетами (Series). Учитывает метод, cluster, cluster_co."""
    method_dist = 0 if p1["method"] == p2["method"] else 1.0
    cluster_dist = 0 if p1["cluster"] == p2["cluster"] else 1.0
    cluster_co_dist = 0 if p1["cluster_co"] == p2["cluster_co"] else 1.0
    total = 0.4 * method_dist + 0.3 * cluster_dist + 0.3 * cluster_co_dist
    return total


def diverse_top_k(df, k=5, score_col="pred_f1", candidate_pool_size=None):
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


def analyze(
    df,
    output_dir=None,
    skip_metrics=False,
    iou_thresholds=None,
    use_diversity=True,
    use_iou_penalty=False,
    penalty_alpha=0.5,
    per_method=True,
    cluster_col="cluster",
    cluster_co_col="cluster_co",
):
    """Rank exact presets. Never pair a group's best score with another preset."""
    from .reports import preset_from_row

    if df is None or df.empty:
        return pd.DataFrame()
    if not skip_metrics:
        df, _ = add_meanap_metrics(df, iou_thresholds=iou_thresholds)
    for _, row in df.iterrows():
        preset_from_row(row)
    # Mean performance across images for exactly the same preset.
    representatives = df.drop_duplicates("preset_json").set_index("preset_json").copy()
    scores = df.groupby("preset_json")["mean_f1"].mean()
    representatives["actual_f1"] = scores
    representatives["pred_f1"] = scores
    representatives["cluster"] = representatives.get(cluster_col, 0)
    representatives["cluster_co"] = representatives.get(cluster_co_col, 0)
    if use_iou_penalty:
        representatives["pred_f1"] *= (
            df.groupby("preset_json")["avg_iou_tp"].mean().clip(lower=0)
            ** penalty_alpha
        )
    # A descriptive regression estimate, not a held-out quality metric.
    if len(df) >= 10:
        import json

        features = pd.DataFrame(
            [
                dict(
                    json.loads(row["params"]),
                    close_px=row["close_px"],
                    open_px=row["open_px"],
                    erode_px=row["erode_px"],
                    dilate_px=row["dilate_px"],
                    invert=int(row["invert"]),
                    method=row["method"],
                    shape=json.loads(row["preset_json"])["morphology"]["shape"],
                    order=",".join(
                        op["operation"]
                        for op in json.loads(row["preset_json"])["morphology"][
                            "operations"
                        ]
                    ),
                    mode=json.loads(row["preset_json"])["annotation_mode"],
                    hull=int(json.loads(row["preset_json"])["use_hull"]),
                )
                for _, row in df.iterrows()
            ],
            index=df.index,
        )
        x = pd.get_dummies(features).fillna(0)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        target = df["mean_f1"]
        if use_iou_penalty:
            target = target * df["avg_iou_tp"].clip(lower=0) ** penalty_alpha
        model.fit(x, target)
        predicted = (
            pd.Series(model.predict(x), index=df["preset_json"]).groupby(level=0).mean()
        )
        representatives["pred_f1"] = predicted
    representatives = representatives.reset_index()
    groups = (
        representatives.groupby("method") if per_method else [("all", representatives)]
    )
    results = []
    for _, group in groups:
        group = group.sort_values("pred_f1", ascending=False)
        top = diverse_top_k(group, k=5) if use_diversity else group.head(5)
        top = top.copy()
        top["rank"] = range(1, len(top) + 1)
        results.append(top)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def global_top_diverse(
    df,
    k=10,
    score_col="actual_f1",
    method_col="method",
    params_col="params",
    invert_col="invert",
    close_col="close_px",
    open_col="open_px",
):
    """
    Жадный выбор топ-k разнообразных пресетов из глобального датасета.
    Использует distance_between_presets для оценки разнообразия.
    Требует наличия колонок 'cluster' и 'cluster_co' в df.
    """
    df = df.copy()
    if "preset_json" in df:
        means = df.groupby("preset_json")[score_col].mean()
        df = df.drop_duplicates("preset_json").set_index("preset_json")
        df[score_col] = means
        df = df.reset_index()
    for column in ("cluster", "cluster_co"):
        if column not in df:
            df[column] = 0
    if len(df) <= k:
        return df.copy()

    required = ["cluster", "cluster_co", method_col, score_col]
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
            best_idx = max(
                range(len(scores)), key=lambda i: (scores[i][0], scores[i][1])
            )
            selected.append(candidates.iloc[best_idx])
            candidates = candidates.drop(candidates.index[best_idx])

    result = pd.DataFrame(selected).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result
