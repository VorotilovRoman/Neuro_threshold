"""Numerical task bodies. Inputs are snapshots; no QWidget or QPixmap access."""

from concurrent.futures import CancelledError

import cv2
import numpy as np


def checkpoint(cancel):
    if cancel.is_set():
        raise CancelledError()


def interactive(method, image, gray, rect, fg, bg, params, cancel):
    from utils_new.segmentation.backends import interactive as ops

    checkpoint(cancel)
    if method == "GrabCut":
        return ops.grabcut_segmentation(
            image,
            rect,
            None,
            fg,
            bg,
            iterations=params["iterations"],
            mode=params["mode"],
        )
    if method == "Watershed (marker)":
        return ops.watershed_segmentation(
            gray,
            distance_threshold=params.get("threshold", 50),
            min_distance=params.get("min_distance", 20),
            fg_scribbles=fg, bg_scribbles=bg, mode=params.get("mode", "automatic"),
        )
    if method == "Lazy Snapping":
        return ops.lazy_snapping(image, fg, bg, **params)
    if method == "SuperCut":
        return ops.supercut_segmentation(
            image,
            rect,
            **{key: value for key, value in params.items() if key != "color_sigma"},
        )
    if method == "OneCut":
        return ops.onecut_segmentation(
            image,
            rect,
            **params,
        )
    if method == "Random Walker":
        ops._seeds(fg, bg, gray.shape)
        markers = np.zeros(gray.shape, np.int32)
        markers[fg == 255] = 1
        markers[bg == 255] = 2
        return ops.random_walker_segmentation(image, markers, **params)
    if method == "Active Contours":
        return ops.active_contour_segmentation(gray, rect, **params)
    if method == "Morphological Chan–Vese":
        return ops.morphological_chan_vese_segmentation(gray, rect, **params)
    raise ValueError(f"Неизвестный интерактивный метод: {method}")


def features(image, spec):
    from utils_new.segmentation.backends import traditional_ml as ops

    flags = (spec["use_intensity"], spec["use_texture"], spec["use_spatial"])
    if spec["use_superpixels"]:
        segments = ops.create_superpixels(image, spec["superpixel_size"])
        return ops.extract_superpixel_features(image, segments, *flags), segments
    return ops.extract_pixel_features(image, *flags), None


def prepare_training(images, grays, spec, cancel, paths=None):
    all_x, all_y = [], []
    for index, (image, gray) in enumerate(zip(images, grays)):
        checkpoint(cancel)
        if image is None:
            continue
        x, segments = features(image, spec)
        if spec["model_type"] in ("K-Means Clustering", "MeanShift"):
            pseudo = np.zeros(gray.shape, np.uint8)  # clustering does not consume labels
        elif spec.get("training_source") == "Otsu (псевдометки)":
            _, pseudo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif spec.get("training_source") == "Ручная разметка YOLO":
            from utils_new.segmentation.training_labels import manual_target
            if paths is None or len(paths) != len(images):
                raise ValueError("Для ручного обучения нужны пути всех изображений.")
            pseudo = manual_target(paths[index], gray.shape) * 255
        else:
            raise ValueError("Укажите источник обучающей разметки: ручной YOLO или явные псевдометки Otsu.")
        if spec["use_superpixels"]:
            y = np.zeros(len(np.unique(segments)), np.int32)
            for segment_id in np.unique(segments):
                y[segment_id] = (
                    1 if np.mean(pseudo[segments == segment_id]) > 127 else 0
                )
        else:
            y = (pseudo.ravel() > 127).astype(np.int32)
        all_x.append(x)
        all_y.append(y)
    if not all_x:
        raise ValueError("Нет корректных изображений для обучения.")
    return np.vstack(all_x), np.hstack(all_y)


def train_prepared(x, y, spec, cancel):
    from utils_new.segmentation.backends import traditional_ml as ops

    checkpoint(cancel)
    metadata = dict(spec)
    if spec["model_type"] not in ("K-Means Clustering", "MeanShift") and len(np.unique(y)) < 2:
        raise ValueError("В обучающей разметке нужны и фон, и объект.")
    if spec["normalize_features"]:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        metadata["scaler"] = scaler
    name = spec["model_type"]
    if name == "K-Means Clustering":
        model = ops.train_kmeans(x, spec["n_clusters"])
    elif name == "MeanShift":
        model = ops.train_meanshift(x)
    elif name == "SVM (pixel-wise)":
        model = ops.train_svm(x, y, spec["kernel"], spec["C"])
    elif name == "Random Forest":
        model = ops.train_random_forest(x, y, spec["n_trees"])
    elif name == "XGBoost":
        model = ops.train_xgboost(x, y, spec["n_trees"])
    elif name == "Decision Tree":
        model = ops.train_decision_tree(x, y)
    else:
        raise ValueError(f"Неизвестная модель: {name}")
    checkpoint(cancel)
    return model, metadata


def predict_ml(image, gray, model, metadata, spec, cluster, cancel):

    checkpoint(cancel)
    feature_keys = ("use_intensity", "use_texture", "use_spatial", "use_superpixels", "normalize_features")
    if not metadata or any(key not in metadata or metadata[key] != spec[key] for key in feature_keys):
        raise ValueError("Признаки отличаются от обучения. Нажмите «Сбросить к настройкам модели» или переобучите модель.")
    if spec["use_superpixels"] and metadata.get("superpixel_size") != spec["superpixel_size"]:
        raise ValueError("Размер суперпикселей отличается от обучения. Сбросьте настройки модели.")
    if spec["normalize_features"] and "scaler" not in metadata:
        raise ValueError("В модели отсутствует обученный StandardScaler. Переобучите модель.")
    x, segments = features(image, spec)
    if metadata and "scaler" in metadata:
        x = metadata["scaler"].transform(x)
    labels = model.predict(x)
    if spec["use_superpixels"]:
        mask = np.zeros_like(segments, dtype=np.asarray(labels).dtype)
        for segment_id, label in enumerate(labels):
            mask[segments == segment_id] = label
    else:
        mask = labels.reshape(gray.shape)
    if not cluster:
        return mask
    unique = np.unique(mask)
    means = [float(np.mean(gray[mask == label])) for label in unique]
    return mask, unique, means


def optimal_k(images, spec, cancel):
    from utils_new.segmentation.backends import traditional_ml as ops

    values = []
    for image in images:
        checkpoint(cancel)
        if image is not None:
            values.append(features(image, spec)[0])
    if not values:
        raise ValueError("Нет корректных изображений.")
    return ops.find_optimal_k_elbow(
        ops.compute_elbow_kmeans(np.vstack(values), max_k=10)
    )


def predict_dl(model, image, parameters, overrides, cancel):
    checkpoint(cancel)
    metadata = getattr(model, "metadata", None)
    if metadata is not None:
        metadata.update(overrides)
    predict = getattr(model, 'predict_result', model.predict)
    mask = predict(image, **parameters)
    checkpoint(cancel)
    return mask


def load_dl(name, path, device, cancel):
    from utils_new.segmentation.backends.deep_learning import create_segmentor

    checkpoint(cancel)
    model = create_segmentor(name, device=device)
    if name in ("U-Net", "DeepLabV3", "DeepLabV3+", "SegFormer"):
        model.load(weights_path=path)
    else:
        model.load(model_path=path)
    checkpoint(cancel)
    return model


def load_ml(path, cancel):
    import joblib

    checkpoint(cancel)
    data = joblib.load(path)
    checkpoint(cancel)
    return (
        (data["model"], data.get("metadata", {}))
        if isinstance(data, dict) and "model" in data
        else (data, None)
    )
