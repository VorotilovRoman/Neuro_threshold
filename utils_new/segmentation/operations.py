"""Numerical task bodies. Inputs are snapshots; no QWidget or QPixmap access."""

from concurrent.futures import CancelledError

import cv2
import numpy as np


def checkpoint(cancel):
    if cancel.is_set():
        raise CancelledError()


def interactive(method, image, gray, rect, fg, bg, params, cancel):
    from utils_ops import interactive_ops as ops

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
            distance_threshold=params["threshold"],
            min_area=params["min_distance"],
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
            **{key: value for key, value in params.items() if key != "sigma"},
        )
    if method == "Random Walker":
        markers = np.zeros(gray.shape, np.int32)
        markers[fg == 255] = 1
        markers[bg == 255] = 2
        return ops.random_walker_segmentation(image, markers, **params)
    if method == "Active Contours":
        return ops.active_contour_segmentation(gray, rect, **params)
    raise ValueError(f"Неизвестный интерактивный метод: {method}")


def features(image, spec):
    from utils_ops import traditional_ml_ops as ops

    flags = (spec["use_intensity"], spec["use_texture"], spec["use_spatial"])
    if spec["use_superpixels"]:
        segments = ops.create_superpixels(image, spec["superpixel_size"])
        return ops.extract_superpixel_features(image, segments, *flags), segments
    return ops.extract_pixel_features(image, *flags), None


def prepare_training(images, grays, spec, cancel):
    all_x, all_y = [], []
    for image, gray in zip(images, grays):
        checkpoint(cancel)
        if image is None:
            continue
        x, segments = features(image, spec)
        _, pseudo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    from utils_ops import traditional_ml_ops as ops

    checkpoint(cancel)
    metadata = dict(spec)
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
    from utils_ops import traditional_ml_ops as ops

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
    mask = model.predict(image, **parameters)
    checkpoint(cancel)
    return mask


def load_dl(name, path, device, cancel):
    from utils_ops.deep_learning_ops import create_segmentor

    checkpoint(cancel)
    model = create_segmentor(name, device=device)
    if name in ("U-Net", "DeepLabV3", "DeepLabV3+"):
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
