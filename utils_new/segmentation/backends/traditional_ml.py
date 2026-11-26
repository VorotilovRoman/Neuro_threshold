"""Pixel/superpixel features and classical estimator training."""

from importlib.util import find_spec

import cv2
import numpy as np
from skimage import feature, segmentation
from sklearn.cluster import KMeans, MeanShift
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

XGBOOST_AVAILABLE = find_spec("xgboost") is not None

def extract_pixel_features(image, use_intensity=True, use_texture=False, use_spatial=False):
    h, w = image.shape[:2]
    features = []
    if use_intensity:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        features.append(gray.ravel())
    if use_texture:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        features.append(lbp.ravel())
    if use_spatial:
        h, w = image.shape[:2]
        x_coords = np.tile(np.arange(w), h) / w
        y_coords = np.repeat(np.arange(h), w) / h
        features.append(x_coords)
        features.append(y_coords)
    if not features:
        raise ValueError("At least one feature type must be selected")
    X = np.column_stack(features)
    return X

def extract_superpixel_features(image, segments, use_intensity=True, use_texture=False, use_spatial=False):
    h, w = image.shape[:2]
    n_segments = len(np.unique(segments))
    features = []
    if use_intensity:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean_intensity = np.zeros(n_segments)
        for seg_id in range(n_segments):
            mean_intensity[seg_id] = np.mean(gray[segments == seg_id])
        features.append(mean_intensity)
    if use_texture:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        mean_lbp = np.zeros(n_segments)
        for seg_id in range(n_segments):
            mean_lbp[seg_id] = np.mean(lbp[segments == seg_id])
        features.append(mean_lbp)
    if use_spatial:
        x_coords = np.zeros(n_segments)
        y_coords = np.zeros(n_segments)
        for seg_id in range(n_segments):
            ys, xs = np.where(segments == seg_id)
            if len(xs) > 0:
                x_coords[seg_id] = np.mean(xs) / w
                y_coords[seg_id] = np.mean(ys) / h
        features.append(x_coords)
        features.append(y_coords)
    if not features:
        raise ValueError("At least one feature type must be selected")
    X = np.column_stack(features)
    return X

def train_kmeans(X, n_clusters=3):
    if n_clusters > min(len(X), len(np.unique(X, axis=0))):
        raise ValueError("K превышает число различных обучающих образцов.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans

def train_meanshift(X):
    ms = MeanShift(bandwidth=None, bin_seeding=True)
    ms.fit(X)
    return ms

def train_svm(X, y, kernel='rbf', C=10):
    svm = SVC(kernel=kernel, C=C, random_state=42)
    svm.fit(X, y)
    return svm

def train_random_forest(X, y, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf

def train_xgboost(X, y, n_estimators=100):
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost не установлен. Обновите базовые зависимости: python install.py")
    try:
        import xgboost as xgb
    except (ImportError, OSError) as error:
        raise ImportError("XGBoost установлен, но не загружается. Выполните python dependency_health.py --load") from error
    model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def train_decision_tree(X, y):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    return dt



def create_superpixels(image, superpixel_size=50):
    h, w = image.shape[:2]
    if superpixel_size <= 0:
        raise ValueError("Superpixel size должен быть положительным.")
    n_segments = min(h * w, max(1, round(h * w / superpixel_size ** 2)))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    segments = segmentation.slic(rgb, n_segments=n_segments, compactness=10, start_label=0,
                                 channel_axis=-1 if image.ndim == 3 else None)
    return segments

def compute_elbow_kmeans(X, max_k=10):
    """
    Вычисляет инерцию (сумму квадратов расстояний) для K от 1 до max_k.
    Возвращает список инерций.
    """
    inertias = []
    max_k = min(max_k, len(X), len(np.unique(X, axis=0)))
    if max_k < 1:
        raise ValueError("Нет образцов для поиска K.")
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias

def find_optimal_k_elbow(inertias):
    """
    Находит оптимальное K как точку максимального изгиба (метод локтя).
    """
    if not inertias:
        raise ValueError("Нет значений инерции.")
    if len(inertias) < 3 or inertias[0] == inertias[-1]:
        return 1
    # Maximum distance from the normalized endpoint line. A heuristic, not validation.
    x = np.linspace(0, 1, len(inertias))
    y = (np.asarray(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
    return int(np.argmax(1 - x - y)) + 1
