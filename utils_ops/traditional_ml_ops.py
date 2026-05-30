import numpy as np
import cv2
from skimage import segmentation, feature
from sklearn.cluster import KMeans, MeanShift
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Для XGBoost (если установлен)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed, XGBoost model will be disabled")

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
        raise ImportError("XGBoost is not installed")
    model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

def train_decision_tree(X, y):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    return dt

def predict_pixelwise(model, X, original_shape):
    labels = model.predict(X)
    return labels.reshape(original_shape).astype(np.uint8)

def predict_superpixel(model, segments, X_superpixel):
    labels = model.predict(X_superpixel)
    result = np.zeros_like(segments, dtype=np.uint8)
    for seg_id, label in enumerate(labels):
        result[segments == seg_id] = 255 if label == 1 else 0
    return result

def create_superpixels(image, superpixel_size=50):
    h, w = image.shape[:2]
    n_segments = int((h * w) / (superpixel_size * superpixel_size))
    segments = segmentation.slic(image, n_segments=n_segments, compactness=10, start_label=0)
    return segments

def compute_elbow_kmeans(X, max_k=10):
    """
    Вычисляет инерцию (сумму квадратов расстояний) для K от 1 до max_k.
    Возвращает список инерций.
    """
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias

def find_optimal_k_elbow(inertias):
    """
    Находит оптимальное K как точку максимального изгиба (метод локтя).
    """
    if len(inertias) < 3:
        return 2
    # Вычисляем разности и нормализуем
    diffs = np.diff(inertias)
    # Ищем точку, где падение замедляется
    # Простой метод: выбираем K, где производная меньше среднего
    mean_diff = np.mean(diffs)
    for i, diff in enumerate(diffs):
        if diff < mean_diff and i >= 1:
            return i + 1  # K = i+1 (так как diff[i] между i и i+1)
    return 2