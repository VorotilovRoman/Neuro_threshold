from import_libs_external import *

# ========== Константы ==========
DEFAULT_IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ========== Преобразования координат ==========
def yolo_to_xyxy(cx, cy, w, h):
    """
    Преобразует YOLO-координаты (нормализованные, 0..1) в абсолютные [x1, y1, x2, y2].
    Возвращает координаты также в диапазоне 0..1.
    """
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.clip([x1, y1, x2, y2], 0, 1)


def xyxy_to_yolo(x1, y1, x2, y2):
    """
    Преобразует абсолютные координаты [x1, y1, x2, y2] (0..1) в YOLO-формат (cx, cy, w, h).
    """
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, w, h


# ========== Функции IoU ==========
def iou(box1, box2):
    """
    Вычисляет IoU для двух боксов в YOLO-формате (cls, cx, cy, w, h).
    Все координаты нормализованы (0..1).
    """
    x1, y1, w1, h1 = box1[1], box1[2], box1[3], box1[4]
    x2, y2, w2, h2 = box2[1], box2[2], box2[3], box2[4]
    x1_min = x1 - w1/2
    x1_max = x1 + w1/2
    y1_min = y1 - h1/2
    y1_max = y1 + h1/2
    x2_min = x2 - w2/2
    x2_max = x2 + w2/2
    y2_min = y2 - h2/2
    y2_max = y2 + h2/2
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


def iou_xyxy(box1, box2):
    """
    Вычисляет IoU для двух боксов в формате [x1, y1, x2, y2] (абсолютные координаты).
    Возвращает значение в диапазоне [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    if union == 0:
        return 0.0
    return inter / union


# ========== Парсинг объектов ==========
def parse_objects(s):
    """
    Преобразует строковое представление списка объектов в список кортежей.
    Если строка пустая или невалидная, возвращает пустой список.
    """
    if pd.isna(s) or s == '' or s == '[]':
        return []
    try:
        return literal_eval(s)
    except Exception:
        return []


# ========== Precision, Recall, F1 ==========
def compute_precision_recall_f1(tp, fp, fn):
    """
    Вычисляет precision, recall и F1-меру по количеству TP, FP, FN.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Для одного изображения вычисляет F1, TP, FP, FN.
    pred_boxes и gt_boxes – списки в формате (cls, cx, cy, w, h).
    """
    if not pred_boxes and not gt_boxes:
        return 1.0, 0, 0, 0
    if not pred_boxes:
        return 0.0, 0, 0, len(gt_boxes)
    if not gt_boxes:
        return 0.0, len(pred_boxes), len(pred_boxes), 0

    used_gt = [False] * len(gt_boxes)
    tp, fp = 0, 0
    for p in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, g in enumerate(gt_boxes):
            if used_gt[i]:
                continue
            iou_val = iou(p, g)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i
        if best_iou > iou_threshold:
            tp += 1
            used_gt[best_idx] = True
        else:
            fp += 1
    fn = len(gt_boxes) - sum(used_gt)
    _, _, f1 = compute_precision_recall_f1(tp, fp, fn)
    return f1, tp, fp, fn


# ========== Average Precision (AP) и mAP ==========
def compute_ap(precision, recall):
    """
    Вычисляет Average Precision (AP) по кривой precision-recall.
    precision и recall – списки значений при разных порогах.
    """
    # Добавляем граничные точки
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Монотонное убывание precision
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    # Индексы, где recall меняется
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap


def compute_map(df, iou_thresholds=None):
    """
    Вычисляет mAP (mean Average Precision) для датасета.
    Упрощённая версия: возвращает средний mean_f1 по всем изображениям.
    Для полной mAP требуется группировка по классам и порогам.
    """
    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS
    # Используем существующую функцию add_meanap_metrics из prepare.py
    # Импорт внутри функции для избежания циклических зависимостей
    from .prepare import add_meanap_metrics
    df, _ = add_meanap_metrics(df, iou_thresholds)
    return df['mean_f1'].mean() if 'mean_f1' in df.columns else 0.0


# ========== Извлечение числовых параметров для кластеризации ==========
def extract_numeric_params(method: str, params_str: str) -> dict:
    """
    Извлекает числовые параметры из строки params (поддерживает кириллицу и латиницу).
    Используется для кластеризации.
    """
    if not params_str or pd.isna(params_str):
        return {}
    params_str = params_str.strip()
    result = {}
    if method in ("Adaptive Mean", "Adaptive Gauss"):
        win_match = re.search(r'(?:окно|window)\s*=\s*([+-]?\d+)', params_str, re.IGNORECASE)
        if win_match:
            result['window'] = int(win_match.group(1))
        c_match = re.search(r'[Cc]\s*=\s*([+-]?\d+)', params_str)
        if c_match:
            result['c'] = int(c_match.group(1))
    elif method == "Background Symmetry":
        ex_match = re.search(r'excess\s*=\s*([+-]?\d*\.?\d+)', params_str, re.IGNORECASE)
        if ex_match:
            result['excess'] = float(ex_match.group(1))
    elif method == "ISODATA":
        th_match = re.search(r'(?:начальный порог|init)\s*=\s*([+-]?\d+)', params_str, re.IGNORECASE)
        if th_match:
            result['init'] = int(th_match.group(1))
    elif method in ("Niblack", "Row Adaptive"):
        win_match = re.search(r'(?:окно|window)\s*=\s*([+-]?\d+)', params_str, re.IGNORECASE)
        if win_match:
            result['window'] = int(win_match.group(1))
        k_match = re.search(r'[kK]\s*=\s*([+-]?\d*\.?\d+)', params_str)
        if k_match:
            result['k'] = float(k_match.group(1))
    elif method == "Sauvola":
        win_match = re.search(r'(?:окно|window)\s*=\s*([+-]?\d+)', params_str, re.IGNORECASE)
        if win_match:
            result['window'] = int(win_match.group(1))
        k_match = re.search(r'[kK]\s*=\s*([+-]?\d*\.?\d+)', params_str)
        if k_match:
            result['k'] = float(k_match.group(1))
        r_match = re.search(r'[Rr]\s*=\s*([+-]?\d+)', params_str)
        if r_match:
            result['r'] = int(r_match.group(1))
    elif method == "Simple Threshold":
        th_match = re.search(r'(?:порог|threshold)\s*=\s*([+-]?\d+)', params_str, re.IGNORECASE)
        if th_match:
            result['threshold'] = int(th_match.group(1))
    return result