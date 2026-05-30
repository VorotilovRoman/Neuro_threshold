from import_libs_internal import *
# ========== Вспомогательные функции для вложенности ==========

def contains_center(box1, box2):
    """
    Проверяет, лежит ли центр box2 внутри box1.
    box1, box2: [x1, y1, x2, y2]
    """
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return (box1[0] <= cx2 <= box1[2]) and (box1[1] <= cy2 <= box1[3])


def boxes_contain_each_other(box1, box2):
    """
    Проверяет, содержит ли один бокс другой (центр одного внутри другого).
    Симметричная проверка.
    """
    return contains_center(box1, box2) or contains_center(box2, box1)


# ========== Классический Non-Maximum Suppression (NMS) ==========

def non_max_suppression(boxes, scores, iou_threshold=0.5, conf_threshold=0.1):
    """
    Классический NMS: сортировка по уверенности, выбор лучшего, подавление по IoU.

    Параметры:
        boxes: list of [x1, y1, x2, y2]
        scores: list of raw confidence scores
        iou_threshold: порог IoU для подавления
        conf_threshold: минимальная уверенность для включения

    Возвращает:
        kept_boxes: list of объединённых боксов
        kept_scores: list of итоговых уверенностей
    """
    # Фильтрация по уверенности
    mask = np.array(scores) >= conf_threshold
    boxes = [boxes[i] for i in range(len(boxes)) if mask[i]]
    scores = [scores[i] for i in range(len(scores)) if mask[i]]
    if not boxes:
        return [], []

    # Сортировка по убыванию уверенности
    order = np.argsort(scores)[::-1]
    boxes = [boxes[i] for i in order]
    scores = [scores[i] for i in order]

    kept_boxes = []
    kept_scores = []

    while boxes:
        # Выбираем лучший (первый)
        best_box = boxes.pop(0)
        best_score = scores.pop(0)
        kept_boxes.append(best_box)
        kept_scores.append(best_score)

        # Подавляем остальные, которые сильно пересекаются с лучшим
        remaining_boxes = []
        remaining_scores = []
        for box, score in zip(boxes, scores):
            if iou_xyxy(best_box, box) < iou_threshold:
                remaining_boxes.append(box)
                remaining_scores.append(score)
        boxes = remaining_boxes
        scores = remaining_scores

    return kept_boxes, kept_scores


# ========== Weighted Boxes Fusion (WBF) ==========


def weighted_boxes_fusion(boxes, scores, weights, iou_threshold=0.5, conf_threshold=0.1,
                          use_containment=False, verbose=False):
    """
    Weighted Boxes Fusion (WBF) – взвешенное усреднение координат и уверенностей.
    """
    weighted_scores = [s * w for s, w in zip(scores, weights)]
    order = np.argsort(weighted_scores)[::-1]
    boxes = [boxes[i] for i in order]
    scores = [scores[i] for i in order]
    weights = [weights[i] for i in order]

    used = [False] * len(boxes)
    fused_boxes = []
    fused_scores = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        group_indices = [i]
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            iou_val = iou_xyxy(boxes[i], boxes[j])
            if iou_val >= iou_threshold or (use_containment and boxes_contain_each_other(boxes[i], boxes[j])):
                group_indices.append(j)
                used[j] = True
        used[i] = True

        if verbose:
            print(f"\nGroup {len(fused_boxes)+1}: indices {group_indices}")
            for idx in group_indices:
                print(f"  Box {idx}: box={boxes[idx]}, score={scores[idx]:.4f}, weight={weights[idx]:.4f}")

        total_weight = sum(weights[idx] for idx in group_indices)
        if total_weight == 0:
            continue

        x1 = sum(weights[idx] * boxes[idx][0] for idx in group_indices) / total_weight
        y1 = sum(weights[idx] * boxes[idx][1] for idx in group_indices) / total_weight
        x2 = sum(weights[idx] * boxes[idx][2] for idx in group_indices) / total_weight
        y2 = sum(weights[idx] * boxes[idx][3] for idx in group_indices) / total_weight
        conf = sum(weights[idx] * scores[idx] for idx in group_indices) / total_weight

        if verbose:
            print(f"  Fused box: [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}], confidence={conf:.4f}")

        if conf >= conf_threshold:
            fused_boxes.append([x1, y1, x2, y2])
            fused_scores.append(conf)

    return fused_boxes, fused_scores



# ========== Основная функция ансамбля с выбором метода ==========

def ensemble_detections(detections_list, weights, iou_threshold=0.5, conf_threshold=0.1,
                        use_containment=False, method='wbf', verbose=False):
    """
    Объединяет детекции из нескольких источников с помощью выбранного метода.

    Параметры:
        detections_list: список списков, каждый элемент – список детекций в формате (cls, cx, cy, w, h, score)
        weights: список весов для каждого источника
        iou_threshold: порог IoU для группировки/подавления
        conf_threshold: порог отбора итоговых детекций
        use_containment: если True, объединять также вложенные боксы (только для WBF)
        method: 'nms' – классический NMS, 'wbf' – Weighted Boxes Fusion
        verbose: если True, выводит подробности в консоль

    Возвращает:
        result_detections: список детекций в формате (cls, cx, cy, w, h, conf)
    """
    all_boxes = []
    all_scores = []
    all_weights = []
    all_classes = []
    source_indices = []

    for src_idx, detections in enumerate(detections_list):
        weight = weights[src_idx]
        for det in detections:
            if len(det) == 6:
                cls, cx, cy, w, h, score = det
            else:
                cls, cx, cy, w, h = det[:5]
                score = 1.0
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h)
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)
            all_weights.append(weight)
            all_classes.append(cls)
            source_indices.append(src_idx)

    if verbose:
        print("\n=== Ensemble Input ===")
        for i, (box, score, weight, src) in enumerate(zip(all_boxes, all_scores, all_weights, source_indices)):
            print(f"  Box {i}: src={src}, box={box}, score={score:.4f}, weight={weight:.4f}, weighted_conf={score * weight:.4f}")

    if method == 'nms':
        if use_containment and verbose:
            print("\n⚠️  WARNING: 'use_containment' is ignored when method='nms' (only IoU is used).")
        fused_boxes, fused_scores = non_max_suppression(
            all_boxes, all_scores,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold
        )
    else:  # 'wbf'
        fused_boxes, fused_scores = weighted_boxes_fusion(
            all_boxes, all_scores, all_weights,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            use_containment=use_containment,
            verbose=verbose
        )

    if verbose:
        print("\n=== Ensemble Output ===")
        for i, (box, conf) in enumerate(zip(fused_boxes, fused_scores)):
            cx, cy, w, h = xyxy_to_yolo(box[0], box[1], box[2], box[3])
            print(f"  Detection {i}: box={box}, conf={conf:.4f} -> YOLO: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")

    result_detections = []
    for box, conf in zip(fused_boxes, fused_scores):
        cx, cy, w, h = xyxy_to_yolo(box[0], box[1], box[2], box[3])
        result_detections.append((0, cx, cy, w, h, conf))
    return result_detections




# ========== Дополнительные функции для работы с пресетами и полным ансамблем ==========

def extract_threshold_params(preset, method):
    """
    Извлекает параметры для apply_threshold_method из плоского словаря пресета.
    """
    params = {}
    if method == "Simple Threshold":
        params['threshold'] = preset.get('threshold', 128)
    elif method in ["Adaptive Mean", "Adaptive Gauss"]:
        params['window'] = preset.get('window', 25)
        params['c'] = preset.get('c', 3)
    elif method == "Niblack":
        params['window'] = preset.get('window', 25)
        params['k'] = preset.get('k', -0.2)
    elif method == "Sauvola":
        params['window'] = preset.get('window', 25)
        params['k'] = preset.get('k', 0.2)
        params['r'] = preset.get('r', 128)
    elif method == "ISODATA":
        params['init'] = preset.get('init', 128)
    elif method == "Background Symmetry":
        params['excess'] = preset.get('excess', 0.2)
    elif method == "Row Adaptive":
        params['window'] = preset.get('window', 50)
        params['k'] = preset.get('k', 0.5)
    return params


def get_detections_from_preset(image, gray, preset, confidence=0.7):
    """
    Применяет пресет к изображению и возвращает список детекций.
    Каждая детекция: (class, cx, cy, w, h, confidence)
    class всегда 0 (можно изменить при необходимости).
    confidence используется как порог отбора, но для пресетов все найденные объекты получают confidence = 1.0.
    """
    method = preset.get('method', 'Simple Threshold')
    params = extract_threshold_params(preset, method)

    # 1. Бинаризация
    binary, _ = apply_threshold_method(gray, method, params)

    # 2. Принудительная инверсия
    binary = cv2.bitwise_not(binary)

    # 3. Морфология
    close_factor = preset.get('close_factor', 0.0)
    open_factor = preset.get('open_factor', 0.0)
    kernel_shape = preset.get('kernel_shape', 'Rectangle')
    processed = apply_morphology(binary, close_factor, open_factor, kernel_shape, gray.shape)

    # 4. Опциональная инверсия результата
    if preset.get('invert', False):
        processed = cv2.bitwise_not(processed)

    # 5. Выделение объектов в зависимости от draw_mode
    draw_mode = preset.get('draw_mode', 'Segmentation (Polygon)')
    use_hull = preset.get('use_hull', False)

    if draw_mode == "None":
        objects = []
    elif draw_mode == "Segmentation (Polygon)":
        _, objects = segment_contours(image.copy(), processed, use_hull, draw=False)
    elif draw_mode == "Bounding Box (Detect)":
        _, objects = segment_projections(image.copy(), processed, draw=False)
    elif draw_mode == "OBB (Oriented Box)":
        _, objects = segment_min_area_rect(image.copy(), processed, use_hull, draw=False)
    else:
        objects = []

    # 6. Преобразование в список детекций (class, cx, cy, w, h, conf)
    detections = []
    h_img, w_img = image.shape[:2]
    for obj in objects:
        # Определяем тип объекта
        if isinstance(obj, (tuple, list)) and len(obj) >= 2 and obj[0] in ('detect', 'obb', 'segment'):
            typ = obj[0]
            if typ == 'detect':
                _, cls, cx, cy, w, h = obj
                # Уверенность = 1.0 (или можно вычислить на основе площади, но пока 1.0)
                detections.append((int(cls), float(cx), float(cy), float(w), float(h), 1.0))
            elif typ == 'obb' or typ == 'segment':
                # Для OBB и сегментации можно преобразовать в bounding box для упрощения
                # Альтернативно: можно игнорировать, если ансамбль работает только с detect
                # Получим охватывающий прямоугольник из точек
                _, cls, points = obj
                xs = [points[i] for i in range(0, len(points), 2)]
                ys = [points[i + 1] for i in range(0, len(points), 2)]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                detections.append((int(cls), float(cx), float(cy), float(w), float(h), 1.0))
        else:
            # Старый формат (x, y, w, h) в пикселях
            if len(obj) == 4:
                x, y, w_px, h_px = obj
                cx = (x + w_px / 2) / w_img
                cy = (y + h_px / 2) / h_img
                w_norm = w_px / w_img
                h_norm = h_px / h_img
                detections.append((0, cx, cy, w_norm, h_norm, 1.0))
            # len(obj) == 5 для OBB в пикселях – можно аналогично

    # Применяем порог уверенности (отбрасываем детекции с conf < confidence)
    # В нашем случае conf=1.0, поэтому все сохранятся, если confidence <= 1.0
    # Но для единообразия:
    result = [d for d in detections if d[5] >= confidence]
    return result

def run_ensemble(image, gray, yolo_detections, yolo_weight,
                 presets_with_weights, iou_threshold, conf_threshold,
                 use_containment=False, method='wbf', verbose=True):
    """
    Выполняет полный ансамбль: собирает детекции из YOLO и всех пресетов,
    взвешивает их и применяет выбранный метод объединения (NMS или WBF).

    Параметры:
        image: исходное изображение
        gray: grayscale версия
        yolo_detections: список детекций YOLO (cls, cx, cy, w, h, score)
        yolo_weight: вес YOLO
        presets_with_weights: список кортежей (preset_dict, weight, preset_confidence)
        iou_threshold: порог IoU
        conf_threshold: порог уверенности
        use_containment: объединять также вложенные боксы (только для WBF)
        method: 'nms' или 'wbf'
        verbose: печатать отладочную информацию
    """
    detections_list = [yolo_detections]
    weights = [yolo_weight]

    for preset, weight, preset_conf in presets_with_weights:
        preset_dets = get_detections_from_preset(image, gray, preset, confidence=preset_conf)
        detections_list.append(preset_dets)
        weights.append(weight)
        if verbose:
            print(f"Пресет: {len(preset_dets)} объектов, вес {weight:.3f}, уверенность {preset_conf:.2f}")

    final_detections = ensemble_detections(
        detections_list, weights,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        use_containment=use_containment,
        method=method,
        verbose=verbose
    )
    return final_detections