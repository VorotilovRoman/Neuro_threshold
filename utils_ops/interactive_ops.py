# interactive_ops.py
from import_libs_external import *

# Проверка наличия networkx для Lazy Snapping
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False
    print("Предупреждение: networkx не установлена. Lazy Snapping будет недоступен.")



def grabcut_segmentation(img, rect=None, mask=None, fg_scribbles=None, bg_scribbles=None,
                         iterations=5, mode="Both"):
    """
    GrabCut segmentation with mode selection.
    mode: "GC_INIT_WITH_RECT" (only rectangle), "GC_INIT_WITH_MASK" (only scribbles), "Both" (both)
    """


    h, w = img.shape[:2]
    if mask is None:
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    if fg_scribbles is not None:
        mask[fg_scribbles == 255] = cv2.GC_FGD
    if bg_scribbles is not None:
        mask[bg_scribbles == 255] = cv2.GC_BGD

    has_def_fg = np.any(mask == cv2.GC_FGD)
    has_def_bg = np.any(mask == cv2.GC_BGD)

    if mode == "GC_INIT_WITH_MASK" and (not has_def_fg or not has_def_bg):
        # fallback: использовать инициализацию с прямоугольником, если он есть
        if rect is not None:
            mode = "GC_INIT_WITH_RECT"
        else:
            raise ValueError("Для GrabCut с маской нужны и FG, и BG метки, либо прямоугольник.")

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if mode == "GC_INIT_WITH_RECT" and rect is not None:
        x1, y1, x2, y2 = rect
        rect_tuple = (x1, y1, x2 - x1, y2 - y1)
        cv2.grabCut(img, mask, rect_tuple, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    elif mode == "GC_INIT_WITH_MASK" and (fg_scribbles is not None or bg_scribbles is not None):
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)
    elif mode == "Both":
        if rect is not None:
            x1, y1, x2, y2 = rect
            rect_tuple = (x1, y1, x2 - x1, y2 - y1)
            cv2.grabCut(img, mask, rect_tuple, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
            if fg_scribbles is not None or bg_scribbles is not None:
                cv2.grabCut(img, mask, rect_tuple, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)
    else:
        # fallback: if nothing suitable, just run with mask if available
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_MASK)

    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return result


def watershed_segmentation(gray, distance_threshold=50, min_area=20):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, distance_threshold / 100.0 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Удаление мелких маркеров
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg, connectivity=8)
        cleaned = np.zeros_like(sure_fg, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        sure_fg = cleaned

    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    result = np.where(markers > 1, 255, 0).astype(np.uint8)
    return result


def random_walker_segmentation(img, markers, beta=130, mode='cg_j'):
    """
    Random Walker segmentation.
    img: BGR image
    markers: integer mask with different labels for foreground/background
    beta: edge weight parameter (default 130)
    mode: solver mode ('cg_j', 'cg_mg', 'bf')
    Returns: binary mask where markers == 1 are foreground
    """
    from skimage.segmentation import random_walker
    import warnings

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = random_walker(img, markers, beta=beta, mode=mode)

    result = np.where(labels == 1, 255, 0).astype(np.uint8)
    return result


def active_contour_segmentation(gray, rect, alpha=0.01, beta=0.1, gamma=0.01,
                                max_iter=250, convergence=0.001, smooth=True,
                                init_type="Ellipse"):
    """
    Active Contours (Snakes) segmentation.

    Args:
        gray: grayscale image
        rect: (x1, y1, x2, y2) bounding box
        alpha: elasticity parameter (tension)
        beta: rigidity parameter (bending)
        gamma: step size for evolution
        max_iter: maximum iterations
        convergence: convergence threshold
        smooth: apply Gaussian smoothing to image
        init_type: type of initial contour ("Rectangle", "Circle", "Ellipse")
    Returns:
        binary mask (0=bg, 255=fg)
    """
    from skimage.segmentation import active_contour

    x1, y1, x2, y2 = rect

    if smooth:
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 2)
    else:
        gray_blurred = gray

    gray_norm = cv2.normalize(gray_blurred.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = (x2 - x1) / 2.0
    ry = (y2 - y1) / 2.0

    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)

    if init_type == "Rectangle":
        rect_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        t = np.linspace(0, 1, n_points)
        snake_init = np.zeros((n_points, 2), dtype=np.float32)
        seg_len = n_points // 4
        for i in range(4):
            start = rect_pts[i]
            end = rect_pts[(i + 1) % 4]
            for j in range(seg_len):
                idx = i * seg_len + j
                if idx < n_points:
                    frac = j / seg_len
                    snake_init[idx] = start + frac * (end - start)
        for idx in range(4 * seg_len, n_points):
            snake_init[idx] = snake_init[4 * seg_len - 1]
    elif init_type == "Circle":
        radius = min(rx, ry)
        snake_init = np.array([[cx + radius * np.cos(t), cy + radius * np.sin(t)] for t in theta], dtype=np.float32)
    else:  # Ellipse
        snake_init = np.array([[cx + rx * np.cos(t), cy + ry * np.sin(t)] for t in theta], dtype=np.float32)

    snake = active_contour(
        gray_norm,
        snake_init,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=max_iter,
        convergence=convergence
    )

    h, w = gray.shape
    snake[:, 0] = np.clip(snake[:, 0], 0, w - 1)
    snake[:, 1] = np.clip(snake[:, 1], 0, h - 1)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [snake.astype(np.int32)], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask




def supercut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0, lambda_val=0.5, sigma_color=10.0):
    """
    SuperCut using graph cut (max-flow/min-cut) on superpixels.
    """
    from skimage.segmentation import slic, find_boundaries
    import networkx as nx

    # Приведение размеров к целым числам
    h, w = int(img.shape[0]), int(img.shape[1])
    n_segments = max(50, int((h * w) / (superpixel_size * superpixel_size)))

    # Сглаживание изображения
    img_smooth = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img
    segments = slic(img_smooth, n_segments=n_segments, compactness=compactness, start_label=0)

    x1, y1, x2, y2 = rect

    n_sp = len(np.unique(segments))
    sp_centers = np.zeros((n_sp, 2), dtype=np.float32)
    sp_colors = np.zeros((n_sp, 3), dtype=np.float32)
    sp_in_ratio = np.zeros(n_sp, dtype=np.float32)

    for seg_id in range(n_sp):
        mask = (segments == seg_id)
        if not np.any(mask):
            sp_in_ratio[seg_id] = 0.0
            continue
        coords = np.argwhere(mask)
        sp_centers[seg_id] = np.mean(coords, axis=0)[::-1]
        sp_colors[seg_id] = np.mean(img_smooth[mask], axis=0)

        ys, xs = coords[:, 0], coords[:, 1]
        inside = np.sum((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2))
        total = len(xs)
        sp_in_ratio[seg_id] = inside / total if total > 0 else 0.0

    # Построение графа соседства
    adj = [[] for _ in range(n_sp)]
    boundaries = find_boundaries(segments, mode='inner')
    bound_coords = np.argwhere(boundaries)
    for y, x in bound_coords:
        seg_id = segments[y, x]
        neighbors = np.unique(segments[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)])
        for nb in neighbors:
            if nb != seg_id and nb not in adj[seg_id]:
                adj[seg_id].append(nb)

    # Унарные стоимости (логарифмические)
    eps = 1e-6
    unary_fg = np.zeros(n_sp)   # стоимость отнесения к объекту
    unary_bg = np.zeros(n_sp)   # стоимость отнесения к фону
    for i in range(n_sp):
        r = sp_in_ratio[i]
        unary_fg[i] = -lambda_val * np.log(r + eps)
        unary_bg[i] = -lambda_val * np.log(1 - r + eps)

    # Парные стоимости (цветовое расстояние)
    def pairwise_cost(i, j):
        color_dist = np.linalg.norm(sp_colors[i] - sp_colors[j])
        return np.exp(- (color_dist ** 2) / (2 * sigma_color ** 2 + 1e-6))

    # ---- Построение графа для min-cut ----
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    G.add_node(source)
    G.add_node(sink)
    for i in range(n_sp):
        G.add_node(i)

    # Терминальные рёбра (унарные веса)
    for i in range(n_sp):
        G.add_edge(source, i, capacity=unary_bg[i])
        G.add_edge(i, sink, capacity=unary_fg[i])

    # Парные рёбра (между соседями)
    for i in range(n_sp):
        for j in adj[i]:
            if j <= i:
                continue
            w = pairwise_cost(i, j)
            G.add_edge(i, j, capacity=w)
            G.add_edge(j, i, capacity=w)

    # Вычисление минимального разреза
    try:
        cut_value, partition = nx.minimum_cut(G, source, sink)
        source_side, sink_side = partition
    except Exception as e:
        print(f"Ошибка при min-cut в SuperCut: {e}")
        return np.zeros_like(segments, dtype=np.uint8)

    # Формирование маски (размерность берётся из segments)
    result = np.zeros_like(segments, dtype=np.uint8)
    for sp_id in source_side:
        if isinstance(sp_id, int) and 0 <= sp_id < n_sp:
            result[segments == sp_id] = 255

    # Морфологическое сглаживание (опционально)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result

def onecut_segmentation(img, rect, superpixel_size=50, compactness=10, sigma=1.0,
                        spatial_weight=1.0, data_weight=1.0, color_sigma=10.0):
    """
    OneCut segmentation using superpixels and graph cut.
    - Uses color models from inside (foreground) and outside (background) bounding box.
    - Unary costs: data_weight * log probability from GMM color models.
    - Pairwise costs: spatial_weight * exp(-color_diff^2 / (2*color_sigma^2)).
    """
    from skimage.segmentation import slic, find_boundaries
    import networkx as nx
    from sklearn.mixture import GaussianMixture

    h, w = int(img.shape[0]), int(img.shape[1])
    x1, y1, x2, y2 = rect
    # Убедимся, что прямоугольник в пределах изображения
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Некорректный прямоугольник")

    # Сглаживание изображения
    img_smooth = cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img

    # Суперпиксели
    n_segments = max(50, int((h * w) / (superpixel_size * superpixel_size)))
    segments = slic(img_smooth, n_segments=n_segments, compactness=compactness,
                    start_label=0, convert2lab=False)

    n_sp = len(np.unique(segments))
    sp_colors = np.zeros((n_sp, 3), dtype=np.float32)   # средний цвет суперпикселя

    # Сбор статистики цветов для каждого суперпикселя
    for sp_id in range(n_sp):
        mask = (segments == sp_id)
        if np.any(mask):
            sp_colors[sp_id] = np.mean(img_smooth[mask], axis=0)

    # Собираем пиксели для обучения GMM: FG - внутри прямоугольника, BG - снаружи (с отступом)
    # Для фона берём область, расширенную на 20 пикселей вокруг прямоугольника, исключая сам прямоугольник
    fg_pixels = img_smooth[y1:y2, x1:x2].reshape(-1, 3)
    # Фон: полоса вокруг прямоугольника (расширение на 20 пикселей), но не выходящая за границы
    pad = 20
    bg_x1 = max(0, x1 - pad)
    bg_y1 = max(0, y1 - pad)
    bg_x2 = min(w, x2 + pad)
    bg_y2 = min(h, y2 + pad)
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask[bg_y1:bg_y2, bg_x1:bg_x2] = 1
    bg_mask[y1:y2, x1:x2] = 0
    bg_pixels = img_smooth[bg_mask == 1].reshape(-1, 3)

    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        # Если слишком мало пикселей – простой fallback
        # Используем прямоугольник как объект, остальное фон
        result = np.zeros((h, w), dtype=np.uint8)
        result[y1:y2, x1:x2] = 255
        return result

    # Обучаем GMM для переднего и фонового плана (2 компоненты каждый)
    gmm_fg = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm_bg = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm_fg.fit(fg_pixels)
    gmm_bg.fit(bg_pixels)

    # Вычисляем унарные стоимости (логарифмы вероятностей)
    eps = 1e-6
    unary_fg = np.zeros(n_sp)
    unary_bg = np.zeros(n_sp)
    for sp_id in range(n_sp):
        color = sp_colors[sp_id].reshape(1, -1)
        # Логарифмическая вероятность (плотность)
        log_prob_fg = gmm_fg.score_samples(color)[0]
        log_prob_bg = gmm_bg.score_samples(color)[0]
        # Нормализуем в softmax стиле
        # Стоимость = -log(вероятность) * data_weight
        prob_fg = np.exp(log_prob_fg - np.maximum(log_prob_fg, log_prob_bg))
        prob_bg = np.exp(log_prob_bg - np.maximum(log_prob_fg, log_prob_bg))
        # Избегаем нулей
        prob_fg = max(prob_fg, eps)
        prob_bg = max(prob_bg, eps)
        unary_fg[sp_id] = -data_weight * np.log(prob_fg)
        unary_bg[sp_id] = -data_weight * np.log(prob_bg)

    # Построение графа соседства суперпикселей (как в SuperCut)
    adj = [[] for _ in range(n_sp)]
    boundaries = find_boundaries(segments, mode='inner')
    bound_coords = np.argwhere(boundaries)
    for y, x in bound_coords:
        seg_id = segments[y, x]
        neighbors = np.unique(segments[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)])
        for nb in neighbors:
            if nb != seg_id and nb not in adj[seg_id]:
                adj[seg_id].append(nb)

    # Парная стоимость: spatial_weight * exp(-цвет_разница^2 / (2*color_sigma^2))
    def pairwise_cost(i, j):
        color_diff = np.linalg.norm(sp_colors[i] - sp_colors[j])
        w = np.exp(- (color_diff ** 2) / (2 * color_sigma ** 2 + 1e-6))
        return spatial_weight * w

    # Построение графа (networkx)
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    G.add_node(source)
    G.add_node(sink)
    for i in range(n_sp):
        G.add_node(i)

    # Терминальные рёбра
    for i in range(n_sp):
        G.add_edge(source, i, capacity=unary_bg[i])
        G.add_edge(i, sink, capacity=unary_fg[i])

    # Парные рёбра
    for i in range(n_sp):
        for j in adj[i]:
            if j <= i:
                continue
            w = pairwise_cost(i, j)
            if w > 0:
                G.add_edge(i, j, capacity=w)
                G.add_edge(j, i, capacity=w)

    # Вычисление минимального разреза
    try:
        cut_value, partition = nx.minimum_cut(G, source, sink)
        source_side, sink_side = partition
    except Exception as e:
        print(f"Ошибка при min-cut в OneCut: {e}")
        return np.zeros_like(segments, dtype=np.uint8)

    # Формирование маски
    result = np.zeros_like(segments, dtype=np.uint8)
    for sp_id in source_side:
        if isinstance(sp_id, int) and 0 <= sp_id < n_sp:
            result[segments == sp_id] = 255

    # Морфологическое сглаживание
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result



def lazy_snapping(img, fg_scribbles, bg_scribbles, superpixel_size=50, compactness=10, sigma=1.0, color_sigma=10.0):
    """
    Lazy Snapping using graph cut on superpixels.
    color_sigma: sensitivity of color distance (higher = less influence of color contrast)
    """
    if not HAS_NETWORKX:
        raise ImportError("Для Lazy Snapping требуется установить библиотеку networkx. Установите её командой: pip install networkx")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = int(img.shape[0]), int(img.shape[1])
    if sigma > 0:
        img_smooth = cv2.GaussianBlur(img, (0, 0), sigma)
    else:
        img_smooth = img

    # 1. Суперпиксели
    n_segments = max(30, int((h * w) / (superpixel_size * superpixel_size)))
    segments = slic(img_smooth, n_segments=n_segments, compactness=compactness,
                    start_label=0, convert2lab=False)

    # 2. Подготовка структур
    n_sp = len(np.unique(segments))
    sp_colors = np.zeros((n_sp, 3), dtype=np.float32)
    sp_fg_ratio = np.zeros(n_sp, dtype=np.float32)
    sp_bg_ratio = np.zeros(n_sp, dtype=np.float32)

    for sp_id in range(n_sp):
        mask = (segments == sp_id)
        if not np.any(mask):
            continue
        sp_colors[sp_id] = np.mean(img_smooth[mask], axis=0)
        total = mask.sum()
        if total > 0:
            sp_fg_ratio[sp_id] = np.sum(fg_scribbles[mask] == 255) / total
            sp_bg_ratio[sp_id] = np.sum(bg_scribbles[mask] == 255) / total

    # 3. Соседство суперпикселей
    boundaries = find_boundaries(segments, mode='inner')
    bound_coords = np.argwhere(boundaries)
    adj = {sp_id: set() for sp_id in range(n_sp)}
    for y, x in bound_coords:
        sp_center = segments[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    nb = segments[ny, nx]
                    if nb != sp_center:
                        adj[sp_center].add(nb)
    adj_list = {sp_id: list(neighbors) for sp_id, neighbors in adj.items()}

    # 4. Построение графа (networkx)
    import networkx as nx
    G = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    BIG = 1e6

    G.add_node(source)
    G.add_node(sink)
    for sp_id in range(n_sp):
        G.add_node(sp_id)

    # Терминальные веса
    for sp_id in range(n_sp):
        if sp_fg_ratio[sp_id] > 0 and sp_bg_ratio[sp_id] == 0:
            w_fg, w_bg = BIG, 0
        elif sp_bg_ratio[sp_id] > 0 and sp_fg_ratio[sp_id] == 0:
            w_fg, w_bg = 0, BIG
        else:
            w_fg = sp_fg_ratio[sp_id] * BIG
            w_bg = sp_bg_ratio[sp_id] * BIG
            if w_fg == 0 and w_bg == 0:
                w_fg = 1.0
                w_bg = 1.0


        G.add_edge(source, sp_id, capacity=w_fg)
        G.add_edge(sp_id, sink, capacity=w_bg)

    # Парные веса
    for sp_id in range(n_sp):
        for nb_id in adj_list[sp_id]:
            if nb_id <= sp_id:
                continue
            color_diff = np.linalg.norm(sp_colors[sp_id] - sp_colors[nb_id])
            w = np.exp(- (color_diff ** 2) / (2 * color_sigma ** 2))
            G.add_edge(sp_id, nb_id, capacity=w)
            G.add_edge(nb_id, sp_id, capacity=w)

    # 5. Min-cut
    try:
        cut_value, partition = nx.minimum_cut(G, source, sink)
        source_side, sink_side = partition
    except Exception as e:
        print(f"Ошибка при min-cut: {e}")
        return np.zeros((h, w), dtype=np.uint8)

    # 6. Формирование маски
    result = np.zeros_like(segments, dtype=np.uint8)
    for sp_id in source_side:
        if isinstance(sp_id, int) and 0 <= sp_id < n_sp:
            result[segments == sp_id] = 255

    # 7. Морфологическое сглаживание
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result