"""Interactive binary segmentation, without implicit morphology or fallback masks.

SuperCut/OneCut/Lazy Snapping are project variants on a SLIC graph, not
reproductions of the similarly named publications. See docs/interactive.md.
"""

import cv2
import numpy as np
from skimage.segmentation import slic


def _image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Нужно цветное изображение BGR uint8.")
    return image


def _rect(rect, shape):
    if rect is None or len(rect) != 4:
        raise ValueError("Выделите прямоугольную область объекта (ROI).")
    x1, y1, x2, y2 = map(int, rect)
    h, w = shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            "ROI должен иметь положительную площадь и находиться внутри изображения."
        )
    return x1, y1, x2, y2


def _seeds(fg, bg, shape, required=True):
    result = []
    for value in (fg, bg):
        if value is None:
            result.append(np.zeros(shape[:2], bool))
        elif np.shape(value) != shape[:2]:
            raise ValueError("Размер штрихов не совпадает с изображением.")
        else:
            result.append(np.asarray(value) > 0)
    fg, bg = result
    if np.any(fg & bg):
        raise ValueError("Штрихи объекта и фона пересекаются. Исправьте разметку.")
    if required and (not fg.any() or not bg.any()):
        raise ValueError("Отметьте штрихами и объект (FG), и фон (BG).")
    return fg, bg


def _positive(name, value, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{name}: требуется {'неотрицательное' if zero else 'положительное'} конечное значение."
        )


def grabcut_segmentation(
    img,
    rect=None,
    mask=None,
    fg_scribbles=None,
    bg_scribbles=None,
    iterations=5,
    mode="Both",
):
    """Initialize once: rectangle, seeds, or rectangle with hard seed constraints."""
    img = _image(img)
    _positive("Iterations", iterations)
    if mode not in ("GC_INIT_WITH_RECT", "GC_INIT_WITH_MASK", "Both"):
        raise ValueError("Неизвестный режим GrabCut.")
    fg, bg = _seeds(fg_scribbles, bg_scribbles, img.shape, required=False)
    labels = np.full(img.shape[:2], cv2.GC_PR_BGD, np.uint8)
    if mask is not None:
        if np.shape(mask) != labels.shape or not np.isin(mask, [0, 1, 2, 3]).all():
            raise ValueError("Маска GrabCut должна содержать только метки 0, 1, 2, 3.")
        labels = np.array(mask, dtype=np.uint8, copy=True)
    cv_rect = None
    init = cv2.GC_INIT_WITH_MASK
    if mode == "GC_INIT_WITH_RECT" or (mode == "Both" and rect is not None):
        x1, y1, x2, y2 = _rect(rect, img.shape)
        labels.fill(cv2.GC_BGD)
        labels[y1:y2, x1:x2] = cv2.GC_PR_FGD
        cv_rect = (x1, y1, x2 - x1, y2 - y1)
        if mode == "GC_INIT_WITH_RECT":
            init = cv2.GC_INIT_WITH_RECT
        elif np.any(fg & (labels == cv2.GC_BGD)):
            raise ValueError("Штрихи FG должны находиться внутри ROI GrabCut.")
    elif mask is None and (not fg.any() or not bg.any()):
        raise ValueError("Без ROI для GrabCut нужны штрихи и FG, и BG.")
    if mode != "GC_INIT_WITH_RECT":
        labels[fg] = cv2.GC_FGD
        labels[bg] = cv2.GC_BGD
    for kinds in ((cv2.GC_BGD, cv2.GC_PR_BGD), (cv2.GC_FGD, cv2.GC_PR_FGD)):
        if np.isin(labels, kinds).sum() < 5:
            raise ValueError(
                "Для цветовых моделей GrabCut нужны минимум 5 пикселей объекта и фона. Уменьшите ROI или добавьте штрихи."
            )
    cv2.grabCut(
        img,
        labels,
        cv_rect,
        np.zeros((1, 65)),
        np.zeros((1, 65)),
        int(iterations),
        init,
    )
    return np.where(np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]), 255, 0).astype(
        np.uint8
    )


def watershed_segmentation(
    gray,
    distance_threshold=50,
    min_distance=20,
    fg_scribbles=None,
    bg_scribbles=None,
    mode="automatic",
):
    """Marker watershed; automatic mode assumes dark objects on a light background."""
    from skimage.feature import peak_local_max
    from skimage.filters import sobel
    from skimage.segmentation import watershed

    if mode == "scribbles":
        fg, bg = _seeds(fg_scribbles, bg_scribbles, gray.shape)
        _, labels = cv2.connectedComponents(fg.astype(np.uint8))
        markers = np.where(fg, labels + 1, 0).astype(np.int32)
        markers[bg] = 1
        result = watershed(sobel(gray.astype(float) / 255), markers)
        return np.where(result > 1, 255, 0).astype(np.uint8)
    if mode != "automatic":
        raise ValueError("Неизвестный режим Watershed.")
    if not 0 < distance_threshold <= 100:
        raise ValueError("Distance threshold должен быть в диапазоне (0, 100].")
    _positive("Min distance", min_distance)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if not dist.any():
        return np.zeros_like(gray, np.uint8)
    peaks = peak_local_max(
        dist,
        min_distance=int(min_distance),
        exclude_border=False,
        threshold_abs=np.nextafter(float(dist.max()) * distance_threshold / 100, 0),
        labels=(binary > 0).astype(np.uint8),
    )
    markers = np.zeros(gray.shape, np.int32)
    for label, (y, x) in enumerate(peaks, 1):
        markers[y, x] = label
    result = watershed(-dist, markers, mask=binary > 0, watershed_line=True)
    return np.where(result > 0, 255, 0).astype(np.uint8)


def random_walker_segmentation(img, markers, beta=130, mode="cg_j", use_color=True):
    from importlib.util import find_spec

    from skimage.segmentation import random_walker

    img = _image(img)
    if np.shape(markers) != img.shape[:2] or not np.isin(markers, [0, 1, 2]).all():
        raise ValueError(
            "Random Walker: нужны метки 0 (неизвестно), 1 (объект), 2 (фон)."
        )
    _seeds(markers == 1, markers == 2, img.shape)
    _positive("Beta", beta, zero=True)
    if mode not in ("cg_j", "cg_mg", "bf"):
        raise ValueError("Неизвестный решатель Random Walker.")
    if mode == "cg_mg" and find_spec("pyamg") is None:
        raise ImportError("Решатель cg_mg требует pyamg. Выберите cg_j.")
    if mode == "bf" and markers.size > 65536:
        raise ValueError(
            "Прямой решатель bf предназначен для небольших изображений (до 256×256). Выберите cg_j."
        )
    data = (
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if use_color
        else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ).astype(float) / 255
    if np.max(np.std(data, axis=(0, 1))) == 0:
        raise ValueError(
            "Random Walker: изображение постоянно, цветовые границы отсутствуют."
        )
    labels = random_walker(
        data,
        markers.copy(),
        beta=beta,
        mode=mode,
        channel_axis=-1 if use_color else None,
    )
    return np.where(labels == 1, 255, 0).astype(np.uint8)


def active_contour_segmentation(
    gray,
    rect,
    alpha=0.01,
    beta=0.1,
    gamma=0.01,
    max_iter=250,
    convergence=0.001,
    smooth=True,
    init_type="Ellipse",
):
    from skimage.segmentation import active_contour

    x1, y1, x2, y2 = _rect(rect, gray.shape)
    for name, value in (("Alpha", alpha), ("Beta", beta)):
        _positive(name, value, zero=True)
    for name, value in (
        ("Gamma", gamma),
        ("Max iterations", max_iter),
        ("Convergence", convergence),
    ):
        _positive(name, value)
    image = cv2.GaussianBlur(gray, (5, 5), 2) if smooth else gray
    image = image.astype(float) / 255
    # skimage uses (row, column), OpenCV uses (x, y).
    cx, cy = (x1 + x2 - 1) / 2, (y1 + y2 - 1) / 2
    rx, ry = (x2 - x1 - 1) / 2, (y2 - y1 - 1) / 2
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    if init_type == "Rectangle":
        corners = np.array([[y1, x1], [y1, x2 - 1], [y2 - 1, x2 - 1], [y2 - 1, x1]])
        initial = np.concatenate(
            [
                a + np.arange(25)[:, None] / 25 * (b - a)
                for a, b in zip(corners, np.roll(corners, -1, axis=0))
            ]
        )
    elif init_type in ("Circle", "Ellipse"):
        if init_type == "Circle":
            rx = ry = min(rx, ry)
        initial = np.column_stack((cy + ry * np.sin(theta), cx + rx * np.cos(theta)))
    else:
        raise ValueError("Неизвестный начальный контур.")
    snake = active_contour(
        image,
        initial,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=int(max_iter),
        convergence=convergence,
    )
    snake = np.clip(snake, [0, 0], np.array(gray.shape) - 1)
    result = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(result, [np.rint(snake[:, ::-1]).astype(np.int32)], 255)
    return result


def morphological_chan_vese_segmentation(
    gray, rect, iterations=100, smoothing=2, lambda_inside=1.0, lambda_outside=1.0
):
    """Region-based evolution, useful when an object has a weak edge."""
    from skimage.segmentation import morphological_chan_vese

    x1, y1, x2, y2 = _rect(rect, gray.shape)
    for name, value in (
        ("Iterations", iterations),
        ("Lambda inside", lambda_inside),
        ("Lambda outside", lambda_outside),
    ):
        _positive(name, value)
    _positive("Smoothing", smoothing, zero=True)
    initial = np.zeros(gray.shape, np.uint8)
    initial[y1:y2, x1:x2] = 1
    result = morphological_chan_vese(
        gray.astype(float) / 255,
        int(iterations),
        initial,
        smoothing=int(smoothing),
        lambda1=lambda_outside,
        lambda2=lambda_inside,
    )
    return result.astype(np.uint8) * 255


def _superpixels(img, superpixel_size, compactness, sigma):
    img = _image(img)
    _positive("Superpixel size", superpixel_size)
    _positive("Compactness", compactness)
    _positive("Sigma", sigma, zero=True)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    count = min(
        img.shape[0] * img.shape[1],
        max(2, round(img.shape[0] * img.shape[1] / superpixel_size**2)),
    )
    # SLIC's sigma is applied once, in RGB before Lab conversion.
    segments = slic(
        rgb,
        n_segments=count,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        convert2lab=True,
    )
    _, segments = np.unique(segments, return_inverse=True)
    segments = segments.reshape(img.shape[:2])
    counts = np.bincount(segments.ravel())
    colors = np.column_stack(
        [
            np.bincount(segments.ravel(), weights=img[..., c].ravel()) / counts
            for c in range(3)
        ]
    )
    pairs = np.concatenate(
        (
            np.column_stack((segments[:, :-1].ravel(), segments[:, 1:].ravel())),
            np.column_stack((segments[:-1].ravel(), segments[1:].ravel())),
        )
    )
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return segments, colors, pairs[pairs[:, 0] != pairs[:, 1]]


def _appearance_costs(colors, fg_pixels, bg_pixels):
    from sklearn.mixture import GaussianMixture

    scores = []
    for pixels in (fg_pixels, bg_pixels):
        if len(pixels) < 2:
            raise ValueError(
                "Нужно минимум 2 пикселя каждого класса для цветовой модели."
            )
        # Bound memory/cost deterministically; constant-color marks need one component.
        pixels = pixels[
            np.linspace(0, len(pixels) - 1, min(len(pixels), 30000), dtype=int)
        ]
        components = min(5, len(np.unique(pixels, axis=0)))
        model = GaussianMixture(n_components=components, random_state=0, reg_covar=1e-3)
        model.fit(pixels)
        scores.append(model.score_samples(colors))
    scores = np.array(scores)
    # Subtract a shared per-node constant: nonnegative costs, same optimal cut.
    return np.max(scores, axis=0) - scores


def _cut(
    segments, colors, pairs, costs, color_sigma, spatial_weight=1.0, fg=None, bg=None
):
    import networkx as nx

    _positive("Color sigma", color_sigma)
    _positive("Spatial weight", spatial_weight, zero=True)
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Некорректные унарные стоимости графа.")
    costs = np.array(costs, copy=True)
    weights = spatial_weight * np.exp(
        -np.sum((colors[pairs[:, 0]] - colors[pairs[:, 1]]) ** 2, axis=1)
        / (2 * color_sigma**2)
    )
    if fg is not None:
        fg_ids, bg_ids = np.unique(segments[fg]), np.unique(segments[bg])
        if np.intersect1d(fg_ids, bg_ids).size:
            raise ValueError(
                "FG и BG попали в один суперпиксель. Уменьшите Superpixel size или разнесите штрихи."
            )
        hard = 1 + costs.sum() + weights.sum()
        costs[:, fg_ids] = np.array([[0], [hard]])
        costs[:, bg_ids] = np.array([[hard], [0]])
    graph = nx.DiGraph()
    graph.add_nodes_from(("source", "sink"))
    for i in range(len(colors)):
        graph.add_edge("source", i, capacity=float(costs[1, i]))
        graph.add_edge(i, "sink", capacity=float(costs[0, i]))
    for (i, j), weight in zip(pairs, weights):
        graph.add_edge(int(i), int(j), capacity=float(weight))
        graph.add_edge(int(j), int(i), capacity=float(weight))
    _, (foreground, _) = nx.minimum_cut(graph, "source", "sink")
    lut = np.array(
        [255 if i in foreground else 0 for i in range(len(colors))], np.uint8
    )
    return lut[segments]


def supercut_segmentation(
    img,
    rect,
    superpixel_size=50,
    compactness=10,
    sigma=1.0,
    lambda_val=0.5,
    sigma_color=10.0,
):
    """Project variant: ROI-occupancy prior and color-sensitive SLIC graph."""
    x1, y1, x2, y2 = _rect(rect, img.shape)
    _positive("Lambda", lambda_val, zero=True)
    segments, colors, pairs = _superpixels(img, superpixel_size, compactness, sigma)
    inside = np.zeros(img.shape[:2], np.uint8)
    inside[y1:y2, x1:x2] = 1
    ratio = np.bincount(segments.ravel(), weights=inside.ravel()) / np.bincount(
        segments.ravel()
    )
    ratio = np.clip(ratio, 1e-6, 1 - 1e-6)
    costs = -lambda_val * np.log(np.array([ratio, 1 - ratio]))
    return _cut(segments, colors, pairs, costs, sigma_color)


def onecut_segmentation(
    img,
    rect,
    superpixel_size=50,
    compactness=10,
    sigma=1.0,
    spatial_weight=1.0,
    data_weight=1.0,
    color_sigma=10.0,
):
    """Project variant: GMM learned from ROI/exterior and a SLIC graph cut."""
    x1, y1, x2, y2 = _rect(rect, img.shape)
    _positive("Data weight", data_weight, zero=True)
    segments, colors, pairs = _superpixels(img, superpixel_size, compactness, sigma)
    inside = np.zeros(img.shape[:2], bool)
    inside[y1:y2, x1:x2] = True
    costs = data_weight * _appearance_costs(colors, img[inside], img[~inside])
    return _cut(segments, colors, pairs, costs, color_sigma, spatial_weight)


def lazy_snapping(
    img,
    fg_scribbles,
    bg_scribbles,
    superpixel_size=50,
    compactness=10,
    sigma=1.0,
    color_sigma=10.0,
):
    """Project variant: seed-trained GMM unary costs with hard superpixel seeds."""
    fg, bg = _seeds(fg_scribbles, bg_scribbles, img.shape)
    segments, colors, pairs = _superpixels(img, superpixel_size, compactness, sigma)
    costs = _appearance_costs(colors, img[fg], img[bg])
    return _cut(segments, colors, pairs, costs, color_sigma, fg=fg, bg=bg)
