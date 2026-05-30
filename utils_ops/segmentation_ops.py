# utils/segmentation_ops.py
from import_libs_external import *
from utils.settings import settings


# ---------- Вспомогательные функции для отображения ----------
def get_display_params(img_shape):
    h, w = img_shape[:2]
    base = min(h, w)
    base_thickness = max(1, int(0.005 * base))
    base_font_scale = max(0.3, 0.001 * base)
    font_thickness = max(1, int(0.001 * base))
    min_area = max(1, int(0.0005 * h * w))

    thickness = int(base_thickness * settings.get_line_thickness_factor())
    font_scale = base_font_scale * settings.get_font_scale_factor()

    return thickness, font_scale, font_thickness, min_area


def draw_label(img_color, label, x, y, box_w, box_h, font_scale, font_thickness, thickness, color=None):
    if color is None:
        color = settings.get_color('label_text')
    h, w = img_color.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    margin = 5
    if y - text_h - margin >= 0:
        text_x = x
        text_y = y - margin
    elif y + box_h + text_h + margin <= h:
        text_x = x
        text_y = y + box_h + text_h + margin
    else:
        text_x = x + margin
        text_y = y + text_h + margin
    if text_x + text_w > w:
        text_x = w - text_w - margin
    if text_x < 0:
        text_x = margin
    cv2.putText(img_color, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, font_thickness, cv2.LINE_AA)


def draw_label_rotated(img_color, label, center, font_scale, font_thickness, color=None):
    if color is None:
        color = settings.get_color('label_text')
    h, w = img_color.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cx, cy = center
    text_x = int(cx - text_w // 2)
    text_y = int(cy - text_h // 2 - 5)
    if text_x < 0:
        text_x = 5
    if text_x + text_w > w:
        text_x = w - text_w - 5
    if text_y < text_h + 5:
        text_y = int(cy + text_h // 2 + 15)
    if text_y > h - 5:
        text_y = h - 5
    cv2.putText(img_color, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, font_thickness, cv2.LINE_AA)


# ---------- Методы бинаризации ----------
def simple_threshold(img, thresh):
    _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return binary, thresh


def otsu_threshold(img):
    th, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, th


def triangle_threshold(img):
    th, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary, th


def niblack_threshold(img, window_size=15, k=-0.2):
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter((img.astype(np.float32))**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean + k * std
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None


def sauvola_threshold(img, window_size=25, k=0.2, R=128):
    mean = cv2.boxFilter(img.astype(np.float32), cv2.CV_32F, (window_size, window_size))
    mean_sq = cv2.boxFilter((img.astype(np.float32))**2, cv2.CV_32F, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    threshold = mean * (1 + k * (std / R - 1))
    binary = (img > threshold).astype(np.uint8) * 255
    return binary, None


def isodata_threshold(img, initial_threshold=128):
    threshold = initial_threshold
    img_flat = img.ravel().astype(np.float32)
    while True:
        bg = img_flat[img_flat < threshold]
        fg = img_flat[img_flat >= threshold]
        if len(bg) == 0 or len(fg) == 0:
            break
        mean_bg = np.mean(bg)
        mean_fg = np.mean(fg)
        new_threshold = (mean_bg + mean_fg) / 2
        if abs(new_threshold - threshold) < 0.5:
            break
        threshold = new_threshold
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary, int(threshold)


def background_symmetry_threshold(img, excess_threshold=0.2):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    bg_peak = np.argmax(hist)
    bg_hist_expected = np.zeros_like(hist)
    bg_hist_expected[:bg_peak + 1] = hist[:bg_peak + 1]
    for i in range(1, bg_peak + 1):
        if bg_peak + i < 256:
            bg_hist_expected[bg_peak + i] = hist[bg_peak - i]
    excess = hist - bg_hist_expected
    excess = np.maximum(excess, 0)
    if excess.max() > 0:
        excess_norm = excess / excess.max()
        threshold_candidates = np.where(excess_norm > excess_threshold)[0]
        if len(threshold_candidates) > 0:
            threshold = threshold_candidates[0]
        else:
            threshold = bg_peak + 20
    else:
        threshold = bg_peak + 20
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary, threshold


def row_adaptive_threshold(img, window_size=50, k=0.5):
    h, w = img.shape
    binary = np.zeros_like(img)
    for row in range(h):
        row_data = img[row, :].astype(np.float32)
        for col in range(0, w, window_size):
            end = min(col + window_size, w)
            window = row_data[col:end]
            if len(window) > 0:
                mean = np.mean(window)
                std = np.std(window)
                thresh = max(0, mean - k * std)
                binary[row, col:end] = (row_data[col:end] < thresh).astype(np.uint8) * 255
    return binary, None


# ---------- Выделение объектов из бинарной маски (с поддержкой draw) ----------
def segment_contours(img, binary, use_hull, draw=True):
    """
    Возвращает (img_color, objects) где objects = список ('segment', 0, points) в нормализованных координатах.
    Если draw=False, img_color будет просто скопирован (без рисования), но объекты всё равно вычислены.
    """
    img_color = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thickness, font_scale, font_thickness, min_area = get_display_params(img.shape)
    color_rect = settings.get_color('annotation')
    color_label = settings.get_color('label_text')

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    obj_count = 0
    h, w = img.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        obj_count += 1
        if use_hull:
            cnt = cv2.convexHull(cnt)

        if draw:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w_box, y + h_box), color_rect, thickness)
            draw_label(img_color, f"obj {obj_count}", x, y, w_box, h_box,
                       font_scale, font_thickness, thickness, color=color_label)

        # Получаем точки контура в нормализованных координатах
        pts = cnt.squeeze().astype(np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        pts_norm = []
        for pt in pts:
            pts_norm.append(pt[0] / w)
            pts_norm.append(pt[1] / h)
        objects.append(('segment', 0, pts_norm))   # class 0 временно; позже можно задать
    return img_color, objects


def segment_projections(img, binary, draw=True):
    """
    Возвращает (img_color, objects) где objects = список ('detect', 0, cx, cy, w, h)
    Если draw=False, img_color будет просто скопирован (без рисования).
    """
    img_color = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thickness, font_scale, font_thickness, min_area = get_display_params(img.shape)
    color_rect = settings.get_color('annotation')
    color_label = settings.get_color('label_text')

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    objects = []
    obj_count = 0
    h, w = img.shape[:2]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        obj_count += 1
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]

        if draw:
            cv2.rectangle(img_color, (x, y), (x + w_box, y + h_box), color_rect, thickness)
            draw_label(img_color, f"obj {obj_count}", x, y, w_box, h_box,
                       font_scale, font_thickness, thickness, color=color_label)

        # Нормализованные координаты для detect
        cx = (x + w_box / 2) / w
        cy = (y + h_box / 2) / h
        wn = w_box / w
        hn = h_box / h
        objects.append(('detect', 0, cx, cy, wn, hn))
    return img_color, objects


def segment_min_area_rect(img, binary, use_hull, draw=True):
    """
    Возвращает (img_color, objects) где objects = список ('obb', 0, points) где points = 8 нормализованных координат углов.
    Если draw=False, img_color будет просто скопирован (без рисования).
    """
    img_color = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    thickness, font_scale, font_thickness, min_area = get_display_params(img.shape)
    color_rect = settings.get_color('annotation')
    color_label = settings.get_color('label_text')

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    obj_count = 0
    h, w = img.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        obj_count += 1
        if use_hull:
            cnt = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        if draw:
            cv2.drawContours(img_color, [box], 0, color_rect, thickness)
            cx, cy = rect[0]
            label = f"obj {obj_count}"
            draw_label_rotated(img_color, label, (cx, cy), font_scale, font_thickness, color=color_label)

        # Нормализуем 4 угла
        pts_norm = []
        for pt in box:
            pts_norm.append(pt[0] / w)
            pts_norm.append(pt[1] / h)
        objects.append(('obb', 0, pts_norm))
    return img_color, objects


def segment_polygons(img, binary, use_hull, draw=True):
    """Алиас для segment_contours (уже возвращает segment)."""
    return segment_contours(img, binary, use_hull, draw=draw)


# ---------- Отрисовка YOLO аннотаций ----------
def draw_yolo_annotations(img, annotations, color=(0, 255, 0)):
    """Рисует аннотации любого типа (detect, obb, segment) на изображении."""
    if not annotations:
        return img
    img_copy = img.copy()
    thickness, font_scale, font_thickness, _ = get_display_params(img.shape)
    h, w = img.shape[:2]
    color_rect = settings.get_color('annotation')
    color_label = settings.get_color('label_text')

    for idx, ann in enumerate(annotations):
        if ann[0] == 'detect':
            _, cls, cx, cy, bw, bh = ann
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            x2 = x + int(bw * w)
            y2 = y + int(bh * h)
            cv2.rectangle(img_copy, (x, y), (x2, y2), color_rect, thickness)
            draw_label(img_copy, f"obj {idx + 1}", x, y, x2 - x, y2 - y,
                       font_scale, font_thickness, thickness, color=color_label)
        elif ann[0] == 'obb':
            _, cls, points = ann   # 8 нормализованных координат
            pts = []
            for i in range(0, len(points), 2):
                px = int(points[i] * w)
                py = int(points[i + 1] * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_copy, [pts], True, color_rect, thickness)
            # центр для подписи (среднее арифметическое углов)
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            draw_label_rotated(img_copy, f"obj {idx + 1}", (cx, cy),
                               font_scale, font_thickness, color=color_label)
        elif ann[0] == 'segment':
            _, cls, points = ann   # список нормализованных координат (чётное число)
            pts = []
            for i in range(0, len(points), 2):
                px = int(points[i] * w)
                py = int(points[i + 1] * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_copy, [pts], True, color_rect, thickness)
            # подпись у первой точки или центроид
            if len(pts) > 0:
                draw_label(img_copy, f"obj {idx + 1}", pts[0][0], pts[0][1], 0, 0,
                           font_scale, font_thickness, thickness, color=color_label)
    return img_copy


# ---------- Функции для работы со списками объектов ----------
def format_annotation_display(index, ann, img_w, img_h):
    """Форматирует аннотацию для отображения в QListWidget.
       Поддерживает форматы ('detect',...), ('obb',...), ('segment',...)."""
    typ = ann[0]
    if typ == 'detect':
        _, cls, cx, cy, w, h = ann
        x_px = int((cx - w / 2) * img_w)
        y_px = int((cy - h / 2) * img_h)
        width_px = int(w * img_w)
        height_px = int(h * img_h)
        markup = f"Obj {index + 1}: rect x={x_px} y={y_px} w={width_px} h={height_px}"
        yolo = f"YOLO: {cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        return f"{markup} | {yolo}"
    elif typ == 'obb':
        _, cls, points = ann  # 8 координат
        pts_str = ' '.join(f"{p:.6f}" for p in points)
        return f"Obj {index + 1}: OBB | YOLO-OBB: {cls} {pts_str}"
    elif typ == 'segment':
        _, cls, points = ann
        num_pts = len(points) // 2
        if len(points) > 6:
            pts_preview = points[:6]
            preview = ' '.join(f"{p:.6f}" for p in pts_preview) + ' ...'
        else:
            preview = ' '.join(f"{p:.6f}" for p in points)
        return f"Obj {index + 1}: polygon {num_pts} pts | YOLO-seg: {cls} {preview}"
    else:
        return f"Obj {index + 1}: unknown type"


def update_annotation_list(list_widget: QListWidget, annotations, img_w: int, img_h: int):
    """Очищает и заполняет QListWidget строками аннотаций. Каждому элементу в UserRole записывается индекс."""
    list_widget.clear()
    for i, ann in enumerate(annotations):
        text = format_annotation_display(i, ann, img_w, img_h)
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, i)
        list_widget.addItem(item)


def delete_annotation_by_index(current_annotations, idx, all_annotations=None, current_index=None):
    """Удаляет аннотацию с индексом idx. Возвращает (new_annotations, new_selected_index)."""
    if idx < 0 or idx >= len(current_annotations):
        return current_annotations, -1
    new_annotations = current_annotations.copy()
    del new_annotations[idx]
    new_selected = -1 if idx >= len(new_annotations) else idx
    if all_annotations is not None and current_index is not None:
        if current_index < len(all_annotations):
            all_annotations[current_index] = new_annotations.copy()
    return new_annotations, new_selected


def draw_selected_objects(img, objects, selected_indices, draw_mode, use_hull,
                          color_rect, color_label, thickness, font_scale, font_thickness):
    """Рисует выбранные объекты на изображении. Поддерживает ('detect',...), ('obb',...), ('segment',...)."""
    img_copy = img.copy()
    if not selected_indices:
        return img_copy
    h, w = img.shape[:2]
    for idx_in_list, obj_idx in enumerate(selected_indices):
        if obj_idx >= len(objects):
            continue
        obj = objects[obj_idx]
        ann_type = obj[0]
        if ann_type == 'detect':
            _, cls, cx, cy, bw, bh = obj
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            x2 = x + int(bw * w)
            y2 = y + int(bh * h)
            cv2.rectangle(img_copy, (x, y), (x2, y2), color_rect, thickness)
            draw_label(img_copy, f"obj {obj_idx + 1}", x, y, x2 - x, y2 - y,
                       font_scale, font_thickness, thickness, color=color_label)
        elif ann_type == 'obb':
            _, cls, points = obj
            pts = []
            for i in range(0, len(points), 2):
                px = int(points[i] * w)
                py = int(points[i + 1] * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_copy, [pts], True, color_rect, thickness)
            cx = sum(p[0] for p in pts) / 4
            cy = sum(p[1] for p in pts) / 4
            draw_label_rotated(img_copy, f"obj {obj_idx + 1}", (cx, cy),
                               font_scale, font_thickness, color=color_label)
        elif ann_type == 'segment':
            _, cls, points = obj
            pts = []
            for i in range(0, len(points), 2):
                px = int(points[i] * w)
                py = int(points[i + 1] * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_copy, [pts], True, color_rect, thickness)
            if len(pts) > 0:
                draw_label(img_copy, f"obj {obj_idx + 1}", pts[0][0], pts[0][1], 0, 0,
                           font_scale, font_thickness, thickness, color=color_label)
    return img_copy


def format_object_for_list(index, obj, img_w, img_h):
    """Универсальное форматирование для старого стиля (используется в threshold_methods.py)."""
    if isinstance(obj, tuple) and len(obj) > 0 and obj[0] in ('detect', 'obb', 'segment'):
        return format_annotation_display(index, obj, img_w, img_h)
    elif len(obj) == 4:
        x, y, w, h = obj
        cx_norm = (x + w / 2) / img_w
        cy_norm = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        markup = f"obj {index + 1}: x={x}, y={y}, w={w}, h={h}"
        yolo = f"YOLO: 0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        return f"{markup} | {yolo}"
    elif len(obj) == 5:
        cx, cy, w, h, angle = obj
        return f"obj {index + 1}: center=({cx:.1f},{cy:.1f}), size=({w:.1f}x{h:.1f}), angle={angle:.1f}°"
    else:
        return f"obj {index + 1}: unknown"


# ---------- Вспомогательные функции для пороговых методов ----------
def normalize_method_name(method_name):
    """Приводит имя метода к каноническому виду для UI."""
    method_map = {
        'otsu': 'Otsu',
        'otsu (opencv)': 'Otsu',
        'triangle': 'Triangle',
        'triangle (opencv)': 'Triangle',
        'simple threshold': 'Simple Threshold',
        'adaptive mean': 'Adaptive Mean',
        'adaptive gauss': 'Adaptive Gauss',
        'niblack': 'Niblack',
        'sauvola': 'Sauvola',
        'isodata': 'ISODATA',
        'background symmetry': 'Background Symmetry',
        'row adaptive': 'Row Adaptive',
    }
    lower_name = method_name.lower().strip()
    if lower_name in method_map:
        return method_map[lower_name]
    if lower_name.endswith(' (opencv)'):
        base = lower_name[:-9]
        if base in method_map:
            return method_map[base]
    return method_name


def apply_threshold_method(gray_img, method_name, params):
    """Применяет метод бинаризации по имени и параметрам."""
    method_name = normalize_method_name(method_name)
    if method_name == "Simple Threshold":
        thresh = params.get('threshold', 128)
        binary, _ = simple_threshold(gray_img, thresh)
        return binary, thresh
    elif method_name == "Otsu":
        binary, thresh = otsu_threshold(gray_img)
        return binary, thresh
    elif method_name == "Triangle":
        binary, thresh = triangle_threshold(gray_img)
        return binary, thresh
    elif method_name == "Adaptive Mean":
        ws = params.get('window', 25)
        if ws % 2 == 0:
            ws += 1
        if ws < 3:
            ws = 3
        c = params.get('c', 3)
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, ws, c)
        return binary, None
    elif method_name == "Adaptive Gauss":
        ws = params.get('window', 25)
        if ws % 2 == 0:
            ws += 1
        if ws < 3:
            ws = 3
        c = params.get('c', 3)
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, ws, c)
        return binary, None
    elif method_name == "Niblack":
        ws = params.get('window', 25)
        k = params.get('k', -0.2)
        binary, _ = niblack_threshold(gray_img, ws, k)
        return binary, None
    elif method_name == "Sauvola":
        ws = params.get('window', 25)
        k = params.get('k', 0.2)
        r = params.get('r', 128)
        binary, _ = sauvola_threshold(gray_img, ws, k, r)
        return binary, None
    elif method_name == "ISODATA":
        init = params.get('init', 128)
        binary, thresh = isodata_threshold(gray_img, init)
        return binary, thresh
    elif method_name == "Background Symmetry":
        excess = params.get('excess', 0.2)
        binary, thresh = background_symmetry_threshold(gray_img, excess)
        return binary, thresh
    elif method_name == "Row Adaptive":
        ws = params.get('window', 50)
        k = params.get('k', 0.5)
        binary, _ = row_adaptive_threshold(gray_img, ws, k)
        return binary, None
    else:
        raise ValueError(f"Unknown method: {method_name}")


def get_morph_kernel(size, shape_name):
    shape_map = {
        "Rectangle": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS
    }
    shape = shape_map.get(shape_name, cv2.MORPH_RECT)
    return cv2.getStructuringElement(shape, (size, size))


def apply_morphology(binary, close_factor, open_factor, kernel_shape, img_shape=None):
    if img_shape is None:
        h, w = binary.shape[:2]
    else:
        h, w = img_shape[:2]
    processed = binary.copy()
    if close_factor > 0:
        ksize = max(1, int(close_factor * min(h, w)))
        kernel = get_morph_kernel(ksize, kernel_shape)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    if open_factor > 0:
        ksize = max(1, int(open_factor * min(h, w)))
        kernel = get_morph_kernel(ksize, kernel_shape)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    return processed