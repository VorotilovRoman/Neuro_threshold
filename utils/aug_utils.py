from import_libs_internal import *

# Настройка логгера
logger = logging.getLogger("aug_utils")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Флаг для очень подробного лога
VERBOSE_ANNOTATION_LOG = 0


def repair_obb_to_rectangle(pts_norm_flat):
    """
    Принимает список из 8 чисел (4 точки x1,y1,x2,y2,x3,y3,x4,y4) в нормализованных координатах.
    Возвращает список из 8 чисел – вершины минимального ориентированного прямоугольника,
    охватывающего исходные точки. Гарантирует, что аннотация остаётся прямоугольником.
    """
    if not pts_norm_flat:
        return pts_norm_flat
    pts = np.array(pts_norm_flat).reshape(-1, 2)
    if len(pts) != 4:
        # Не 4 точки – не OBB, возвращаем как есть
        return pts_norm_flat

    # Вычисляем минимальный ориентированный прямоугольник в нормализованном пространстве
    # (масштаб не влияет на углы, так что можно работать с нормализованными координатами)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    # rect: ((cx, cy), (width, height), angle)
    box = cv2.boxPoints(rect)  # 4 точки в порядке: [x1,y1], [x2,y2], ...
    # Ограничиваем координаты диапазоном [0,1]
    box = np.clip(box, 0.0, 1.0)
    return box.flatten().tolist()


def log_annotations_summary(prefix, annotations, max_items=5):
    """Выводит краткую информацию об аннотациях (тип, класс, кол-во точек)"""
    if not annotations:
        logger.debug(f"{prefix} пустой список")
        return
    logger.debug(f"{prefix} всего {len(annotations)} аннотаций")
    for i, ann in enumerate(annotations[:max_items]):
        typ = ann[0]
        cls = ann[1]
        if len(ann) == 3:
            coords = ann[2]
        else:
            coords = ann[2:]
        coord_len = len(coords)
        if typ == 'detect':
            logger.debug(f"  {i+1}: detect class={cls}, bbox={[round(c,4) for c in coords]}")
        elif typ == 'obb':
            pts = np.array(coords).reshape(-1,2)
            logger.debug(f"  {i+1}: obb class={cls}, points={pts.tolist()}")
        elif typ == 'segment':
            pts = np.array(coords).reshape(-1,2)
            logger.debug(f"  {i+1}: segment class={cls}, points_count={len(pts)}")
    if len(annotations) > max_items:
        logger.debug(f"  ... и еще {len(annotations)-max_items} аннотаций")

def _log_verbose_annotations(prefix, annotations):
    """Выводит полные данные аннотаций только при VERBOSE_ANNOTATION_LOG=True"""
    if not VERBOSE_ANNOTATION_LOG:
        return
    for i, ann in enumerate(annotations):
        typ = ann[0]
        cls = ann[1]
        if len(ann) == 3:
            coords = ann[2]
        else:
            coords = ann[2:]
        logger.debug(f"{prefix} [{i}] {typ} cls={cls} coords={coords}")

def _get_fill_params(fill_color):
    try:
        A.PadIfNeeded(1, 1, fill=fill_color)
        return {'fill': fill_color}
    except TypeError:
        return {'value': fill_color}

# ---------- Вспомогательные функции для аффинных преобразований ----------
def apply_affine_to_points_np(points, M_3x3, orig_w, orig_h, new_w, new_h):
    """Применяет аффинное преобразование (3x3) к массиву точек (Nx2) в пикселях"""
    if len(points) == 0:
        return np.empty((0,2))
    pts_hom = np.hstack([points, np.ones((len(points),1))])  # Nx3
    transformed = (M_3x3 @ pts_hom.T).T  # Nx3
    return transformed[:,:2]

def compose_affine_matrices(matrices_3x3):
    result = np.eye(3, dtype=np.float32)
    for m in matrices_3x3:
        result = m @ result
    return result

# ---------- Общие функции для ресайза ----------
def resize_image_and_annotations(image, annotations, target_size, fill_color=(0,0,0)):
    """
    Ресайз с сохранением пропорций (LongestMaxSize) + паддинг до target_size.
    Пересчитывает аннотации.
    """
    if target_size is None:
        return image, annotations
    h,w = image.shape[:2]
    logger.info(f"resize: {w}x{h} -> {target_size}x{target_size}")
    scale = target_size / max(h,w)
    new_w = int(round(w*scale))
    new_h = int(round(h*scale))
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_size - new_h)//2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w)//2
    pad_right = target_size - new_w - pad_left
    padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=fill_color)
    logger.debug(f"  scale={scale:.4f}, new_size={new_w}x{new_h}, paddings: L={pad_left}, R={pad_right}, T={pad_top}, B={pad_bottom}")
    log_annotations_summary("Входные аннотации в resize", annotations)
    _log_verbose_annotations("  resize вход подробно", annotations)

    new_annotations = []
    for ann in annotations:
        typ = ann[0]
        cls = ann[1]
        if len(ann)==3:
            coords = ann[2]
        else:
            coords = ann[2:]
        if typ == 'detect':
            cx,cy,bw,bh = coords
            cx_abs = cx*w; cy_abs = cy*h
            half_w = (bw*w)/2; half_h = (bh*h)/2
            new_cx_abs = cx_abs*scale + pad_left
            new_cy_abs = cy_abs*scale + pad_top
            new_half_w = half_w*scale; new_half_h = half_h*scale
            new_cx = new_cx_abs/target_size
            new_cy = new_cy_abs/target_size
            new_bw = (new_half_w*2)/target_size
            new_bh = (new_half_h*2)/target_size
            new_annotations.append(('detect', cls, new_cx, new_cy, new_bw, new_bh))
        elif typ == 'obb':
            pts = np.array(coords).reshape(-1, 2)
            pts_abs = pts * np.array([w, h])
            pts_abs = pts_abs * scale
            pts_abs[:, 0] += pad_left
            pts_abs[:, 1] += pad_top
            pts_norm = pts_abs / target_size
            # Применяем repair для гарантии прямоугольной формы
            pts_fixed = repair_obb_to_rectangle(pts_norm.flatten().tolist())
            new_annotations.append(('obb', cls, pts_fixed))
        elif typ == 'segment':
            pts = np.array(coords).reshape(-1,2)
            pts_abs = pts * np.array([w,h])
            pts_abs = pts_abs * scale
            pts_abs[:,0] += pad_left
            pts_abs[:,1] += pad_top
            pts_norm = pts_abs / target_size
            new_annotations.append(('segment', cls, pts_norm.flatten().tolist()))
    new_annotations = clip_annotations(new_annotations)
    log_annotations_summary("Выходные аннотации после resize", new_annotations)
    _log_verbose_annotations("  resize выход подробно", new_annotations)
    return padded_img, new_annotations


def resize_image_and_annotations_stretch(image, annotations, target_size):
    """
    Растягивает изображение до квадрата target_size x target_size без сохранения пропорций.
    Аннотации не изменяются (нормализованные координаты остаются теми же).
    """
    h, w = image.shape[:2]
    scale_x = target_size / w
    scale_y = target_size / h
    img_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    logger.info(
        f"Stretch resize: {w}x{h} -> {target_size}x{target_size} (scale_x={scale_x:.4f}, scale_y={scale_y:.4f})")
    log_annotations_summary("Входные аннотации (stretch)", annotations)

    # При растяжении нормализованные координаты не меняются
    new_annotations = []
    for ann in annotations:
        typ = ann[0]
        cls = ann[1]
        if len(ann) == 3:
            coords = ann[2]
        else:
            coords = ann[2:]
        if typ == 'detect':
            cx, cy, bw, bh = coords
            new_annotations.append(('detect', cls, cx, cy, bw, bh))
        elif typ == 'obb':
            # Растяжение не меняет нормализованные координаты, но для единообразия вызываем repair
            pts_fixed = repair_obb_to_rectangle(coords)
            new_annotations.append(('obb', cls, pts_fixed))
        else:  # segment
            new_annotations.append(('segment', cls, coords))

    new_annotations = clip_annotations(new_annotations)
    log_annotations_summary("Выходные аннотации (stretch)", new_annotations)
    return img_resized, new_annotations



# ---------- Обрезка аннотаций ----------
def clip_annotations(annotations, epsilon=1e-7):
    before = len(annotations)
    clipped = []
    for ann in annotations:
        typ = ann[0]
        cls = ann[1]
        if len(ann) == 3:
            coords = ann[2]
        else:
            coords = ann[2:]
        if typ == 'detect':
            cx,cy,w,h = coords
            cx = max(epsilon, min(1.0-epsilon, cx))
            cy = max(epsilon, min(1.0-epsilon, cy))
            w = max(epsilon, min(1.0-epsilon, w))
            h = max(epsilon, min(1.0-epsilon, h))
            if w>epsilon and h>epsilon:
                clipped.append(('detect', cls, cx,cy,w,h))
        elif typ == 'obb':
            new_points = [max(epsilon, min(1.0-epsilon, p)) for p in coords]
            clipped.append(('obb', cls, new_points))
        elif typ == 'segment':
            new_points = [max(epsilon, min(1.0-epsilon, p)) for p in coords]
            clipped.append(('segment', cls, new_points))
    if before != len(clipped):
        logger.info(f"clip_annotations: {before} -> {len(clipped)}")
        if VERBOSE_ANNOTATION_LOG:
            _log_verbose_annotations("После clip", clipped)
    return clipped
