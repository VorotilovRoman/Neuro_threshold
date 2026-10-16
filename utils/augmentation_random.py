# utils/augmentation_random.py
from import_libs_internal import *

# ---------------------------------------------------------------------
# Для детекции (аннотации)
# ---------------------------------------------------------------------

# Настройка логгера
logger = logging.getLogger("aug_utils")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =====================================================================
# Общие вспомогательные функции для сложных аугментаций
# =====================================================================

def _random_crop_factor(aug_config):
    """Возвращает случайный scale для crop в [min_scale, 1.0].
    min_scale берётся из aug_config['random_crop_scale'], по умолчанию 0.7."""
    try:
        min_scale = float(aug_config.get('random_crop_scale', 0.7))
    except (TypeError, ValueError):
        min_scale = 0.7
    min_scale = max(0.1, min(1.0, min_scale))
    if min_scale >= 1.0:
        return 1.0
    return random.uniform(min_scale, 1.0)


def _pick_crop_box(w, h, aug_config):
    """Случайный прямоугольник кропа внутри (w, h).
    Возвращает (x0, y0, cw, ch) в пикселях."""
    factor = _random_crop_factor(aug_config)
    cw = max(16, int(round(w * factor)))
    ch = max(16, int(round(h * factor)))
    cw = min(cw, w)
    ch = min(ch, h)
    if cw >= w and ch >= h:
        return 0, 0, w, h
    x0 = random.randint(0, max(0, w - cw))
    y0 = random.randint(0, max(0, h - ch))
    return x0, y0, cw, ch


def _crop_annotations(annotations, x0, y0, cw, ch, prev_w, prev_h):
    """Пересчитывает нормализованные аннотации после кропа.
    x0, y0 — левый верхний угол кропа в пикселях изображения prev_w × prev_h.
    cw, ch — размер кропа.
    Объекты, полностью вышедшие за пределы кропа, удаляются."""
    new_ann = []
    x0f, y0f = float(x0), float(y0)
    x1f, y1f = float(x0 + cw), float(y0 + ch)

    for ann in annotations:
        typ, cls = ann[0], ann[1]
        coords = ann[2] if len(ann) == 3 else ann[2:]

        if typ == 'detect':
            cx, cy, bw, bh = coords
            cx_pix = cx * prev_w
            cy_pix = cy * prev_h
            bw_pix = bw * prev_w
            bh_pix = bh * prev_h
            xmin = cx_pix - bw_pix / 2.0 - x0f
            xmax = cx_pix + bw_pix / 2.0 - x0f
            ymin = cy_pix - bh_pix / 2.0 - y0f
            ymax = cy_pix + bh_pix / 2.0 - y0f
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(cw), xmax)
            ymax = min(float(ch), ymax)
            if xmax - xmin < 2 or ymax - ymin < 2:
                continue
            new_cx = (xmin + xmax) / 2.0 / cw
            new_cy = (ymin + ymax) / 2.0 / ch
            new_bw = (xmax - xmin) / cw
            new_bh = (ymax - ymin) / ch
            new_ann.append(('detect', cls, new_cx, new_cy, new_bw, new_bh))

        elif typ in ('obb', 'segment'):
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts_pix = pts * np.array([prev_w, prev_h], dtype=np.float32)

            # Bbox полигона
            xmin_p = float(pts_pix[:, 0].min())
            xmax_p = float(pts_pix[:, 0].max())
            ymin_p = float(pts_pix[:, 1].min())
            ymax_p = float(pts_pix[:, 1].max())

            # Полностью вне кропа — пропускаем
            if xmax_p < x0f or xmin_p > x1f or ymax_p < y0f or ymin_p > y1f:
                continue

            pts_pix[:, 0] -= x0f
            pts_pix[:, 1] -= y0f
            pts_norm = pts_pix / np.array([cw, ch], dtype=np.float32)
            pts_norm = np.clip(pts_norm, 0.0, 1.0)

            if typ == 'obb':
                pts_fixed = repair_obb_to_rectangle(pts_norm.flatten().tolist())
                new_ann.append(('obb', cls, pts_fixed))
            else:
                new_ann.append(('segment', cls, pts_norm.flatten().tolist()))

    return new_ann


# =====================================================================
# Сложные аугментации для детекции/сегментации (аннотации)
# =====================================================================
def apply_complex_augmentations(image, annotations, aug_config, border_color=(0, 0, 0)):
    """
    Применяет случайный поворот, shear и кроп к изображению и аннотациям.
    Размер результата не обязан совпадать с исходным — при необходимости
    родитель потом приведёт его к target_size.

    Возвращает (aug_img, new_annotations).
    """
    h, w = image.shape[:2]
    ann = list(annotations) if annotations else []

    matrices = []

    cx_c = (w - 1) / 2.0
    cy_c = (h - 1) / 2.0
    T = np.array([[1, 0, cx_c], [0, 1, cy_c], [0, 0, 1]], dtype=np.float32)
    T_inv = np.array([[1, 0, -cx_c], [0, 1, -cy_c], [0, 0, 1]], dtype=np.float32)

    if aug_config.get('random_rotate', False):
        try:
            max_angle = float(aug_config.get('random_rotate_angle', 10.0))
        except (TypeError, ValueError):
            max_angle = 10.0
        angle_deg = random.uniform(-max_angle, max_angle)
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        matrices.append(T @ R @ T_inv)
        logger.debug(f"apply_complex: rotation {angle_deg:.2f} deg")

    if aug_config.get('shear', False):
        try:
            max_shear = float(aug_config.get('shear_angle', 10.0))
        except (TypeError, ValueError):
            max_shear = 10.0
        # Одна общая величина для X и Y — иначе картинку сильно «расплющивает»,
        # а OBB/segment теряют прямоугольную форму.
        shear_deg = random.uniform(-max_shear, max_shear)
        shear = math.tan(math.radians(shear_deg))
        S = np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        matrices.append(T @ S @ T_inv)
        logger.debug(f"apply_complex: shear {shear_deg:.2f} deg")

    do_crop = bool(aug_config.get('random_crop', False))

    if not matrices and not do_crop:
        return image.copy(), ann

    # --- Единая матрица rotate+shear ---
    M_total = compose_affine_matrices(matrices) if matrices else np.eye(3, dtype=np.float32)

    # --- Bbox всего изображения после трансформации ---
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    corners_hom = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    tc = (M_total @ corners_hom.T).T[:, :2]
    x_min, y_min = tc.min(axis=0)
    x_max, y_max = tc.max(axis=0)
    new_w = max(1, int(round(x_max - x_min)))
    new_h = max(1, int(round(y_max - y_min)))

    shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    M_shifted = shift @ M_total

    # --- Применяем к изображению ---
    aug_img = cv2.warpAffine(
        image, M_shifted[:2, :], (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )

    # --- Трансформируем аннотации ---
    new_annotations = []
    for ann_item in ann:
        typ, cls = ann_item[0], ann_item[1]
        coords = ann_item[2] if len(ann_item) == 3 else ann_item[2:]

        if typ == 'detect':
            cx_n, cy_n, bw_n, bh_n = coords
            cx_pix = cx_n * w
            cy_pix = cy_n * h
            half_w = (bw_n * w) / 2.0
            half_h = (bh_n * h) / 2.0
            pts = np.array([
                [cx_pix - half_w, cy_pix - half_h],
                [cx_pix + half_w, cy_pix - half_h],
                [cx_pix + half_w, cy_pix + half_h],
                [cx_pix - half_w, cy_pix + half_h],
            ], dtype=np.float32)
            pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
            pts_t = (M_shifted @ pts_h.T).T[:, :2]

            xmin, ymin = pts_t.min(axis=0)
            xmax, ymax = pts_t.max(axis=0)
            xmin = max(0.0, xmin); ymin = max(0.0, ymin)
            xmax = min(float(new_w), xmax); ymax = min(float(new_h), ymax)
            if xmax - xmin < 2 or ymax - ymin < 2:
                continue
            new_cx = (xmin + xmax) / 2.0 / new_w
            new_cy = (ymin + ymax) / 2.0 / new_h
            new_bw = (xmax - xmin) / new_w
            new_bh = (ymax - ymin) / new_h
            new_annotations.append(('detect', cls, new_cx, new_cy, new_bw, new_bh))

        elif typ in ('obb', 'segment'):
            pts_norm = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts_pix = pts_norm * np.array([w, h], dtype=np.float32)
            pts_h = np.hstack([pts_pix, np.ones((len(pts_pix), 1), dtype=np.float32)])
            pts_t = (M_shifted @ pts_h.T).T[:, :2]

            # Bbox полигона после трансформации
            xmin_p = float(pts_t[:, 0].min())
            xmax_p = float(pts_t[:, 0].max())
            ymin_p = float(pts_t[:, 1].min())
            ymax_p = float(pts_t[:, 1].max())
            # Полностью вне — пропускаем
            if xmax_p < 0 or xmin_p > new_w or ymax_p < 0 or ymin_p > new_h:
                continue

            pts_norm_new = pts_t / np.array([new_w, new_h], dtype=np.float32)
            pts_norm_new = np.clip(pts_norm_new, 0.0, 1.0)

            if typ == 'obb':
                pts_fixed = repair_obb_to_rectangle(pts_norm_new.flatten().tolist())
                new_annotations.append(('obb', cls, pts_fixed))
            else:
                new_annotations.append(('segment', cls, pts_norm_new.flatten().tolist()))

    # --- Random Crop ---
    if do_crop and new_annotations:
        x0, y0, cw, ch = _pick_crop_box(new_w, new_h, aug_config)
        if cw < new_w or ch < new_h:
            aug_img = aug_img[y0:y0 + ch, x0:x0 + cw].copy()
            new_annotations = _crop_annotations(
                new_annotations, x0, y0, cw, ch, new_w, new_h
            )
            logger.debug(f"apply_complex: crop to {cw}x{ch} at ({x0},{y0})")

    new_annotations = clip_annotations(new_annotations)
    logger.debug(f"apply_complex: image {w}x{h} -> {aug_img.shape[1]}x{aug_img.shape[0]}, "
                 f"anns {len(ann)} -> {len(new_annotations)}")
    return aug_img, new_annotations


# =====================================================================
# Сложные аугментации для масок (сегментация)
# =====================================================================
def apply_complex_augmentations_mask(image, mask, aug_config, border_color=(0, 0, 0)):
    """
    Применяет случайный поворот, shear и кроп к изображению и маске.
    Маска — одноканальная (class_id), фон = 0.

    Возвращает (aug_img, aug_mask).
    """
    img = image.copy()
    msk = mask.copy()

    transforms = []
    if aug_config.get('random_rotate', False):
        try:
            angle = float(aug_config.get('random_rotate_angle', 10.0))
        except (TypeError, ValueError):
            angle = 10.0
        transforms.append(A.Rotate(
            limit=(-angle, angle), p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_color,
            fill_mask=0,
        ))
    if aug_config.get('shear', False):
        try:
            angle = float(aug_config.get('shear_angle', 10.0))
        except (TypeError, ValueError):
            angle = 10.0
        transforms.append(A.Affine(
            shear=(-angle, angle), p=1.0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=border_color,
            fill_mask=0,
        ))

    if transforms:
        try:
            aug = A.Compose(transforms, is_check_shapes=False)
            transformed = aug(image=img, mask=msk)
            img = transformed['image']
            msk = transformed['mask']
        except Exception as e:
            logger.warning(f"apply_complex_augmentations_mask: albumentations failed: {e}")

    if aug_config.get('random_crop', False):
        h, w = img.shape[:2]
        x0, y0, cw, ch = _pick_crop_box(w, h, aug_config)
        if cw < w or ch < h:
            img = img[y0:y0 + ch, x0:x0 + cw].copy()
            msk = msk[y0:y0 + ch, x0:x0 + cw].copy()
            logger.debug(f"apply_complex_mask: crop to {cw}x{ch} at ({x0},{y0})")

    return img, msk