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


def apply_complex_augmentations(image, annotations, aug_config, border_color=(0, 0, 0)):
    """
    Применяет случайный поворот и сдвиг (shear) к изображению и аннотациям.
    Возвращает изображение и аннотации (размер может измениться).
    """
    h, w = image.shape[:2]
    img = image.copy()
    ann = annotations.copy()
    matrices = []

    if aug_config.get('random_rotate', False):
        max_angle = aug_config.get('random_rotate_angle', 10.0)
        angle_deg = random.uniform(-max_angle, max_angle)
        angle_rad = angle_deg * math.pi / 180.0
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        R = np.array([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                      [math.sin(angle_rad), math.cos(angle_rad), 0],
                      [0, 0, 1]], dtype=np.float32)
        T_inv = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        rot_mat = T @ R @ T_inv
        matrices.append(rot_mat)
        logger.debug(f"apply_complex: rotation {angle_deg:.2f} deg")

    if aug_config.get('shear', False):
        max_shear = aug_config.get('shear_angle', 10.0)
        shear_x_deg = random.uniform(-max_shear, max_shear)
        shear_y_deg = random.uniform(-max_shear, max_shear)
        shear_x = shear_x_deg * math.pi / 180.0
        shear_y = shear_y_deg * math.pi / 180.0
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        S = np.array([[1, math.tan(shear_x), 0],
                      [math.tan(shear_y), 1, 0],
                      [0, 0, 1]], dtype=np.float32)
        T_inv = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        shear_mat = T @ S @ T_inv
        matrices.append(shear_mat)
        logger.debug(f"apply_complex: shear x={shear_x_deg:.2f}°, y={shear_y_deg:.2f}°")

    if matrices:
        M_total = compose_affine_matrices(matrices)
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        corners_hom = np.hstack([corners, np.ones((4, 1))])
        transformed_corners = (M_total @ corners_hom.T).T[:, :2]
        x_min, y_min = transformed_corners.min(axis=0)
        x_max, y_max = transformed_corners.max(axis=0)
        new_w = int(round(x_max - x_min))
        new_h = int(round(y_max - y_min))
        shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        M_shifted = shift @ M_total
        M_2x3 = M_shifted[:2, :]

        # Применяем аффинное преобразование к изображению
        aug_img = cv2.warpAffine(image, M_2x3, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=border_color)

        # Функция для преобразования полигонов (obb и segment)
        # Вход: нормализованные координаты (список float)
        # Выход: нормализованные координаты после преобразования
        def transform_polygon(pts_norm_flat):
            if not pts_norm_flat:
                return pts_norm_flat
            # Преобразуем в массив точек (N,2)
            pts_norm = np.array(pts_norm_flat).reshape(-1, 2)
            # 1. Переводим в абсолютные пиксели исходного изображения
            pts_abs = pts_norm * np.array([w, h])
            # 2. Применяем аффинное преобразование
            pts_abs_trans = apply_affine_to_points_np(pts_abs, M_shifted, w, h, new_w, new_h)
            # 3. Нормализуем относительно нового размера
            pts_norm_new = pts_abs_trans / np.array([new_w, new_h])
            return pts_norm_new.flatten().tolist()

        # Логируем изменение размеров
        logger.debug(f"Complex aug: image {w}x{h} -> {new_w}x{new_h}")
        log_annotations_summary("До сложных аугментаций", annotations)

        # Обрабатываем аннотации
        new_annotations = []
        for ann_item in ann:
            typ = ann_item[0]
            cls = ann_item[1]
            if len(ann_item) == 3:
                coords = ann_item[2]
            else:
                coords = ann_item[2:]

            if typ == 'detect':
                # Для bbox: переводим в абсолютные координаты углов, трансформируем, находим новый bbox
                cx, cy, bw, bh = coords
                cx_abs = cx * w
                cy_abs = cy * h
                half_w = (bw * w) / 2
                half_h = (bh * h) / 2
                pts = np.array([[cx_abs - half_w, cy_abs - half_h],
                                [cx_abs + half_w, cy_abs - half_h],
                                [cx_abs + half_w, cy_abs + half_h],
                                [cx_abs - half_w, cy_abs + half_h]], dtype=np.float32)
                pts_trans = apply_affine_to_points_np(pts, M_shifted, w, h, new_w, new_h)
                if len(pts_trans) == 0:
                    continue
                xmin, ymin = pts_trans.min(axis=0)
                xmax, ymax = pts_trans.max(axis=0)
                new_cx = (xmin + xmax) / 2.0 / new_w
                new_cy = (ymin + ymax) / 2.0 / new_h
                new_bw = (xmax - xmin) / new_w
                new_bh = (ymax - ymin) / new_h
                new_annotations.append(('detect', cls, new_cx, new_cy, new_bw, new_bh))


            elif typ == 'obb':
                new_coords = transform_polygon(coords)
                if new_coords:
                    # После аффинного преобразования координаты могут быть не прямоугольником – исправляем
                    new_coords = repair_obb_to_rectangle(new_coords)
                    new_annotations.append(('obb', cls, new_coords))

            else:  # segment
                new_coords = transform_polygon(coords)
                if new_coords:
                    new_annotations.append(('segment', cls, new_coords))

        new_annotations = clip_annotations(new_annotations)
        log_annotations_summary("После сложных аугментаций", new_annotations)
        return aug_img, new_annotations

    else:
        # Если ни одна сложная аугментация не активна, возвращаем оригинал
        return img, ann
# ---------------------------------------------------------------------
# Для масок (сегментация)
# ---------------------------------------------------------------------
def apply_complex_augmentations_mask(image, mask, aug_config, border_color=(0,0,0)):
    img = image.copy()
    msk = mask.copy()
    transforms = []
    if aug_config.get('random_rotate', False):
        angle = aug_config.get('random_rotate_angle', 10.0)
        transforms.append(A.Rotate(limit=angle, p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   fill=border_color,          # цвет для изображения
                                   fill_mask=0))               # для маски фон всегда 0
    if aug_config.get('shear', False):
        angle = aug_config.get('shear_angle', 10.0)
        transforms.append(A.Affine(shear=(-angle, angle), p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT,
                                   fill=border_color,           # цвет для изображения
                                   fill_mask=0))                # для маски фон всегда 0
    if transforms:
        aug = A.Compose(transforms, is_check_shapes=False)
        transformed = aug(image=img, mask=msk)
        img = transformed['image']
        msk = transformed['mask']
    return img, msk
