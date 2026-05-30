# utils/augmentation_base.py
from import_libs_internal import *
# ---------------------------------------------------------------------
# Для детекции (аннотации)
# -------------------------------------------------
# Настройка логгера
logger = logging.getLogger("aug_utils")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
def apply_simple_augmentations(image, annotations, aug_config):
    """Применяет горизонтальный/вертикальный флип и повороты на 90°."""
    img = image.copy()
    ann = annotations.copy()
    if aug_config.get('horizontal_flip', False) and random.random() < 0.5:
        img, ann = _flip_horizontal(img, ann)
    if aug_config.get('vertical_flip', False) and random.random() < 0.5:
        img, ann = _flip_vertical(img, ann)
    if aug_config.get('rotate_90', False):
        k = random.choice([1, 2, 3])
        img, ann = _rotate_90_multiple(img, ann, k)
    return img, ann

def _flip_horizontal(img, ann):
    img = cv2.flip(img, 1)
    new_ann = []
    for a in ann:
        typ, cls = a[0], a[1]
        if typ == 'detect':
            _, _, cx, cy, w, h = a
            new_ann.append(('detect', cls, 1-cx, cy, w, h))
        elif typ == 'obb':
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1, 2)
            pts_arr[:, 0] = 1 - pts_arr[:, 0]
            new_coords = pts_arr.flatten().tolist()
            new_coords = repair_obb_to_rectangle(new_coords)  # <-- добавить
            new_ann.append(('obb', cls, new_coords))
        else:   # segment
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1,2)
            pts_arr[:,0] = 1 - pts_arr[:,0]
            new_ann.append(('segment', cls, pts_arr.flatten().tolist()))
    return img, clip_annotations(new_ann)

def _flip_vertical(img, ann):
    img = cv2.flip(img, 0)
    new_ann = []
    for a in ann:
        typ, cls = a[0], a[1]
        if typ == 'detect':
            _, _, cx, cy, w, h = a
            new_ann.append(('detect', cls, cx, 1-cy, w, h))
        elif typ == 'obb':
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1, 2)
            pts_arr[:, 1] = 1 - pts_arr[:, 1]
            new_coords = pts_arr.flatten().tolist()
            new_coords = repair_obb_to_rectangle(new_coords)
            new_ann.append(('obb', cls, new_coords))
        else:
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1,2)
            pts_arr[:,1] = 1 - pts_arr[:,1]
            new_ann.append(('segment', cls, pts_arr.flatten().tolist()))
    return img, clip_annotations(new_ann)

def _rotate_90_multiple(img, ann, k):
    for _ in range(k):
        img, ann = _rotate_90(img, ann)
    return img, ann

def _rotate_90(img, ann):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_ann = []
    for a in ann:
        typ, cls = a[0], a[1]
        if typ == 'detect':
            _, _, cx, cy, w, h = a
            # cx' = 1-cy, cy' = cx, w' = h, h' = w
            new_ann.append(('detect', cls, 1-cy, cx, h, w))
        elif typ == 'obb':
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1, 2)
            new_pts = []
            for x, y in pts_arr:
                new_pts.extend([1 - y, x])
            new_coords = repair_obb_to_rectangle(new_pts)
            new_ann.append(('obb', cls, new_coords))
        else:
            _, _, pts = a
            pts_arr = np.array(pts).reshape(-1,2)
            new_pts = []
            for x, y in pts_arr:
                new_pts.extend([1-y, x])
            new_ann.append(('segment', cls, new_pts))
    return img, clip_annotations(new_ann)

# ---------------------------------------------------------------------
# Для масок (сегментация)
# ---------------------------------------------------------------------
def apply_simple_augmentations_mask(image, mask, aug_config):
    """Применяет простые аугментации к изображению и маске (флип, поворот 90°) с помощью OpenCV."""
    img = image.copy()
    msk = mask.copy()

    # Горизонтальный флип
    if aug_config.get('horizontal_flip', False) and random.random() < 0.5:
        img = cv2.flip(img, 1)
        msk = cv2.flip(msk, 1)

    # Вертикальный флип
    if aug_config.get('vertical_flip', False) and random.random() < 0.5:
        img = cv2.flip(img, 0)
        msk = cv2.flip(msk, 0)

    # Поворот на 90° (случайное количество раз 1-3)
    if aug_config.get('rotate_90', False):
        k = random.choice([1, 2, 3])
        for _ in range(k):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            msk = cv2.rotate(msk, cv2.ROTATE_90_CLOCKWISE)

    return img, msk