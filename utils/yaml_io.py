# utils/yaml_io.py
"""
Унифицированный парсинг YAML-датасетов: YOLO (data.yaml) и U-Net (masks_data.yaml).

Используется в:
    • tabs/preparing_dataset_yaml.py
    • tabs/layout_dataset.py

Возвращает плоский список пар (img_path, label_or_none) + мета.
"""
from import_libs_external import *


_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


# ============================================================
# Общие вспомогательные
# ============================================================
def _resolve_relative_to_yaml(raw_path, yaml_path):
    """Абсолютный путь: если raw_path абсолютный — берём как есть,
    иначе — относительно директории YAML."""
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return os.path.normpath(raw_path)
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(yaml_path)), raw_path)
    )


def _resolve_split_dirs(split_val, base_path):
    """
    split_val может быть:
        • строкой — путь к папке с изображениями;
        • строкой — путь к .txt со списком путей;
        • списком таких строк.
    Возвращает плоский список папок с изображениями.
    """
    if not split_val:
        return []

    items = split_val if isinstance(split_val, (list, tuple)) else [split_val]
    out = []
    for it in items:
        if not isinstance(it, str):
            continue
        p = it if os.path.isabs(it) else os.path.join(base_path, it)
        p = os.path.normpath(p)
        if os.path.isdir(p):
            out.append(p)
        elif os.path.isfile(p):
            # txt-list: строки — относительные пути к папкам или изображениям
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        q = line if os.path.isabs(line) else os.path.join(base_path, line)
                        q = os.path.normpath(q)
                        if os.path.isdir(q):
                            out.append(q)
                        elif os.path.isfile(q) and q.lower().endswith(_IMG_EXTS):
                            # Одиночный файл — берём его директорию
                            out.append(os.path.dirname(q))
            except Exception:
                pass
    # Уникализируем, сохраняя порядок
    seen = set()
    uniq = []
    for d in out:
        key = os.path.abspath(d)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)
    return uniq


def _find_label_for_image(img_path, base_path, split, rel_dir):
    """
    Ищет .txt для изображения в стандартных местах YOLO-структуры.
    Возвращает путь или None.
    """
    img_path_abs = os.path.abspath(img_path)
    d = os.path.dirname(img_path_abs)
    base = os.path.splitext(os.path.basename(img_path_abs))[0]

    # 1) рядом с изображением
    same = os.path.join(d, base + '.txt')
    if os.path.exists(same):
        return same

    # 2) подстановка images → labels в пути
    norm = img_path_abs.replace('\\', '/')
    parts = norm.split('/')
    for i, p in enumerate(parts):
        if p == 'images':
            new_parts = parts[:i] + ['labels'] + parts[i + 1:]
            new_parts[-1] = base + '.txt'
            cand = os.sep.join(new_parts)
            if os.path.exists(cand):
                return cand

    # 3) стандартные относительные пути от base_path
    for lbl_dir in [
        os.path.join(base_path, 'labels', split),
        os.path.join(base_path, 'labels', os.path.basename(rel_dir)) if rel_dir else None,
        os.path.join(base_path, 'labels'),
        os.path.join(os.path.dirname(d), 'labels'),
    ]:
        if not lbl_dir:
            continue
        cand = os.path.join(lbl_dir, base + '.txt')
        if os.path.exists(cand):
            return cand
    return None

def find_mask_for_image(img_path, base_path=None):
    """
    Ищет маску для изображения в стандартных местах YAML-структуры.
    Возвращает путь к маске или None.
    """
    img_path_abs = os.path.abspath(img_path)
    d = os.path.dirname(img_path_abs)
    base = os.path.splitext(os.path.basename(img_path_abs))[0]
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 1) рядом с изображением (только другой файл)
    for ext in exts:
        cand = os.path.join(d, base + ext)
        if os.path.exists(cand) and os.path.abspath(cand) != img_path_abs:
            return cand

    # 2) подстановка images → masks в пути
    norm = img_path_abs.replace('\\', '/')
    parts = norm.split('/')
    for i, p in enumerate(parts):
        if p == 'images':
            for ext in exts:
                new_parts = parts[:i] + ['masks'] + parts[i + 1:]
                new_parts[-1] = base + ext
                cand = os.sep.join(new_parts)
                if os.path.exists(cand):
                    return cand

    # 3) base_path/masks/ и соседняя masks/
    dirs_to_try = []
    if base_path:
        dirs_to_try.append(os.path.join(base_path, 'masks'))
    dirs_to_try.append(os.path.join(os.path.dirname(d), 'masks'))
    for mask_dir in dirs_to_try:
        for ext in exts:
            cand = os.path.join(mask_dir, base + ext)
            if os.path.exists(cand):
                return cand
    return None
# ============================================================
# YOLO YAML
# ============================================================
def load_yolo_yaml(yaml_path, log_cb=None):
    """
    Парсит YOLO data.yaml.

    Возвращает:
        pairs:      [(img_path, label_path_or_None), ...]
        names:      {cls_id: name} | None
        base_path:  str — абсолютный корень датасета
        splits_info: {'train': n, 'val': n, 'test': n} — сколько найдено в каждом
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML не содержит словаря")

    root_raw = cfg.get('path', '')
    if root_raw:
        base_path = _resolve_relative_to_yaml(root_raw, yaml_path)
    else:
        base_path = os.path.dirname(os.path.abspath(yaml_path))

    names = cfg.get('names')
    # names может быть списком ['cat','dog'] или словарём {0:'cat',1:'dog'}
    names_dict = None
    if isinstance(names, dict):
        try:
            names_dict = {int(k): str(v) for k, v in names.items()}
        except Exception:
            names_dict = None
    elif isinstance(names, (list, tuple)):
        names_dict = {i: str(v) for i, v in enumerate(names)}

    # Секции split'ов, которые поддерживаем
    split_keys = [
        ('train', cfg.get('train')),
        ('val', cfg.get('val')),
        ('valid', cfg.get('valid')),
        ('validation', cfg.get('validation')),
        ('test', cfg.get('test')),
    ]

    pairs = []
    splits_info = {}
    seen_paths = set()

    for split, split_val in split_keys:
        if not split_val:
            continue
        dirs = _resolve_split_dirs(split_val, base_path)
        if not dirs:
            continue
        n_in_split = 0
        for d in dirs:
            try:
                files = sorted(os.listdir(d))
            except Exception as e:
                if log_cb:
                    log_cb(f"Ошибка чтения {d}: {e}")
                continue
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _IMG_EXTS:
                    continue
                img_path = os.path.join(d, fname)
                key = os.path.abspath(img_path)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                lbl = _find_label_for_image(img_path, base_path, split, d)
                pairs.append((img_path, lbl))
                n_in_split += 1
        splits_info[split] = n_in_split

    return pairs, names_dict, base_path, splits_info


# ============================================================
# U-Net YAML (masks_data.yaml)
# ============================================================
def load_unet_yaml_pairs(yaml_path, log_cb=None):
    """
    Парсит U-Net masks_data.yaml.
    Возвращает:
        pairs:       [(img_path, mask_path), ...]
        class_names: {yolo_cls_id: name} — плотная нумерация
        mask_offset: int
        base_path:   str
        num_classes: int (включая фон)
    """
    from utils.unet_io import load_unet_yaml
    data = load_unet_yaml(yaml_path, log_cb=log_cb)
    pairs = [(img, mask) for (img, mask) in data['pairs']]
    return (pairs,
            data['class_names'],
            int(data['mask_class_offset']),
            data['root'],
            int(data['num_classes']))