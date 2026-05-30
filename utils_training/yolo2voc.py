#!/usr/bin/env python3
"""
Label Convert: YOLO (detection + segmentation) -> VOC Pascal format.

Поддерживает:
- Конвертацию YOLO датасета (images, labels, masks) в структуру VOC.
- Создание папки VOC/ с подпапками:
    JPEGImages/            – копии всех изображений
    SegmentationClass/     – бинарные маски (0 и 255) в формате PNG с палитрой
    Annotations/           – XML описания bounding box (из YOLO .txt)
    ImageSets/Segmentation/ – файлы train.txt, val.txt, test.txt

Использование:
    python yolo2voc.py --yaml data.yaml --output ./VOC_output
    или без аргументов – интерактивный режим.

Требования:
    pip install pascal-voc-writer pyyaml pillow
"""

import argparse
import os
import sys
import shutil
from PIL import Image

try:
    from pascal_voc_writer import Writer
except ImportError:
    print("Ошибка: библиотека 'pascal-voc-writer' не установлена.")
    print("Установите: pip install pascal-voc-writer")
    sys.exit(1)

YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("Предупреждение: PyYAML не установлен. Работа с YAML-файлами недоступна.")

# ------------------------------------------------------------
# Палитра для бинарной маски (0 = фон, 1 = объект)
# Индекс 0 -> чёрный, индекс 1 -> белый
# ------------------------------------------------------------
def create_binary_palette():
    """Создаёт палитру для 8-битного PNG: индекс 0 -> чёрный, индекс 1 -> белый."""
    palette = [0, 0, 0,   255, 255, 255] + [0, 0, 0] * 254
    return bytes(palette)

def convert_mask_to_voc_format(src_mask_path, dst_mask_path):
    """
    Конвертирует бинарную маску (0 и 255) в P-режим с палитрой.
    Исходная маска может быть в градациях серого (L) или RGB.
    """
    mask = Image.open(src_mask_path).convert('L')   # одноканальная
    # Бинаризуем: значения > 127 считаем объектом (1)
    mask_array = mask.point(lambda p: 1 if p > 127 else 0, mode='L')
    # Переводим в P-режим и присваиваем палитру
    mask_p = mask_array.convert('P')
    mask_p.putpalette(create_binary_palette())
    mask_p.save(dst_mask_path, format='PNG')

def copy_image(src_img_path, dst_img_path):
    """Копирует изображение без изменений."""
    shutil.copy2(src_img_path, dst_img_path)

def convert_yolo_txt_to_voc_xml(txt_path, img_path, class_names, output_xml_path):
    """
    Конвертирует один YOLO .txt файл (детекция) в VOC XML.
    txt_path: путь к .txt файлу YOLO (нормализованные координаты)
    img_path: путь к соответствующему изображению (для получения w, h)
    class_names: список имён классов (индекс = id)
    output_xml_path: куда сохранить .xml
    """
    with Image.open(img_path) as img:
        w, h = img.size

    writer = Writer(output_xml_path, w, h)

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x_min = max(0, int((x_center - width / 2) * w))
            y_min = max(0, int((y_center - height / 2) * h))
            x_max = min(w, int((x_center + width / 2) * w))
            y_max = min(h, int((y_center + height / 2) * h))

            if x_min >= x_max or y_min >= y_max:
                continue

            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            writer.addObject(class_name, x_min, y_min, x_max, y_max)

    writer.save(output_xml_path)

def process_set(set_name, base_dir, images_rel, masks_rel, labels_rel, class_names, voc_root):
    """
    Обрабатывает один набор данных (train/val/test).
    - Копирует изображения из base_dir/images_rel в voc_root/JPEGImages/
    - Копирует (конвертирует) маски из base_dir/masks_rel в voc_root/SegmentationClass/
    - Конвертирует YOLO метки (если есть) в voc_root/Annotations/
    - Добавляет имена файлов (без расширения) в список для ImageSets/Segmentation/{set_name}.txt
    """
    src_images_dir = os.path.join(base_dir, images_rel)
    src_masks_dir = os.path.join(base_dir, masks_rel) if masks_rel else None
    src_labels_dir = os.path.join(base_dir, labels_rel) if labels_rel else None

    if not os.path.isdir(src_images_dir):
        print(f"Предупреждение: папка {src_images_dir} не найдена, пропускаем {set_name}")
        return []

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    img_files = [f for f in os.listdir(src_images_dir)
                 if f.lower().endswith(img_extensions)]

    if not img_files:
        print(f"Нет изображений в {src_images_dir}, пропускаем {set_name}")
        return []

    file_names = []
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        # Копируем изображение
        src_img = os.path.join(src_images_dir, img_file)
        dst_img = os.path.join(voc_root, 'JPEGImages', img_file)
        copy_image(src_img, dst_img)

        # Обрабатываем маску (если есть)
        if src_masks_dir:
            mask_file = base_name + '.png'
            src_mask = os.path.join(src_masks_dir, mask_file)
            if os.path.exists(src_mask):
                dst_mask = os.path.join(voc_root, 'SegmentationClass', mask_file)
                convert_mask_to_voc_format(src_mask, dst_mask)
            else:
                print(f"Предупреждение: маска для {img_file} не найдена в {src_masks_dir}")

        # Обрабатываем YOLO .txt (если есть)
        if src_labels_dir:
            txt_file = base_name + '.txt'
            src_txt = os.path.join(src_labels_dir, txt_file)
            if os.path.exists(src_txt):
                dst_xml = os.path.join(voc_root, 'Annotations', base_name + '.xml')
                convert_yolo_txt_to_voc_xml(src_txt, src_img, class_names, dst_xml)

        file_names.append(base_name)

    # Записываем список файлов в ImageSets/Segmentation/{set_name}.txt
    seg_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
    os.makedirs(seg_dir, exist_ok=True)
    list_file = os.path.join(seg_dir, f'{set_name}.txt')
    with open(list_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(file_names))
    print(f"Создан {list_file} ({len(file_names)} файлов)")

    return file_names

def build_voc_from_yolo(yaml_file, output_root):
    """Основная функция конвертации, использующая YAML-файл датасета YOLO."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML не установлен")

    # Проверка существования YAML-файла
    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(f"YAML-файл не найден по пути: {yaml_file}")

    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Определяем корневую папку данных
    path_root = data.get('path', '')
    if path_root and os.path.exists(path_root):
        base_dir = path_root
    else:
        base_dir = os.path.dirname(yaml_file)
        if path_root:
            candidate = os.path.join(base_dir, path_root)
            if os.path.exists(candidate):
                base_dir = candidate

    # Извлекаем имена классов
    class_names = data.get('names', [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    # Функция для безопасного получения пути (строка или первый элемент списка)
    def get_path(key):
        val = data.get(key)
        if isinstance(val, list):
            if val:
                return val[0]
            return None
        return val

    train_images = get_path('train')
    val_images = get_path('val')
    test_images = get_path('test')

    # Пути к маскам и меткам (можно задать в YAML или использовать стандартные)
    train_masks = get_path('train_masks') or 'masks/train'
    val_masks = get_path('val_masks') or 'masks/val'
    test_masks = get_path('test_masks') or 'masks/test'

    train_labels = get_path('train_labels') or 'labels/train'
    val_labels = get_path('val_labels') or 'labels/val'
    test_labels = get_path('test_labels') or 'labels/test'

    # Создаём структуру VOC
    voc_root = os.path.join(output_root, 'VOC')
    for sub in ['JPEGImages', 'SegmentationClass', 'Annotations', 'ImageSets/Segmentation']:
        os.makedirs(os.path.join(voc_root, sub), exist_ok=True)

    # Обрабатываем train/val/test
    if train_images:
        process_set('train', base_dir, train_images, train_masks, train_labels,
                    class_names, voc_root)
    if val_images:
        process_set('val', base_dir, val_images, val_masks, val_labels,
                    class_names, voc_root)
    if test_images and test_images != 'null':
        test_path = os.path.join(base_dir, test_images)
        if os.path.exists(test_path):
            process_set('test', base_dir, test_images, test_masks, test_labels,
                        class_names, voc_root)
        else:
            print(f"Предупреждение: папка {test_path} не найдена, test пропущен")

    print(f"Конвертация завершена. VOC-датасет создан в {voc_root}")

def interactive_mode():
    """Интерактивный режим с запросом параметров."""
    print("=== Интерактивный конвертер YOLO -> VOC ===")

    use_yaml = input("Использовать YAML-файл датасета YOLO (data.yaml)? (y/n): ").strip().lower()
    if (use_yaml == 'y' or use_yaml == 'н') and YAML_AVAILABLE:
        yaml_path = input("Путь к data.yaml: ").strip()
        if not os.path.isfile(yaml_path):
            print("Файл не найден.")
            return
        output_dir = input("Куда сохранить VOC-датасет? (по умолчанию ./VOC_output): ").strip()
        if not output_dir:
            output_dir = "./VOC_output"
        build_voc_from_yolo(yaml_path, output_dir)
    else:
        print("Ручной режим. Укажите:")
        base_dir = input("Корневая папка датасета (содержит images/, masks/ и labels/): ").strip()
        if not os.path.isdir(base_dir):
            print("Папка не найдена.")
            return

        class_names_str = input("Введите имена классов через запятую (например, object): ").strip()
        class_names = [c.strip() for c in class_names_str.split(',')] if class_names_str else ['object']

        train_images = input("Относительный путь к train изображениям (например images/train): ").strip()
        val_images = input("Относительный путь к val изображениям: ").strip()
        test_images = input("Относительный путь к test изображениям (если есть, иначе Enter): ").strip() or None

        train_masks = input("Относительный путь к train маскам (например masks/train): ").strip()
        val_masks = input("Относительный путь к val маскам: ").strip()
        test_masks = input("Относительный путь к test маскам (если есть, иначе Enter): ").strip() or None

        train_labels = input("Относительный путь к train YOLO .txt (например labels/train): ").strip()
        val_labels = input("Относительный путь к val YOLO .txt: ").strip()
        test_labels = input("Относительный путь к test YOLO .txt (если есть, иначе Enter): ").strip() or None

        output_dir = input("Куда сохранить VOC-датасет? (по умолчанию ./VOC_output): ").strip()
        if not output_dir:
            output_dir = "./VOC_output"

        voc_root = os.path.join(output_dir, 'VOC')
        for sub in ['JPEGImages', 'SegmentationClass', 'Annotations', 'ImageSets/Segmentation']:
            os.makedirs(os.path.join(voc_root, sub), exist_ok=True)

        if train_images:
            process_set('train', base_dir, train_images, train_masks, train_labels,
                        class_names, voc_root)
        if val_images:
            process_set('val', base_dir, val_images, val_masks, val_labels,
                        class_names, voc_root)
        if test_images:
            process_set('test', base_dir, test_images, test_masks, test_labels,
                        class_names, voc_root)

        print(f"Конвертация завершена. VOC-датасет создан в {voc_root}")

def main():
    parser = argparse.ArgumentParser(description="Конвертация YOLO датасета в VOC формат")
    parser.add_argument("--yaml", type=str, help="Путь к data.yaml")
    parser.add_argument("--output", type=str, default="./VOC_output", help="Папка для сохранения VOC")
    args = parser.parse_args()

    if args.yaml:
        # Проверка существования файла перед вызовом
        if not os.path.isfile(args.yaml):
            print(f"Ошибка: YAML-файл не найден по пути: {args.yaml}")
            sys.exit(1)
        build_voc_from_yolo(args.yaml, args.output)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()