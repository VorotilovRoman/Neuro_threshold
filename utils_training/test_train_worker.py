#!/usr/bin/env python3
# test_train_worker.py
import subprocess
import json
import os
import sys
import time
from itertools import product

# ============================================================================
# Параметры моделей (пути приведены к вашей структуре после конвертации yolo2voc)
# ============================================================================

BASE_PROJECT_PATH = "C:/Users/romka/PycharmProjects/Neuro_threshold/result/test_train"
# После конвертации yolo2voc папка VOC находится, например, в ./VOC_output
BASE_DATASET_PATH = {
    "yolo": "C:/Users/romka/PycharmProjects/Neuro_threshold/datasetPCB_for_net_white",
    "unet": "C:/Users/romka/PycharmProjects/Neuro_threshold/datasetPCB_for_net_white",
    "deeplab": "C:/Users/romka/PycharmProjects/Neuro_threshold/datasetPCB_for_net_norm_masks/only_masks_dataset/dataset_voc_xml/VOC",          # <- путь к VOC-датасету (содержит JPEGImages, SegmentationClass, ImageSets)
    "segformer": "/home/user/dataset",
    "sam": "/home/user/sam_dataset"
}
EXPERIMENT_NAME = "test_run"
# ===== YOLO (0) =====
YOLO_PARAMS = {
    "model": ["yolov8n.pt", "yolo11s.pt", "yolo26m-seg.pt"],
    "data": [os.path.join(BASE_DATASET_PATH["yolo"], "data.yaml")],
    "project": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [5],
    "imgsz": [640],
    "batch": [-1],
    "device": ["auto"],
    "optimizer": ["auto"],
    "lr0": [0.001, 0.01],
    "lrf": [0.001, 0.01],
    "disable_aug": [True, False],
    "close_mosaic": [0],
    "mosaic": [0.0],
}
# ===== YOLO (0) =====
YOLO_PARAMS_all = {
    "model": ["yolov8n.pt", "yolo11n.pt", "yolo26n-seg.pt"],
    "data": [os.path.join(BASE_DATASET_PATH["yolo"], "data.yaml")],
    "project": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [5, 10, 20],
    "imgsz": [320, 640, 1280],
    "batch": [-1, 4, 8, 16],
    "device": ["auto"],
    "optimizer": ["auto", "SGD", "AdamW", "Adam", "MuSGD", "NAdam", "RMSProp"],
    "lr0": [0.0001, 0.001, 0.01],
    "lrf": [0.0001, 0.001, 0.01],
    "disable_aug": [True, False],
    "close_mosaic": [0, 5, 10],
    "mosaic": [0.0, 0.5, 1.0],
}

# ===== U-Net (1) =====
UNET_PARAMS = {
    "dataset_path": [BASE_DATASET_PATH["unet"]],
    "model_path": [None],
    "project_path": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [1],
    "batch_size": [-1],
    #"batch_size": [-1, 4],
    "device": ["auto"],
    "optimizer": ["Adam"],
    #"optimizer": ["Adam", "SGD"],
    "lr": [0.0001, 0.001],
    "encoder": ["resnet50"],
    "input_size": [512],
    "num_classes": [2],
    "loss": ["BCE", "Dice"],
    "dropout": [0.0, 0.2],
    "scheduler": ["None", "ReduceLROnPlateau"],
    "augmentations": [
        {"hflip": True, "vflip": True, "rotate": False, "scale": False, "noise": False},
        {"hflip": True, "vflip": True, "rotate": True, "scale": True, "noise": True},
    ],
}


# ===== U-Net (1) =====
UNET_PARAMS_all = {
    "dataset_path": [BASE_DATASET_PATH["unet"]],
    "model_path": [None],
    "project_path": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [1, 5, 10],
    "batch_size": [-1, 2, 4, 8, 16],
    "device": ["auto", "0", "cpu", "mps"],
    "optimizer": ["Adam", "SGD", "RMSprop", "AdamW"],
    "lr": [0.00001, 0.0001, 0.001],
    "encoder": ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "vgg16"],
    "input_size": [256, 512, 1024],
    "num_classes": [1, 2],
    "loss": ["BCE", "Dice", "BCE+Dice", "Focal", "Tversky"],
    "dropout": [0.0, 0.1, 0.2, 0.5],
    "scheduler": ["None", "ReduceLROnPlateau", "CosineAnnealing", "StepLR"],
    "augmentations": [
        {"hflip": True, "vflip": True, "rotate": False, "scale": False, "noise": False},
        {"hflip": True, "vflip": True, "rotate": True, "scale": True, "noise": True},
        {"hflip": False, "vflip": False, "rotate": False, "scale": False, "noise": False},
        {"hflip": True, "vflip": False, "rotate": False, "scale": False, "noise": False},
    ],
}



# ===== DeepLabV3+ (2) — обновлённые параметры, совместимые с train_deeplabv3_worker.py =====
DEEPLAB_PARAMS = {
    # Общие
    "model_path": [None],                         # None — использовать предустановленный backbone
    "dataset_path": [BASE_DATASET_PATH["deeplab"]],
    "project_path": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [2],                             # короткие тесты
    "batch_size": [-1],                        # -1 = auto batch
    "device": ["auto"],
    # Архитектура
    "backbone": ["resnet50"],
    "output_stride": [16],
    "output_activation": ["argmax"],              # 'argmax', 'sigmoid', 'softmax'
    "num_classes": [2],                           # 1 = бинарная сегментация, >1 = мультикласс
    "ignore_index": [None],                       # 255 для VOC, None — не игнорировать
    # Потери
    "loss": ["CrossEntropy+Dice"],
    "loss_weights": [{"ce": 1.0, "dice": 1.0, "focal": 0.0}],
    "class_weights": [None, "auto"],              # None, "auto" или путь к файлу
    # Оптимизация
    "optimizer": ["Adam"],
    "lr": [0.001],
    "momentum": [0.9],
    "weight_decay": [0.0001],
    "scheduler": ["Poly"],
    # Двухфазное обучение
    "two_phase_training": [False, True],
    "freeze_epochs": [2],
    "unfreeze_epochs": [3],
    # Аугментации
    "augmentations": [
        {"hflip": True, "vflip": True, "rotate": True, "scale": True, "noise": True},
    ],
    # Дополнительно
    "threshold": [0.5],                           # порог для бинарной сегментации
    "num_workers": [4],                           # количество workers для DataLoader
}
# ===== SegFormer (3) =====
SEGFORMER_PARAMS = {
    "model_path": [None],
    "dataset_path": [BASE_DATASET_PATH["segformer"]],
    "project_path": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [1, 5],
    "batch_size": [-1, 4],
    "device": ["auto"],
    "optimizer": ["AdamW", "SGD"],
    "lr": [0.0001, 0.0005],
    "variant": ["mit-b0", "mit-b1"],
    "num_classes": [2],
}

# ===== SAM (4) =====
SAM_PARAMS = {
    "model_path": [None],
    "dataset_path": [BASE_DATASET_PATH["sam"]],
    "project_path": [BASE_PROJECT_PATH],
    "name": [None],
    "epochs": [1, 5],
    "batch_size": [-1, 4],
    "device": ["auto"],
    "optimizer": ["AdamW", "Adam"],
    "lr": [0.0001, 0.001],
    "model_type": ["vit_b"],
    "prompt_type": ["box", "points"],
}
# Сопоставление индексов с именами воркеров (имя файла должно совпадать)
MODELS = {
    0: {"name": "YOLO", "params": YOLO_PARAMS, "worker": "train_yolo_worker.py"},
    1: {"name": "U-Net", "params": UNET_PARAMS, "worker": "train_unet_worker.py"},
    2: {"name": "DeepLabV3+", "params": DEEPLAB_PARAMS, "worker": "train_deeplabv3_worker.py"},  # исправлено имя
    3: {"name": "SegFormer", "params": SEGFORMER_PARAMS, "worker": "train_segformer_worker.py"},
    4: {"name": "SAM", "params": SAM_PARAMS, "worker": "train_sam_worker.py"},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER_BASE = os.path.join(SCRIPT_DIR, "..", "utils_training")
RESULTS_FILE = "results_test_model.txt"

def generate_combinations(params_dict):
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    for combination in product(*values):
        yield dict(zip(keys, combination))

def save_metrics(model_name, params, metrics):
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if os.path.getsize(RESULTS_FILE) == 0:
            f.write("model|parameters|metrics\n")
        params_str = json.dumps(params, ensure_ascii=False)
        metrics_str = json.dumps(metrics, ensure_ascii=False)
        f.write(f"{model_name}|{params_str}|{metrics_str}\n")

def run_worker(worker_name, params):
    worker_path = os.path.join(WORKER_BASE, worker_name)
    if not os.path.exists(worker_path):
        print(f"[ERROR] Воркер не найден: {worker_path}")
        return False

    params_json = json.dumps(params, ensure_ascii=False)
    cmd = [sys.executable, worker_path, "--params", params_json]

    print(f"\n>>> Запуск: {worker_name}")
    print(f">>> Параметры: {params}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, encoding='utf-8', errors='replace')

    for line in process.stdout:
        line = line.rstrip()
        print(f"[{worker_name}] {line}")
        if "METRICS:" in line:
            try:
                metrics_part = line.split("METRICS:")[-1].strip()
                metrics = json.loads(metrics_part)
                model_short = worker_name.replace('train_', '').replace('_worker.py', '')
                save_metrics(model_short, params, metrics)
            except Exception as e:
                print(f"[WARN] Не удалось распарсить METRICS: {e}")

    process.wait()
    if process.returncode == 0:
        print(f">>> {worker_name} завершён успешно.")
        return True
    else:
        print(f">>> {worker_name} завершён с ошибкой (код {process.returncode})")
        return False

def main():
    print("Доступные модели:")
    for idx, info in MODELS.items():
        print(f"{idx}: {info['name']}")
    try:
        model_idx = int(input("Выберите номер модели: "))
    except ValueError:
        print("Неверный ввод.")
        return

    if model_idx not in MODELS:
        print("Модель не найдена.")
        return

    model_info = MODELS[model_idx]
    params_dict = model_info["params"]
    worker_name = model_info["worker"]

    combinations = list(generate_combinations(params_dict))
    print(f"Всего комбинаций для {model_info['name']}: {len(combinations)}")

    try:
        limit = input("Сколько комбинаций запустить? (Enter - все): ")
        if limit.strip():
            limit = int(limit)
        else:
            limit = len(combinations)
    except ValueError:
        limit = len(combinations)

    success_count = 0
    for i, params in enumerate(combinations[:limit]):
        print(f"\n========== Запуск {i+1}/{min(limit, len(combinations))} ==========")
        name_suffix = f"test_{model_info['name']}_{i}"
        if "name" in params and params["name"] is not None:
            params["name"] = f"{params['name']}_{name_suffix}"
        else:
            params["name"] = name_suffix

        try:
            ok = run_worker(worker_name, params)
            if ok:
                success_count += 1
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n[MAIN] Прерывание текущего обучения пользователем (Ctrl+C).")
            print("Для выхода из программы нажмите Ctrl+C ещё раз.")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n[MAIN] Выход по запросу.")
                break

    print(f"\n===== ИТОГО: успешно {success_count} из {limit} =====")

if __name__ == "__main__":
    main()