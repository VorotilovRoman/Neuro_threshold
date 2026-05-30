#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import platform
import importlib

# ------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------
def run_pip_install(package, extra_index=None):
    """Устанавливает пакет через pip и возвращает True/False."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_index:
        cmd.extend(["--index-url", extra_index])
    cmd.append(package)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Установлен: {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка установки {package}: {e.stderr}")
        return False

def get_pip():
    try:
        import pip
        return True
    except ImportError:
        return False

# ------------------------------------------------------------
# Установка PyTorch с поддержкой CUDA 13
# ------------------------------------------------------------
def install_pytorch():
    print("Установка PyTorch и torchvision...")
    # Пробуем официальный nightly‑индекс для CUDA 13 (если доступен)
    # Реальный индекс для cu130 отсутствует, поэтому используем последний стабильный с CUDA 12.4
    # (это удовлетворяет условию "или больше", т.к. 2.5.1 новее формально)
    torch_version = "2.11.0"
    torchvision_version = "0.26.0"
    cuda_version = "cu130"
    index_url = f"https://download.pytorch.org/whl/{cuda_version}"
    print(f"torch=={torch_version}")
    success = run_pip_install(f"torch=={torch_version}", extra_index=index_url)
    print(f"torchvision=={torchvision_version}")
    success &= run_pip_install(f"torchvision=={torchvision_version}", extra_index=index_url)
    if not success:
        print("Не удалось установить CUDA‑версию. Пробуем CPU‑версию.")
        run_pip_install("torch")
        run_pip_install("torchvision")
    # Дополнительно проверяем, установилась ли нужная версия
    try:
        import torch
        print(f"PyTorch установлен: {torch.__version__}, CUDA доступна: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch не установлен!")

# ------------------------------------------------------------
# Установка остальных пакетов из requirements.txt
# ------------------------------------------------------------
def install_other_packages():
    print("\nУстановка остальных библиотек...")
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Пропускаем torch и torchvision, т.к. уже обработаны
            if line.startswith("torch") or line.startswith("torchvision"):
                continue
            run_pip_install(line)

# ------------------------------------------------------------
# Проверка установки
# ------------------------------------------------------------
def verify_installation():
    print("\n=== Проверка установленных библиотек ===")
    modules = [
        ("cv2", "opencv-python"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLO"),
        ("albumentations", "Albumentations"),
        ("PyQt5.QtCore", "PyQt5"),
        ("sklearn", "scikit-learn"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]
    for mod, name in modules:
        try:
            m = __import__(mod)
            version = getattr(m, "__version__", "unknown")
            print(f"{name}: {version}")
        except ImportError:
            print(f"{name}: не установлен")

# ------------------------------------------------------------
# Главная функция
# ------------------------------------------------------------
def main():
    print("=== Установка зависимостей для проекта ===\n")
    print(f"ОС: {platform.system()}")
    print(f"Python: {sys.version}")

    if not get_pip():
        print("Ошибка: pip не найден. Установите pip и повторите попытку.")
        sys.exit(1)

    install_pytorch()
    install_other_packages()
    verify_installation()
    print("\nГотово!")

if __name__ == "__main__":
    main()