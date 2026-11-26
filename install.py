#!/usr/bin/env python3
"""Install a consistent dependency profile into the current Python environment."""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from dependency_health import (
    OPENCV_PROVIDERS, SELECTED_OPENCV, TORCH_PROFILES,
    inherited_opencv, installed_opencv, installed_torch_profile,
    EXTRA_MODULES, installed_xgboost, selected_extras, selected_xgboost,
)

ROOT = Path(__file__).resolve().parent
EXTRAS = tuple(EXTRA_MODULES)


def run_pip(*args, dry_run=False):
    command = [sys.executable, '-m', 'pip', *args]
    print('+', ' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def check_installation_environment():
    """Fail before pip changes packages; never import cv2 just to diagnose it."""
    external = inherited_opencv()
    if external:
        locations = '; '.join(f'{name}: {path}' for name, path in external.items())
        raise ValueError(
            'OpenCV из глобального Python виден внутри .venv: ' + locations + '. '
            'pip не может удалить эти пакеты из текущего окружения. Создайте изолированное '
            'окружение без Inherit global site-packages и внешнего PYTHONPATH: '
            'python -m venv .venv_clean. Веса в models сохраняются.')
    if sys.platform != 'win32':
        return
    spec = importlib.util.find_spec('cv2')
    if spec is None:
        return
    files = []
    if spec.origin and spec.origin.lower().endswith('.pyd'):
        files.append(Path(spec.origin))
    for directory in spec.submodule_search_locations or ():
        files.extend(Path(directory).glob('*.pyd'))
    for path in files:
        try:
            # OPEN_EXISTING semantics: request write access without changing bytes.
            descriptor = os.open(path, os.O_RDWR | getattr(os, 'O_BINARY', 0))
            os.close(descriptor)
        except OSError as error:
            raise ValueError(
                f'Нельзя обновить {path}: {error}. Закройте приложение, Python Console '
                'в PyCharm и Jupyter, использующие это окружение, затем повторите установку. '
                'Если доступ по-прежнему запрещён, проверьте права и защиту файла. '
                'Проверка не изменяла DLL; pip ещё не запускался.') from error


def normalize_opencv(dry_run=False, force=False):
    """Repair shared cv2 files after removing competing OpenCV wheels."""
    providers = installed_opencv()
    competing = [name for name in OPENCV_PROVIDERS if name != SELECTED_OPENCV and name in providers]
    if dry_run:
        competing = [name for name in OPENCV_PROVIDERS if name != SELECTED_OPENCV]
    if competing:
        run_pip('uninstall', '-y', *competing, dry_run=dry_run)
    pin = next(line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
               if line.startswith(SELECTED_OPENCV + '=='))
    expected_version = pin.split('==', 1)[1]
    if force or competing or providers.get(SELECTED_OPENCV) != expected_version:
        # Removal can delete shared files; explicit repair also restores a lone
        # wheel whose metadata survived corruption. Pin drift must be corrected.
        run_pip('install', '--no-deps', '--force-reinstall', pin, dry_run=dry_run)


def normalize_xgboost(dry_run=False):
    """Restore the selected provider after removing a legacy wheel sharing xgboost/ files."""
    selected = selected_xgboost()
    competing = [name for name in installed_xgboost() if name != selected]
    if competing:
        run_pip('uninstall', '-y', *competing, dry_run=dry_run)
        pin = next(line.split(';')[0].strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                   if line.startswith(selected + '=='))
        run_pip('install', '--no-deps', '--force-reinstall', pin, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch', choices=TORCH_PROFILES,
                        help='Torch/torchvision profile (default: keep installed build; cpu for a new environment)')
    parser.add_argument('--extra', action='append', choices=(*EXTRAS, 'all'), default=[],
                        help='Optional feature dependencies; may be repeated')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without installing')
    parser.add_argument('--repair-opencv', action='store_true',
                        help='Only normalize cv2 providers and verify the existing environment')
    args = parser.parse_args(argv)
    if sys.prefix == sys.base_prefix and not args.dry_run:
        parser.error('Activate a virtual environment first: python -m venv .venv')
    try:
        if not args.dry_run:
            check_installation_environment()
        if not args.repair_opencv:
            selected_profile = args.torch or installed_torch_profile() or 'cpu'
            # One resolution includes the exact local Torch build and all extras.
            # A conflicting extra must fail rather than replace CUDA with CPU.
            requirements = ['-r', str(ROOT / 'requirements' / f'torch-{selected_profile}.txt')]
            extras = selected_extras(args.extra)
            for extra in sorted(extras):
                requirements.extend(['-r', str(ROOT / 'requirements' / f'optional-{extra}.txt')])
            run_pip('install', *requirements, dry_run=args.dry_run)
            normalize_xgboost(dry_run=args.dry_run)
        normalize_opencv(dry_run=args.dry_run, force=args.repair_opencv)
        if not args.dry_run:
            # No native modules have been imported by this installer. Validate in
            # a new process after all wheel files have reached their final state.
            verification = [sys.executable, str(ROOT / 'dependency_health.py'), '--load']
            if not args.repair_opencv:
                for extra in extras:
                    verification.extend(['--extra', extra])
                verification += ['--torch-profile', selected_profile]
                if selected_profile != 'cpu':
                    verification.append('--cuda-test')
            subprocess.run(verification,
                           cwd=ROOT, check=True)
            print('Installation verified. See any explicit OpenCV metadata exceptions above.')
        return 0
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f'Installation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
