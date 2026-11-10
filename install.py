#!/usr/bin/env python3
"""Install a consistent dependency profile into the current Python environment."""
import argparse
import subprocess
import sys
from pathlib import Path

from dependency_health import OPENCV_PROVIDERS, SELECTED_OPENCV, installed_opencv

ROOT = Path(__file__).resolve().parent
TORCH_PROFILES = ('cpu', 'cu130')
EXTRAS = ('training', 'audit', 'sahi', 'export')


def run_pip(*args, dry_run=False):
    command = [sys.executable, '-m', 'pip', *args]
    print('+', ' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def normalize_opencv(dry_run=False):
    """Repair shared cv2 files after removing competing OpenCV wheels."""
    providers = installed_opencv()
    competing = [name for name in OPENCV_PROVIDERS if name != SELECTED_OPENCV and name in providers]
    if dry_run:
        competing = [name for name in OPENCV_PROVIDERS if name != SELECTED_OPENCV]
    if competing:
        run_pip('uninstall', '-y', *competing, dry_run=dry_run)
    if competing or SELECTED_OPENCV not in providers:
        # Uninstall removes shared files even if the headless metadata survives.
        pin = next(line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                   if line.startswith(SELECTED_OPENCV + '=='))
        run_pip('install', '--no-deps', '--force-reinstall', pin, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch', choices=TORCH_PROFILES, default='cpu',
                        help='Explicit Torch/torchvision wheel profile (default: cpu)')
    parser.add_argument('--extra', action='append', choices=EXTRAS, default=[],
                        help='Optional feature dependencies; may be repeated')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without installing')
    parser.add_argument('--repair-opencv', action='store_true',
                        help='Only normalize cv2 providers and verify the existing environment')
    args = parser.parse_args(argv)
    if sys.prefix == sys.base_prefix and not args.dry_run:
        parser.error('Activate a virtual environment first: python -m venv .venv')
    try:
        if not args.repair_opencv:
            # Resolve the matching pair together. Never silently switch GPU/CPU profiles.
            run_pip('install', '-r', str(ROOT / 'requirements' / f'torch-{args.torch}.txt'),
                    dry_run=args.dry_run)
            requirements = ['-r', str(ROOT / 'requirements.txt')]
            for extra in sorted(set(args.extra)):
                requirements.extend(['-r', str(ROOT / 'requirements' / f'optional-{extra}.txt')])
            run_pip('install', *requirements, dry_run=args.dry_run)
        normalize_opencv(dry_run=args.dry_run)
        if not args.dry_run:
            # No native modules have been imported by this installer. Validate in
            # a new process after all wheel files have reached their final state.
            subprocess.run([sys.executable, str(ROOT / 'dependency_health.py'), '--load'],
                           cwd=ROOT, check=True)
            print('Installation verified. See any explicit OpenCV metadata exceptions above.')
        return 0
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f'Installation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
