"""Dependency diagnostics without importing native libraries by default."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import re
import subprocess
import sys
from pathlib import Path

SELECTED_OPENCV = 'opencv-python-headless'
OPENCV_PROVIDERS = (
    'opencv-python', 'opencv-python-headless',
    'opencv-contrib-python', 'opencv-contrib-python-headless',
)


def installed_opencv():
    versions = {}
    for name in OPENCV_PROVIDERS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def direct_requirement_errors(requirements_file=None):
    """Validate the declared direct pins, accepting local Torch build suffixes."""
    requirements_file = requirements_file or Path(__file__).with_name('requirements.txt')
    errors = []
    for line in requirements_file.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s#]+)', line.strip())
        if not match:
            continue
        name, expected = match.groups()
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f'{name}: required {expected}, not installed')
            continue
        comparable = actual.split('+')[0] if name in {'torch', 'torchvision'} else actual
        if comparable != expected:
            errors.append(f'{name}: expected {expected}, installed {actual}')
    return errors


def check_environment():
    """Return errors and explicitly allowed metadata exceptions.

    Ultralytics requests opencv-python by distribution name, but uses the same
    cv2 API from the headless wheel. PyQt supplies our GUI. Do not install both to make
    pip check quiet: their shared files would become ambiguous again.
    """
    providers = installed_opencv()
    errors = direct_requirement_errors()
    notes = []
    if set(providers) != {SELECTED_OPENCV}:
        errors.append('Expected only opencv-python-headless; found: ' + repr(providers))
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                            capture_output=True, text=True, check=False)
    for line in result.stdout.splitlines():
        if line == 'No broken requirements found.':
            continue
        fields = line.lower().split()
        if (set(providers) == {SELECTED_OPENCV} and fields and
                fields[0] == 'ultralytics' and
                'requires opencv-python, which is not installed.' in line.lower()):
            notes.append(line + ' Using the selected headless cv2 provider (GUI is PyQt5).')
        elif line.strip():
            errors.append(line)
    if result.returncode and not (errors or notes):
        errors.append(result.stderr.strip() or 'pip check failed without diagnostics')
    return errors, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', action='store_true',
                        help='Also import core runtime libraries in this fresh process')
    args = parser.parse_args(argv)
    errors, notes = check_environment()
    print('OpenCV providers:', json.dumps(installed_opencv(), ensure_ascii=False))
    for note in notes:
        print('METADATA EXCEPTION:', note)
    for error in errors:
        print('ERROR:', error, file=sys.stderr)
    if args.load and not errors:
        for name in ('numpy', 'cv2', 'torch', 'torchvision', 'ultralytics',
                     'onnxruntime', 'albumentations', 'PyQt5.QtCore'):
            try:
                if name == 'ultralytics':
                    from runtime_policy import configure_ultralytics
                    configure_ultralytics()
                module = importlib.import_module(name)
                version = getattr(module, '__version__', getattr(module, 'QT_VERSION_STR', 'unknown'))
                print(name, version, getattr(module, '__file__', 'built-in'))
            except Exception as exc:
                errors.append(f'{name}: {exc}')
                print('ERROR:', errors[-1], file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
