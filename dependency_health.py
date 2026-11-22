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
TORCH_PROFILES = ('cpu', 'cu126', 'cu130')


def installed_torch_profile():
    """Read wheel metadata without initializing Torch or guessing a CUDA build."""
    try:
        version = metadata.version('torch')
    except metadata.PackageNotFoundError:
        return None
    profile = version.partition('+')[2]
    if profile in TORCH_PROFILES:
        return profile
    raise ValueError(
        f'Installed Torch {version} has no supported explicit build tag. '
        'Select --torch cpu, --torch cu126 or --torch cu130 to change it.'
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
        comparable = (actual.split('+')[0]
                      if name in {'torch', 'torchvision'} and '+' not in expected else actual)
        if comparable != expected:
            errors.append(f'{name}: expected {expected}, installed {actual}')
    return errors


def check_environment(torch_profile=None):
    """Return errors and explicitly allowed metadata exceptions.

    Ultralytics requests opencv-python by distribution name, but uses the same
    cv2 API from the headless wheel. PyQt supplies our GUI. Do not install both to make
    pip check quiet: their shared files would become ambiguous again.
    """
    providers = installed_opencv()
    errors = direct_requirement_errors()
    if torch_profile is not None:
        if torch_profile not in TORCH_PROFILES:
            raise ValueError(f'Unknown Torch profile: {torch_profile}')
        profile = Path(__file__).with_name('requirements') / f'torch-{torch_profile}.txt'
        errors.extend(direct_requirement_errors(profile))
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


def check_torch_device(torch, torchvision, *, profile=None, cuda_test=False):
    """Validate runtime selection and, when requested, real forward/backward/NMS.

    Imported wheels alone do not prove that the driver can execute their CUDA
    kernels. Keep this explicit so ordinary metadata checks never initialize CUDA.
    """
    cuda_version = torch.version.cuda
    if profile == 'cpu' and cuda_version is not None:
        raise RuntimeError(f'CPU profile requested, but Torch uses CUDA {cuda_version}.')
    if profile and profile.startswith('cu'):
        expected = f'{profile[2:-1]}.{profile[-1]}'
        if cuda_version != expected:
            raise RuntimeError(f'{profile} requires CUDA runtime {expected}; loaded {cuda_version}.')
    print('Torch CUDA runtime:', cuda_version or 'none (CPU build)')
    available = torch.cuda.is_available()
    print('CUDA available:', available)
    if not available:
        if cuda_test or (profile and profile != 'cpu'):
            raise RuntimeError(
                'CUDA is unavailable. Check the selected CUDA wheels, NVIDIA driver and GPU. '
                'The installer does not fall back to CPU; a separate CUDA Toolkit is not required.'
            )
        return
    print('GPU:', torch.cuda.get_device_name(0),
          'capability:', '.'.join(map(str, torch.cuda.get_device_capability(0))))
    print('Torch CUDA architectures:', ', '.join(torch.cuda.get_arch_list()))
    if not cuda_test:
        return
    device = 'cuda:0'
    image = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
    kernel = torch.randn(4, 3, 3, 3, device=device, requires_grad=True)
    logits = torch.nn.functional.conv2d(image, kernel, padding=1)
    logits.square().mean().backward()
    if not all(torch.isfinite(value).all().item()
               for value in (logits, image.grad, kernel.grad)):
        raise RuntimeError('CUDA forward/backward produced non-finite values.')
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]],
                         device=device)
    scores = torch.tensor([.9, .8, .7], device=device)
    indices = torchvision.ops.nms(boxes, scores, .5)
    torch.cuda.synchronize()
    if not indices.is_cuda or indices.tolist() != [0, 2]:
        raise RuntimeError('torchvision CUDA NMS returned an unexpected result.')
    print('CUDA TEST OK: convolution, backward and torchvision NMS executed on cuda:0.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', action='store_true',
                        help='Also import core runtime libraries in this fresh process')
    parser.add_argument('--torch-profile', choices=TORCH_PROFILES,
                        help='Require this exact Torch/torchvision CPU/CUDA wheel pair')
    parser.add_argument('--cuda-test', action='store_true',
                        help='Load libraries and execute CUDA forward/backward and torchvision NMS')
    args = parser.parse_args(argv)
    errors, notes = check_environment(args.torch_profile)
    print('OpenCV providers:', json.dumps(installed_opencv(), ensure_ascii=False))
    for note in notes:
        print('METADATA EXCEPTION:', note)
    for error in errors:
        print('ERROR:', error, file=sys.stderr)
    if (args.load or args.cuda_test) and not errors:
        loaded = {}
        # Match the application bootstrap: Torch before cv2/Qt native DLLs on Windows.
        for name in ('torch', 'torchvision', 'numpy', 'cv2', 'ultralytics',
                     'onnxruntime', 'albumentations', 'PyQt5.QtCore'):
            try:
                if name == 'ultralytics':
                    from runtime_policy import configure_ultralytics
                    configure_ultralytics()
                module = importlib.import_module(name)
                loaded[name] = module
                version = getattr(module, '__version__', getattr(module, 'QT_VERSION_STR', 'unknown'))
                print(name, version, getattr(module, '__file__', 'built-in'))
            except Exception as exc:
                errors.append(f'{name}: {exc}')
                print('ERROR:', errors[-1], file=sys.stderr)
        if not errors:
            try:
                check_torch_device(loaded['torch'], loaded['torchvision'],
                                   profile=args.torch_profile, cuda_test=args.cuda_test)
            except Exception as exc:
                errors.append(f'CUDA runtime: {exc}')
                print('ERROR:', errors[-1], file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
