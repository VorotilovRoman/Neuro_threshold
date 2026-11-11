"""One Torch initialization attempt per process, including failed DLL loads."""
import importlib
import sys
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class TorchRuntime:
    module: object = None
    device: str = 'cpu'
    gpu_count: int = 0
    gpu_name: str = ''
    error: str = ''


_lock = threading.RLock()
_runtime = None


def get_torch_runtime():
    """Cache failure as well as success; never retry a broken native loader."""
    global _runtime
    with _lock:
        if _runtime is not None:
            return _runtime
        try:
            torch = importlib.import_module('torch')
        except (ImportError, OSError, RuntimeError) as exc:
            _runtime = TorchRuntime(error=f'{type(exc).__name__}: {exc}')
            return _runtime
        try:
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            _runtime = TorchRuntime(torch, 'cuda:0' if count else 'cpu', count,
                                    torch.cuda.get_device_name(0) if count else '')
        except Exception as exc:
            # A failed CUDA probe does not mean a successfully imported Torch
            # cannot execute CPU operations. Keep the original diagnostic.
            _runtime = TorchRuntime(torch, error=f'CUDA probe: {type(exc).__name__}: {exc}')
        return _runtime


def prepare_windows_torch():
    """Load Torch before GUI/native stacks on Windows; keep Linux loading lazy.

    Load order can affect native DLL initialization. This is a precaution, not
    a substitute for valid Torch wheels and Windows runtime dependencies.
    """
    if sys.platform == 'win32':
        return get_torch_runtime()
    return None


def require_torch():
    """Return the loaded module or report the cached failure without retrying."""
    runtime = get_torch_runtime()
    if runtime.module is None:
        raise RuntimeError('PyTorch недоступен. ' + runtime.error)
    return runtime.module
