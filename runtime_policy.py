"""Runtime dependency policy: library operations must never run pip implicitly."""
import os
import sys


def configure_ultralytics():
    """Keep optional backend installation under install.py's control.

    Ultralytics caches this environment flag at import time. Also disable the
    cached flags when a caller imported it before the application was loaded.
    This prevents ONNX/DNN/export checks from reinstalling a competing cv2 wheel.
    """
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    for name in ('ultralytics.utils', 'ultralytics.utils.checks'):
        module = sys.modules.get(name)
        if module is not None:
            module.AUTOINSTALL = False
