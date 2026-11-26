"""Local portable model contracts shared by all inference and export screens."""
from pathlib import Path


GGUF_UNSUPPORTED = (
    "GGUF: для YOLO, U-Net, DeepLab и SAM нет подключённого совместимого "
    "экспортёра/runtime. GGUF хранит тензоры и метаданные, но не универсальный "
    "граф вычислений. Нужен backend конкретной архитектуры; используйте ONNX "
    "или NCNN. Переименование весов в .gguf не является конвертацией."
)


def reject_gguf(path):
    if Path(path).suffix.lower() == ".gguf":
        raise ValueError(GGUF_UNSUPPORTED)


def ncnn_param(path):
    path = Path(path).expanduser().resolve(strict=True)
    reject_gguf(path)
    if path.is_dir():
        params = sorted(path.glob("*.param"))
        if len(params) != 1:
            raise ValueError("В папке NCNN должен быть ровно один файл .param.")
        path = params[0]
    if path.suffix.lower() != ".param" or not path.with_suffix(".bin").is_file():
        raise ValueError("NCNN: нужны одноимённые файлы .param и .bin рядом.")
    # Reject an unrelated/broken file before entering the native parser.
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "7767517":
            raise ValueError("Неверный заголовок NCNN .param.")
    return path


class NCNNRuntime:
    """One CPU RGB CHW input / semantic CHW output; no preprocessing here."""
    def __init__(self, path):
        try:
            import ncnn
        except ImportError as error:
            raise RuntimeError("Для NCNN установите зависимости: python install.py --extra export") from error

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 2
        # Disable reduced precision for parity with CPU Torch / ONNX.
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        param = ncnn_param(path)
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(param.with_suffix(".bin"))) != 0:
            raise ValueError("NCNN не смог прочитать граф или веса.")
        self.inputs = list(self.net.input_names())
        self.outputs = list(self.net.output_names())

    def run(self, tensor):
        import numpy as np

        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("NCNN: поддерживается batch=1, NCHW.")
        with self.net.create_extractor() as extractor:
            status = extractor.input(self.inputs[0], self.ncnn.Mat(np.ascontiguousarray(tensor[0])))
            if status != 0:
                raise ValueError("NCNN: ошибка передачи входа.")
            outputs = []
            for name in self.outputs:
                status, value = extractor.extract(name)
                if status != 0:
                    raise ValueError(f"NCNN: ошибка вычисления выхода {name}.")
                outputs.append(np.array(value, copy=True)[None])
            return outputs


def load_yolo(path, **kwargs):
    """Load only local inference artifacts; normalize NCNN param to its folder."""
    from runtime_policy import configure_ultralytics

    configure_ultralytics()
    reject_gguf(path)
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Локальная модель YOLO не найдена: {candidate}")
    if candidate.suffix.lower() == ".onnx" and candidate.is_file():
        import onnxruntime as ort

        session = ort.InferenceSession(str(candidate), providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        task = metadata.get('task')
        if task not in ('detect', 'segment', 'obb', 'pose', 'classify'):
            raise ValueError("YOLO ONNX: отсутствует task в метаданных Ultralytics. Семантическую модель загружайте в ONNX / NCNN.")
        kwargs.setdefault('task', task)
        del session
    if candidate.is_dir() or candidate.suffix.lower() == ".param":
        import yaml

        param = ncnn_param(candidate)
        folder = param.parent
        if not folder.name.endswith("_ncnn_model"):
            raise ValueError("Для YOLO имя папки NCNN должно заканчиваться на _ncnn_model.")
        meta_file = folder / "metadata.yaml"
        if not meta_file.is_file():
            raise ValueError("YOLO NCNN: отсутствует metadata.yaml с task, names и imgsz.")
        metadata = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("task") not in ("detect", "segment", "obb", "pose", "classify"):
            raise ValueError("YOLO NCNN: неверный task в metadata.yaml.")
        kwargs.setdefault("task", metadata["task"])
        path = str(folder)
    from ultralytics import YOLO

    return YOLO(str(path), **kwargs)
