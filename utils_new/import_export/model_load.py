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


def _native_path_unsafe(path):
    import sys
    return sys.platform == 'win32' and (not str(path).isascii() or len(str(path)) >= 240)


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
        # The native NCNN filename API uses narrow paths on Windows. Parse the
        # text from Python and stage only weights when the native path is unsafe.
        binary = param.with_suffix('.bin')
        if _native_path_unsafe(binary):
            import shutil
            import tempfile
            with tempfile.TemporaryDirectory(prefix='neuro-ncnn-') as directory:
                local = Path(directory) / 'weights.bin'
                if not str(local).isascii() or len(str(local)) >= 240:
                    raise ValueError('NCNN: задайте TEMP/TMP короткой папкой с латинским именем для загрузки весов.')
                shutil.copyfile(binary, local)
                status = self.net.load_param_mem(param.read_text(encoding='utf-8'))
                status = status or self.net.load_model(str(local))
        else:
            status = self.net.load_param(str(param)) or self.net.load_model(str(binary))
        if status != 0:
            raise ValueError("NCNN не смог прочитать граф или веса.")
        self.inputs = list(self.net.input_names())
        self.outputs = list(self.net.output_names())

    def run(self, tensor):
        import numpy as np

        if (not isinstance(tensor, np.ndarray) or tensor.ndim != 4
                or tensor.shape[:2] != (1, 3) or min(tensor.shape[2:]) < 1
                or tensor.dtype != np.float32 or not np.isfinite(tensor).all()):
            raise ValueError("NCNN: нужен конечный float32 массив [1,3,H,W] с положительными H/W.")
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
    path = str(candidate.resolve())
    graph_metadata = {}
    meta_target = candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Локальная модель YOLO не найдена: {candidate}")
    if candidate.suffix.lower() == ".onnx" and candidate.is_file():
        import onnxruntime as ort

        session = ort.InferenceSession(str(candidate), providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        graph_metadata = metadata
        task = metadata.get('task')
        if task not in ('detect', 'segment', 'obb', 'pose', 'classify'):
            raise ValueError("YOLO ONNX: отсутствует task в метаданных Ultralytics. Семантическую модель загружайте в ONNX / NCNN.")
        kwargs.setdefault('task', task)
        del session
    if candidate.is_dir() or candidate.suffix.lower() == ".param":
        import yaml

        param = ncnn_param(candidate)
        meta_target = param
        folder = param.parent
        if not folder.name.endswith("_ncnn_model"):
            raise ValueError("Для YOLO имя папки NCNN должно заканчиваться на _ncnn_model.")
        meta_file = folder / "metadata.yaml"
        if not meta_file.is_file():
            raise ValueError("YOLO NCNN: отсутствует metadata.yaml с task, names и imgsz.")
        metadata = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        graph_metadata = metadata if isinstance(metadata, dict) else {}
        if not isinstance(metadata, dict) or metadata.get("task") not in ("detect", "segment", "obb", "pose", "classify"):
            raise ValueError("YOLO NCNN: неверный task в metadata.yaml.")
        kwargs.setdefault("task", metadata["task"])
        path = str(folder)
    from ultralytics import YOLO
    from utils_new.import_export.training import read_model_meta

    temporary = None
    if candidate.is_dir() or candidate.suffix.lower() == '.param':
        if _native_path_unsafe(Path(path)):
            import shutil
            import tempfile
            temporary = tempfile.TemporaryDirectory(prefix='neuro-yolo-ncnn-')
            local = Path(temporary.name) / 'model_ncnn_model'
            if _native_path_unsafe(local):
                temporary.cleanup()
                raise ValueError('NCNN: задайте TEMP/TMP короткой папкой с латинским именем.')
            try:
                shutil.copytree(path, local)
            except BaseException:
                temporary.cleanup()
                raise
            path = str(local)
    try:
        model = YOLO(str(path), **kwargs)
    except BaseException:
        if temporary:
            temporary.cleanup()
        raise
    # Ultralytics loads the native backend lazily on first predict().
    if temporary:
        model._neuro_runtime_directory = temporary
    sidecar = read_model_meta(meta_target)
    task = getattr(model, 'task', None)
    if sidecar.get('task') and sidecar['task'] != task:
        raise ValueError('Тип задачи в метаданных не соответствует загруженной модели YOLO.')
    recipe = sidecar.get('export_parameters', {})
    if not isinstance(recipe, dict):
        raise ValueError('export_parameters должен быть JSON-объектом.')
    if graph_metadata.get('neuro_export'):
        import json

        embedded = json.loads(graph_metadata['neuro_export'])
        if recipe and recipe != embedded:
            raise ValueError('Параметры экспорта в JSON не соответствуют ONNX. Используйте метаданные этой модели.')
        recipe = embedded
    model.neuro_metadata = {**graph_metadata, **sidecar, 'export_parameters': recipe}
    native_schema=getattr(getattr(model,'model',None),'neuro_pose_schema',None)
    if native_schema:
        from utils_new.annotations.model import PoseSchema
        declared=model.neuro_metadata.get('pose_schema')
        if declared and PoseSchema(**declared)!=PoseSchema(**native_schema):
            raise ValueError('Схема Pose в JSON не соответствует сохранённой внутри весов.')
        model.neuro_metadata['pose_schema']=native_schema
    if 'imgsz' in model.neuro_metadata:
        import ast

        value = model.neuro_metadata['imgsz']
        value = ast.literal_eval(value) if isinstance(value, str) else value
        if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] == value[1]:
            value = value[0]
        model.neuro_metadata['imgsz'] = value
    if candidate.suffix.lower() == '.onnx' and recipe.get('nms'):
        from utils_new.import_export.model_export import validate_export_options

        recipe = validate_export_options('YOLO', 'ONNX', recipe)
        return _EmbeddedNMSModel(model, recipe)
    return model


class _EmbeddedNMSModel:
    """Do not pretend runtime options can undo filtering compiled into a graph."""
    def __init__(self, model, parameters):
        self._native, self._parameters = model, parameters

    def __getattr__(self, name):
        return getattr(self._native, name)

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream=stream, **kwargs)

    def predict(self, source=None, stream=False, **kwargs):
        return self._native.predict(source, stream=stream, **self._inference_options(kwargs))

    def val(self, **kwargs):
        return self._native.val(**self._inference_options(kwargs))

    def _inference_options(self, kwargs):
        import math

        for name in ('conf', 'iou', 'max_det'):
            kwargs.setdefault(name, self._parameters[name])
        if kwargs['conf'] < self._parameters['conf']:
            raise ValueError(f"NMS встроен в ONNX: conf нельзя уменьшить ниже {self._parameters['conf']}. Экспортируйте модель повторно без NMS или с меньшим conf.")
        if not math.isclose(kwargs['iou'], self._parameters['iou'], abs_tol=1e-8):
            raise ValueError(f"NMS встроен в ONNX: IoU фиксирован = {self._parameters['iou']}.")
        if kwargs['max_det'] > self._parameters['max_det']:
            raise ValueError(f"NMS встроен в ONNX: max_det не может превышать {self._parameters['max_det']}.")
        return kwargs
