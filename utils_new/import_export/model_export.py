"""Export the same stride padding/crop used during semantic model inference."""


def stride_padded_model(model):
    import torch

    class PaddedModel(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network
            self.stride = int(
                getattr(getattr(network, "encoder", None), "output_stride", 32)
            )

        def forward(self, image):
            height, width = image.shape[-2:]
            padded = torch.nn.functional.pad(
                image, (0, (-width) % self.stride, 0, (-height) % self.stride)
            )
            logits = self.network(padded)
            return logits[..., :height, :width]

    return PaddedModel(model).eval()


from pathlib import Path
from datetime import datetime
import shutil
import traceback

from utils_new.import_export.model_load import GGUF_UNSUPPORTED


def load_export_model(kind, path, *, model_key="sam2.1-tiny"):
    """Reuse inference loaders, on CPU; never guess a semantic architecture."""
    from runtime_torch import require_torch

    require_torch().set_num_threads(2)
    from utils_new.import_export.model_load import load_yolo, reject_gguf
    from utils_new.segmentation.backends.deep_learning import create_segmentor

    path = Path(path).expanduser().resolve(strict=True)
    reject_gguf(path)
    if kind == "YOLO":
        model = load_yolo(path)
        metadata = dict(getattr(model, 'neuro_metadata', {}), task=model.task, names=model.names, model_type="YOLO")
        return model, metadata
    if kind == "SAM":
        from utils_new.segmentation.backends.sam import load_sam_backend
        from utils_new.import_export.training import read_model_meta

        model = load_sam_backend(model_key, str(path), device="cpu")
        return model, dict(read_model_meta(path), model_type="SAM", model_key=model_key)
    model = create_segmentor(kind, device="cpu")
    model.load(**({"weights_path": str(path)} if kind in ("U-Net", "DeepLabV3+")
                  else {"model_path": str(path)}))
    return model, dict(model.metadata)


def export_capabilities(kind, path):
    """Capabilities reflect implemented executable contracts, not file extensions."""
    reasons = {"GGUF": GGUF_UNSUPPORTED}
    if kind == "SAM":
        reason = "SAM — модель с подсказками: нужны отдельные адаптеры encoder/decoder, точек, рамок и токенизатора. Универсальный экспорт не подключён."
    elif kind == "ONNX / NCNN" or (kind == "YOLO" and Path(path).suffix.lower() != ".pt"):
        reason = "Для экспорта загрузите исходные обучаемые веса PyTorch. Обратное преобразование ONNX/NCNN в PyTorch не поддерживается."
    else:
        reason = ""
    reasons.update(ONNX=reason, NCNN=reason)
    return reasons


def inspect_model(kind, path, *, model_key="sam2.1-tiny"):
    model, metadata = load_export_model(kind, path, model_key=model_key)
    network = getattr(model, "model", None)
    parameters = sum(value.numel() for value in network.parameters()) if hasattr(network, "parameters") else None
    return dict(kind=kind, path=str(Path(path).resolve()), metadata=metadata,
                parameters=parameters, capabilities=export_capabilities(kind, path))


def _semantic_metadata(metadata, size):
    if not {"normalize", "resize_mode"} <= metadata.keys():
        raise ValueError("В *_meta.json укажите normalize и resize_mode: предобработку нельзя угадывать при экспорте.")
    return dict(metadata, input_size=size, output_kind="logits")


def validate_export_options(kind, format, options=None):
    import math

    values = dict(opset=18, dynamic=kind != 'YOLO', nms=False,
                  conf=.25, iou=.45, max_det=300, threshold=.5)
    if options is not None:
        if not isinstance(options, dict) or set(options) - values.keys():
            raise ValueError('Неизвестные параметры экспорта.')
        values.update(options)
    if type(values['opset']) is not int or values['opset'] not in (13, 16, 18):
        raise ValueError('ONNX opset: 13, 16 или 18.')
    for key in ('dynamic', 'nms'):
        if type(values[key]) is not bool:
            raise ValueError(f'{key}: требуется bool.')
    for key in ('conf', 'iou', 'threshold'):
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key}: конечное число от 0 до 1.')
    if type(values['max_det']) is not int or not 1 <= values['max_det'] <= 3000:
        raise ValueError('max_det: целое число от 1 до 3000.')
    if values['nms'] and (kind != 'YOLO' or format != 'ONNX'):
        raise ValueError('Встроенный NMS доступен только для YOLO/ONNX. В NCNN NMS выполняется runtime.')
    if format == 'NCNN':
        values['dynamic'] = False
    return values


def export_semantic(model, metadata, size, target, format, *, options=None):
    """Strict semantic export; validate raw logits before publishing any artifact."""
    if format == "NCNN" and size % 32:
        raise ValueError("Для NCNN укажите размер входа, кратный 32 (например 512).")
    import numpy as np
    from runtime_torch import require_torch

    torch = require_torch()
    options = validate_export_options('semantic', format, options)
    model = stride_padded_model(model.to("cpu").eval())
    metadata = dict(_semantic_metadata(metadata, size), threshold=options['threshold'], export_parameters=options)
    # Nonzero deterministic inputs detect broken constants/channel ordering too.
    example = torch.from_numpy(np.random.default_rng(723).random((1, 3, size, size), dtype=np.float32))
    with torch.inference_mode():
        expected = model(example).numpy()
    if format == "ONNX":
        import onnx
        import onnxruntime as ort

        torch.onnx.export(model, example, str(target), opset_version=options['opset'], dynamo=False,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {2: "height", 3: "width"},
                                        "output": {2: "height", 3: "width"}} if options['dynamic'] else None)
        import json
        from utils_new.import_export.training import json_values

        graph = onnx.load(str(target), load_external_data=False)
        onnx.helper.set_model_props(graph, {'neuro_semantic': json.dumps(json_values(metadata))})
        onnx.save(graph, str(target))
        onnx.checker.check_model(onnx.load(str(target)))
        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        actual = session.run(None, {"input": example.numpy()})[0]
    elif format == "NCNN":
        import pnnx
        import subprocess
        from utils_new.import_export.model_load import NCNNRuntime

        target.mkdir()
        # TorchScript's native filename API can reject existing Unicode/long
        # Windows paths. Python opens the file; PNNX sees only relative names.
        traced = torch.jit.trace(model, example)
        with (target / 'trace.pt').open('wb') as stream:
            torch.jit.save(traced, stream)
        command = [pnnx.EXEC_PATH, 'trace.pt', f'inputshape=[1,3,{size},{size}]',
                   'device=cpu', 'fp16=0', 'pnnxparam=trace.pnnx.param',
                   'pnnxbin=trace.pnnx.bin', 'pnnxpy=trace_pnnx.py',
                   'pnnxonnx=trace.onnx', 'ncnnparam=model.ncnn.param',
                   'ncnnbin=model.ncnn.bin', 'ncnnpy=trace_ncnn.py']
        converted = subprocess.run(command, cwd=target, capture_output=True)
        if converted.returncode:
            diagnostic = (converted.stderr + converted.stdout).decode('utf-8', errors='replace')
            raise RuntimeError(f'PNNX conversion failed ({converted.returncode}):\n{diagnostic[-12000:]}')
        if not all((target / name).is_file() for name in ('model.ncnn.param', 'model.ncnn.bin')):
            raise RuntimeError('PNNX не создал файлы графа и весов NCNN.')
        runtime = NCNNRuntime(target / "model.ncnn.param")
        if len(runtime.inputs) != 1 or len(runtime.outputs) != 1:
            raise ValueError("NCNN: ожидался один вход и один выход семантической модели.")
        actual = runtime.run(example.numpy())[0]
        for path in target.iterdir():
            if path.name.startswith("trace"):
                path.unlink()
    else:
        raise ValueError(f"Неизвестный формат {format}.")
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4,
                               err_msg="Выход экспортированной модели отличается от PyTorch")
    # Second independent input catches shape-dependent and data-dependent errors.
    other = torch.flip(example, (-1,))
    with torch.inference_mode():
        reference = model(other).numpy()
    output = (session.run(None, {"input": other.numpy()})[0] if format == "ONNX"
              else runtime.run(other.numpy())[0])
    np.testing.assert_allclose(output, reference, rtol=2e-3, atol=2e-4)
    from utils_new.import_export.training import save_model_meta

    meta_target = target if format == "ONNX" else target / "model.ncnn.param"
    save_model_meta(dict(metadata, format=format, validation="CPU raw logits parity, two inputs"), meta_target)
    return target


def export_model(kind, path, format, output_dir, *, size=640, model_key="sam2.1-tiny", work_dir=None, options=None):
    """Export into a fresh bundle: source files and existing exports stay intact."""
    import tempfile
    import uuid

    if format not in ("ONNX", "NCNN", "GGUF"):
        raise ValueError(f"Неизвестный формат: {format}")
    reason = export_capabilities(kind, path)[format]
    if reason:
        raise ValueError(reason)
    options = validate_export_options(kind, format, options)
    if not isinstance(size, int) or not 32 <= size <= 4096:
        raise ValueError("Размер входа: целое число от 32 до 4096.")
    if kind == "YOLO" and size % 32:
        raise ValueError("Для YOLO укажите размер, кратный 32.")
    if format == "NCNN" and size % 32:
        raise ValueError("Для NCNN укажите размер входа, кратный 32 (например 512).")
    from importlib.util import find_spec

    required = ('onnx', 'onnxruntime') if format == 'ONNX' else ('ncnn', 'pnnx')
    missing = [name for name in required if find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Не установлены {', '.join(missing)}. Выполните: python install.py --extra export")
    destination = Path(output_dir).expanduser().resolve(strict=True)
    if not destination.is_dir():
        raise ValueError("Выберите папку результатов.")
    model, metadata = load_export_model(kind, path, model_key=model_key)
    if options['nms'] and metadata.get('task') == 'classify':
        raise ValueError('Классификатору YOLO NMS не применяется.')
    metadata = dict(metadata, export_parameters=options, source_weights=str(Path(path).resolve()),
                    export_date=datetime.now().isoformat())
    with tempfile.TemporaryDirectory(prefix=".export-", dir=work_dir or destination) as temporary:
        stage = Path(temporary)
        if kind == "YOLO":
            # Ultralytics writes next to its input: export a copy, never the original.
            source_dir = stage / 'source'
            source_dir.mkdir()
            source = source_dir / "model.pt"
            shutil.copy2(path, source)
            model.ckpt_path = str(source)
            model.model.pt_path = str(source)
            kwargs = dict(format=format.lower(), imgsz=size, batch=1, device="cpu", half=False)
            if format == "ONNX":
                kwargs.update(simplify=False, opset=options['opset'], dynamic=options['dynamic'],
                              nms=options['nms'], conf=options['conf'], iou=options['iou'], max_det=options['max_det'])
            captured = []
            model.add_callback("on_export_end", lambda exporter: captured.append((exporter.model, exporter.args)))
            exported = Path(model.export(**kwargs))
            if not captured:
                raise RuntimeError("Ultralytics не вернул исходную сеть для проверки экспорта.")
            reference, arguments = captured[0]
            if options['nms']:
                options['max_det'] = int(arguments.max_det)
            if format == 'ONNX':
                import json
                import onnx

                graph = onnx.load(str(exported), load_external_data=False)
                properties = {item.key: item.value for item in graph.metadata_props}
                properties['neuro_export'] = json.dumps(options)
                onnx.helper.set_model_props(graph, properties)
                onnx.save(graph, str(exported))
            if options['nms']:
                from ultralytics.engine.exporter import NMSModel

                reference = NMSModel(reference, arguments)
            _validate_yolo_export(reference, exported, size, format)
            from utils_new.import_export.training import save_model_meta

            if format == "ONNX":
                # Keep ONNX external-data files together with their graph.
                bundle = source_dir
                source.unlink()
                save_model_meta(dict(metadata, imgsz=size, format=format), bundle / 'model.onnx')
            else:
                bundle = exported
                save_model_meta(dict(metadata, imgsz=size, format=format), bundle / 'model.ncnn.param')
        else:
            bundle = stage / "bundle"
            bundle.mkdir()
            target = bundle / ("model.onnx" if format == "ONNX" else "model_ncnn_model")
            export_semantic(model.model, metadata, size, target, format, options=options)
            if format == "NCNN":
                bundle = target
        suffix = "_ncnn_model" if format == "NCNN" else "_onnx"
        for cache in bundle.rglob('__pycache__'):
            shutil.rmtree(cache)
        name = f"{Path(path).stem}_{uuid.uuid4().hex[:8]}{suffix}"
        published = destination / name
        # Staging is on the destination volume: publishing is one atomic rename.
        bundle.rename(published)
    return str(published)


def _validate_yolo_export(model, exported, size, format):
    """Check graph loading and raw outputs using Ultralytics' task-aware backend."""
    import numpy as np
    from runtime_torch import require_torch
    from ultralytics.nn.autobackend import AutoBackend

    torch = require_torch()
    if format == "ONNX":
        import onnx

        onnx.checker.check_model(onnx.load(str(exported)))
    backend = AutoBackend(str(exported), device=torch.device("cpu"))
    reference = model.to("cpu").eval()

    def arrays(value):
        if isinstance(value, (tuple, list)):
            return [item for part in value for item in arrays(part)]
        return [value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)]

    for seed in (173, 284):
        image = torch.from_numpy(np.random.default_rng(seed).random((1, 3, size, size), dtype=np.float32))
        with torch.inference_mode():
            expected, actual = arrays(reference(image)), arrays(backend(image))
        if len(expected) != len(actual):
            raise ValueError("YOLO: число выходов изменилось после экспорта.")
        # NCNN sorts blob names; match unique shapes before numeric comparison.
        for item in expected:
            matches = [i for i, output in enumerate(actual) if output.shape == item.shape]
            if not matches:
                raise ValueError(f"YOLO: отсутствует выход {item.shape}.")
            output = actual.pop(matches[0])
            np.testing.assert_allclose(output, item, rtol=5e-3, atol=5e-3)


def main(argv=None):
    """Isolated Qt export worker; structured results never mix with native logs."""
    import argparse
    import json
    import os
    from utils_new.import_export.json_objects import write_json_atomic
    from utils_new.import_export.training import json_values

    parser = argparse.ArgumentParser(description="Inspect/export local neural models")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    args.request = args.request.resolve()
    args.result = args.result.resolve()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        operation = request.pop("operation")
        if operation not in ('inspect', 'export'):
            raise ValueError(f'Неизвестная операция: {operation}')
        for key in ('path', 'output_dir', 'work_dir'):
            if request.get(key):
                request[key] = str(Path(request[key]).expanduser().resolve())
        # Native converters sometimes create/delete debug files in cwd.
        # Confine those effects to the job workspace, away from project files.
        os.chdir(args.request.parent)
        value = inspect_model(**request) if operation == "inspect" else export_model(**request)
        write_json_atomic(args.result, dict(ok=True, value=json_values(value)))
        return 0
    except Exception as error:
        traceback.print_exc()
        write_json_atomic(args.result, dict(ok=False, error=str(error)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
