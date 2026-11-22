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


def export_onnx(model, metadata, imgsz, run_dir=None, *, log=print):
    """Optional export. Its failure must not hide successful training/validation."""
    try:
        import onnx
    except ImportError:
        log("[EXPORT][SKIPPED] ONNX не установлен; веса и метрики обучения сохранены.")
        return
    try:
        result = model.export(format="onnx", imgsz=imgsz, device="cpu", batch=1)
        path = Path(result) if result else None
        if path is None or not path.is_file():
            raise RuntimeError("Экспорт не создал ONNX-файл.")
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        info = {
            "format": "ONNX",
            "source_weights": metadata.get("validation_weights"),
            "onnx_opset": graph.opset_import[0].version if graph.opset_import else None,
            "model_type": metadata["model_type"],
            "task": metadata["task"],
            "imgsz": imgsz,
            "export_date": datetime.now().isoformat(),
            "description": "Выходы Ultralytics YOLO; требуется соответствующий task-декодер. Это не semantic NCHW-адаптер Custom ONNX.",
        }
        from utils_new.import_export.json_objects import write_json_atomic

        write_json_atomic(path.with_name(path.stem + "_onnx_meta.json"), info)
        if run_dir and path.parent.resolve() != (run_dir / "weights").resolve():
            destination = run_dir / "weights" / path.name
            shutil.copy2(path, destination)
            shutil.copy2(
                path.with_name(path.stem + "_onnx_meta.json"),
                destination.with_name(destination.stem + "_onnx_meta.json"),
            )
        log(f"[EXPORT][OK] ONNX проверен и сохранён: {path}")
    except Exception as error:  # noqa: BLE001 - optional export boundary
        log(f"[EXPORT][WARNING] Обучение завершено, но ONNX не экспортирован: {error}")
        log(traceback.format_exc())
