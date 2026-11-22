"""Atomic model weights, per-checkpoint metadata and bounded training reports."""

from pathlib import Path
import csv
import json
import math


from utils_new.import_export.atomic import atomic_path
from utils_new.import_export.json_objects import write_json_atomic


def json_values(value):
    if isinstance(value, dict):
        return {str(k): json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_meta(metadata, model_path):
    path = Path(model_path)
    write_json_atomic(path.with_name(path.stem + "_meta.json"), json_values(metadata))


def save_weights(path, model_or_state, metadata):
    import torch

    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    state = {key: value.detach().cpu().clone() for key, value in state.items()}
    with atomic_path(path) as temporary:
        torch.save(state, temporary)
    save_model_meta(metadata, path)


def read_weights(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError(f"Ожидается state_dict сегментации: {path}")
    return state


def read_model_meta(path):
    path = Path(path).with_name(Path(path).stem + "_meta.json")
    if not path.exists():
        return {}
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict):
        raise ValueError(f"Некорректные метаданные: {path}")
    return values


def save_training_history(directory, history):
    directory = Path(directory)
    with atomic_path(directory / "results.csv") as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            keys = list(history[0]) if history else ["epoch"]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(json_values(history))
    write_json_atomic(directory / "history.json", dict(epochs=json_values(history)))


def save_training_reports(
    directory,
    history,
    confusion=None,
    probabilities=None,
    examples=(),
    train_examples=(),
):
    """Render small retained samples; no full-dataset logits and no pyplot state."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.metrics import precision_recall_curve

    directory = Path(directory)

    def figure(rows=1, cols=1, size=(10, 5)):
        value = Figure(figsize=size)
        FigureCanvasAgg(value)
        return value, value.subplots(rows, cols, squeeze=False)

    def save(value, name):
        with atomic_path(directory / name) as temporary:
            value.savefig(temporary, format=Path(name).suffix[1:], dpi=120)

    fig, axes = figure()
    for key in ("train_loss", "val_loss", "iou", "dice"):
        if history and any(row.get(key) is not None for row in history):
            axes[0, 0].plot(
                [r["epoch"] for r in history],
                [r.get(key, np.nan) for r in history],
                label=key,
            )
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    save(fig, "training_plots.png")
    if confusion is not None:
        matrix = np.asarray(confusion)
        fig, axes = figure()
        axes[0, 0].imshow(matrix, cmap="Blues")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")
        save(fig, "confusion_matrix.png")
        write_json_atomic(
            directory / "confusion_matrix.json", dict(matrix=matrix.tolist())
        )
    if probabilities is not None and len(probabilities[0]):
        scores, targets = probabilities
        fig, axes = figure()
        for column in range(scores.shape[1]):
            label = column if scores.shape[1] > 1 else 1
            precision, recall, _ = precision_recall_curve(
                (targets == label).astype(np.uint8), scores[:, column]
            )
            axes[0, 0].plot(recall, precision, label=str(label))
        axes[0, 0].set_xlabel("Recall")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].legend()
        save(fig, "pr_curve.png")
    for values, name in (
        (examples, "val_examples.jpg"),
        (train_examples, "train_batch_examples.jpg"),
    ):
        if not values:
            continue
        fig, axes = figure(len(values), 3, size=(10, len(values) * 2.6))
        for row, (image, target, prediction) in enumerate(values):
            for column, array in enumerate((image, target, prediction)):
                axes[row, column].imshow(array, cmap="tab20" if column else None)
                axes[row, column].axis("off")
        save(fig, name)


def export_semantic_onnx(model, metadata, output, *, log=print):
    """Export on CPU and publish only after graph and runtime validation."""
    import torch
    import numpy as np
    from utils_new.import_export.model_export import stride_padded_model

    try:
        import onnx
        import onnxruntime as ort

        model = model.to("cpu").eval()
        size = metadata["input_size"]
        dummy = torch.zeros(1, 3, size, size)
        with atomic_path(output, suffix=".onnx") as temporary:
            torch.onnx.export(
                stride_padded_model(model),
                dummy,
                temporary,
                opset_version=18,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
                dynamo=False,
            )
            onnx.checker.check_model(onnx.load(str(temporary)))
            session = ort.InferenceSession(
                str(temporary), providers=["CPUExecutionProvider"]
            )
            actual = session.run(None, {"input": dummy.numpy()})[0]
            with torch.inference_mode():
                expected = stride_padded_model(model)(dummy).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        save_model_meta(
            dict(metadata, format="ONNX", output_kind="logits", onnx_opset=18), output
        )
        log(f"[EXPORT][OK] {output}")
        return True
    except Exception as error:
        import traceback

        log(
            f"[EXPORT][WARNING] Веса сохранены, ONNX не экспортирован: {error}\n{traceback.format_exc()}"
        )
        return False
