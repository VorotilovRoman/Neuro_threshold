"""Local-only prompt segmentation. Coordinates and masks use working-image pixels."""

from concurrent.futures import CancelledError
from dataclasses import dataclass
import importlib
from pathlib import Path

from runtime_torch import prepare_windows_torch, require_torch

prepare_windows_torch()

import cv2
import numpy as np

from utils_new.annotations.adapters import to_legacy_many
from utils_new.annotations.masks import mask_to_annotations
from utils_new.annotations.model import Annotation, ImageSize
from utils_new.images.processing import ImageStats
from utils_new.import_export.sam_weights import MODELS


MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def find_sam3_tokenizer(weights):
    """Find local BPE assets for a separately downloaded checkpoint, without I/O scans."""
    parent = Path(weights).expanduser().resolve().parent
    name = "bpe_simple_vocab_16e6.txt.gz"
    candidates = (parent / name, parent / "sam3" / name,
                  parent / "segment_anything" / "sam3" / name,
                  MODELS_DIR / "segment_anything" / "sam3" / name)
    return next((path for path in candidates if path.is_file()), None)


@dataclass(frozen=True)
class SamPrompt:
    points: tuple = ()
    labels: tuple = ()
    box: tuple | None = None
    text: str = ""

    def validated(self, width, height, *, supports_text=False):
        points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(self.labels)
        if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("У каждой точки должна быть метка 1 (объект) или 0 (фон)")
        if (not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()):
            raise ValueError("Точки должны находиться внутри снимка")
        box = None
        if self.box is not None:
            box = np.asarray(self.box, dtype=np.float32)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError("Рамка должна иметь формат x1,y1,x2,y2 и находиться внутри снимка")
        text = self.text.strip()
        if text and not supports_text:
            raise ValueError("Текст поддерживают SAM3 и SAM2 с Grounding DINO")
        if text and len(points):
            raise ValueError("Выберите текст с рамкой либо точки с рамкой")
        if not text and box is None and not (labels == 1).any():
            raise ValueError("Укажите рамку, точку объекта или текстовый запрос SAM3")
        return points, labels.astype(np.int32), box, text


@dataclass
class SamResult:
    masks: np.ndarray
    scores: np.ndarray
    detection_scores: np.ndarray | None = None
    phrases: tuple = ()
    detected_count: int | None = None

    def annotations(self, *, class_id=0, kind="segment", min_area=10):
        """One model instance per mask; explicitly discard holes for YOLO polygons."""
        if not isinstance(class_id, int) or class_id < 0:
            raise ValueError("ID класса должен быть неотрицательным целым числом")
        size = ImageSize(self.masks.shape[2], self.masks.shape[1])
        result = []
        for mask in self.masks:
            components = mask_to_annotations(mask.astype(np.uint8), kind=kind,
                                             min_area=min_area, holes="drop")
            result.extend(Annotation(a.kind, class_id, a.points_px) for a in components)
        return to_legacy_many(result, size)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError()


class LocalSAM:
    """Load once, reuse image embeddings; access serially from a worker."""

    def __init__(self):
        self.model = self.predictor = self.torch = None
        self.model_key = None
        self.device = "cpu"
        self._image_key = self._state = None
        self._image_ref = None
        self.prompt_mode = "points"

    @classmethod
    def load(cls, model_key, weights, *, device="auto", tokenizer=None, prompt_mode="points"):
        if model_key not in MODELS:
            raise ValueError(f"Неизвестная архитектура SAM: {model_key}")
        path = Path(weights).expanduser().resolve(strict=True)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Выберите непустой локальный файл весов")
        instance = cls()
        instance.torch = torch = require_torch()
        device = "cuda:0" if device in ("0", "cuda") else device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна; выберите CPU или проверьте профиль PyTorch")
        if device not in ("cpu", "cuda:0"):
            raise ValueError("Поддерживаются устройства auto, cpu и cuda:0")
        instance.device, instance.model_key = device, model_key
        if prompt_mode not in ("points", "text"):
            raise ValueError("Неизвестный режим подсказок")
        instance.prompt_mode = prompt_mode
        try:
            if model_key.startswith("sam-vit-"):
                package = importlib.import_module("segment_anything")
                model_type = "vit_" + model_key[-1]
                instance.model = package.sam_model_registry[model_type](checkpoint=str(path))
                instance.model.to(device=device).eval()
                instance.predictor = package.SamPredictor(instance.model)
            elif model_key.startswith(("sam2-", "sam2.1-")):
                config = MODELS[model_key].assets[1].path.removeprefix("sam2/")
                builder = importlib.import_module("sam2.build_sam").build_sam2
                instance.model = builder(config, str(path), device=device)
                predictor = importlib.import_module("sam2.sam2_image_predictor").SAM2ImagePredictor
                instance.predictor = predictor(instance.model)
            else:
                instance._load_sam3(path, tokenizer)
        except ImportError as error:
            package = "segment_anything" if model_key.startswith("sam-vit-") else "Ultralytics/CLIP" if model_key == "sam3" else "sam2"
            raise RuntimeError(f"Не удалось импортировать {package}: {error}. Установите официальный "
                               "пакет модели и его зависимости: python install.py --extra sam.") from error
        return instance

    def _load_sam3(self, path, tokenizer):
        # Official sam3 currently pins numpy<2. Use the project's pinned
        # Ultralytics implementation, with no downloads or pip at runtime.
        from runtime_policy import configure_ultralytics

        configure_ultralytics()
        checkpoint = self.torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            checkpoint = checkpoint["model"]
        if (not isinstance(checkpoint, dict)
                or not any(str(key).startswith("detector.backbone.") for key in checkpoint)
                or (self.prompt_mode == "points" and not any(str(key).startswith("tracker.") for key in checkpoint))):
            raise ValueError("Файл не содержит ожидаемых весов SAM3 для выбранного режима")
        del checkpoint
        predictors = importlib.import_module("ultralytics.models.sam.predict")
        builders = importlib.import_module("ultralytics.models.sam.build_sam3")
        if self.prompt_mode == "text":
            bpe = Path(tokenizer).expanduser().resolve(strict=True) if tokenizer else find_sam3_tokenizer(path)
            if bpe is None or not bpe.is_file():
                raise ValueError("Не найден bpe_simple_vocab_16e6.txt.gz: выберите токенизатор SAM3")
            clip = importlib.import_module("clip")  # Check before the builder can request installation.
            tokenizer = clip.simple_tokenizer.SimpleTokenizer(bpe_path=str(bpe))
            if not callable(tokenizer):
                raise RuntimeError("Нужен CLIP из Ultralytics: python install.py --extra sam")
            self.model = builders.build_sam3_image_model(str(path))
            self.model.backbone.language_backbone.tokenizer = tokenizer
            predictor_type = predictors.SAM3SemanticPredictor
        else:
            self.model = builders.build_interactive_sam3(str(path))
            predictor_type = predictors.SAM3Predictor
        self.predictor = predictor_type(overrides=dict(task="segment", mode="predict", model=str(path),
                                                       device=self.device, imgsz=1008,
                                                       half=False, save=False, verbose=False))
        self.predictor.setup_model(model=self.model, verbose=False)

    def _sam3_predict(self, height, width, points, labels, box, text, min_score):
        if not text:
            # Explicit batch axis groups FG/BG clicks into ONE object prompt.
            masks, boxes = self.predictor.inference_features(
                self.predictor.features, (height, width), bboxes=box,
                points=points[None] if len(points) else None,
                labels=labels[None] if len(points) else None, multimask_output=True)
            if masks is None:
                return np.empty((0, height, width), bool), np.empty(0)
            return _numpy(masks), _numpy(boxes)[:, 4]
        geometric = self.predictor._prepare_geometric_prompts((height, width), box, None)
        output = self.predictor._inference_features(self.predictor.features, *geometric, text=[text])
        scores = (output["pred_logits"].sigmoid()
                  * output["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        keep = scores >= min_score
        logits = output["pred_masks"][keep]
        if not len(logits):
            return np.empty((0, height, width), bool), np.empty(0)
        # Raw decoder outputs are logits. Official SAM3 uses sigmoid>0.5,
        # i.e. logits>0, rather than applying a 0.5 threshold to raw logits.
        masks = self.torch.nn.functional.interpolate(logits[:, None].float(), (height, width),
                                                     mode="bilinear", align_corners=False)[:, 0] > 0
        return _numpy(masks), _numpy(scores[keep])

    def predict(self, image, prompt, *, image_key=None, min_score=0.0, cancel=None):
        if self.model is None:
            raise RuntimeError("Модель SAM не загружена")
        if not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Порог качества должен находиться в [0, 1]")
        height, width = image.shape[:2]
        points, labels, box, text = prompt.validated(width, height, supports_text=self.model_key == "sam3")
        if self.model_key == "sam3" and bool(text) != (self.prompt_mode == "text"):
            raise ValueError("Сменился режим SAM3: загрузите модель для выбранного режима подсказок")
        key = (image_key, id(image), image.shape, image.dtype.str)
        _cancelled(cancel)
        with self.torch.inference_mode():
            if key != self._image_key:
                # If encoding fails/cancels, never reuse the previous image's features.
                self._image_key = self._state = self._image_ref = None
                rgb = ImageStats.to_rgb3(image)
                if self.model_key == "sam3":
                    self.predictor.reset_image()
                    self.predictor.set_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                else:
                    self.predictor.set_image(rgb)
                _cancelled(cancel)
                self._image_key, self._image_ref = key, image
            if self.model_key == "sam3":
                masks, scores = self._sam3_predict(height, width, points, labels, box, text, min_score)
            else:
                arguments = dict(point_coords=points if len(points) else None,
                                 point_labels=labels if len(points) else None,
                                 box=box, multimask_output=True)
                masks, scores, _ = self.predictor.predict(**arguments)
                masks, scores = _numpy(masks), _numpy(scores)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            scores = scores.reshape(-1)
            if (masks.ndim != 3 or masks.shape[1:] != (height, width)
                    or len(masks) != len(scores) or not np.isfinite(scores).all()
                    or not np.isfinite(masks).all()):
                raise ValueError("SAM вернул некорректные маски или оценки качества")
            if not text and len(scores):
                best = int(np.argmax(scores))
                masks, scores = masks[best:best + 1], scores[best:best + 1]
            keep = scores >= min_score
            _cancelled(cancel)
            return SamResult(masks[keep] > 0.5, scores[keep])
