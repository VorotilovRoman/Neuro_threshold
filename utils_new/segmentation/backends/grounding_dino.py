"""Offline Grounding DINO detection and text-to-mask composition with SAM2."""

from dataclasses import dataclass
import importlib
from pathlib import Path

from runtime_torch import prepare_windows_torch, require_torch

prepare_windows_torch()

import numpy as np

from utils_new.annotations.boxes import non_max_suppression_indices
from utils_new.images.processing import ImageStats
from utils_new.segmentation.backends.sam import LocalSAM, SamPrompt, SamResult, _cancelled, _numpy


@dataclass
class GroundingDetections:
    boxes: np.ndarray
    scores: np.ndarray
    phrases: tuple


class LocalGroundingDINO:
    """A complete Hugging Face snapshot, never a remote model identifier."""

    @staticmethod
    def local_directory(directory):
        path = Path(directory).expanduser().resolve()
        required = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer_config.json")
        missing = [name for name in required if not (path / name).is_file()]
        if missing or not ((path / "tokenizer.json").is_file() or (path / "vocab.txt").is_file()):
            raise ValueError("Нужна полная локальная папка Grounding DINO (Transformers): "
                             "config.json, model.safetensors, препроцессор и токенизатор. "
                             "Скачайте --models grounding-dino-tiny.")
        return path

    @classmethod
    def load(cls, directory, *, device="cpu"):
        path = cls.local_directory(directory)
        torch = require_torch()
        if device not in ("cpu", "cuda:0") or (device == "cuda:0" and not torch.cuda.is_available()):
            raise ValueError("Для Grounding DINO выберите CPU или доступную cuda:0")
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("Установите зависимости: python install.py --extra sam --extra grounding-dino") from error
        local = dict(local_files_only=True, trust_remote_code=False)
        config = transformers.AutoConfig.from_pretrained(str(path), **local)
        if config.model_type != "grounding-dino" or getattr(config, "use_timm_backbone", False):
            raise ValueError("Нужен Grounding DINO с встроенной конфигурацией backbone Transformers")
        # Avoid optional compiled/deformable CUDA kernels and backbone downloads.
        config.disable_custom_kernels = True
        config.use_pretrained_backbone = False
        instance = cls()
        instance.torch, instance.device = torch, device
        instance.processor = transformers.AutoProcessor.from_pretrained(str(path), **local)
        instance.model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            str(path), config=config, use_safetensors=True, dtype=torch.float32, **local).to(device).eval()
        return instance

    def detect(self, image, text, *, roi=None, box_threshold=0.35, text_threshold=0.25,
               nms_threshold=0.5, max_objects=50, cancel=None):
        if not all(np.isfinite(value) and 0 <= value <= 1
                   for value in (box_threshold, text_threshold, nms_threshold)):
            raise ValueError("Пороги Grounding DINO должны находиться в [0, 1]")
        if not isinstance(max_objects, int) or not 1 <= max_objects <= 500:
            raise ValueError("Количество объектов должно находиться в диапазоне 1–500")
        text = text.strip().lower()
        if not text:
            raise ValueError("Введите текстовый запрос Grounding DINO")
        if not text.endswith("."):
            text += "."
        height, width = image.shape[:2]
        if roi is not None:
            SamPrompt(box=tuple(roi)).validated(width, height)
            x1, y1 = np.floor(roi[:2]).astype(int)
            x2, y2 = np.ceil(roi[2:]).astype(int)
        else:
            x1, y1, x2, y2 = 0, 0, width, height
        _cancelled(cancel)
        from PIL import Image

        rgb = ImageStats.to_rgb3(image[y1:y2, x1:x2])
        with self.torch.inference_mode():
            inputs = self.processor(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(self.device)
            _cancelled(cancel)
            output = self.model(**inputs)
            _cancelled(cancel)
            result = self.processor.post_process_grounded_object_detection(
                output, input_ids=inputs["input_ids"], threshold=box_threshold,
                text_threshold=text_threshold, target_sizes=[(y2 - y1, x2 - x1)])[0]
        boxes, scores = _numpy(result["boxes"]).astype(np.float64), _numpy(result["scores"]).reshape(-1)
        phrases = tuple(result["text_labels"])
        if (boxes.shape != (len(scores), 4) or len(phrases) != len(scores)
                or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
            raise ValueError("Grounding DINO вернул некорректные рамки или оценки")
        # The detector operates on an ROI; SAM always receives full-image pixels.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, x2 - x1) + x1
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, y2 - y1) + y1
        valid = np.flatnonzero((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        keep = non_max_suppression_indices(boxes[valid], scores[valid], nms_threshold, box_threshold)[:max_objects]
        indices = valid[keep]
        _cancelled(cancel)
        return GroundingDetections(boxes[indices], scores[indices], tuple(phrases[index] for index in indices))


class GroundedSAM2:
    """Detect by text, segment every surviving box, retain separate confidence/IoU."""

    supports_text = True

    @classmethod
    def load(cls, model_key, weights, directory, *, device="auto", detector_device="cpu", cancel=None):
        if not model_key.startswith(("sam2-", "sam2.1-")):
            raise ValueError("Связка Grounding DINO доступна для SAM2 и SAM2.1")
        _cancelled(cancel)
        directory = LocalGroundingDINO.local_directory(directory)
        instance = cls()
        instance.sam = LocalSAM.load(model_key, weights, device=device)
        _cancelled(cancel)
        instance.detector = LocalGroundingDINO.load(directory, device=(instance.sam.device if detector_device == "same" else detector_device))
        instance.model_key, instance.device = model_key, instance.sam.device
        _cancelled(cancel)
        return instance

    def predict(self, image, prompt, *, image_key=None, min_score=0.0, cancel=None, **detector_options):
        height, width = image.shape[:2]
        _, _, box, text = prompt.validated(width, height, supports_text=True)
        if not text:
            raise ValueError("Для Grounding DINO нужен текстовый запрос")
        if not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Порог IoU SAM2 должен находиться в [0, 1]")
        detections = self.detector.detect(image, text, roi=box, cancel=cancel, **detector_options)
        masks, scores, confidences, phrases = [], [], [], []
        for detection_box, confidence, phrase in zip(detections.boxes, detections.scores, detections.phrases):
            _cancelled(cancel)
            result = self.sam.predict(image, SamPrompt(box=tuple(detection_box)), image_key=image_key,
                                      min_score=min_score, cancel=cancel)
            masks.extend(result.masks)
            scores.extend(result.scores)
            confidences.extend([confidence] * len(result.masks))
            phrases.extend([phrase] * len(result.masks))
        _cancelled(cancel)
        return SamResult(np.stack(masks) if masks else np.empty((0, height, width), bool),
                         np.asarray(scores), np.asarray(confidences), tuple(phrases), len(detections.boxes))
