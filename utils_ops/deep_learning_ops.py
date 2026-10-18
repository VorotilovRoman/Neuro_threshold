# deep_learning_ops.py
from import_libs_external import *

# ------------------ Логирование с немедленным выводом ------------------
def _log(msg):
    print(msg)
    sys.stdout.flush()

try:
    print(f"PyTorch version for deep_learning: {torch.__version__}")
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# Допустимые режимы ресайза
# Для ОБУЧЕНИЯ:      hard / keep_aspect_pad / random_crop
# Для ИНФЕРЕНСА:     hard / keep_aspect_pad  (random_crop не используется — теряет кадр)
VALID_TRAIN_RESIZE_MODES = ("hard", "keep_aspect_pad", "random_crop")
VALID_INFERENCE_RESIZE_MODES = ("hard", "keep_aspect_pad")


class BaseSegmentor:
    """Базовый класс для всех сегментаторов."""
    def __init__(self, model_path=None, device='cpu'):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.metadata = {}

    def load(self, **kwargs):
        raise NotImplementedError

    def predict(self, image, **kwargs):
        raise NotImplementedError


class TorchSegmentor(BaseSegmentor):
    """Базовый класс для PyTorch моделей."""
    def __init__(self, model_path=None, device='cpu'):
        super().__init__(model_path, device)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed")

    def _get_device(self):
        # Поддержка режима "auto"
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        elif self.device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif self.device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    # ------------------------------------------------------------
    # Разрешение режима ресайза для ИНФЕРЕНСА
    # ------------------------------------------------------------
    def _resolve_resize_mode(self):
        """
        Возвращает эффективный режим ресайза для ИНФЕРЕНСА.
        Гарантированно возвращает только 'hard' или 'keep_aspect_pad'.

        Логика:
          • metadata['resize_mode'] == 'hard'            → 'hard';
          • metadata['resize_mode'] == 'keep_aspect_pad' → 'keep_aspect_pad';
          • metadata['resize_mode'] == 'random_crop'     → берём
            metadata['inference_resize_mode'] (задаётся пользователем в UI);
            если не задано — безопасный дефолт 'keep_aspect_pad'
            (не теряет информацию о кадре);
          • backward compat: старое поле metadata['keep_aspect_ratio'] (bool).
        """
        mode = self.metadata.get('resize_mode')

        if mode == 'hard':
            return 'hard'
        if mode == 'keep_aspect_pad':
            return 'keep_aspect_pad'
        if mode == 'random_crop':
            inference_mode = self.metadata.get('inference_resize_mode')
            if inference_mode in VALID_INFERENCE_RESIZE_MODES:
                return inference_mode
            # Safe default — не теряем кадр
            return 'keep_aspect_pad'

        # Backward compat со старым bool-полем
        if self.metadata.get('keep_aspect_ratio', False):
            return 'keep_aspect_pad'
        return 'hard'

    # ------------------------------------------------------------
    # Общий препроцессинг
    # ------------------------------------------------------------
    def _preprocess(self, image, input_size):
        """
        Ресайз + опциональный padding + опциональная нормализация.

        Режим инференса берётся из self._resolve_resize_mode() —
        только 'hard' или 'keep_aspect_pad'. Никаких кропов —
        информация о кадре не теряется.

        Возвращает:
            tensor, (orig_h, orig_w), (new_h, new_w), pad_info
            pad_info = (pad_left, pad_top, new_h, new_w) — для обратного вырезания
                       рабочей области из маски.
        """
        _log(f"[Preprocess] Входное изображение: shape={image.shape}, dtype={image.dtype}")

        # --- Каналы в RGB ---
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        normalize = bool(self.metadata.get('normalize', False))
        resize_mode = self._resolve_resize_mode()

        if isinstance(input_size, int):
            tgt_h, tgt_w = input_size, input_size
        else:
            tgt_h, tgt_w = input_size

        pad_info = (0, 0, tgt_h, tgt_w)

        if resize_mode == 'keep_aspect_pad':
            scale = min(tgt_h / h, tgt_w / w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            resized = cv2.resize(image, (new_w, new_h))
            pad_h = tgt_h - new_h
            pad_w = tgt_w - new_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            image = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            pad_info = (pad_left, pad_top, new_h, new_w)
        else:  # 'hard'
            image = cv2.resize(image, (tgt_w, tgt_h))
            pad_info = (0, 0, tgt_h, tgt_w)

        # --- Тензор + нормализация ---
        ops = [T.ToTensor()]
        if normalize:
            ops.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ))
        tensor = T.Compose(ops)(image).unsqueeze(0)
        _log(f"[Preprocess] Тензор shape={tensor.shape}, device={self._get_device()}, "
             f"normalize={normalize}, resize_mode={resize_mode}")
        return tensor, (h, w), (tgt_h, tgt_w), pad_info

    # ------------------------------------------------------------
    # Обратное преобразование маски к исходному размеру
    # ------------------------------------------------------------
    def _postprocess_mask(self, mask_np, orig_size, pad_info, resize_mode):
        """
        Приводит маску (H', W') к исходному (orig_h, orig_w).

        Для 'keep_aspect_pad':
            вырезаем padding-область, затем ресайзим NEAREST.
        Для 'hard':
            просто ресайзим NEAREST.
        """
        if resize_mode == 'keep_aspect_pad':
            pl, pt, nh, nw = pad_info
            # Обрезаем padding (если рабочая область меньше целевого квадрата)
            mask_np = mask_np[pt:pt + nh, pl:pl + nw]

        mask_np = cv2.resize(
            mask_np, (orig_size[1], orig_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask_np


class UNetSegmentor(TorchSegmentor):
    """
    Сегментатор U-Net с использованием библиотеки segmentation_models_pytorch.
    При загрузке требует файл *_meta.json рядом с весами.

    Поддерживает:
        num_classes = 1  → бинарная через sigmoid + threshold
        num_classes = 2  → «фон + один объект»: softmax[:, 1] + threshold
        num_classes > 2  → multiclass через softmax + argmax

    Препроцессинг:
        normalize           — ImageNet-нормализация (bool)
        resize_mode         — 'hard' / 'keep_aspect_pad' / 'random_crop'
                              (random_crop на инференсе заменяется на
                               inference_resize_mode, см. _resolve_resize_mode)
        inference_resize_mode — 'hard' / 'keep_aspect_pad', используется только
                                если обучение шло с random_crop.
    """
    def load(self, weights_path=None, encoder=None, input_size=None):
        """
        Загружает модель U-Net.

        Параметры:
            weights_path (str): путь к файлу весов (.pth)
            encoder (str): игнорируется, если есть метафайл
            input_size (int): игнорируется, если есть метафайл

        Исключения:
            FileNotFoundError: если файл весов или метафайл не найден
            ValueError: если метафайл имеет неверный формат
        """
        import segmentation_models_pytorch as smp
        import json

        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл модели U-Net не найден: {weights_path}")

        # Поиск метафайла
        meta_path = os.path.splitext(weights_path)[0] + '_meta.json'
        if not os.path.exists(meta_path):
            raise ValueError(
                f"Файл метаданных не найден: {meta_path}\n"
                "Для загрузки модели U-Net необходим файл *_meta.json в той же папке.\n"
                "Формат файла:\n"
                "{\n"
                '  "encoder": "resnet50",                    # имя энкодера\n'
                '  "input_size": 512,                        # размер входа\n'
                '  "num_classes": 2,                         # число классов, включая фон\n'
                '  "decoder_channels": [256,128,64,32,16],   # опционально\n'
                '  "normalize": false,                       # ImageNet-нормализация\n'
                '  "resize_mode": "hard",                    # hard | keep_aspect_pad | random_crop\n'
                '  "inference_resize_mode": "keep_aspect_pad"  # только если обучение шло с random_crop\n'
                "}\n"
                "Убедитесь, что метафайл создан при обучении модели."
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # --- Архитектурные параметры ---
        encoder_name = meta.get('encoder')
        input_size_val = meta.get('input_size')
        num_classes = meta.get('num_classes', 1)
        decoder_channels = meta.get('decoder_channels', (256, 128, 64, 32, 16))
        architecture = meta.get('architecture', 'unet').lower()

        if not encoder_name or not input_size_val:
            raise ValueError(
                "Метафайл должен содержать поля 'encoder' и 'input_size'.\n"
                f"Текущее содержимое: {meta}"
            )


        # --- Модель (ВЫБОР АРХИТЕКТУРЫ) ---
        if architecture in ('unet++', 'unetplusplus'):
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=num_classes,
                decoder_channels=decoder_channels,
            )
        else:
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=3,
                classes=num_classes,
                decoder_channels=decoder_channels,
            )

        # --- Веса ---
        state_dict = torch.load(weights_path, map_location='cpu')

        cleaned = {k: v for k, v in state_dict.items()
                   if not k.endswith('num_batches_tracked')}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            _log(f"[UNet] Отсутствуют ключи: {missing[:3]}...")
        if unexpected:
            _log(f"[UNet] Лишние ключи (игнорируются): {unexpected[:3]}...")

        self.model.to(self._get_device())
        self.model.eval()
        _log(f"[UNet] Модель загружена, устройство: {self._get_device()}")



        # --- Формируем metadata ТОЛЬКО из того, что реально есть в файле ---
        # Ключевое отличие: если параметра нет в meta — мы НЕ записываем
        # его со значением по умолчанию. Это позволяет UI различить
        # «параметр известен из обучения» vs «параметр неизвестен,
        # пользователь должен выбрать сам».
        self.metadata = {
            'model_type': 'U-Net++' if architecture in ('unet++', 'unetplusplus') else 'U-Net',
            'architecture': architecture,
            'encoder': encoder_name,
            'input_size': input_size_val,
            'num_classes': num_classes,
            'decoder_channels': decoder_channels,
            'weights_path': weights_path,
        }

        # --- normalize: добавляем только если был в файле ---
        if 'normalize' in meta:
            self.metadata['normalize'] = bool(meta['normalize'])

        # --- resize_mode: добавляем только если был в файле ---
        # Обратная совместимость: старый bool-ключ keep_aspect_ratio
        if 'resize_mode' in meta and meta['resize_mode'] in VALID_TRAIN_RESIZE_MODES:
            self.metadata['resize_mode'] = meta['resize_mode']
            self.metadata['keep_aspect_ratio'] = (meta['resize_mode'] == 'keep_aspect_pad')
        elif 'keep_aspect_ratio' in meta:
            # Старое поле: конвертируем в новый формат
            self.metadata['resize_mode'] = ('keep_aspect_pad'
                                            if meta['keep_aspect_ratio'] else 'hard')
            self.metadata['keep_aspect_ratio'] = bool(meta['keep_aspect_ratio'])

        # --- inference_resize_mode: добавляем только если был в файле
        # (или если train resize_mode == random_crop и мы хотим явный
        # safe-default, чтобы пользователю не пришлось выбирать заново
        # при перезагрузке). ---
        if 'inference_resize_mode' in meta and \
                meta['inference_resize_mode'] in VALID_INFERENCE_RESIZE_MODES:
            self.metadata['inference_resize_mode'] = meta['inference_resize_mode']
        elif self.metadata.get('resize_mode') == 'random_crop':
            # Для random_crop даём безопасный дефолт — keep_aspect_pad
            self.metadata['inference_resize_mode'] = 'keep_aspect_pad'

        _log(f"[UNet] Метаданные загружены: encoder={encoder_name}, "
             f"input_size={input_size_val}, num_classes={num_classes}")
        _log(f"[UNet]   присутствуют в meta: "
             f"normalize={'yes' if 'normalize' in self.metadata else 'NO'}, "
             f"resize_mode={'yes' if 'resize_mode' in self.metadata else 'NO'}, "
             f"inference_resize_mode="
             f"{'yes' if 'inference_resize_mode' in self.metadata else 'NO'}")




    def predict(self, image, threshold=0.5):
        """
        Инференс U-Net.

        Логика по num_classes:
            1 → sigmoid + threshold
            2 → softmax[:, 1] + threshold (фон + объект)
           >2 → softmax + argmax (истинный multiclass)

        Параметр threshold игнорируется для num_classes > 2.
        """
        start_time = time.time()

        input_size = self.metadata.get('input_size', 512)
        num_classes = self.metadata.get('num_classes', 1)
        train_resize_mode = self.metadata.get('resize_mode', '?')
        resize_mode = self._resolve_resize_mode()

        _log(f"[UNet predict] Начало: shape={image.shape}, "
             f"num_classes={num_classes}, threshold={threshold}, "
             f"train_resize_mode={train_resize_mode}, "
             f"inference_resize_mode={resize_mode}")

        tensor, orig_size, _, pad_info = self._preprocess(image, input_size)
        device = self._get_device()
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = self.model(tensor)   # [1, C, H, W]
            if num_classes == 1:
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()   # [H, W]
                mask_np = (probs > threshold).astype(np.uint8) * 255
            elif num_classes == 2:
                # «Фон + один объект»: берём вероятность класса «объект» (index 1)
                probs = torch.softmax(logits, dim=1)                    # [1, 2, H, W]
                obj_prob = probs[0, 1].cpu().numpy()                    # [H, W]
                mask_np = (obj_prob > threshold).astype(np.uint8) * 255
            else:
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                mask_np = pred_class.squeeze().cpu().numpy().astype(np.uint8)
                mask_np = (mask_np * (255 // (num_classes - 1))).astype(np.uint8)

        _log(f"[UNet predict] Предсказание shape={mask_np.shape}, "
             f"min={mask_np.min()}, max={mask_np.max()}")

        # Обратный ресайз к исходному размеру (без center crop)
        mask_np = self._postprocess_mask(mask_np, orig_size, pad_info, resize_mode)

        elapsed = time.time() - start_time
        _log(f"[UNet predict] Готово за {elapsed:.3f} сек. "
             f"Выходная маска shape={mask_np.shape}")
        return mask_np



class DeepLabV3Segmentor(TorchSegmentor):
    """
    Сегментатор DeepLabV3 / DeepLabV3+ на основе segmentation_models_pytorch.
    При загрузке требует файл *_meta.json рядом с весами.

    Поддерживает:
        num_classes = 1  → sigmoid + threshold
        num_classes = 2  → softmax[:, 1] + threshold
                           (или sigmoid+threshold, если output_activation='sigmoid')
        num_classes > 2  → softmax + argmax (класс масштабируется в 0..255)

    Препроцессинг:
        normalize             — ImageNet-нормализация (bool)
        resize_mode           — 'hard' / 'keep_aspect_pad' / 'random_crop'
                                (random_crop на инференсе заменяется на
                                 inference_resize_mode, см. _resolve_resize_mode)
        inference_resize_mode — 'hard' / 'keep_aspect_pad', используется только
                                если обучение шло с random_crop.

    Архитектура:
        meta['architecture']  — 'deeplabv3plus' | 'deeplabv3'
                                (если нет — считается deeplabv3plus для обратной совместимости)
    """
    def load(self, weights_path=None, backbone=None, output_stride=None):
        """
        Загружает модель DeepLabV3/V3+.

        Параметры:
            weights_path (str): путь к файлу весов (.pth) — обязателен
            backbone (str): игнорируется, если в *_meta.json есть 'backbone'
            output_stride (int): игнорируется, если есть в *_meta.json

        Исключения:
            FileNotFoundError: если файл весов не найден
            ValueError: если отсутствует *_meta.json
        """
        import segmentation_models_pytorch as smp
        import json

        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл модели DeepLab не найден: {weights_path}")

        # ---------- Поиск и чтение метафайла ----------
        meta_path = os.path.splitext(weights_path)[0] + '_meta.json'
        if not os.path.exists(meta_path):
            raise ValueError(
                f"Файл метаданных не найден: {meta_path}\n"
                "Для загрузки DeepLabV3/V3+ необходим *_meta.json в той же папке.\n"
                "Формат файла:\n"
                "{\n"
                '  "architecture": "deeplabv3plus",          # или "deeplabv3"\n'
                '  "backbone": "resnet50",\n'
                '  "output_stride": 16,                      # 8 | 16 | 32\n'
                '  "atrous_rates": [6, 12, 18],              # опционально\n'
                '  "input_size": 513,                        # 513 (OS=16) | 1025 (OS=8)\n'
                '  "num_classes": 21,\n'
                '  "output_activation": "argmax",           # argmax | softmax | sigmoid\n'
                '  "normalize": true,\n'
                '  "resize_mode": "hard"\n'
                "}\n"
                "Убедитесь, что метафайл создан при обучении модели."
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # ---------- Архитектурные параметры ----------
        architecture = str(meta.get('architecture', 'deeplabv3plus')).lower()
        if architecture in ('deeplabv3plus', 'deeplabv3+'):
            architecture = 'deeplabv3plus'
        elif architecture == 'deeplabv3':
            architecture = 'deeplabv3'
        else:
            _log(f"[DeepLab] Неизвестная архитектура '{architecture}', "
                 f"использую deeplabv3plus")
            architecture = 'deeplabv3plus'

        # Backbone: приоритет — meta, затем аргумент, затем дефолт
        backbone_name = meta.get('backbone') or backbone or 'resnet50'

        # Output stride: приоритет — meta, затем аргумент, затем дефолт
        output_stride_val = meta.get('output_stride', output_stride if output_stride else 16)
        try:
            output_stride_val = int(output_stride_val)
        except (TypeError, ValueError):
            output_stride_val = 16

        num_classes = int(meta.get('num_classes', 21))
        input_size_val = meta.get('input_size', 513)
        output_activation = str(meta.get('output_activation', 'argmax')).lower()
        if output_activation not in ('argmax', 'softmax', 'sigmoid'):
            output_activation = 'argmax'

        # ---------- Atrous rates ----------
        # Приоритет: meta → авто по OS
        atrous_rates_meta = meta.get('atrous_rates')
        if atrous_rates_meta:
            try:
                atrous_rates = tuple(int(r) for r in atrous_rates_meta)
            except (TypeError, ValueError):
                atrous_rates = None
        else:
            atrous_rates = None

        if atrous_rates is None:
            atrous_rates = {
                32: (3, 6, 9),
                16: (6, 12, 18),
                8:  (12, 24, 36),
            }.get(output_stride_val, (6, 12, 18))

        # ---------- Создание модели ----------
        _log(f"[DeepLab] Создаю {architecture}: backbone={backbone_name}, "
             f"OS={output_stride_val}, atrous_rates={atrous_rates}, "
             f"classes={num_classes}")

        if architecture == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone_name,
                encoder_weights=None,       # веса загрузим из файла
                in_channels=3,
                classes=num_classes,
                encoder_output_stride=output_stride_val,
                decoder_atrous_rates=atrous_rates,
                decoder_use_batchnorm=True,
            )
        else:  # deeplabv3
            self.model = smp.DeepLabV3(
                encoder_name=backbone_name,
                encoder_weights=None,
                in_channels=3,
                classes=num_classes,
                encoder_output_stride=output_stride_val,
                decoder_atrous_rates=atrous_rates,
            )

        # ---------- Загрузка весов ----------
        state_dict = torch.load(weights_path, map_location='cpu')
        cleaned = {k: v for k, v in state_dict.items()
                   if not k.endswith('num_batches_tracked')}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            _log(f"[DeepLab] Отсутствуют ключи ({len(missing)}): {missing[:3]}...")
        if unexpected:
            _log(f"[DeepLab] Лишние ключи ({len(unexpected)}): {unexpected[:3]}...")

        self.model.to(self._get_device())
        self.model.eval()
        _log(f"[DeepLab] Модель загружена, устройство: {self._get_device()}")

        # ---------- Формируем metadata ----------
        self.metadata = {
            'model_type': 'DeepLabV3+' if architecture == 'deeplabv3plus' else 'DeepLabV3',
            'architecture': architecture,
            'backbone': backbone_name,
            'output_stride': output_stride_val,
            'atrous_rates': list(atrous_rates),
            'input_size': int(input_size_val),
            'num_classes': num_classes,
            'output_activation': output_activation,
            'weights_path': weights_path,
        }

        # normalize: только если был в meta
        if 'normalize' in meta:
            self.metadata['normalize'] = bool(meta['normalize'])

        # resize_mode: с backward-compat
        if 'resize_mode' in meta and meta['resize_mode'] in VALID_TRAIN_RESIZE_MODES:
            self.metadata['resize_mode'] = meta['resize_mode']
            self.metadata['keep_aspect_ratio'] = (meta['resize_mode'] == 'keep_aspect_pad')
        elif 'keep_aspect_ratio' in meta:
            self.metadata['resize_mode'] = ('keep_aspect_pad'
                                            if meta['keep_aspect_ratio'] else 'hard')
            self.metadata['keep_aspect_ratio'] = bool(meta['keep_aspect_ratio'])

        # inference_resize_mode
        if 'inference_resize_mode' in meta and \
                meta['inference_resize_mode'] in VALID_INFERENCE_RESIZE_MODES:
            self.metadata['inference_resize_mode'] = meta['inference_resize_mode']
        elif self.metadata.get('resize_mode') == 'random_crop':
            self.metadata['inference_resize_mode'] = 'keep_aspect_pad'

        _log(f"[DeepLab] Метаданные: arch={architecture}, backbone={backbone_name}, "
             f"OS={output_stride_val}, input_size={input_size_val}, "
             f"num_classes={num_classes}, activation={output_activation}")
        _log(f"[DeepLab]   в meta: "
             f"normalize={'yes' if 'normalize' in self.metadata else 'NO'}, "
             f"resize_mode={'yes' if 'resize_mode' in self.metadata else 'NO'}, "
             f"inference_resize_mode="
             f"{'yes' if 'inference_resize_mode' in self.metadata else 'NO'}")

    def predict(self, image, threshold=0.5):
        """
        Инференс DeepLabV3/V3+.

        Логика по num_classes:
            1 → sigmoid + threshold
            2 → softmax[:, 1] + threshold
                (или sigmoid[:, 1] + threshold при output_activation='sigmoid')
           >2 → softmax + argmax (класс масштабируется в 0..255 для визуализации)

        Параметр threshold игнорируется для num_classes > 2.
        """
        start_time = time.time()

        input_size = self.metadata.get('input_size', 513)
        num_classes = self.metadata.get('num_classes', 21)
        output_activation = self.metadata.get('output_activation', 'argmax')
        train_resize_mode = self.metadata.get('resize_mode', '?')
        resize_mode = self._resolve_resize_mode()

        _log(f"[DeepLab predict] Начало: shape={image.shape}, "
             f"num_classes={num_classes}, activation={output_activation}, "
             f"threshold={threshold}, train_rm={train_resize_mode}, "
             f"infer_rm={resize_mode}")

        tensor, orig_size, _, pad_info = self._preprocess(image, input_size)
        device = self._get_device()
        tensor = tensor.to(device)

        # smp.DeepLabV3/V3+ требуют кратности output_stride.
        # Паддим до ближайшего кратного 16 и обрезаем обратно.
        h_in, w_in = tensor.shape[-2], tensor.shape[-1]
        nh = ((h_in + 15) // 16) * 16
        nw = ((w_in + 15) // 16) * 16
        if nh != h_in or nw != w_in:
            tensor = torch.nn.functional.pad(
                tensor, (0, nw - w_in, 0, nh - h_in), mode='constant', value=0
            )


        with torch.no_grad():
            logits = self.model(tensor)  # [1, C, H, W]
            if nh != h_in or nw != w_in:
                logits = logits[..., :h_in, :w_in]

            if num_classes == 1:
                # Бинарная сегментация — всегда sigmoid
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                mask_np = (probs > threshold).astype(np.uint8) * 255

            elif num_classes == 2:
                # «Фон + один объект»
                if output_activation == 'sigmoid':
                    probs = torch.sigmoid(logits)
                else:
                    if output_activation == 'sigmoid':
                        _log("[DeepLab predict] ВНИМАНИЕ: output_activation='sigmoid' при "
                             f"num_classes={num_classes} > 2 — использую softmax + argmax.")
                    probs = torch.softmax(logits, dim=1)
                obj_prob = probs[0, 1].cpu().numpy()  # [H, W]
                mask_np = (obj_prob > threshold).astype(np.uint8) * 255

            else:
                # Истинный multiclass
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                mask_np = pred_class.squeeze().cpu().numpy().astype(np.uint8)
                # Масштабируем индексы классов в 0..255 для совместимости
                # с downstream-морфологией (бинарная обработка).
                mask_np = (mask_np * (255 // (num_classes - 1))).astype(np.uint8)

        _log(f"[DeepLab predict] Предсказание shape={mask_np.shape}, "
             f"min={mask_np.min()}, max={mask_np.max()}")

        # Обратный ресайз к исходному размеру
        mask_np = self._postprocess_mask(mask_np, orig_size, pad_info, resize_mode)

        elapsed = time.time() - start_time
        _log(f"[DeepLab predict] Готово за {elapsed:.3f} сек. "
             f"Выходная маска shape={mask_np.shape}")
        return mask_np






# =============================================================================
# YOLO
# =============================================================================
class YOLOSegmentor(BaseSegmentor):
    """
    YOLO сегментатор (любая версия: v8, v11, v26, ...).
    Модель загружается из файла .pt.

    Тип вывода определяется автоматически по результату инференса:
        • segmentation — если у результата есть .masks: используем их как есть;
        • OBB         — если у результата есть .obb: для каждой OBB-детекции
                        берём bbox, внутри ROI применяем Otsu (THRESH_BINARY_INV),
                        результат маскируем полигоном OBB;
        • detect      — если есть .boxes: для каждого бокса внутри ROI применяем Otsu.

    Параметры инференса: conf, iou, imgsz, save.
    """
    def load(self, model_path=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics is required for YOLO")
        _log(f"[YOLO] Загрузка модели из {model_path}")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(model_path)
        model_name = os.path.basename(model_path)
        self.metadata = {
            'model_type': 'YOLO-seg',
            'model_path': model_path,
            'model_name': model_name,
            'device': self.device,
            'task': getattr(self.model, 'task', 'unknown'),
        }
        _log(f"[YOLO] Модель загружена, task={self.metadata['task']}, "
             f"устройство по умолчанию: {self.device}")

    @staticmethod
    def _otsu_inside_roi(image, x1, y1, x2, y2):
        """
        Возвращает бинарную маску (uint8, 0/255) для ROI [x1:x2, y1:y2]
        методом Otsu (THRESH_BINARY_INV) — «тёмный объект на светлом фоне».
        """
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, roi_mask = cv2.threshold(
            roi_gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return roi_mask

    def predict(self, image, conf=0.25, iou=0.45, imgsz=640, save=False):
        start_time = time.time()
        original_h, original_w = image.shape[:2]
        _log(f"[YOLO predict] Начало, изображение ({original_h}, {original_w}), "
             f"conf={conf}, iou={iou}, imgsz={imgsz}, save={save}")

        device_arg = None if self.device == 'auto' else self.device
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            verbose=False,
            device=device_arg,
        )
        r = results[0]
        combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)

        # ============== 1. SEGMENTATION ==============
        masks = getattr(r, 'masks', None)
        if masks is not None and len(masks) > 0:
            try:
                masks_np = masks.data.cpu().numpy()          # (N, H', W')
            except AttributeError:
                masks_np = np.asarray(masks.data)
            _log(f"[YOLO predict] Тип вывода: SEGMENTATION, "
                 f"{masks_np.shape[0]} масок, размер внутри модели: "
                 f"{masks_np.shape[1]}x{masks_np.shape[2]}")

            combined = np.max(masks_np, axis=0)
            combined = cv2.resize(combined, (original_w, original_h),
                                  interpolation=cv2.INTER_LINEAR)
            combined_mask = (combined * 255).astype(np.uint8)
            object_pixels = int(np.sum(combined_mask > 127))
            _log(f"[YOLO predict] Маска (segmentation): "
                 f"объектных пикселей {object_pixels} "
                 f"({100 * object_pixels / combined_mask.size:.2f}%)")
            elapsed = time.time() - start_time
            _log(f"[YOLO predict] Готово за {elapsed:.3f} сек. "
                 f"Выходная маска shape={combined_mask.shape}")
            return combined_mask

        # ============== 2. OBB ==============
        obb = getattr(r, 'obb', None)
        if obb is not None and len(obb) > 0:
            try:
                corners_all = obb.xyxyxyxy.cpu().numpy()     # (N, 4, 2)
            except AttributeError:
                corners_all = np.asarray(obb.xyxyxyxy)

            n = len(corners_all)
            _log(f"[YOLO predict] Тип вывода: OBB, {n} детекций. "
                 f"Otsu внутри bbox, маскирование полигоном OBB.")

            for corners in corners_all:
                pts = corners.reshape(-1, 2).astype(np.int32)
                x1 = max(0, int(pts[:, 0].min()))
                y1 = max(0, int(pts[:, 1].min()))
                x2 = min(original_w, int(pts[:, 0].max()) + 1)
                y2 = min(original_h, int(pts[:, 1].max()) + 1)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                roi_mask = self._otsu_inside_roi(image, x1, y1, x2, y2)
                if roi_mask is None:
                    continue

                poly_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                shifted = pts - np.array([x1, y1])
                cv2.fillPoly(poly_mask, [shifted], 255)
                roi_mask = cv2.bitwise_and(roi_mask, poly_mask)

                combined_mask[y1:y2, x1:x2] = np.maximum(
                    combined_mask[y1:y2, x1:x2], roi_mask
                )

            object_pixels = int(np.sum(combined_mask > 127))
            _log(f"[YOLO predict] Маска (OBB): объектных пикселей "
                 f"{object_pixels} ({100 * object_pixels / combined_mask.size:.2f}%)")
            elapsed = time.time() - start_time
            _log(f"[YOLO predict] Готово за {elapsed:.3f} сек. "
                 f"Выходная маска shape={combined_mask.shape}")
            return combined_mask

        # ============== 3. DETECT (boxes) ==============
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and len(boxes) > 0:
            try:
                xyxy_all = boxes.xyxy.cpu().numpy()          # (N, 4)
            except AttributeError:
                xyxy_all = np.asarray(boxes.xyxy)

            n = len(xyxy_all)
            _log(f"[YOLO predict] Тип вывода: DETECT, {n} боксов. "
                 f"Otsu внутри bbox.")

            for box in xyxy_all:
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(original_w, int(box[2]))
                y2 = min(original_h, int(box[3]))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                roi_mask = self._otsu_inside_roi(image, x1, y1, x2, y2)
                if roi_mask is None:
                    continue

                combined_mask[y1:y2, x1:x2] = np.maximum(
                    combined_mask[y1:y2, x1:x2], roi_mask
                )

            object_pixels = int(np.sum(combined_mask > 127))
            _log(f"[YOLO predict] Маска (detect): объектных пикселей "
                 f"{object_pixels} ({100 * object_pixels / combined_mask.size:.2f}%)")
            elapsed = time.time() - start_time
            _log(f"[YOLO predict] Готово за {elapsed:.3f} сек. "
                 f"Выходная маска shape={combined_mask.shape}")
            return combined_mask

        # ============== 4. Пусто ==============
        task = getattr(self.model, 'task', 'unknown')
        _log(f"[YOLO predict] Модель не вернула ни масок, ни боксов "
             f"(task={task}). Возвращаем нулевую маску.")
        return combined_mask


# =============================================================================
# SAM (заглушка)
# =============================================================================
class SAMSegmentor(BaseSegmentor):
    """Segment Anything Model (заглушка)."""
    def load(self, model_type='vit_h', model_path=None):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError("segment-anything is required for SAM")
        self.metadata = {
            'model_type': 'SAM',
            'model_type_detail': model_type,
            'model_path': model_path,
        }
        _log("[SAM] Загрузка не реализована, выбрасываем исключение")
        raise NotImplementedError("SAM loading not implemented in this stub")

    def predict(self, image, prompt_type='everything'):
        raise NotImplementedError("SAM prediction not implemented")


# =============================================================================
# ONNX
# =============================================================================
class ONNXSegmentor(BaseSegmentor):
    """ONNX Runtime инференс."""
    def load(self, model_path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed")
        self.metadata = {'model_type': 'Custom ONNX', 'path': model_path}
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        _log(f"[ONNX] Сессия создана для {model_path}")

    def predict(self, image):
        start_time = time.time()
        original_h, original_w = image.shape[:2]
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        h_in, w_in = input_shape[2], input_shape[3]
        _log(f"[ONNX predict] Ожидаемый вход: {input_shape}, "
             f"текущее изображение: ({original_h}, {original_w})")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) \
            if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (w_in, h_in))
        _log(f"[ONNX predict] После ресайза: {img.shape}")
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {input_name: img})
        mask = outputs[0].squeeze()
        _log(f"[ONNX predict] Выход модели (до ресайза): {mask.shape}, "
             f"min={mask.min()}, max={mask.max()}")
        mask = cv2.resize(mask, (original_w, original_h))
        result = (mask * 255).astype(np.uint8)
        elapsed = time.time() - start_time
        _log(f"[ONNX predict] Готово за {elapsed:.3f} сек. "
             f"Итоговая маска shape={result.shape}")
        return result


# =============================================================================
# Фабрика
# =============================================================================
def create_segmentor(model_type, **kwargs):
    segmentors = {
        'U-Net': UNetSegmentor,
        'DeepLabV3+': DeepLabV3Segmentor,
        'DeepLabV3': DeepLabV3Segmentor,      # <--- добавить
        'SegFormer': None,
        'SAM': SAMSegmentor,
        'YOLOv8-seg': YOLOSegmentor,
        'YOLO-seg': YOLOSegmentor,
        'Custom ONNX': ONNXSegmentor,
    }
    if model_type not in segmentors:
        raise ValueError(f"Unknown model type: {model_type}")
    segmentor_class = segmentors[model_type]
    if segmentor_class is None:
        raise NotImplementedError(f"Model {model_type} not yet implemented")
    return segmentor_class(**kwargs)