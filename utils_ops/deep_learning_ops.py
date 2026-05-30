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

    def _preprocess(self, image, input_size):
        # image: numpy array (H,W,3) or (H,W)
        _log(f"[Preprocess] Входное изображение: shape={image.shape}, dtype={image.dtype}")
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            _log("[Preprocess] Преобразовано из серого в RGB")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            _log("[Preprocess] Преобразовано из BGRA в RGB")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _log("[Preprocess] Преобразовано из BGR в RGB")

        h, w = image.shape[:2]
        if isinstance(input_size, int):
            scale = input_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = input_size
        _log(f"[Preprocess] Исходный размер: ({h}, {w}) -> новый размер: ({new_h}, {new_w})")
        image = cv2.resize(image, (new_w, new_h))

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(image).unsqueeze(0)
        _log(f"[Preprocess] Тензор shape={tensor.shape}, device={self._get_device()}")
        return tensor, (h, w), (new_h, new_w)


class UNetSegmentor(TorchSegmentor):
    """
    Сегментатор U-Net с использованием библиотеки segmentation_models_pytorch.
    При загрузке обязательно требует наличия файла *_meta.json рядом с весами.
    Поддерживает как бинарную, так и многоклассовую сегментацию.
    """
    def load(self, weights_path=None, encoder=None, input_size=None):
        """
        Загружает модель U-Net.

        Параметры:
            weights_path (str): путь к файлу весов (.pth)
            encoder (str): имя энкодера (игнорируется, если есть метафайл)
            input_size (int): размер входа (игнорируется, если есть метафайл)

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
                '  "encoder": "resnet50",           # имя энкодера (resnet18, resnet34, resnet50, efficientnet-b0, vgg16)\n'
                '  "input_size": 512,               # размер входа (256, 512, 1024)\n'
                '  "num_classes": 1,                # количество классов (1 для бинарной сегментации)\n'
                '  "decoder_channels": [256,128,64,32,16]  # опционально, каналы декодера\n'
                "}\n"
                "Убедитесь, что метафайл создан при обучении модели."
            )

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        # Извлекаем параметры архитектуры
        encoder_name = meta.get('encoder')
        input_size_val = meta.get('input_size')
        num_classes = meta.get('num_classes', 1)
        decoder_channels = meta.get('decoder_channels', (256, 128, 64, 32, 16))

        if not encoder_name or not input_size_val:
            raise ValueError(
                "Метафайл должен содержать поля 'encoder' и 'input_size'.\n"
                f"Текущее содержимое: {meta}"
            )

        # Сохраняем метаданные для использования в predict
        self.metadata = {
            'model_type': 'U-Net',
            'encoder': encoder_name,
            'input_size': input_size_val,
            'num_classes': num_classes,
            'decoder_channels': decoder_channels,
            'weights_path': weights_path
        }

        _log(f"[UNet] Создание модели: encoder={encoder_name}, input_size={input_size_val}, num_classes={num_classes}")

        # Создаём модель с параметрами из метафайла
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,      # веса загружаем из файла, imagenet не используем
            in_channels=3,
            classes=num_classes,
            decoder_channels=decoder_channels
        )

        # Загружаем веса
        state_dict = torch.load(weights_path, map_location='cpu')
        # Удаляем служебные ключи, которые могут вызвать предупреждения
        cleaned = {k: v for k, v in state_dict.items() if not k.endswith('num_batches_tracked')}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            _log(f"[UNet] Отсутствуют ключи: {missing[:3]}...")
        if unexpected:
            _log(f"[UNet] Лишние ключи (игнорируются): {unexpected[:3]}...")

        self.model.to(self._get_device())
        self.model.eval()
        _log(f"[UNet] Модель загружена, устройство: {self._get_device()}")

    def predict(self, image, threshold=0.5):
        """
        Выполняет инференс и возвращает бинарную маску (0 и 255).

        Параметры:
            image (np.ndarray): входное изображение (BGR или серое)
            threshold (float): порог для бинаризации (0..1)

        Возвращает:
            np.ndarray: uint8 маска, где 255 — объект, 0 — фон
        """
        start_time = time.time()
        _log(f"[UNet predict] Начало, входное изображение shape={image.shape}")

        input_size = self.metadata.get('input_size', 512)
        num_classes = self.metadata.get('num_classes', 1)
        print("threshold", threshold)
        tensor, orig_size, _ = self._preprocess(image, input_size)
        device = self._get_device()
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = self.model(tensor)  # [1, C, H, W]
            if num_classes == 1:
                probs = torch.sigmoid(logits)  # [1, 1, H, W] значения 0..1
                mask_np = probs.squeeze().cpu().numpy()  # [H, W]
                # Бинаризация по порогу
                mask_np = (mask_np > threshold).astype(np.uint8) * 255
            else:
                probs = torch.softmax(logits, dim=1)  # [1, C, H, W]
                pred_class = torch.argmax(probs, dim=1)  # [1, H, W]
                mask_np = pred_class.squeeze().cpu().numpy().astype(np.uint8)
                # Масштабируем индексы классов для визуализации
                if num_classes > 1:
                    mask_np = (mask_np * (255 // (num_classes - 1))).astype(np.uint8)

        _log(f"[UNet predict] Предсказание shape={mask_np.shape}, min={mask_np.min()}, max={mask_np.max()}")

        # Ресайз к исходному размеру изображения
        mask_np = cv2.resize(mask_np, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)

        elapsed = time.time() - start_time
        _log(f"[UNet predict] Готово за {elapsed:.3f} сек. Выходная маска shape={mask_np.shape}")
        return mask_np

class DeepLabV3Segmentor(TorchSegmentor):
    """DeepLabV3+ из torchvision."""
    def load(self, backbone='resnet50', output_stride=16, input_size=520):
        from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
        _log(f"[DeepLabV3] Загрузка модели: backbone={backbone}, output_stride={output_stride}, input_size={input_size}")
        self.metadata = {
            'model_type': 'DeepLabV3+',
            'backbone': backbone,
            'output_stride': output_stride,
            'input_size': input_size
        }
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=True, progress=False)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=True, progress=False)
        elif backbone == 'mobilenet_v3':
            self.model = deeplabv3_mobilenet_v3_large(pretrained=True, progress=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.model.to(self._get_device())
        self.model.eval()
        _log(f"[DeepLabV3] Модель загружена, устройство: {self._get_device()}")

    def predict(self, image):
        start_time = time.time()
        _log(f"[DeepLabV3 predict] Начало, shape={image.shape}")
        input_size = self.metadata.get('input_size', 520)
        tensor, orig_size, _ = self._preprocess(image, input_size)
        tensor = tensor.to(self._get_device())
        with torch.no_grad():
            output = self.model(tensor)['out']
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        _log(f"[DeepLabV3 predict] Предсказание shape={pred.shape}, min={pred.min()}, max={pred.max()}")
        pred = cv2.resize(pred, (orig_size[1], orig_size[0]))
        mask = (pred * 255).astype(np.uint8)
        elapsed = time.time() - start_time
        _log(f"[DeepLabV3 predict] Готово за {elapsed:.3f} сек. Маска shape={mask.shape}")
        return mask



# deep_learning_ops.py - класс YOLOSegmentor

class YOLOSegmentor(BaseSegmentor):
    """
    YOLO сегментатор (любая версия: v8, v11, v26, ...).
    Модель загружается из файла .pt (поддерживаются как официальные, так и пользовательские).
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
            'device': self.device
        }
        _log(f"[YOLO] Модель загружена, устройство по умолчанию: {self.device}")

    def predict(self, image, conf=0.25, iou=0.45, imgsz=640, save=False):
        start_time = time.time()
        original_h, original_w = image.shape[:2]
        _log(f"[YOLO predict] Начало, исходное изображение размером ({original_h}, {original_w}), "
             f"conf={conf}, iou={iou}, imgsz={imgsz}, save={save}")

        device_arg = None if self.device == 'auto' else self.device
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=save,
            verbose=False,
            device=device_arg
        )
        if results[0].masks is None:
            _log("[YOLO predict] Маски не обнаружены, возвращаем нулевую маску")
            return np.zeros((original_h, original_w), dtype=np.uint8)

        masks = results[0].masks.data.cpu().numpy()  # (N, H', W')
        _log(f"[YOLO predict] Получено {len(masks)} масок, размер маски внутри модели: {masks.shape[1]}x{masks.shape[2]}")
        combined = np.max(masks, axis=0)
        combined = cv2.resize(combined, (original_w, original_h))
        _log(f"[YOLO predict] После ресайза к исходному размеру: {combined.shape}")
        mask = (combined * 255).astype(np.uint8)
        elapsed = time.time() - start_time
        _log(f"[YOLO predict] Готово за {elapsed:.3f} сек. Выходная маска shape={mask.shape}, min={mask.min()}, max={mask.max()}")
        return mask



class SAMSegmentor(BaseSegmentor):
    """Segment Anything Model (заглушка)."""
    def load(self, model_type='vit_h', model_path=None):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError("segment-anything is required for SAM")
        self.metadata = {'model_type': 'SAM', 'model_type_detail': model_type, 'model_path': model_path}
        _log("[SAM] Загрузка не реализована, выбрасываем исключение")
        raise NotImplementedError("SAM loading not implemented in this stub")

    def predict(self, image, prompt_type='everything'):
        raise NotImplementedError("SAM prediction not implemented")


class ONNXSegmentor(BaseSegmentor):
    """ONNX Runtime инференс."""
    def load(self, model_path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed")
        self.metadata = {'model_type': 'Custom ONNX', 'path': model_path}
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        _log(f"[ONNX] Сессия создана для {model_path}")

    def predict(self, image):
        start_time = time.time()
        original_h, original_w = image.shape[:2]
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        h_in, w_in = input_shape[2], input_shape[3]
        _log(f"[ONNX predict] Ожидаемый вход модели: {input_shape}, текущее изображение: ({original_h}, {original_w})")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape)==3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (w_in, h_in))
        _log(f"[ONNX predict] После ресайза: {img.shape}")
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        outputs = self.session.run(None, {input_name: img})
        mask = outputs[0].squeeze()
        _log(f"[ONNX predict] Выход модели (до ресайза): {mask.shape}, min={mask.min()}, max={mask.max()}")
        mask = cv2.resize(mask, (original_w, original_h))
        result = (mask * 255).astype(np.uint8)
        elapsed = time.time() - start_time
        _log(f"[ONNX predict] Готово за {elapsed:.3f} сек. Итоговая маска shape={result.shape}")
        return result


def create_segmentor(model_type, **kwargs):
    """Фабрика для создания сегментатора по имени."""
    segmentors = {
        'U-Net': UNetSegmentor,
        'DeepLabV3+': DeepLabV3Segmentor,
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