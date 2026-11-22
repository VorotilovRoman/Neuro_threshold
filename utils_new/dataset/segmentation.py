"""One paired image/index-mask dataset for U-Net and DeepLab training."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from utils_new.import_export.images import (
    collect_image_paths,
    image_size_from_file,
    read_index_mask,
    index_mask_files,
)
from utils_new.images.processing import resize_segmentation_pair
from utils_new.segmentation.training_contracts import validate_mask_files


class ImageMaskDataset(Dataset):
    def __init__(
        self,
        pairs,
        input_size=512,
        num_classes=1,
        ignore_index=None,
        augmentations=None,
        *,
        normalize=False,
        resize_mode="hard",
        is_train=True,
    ):
        self.pairs = list(pairs)
        self.input_size = input_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.augmentations = augmentations or {}
        self.normalize = normalize
        self.resize_mode = resize_mode
        self.is_train = is_train
        if resize_mode not in ("hard", "keep_aspect_pad", "random_crop"):
            raise ValueError("Неизвестный resize_mode.")
        if not self.pairs:
            raise ValueError("Нет пар изображение-маска.")
        self.image_paths = [str(p[0]) for p in self.pairs]
        self.mask_paths = [str(p[1]) for p in self.pairs]
        validate_mask_files(self.mask_paths, num_classes, ignore_index)
        for image, mask in self.pairs:
            if image_size_from_file(image) != image_size_from_file(mask):
                raise ValueError(f"Размер снимка и маски различается: {image}, {mask}")

    @classmethod
    def from_folders(
        cls,
        images_dir,
        masks_dir,
        input_size=512,
        num_classes=1,
        augmentations=None,
        **options,
    ):
        images = collect_image_paths(images_dir, recursive=True)
        root = Path(images_dir)
        masks = index_mask_files(masks_dir)
        pairs = []
        for image in images:
            key = Path(image).relative_to(root).with_suffix("").as_posix().casefold()
            if key not in masks:
                raise FileNotFoundError(f"Нет маски: {image}")
            pairs.append((image, str(masks[key])))
        return cls(
            pairs, input_size, num_classes, augmentations=augmentations, **options
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = np.array(source.convert("RGB"))
        mask = read_index_mask(mask_path, ignore_index=self.ignore_index)
        if self.num_classes == 1:
            void = (
                mask == self.ignore_index
                if self.ignore_index is not None
                else np.zeros(mask.shape, bool)
            )
            mask = (mask > 0).astype(np.int32)
            if self.ignore_index is not None:
                mask[void] = self.ignore_index
        else:
            mask = mask.astype(np.int32)
        fill = self.ignore_index if self.ignore_index is not None else 0
        image, mask, _ = resize_segmentation_pair(
            image,
            mask,
            self.input_size,
            self.resize_mode,
            is_train=self.is_train,
            mask_fill=fill,
        )
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        aug = self.augmentations if self.is_train else {}
        for flag, transform in (("hflip", TF.hflip), ("vflip", TF.vflip)):
            if aug.get(flag) and torch.rand(()) < 0.5:
                image, mask = transform(image), transform(mask)
        if aug.get("rotate"):
            angle = float(torch.empty(()).uniform_(-10, 10))
            image = TF.rotate(
                image, angle, interpolation=InterpolationMode.BILINEAR, fill=0
            )
            mask = TF.rotate(
                mask, angle, interpolation=InterpolationMode.NEAREST, fill=fill
            )
        if aug.get("scale"):
            scale = float(torch.empty(()).uniform_(0.9, 1.1))
            image = TF.affine(
                image,
                0,
                (0, 0),
                scale,
                0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                0,
                (0, 0),
                scale,
                0,
                interpolation=InterpolationMode.NEAREST,
                fill=fill,
            )
        image = TF.to_tensor(image)
        if aug.get("noise"):
            image = (image + torch.randn_like(image) * 0.05).clamp(0, 1)
        if self.normalize:
            image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.from_numpy(np.array(mask, dtype=np.int64))
