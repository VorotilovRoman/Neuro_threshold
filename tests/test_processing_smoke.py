"""Small data-path checks after changing dependency boundaries (no model weights)."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
RUNTIME = all(importlib.util.find_spec(name) for name in ('cv2', 'numpy', 'PyQt5', 'skimage'))


@unittest.skipUnless(RUNTIME, 'image runtime dependencies are not installed')
class ProcessingSmokeTests(unittest.TestCase):
    def test_image_threshold_and_gradient(self):
        import cv2
        import numpy as np

        from utils_ops.gradient_ops import apply_gradient_method
        from utils_ops.segmentation_ops import apply_threshold_method
        image = np.zeros((64, 64), dtype=np.uint8)
        image[16:48, 16:48] = 220
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'изображение.png'
            self.assertTrue(cv2.imwrite(str(path), image))
            np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), image)
        for name in ('Simple Threshold', 'Otsu', 'Adaptive Mean', 'Niblack', 'Sauvola'):
            mask, _ = apply_threshold_method(image, name, {})
            self.assertEqual(mask.shape, image.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 255})
            self.assertGreater(mask.max(), 0)
        for name in ('Sobel', 'Canny'):
            result = apply_gradient_method(image, name, {})
            self.assertEqual(result.shape, image.shape)
            self.assertGreater(result.max(), 0)

    def test_geometry_and_generated_formats(self):
        import json

        import cv2
        import numpy as np
        import yaml

        from utils_new.annotations import (
            AnnotationSample,
            ImageSize,
            Resize,
            from_legacy,
            rasterize_annotations,
            to_legacy,
            transform_sample,
        )
        from utils_new.DatasetGenerator_new import INPUT_LABELS, DatasetGeneratorCoreNew
        size = ImageSize(32, 24)
        legacy = ('detect', 0, .5, .5, .5, .5)
        annotation = from_legacy(legacy, size)
        np.testing.assert_allclose(to_legacy(annotation, size)[2:], legacy[2:])
        mask = rasterize_annotations([annotation], size)
        self.assertEqual(set(np.unique(mask)), {0, 1})
        self.assertEqual(int((mask == 1).sum()), 192)
        sample = AnnotationSample(np.zeros((24, 32, 3), dtype=np.uint8), [annotation])
        changed = transform_sample(sample, [Resize((64, 48), mode='stretch')])
        self.assertEqual(changed.image.shape[:2], (48, 64))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            image, label, output = source / 'source.png', source / 'source.txt', source / 'out'
            cv2.imwrite(str(image), np.full((24, 32, 3), 127, dtype=np.uint8))
            label.write_text('0 0.5 0.5 0.5 0.5\n')
            generator = DatasetGeneratorCoreNew(
                images_folder=directory, labels_folder=directory, output_folder=str(output),
                pairs=[(str(image), str(label), None)], train_pct=1, val_pct=0, test_pct=0,
                target_size=None, augmentations={}, has_augmentations=False, aug_multiplier=1,
                class_names={0: 'object'}, class_mapping={0: 0}, bg_color=0, input_type=INPUT_LABELS,
                masks_folder=None, num_classes=1, mask_class_remap={}, resize_mode='keep',
                out_masks=True, out_unet_yaml=True, out_coco=True, random_seed=42,
                split_by_source=True, strict_complete=True, log_cb=lambda text: None)
            ok, message = generator.execute()
            self.assertTrue(ok, message)
            self.assertEqual(len(list((output / 'images/train').glob('*'))), 1)
            labels = list((output / 'labels/train').glob('*.txt'))
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0].read_text().startswith('0 '))
            masks = list((output / 'masks/train').glob('*.png'))
            self.assertEqual(len(masks), 1)
            stored = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(set(np.unique(stored)), {0, 1})
            self.assertEqual(yaml.safe_load((output / 'data.yaml').read_text())['nc'], 1)
            self.assertTrue((output / 'masks_data.yaml').exists())
            self.assertTrue(any(isinstance(json.loads(p.read_text()).get('annotations'), list)
                                for p in output.rglob('*.json')))
            self.assertTrue(image.exists() and label.exists())

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('albumentations'),
                         'Torch/Albumentations are not installed')
    def test_torch_and_mask_augmentation(self):
        import numpy as np
        import torch

        from utils.augmentation_random import apply_complex_augmentations_mask
        self.assertEqual(torch.mm(torch.ones(2, 2), torch.ones(2, 2)).tolist(), [[2, 2], [2, 2]])
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[6:18, 8:24] = 1
        image = np.repeat((mask * 255)[:, :, None], 3, axis=2)
        changed, changed_mask = apply_complex_augmentations_mask(
            image, mask, {'random_rotate': True, 'random_rotate_angle': 30})
        self.assertEqual(changed.shape, image.shape)
        self.assertEqual(changed_mask.shape, mask.shape)
        self.assertEqual(set(np.unique(changed_mask)), {0, 1})


if __name__ == '__main__':
    unittest.main()
