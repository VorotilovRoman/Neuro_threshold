"""Model mechanics with tiny CPU inputs; no pretrained weight downloads."""

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from utils_new.segmentation.model_output import decode_output
from utils_new.segmentation.training_contracts import validate_mask_files


class OutputContracts(unittest.TestCase):
    def test_logit_probability_and_multiclass_decoding(self):
        logits = np.array([[[[-8, 8]]]], np.float32)
        np.testing.assert_array_equal(decode_output(logits), [[0, 255]])
        np.testing.assert_array_equal(
            decode_output(np.array([[[[0.2, 0.8]]]]), output_kind="probabilities"),
            [[0, 255]],
        )
        two = np.array([[[[5, -5]], [[-5, 5]]]], np.float32)
        np.testing.assert_array_equal(decode_output(two), [[0, 255]])
        multiple = np.zeros((1, 300, 1, 2), np.float32)
        multiple[0, 299, 0, 1] = 5
        np.testing.assert_array_equal(decode_output(multiple), [[0, 299]])
        for values in (np.array([[[[np.nan]]]]), np.zeros((2, 1, 3, 4))):
            with self.assertRaises(ValueError):
                decode_output(values)
        with self.assertRaises(ValueError):
            decode_output(logits, output_kind="probabilities")

    def test_all_masks_are_checked_including_after_twentieth(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(22):
                path = Path(directory) / f"{i}.png"
                Image.fromarray(np.full((3, 5), 8 if i == 21 else 1, np.uint8)).save(
                    path
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "21.png"):
                validate_mask_files(paths, 2)
            validate_mask_files(paths, 1)  # binary union accepts all positive IDs

    def test_onnx_dynamic_input_and_centered_padding_roundtrip(self):
        from utils_new.segmentation.backends.deep_learning import ONNXSegmentor

        model = ONNXSegmentor()

        def run(_, inputs):
            tensor = inputs["image"]
            self.assertEqual(tensor.shape, (1, 3, 32, 32))
            # Object indicated by red RGB channel; zero-padding is background.
            return [tensor[:, :1] * 20 - 10]

        model.session = SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="image", shape=[None, 3, "h", "w"])
            ],
            run=run,
        )
        model.metadata = {
            "input_size": 32,
            "resize_mode": "keep_aspect_pad",
            "normalize": False,
            "output_kind": "logits",
        }
        image = np.zeros((12, 36, 3), np.uint8)
        image[:, :, 2] = 255
        np.testing.assert_array_equal(
            model.predict(image), np.full((12, 36), 255, np.uint8)
        )


@unittest.skipUnless(
    find_spec("torch") and find_spec("segmentation_models_pytorch"),
    "optional training backend",
)
class RealTorchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def test_actual_smp_losses_backward_optimizer_and_dropout(self):
        import segmentation_models_pytorch as smp
        import torch

        from utils_new.segmentation.training_contracts import apply_decoder_dropout
        from utils_training.train_DeepLabV3_worker import (
            get_loss_function as deeplab_loss,
        )
        from utils_training.train_unet_worker import get_loss_function

        for classes in (1, 3):
            logits = torch.randn(2, classes, 16, 16, requires_grad=True)
            target = torch.randint(0, 2 if classes == 1 else classes, (2, 16, 16))
            for name in ("BCE+Dice", "Dice", "Focal", "Tversky"):
                loss = get_loss_function(name, classes)(
                    logits, target[:, None].float() if classes == 1 else target
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward(retain_graph=True)
            for name in ("CrossEntropy", "Dice", "CrossEntropy+Dice", "Focal"):
                loss = deeplab_loss(name, classes)(
                    logits, target[:, None].float() if classes == 1 else target
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward(retain_graph=True)
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            decoder_channels=(16, 8, 4, 2, 1),
            classes=1,
        )
        keys = tuple(model.state_dict())
        apply_decoder_dropout(model, 0.5)
        self.assertEqual(tuple(model.state_dict()), keys)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        data = torch.rand(2, 3, 32, 32)
        model.train()
        prediction = model(data)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction, torch.zeros_like(prediction)
        )
        loss.backward()
        previous = model.segmentation_head[0].bias.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(previous, model.segmentation_head[0].bias))
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(data), model(data)))

    def test_real_weights_load_and_predict_unet_and_deeplab(self):
        import segmentation_models_pytorch as smp
        import torch

        from utils_new.segmentation.backends.deep_learning import DeepLabV3Segmentor, UNetSegmentor

        with tempfile.TemporaryDirectory() as directory:
            for name, network, adapter, metadata in (
                (
                    "unet",
                    smp.Unet(
                        encoder_name="resnet18",
                        encoder_weights=None,
                        decoder_channels=(16, 8, 4, 2, 1),
                        classes=3,
                    ),
                    UNetSegmentor,
                    {
                        "architecture": "unet",
                        "encoder": "resnet18",
                        "decoder_channels": [16, 8, 4, 2, 1],
                    },
                ),
                (
                    "deeplab",
                    smp.DeepLabV3Plus(
                        encoder_name="resnet18",
                        encoder_weights=None,
                        classes=3,
                        encoder_output_stride=16,
                    ),
                    DeepLabV3Segmentor,
                    {
                        "architecture": "deeplabv3plus",
                        "backbone": "resnet18",
                        "output_stride": 16,
                    },
                ),
            ):
                path = Path(directory) / (name + ".pth")
                torch.save(network.state_dict(), path)
                metadata.update(
                    input_size=33,
                    num_classes=3,
                    normalize=True,
                    resize_mode="keep_aspect_pad",
                )
                path.with_name(name + "_meta.json").write_text(json.dumps(metadata))
                model = adapter(device="cpu")
                model.load(weights_path=str(path))
                result = model.predict(np.zeros((21, 57, 3), np.uint8))
                self.assertEqual(result.shape, (21, 57))
                self.assertTrue(np.isin(result, [0, 1, 2]).all())

    def test_binary_dataset_preserves_index_one(self):
        from utils_training.train_DeepLabV3_worker import VOCSegmentationPairs
        from utils_training.train_unet_worker import SegmentationDataset

        with tempfile.TemporaryDirectory() as directory:
            image_dir, mask_dir = Path(directory) / "images", Path(directory) / "masks"
            image_dir.mkdir()
            mask_dir.mkdir()
            path, mask = image_dir / "a.png", mask_dir / "a.png"
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
            pixels = np.zeros((8, 8), np.uint8)
            pixels[2:6, 2:6] = 1
            Image.fromarray(pixels).save(mask)
            unet = SegmentationDataset(
                str(image_dir), str(mask_dir), input_size=8, num_classes=1
            )[0][1]
            deeplab = VOCSegmentationPairs([(str(path), str(mask))], 8, 1, None, {})[0][
                1
            ]
            self.assertEqual(int(unet.sum()), 16)
            self.assertEqual(int(deeplab.sum()), 16)


@unittest.skipUnless(find_spec("torch"), "optional torch backend")
class SemanticMetrics(unittest.TestCase):
    def test_confusion_metrics_exclude_absent_classes_and_do_not_claim_ap(self):
        import torch

        from utils_new.segmentation.model_metrics import segmentation_metrics

        target = torch.tensor([[[0, 0], [1, 1]]])
        predicted = torch.tensor([[[0, 1], [1, 1]]])
        logits = (
            torch.nn.functional.one_hot(predicted, num_classes=3)
            .permute(0, 3, 1, 2)
            .float()
            * 20
        )
        result = segmentation_metrics(logits, target, 3)
        self.assertAlmostEqual(result["IoU"], (1 / 2 + 2 / 3) / 2)
        self.assertAlmostEqual(result["Dice"], (2 / 3 + 4 / 5) / 2)
        self.assertAlmostEqual(result["F1"], result["Dice"])
        self.assertEqual(result["Accuracy"], 0.75)
        self.assertFalse(any("AP" in key for key in result))
        # Void pixels do not inflate accuracy; entirely void data has no score.
        ignored = target.clone()
        ignored[0, 0, 1] = 255
        self.assertEqual(
            segmentation_metrics(logits, ignored, 3, ignore_index=255)["Accuracy"], 1
        )
        self.assertTrue(
            np.isnan(
                segmentation_metrics(
                    logits, torch.full_like(target, 255), 3, ignore_index=255
                )["IoU"]
            )
        )
        # Confidence threshold is meaningful for a two-channel binary head too.
        two_logits = torch.zeros(1, 2, 2, 2)
        two_logits[:, 1] = 1
        self.assertLess(
            segmentation_metrics(two_logits, torch.ones_like(target), 2, threshold=0.9)[
                "IoU"
            ],
            0.01,
        )


@unittest.skipUnless(
    find_spec("onnx") and find_spec("onnxruntime"), "optional ONNX backend"
)
class RealONNXContracts(unittest.TestCase):
    def test_exported_graph_session_and_sidecar_load(self):
        import onnx
        from onnx import TensorProto, helper

        from utils_new.segmentation.backends.deep_learning import ONNXSegmentor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.onnx"
            inp = helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [1, 3, "height", "width"]
            )
            out = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, [1, 1, "height", "width"]
            )
            node = helper.make_node(
                "ReduceMean", ["image"], ["output"], axes=[1], keepdims=1
            )
            graph = helper.make_graph([node], "semantic-test", [inp], [out])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 13)]
            )
            model.ir_version = 10
            onnx.save(model, str(path))
            metadata = {
                "input_size": 16,
                "normalize": False,
                "resize_mode": "hard",
                "output_kind": "logits",
            }
            path.with_name("semantic_meta.json").write_text(json.dumps(metadata))
            adapter = ONNXSegmentor()
            adapter.load(str(path))
            result = adapter.predict(np.full((7, 23, 3), 255, np.uint8), threshold=0.6)
            np.testing.assert_array_equal(result, np.full((7, 23), 255, np.uint8))
            result = adapter.predict(np.zeros((7, 23, 3), np.uint8), threshold=0.6)
            self.assertFalse(result.any())


class YOLOContracts(unittest.TestCase):
    def test_mask_geometry_is_original_size_and_detection_fallback_is_explicit(self):
        from utils_new.segmentation.backends.deep_learning import YOLOSegmentor

        model = YOLOSegmentor()
        image = np.zeros((10, 30, 3), np.uint8)
        mask = np.zeros((1, 10, 30), np.float32)
        mask[:, 2:7, 15:25] = 1
        args = {}

        def predict(image, **kwargs):
            args.update(kwargs)
            return [
                SimpleNamespace(masks=SimpleNamespace(data=mask, __len__=lambda: 1))
            ]

        class Masks:
            data = mask

            def __len__(self):
                return 1

        model.model = SimpleNamespace(
            predict=lambda image, **kwargs: (
                args.update(kwargs) or [SimpleNamespace(masks=Masks())]
            )
        )
        result = model.predict(image)
        self.assertTrue(args["retina_masks"])
        np.testing.assert_array_equal(result, mask[0].astype(np.uint8) * 255)

        class Boxes:
            xyxy = np.array([[5, 2, 20, 8]])

            def __len__(self):
                return 1

        model.model = SimpleNamespace(
            predict=lambda image, **kwargs: [SimpleNamespace(masks=None, boxes=Boxes())]
        )
        with self.assertRaisesRegex(ValueError, "Detection-модель"):
            model.predict(image)
        result = model.predict(image, allow_box_otsu=True)
        self.assertEqual(result.shape, image.shape[:2])

    def test_validation_uses_mask_aggregate_ap_not_per_class_box_array(self):
        from utils_new.segmentation.yolo_metrics import validation_metrics

        box = SimpleNamespace(
            mp=0.3, mr=0.4, map50=0.5, map=0.2, ap50=np.array([0.1, 0.9])
        )
        mask = SimpleNamespace(mp=0.6, mr=0.7, map50=0.8, map=0.55)
        result = validation_metrics(SimpleNamespace(box=box, seg=mask, results_dict={}))
        self.assertEqual(result["mAP50"], 0.8)
        self.assertEqual(result["mask/mAP50"], 0.8)
        self.assertEqual(result["box/mAP50"], 0.5)


@unittest.skipUnless(
    find_spec("torch") and find_spec("onnx") and find_spec("onnxruntime"),
    "optional export backend",
)
class ExportGeometryContracts(unittest.TestCase):
    def test_stride_padding_is_dynamic_and_output_is_cropped_before_decoding(self):
        import torch

        from utils_new.segmentation.model_export import stride_padded_model
        from utils_new.segmentation.backends.deep_learning import ONNXSegmentor

        class Network(torch.nn.Module):
            encoder = SimpleNamespace(output_stride=32)

            def forward(self, image):
                return torch.nn.functional.interpolate(
                    torch.nn.functional.avg_pool2d(image[:, :1], 32),
                    scale_factor=32,
                    mode="nearest",
                )

        wrapped = stride_padded_model(Network())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "padded.onnx"
            torch.onnx.export(
                wrapped,
                torch.zeros(1, 3, 33, 47),
                str(path),
                opset_version=18,
                input_names=["image"],
                output_names=["output"],
                dynamic_axes={
                    "image": {2: "height", 3: "width"},
                    "output": {2: "height", 3: "width"},
                },
                dynamo=False,
            )
            path.with_name("padded_meta.json").write_text(
                json.dumps(
                    {
                        "input_size": [33, 47],
                        "normalize": False,
                        "resize_mode": "hard",
                        "output_kind": "logits",
                    }
                )
            )
            adapter = ONNXSegmentor()
            adapter.load(str(path))
            for h, w in ((33, 47), (17, 61)):
                adapter.metadata["input_size"] = [h, w]
                image = np.full((h, w, 3), 255, np.uint8)
                with torch.no_grad():
                    expected = (
                        torch.sigmoid(wrapped(torch.ones(1, 3, h, w)))[0, 0].numpy()
                        > 0.6
                    ).astype(np.uint8) * 255
                self.assertEqual(expected.shape, (h, w))
                np.testing.assert_array_equal(
                    adapter.predict(image, threshold=0.6), expected
                )


@unittest.skipUnless(
    find_spec("torch") and find_spec("segmentation_models_pytorch"),
    "optional training backend",
)
class BinaryVoidContracts(unittest.TestCase):
    def test_binary_ignored_pixels_have_zero_loss_gradient(self):
        import torch

        from utils_training.train_DeepLabV3_worker import get_loss_function

        logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
        labels = torch.tensor([[[[0.0, 1.0, 255.0]]]])
        loss = get_loss_function("CrossEntropy", 1, ignore_index=255)(logits, labels)
        loss.backward()
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        self.assertEqual(logits.grad[0, 0, 0, 2].item(), 0.0)
