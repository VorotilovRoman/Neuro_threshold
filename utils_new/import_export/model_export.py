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
