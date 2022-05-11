from tlxzoo.module import *
import tensorlayerx as tlx


class OpticalCharacterRecognition(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(OpticalCharacterRecognition, self).__init__()
        if backbone == "trocr":
            self.backbone = TrOCR(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def forward(self, inputs, **kwargs):
        return self.backbone(inputs, **kwargs)

    def generate_one(self, inputs, **kwargs):
        return self.backbone.generate_one(inputs, **kwargs)

    def loss_fn(self, output, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, **kwargs)
        else:
            raise ValueError("loss fn isn't defined.")
