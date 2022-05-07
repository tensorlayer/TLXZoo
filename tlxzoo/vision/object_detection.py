from tlxzoo.module import *
import tensorlayerx as tlx


class ObjectDetection(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(ObjectDetection, self).__init__()
        if backbone == "detr":
            self.backbone = Detr(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, target, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs, **kwargs):
        return self.backbone(inputs, **kwargs)

    def predict(self, inputs):
        self.set_eval()
        out = self.backbone(inputs)
        return out