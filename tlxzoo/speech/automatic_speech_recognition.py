from tlxzoo.module import *
import tensorlayerx as tlx


class AutomaticSpeechRecognition(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(AutomaticSpeechRecognition, self).__init__()
        if backbone == "wav2vec":
            self.backbone = Wav2Vec2(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, target, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target, **kwargs)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs, **kwargs):
        return self.backbone(inputs, **kwargs)

    def predict(self, inputs, **kwargs):
        self.set_eval()
        out = self.backbone(inputs, **kwargs)
        return tlx.argmax(out, axis=-1)