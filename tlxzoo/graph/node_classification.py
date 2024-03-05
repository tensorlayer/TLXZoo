import tensorlayerx as tlx
from tlxzoo.module import *


class NodeClassification(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        if backbone == "gcn":
            nfeat = kwargs.pop("nfeat", 1433)
            nhid = kwargs.pop("nhid", 16)
            nclass = kwargs.pop("nclass", 7)
            dropout = kwargs.pop("dropout", 0.5)
            self.backbone = GCN(nfeat, nhid, nclass, dropout)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, target):
        loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)
        return loss

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs):
        self.set_eval()
        out = self.backbone(inputs)
        return tlx.argmax(out, axis=-1)
