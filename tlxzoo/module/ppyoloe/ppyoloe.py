import tensorlayerx.nn as nn

from .models import *


class PPYOLOE(nn.Module):
    def __init__(self, backbone, neck, head):
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, scale_factor=None, targets=None):
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.head(fpn_feats)
        if self.is_train:
            return out
        else:
            out = self.head.post_process(out, scale_factor)
            return out

    def loss_fn(self, outputs, targets):
        losses = self.head.get_loss(outputs, targets)
        loss = losses["total_loss"]
        return loss


def ppyoloe(arch, num_classes, data_format, **kwargs):
    if arch == 'ppyoloe_s':
        depth_mult = 0.33
        width_mult = 0.50
    elif arch == 'ppyoloe_m':
        depth_mult = 0.67
        width_mult = 0.75
    elif arch == 'ppyoloe_l':
        depth_mult = 1.0
        width_mult = 1.0
    elif arch == 'ppyoloe_x':
        depth_mult = 1.33
        width_mult = 1.25
    else:
        raise ValueError(f"tlxzoo don`t support {arch}")

    backbone = CSPResNet(layers=[3, 6, 6, 3],
                         channels=[64, 128, 256, 512, 1024],
                         return_idx=[1, 2, 3],
                         use_large_stem=True,
                         depth_mult=depth_mult,
                         width_mult=width_mult,
                         data_format=data_format)
    fpn = CustomCSPPAN(in_channels=[int(256 * width_mult), int(512 * width_mult), int(1024 * width_mult)],
                       out_channels=[768, 384, 192],
                       stage_num=1,
                       block_num=3,
                       act='swish',
                       spp=True,
                       depth_mult=depth_mult,
                       width_mult=width_mult,
                       data_format=data_format)
    static_assigner = ATSSAssigner(topk=9, num_classes=num_classes)
    assigner = TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0)
    head = PPYOLOEHead(static_assigner=static_assigner,
                       assigner=assigner,
                       nms_cfg=dict(
                           score_threshold=0.01,
                           nms_threshold=0.6,
                           nms_top_k=1000,
                           keep_top_k=100
                       ),
                       in_channels=[
                           int(768 * width_mult), int(384 * width_mult), int(192 * width_mult)],
                       fpn_strides=[32, 16, 8],
                       grid_cell_scale=5.0,
                       grid_cell_offset=0.5,
                       static_assigner_epoch=100,
                       use_varifocal_loss=True,
                       num_classes=80,
                       loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, },
                       eval_size=None,
                       data_format=data_format)
    model = PPYOLOE(backbone, fpn, head)
    return model
