import tensorlayerx as tlx
from tensorlayerx import nn
from ..resnet.resnet50 import ResNet50


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tlx.initializers.he_normal()


class ConvUnit(nn.Module):
    """Conv + BN + Act"""

    def __init__(self, f, k, s, wd, act=None, name='ConvBN', **kwargs):
        super(ConvUnit, self).__init__(name=name, **kwargs)
        self.conv = nn.layers.Conv2d(out_channels=f,
                                     kernel_size=(k, k),
                                     stride=(s, s),
                                     padding="same",
                                     W_init=_kernel_init(),
                                     b_init=None,
                                     name=name+'/conv'
                                     )
        self.bn = nn.BatchNorm(name=name+"/bn")

        if act is None:
            self.act_fn = tlx.identity
        elif act == 'relu':
            self.act_fn = tlx.ReLU()
        elif act == 'lrelu':
            self.act_fn = tlx.LeakyReLU(0.1)
        else:
            raise NotImplementedError(
                'Activation function type {} is not recognized.'.format(act))

    def forward(self, x):
        return self.act_fn(self.bn(self.conv(x)))


class FPN(tlx.nn.Module):
    """Feature Pyramid Network"""

    def __init__(self, out_ch, wd, name='FPN', **kwargs):
        super(FPN, self).__init__(name=name, **kwargs)
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.output1 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, name=name+"/output1")
        self.output2 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, name=name+"/output2")
        self.output3 = ConvUnit(f=out_ch, k=1, s=1, wd=wd, act=act, name=name+"/output3")
        self.merge1 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act, name=name+"/merge1")
        self.merge2 = ConvUnit(f=out_ch, k=3, s=1, wd=wd, act=act, name=name+"/merge2")

    def forward(self, x):
        output1 = self.output1(x[0])  # [80, 80, out_ch]
        output2 = self.output2(x[1])  # [40, 40, out_ch]
        output3 = self.output3(x[2])  # [20, 20, out_ch]

        up_h, up_w = tlx.get_tensor_shape(output2)[1], tlx.get_tensor_shape(output2)[2]
        up3 = tlx.resize(output3, [up_h, up_w], method='nearest', antialias=False)
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up_h, up_w = tlx.get_tensor_shape(output1)[1], tlx.get_tensor_shape(output1)[2]
        up2 = tlx.resize(output2, [up_h, up_w], method='nearest', antialias=False)
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return output1, output2, output3


class SSH(tlx.nn.Module):
    """Single Stage Headless Layer"""

    def __init__(self, out_ch, wd, name='SSH', **kwargs):
        super(SSH, self).__init__(name=name, **kwargs)
        assert out_ch % 4 == 0
        act = 'relu'
        if (out_ch <= 64):
            act = 'lrelu'

        self.conv_3x3 = ConvUnit(f=out_ch // 2, k=3, s=1, wd=wd, act=None, name=name+"/conv_3x3")

        self.conv_5x5_1 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act, name=name+"/conv_5x5_1")
        self.conv_5x5_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None, name=name+"/conv_5x5_2")

        self.conv_7x7_2 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=act, name=name+"/conv_7x7_2")
        self.conv_7x7_3 = ConvUnit(f=out_ch // 4, k=3, s=1, wd=wd, act=None, name=name+"/conv_7x7_3")

        self.relu = tlx.ReLU()

    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)

        conv_5x5_1 = self.conv_5x5_1(x)
        conv_5x5 = self.conv_5x5_2(conv_5x5_1)

        conv_7x7_2 = self.conv_7x7_2(conv_5x5_1)
        conv_7x7 = self.conv_7x7_3(conv_7x7_2)

        output = tlx.concat([conv_3x3, conv_5x5, conv_7x7], axis=3)
        output = self.relu(output)

        return output


class BboxHead(nn.Module):
    """Bbox Head Layer"""

    def __init__(self, num_anchor, wd, name='BboxHead', **kwargs):
        super(BboxHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.layers.Conv2d(out_channels=num_anchor * 4,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding="same",
                                     name=name+'/conv'
                                     )

    def forward(self, x):
        h, w = tlx.get_tensor_shape(x)[1], tlx.get_tensor_shape(x)[2]
        x = self.conv(x)

        return tlx.reshape(x, [-1, h * w * self.num_anchor, 4])


class LandmarkHead(nn.Module):
    """Landmark Head Layer"""

    def __init__(self, num_anchor, wd, name='LandmarkHead', **kwargs):
        super(LandmarkHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.layers.Conv2d(out_channels=num_anchor * 10,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding="same",
                                     name=name+'/conv'
                                     )

    def forward(self, x):
        h, w = tlx.get_tensor_shape(x)[1], tlx.get_tensor_shape(x)[2]
        x = self.conv(x)

        return tlx.reshape(x, [-1, h * w * self.num_anchor, 10])


class ClassHead(nn.Module):
    """Class Head Layer"""

    def __init__(self, num_anchor, wd, name='ClassHead', **kwargs):
        super(ClassHead, self).__init__(name=name, **kwargs)
        self.num_anchor = num_anchor
        self.conv = nn.layers.Conv2d(out_channels=num_anchor * 2,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding="same",
                                     name=name+'/conv'
                                     )

    def forward(self, x):
        h, w = tlx.get_tensor_shape(x)[1], tlx.get_tensor_shape(x)[2]
        x = self.conv(x)

        return tlx.reshape(x, [-1, h * w * self.num_anchor, 2])


class RetinaFaceModel(nn.Module):
    def __init__(self, cfg, iou_th=0.4, score_th=0.02, name='RetinaFaceModel'):
        super(RetinaFaceModel, self).__init__(name=name)

        input_size = cfg.input_size
        self.cfg = cfg
        self.wd = cfg.weights_decay
        out_ch = cfg.out_channel
        self.num_anchor = len(cfg.min_sizes[0])
        self.iou_th = iou_th
        self.score_th = score_th

        self.backbone_pick_layers = [80, 142, 174]

        self.backbone = ResNet50(None)

        self.fpn = FPN(out_ch=out_ch, wd=self.wd)

        features = [SSH(out_ch=out_ch, wd=self.wd, name=f'SSH_{i}')
                    for i, f in enumerate(self.backbone_pick_layers)]
        self.features = nn.LayerList(features)

        bboxheads = [BboxHead(self.num_anchor, wd=self.wd, name=f'BboxHead_{i}')
                     for i, _ in enumerate(features)]
        self.bboxheads = nn.LayerList(bboxheads)

        landmarkheads = [LandmarkHead(self.num_anchor, wd=self.wd, name=f'LandmarkHead_{i}')
                         for i, _ in enumerate(features)]
        self.landmarkheads = nn.LayerList(landmarkheads)

        classheads = [ClassHead(self.num_anchor, wd=self.wd, name=f'ClassHead_{i}')
                      for i, _ in enumerate(features)]
        self.classheads = nn.LayerList(classheads)

        self.build(inputs_shape=[2, input_size, input_size, 3])

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, inputs):
        _, _, x = self.backbone(inputs, self.backbone_pick_layers)

        x = self.fpn(x)

        features = []
        for ssh, f in zip(self.features, x):
            features.append(ssh(f))

        bbox_regressions = tlx.concat(
            [bboxhead(f) for bboxhead, f in zip(self.bboxheads, features)], axis=1)

        landm_regressions = tlx.concat(
            [landmarkheads(f) for landmarkheads, f in zip(self.landmarkheads, features)], axis=1)

        classifications = tlx.concat(
            [classhead(f) for classhead, f in zip(self.classheads, features)], axis=1)

        classifications = tlx.softmax(classifications, axis=-1)

        return bbox_regressions, landm_regressions, classifications


