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


class RetinaFace(nn.Module):
    def __init__(self, input_size=640, weights_decay=5e-4, out_channel=256,
                 min_sizes=None, iou_th=0.4, score_th=0.02, name='RetinaFace'):
        """
        :param input_size: (:obj:`int`, `optional`, defaults to 640):
            input size for build model.
        :param weights_decay: (:obj:`float`, `optional`, defaults to 5e-4):
            weights decay of ConvUnit.
        :param out_channel: (:obj:`int`, `optional`, defaults to 256):
            out Dimensionality of SSH.
        :param min_sizes: (:obj:`list`, `optional`, defaults to [[16, 32], [64, 128], [256, 512]]):
            sizes for anchor.
        :param iou_th: (:obj:`float`, `optional`, defaults to 0.4):
            iou threshold
        :param score_th: (:obj:`float`, `optional`, defaults to 0.02):
            score threshold
        """
        super(RetinaFace, self).__init__(name=name)

        min_sizes = min_sizes if min_sizes else [[16, 32], [64, 128], [256, 512]]
        self.input_size = input_size
        self.wd = weights_decay
        out_ch = out_channel
        self.num_anchor = len(min_sizes[0])
        self.iou_th = iou_th
        self.score_th = score_th

        self.backbone_pick_layers = [80, 142, 174]

        self.backbone = ResNet50(None)

        self.fpn = FPN(out_ch=out_ch, wd=self.wd)

        features = [SSH(out_ch=out_ch, wd=self.wd, name=f'SSH_{i}')
                    for i, f in enumerate(self.backbone_pick_layers)]
        self.features = nn.ModuleList(features)

        bboxheads = [BboxHead(self.num_anchor, wd=self.wd, name=f'BboxHead_{i}')
                     for i, _ in enumerate(features)]
        self.bboxheads = nn.ModuleList(bboxheads)

        landmarkheads = [LandmarkHead(self.num_anchor, wd=self.wd, name=f'LandmarkHead_{i}')
                         for i, _ in enumerate(features)]
        self.landmarkheads = nn.ModuleList(landmarkheads)

        classheads = [ClassHead(self.num_anchor, wd=self.wd, name=f'ClassHead_{i}')
                      for i, _ in enumerate(features)]
        self.classheads = nn.ModuleList(classheads)

        self.build(inputs_shape=[2, input_size, input_size, 3])

        self.multi_box_loss = MultiBoxLoss()

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def loss_fn(self, predictions, labels):
        losses = {}
        losses['loc'], losses['landm'], losses['class'] = self.multi_box_loss(labels, predictions)
        total_loss = tlx.add_n([l for l in losses.values()])
        return total_loss

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


def _smooth_l1_loss(y_true, y_pred):
    t = tlx.abs(y_pred - y_true)
    return tlx.where(t < 1, 0.5 * t ** 2, t - 0.5)


def MultiBoxLoss(num_class=2, neg_pos_ratio=3):
    """multi-box loss"""
    def multi_box_loss(y_true, y_pred):
        num_batch = tlx.get_tensor_shape(y_true)[0]
        num_prior = tlx.get_tensor_shape(y_true)[1]

        loc_pred = tlx.reshape(y_pred[0], [num_batch * num_prior, 4])
        landm_pred = tlx.reshape(y_pred[1], [num_batch * num_prior, 10])
        class_pred = tlx.reshape(y_pred[2], [num_batch * num_prior, num_class])
        loc_true = tlx.reshape(y_true[..., :4], [num_batch * num_prior, 4])
        landm_true = tlx.reshape(y_true[..., 4:14], [num_batch * num_prior, 10])
        landm_valid = tlx.reshape(y_true[..., 14], [num_batch * num_prior, 1])
        class_true = tlx.reshape(y_true[..., 15], [num_batch * num_prior, 1])

        # define filter mask: class_true = 1 (pos), 0 (neg), -1 (ignore)
        #                     landm_valid = 1 (w landm), 0 (w/o landm)
        mask_pos = tlx.equal(class_true, 1)
        mask_neg = tlx.equal(class_true, 0)
        mask_landm = tlx.logical_and(tlx.equal(landm_valid, 1), mask_pos)

        # landm loss (smooth L1)
        mask_landm_b = tlx.tile(mask_landm, [1, tlx.get_tensor_shape(landm_true)[1]])
        loss_landm = _smooth_l1_loss(landm_true[mask_landm_b],
                                     landm_pred[mask_landm_b])
        loss_landm = tlx.reduce_mean(loss_landm)

        # localization loss (smooth L1)
        mask_pos_b = tlx.tile(mask_pos, [1, tlx.get_tensor_shape(loc_true)[1]])

        loss_loc = _smooth_l1_loss(loc_true[mask_pos_b],
                                   loc_pred[mask_pos_b])
        loss_loc = tlx.reduce_mean(loss_loc)

        # classification loss (crossentropy)
        # 1. compute max conf across batch for hard negative mining
        loss_class = tlx.where(mask_neg,
                              1 - class_pred[:, 0][..., None], 0)

        # 2. hard negative mining
        loss_class = tlx.reshape(loss_class, [num_batch, num_prior])
        loss_class_idx = tlx.argsort(loss_class, axis=1, descending=True)
        loss_class_idx_rank = tlx.argsort(loss_class_idx, axis=1)
        mask_pos_per_batch = tlx.reshape(mask_pos, [num_batch, num_prior])
        num_pos_per_batch = tlx.reduce_sum(
                tlx.cast(mask_pos_per_batch, tlx.float32), 1, keepdims=True)
        num_pos_per_batch = tlx.maximum(num_pos_per_batch, 1)
        num_neg_per_batch = tlx.minimum(neg_pos_ratio * num_pos_per_batch,
                                       tlx.cast(num_prior, tlx.float32) - 1)
        mask_hard_neg = tlx.reshape(
            tlx.cast(loss_class_idx_rank, tlx.float32) < num_neg_per_batch,
            [num_batch * num_prior, 1])

        # 3. classification loss including positive and negative examples
        loss_class_mask = tlx.logical_or(mask_pos, mask_hard_neg)
        loss_class_mask_b = tlx.tile(loss_class_mask, [1, tlx.get_tensor_shape(class_pred)[1]])
        filter_class_true = tlx.cast(mask_pos, tlx.int32)[loss_class_mask]
        filter_class_pred = class_pred[loss_class_mask_b]
        filter_class_pred = tlx.reshape(filter_class_pred, [-1, num_class])

        loss_class = tlx.losses.softmax_cross_entropy_with_logits(output=filter_class_pred,
                                                                  target=filter_class_true, reduction="mean")

        return loss_loc, loss_landm, loss_class

    return multi_box_loss