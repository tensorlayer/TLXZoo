import math

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from .backbone import ConvBNLayer
from .utils import (batch_distance2bbox, flatten,
                    generate_anchors_for_grid_cell, get_act_fn,
                    my_multiclass_nms)


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = tlx.maximum(x1, x1g)
        ykis1 = tlx.maximum(y1, y1g)
        xkis2 = tlx.minimum(x2, x2g)
        ykis2 = tlx.minimum(y2, y2g)
        w_inter = tlx.relu(xkis2 - xkis1)
        h_inter = tlx.relu(ykis2 - ykis1)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = tlx.split(pbox, 4, -1)
        x1g, y1g, x2g, y2g = tlx.split(gbox, 4, -1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = tlx.minimum(x1, x1g)
        yc1 = tlx.minimum(y1, y1g)
        xc2 = tlx.maximum(x2, x2g)
        yc2 = tlx.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = tlx.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - \
                loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = tlx.reduce_sum(giou * iou_weight)
        else:
            loss = tlx.reduce_mean(giou * iou_weight)
        return loss * self.loss_weight


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish', act_name='swish', data_format='channels_first'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='VALID',
            data_format=data_format,
            W_init=tlx.initializers.random_normal(stddev=0.001))
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1,
                                act=act, act_name=act_name, data_format=data_format)

    def forward(self, feat, avg_feat):
        weight = tlx.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 nms_cfg=None,
                 data_format='channels_first'):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        self.nms_cfg = nms_cfg
        self.data_format = data_format
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act_name = act
        act = get_act_fn(act) if act is None or isinstance(
            act, (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(
                ESEAttn(in_c, act=act, act_name=act_name, data_format=data_format))
            self.stem_reg.append(
                ESEAttn(in_c, act=act, act_name=act_name, data_format=data_format))
        # pred head
        bias_cls = float(-math.log((1 - 0.01) / 0.01))
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=self.num_classes,
                    kernel_size=(3, 3),
                    padding='SAME',
                    W_init=tlx.initializers.Constant(0.0),
                    b_init=tlx.initializers.Constant(bias_cls),
                    data_format=data_format))
            self.pred_reg.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=(3, 3),
                    padding='SAME',
                    W_init=tlx.initializers.Constant(0.0),
                    b_init=tlx.initializers.Constant(1.0),
                    data_format=data_format))
        # projection conv
        self.proj_conv = nn.Conv2d(
            in_channels=self.reg_max + 1,
            out_channels=1,
            kernel_size=(1, 1),
            padding='VALID',
            b_init=None,
            data_format=data_format)

        self.proj = tlx.reshape(tlx.cast(tlx.linspace(
            0, self.reg_max, self.reg_max + 1), tlx.float32), (-1, 1))

    def _init_weights(self):
        self.proj = tlx.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset, self.data_format)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = tlx.nn.AdaptiveAvgPool2d(
                (1, 1), data_format=self.data_format)(feat)
            cls_logit = self.pred_cls[i](
                self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = tlx.sigmoid(cls_logit)
            if self.data_format == 'channels_first':
                cls_score_list.append(tlx.transpose(
                    flatten(cls_score, 2), (0, 2, 1)))
                reg_distri_list.append(tlx.transpose(
                    flatten(reg_distri, 2), (0, 2, 1)))
            else:
                cls_score_list.append(flatten(cls_score, 1, 2))
                reg_distri_list.append(flatten(reg_distri, 1, 2))
        cls_score_list = tlx.concat(cls_score_list, 1)
        reg_distri_list = tlx.concat(reg_distri_list, 1)

        return (
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        )

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                if self.data_format == 'channels_first':
                    _, _, h, w = tlx.get_tensor_shape(feats[i])
                else:
                    _, h, w, _ = tlx.get_tensor_shape(feats[i])
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = tlx.arange(0, w) + self.grid_cell_offset
            shift_y = tlx.arange(0, h) + self.grid_cell_offset
            shift_y, shift_x = tlx.meshgrid(shift_y, shift_x)
            anchor_point = tlx.cast(
                tlx.stack([shift_x, shift_y], -1), tlx.float32)
            anchor_points.append(tlx.reshape(anchor_point, [-1, 2]))
            stride_tensor.append(
                tlx.constant(stride, tlx.float32, [h * w, 1]))
        anchor_points = tlx.concat(anchor_points)
        stride_tensor = tlx.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            if self.data_format == 'channels_first':
                b, _, h, w = tlx.get_tensor_shape(feat)
            else:
                b, h, w, _ = tlx.get_tensor_shape(feat)
            l = h * w
            avg_feat = tlx.nn.AdaptiveAvgPool2d(
                (1, 1), data_format=self.data_format)(feat)
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = tlx.reshape(reg_dist, [-1, 4, self.reg_max + 1, l])
            reg_dist = tlx.transpose(reg_dist, (0, 2, 1, 3))
            reg_dist = tlx.softmax(reg_dist, axis=1)
            reg_dist = self.proj_conv(reg_dist)
            # cls and reg
            cls_score = tlx.sigmoid(cls_logit)
            cls_score_list.append(tlx.reshape(
                cls_score, [b, self.num_classes, l]))
            reg_dist_list.append(tlx.reshape(reg_dist, [b, 4, l]))

        cls_score_list = tlx.concat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = tlx.concat(reg_dist_list, -1)    # [N,  4, A]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.is_train:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = tlx.pow(score - label, gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        score = tlx.cast(score, tlx.float32)
        eps = 1e-9
        loss = label * (0 - tlx.log(score + eps)) + \
            (1.0 - label) * (0 - tlx.log(1.0 - score + eps))
        loss *= weight
        loss = tlx.reduce_sum(loss)
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * tlx.pow(pred_score, gamma) * \
            (1 - label) + gt_score * label

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = tlx.cast(pred_score, tlx.float32)
        eps = 1e-9
        loss = gt_score * (0 - tlx.log(pred_score + eps)) + \
            (1.0 - gt_score) * (0 - tlx.log(1.0 - pred_score + eps))
        loss *= weight
        loss = tlx.reduce_sum(loss)
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = tlx.get_tensor_shape(pred_dist)
        pred_dist = tlx.reshape(pred_dist, [b, l, 4, self.reg_max + 1])
        pred_dist = tlx.softmax(pred_dist, axis=-1)
        pred_dist = tlx.squeeze(tlx.matmul(pred_dist, self.proj), -1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = tlx.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return tlx.clip_by_value(tlx.concat([lt, rb], -1), 0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = tlx.cast(target, tlx.int64)
        target_right = target_left + 1
        weight_left = tlx.cast(target_right, tlx.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = tlx.softmax(pred_dist, axis=-1)
        target_left_onehot = tlx.OneHot(pred_dist_act.shape[-1])(target_left)
        target_right_onehot = tlx.OneHot(pred_dist_act.shape[-1])(target_right)
        loss_left = target_left_onehot * (0 - tlx.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - tlx.log(pred_dist_act + eps))
        loss_left = tlx.reduce_sum(loss_left, -1) * weight_left
        loss_right = tlx.reduce_sum(loss_right, -1) * weight_right
        return tlx.reduce_mean(loss_left + loss_right, -1, keepdims=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = tlx.reduce_sum(tlx.cast(mask_positive, tlx.int64))
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = tlx.tile(tlx.expand_dims(mask_positive, -1), [1, 1, 4])
            pred_bboxes_pos = tlx.reshape(tlx.mask_select(pred_bboxes,
                                                          bbox_mask), [-1, 4])
            assigned_bboxes_pos = tlx.reshape(tlx.mask_select(
                assigned_bboxes, bbox_mask), [-1, 4])
            bbox_weight = tlx.expand_dims(tlx.mask_select(
                tlx.reduce_sum(assigned_scores, -1), mask_positive), -1)

            loss_l1 = tlx.reduce_sum(
                tlx.abs(pred_bboxes_pos - assigned_bboxes_pos))

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = tlx.reduce_sum(loss_iou) / assigned_scores_sum

            dist_mask = tlx.tile(tlx.expand_dims(mask_positive, -1),
                                 [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = tlx.reshape(tlx.mask_select(
                pred_dist, dist_mask), [-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = tlx.reshape(tlx.mask_select(
                assigned_ltrb, bbox_mask), [-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = tlx.reduce_sum(loss_dfl) / assigned_scores_sum
        else:
            loss_l1 = tlx.zeros([])
            loss_iou = tlx.zeros([])
            loss_dfl = tlx.zeros([])
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
            anchor_points, num_anchors_list, stride_tensor = head_outs
        anchors = anchors
        anchor_points = anchor_points
        stride_tensor = stride_tensor

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['gt_class']
        gt_labels = tlx.cast(gt_labels, tlx.int64)
        gt_bboxes = gt_meta['gt_bbox']
        pad_gt_mask = gt_meta['pad_gt_mask']

        # miemie2013: 剪掉填充的gt
        num_boxes = tlx.reduce_sum(pad_gt_mask, [1, 2])
        num_max_boxes = tlx.cast(tlx.reduce_max(num_boxes), tlx.int32)
        pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
        gt_labels = gt_labels[:, :num_max_boxes, :]
        gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=tlx.convert_to_tensor(pred_bboxes) * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                    tlx.convert_to_tensor(pred_scores),
                    tlx.convert_to_tensor(pred_bboxes) * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = tlx.OneHot(
                self.num_classes + 1)(assigned_labels)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        # 每张卡上的assigned_scores_sum求平均，而且max(x, 1)
        assigned_scores_sum = tlx.reduce_sum(assigned_scores)
        assigned_scores_sum = tlx.relu(
            assigned_scores_sum - 1.) + 1.  # y = max(x, 1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
            self.loss_weight['iou'] * loss_iou + \
            self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'total_loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(
            anchor_points, tlx.transpose(pred_dist, (0, 2, 1)))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        scale_y, scale_x = tlx.split(scale_factor, 2, -1)
        scale_factor = tlx.reshape(tlx.concat(
            [scale_x, scale_y, scale_x, scale_y], -1), [-1, 1, 4])
        # [N, A, 4]     pred_scores.shape = [N, 80, A]
        pred_bboxes /= scale_factor
        return pred_bboxes
        # nms
        # preds = []
        # yolo_scores = tlx.transpose(pred_scores, (0, 2, 1))  # [N, A, 80]
        # preds = my_multiclass_nms(pred_bboxes, yolo_scores, **self.nms_cfg)
        # return preds
