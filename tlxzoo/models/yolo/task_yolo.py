from ...utils.registry import Registers
import tensorlayerx as tlx
from ...task.task import BaseForObjectDetection
from .config_yolo import YOLOv4ForObjectDetectionTaskConfig
from .yolo import YOLOv4, Convolutional
from ...utils.output import BaseForObjectDetectionTaskOutput, float_tensor, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class YOLOv4ForObjectDetectionTaskOutput(BaseForObjectDetectionTaskOutput):
    logits: Union[float_tensor, list, None] = None


@Registers.tasks.register
class YOLOv4ForObjectDetection(BaseForObjectDetection):
    config_class = YOLOv4ForObjectDetectionTaskConfig

    def __init__(self, config: YOLOv4ForObjectDetectionTaskConfig = None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        super(YOLOv4ForObjectDetection, self).__init__(config)

        if model is not None:
            self.yolo = model
        else:
            self.yolo = YOLOv4(self.config.model_config)

        self.sconv = Convolutional(self.config.sconv_filters_shape, activate=False, bn=False)
        self.mconv = Convolutional(self.config.mconv_filters_shape, activate=False, bn=False)
        self.lconv = Convolutional(self.config.lconv_filters_shape, activate=False, bn=False)

    def forward(self, pixels, labels=None):
        pixels = tlx.cast(pixels, tlx.float32)
        outs = self.yolo(pixels)

        sbbox = self.sconv(outs.soutput)
        mbbox = self.mconv(outs.moutput)
        lbbox = self.lconv(outs.loutput)

        feature_maps = [sbbox, mbbox, lbbox]

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, self.config.train_input_size // 8, self.config.num_labels,
                                           self.config.strides, self.config.anchors, i, self.config.xyscale)
            elif i == 1:
                bbox_tensor = decode_train(fm, self.config.train_input_size // 16, self.config.num_labels,
                                           self.config.strides, self.config.anchors, i, self.config.xyscale)
            else:
                bbox_tensor = decode_train(fm, self.config.train_input_size // 32, self.config.num_labels,
                                           self.config.strides, self.config.anchors, i, self.config.xyscale)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

        return YOLOv4ForObjectDetectionTaskOutput(logits=bbox_tensors)

    def loss_fn(self, pred_result, target):
        giou_loss = conf_loss = prob_loss = 0

        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            target_0 = tlx.cast(target[i][0], tlx.float32)
            target_1 = tlx.cast(target[i][1], tlx.float32)
            loss_items = compute_loss(pred, conv, target_0, target_1, STRIDES=self.config.strides,
                                      NUM_CLASS=self.config.num_labels,
                                      IOU_LOSS_THRESH=self.config.iou_loss_thresh, i=i)
            # if tlx.ops.is_nan(loss_items[0]):
            #     giou_loss += 0
            # else:
            #     giou_loss += loss_items[0]
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        return total_loss


def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE):
    conv_output = tlx.reshape(conv_output, (conv_output.shape[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tlx.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tlx.meshgrid(tlx.range(output_size), tlx.range(output_size))
    xy_grid = tlx.expand_dims(tlx.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tlx.tile(tlx.expand_dims(xy_grid, axis=0), [conv_output.shape[0], 1, 1, 3, 1])

    xy_grid = tlx.cast(xy_grid, tlx.float32)

    pred_xy = ((tlx.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tlx.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tlx.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tlx.sigmoid(conv_raw_conf)
    pred_prob = tlx.sigmoid(conv_raw_prob)

    return tlx.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tlx.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tlx.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tlx.cast(input_size, tlx.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tlx.expand_dims(tlx.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tlx.cast(max_iou < IOU_LOSS_THRESH, tlx.float32)

    conf_focal = tlx.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                  logits=conv_raw_conf)
            +
            respond_bgd * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                 logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=label_prob,
                                                                                      logits=conv_raw_prob)

    giou_loss = tlx.reduce_mean(tlx.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tlx.reduce_mean(tlx.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tlx.reduce_mean(tlx.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tlx.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tlx.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tlx.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tlx.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tlx.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tlx.divide(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tlx.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tlx.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tlx.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tlx.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tlx.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tlx.divide(inter_area, union_area)

    enclose_left_up = tlx.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tlx.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tlx.divide(enclose_area - union_area, enclose_area)

    return giou