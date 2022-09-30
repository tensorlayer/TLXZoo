from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayerx as tlx
import numpy as np


def batch_distance2bbox(points, distance):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = tlx.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = tlx.concat([x1y1, x2y2], -1)
    return out_bbox


def my_multiclass_nms(bboxes, scores, score_threshold=0.7, nms_threshold=0.45, nms_top_k=1000, keep_top_k=100, class_agnostic=False):
    '''
    :param bboxes:   shape = [N, A,  4]   "左上角xy + 右下角xy"格式
    :param scores:   shape = [N, A, 80]
    :param score_threshold:
    :param nms_threshold:
    :param nms_top_k:
    :param keep_top_k:
    :param class_agnostic:
    :return:
    '''

    # 每张图片的预测结果
    output = [None for _ in range(len(bboxes))]
    # 每张图片分开遍历
    for i, (xyxy, score) in enumerate(zip(bboxes, scores)):
        '''
        :var xyxy:    shape = [A, 4]   "左上角xy + 右下角xy"格式
        :var score:   shape = [A, 80]
        '''

        # 每个预测框最高得分的分数和对应的类别id
        class_conf = tlx.reduce_max(score, 1, keepdims=True)
        class_pred = tlx.argmax(score, 1)

        # 分数超过阈值的预测框为True
        conf_mask = tlx.squeeze((tlx.squeeze(class_conf) >= score_threshold))
        # 这样排序 (x1, y1, x2, y2, 得分, 类别id)
        detections = tlx.concat(xyxy, class_conf, tlx.cast(class_pred, tlx.float32), 1)
        # 只保留超过阈值的预测框
        detections = detections[conf_mask]
        if not tlx.get_tensor_shape(detections.size)[0]:
            continue

        # 使用torchvision自带的nms、batched_nms
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_threshold,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_threshold,
            )

        detections = detections[nms_out_index]

        # 保留得分最高的keep_top_k个
        sort_inds = tlx.argsort(detections[:, 4], descending=True)
        if keep_top_k > 0 and len(sort_inds) > keep_top_k:
            sort_inds = sort_inds[:keep_top_k]
        detections = detections[sort_inds, :]

        # 为了保持和matrix_nms()一样的返回风格 cls、score、xyxy。
        detections = tlx.concat((detections[:, 5:6], detections[:, 4:5], detections[:, :4]), 1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = tlx.concat((output[i], detections))

    return output


def numel(x):
    shape = tlx.get_tensor_shape(x)
    if shape:
        return np.prod(shape)
    else:
        return 0


def flatten(x, start_dim=0, end_dim=-1):
    shape = tlx.get_tensor_shape(x)
    end_dim = (end_dim + len(shape)) % len(shape)
    shape[start_dim:end_dim+1] = [-1]
    return tlx.reshape(x, shape)


def topk(x, k, axis=-1, largest=True):
    x = tlx.convert_to_numpy(x)
    indices = np.argsort(x, axis)
    values = np.sort(x, axis)
    if largest:
        indices = np.flip(indices, axis)
        values = np.flip(values, axis)
    indices = indices.take(range(k), axis)
    values = values.take(range(k), axis)
    return tlx.convert_to_tensor(values), tlx.convert_to_tensor(indices)


def gather_topk_anchors(metrics, k, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        k (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, k], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = topk(
        metrics, k, axis=-1, largest=largest)
    if topk_mask is None:
        topk_mask = tlx.tile(tlx.reduce_max(topk_metrics, -1, keepdims=True) > eps,
            [1, 1, k])
    topk_idxs = tlx.where(topk_mask, topk_idxs, tlx.zeros_like(topk_idxs))
    is_in_topk = tlx.reduce_sum(tlx.OneHot(num_anchors)(topk_idxs), -2)
    is_in_topk = tlx.where(is_in_topk > 1,
                              tlx.zeros_like(is_in_topk), is_in_topk)
    return tlx.cast(is_in_topk, metrics.dtype)


def check_points_inside_bboxes(points,
                               bboxes,
                               center_radius_tensor=None,
                               eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = tlx.expand_dims(tlx.expand_dims(points, 0), 1)
    x, y = tlx.split(points, 2, axis=-1)
    xmin, ymin, xmax, ymax = tlx.split(tlx.expand_dims(bboxes, 2), 4, axis=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = tlx.concat([l, t, r, b], -1)
    delta_ltrb_min = tlx.reduce_min(delta_ltrb, -1)
    is_in_bboxes = (delta_ltrb_min > eps)
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = tlx.expand_dims(tlx.expand_dims(center_radius_tensor, 0), 1)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = tlx.concat([l, t, r, b], -1)
        delta_ltrb_c_min = tlx.reduce_min(delta_ltrb_c, -1)
        is_in_center = (delta_ltrb_c_min > eps)
        return (tlx.logical_and(is_in_bboxes, is_in_center),
                tlx.logical_or(is_in_bboxes, is_in_center))

    return tlx.cast(is_in_bboxes, bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = tlx.get_tensor_shape(ious)[-2]
    max_iou_index = tlx.argmax(ious, axis=-2)
    ## TODO
    is_max_iou = tlx.transpose(tlx.OneHot(num_max_boxes)(max_iou_index), (0, 2, 1))
    return tlx.cast(is_max_iou, ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = tlx.get_tensor_shape(ious)[-1]
    max_iou_index = tlx.argmax(ious, axis=-1)
    is_max_iou = tlx.OneHot(num_anchors)(max_iou_index)
    return tlx.cast(is_max_iou, ious.dtype)


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5,
                                   data_format='channels_first'):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        if data_format == 'channels_first':
            _, _, h, w = tlx.get_tensor_shape(feat)
        else:
            _, h, w, _ = tlx.get_tensor_shape(feat)
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (tlx.arange(0, w, dtype=tlx.float32) + grid_cell_offset) * stride
        shift_y = (tlx.arange(0, h, dtype=tlx.float32) + grid_cell_offset) * stride
        shift_y, shift_x = tlx.meshgrid(shift_y, shift_x)
        anchor = tlx.cast(tlx.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            -1), feat.dtype)
        anchor_point = tlx.cast(tlx.stack(
            [shift_x, shift_y], -1), feat.dtype)

        anchors.append(tlx.reshape(anchor, [-1, 4]))
        anchor_points.append(tlx.reshape(anchor_point, [-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            tlx.constant(stride, feat.dtype, [num_anchors_list[-1], 1]))
    anchors = tlx.concat(anchors, 0)
    anchor_points = tlx.concat(anchor_points, 0)
    stride_tensor = tlx.concat(stride_tensor, 0)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    bboxes_a = tlx.cast(bboxes_a, tlx.float32)
    bboxes_b = tlx.cast(bboxes_b, tlx.float32)
    N = tlx.get_tensor_shape(bboxes_a)[0]
    A = tlx.get_tensor_shape(bboxes_a)[1]
    B = tlx.get_tensor_shape(bboxes_b)[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = tlx.concat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                           bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], dim=-1)
        box_b = tlx.concat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                           bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], dim=-1)

    box_a_rb = tlx.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = tlx.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = tlx.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = tlx.tile(box_b_rb, [1, A, 1, 1])
    max_xy = tlx.minimum(box_a_rb, box_b_rb)

    box_a_lu = tlx.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = tlx.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = tlx.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = tlx.tile(box_b_lu, [1, A, 1, 1])
    min_xy = tlx.maximum(box_a_lu, box_b_lu)

    inter = tlx.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = tlx.reshape(area_a, (N, A, 1))
    area_a = tlx.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = tlx.reshape(area_b, (N, 1, B))
    area_b = tlx.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]


def iou_similarity(box1, box2):
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    box1 = tlx.cast(box1, tlx.float32)
    box2 = tlx.cast(box2, tlx.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)

def index_sample_2d(tensor, index):
    assert len(tlx.get_tensor_shape(tensor)) == 2
    assert len(tlx.get_tensor_shape(index)) == 2
    assert index.dtype == tlx.int64
    d0, d1 = tlx.get_tensor_shape(tensor)
    d2, d3 = tlx.get_tensor_shape(index)
    assert d0 == d2
    tensor_ = tlx.reshape(tensor, (-1, ))
    batch_ind = tlx.expand_dims(tlx.arange(0, d0, dtype=index.dtype), -1) * d1
    index_ = index + batch_ind
    index_ = tlx.reshape(index_, (-1, ))
    out = tlx.gather(tensor_, index_)
    out = tlx.reshape(out, (d2, d3))
    return out


def gather_1d(tensor, index):
    assert len(tlx.get_tensor_shape(index)) == 1
    assert index.dtype == tlx.int64
    out = tensor[index]
    return out


def gather_nd(tensor, index):
    if len(tlx.get_tensor_shape(tensor)) == 4 and len(tlx.get_tensor_shape(index)) == 2:
        N, R, S, T = tlx.get_tensor_shape(tensor)
        index_0 = index[:, 0]  # [M, ]
        index_1 = index[:, 1]  # [M, ]
        index_2 = index[:, 2]  # [M, ]
        index_ = index_0 * R * S + index_1 * S + index_2  # [M, ]
        x2 = tlx.reshape(tensor, (N * R * S, T))  # [N*R*S, T]
        index_ = tlx.cast(index_, tlx.int64)
        out = gather_1d(x2, index_)
    elif len(tlx.get_tensor_shape(tensor)) == 3 and len(tlx.get_tensor_shape(index)) == 3:
        A, B, C = tlx.get_tensor_shape(tensor)
        D, E, F = tlx.get_tensor_shape(index)
        assert F == 2
        # out.shape = [D, E, C]
        tensor_ = tlx.reshape(tensor, (-1, C))   # [A*B, C]
        index_ = tlx.reshape(index, (-1, F))     # [D*E, F]


        index_0 = index_[:, 0]  # [D*E, ]
        index_1 = index_[:, 1]  # [D*E, ]
        index_ = index_0 * B + index_1  # [D*E, ]

        out = gather_1d(tensor_, index_)  # [D*E, C]
        out = tlx.reshape(out, (D, E, C))   # [D, E, C]
    else:
        raise NotImplementedError("not implemented.")
    return out


def identity(x):
    return x

def mish(x):
    return x * tlx.tanh(tlx.softplus(x))

def swish(x):
    return x * tlx.sigmoid(x)

def hardsigmoid(x):
    x = tlx.where(tlx.logical_and(x>-3, x<3), x/6 + 0.5, x)
    x = tlx.where(x<=-3, tlx.zeros_like(x), x)
    x = tlx.where(x>=3, tlx.ones_like(x), x)
    return x

ACT_SPEC = {'mish': mish, 'swish': swish, 'hardsigmoid': hardsigmoid}


def get_act_fn(act=None):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(tlx, name)

    return lambda x: fn(x, **kwargs)