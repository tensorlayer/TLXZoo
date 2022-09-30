from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorlayerx as tlx
import tensorlayerx.nn as nn

from ..utils import (check_points_inside_bboxes, compute_max_iou_anchor, flatten,
                    compute_max_iou_gt, topk, iou_similarity, index_sample_2d)


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return tlx.stack([boxes_cx, boxes_cy], -1)


def l2_norm(x, axis):
    return tlx.sqrt(tlx.reduce_sum(x*x, axis))


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = tlx.cast(tlx.tile(pad_gt_mask, [1, 1, self.topk]), tlx.bool)
        gt2anchor_distances_list = tlx.split(
            gt2anchor_distances, num_anchors_list, -1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = topk(
                distances, self.topk, -1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = tlx.where(pad_gt_mask, topk_idxs,
                                     tlx.zeros_like(topk_idxs))
            is_in_topk = tlx.reduce_sum(tlx.OneHot(num_anchors)(topk_idxs), axis=-2)
            is_in_topk = tlx.where(is_in_topk > 1,
                                      tlx.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(tlx.cast(is_in_topk, gt2anchor_distances.dtype))
        is_in_topk_list = tlx.concat(is_in_topk_list, -1)
        topk_idxs_list = tlx.concat(topk_idxs_list, -1)
        return is_in_topk_list, topk_idxs_list

    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        self.set_eval()
        
        assert len(tlx.get_tensor_shape(gt_labels)) == len(tlx.get_tensor_shape(gt_bboxes)) and \
               len(tlx.get_tensor_shape(gt_bboxes)) == 3

        num_anchors, _ = tlx.get_tensor_shape(anchor_bboxes)
        batch_size, num_max_boxes, _ = tlx.get_tensor_shape(gt_bboxes)

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = tlx.constant(bg_index, gt_labels.dtype, [batch_size, num_anchors])
            assigned_bboxes = tlx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tlx.zeros(
                [batch_size, num_anchors, self.num_classes])

            assigned_labels = assigned_labels
            assigned_bboxes = assigned_bboxes
            assigned_scores = assigned_scores
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        batch_anchor_bboxes = tlx.tile(tlx.expand_dims(anchor_bboxes, 0), [batch_size, 1, 1])
        ious = iou_similarity(gt_bboxes, batch_anchor_bboxes)

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = tlx.expand_dims(bbox_center(tlx.reshape(gt_bboxes, [-1, 4])), 1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = tlx.reshape(l2_norm(gt_centers - tlx.expand_dims(anchor_centers, 0), axis=-1), [batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        aaaaaa1 = tlx.reshape(iou_candidates, (-1, iou_candidates.shape[-1]))
        aaaaaa2 = tlx.reshape(topk_idxs, (-1, topk_idxs.shape[-1]))
        iou_threshold = index_sample_2d(aaaaaa1, aaaaaa2)
        iou_threshold = tlx.reshape(iou_threshold, [batch_size, num_max_boxes, -1])
        iou_threshold = tlx.reduce_mean(iou_threshold, -1, keepdims=True) + \
                        tlx.reduce_std(iou_threshold, -1, keepdims=True)
        is_in_topk = tlx.where(
            iou_candidates > tlx.tile(iou_threshold, [1, 1, num_anchors]),
            is_in_topk, tlx.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        if tlx.reduce_max(mask_positive_sum) > 1:
            mask_multiple_gts = tlx.tile(tlx.expand_dims(mask_positive_sum, 1) > 1,
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            # when use fp16
            mask_positive = tlx.where(mask_multiple_gts, is_max_iou, tlx.cast(mask_positive, is_max_iou.dtype))
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = tlx.tile(tlx.reduce_sum(is_max_iou, -2, keepdims=True) == 1,
                [1, num_max_boxes, 1])
            mask_positive = tlx.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        assigned_gt_index = tlx.argmax(mask_positive, -2)

        # assigned target
        batch_ind = tlx.expand_dims(tlx.arange(0, batch_size, dtype=gt_labels.dtype), -1)
        batch_ind = batch_ind
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tlx.gather(
            flatten(gt_labels), tlx.cast(flatten(assigned_gt_index), tlx.int64))
        assigned_labels = tlx.reshape(assigned_labels, [batch_size, num_anchors])
        assigned_labels = tlx.where(
            mask_positive_sum > 0, assigned_labels,
            tlx.constant(bg_index, assigned_labels.dtype, tlx.get_tensor_shape(assigned_labels)))

        assigned_bboxes = tlx.gather(
            tlx.reshape(gt_bboxes, [-1, 4]), tlx.cast(flatten(assigned_gt_index), tlx.int64))
        assigned_bboxes = tlx.reshape(assigned_bboxes, [batch_size, num_anchors, 4])

        assigned_scores = tlx.OneHot(self.num_classes + 1)(assigned_labels)
        assigned_scores = tlx.cast(assigned_scores, tlx.float32)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tlx.gather(assigned_scores, tlx.cast(tlx.convert_to_tensor(ind), tlx.int64), axis=-1)
        if pred_bboxes is not None:
            # assigned iou
            ious = iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious_max = tlx.reduce_max(ious, -2)
            ious_max = tlx.expand_dims(ious_max, -1)
            assigned_scores *= ious_max
        elif gt_scores is not None:
            gather_scores = tlx.gather(
                flatten(gt_scores), tlx.cast(flatten(assigned_gt_index), tlx.int64))
            gather_scores = tlx.reshape(gather_scores, [batch_size, num_anchors])
            gather_scores = tlx.where(mask_positive_sum > 0, gather_scores,
                                         tlx.zeros_like(gather_scores))
            assigned_scores *= tlx.expand_dims(gather_scores, -1)
        return assigned_labels, assigned_bboxes, assigned_scores
