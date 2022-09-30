from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from ..utils import (gather_topk_anchors, check_points_inside_bboxes, flatten,
                    compute_max_iou_anchor, iou_similarity, gather_nd)



class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        self.set_eval()
        
        assert len(tlx.get_tensor_shape(pred_scores)) == len(tlx.get_tensor_shape(pred_bboxes))
        assert len(tlx.get_tensor_shape(gt_labels)) == len(tlx.get_tensor_shape(gt_bboxes)) and \
               len(tlx.get_tensor_shape(gt_bboxes)) == 3

        batch_size, num_anchors, num_classes = tlx.get_tensor_shape(pred_scores)
        _, num_max_boxes, _ = tlx.get_tensor_shape(gt_bboxes)

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = tlx.constant(bg_index, gt_labels.dtype, [batch_size, num_anchors])
            assigned_bboxes = tlx.zeros([batch_size, num_anchors, 4])
            assigned_scores = tlx.zeros(
                [batch_size, num_anchors, num_classes])

            assigned_labels = assigned_labels
            assigned_bboxes = assigned_bboxes
            assigned_scores = assigned_scores
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = tlx.transpose(pred_scores, [0, 2, 1])
        batch_ind = tlx.expand_dims(tlx.arange(0, batch_size, dtype=gt_labels.dtype), -1)
        gt_labels_ind = tlx.stack([batch_ind.repeat([1, num_max_boxes]), tlx.squeeze(gt_labels, -1)], -1)
        bbox_cls_scores = gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = tlx.pow(bbox_cls_scores, self.alpha) * tlx.pow(ious, self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=tlx.cast(tlx.tile(pad_gt_mask, [1, 1, self.topk]), tlx.bool))

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        if tlx.reduce_max(mask_positive_sum) > 1:
            mask_multiple_gts = tlx.tile(tlx.expand_dims(mask_positive_sum, 1) > 1,
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = tlx.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = tlx.reduce_sum(mask_positive, -2)
        assigned_gt_index = tlx.argmax(mask_positive, -2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = tlx.gather(flatten(gt_labels), flatten(assigned_gt_index))
        assigned_labels = tlx.reshape(assigned_labels, [batch_size, num_anchors])
        assigned_labels = tlx.where(
            mask_positive_sum > 0, assigned_labels,
            tlx.constant(bg_index, assigned_labels.dtype, tlx.get_tensor_shape(assigned_labels)))

        assigned_bboxes = tlx.gather(
            tlx.reshape(gt_bboxes, [-1, 4]), flatten(assigned_gt_index))
        assigned_bboxes = tlx.reshape(assigned_bboxes, [batch_size, num_anchors, 4])

        assigned_scores = tlx.OneHot(num_classes + 1)(assigned_labels)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = tlx.gather(
            assigned_scores, tlx.cast(tlx.convert_to_tensor(ind), tlx.int64), axis=-1)
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = alignment_metrics.max(-1, keepdim=True)
        max_ious_per_instance, _ = (ious * mask_positive).max(-1, keepdim=True)
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = tlx.reduce_max(alignment_metrics, -2)
        alignment_metrics = tlx.expand_dims(alignment_metrics, -1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores
