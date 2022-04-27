from ...feature.feature import BaseImageFeature
from ...config.config import BaseFeatureConfig
from ...utils.registry import Registers
import cv2
import numpy as np
import math
from itertools import product as product
import tensorlayerx as tlx
from .non_max_suppression import non_max_suppression_np


@Registers.feature_configs.register
class RetinaFaceFeatureConfig(BaseFeatureConfig):
    def __init__(self,
                 input_size=640,
                 min_sizes=None,
                 steps=None,
                 clip=False,
                 match_thresh=0.45,
                 ignore_thresh=0.3,
                 max_steps=32,
                 variances=None,
                 **kwargs):
        self.input_size = input_size
        self.min_sizes = min_sizes if min_sizes else [[16, 32], [64, 128], [256, 512]]
        self.steps = steps if steps else [8, 16, 32]
        self.clip = clip
        self.match_thresh = match_thresh
        self.ignore_thresh = ignore_thresh
        self.max_steps = max_steps
        self.variances = variances if variances else [0.1, 0.2]
        super(RetinaFaceFeatureConfig, self).__init__(**kwargs)


@Registers.features.register
class RetinaFaceFeature(BaseImageFeature):
    config_class = RetinaFaceFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config
        super(RetinaFaceFeature, self).__init__(config, **kwargs)
        priors = prior_box((self.config.input_size, self.config.input_size),
                           self.config.min_sizes, self.config.steps, self.config.clip)
        priors = priors.astype(np.float32)

        self.is_train = True

        self.priors = priors

    def _resize(self, img, labels, img_dim):
        img_h, img_w, _ = img.shape
        w_f = img_w * 1.0
        h_f = img_h * 1.0
        locs = np.stack([labels[:, 0] / w_f, labels[:, 1] / h_f,
                         labels[:, 2] / w_f, labels[:, 3] / h_f,
                         labels[:, 4] / w_f, labels[:, 5] / h_f,
                         labels[:, 6] / w_f, labels[:, 7] / h_f,
                         labels[:, 8] / w_f, labels[:, 9] / h_f,
                         labels[:, 10] / w_f, labels[:, 11] / h_f,
                         labels[:, 12] / w_f, labels[:, 13] / h_f], axis=1)

        locs = np.clip(locs, 0, 1)

        labels = np.concatenate([locs, labels[:, 14:15]], axis=1)
        img = self.resize(img, img_dim)

        return img, labels

    def decode_one(self, bbox_regressions, landm_regressions, classifications, inputs, pad_params,
                   iou_th=0.4, score_th=0.02):
        if not isinstance(pad_params[0], int):
            pad_params = [int(i.numpy()) for i in pad_params]
        bbox_regressions_np = tlx.convert_to_numpy(bbox_regressions)
        landm_regressions_np = tlx.convert_to_numpy(landm_regressions)
        classifications_np = tlx.convert_to_numpy(classifications)
        preds_np = np.concatenate(
            [bbox_regressions_np[0], landm_regressions_np[0],
             np.ones_like(classifications_np[0, :, 0][..., np.newaxis]),
             classifications_np[0, :, 1][..., np.newaxis]], 1)
        priors_np = prior_box((tlx.get_tensor_shape(inputs)[1], tlx.get_tensor_shape(inputs)[2]),
                              self.config.min_sizes, self.config.steps, self.config.clip)

        decode_preds_np = decode(preds_np, priors_np, self.config.variances)

        selected_indices = non_max_suppression_np(boxes=decode_preds_np[:, :4],
                                                  scores=decode_preds_np[:, -1],
                                                  conf_thres=score_th,
                                                  iou_thres=iou_th)
        out = decode_preds_np[selected_indices]
        outputs = recover_pad_output(out, pad_params)
        return outputs

    def set_eval(self):
        self.is_train = False

    def set_train(self):
        self.is_train = True

    def __call__(self, image_path, label):
        img_raw = cv2.imread(image_path)
        img_height, img_width, _ = img_raw.shape
        img = np.float32(img_raw.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            img = _pad_to_square(img)
            img, labels = self._resize(img, label, self.config.input_size)
            labels = labels.astype(np.float32)

            labels = encode(labels=labels, priors=self.priors,
                            match_thresh=self.config.match_thresh,
                            ignore_thresh=self.config.ignore_thresh,
                            variances=self.config.variances)

            return img, labels
        else:
            img, pad_params = pad_input_image(img, max_steps=self.config.max_steps)

            if label is not None:
                labels = label.astype(np.float32)
                labels = encode(labels=labels, priors=self.priors,
                                match_thresh=self.config.match_thresh,
                                ignore_thresh=self.config.ignore_thresh,
                                variances=self.config.variances)
            else:
                labels = label
            return img, (labels, pad_params, image_path)


def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params

    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
                 [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs


def _pad_to_square(img):
    img_h, img_w, _ = img.shape
    img_pad_h = 0
    img_pad_w = 0
    if img_w > img_h:
        img_pad_h = img_w - img_h
    else:
        img_pad_w = img_h - img_w

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    return img


def prior_box(image_sizes, min_sizes, steps, clip=False):
    """prior box"""
    feature_maps = [
        [math.ceil(image_sizes[0] / step), math.ceil(image_sizes[1] / step)]
        for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / image_sizes[1]
                s_ky = min_size / image_sizes[0]
                cx = (j + 0.5) * steps[k] / image_sizes[1]
                cy = (i + 0.5) * steps[k] / image_sizes[0]
                anchors += [cx, cy, s_kx, s_ky]

    output = np.asarray(anchors).reshape([-1, 4])

    if clip:
        output = np.clip(output, 0, 1)

    return output


def get_sorted_top_k(array, top_k=1, axis=-1, reverse=True):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        if top_k == 1:
            partition_index = np.argmax(array, axis=-1)
            partition_index = partition_index[..., None]
        else:
            axis_length = array.shape[axis]
            partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                      range(axis_length - top_k, axis_length), axis)

    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


def encode(labels, priors, match_thresh, ignore_thresh, variances=[0.1, 0.2]):
    """tensorflow encoding"""
    assert ignore_thresh <= match_thresh
    priors = priors.astype(np.float32)
    bbox = labels[:, :4]
    landm = labels[:, 4:-1]
    landm_valid = labels[:, -1]  # 1: with landm, 0: w/o landm.

    # jaccard index
    overlaps = _jaccard(bbox, _point_form(priors))

    # (Bipartite Matching)
    # [num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = get_sorted_top_k(overlaps)
    best_prior_overlap = best_prior_overlap[:, 0]
    best_prior_idx = best_prior_idx[:, 0]

    # [num_priors] best ground truth for each prior
    overlaps_t = np.transpose(overlaps)
    best_truth_overlap, best_truth_idx = get_sorted_top_k(overlaps_t)

    best_truth_overlap = best_truth_overlap[:, 0]
    best_truth_idx = best_truth_idx[:, 0]

    for i in range(best_prior_idx.shape[0]):
        if best_prior_overlap[i] > match_thresh:
            bp_mask = np.eye(best_truth_idx.shape[0])[best_prior_idx[i]]
            bp_mask_int = bp_mask.astype(np.int)
            new_bt_idx = best_truth_idx * (1 - bp_mask_int) + bp_mask_int * i
            bp_mask_float = bp_mask.astype(np.float32)
            new_bt_overlap = best_truth_overlap * (1 - bp_mask_float) + bp_mask_float * 2

            best_truth_idx, best_truth_overlap = new_bt_idx, new_bt_overlap

    best_truth_idx = best_truth_idx.astype(np.int32)
    best_truth_overlap = best_truth_overlap.astype(np.float32)

    matches_bbox = bbox[best_truth_idx]
    matches_landm = landm[best_truth_idx]
    matches_landm_v = landm_valid[best_truth_idx]

    loc_t = _encode_bbox(matches_bbox, priors, variances)
    landm_t = _encode_landm(matches_landm, priors, variances)

    landm_valid_t = (matches_landm_v > 0).astype(np.float32)
    conf_t = (best_truth_overlap > match_thresh).astype(np.float32)

    conf_t = np.where(
        np.logical_and(best_truth_overlap < match_thresh,
                       best_truth_overlap > ignore_thresh),
        np.ones_like(conf_t) * -1, conf_t)  # 1: pos, 0: neg, -1: ignore

    return np.concatenate([loc_t, landm_t, landm_valid_t[..., None], conf_t[..., None]], axis=1)


def _point_form(boxes):
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def _intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2]:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.broadcast_to(np.expand_dims(box_a[:, 2:], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(box_b[:, 2:], 0), [A, B, 2]))
    min_xy = np.maximum(
        np.broadcast_to(np.expand_dims(box_a[:, :2], 1), [A, B, 2]),
        np.broadcast_to(np.expand_dims(box_b[:, :2], 0), [A, B, 2]))
    inter = np.maximum((max_xy - min_xy), np.zeros_like(max_xy - min_xy))
    return inter[:, :, 0] * inter[:, :, 1]


def _jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = _intersect(box_a, box_b)
    area_a = np.broadcast_to(
        np.expand_dims(
            (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]), 1),
        inter.shape)  # [A,B]
    area_b = np.broadcast_to(
        np.expand_dims(
            (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]), 0),
        inter.shape)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def _encode_bbox(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return np.concatenate([g_cxcy, g_wh], 1)  # [num_priors,4]


def _encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    matched = np.reshape(matched, [matched.shape[0], 5, 2])
    priors = np.broadcast_to(
        np.expand_dims(priors, 1), [matched.shape[0], 5, 4])
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = np.reshape(g_cxcy, [g_cxcy.shape[0], -1])
    # return target for smooth_l1_loss
    return g_cxcy


def draw_bbox_landm(img, ann, img_height, img_width, index=None):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # confidence
    text = "{:.4f}".format(ann[15])
    if index:
        text = str(index) + ":" + text
    cv2.putText(img, text, (int(ann[0] * img_width), int(ann[1] * img_height)),
                cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        cv2.circle(img, (int(ann[4] * img_width),
                         int(ann[5] * img_height)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(ann[6] * img_width),
                         int(ann[7] * img_height)), 1, (0, 255, 255), 2)
        cv2.circle(img, (int(ann[8] * img_width),
                         int(ann[9] * img_height)), 1, (255, 0, 0), 2)
        cv2.circle(img, (int(ann[10] * img_width),
                         int(ann[11] * img_height)), 1, (0, 100, 255), 2)
        cv2.circle(img, (int(ann[12] * img_width),
                         int(ann[13] * img_height)), 1, (255, 0, 100), 2)


def decode(labels, priors, variances=[0.1, 0.2]):
    bbox = _decode_bbox(labels[:, :4], priors, variances)
    landm = _decode_landm(labels[:, 4:14], priors, variances)
    landm_valid = labels[:, 14][:, np.newaxis]
    conf = labels[:, 15][:, np.newaxis]

    return np.concatenate([bbox, landm, landm_valid, conf], axis=1)


def _decode_bbox(pre, priors, variances=[0.1, 0.2]):
    centers = priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:]
    sides = priors[:, 2:] * np.exp(pre[:, 2:] * variances[1])

    return np.concatenate([centers - sides / 2, centers + sides / 2], axis=1)


def _decode_landm(pre, priors, variances=[0.1, 0.2]):
    landms = np.concatenate(
        [priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
         priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]], axis=1)
    return landms
