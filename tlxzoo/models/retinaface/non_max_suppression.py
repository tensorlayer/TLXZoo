import numpy as np
import time
from typing import List, Callable, Tuple


def nms_np(detections: np.ndarray, scores: np.ndarray, max_det: int,
           thresh: float) -> List[np.ndarray]:

    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]

    areas = (x2 - x1 + 0.001) * (y2 - y1 + 0.001)

    # get boxes with more ious first
    order = scores.argsort()[::-1]

    # final output boxes
    keep = []

    while order.size > 0 and len(keep) < max_det:
        # pick maxmum iou box
        i = order[0]
        keep.append(i)

        # get iou
        ovr = get_iou((x1, y1, x2, y2), order, areas, idx=i)

        # drop overlaping boxes
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def get_iou(xyxy: Tuple[np.ndarray], order: np.ndarray, areas: np.ndarray,
            idx: int) -> float:
    x1, y1, x2, y2 = xyxy
    xx1 = np.maximum(x1[idx], x1[order[1:]])
    yy1 = np.maximum(y1[idx], y1[order[1:]])
    xx2 = np.minimum(x2[idx], x2[order[1:]])
    yy2 = np.minimum(y2[idx], y2[order[1:]])

    max_width = np.maximum(0.0, xx2 - xx1 + 0.001)
    max_height = np.maximum(0.0, yy2 - yy1 + 0.001)
    inter = max_width * max_height

    return inter / (areas[idx] + areas[order[1:]] - inter)


def non_max_suppression_np(boxes: np.ndarray,
                           scores: np.ndarray,
                           conf_thres: float = 0.25,
                           iou_thres: float = 0.45,
                           nms: Callable = nms_np) -> List[np.ndarray]:
    # Settings
    maximum_detections = boxes.shape[0]

    boxes, scores, conf_index = detection_matrix(boxes, scores, conf_thres)

    # Check shape; # number of boxes
    if not boxes.shape[0]:  # no boxes
        return []

    # Batched NMS
    indexes = nms(boxes, scores, maximum_detections, iou_thres)

    new_indexes = []
    for i in indexes:
        new_indexes.append(conf_index[i])

    return np.array(new_indexes)


def detection_matrix(box: np.ndarray, score: np.ndarray, conf_thres: float):

    index = score > conf_thres

    return box[index], score[index], np.where(score > conf_thres)[0]
