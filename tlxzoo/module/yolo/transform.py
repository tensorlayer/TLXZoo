import numpy as np
import cv2
import random


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

    bboxes1_coor = np.concatenate([
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,)

    bboxes2_coor = np.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = np.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = np.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = np.divide(inter_area, union_area)

    return iou


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate(
            [
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(
            0, int(max_bbox[0] - random.uniform(0, max_l_trans))
        )
        crop_ymin = max(
            0, int(max_bbox[1] - random.uniform(0, max_u_trans))
        )
        crop_xmax = max(
            w, int(max_bbox[2] + random.uniform(0, max_r_trans))
        )
        crop_ymax = max(
            h, int(max_bbox[3] + random.uniform(0, max_d_trans))
        )

        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes


def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate(
            [
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes


class YoloTransform(object):
    def __init__(self, train_input_size=416, anchor_per_scale=3, num_classes=91, max_bbox_per_scale=150,
                 strides=(8, 16, 32),
                 anchors=(12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401)):
        self.train_input_size = train_input_size
        self.anchor_per_scale = anchor_per_scale
        self.num_classes = num_classes
        self.max_bbox_per_scale = max_bbox_per_scale
        self.strides = np.array(strides)
        self.anchors = np.array(anchors).reshape(3, 3, 2)
        self.train_output_sizes = self.train_input_size // self.strides

        self.is_train = True

    def set_eval(self):
        self.is_train = False

    def set_train(self):
        self.is_train = True

    def train_random_transform_hook(self, image, label):
        image, bboxes = random_horizontal_flip(
            np.copy(image), np.copy(label)
        )
        image, bboxes = random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = random_translate(
            np.copy(image), np.copy(bboxes)
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes

    def test_random_transform_hook(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(label),
        )
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (
                    1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            )

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = (
                        np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(
                        np.int32
                    )

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return (label_sbbox, sbboxes), (label_mbbox, mbboxes), (label_lbbox, lbboxes)

    def __call__(self, image_path, label):
        """
        coco2017/0.1/train2017/000000388398.jpg 134,53,270,389,0 11,66,153,263,0 32,94,633,370,13
        """
        annotations = label["annotations"]
        image = cv2.imread(label["path"])

        bboxes = []
        for anno in annotations:
            # if anno["category_id"] >= self.num_classes:
            #     continue
            bndbox = {
                "xmin": anno["bbox"][0],
                "ymin": anno["bbox"][1],
                "xmax": anno["bbox"][2] + anno["bbox"][0],
                "ymax": anno["bbox"][3] + anno["bbox"][1]
            }

            bboxes.append([int(i) for i in [bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"], anno["category_id"]]])
        bboxes = np.array(bboxes)

        if self.is_train:
            image, bboxes = self.train_random_transform_hook(image, bboxes)
        else:
            image, bboxes = self.test_random_transform_hook(image, bboxes)

        bboxes = self.preprocess_true_boxes(bboxes)

        return image, bboxes
