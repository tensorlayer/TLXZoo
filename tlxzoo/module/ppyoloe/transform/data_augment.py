import math
import random

import cv2
import numpy as np
from loguru import logger
from numbers import Number, Integral


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


import uuid
from scipy import ndimage
from PIL import Image

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


def is_poly(segm):
    assert isinstance(segm, (list, dict)), \
        "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def crop_segms(self, segms, valid_ids, crop, height, width):
        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                import copy
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                # logging.getLogger("shapely").setLevel(logging.WARNING)
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def __call__(self, sample, context=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h = sample['h']
        w = sample['w']
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                    crop_polys = self.crop_segms(
                        sample['gt_poly'],
                        valid_ids,
                        np.array(
                            crop_box, dtype=np.int64),
                        h,
                        w)
                    if [] in crop_polys:
                        delete_id = list()
                        valid_polys = list()
                        for id, crop_poly in enumerate(crop_polys):
                            if crop_poly == []:
                                delete_id.append(id)
                            else:
                                valid_polys.append(crop_poly)
                        valid_ids = np.delete(valid_ids, delete_id)
                        if len(valid_polys) == 0:
                            return sample
                        sample['gt_poly'] = valid_polys
                    else:
                        sample['gt_poly'] = crop_polys
                sample['image'] = self._crop_image(sample['image'], crop_box)
                # 掩码也被删去与裁剪
                if 'gt_segm' in sample.keys() and sample['gt_segm'] is not None:
                    gt_segm = sample['gt_segm']
                    gt_segm = gt_segm.transpose(1, 2, 0)
                    gt_segm = np.take(gt_segm, valid_ids, axis=-1)
                    gt_segm = self._crop_image(gt_segm, crop_box)
                    gt_segm = gt_segm.transpose(2, 0, 1)
                    sample['gt_segm'] = gt_segm
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                sample['w'] = crop_box[2] - crop_box[0]
                sample['h'] = crop_box[3] - crop_box[1]
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]


class ColorDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings.
            in [lower, upper, probability] format.
        saturation (list): saturation settings.
            in [lower, upper, probability] format.
        contrast (list): contrast settings.
            in [lower, upper, probability] format.
        brightness (list): brightness settings.
            in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        hsv_format (bool): whether to convert color from BGR to HSV
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 hsv_format=False,
                 random_channel=False):
        super(ColorDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.hsv_format = hsv_format
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 0] += random.uniform(low, high)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
            return img

        # XXX works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        if self.hsv_format:
            img[..., 1] *= delta
            return img
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)

        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness,
                self.apply_contrast,
                self.apply_saturation,
                self.apply_hue,
            ]
            distortions = np.random.permutation(functions)
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)

        if np.random.randint(0, 2):
            img = self.apply_contrast(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = self.apply_saturation(img)
            img = self.apply_hue(img)
            if self.hsv_format:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


from numbers import Number


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
        is_mask_expand(bool): whether expand the segmentation.
    """

    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=(127.5,) * 3,
                 is_mask_expand=False):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value
        self.is_mask_expand = is_mask_expand

    def expand_segms(self, segms, x, y, height, width, ratio):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, ratio):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((int(height * ratio), int(width * ratio)),
                                    0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, ratio))
        return expanded_segms

    def __call__(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        height = int(sample['h'])
        width = int(sample['w'])

        expand_ratio = np.random.uniform(1., self.ratio)
        h = int(height * expand_ratio)
        w = int(width * expand_ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas *= np.array(self.fill_value, dtype=np.uint8)
        canvas[y:y + height, x:x + width, :] = img.astype(np.uint8)

        sample['h'] = h
        sample['w'] = w
        sample['image'] = canvas
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] += np.array([x, y] * 2, dtype=np.float32)
        if self.is_mask_expand and 'gt_poly' in sample and len(sample[
                                                                   'gt_poly']) > 0:
            sample['gt_poly'] = self.expand_segms(sample['gt_poly'], x, y,
                                                  height, width, expand_ratio)
        return sample


class RandomFlipImage(BaseOperator):
    def __init__(self, prob=0.5, is_normalized=False, is_mask_flip=False):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomFlipImage, self).__init__()
        self.prob = prob
        self.is_normalized = is_normalized
        self.is_mask_flip = is_mask_flip
        if not (isinstance(self.prob, float) and
                isinstance(self.is_normalized, bool) and
                isinstance(self.is_mask_flip, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def flip_segms(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def flip_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                if self.is_normalized:
                    gt_keypoint[:, i] = 1 - old_x
                else:
                    gt_keypoint[:, i] = width - old_x - 1
        return gt_keypoint

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            gt_bbox = sample['gt_bbox']
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                if gt_bbox.shape[0] == 0:
                    return sample
                oldx1 = gt_bbox[:, 0].copy()
                oldx2 = gt_bbox[:, 2].copy()
                if self.is_normalized:
                    gt_bbox[:, 0] = 1 - oldx2
                    gt_bbox[:, 2] = 1 - oldx1
                else:
                    gt_bbox[:, 0] = width - oldx2 - 1
                    gt_bbox[:, 2] = width - oldx1 - 1
                if gt_bbox.shape[0] != 0 and (
                        gt_bbox[:, 2] < gt_bbox[:, 0]).all():
                    m = "{}: invalid box, x2 should be greater than x1".format(
                        self)
                    raise BboxError(m)
                sample['gt_bbox'] = gt_bbox
                if self.is_mask_flip and len(sample['gt_poly']) != 0:
                    sample['gt_poly'] = self.flip_segms(sample['gt_poly'],
                                                        height, width)
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = self.flip_keypoint(
                        sample['gt_keypoint'], width)

                if 'semantic' in sample.keys() and sample[
                    'semantic'] is not None:
                    sample['semantic'] = sample['semantic'][:, ::-1]

                if 'gt_segm' in sample.keys() and sample['gt_segm'] is not None:
                    sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample


class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[1, 1, 1],
                 is_scale=True,
                 is_channel_first=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im, sample


class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
            resize_box (bool): whether resize ground truth bbox annotations.
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,
                                                              int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(
                    list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['image'] = im
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(sample[
                                                               'gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, use_padded_im_info=True):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.use_padded_im_info = use_padded_im_info

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride
        if coarsest_stride == 0:
            return samples
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)

        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if self.use_padded_im_info:
                data['im_info'][:2] = max_shape[1:3]
            if 'semantic' in data.keys() and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data.keys() and data['gt_segm'] is not None and len(data['gt_segm']) > 0:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

        return samples


class Permute(BaseOperator):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and
                isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, im, sample, context=None):
        if self.channel_first:
            im = np.swapaxes(im, 1, 2)
            im = np.swapaxes(im, 1, 0)
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im, sample


class RandomShape(BaseOperator):
    """
    Randomly reshape a batch. If random_inter is True, also randomly
    select one an interpolation algorithm [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
    cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]. If random_inter is
    False, use cv2.INTER_NEAREST.
    Args:
        sizes (list): list of int, random choose a size from these
        random_inter (bool): whether to randomly interpolation, defalut true.
    """

    def __init__(self, sizes=[], random_inter=False, resize_box=False):
        super(RandomShape, self).__init__()
        self.sizes = sizes
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, samples, context=None):
        shape = np.random.choice(self.sizes)
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        for i in range(len(samples)):
            im = samples[i]['image']
            h, w = im.shape[:2]
            scale_x = float(shape) / w
            scale_y = float(shape) / h
            im = cv2.resize(
                im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
            samples[i]['image'] = im
            if self.resize_box and 'gt_bbox' in samples[i] and len(samples[0][
                                                                       'gt_bbox']) > 0:
                scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
                samples[i]['gt_bbox'] = np.clip(samples[i]['gt_bbox'] *
                                                scale_array, 0,
                                                float(shape) - 1)
        return samples


class RandomShapeSingle(BaseOperator):
    """
    一张图片的RandomShape
    """

    def __init__(self, random_inter=False, resize_box=False):
        super(RandomShapeSingle, self).__init__()
        self.random_inter = random_inter
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ] if random_inter else []
        self.resize_box = resize_box

    def __call__(self, shape, sample, context=None):
        method = np.random.choice(self.interps) if self.random_inter \
            else cv2.INTER_NEAREST
        im = sample['image']
        h, w = im.shape[:2]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        im = cv2.resize(
            im, None, None, fx=scale_x, fy=scale_y, interpolation=method)
        sample['image'] = im
        if self.resize_box and 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale_x, scale_y] * 2, dtype=np.float32)
            # 注意，旧版本的ppdet中float(shape)需要-1，但是PPYOLOE（新版本的ppdet）中不需要-1
            # sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, float(shape) - 1)
            sample['gt_bbox'] = np.clip(sample['gt_bbox'] * scale_array, 0, float(shape))
        return sample


class PadBox(BaseOperator):
    def __init__(self, num_max_boxes=50, init_bbox=None):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        self.init_bbox = init_bbox
        super(PadBox, self).__init__()

    def __call__(self, sample, context=None):
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if self.init_bbox is not None:
            pad_bbox = np.ones((num_max, 4), dtype=np.float32) * self.init_bbox
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in fields:
            pad_class = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in fields:
            pad_score = np.zeros((num_max), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'is_difficult' in fields:
            pad_diff = np.zeros((num_max), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        return sample


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
            sample_bbox[2] <= object_bbox[0] or \
            sample_bbox[1] >= object_bbox[3] or \
            sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
            intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
            sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target
        return samples


class Gt2YoloTargetSingle(BaseOperator):
    """
    一张图片的Gt2YoloTarget
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTargetSingle, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, sample, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = sample['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        gt_score = sample['gt_score']
        for i, (
                mask, downsample_ratio
        ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            grid_h = int(h / downsample_ratio)
            grid_w = int(w / downsample_ratio)
            target = np.zeros(
                (len(mask), 6 + self.num_classes, grid_h, grid_w),
                dtype=np.float32)
            for b in range(gt_bbox.shape[0]):
                gx, gy, gw, gh = gt_bbox[b, :]
                cls = gt_class[b]
                score = gt_score[b]
                if gw <= 0. or gh <= 0. or score <= 0.:
                    continue

                # find best match anchor index
                best_iou = 0.
                best_idx = -1
                for an_idx in range(an_hw.shape[0]):
                    iou = jaccard_overlap(
                        [0., 0., gw, gh],
                        [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = an_idx

                gi = int(gx * grid_w)
                gj = int(gy * grid_h)

                # gtbox should be regresed in this layes if best match
                # anchor index in anchor mask of this layer
                if best_idx in mask:
                    best_n = mask.index(best_idx)

                    # x, y, w, h, scale
                    target[best_n, 0, gj, gi] = gx * grid_w - gi
                    target[best_n, 1, gj, gi] = gy * grid_h - gj
                    target[best_n, 2, gj, gi] = np.log(
                        gw * w / self.anchors[best_idx][0])
                    target[best_n, 3, gj, gi] = np.log(
                        gh * h / self.anchors[best_idx][1])
                    target[best_n, 4, gj, gi] = 2.0 - gw * gh

                    # objectness record gt_score
                    target[best_n, 5, gj, gi] = score

                    # classification
                    target[best_n, 6 + cls, gj, gi] = 1.

                # For non-matched anchors, calculate the target if the iou
                # between anchor and gt is larger than iou_thresh
                if self.iou_thresh < 1:
                    for idx, mask_i in enumerate(mask):
                        if mask_i == best_idx: continue
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                        if iou > self.iou_thresh:
                            # x, y, w, h, scale
                            target[idx, 0, gj, gi] = gx * grid_w - gi
                            target[idx, 1, gj, gi] = gy * grid_h - gj
                            target[idx, 2, gj, gi] = np.log(
                                gw * w / self.anchors[mask_i][0])
                            target[idx, 3, gj, gi] = np.log(
                                gh * h / self.anchors[mask_i][1])
                            target[idx, 4, gj, gi] = 2.0 - gw * gh

                            # objectness record gt_score
                            target[idx, 5, gj, gi] = score

                            # classification
                            target[idx, 6 + cls, gj, gi] = 1.
            sample['target{}'.format(i)] = target
        return sample


class PadGTSingle(BaseOperator):
    def __init__(self, num_max_boxes=200, return_gt_mask=True):
        super(PadGTSingle, self).__init__()
        self.num_max_boxes = num_max_boxes
        self.return_gt_mask = return_gt_mask

    def __call__(self, im, sample, context=None):
        num_max_boxes = self.num_max_boxes
        if self.return_gt_mask:
            sample['pad_gt_mask'] = np.zeros(
                (num_max_boxes, 1), dtype=np.float32)
        if num_max_boxes != 0:
            num_gt = len(sample['gt_bbox'])
            num_gt = min(num_gt, num_max_boxes)
            pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.int32)
            pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
            if num_gt > 0:
                pad_gt_class[:num_gt] = sample['gt_class'][:num_gt]
                pad_gt_bbox[:num_gt] = sample['gt_bbox'][:num_gt]
            sample['gt_class'] = pad_gt_class
            sample['gt_bbox'] = pad_gt_bbox
            # pad_gt_mask
            if 'pad_gt_mask' in sample:
                sample['pad_gt_mask'][:num_gt] = 1
            # gt_score
            if 'gt_score' in sample:
                pad_gt_score = np.zeros((num_max_boxes, 1), dtype=np.float32)
                if num_gt > 0:
                    pad_gt_score[:num_gt] = sample['gt_score'][:num_gt]
                sample['gt_score'] = pad_gt_score
            if 'is_crowd' in sample:
                pad_is_crowd = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_is_crowd[:num_gt] = sample['is_crowd'][:num_gt]
                sample['is_crowd'] = pad_is_crowd
            if 'difficult' in sample:
                pad_diff = np.zeros((num_max_boxes, 1), dtype=np.int32)
                if num_gt > 0:
                    pad_diff[:num_gt] = sample['difficult'][:num_gt]
                sample['difficult'] = pad_diff
            
        return im, sample


class ResizeImage(BaseOperator):
    def __init__(self,
                 target_size=0,
                 max_size=0,
                 interp=cv2.INTER_LINEAR,
                 use_cv2=True,
                 resize_box=False):
        """
        Rescale image to the specified target size, and capped at max_size
        if max_size != 0.
        If target_size is list, selected a scale randomly as the specified
        target size.
        Args:
            target_size (int|list): the target size of image's short side,
                multi-scale training is adopted when type is list.
            max_size (int): the max size of image
            interp (int): the interpolation method
            use_cv2 (bool): use the cv2 interpolation method or use PIL
                interpolation method
            resize_box (bool): whether resize ground truth bbox annotations.
        """
        super(ResizeImage, self).__init__()
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp,
                                                              int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, im, sample, context=None):
        """ Resize the image numpy.
        """
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(self.target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(self.target_size)
        else:
            selected_size = self.target_size
        if float(im_size_min) == 0:
            raise ZeroDivisionError('{}: min size of image is 0'.format(self))
        if self.max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            if 'im_info' in sample and sample['im_info'][2] != 1.:
                sample['im_info'] = np.append(
                    list(sample['im_info']), im_info).astype(np.float32)
            else:
                sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)
        sample['scale_factor'] = [im_scale_x, im_scale_y] * 2
        if 'gt_bbox' in sample and self.resize_box and len(sample[
                                                               'gt_bbox']) > 0:
            bboxes = sample['gt_bbox'] * sample['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, resize_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, resize_h - 1)
            sample['gt_bbox'] = bboxes
        if 'semantic' in sample.keys() and sample['semantic'] is not None:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return im, sample
    

class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end
        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            im_info = sample['im_info']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            gt_score = sample['gt_score']
            no_gt = False
            if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
                no_gt = True
                bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
                gt_class = np.array([[0]]).astype(np.int32)
                gt_score = np.array([[1]]).astype(np.float32)
                # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
            # bboxes的横坐标变成缩放后图片中对应物体的横坐标
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                                np.floor(im_info[1] / im_info[2])
            # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                                np.floor(im_info[0] / im_info[2])
            # calculate the locations
            h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
            points, num_points_each_level = self._compute_points(w,
                                                                 h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
                object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                    bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

            l_res = xs - bboxes[:,
                         0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
            r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
            t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
            b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
            if self.center_sampling_radius > 0:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
                # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
                # (1)第1个正负样本判断依据
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
            lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])

            # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
            # (2)第2个正负样本判断依据
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            # [所有格子数, gt数]   所有gt的面积
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[
                is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2gtarea[
                is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
            points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
            points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
            labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
            labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
            reg_targets = reg_targets[
                range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                   reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(
                np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
            ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
            pos_ind = np.nonzero(
                labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
            reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
            split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
                labels[:, :] = 0
            labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
            reg_targets_by_level = np.split(reg_targets, split_sections,
                                            axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
            ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                            axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

            # 最后一步是reshape，和格子的位置对应上。
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
                if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                            newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return samples


class Gt2FCOSTargetSingle(BaseOperator):
    """
    一张图片的Gt2FCOSTarget
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTargetSingle, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()

            '''
            location.shape = [grid_h*grid_w, 2]
            如果stride=8，
            location = [[4, 4], [12, 4], [20, 4], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            如果stride=16，
            location = [[8, 8], [24, 8], [40, 8], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            ...
            '''
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标。可以看出，每1个gt都会参与到fpn的所有输出特征图。
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0

        # clipped_box即修改之后的gt，和原始gt（bboxes）的中心点相同，但是边长却修改成最大只能是1.5 * 2 = 3个格子边长
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end

        # 如果格子中心点落在clipped_box代表的gt框内，那么这个格子就被选为候选正样本。

        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, sample, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']  # [3, pad_h, pad_w]
        im_info = sample['im_info']  # [3, ]  分别是resize_h, resize_w, im_scale
        bboxes = sample['gt_bbox']  # [m, 4]  x0y0x1y1格式
        gt_class = sample['gt_class']  # [m, 1]
        gt_score = sample['gt_score']  # [m, 1]
        no_gt = False
        if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
            no_gt = True
            bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
            gt_class = np.array([[0]]).astype(np.int32)
            gt_score = np.array([[1]]).astype(np.float32)
            # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
        # bboxes的横坐标变成缩放后图片中对应物体的横坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                            np.floor(im_info[1] / im_info[2])
        # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                            np.floor(im_info[0] / im_info[2])
        # calculate the locations
        h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
        points, num_points_each_level = self._compute_points(w,
                                                             h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        object_scale_exp = []
        for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
            object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                np.tile(
                    np.array([self.object_sizes_of_interest[i]]),
                    reps=[num_pts, 1]))
        object_scale_exp = np.concatenate(object_scale_exp, axis=0)

        gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                bboxes[:, 3] - bboxes[:, 1])
        xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
        xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
        xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
        ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

        l_res = xs - bboxes[:,
                     0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
        r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
        t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
        b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
        reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
        if self.center_sampling_radius > 0:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的中心区域判断格子中心点是否在gt框内。这样做会减少很多中心度很低的低质量正样本。
            is_inside_box = self._check_inside_boxes_limited(
                bboxes, xs, ys, num_points_each_level)
        else:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的完整区域判断格子中心点是否在gt框内。这样做会增加很多中心度很低的低质量正样本。
            is_inside_box = np.min(reg_targets, axis=2) > 0
        # check if the targets is inside the corresponding level
        max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
        lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 0], axis=1),
            reps=[1, max_reg_targets.shape[1]])
        high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 1], axis=1),
            reps=[1, max_reg_targets.shape[1]])

        # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
        # (2)第2个正负样本判断依据
        is_match_current_level = \
            (max_reg_targets > lower_bound) & \
            (max_reg_targets < high_bound)
        # [所有格子数, gt数]   所有gt的面积
        points2gtarea = np.tile(
            np.expand_dims(
                gt_area, axis=0), reps=[xs.shape[0], 1])
        points2gtarea[
            is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2gtarea[
            is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
        points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
        labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
        labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
        reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
        ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                               reg_targets[:, [0, 2]].max(axis=1)) * \
                              (reg_targets[:, [1, 3]].min(axis=1) / \
                               reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
        ctn_targets = np.reshape(
            ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
        ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
        pos_ind = np.nonzero(
            labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
        reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
        split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
        beg = 0
        for lvl in range(len(num_points_each_level)):
            end = beg + num_points_each_level[lvl]
            split_sections.append(end)
            beg = end
        if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
            labels[:, :] = 0
        labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        reg_targets_by_level = np.split(reg_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

        # 最后一步是reshape，和格子的位置对应上。
        for lvl in range(len(self.downsample_ratios)):
            grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
            grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
            if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                sample['reg_target{}'.format(lvl)] = \
                    np.reshape(
                        reg_targets_by_level[lvl] / \
                        self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                        newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
            else:
                sample['reg_target{}'.format(lvl)] = np.reshape(
                    reg_targets_by_level[lvl],
                    newshape=[grid_h, grid_w, 4])
            sample['labels{}'.format(lvl)] = np.reshape(
                labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
            sample['centerness{}'.format(lvl)] = np.reshape(
                ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return sample


class Gt2Solov2Target(BaseOperator):
    """Assign mask target and labels in SOLOv2 network.
    The code of this function is based on:
        https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L271
    Args:
        num_grids (list): The list of feature map grids size.
        scale_ranges (list): The list of mask boundary range.
        coord_sigma (float): The coefficient of coordinate area length.
        sampling_ratio (float): The ratio of down sampling.
    """

    def __init__(self,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768],
                               [384, 2048]],
                 coord_sigma=0.2,
                 sampling_ratio=4.0):
        super(Gt2Solov2Target, self).__init__()
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio

    def _scale_size(self, im, scale):
        h, w = im.shape[:2]
        new_size = (int(w * float(scale) + 0.5), int(h * float(scale) + 0.5))
        resized_img = cv2.resize(
            im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return resized_img

    def __call__(self, samples, context=None):
        sample_id = 0
        max_ins_num = [0] * len(self.num_grids)
        for sample in samples:
            gt_bboxes_raw = sample['gt_bbox']
            gt_labels_raw = sample['gt_class'] + 1   # 类别id+1
            im_c, im_h, im_w = sample['image'].shape[:]
            gt_masks_raw = sample['gt_segm'].astype(np.uint8)
            mask_feat_size = [
                int(im_h / self.sampling_ratio), int(im_w / self.sampling_ratio)
            ]
            gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                               (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))  # gt的平均边长
            ins_ind_label_list = []
            idx = 0
            for (lower_bound, upper_bound), num_grid \
                    in zip(self.scale_ranges, self.num_grids):
                # gt的平均边长位于指定范围内，这个感受野的特征图负责预测这些满足条件的gt
                hit_indices = ((gt_areas >= lower_bound) &
                               (gt_areas <= upper_bound)).nonzero()[0]
                num_ins = len(hit_indices)

                ins_label = []
                grid_order = []
                cate_label = np.zeros([num_grid, num_grid], dtype=np.int64)
                ins_ind_label = np.zeros([num_grid**2], dtype=np.bool)

                if num_ins == 0:
                    ins_label = np.zeros([1, mask_feat_size[0], mask_feat_size[1]], dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray([sample_id * num_grid * num_grid + 0], dtype=np.int32)
                    idx += 1
                    continue
                gt_bboxes = gt_bboxes_raw[hit_indices]   # [M, 4] 这个感受野的gt
                gt_labels = gt_labels_raw[hit_indices]   # [M, 1] 这个感受野的类别id(+1)
                gt_masks = gt_masks_raw[hit_indices, ...]   # [M, h, w] 这个感受野的gt_mask

                # 这个感受野的gt的宽的一半 * self.coord_sigma
                half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.coord_sigma
                # 这个感受野的gt的高的一半 * self.coord_sigma
                half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.coord_sigma

                # 遍历这个感受野的每一个gt
                for seg_mask, gt_label, half_h, half_w in zip(
                        gt_masks, gt_labels, half_hs, half_ws):
                    if seg_mask.sum() == 0:
                        continue
                    # mass center
                    upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                    center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                    coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                    coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                    # left, top, right, down
                    top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                    down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                    left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                    right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(coord_w - 1, left_box)
                    right = min(right_box, coord_w + 1)

                    cate_label[top:(down + 1), left:(right + 1)] = gt_label
                    seg_mask = self._scale_size(
                        seg_mask, scale=1. / self.sampling_ratio)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            label = int(i * num_grid + j)
                            cur_ins_label = np.zeros(
                                [mask_feat_size[0], mask_feat_size[1]],
                                dtype=np.uint8)
                            cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[
                                1]] = seg_mask
                            ins_label.append(cur_ins_label)
                            ins_ind_label[label] = True
                            grid_order.append(sample_id * num_grid * num_grid +
                                              label)
                if ins_label == []:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                else:
                    ins_label = np.stack(ins_label, axis=0)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        grid_order, dtype=np.int32)
                    assert len(grid_order) > 0
                max_ins_num[idx] = max(
                    max_ins_num[idx],
                    sample['ins_label{}'.format(idx)].shape[0])
                idx += 1
            ins_ind_labels = np.concatenate([
                ins_ind_labels_level_img
                for ins_ind_labels_level_img in ins_ind_label_list
            ])
            fg_num = np.sum(ins_ind_labels)
            sample['fg_num'] = fg_num
            sample_id += 1

            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
            sample.pop('gt_poly')
            sample.pop('gt_segm')

        # padding batch
        for data in samples:
            for idx in range(len(self.num_grids)):
                gt_ins_data = np.zeros(
                    [
                        max_ins_num[idx],
                        data['ins_label{}'.format(idx)].shape[1],
                        data['ins_label{}'.format(idx)].shape[2]
                    ],
                    dtype=np.uint8)
                gt_ins_data[0:data['ins_label{}'.format(idx)].shape[
                    0], :, :] = data['ins_label{}'.format(idx)]
                gt_grid_order = np.zeros([max_ins_num[idx]], dtype=np.int32)
                gt_grid_order[0:data['grid_order{}'.format(idx)].shape[
                    0]] = data['grid_order{}'.format(idx)]
                data['ins_label{}'.format(idx)] = gt_ins_data
                data['grid_order{}'.format(idx)] = gt_grid_order

        return samples



class Gt2RepPointsTargetSingle(BaseOperator):
    """
    一张图片的Gt2RepPointsTarget
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2RepPointsTargetSingle, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        # 从小感受野stride=8遍历到大感受野stride=128。location.shape=[格子行数*格子列数, 2]，存放的是每个格子的中心点的坐标。格子顺序是第一行从左到右，第二行从左到右，...
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()

            '''
            location.shape = [grid_h*grid_w, 2]
            如果stride=8，
            location = [[4, 4], [12, 4], [20, 4], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            如果stride=16，
            location = [[8, 8], [24, 8], [40, 8], ...],  这一个输出层的格子的中心点的xy坐标。格子顺序是第一行从左到右，第二行从左到右，...
            即location = [[0.5*stride, 0.5*stride], [1.5*stride, 0.5*stride], [2.5*stride, 0.5*stride], ...]

            ...
            '''
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in
                                 locations]  # num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(  # [gt数, 4] -> [1, gt数, 4]
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])  # [所有格子数, gt数, 4]   gt坐标。可以看出，每1个gt都会参与到fpn的所有输出特征图。
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2  # [所有格子数, gt数]      gt中心点x
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2  # [所有格子数, gt数]      gt中心点y
        beg = 0  # 开始=0

        # clipped_box即修改之后的gt，和原始gt（bboxes）的中心点相同，但是边长却修改成最大只能是1.5 * 2 = 3个格子边长
        clipped_box = bboxes.copy()  # [所有格子数, gt数, 4]   gt坐标，限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
        for lvl, stride in enumerate(self.downsample_ratios):  # 遍历每个感受野，从 stride=8的感受野 到 stride=128的感受野
            end = beg + num_points_each_level[lvl]  # 结束=开始+这个感受野的格子数
            stride_exp = self.center_sampling_radius * stride  # stride_exp = 1.5 * 这个感受野的stride(的格子边长)
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)  # 限制gt的边长，最大只能是1.5 * 2 = 3个格子边长
            beg = end

        # 如果格子中心点落在clipped_box代表的gt框内，那么这个格子就被选为候选正样本。

        # xs  [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        l_res = xs - clipped_box[:, :, 0]  # [所有格子数, gt数]  所有格子需要学习 gt数 个l
        r_res = clipped_box[:, :, 2] - xs  # [所有格子数, gt数]  所有格子需要学习 gt数 个r
        t_res = ys - clipped_box[:, :, 1]  # [所有格子数, gt数]  所有格子需要学习 gt数 个t
        b_res = clipped_box[:, :, 3] - ys  # [所有格子数, gt数]  所有格子需要学习 gt数 个b
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]  所有格子需要学习 gt数 个lrtb
        inside_gt_box = np.min(clipped_box_reg_targets,
                               axis=2) > 0  # [所有格子数, gt数]  需要学习的lrtb如果都>0，表示格子被选中。即只选取中心点落在gt内的格子。
        return inside_gt_box

    def __call__(self, sample, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        # im, gt_bbox, gt_class, gt_score = sample
        im = sample['image']  # [3, pad_h, pad_w]
        im_info = sample['im_info']  # [3, ]  分别是resize_h, resize_w, im_scale
        bboxes = sample['gt_bbox']  # [m, 4]  x0y0x1y1格式
        gt_class = sample['gt_class']  # [m, 1]
        gt_score = sample['gt_score']  # [m, 1]
        no_gt = False
        if len(bboxes) == 0:  # 如果没有gt，虚构一个gt为了后面不报错。
            no_gt = True
            bboxes = np.array([[0, 0, 100, 100]]).astype(np.float32)
            gt_class = np.array([[0]]).astype(np.int32)
            gt_score = np.array([[1]]).astype(np.float32)
            # print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnone')
        # bboxes的横坐标变成缩放后图片中对应物体的横坐标
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * np.floor(im_info[1]) / \
                            np.floor(im_info[1] / im_info[2])
        # bboxes的纵坐标变成缩放后图片中对应物体的纵坐标
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * np.floor(im_info[0]) / \
                            np.floor(im_info[0] / im_info[2])
        # calculate the locations
        h, w = sample['image'].shape[1:3]  # h w是这一批所有图片对齐后的高宽。
        points, num_points_each_level = self._compute_points(w,
                                                             h)  # points是所有格子中心点的坐标，num_points_each_level=[stride=8感受野格子数, ..., stride=128感受野格子数]
        object_scale_exp = []
        for i, num_pts in enumerate(num_points_each_level):  # 遍历每个感受野格子数
            object_scale_exp.append(  # 边界self.object_sizes_of_interest[i] 重复 num_pts=格子数 次
                np.tile(
                    np.array([self.object_sizes_of_interest[i]]),
                    reps=[num_pts, 1]))
        object_scale_exp = np.concatenate(object_scale_exp, axis=0)

        gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (  # [gt数, ]   所有gt的面积
                bboxes[:, 3] - bboxes[:, 1])
        xs, ys = points[:, 0], points[:, 1]  # 所有格子中心点的横坐标、纵坐标
        xs = np.reshape(xs, newshape=[xs.shape[0], 1])  # [所有格子数, 1]
        xs = np.tile(xs, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的横坐标重复 gt数 次
        ys = np.reshape(ys, newshape=[ys.shape[0], 1])  # [所有格子数, 1]
        ys = np.tile(ys, reps=[1, bboxes.shape[0]])  # [所有格子数, gt数]， 所有格子中心点的纵坐标重复 gt数 次

        l_res = xs - bboxes[:,
                     0]  # [所有格子数, gt数] - [gt数, ] = [所有格子数, gt数]     结果是所有格子中心点的横坐标 分别减去 所有gt左上角的横坐标，即所有格子需要学习 gt数 个l
        r_res = bboxes[:, 2] - xs  # 所有格子需要学习 gt数 个r
        t_res = ys - bboxes[:, 1]  # 所有格子需要学习 gt数 个t
        b_res = bboxes[:, 3] - ys  # 所有格子需要学习 gt数 个b
        reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)  # [所有格子数, gt数, 4]   所有格子需要学习 gt数 个lrtb
        if self.center_sampling_radius > 0:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内（gt是被限制边长后的gt）。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的中心区域判断格子中心点是否在gt框内。这样做会减少很多中心度很低的低质量正样本。
            is_inside_box = self._check_inside_boxes_limited(
                bboxes, xs, ys, num_points_each_level)
        else:
            # [所有格子数, gt数]    True表示格子中心点（锚点）落在gt内。
            # FCOS首先将gt框内的锚点（格子中心点）视为候选正样本，然后根据为每个金字塔等级定义的比例范围从候选中选择最终的正样本（而且是负责预测gt里面积最小的），最后那些未选择的锚点为负样本。
            # (1)第1个正负样本判断依据

            # 这里是使用gt的完整区域判断格子中心点是否在gt框内。这样做会增加很多中心度很低的低质量正样本。
            is_inside_box = np.min(reg_targets, axis=2) > 0
        # check if the targets is inside the corresponding level
        max_reg_targets = np.max(reg_targets, axis=2)  # [所有格子数, gt数]   所有格子需要学习 gt数 个lrtb   中的最大值
        lower_bound = np.tile(  # [所有格子数, gt数]   下限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 0], axis=1),
            reps=[1, max_reg_targets.shape[1]])
        high_bound = np.tile(  # [所有格子数, gt数]   上限重复 gt数 次
            np.expand_dims(
                object_scale_exp[:, 1], axis=1),
            reps=[1, max_reg_targets.shape[1]])

        # [所有格子数, gt数]   最大回归值如果位于区间内，就为True
        # (2)第2个正负样本判断依据
        is_match_current_level = \
            (max_reg_targets > lower_bound) & \
            (max_reg_targets < high_bound)
        # [所有格子数, gt数]   所有gt的面积
        points2gtarea = np.tile(
            np.expand_dims(
                gt_area, axis=0), reps=[xs.shape[0], 1])
        points2gtarea[
            is_inside_box == 0] = self.INF  # 格子中心点落在gt外的（即负样本），需要学习的面积置为无穷。     这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2gtarea[
            is_match_current_level == 0] = self.INF  # 最大回归值如果位于区间外（即负样本），需要学习的面积置为无穷。 这是为了points2gtarea.min(axis=1)时，若某格子有最终正样本，那么就应该不让负样本的面积影响到判断。
        points2min_area = points2gtarea.min(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值
        points2min_area_ind = points2gtarea.argmin(axis=1)  # [所有格子数, ]   所有格子需要学习 gt数 个面积  中的最小值的下标
        labels = gt_class[points2min_area_ind] + 1  # [所有格子数, 1]   所有格子需要学习 的类别id，学习的是gt中面积最小值的的类别id
        labels[points2min_area == self.INF] = 0  # [所有格子数, 1]   负样本的points2min_area肯定是self.INF，这里将负样本需要学习 的类别id 置为0
        reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]  # [所有格子数, 4]   所有格子需要学习 的 lrtb（负责预测gt里面积最小的）
        ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                               reg_targets[:, [0, 2]].max(axis=1)) * \
                              (reg_targets[:, [1, 3]].min(axis=1) / \
                               reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)  # [所有格子数, ]  所有格子需要学习的centerness
        ctn_targets = np.reshape(
            ctn_targets, newshape=[ctn_targets.shape[0], 1])  # [所有格子数, 1]  所有格子需要学习的centerness
        ctn_targets[labels <= 0] = 0  # 负样本需要学习的centerness置为0
        pos_ind = np.nonzero(
            labels != 0)  # tuple=( ndarray(shape=[正样本数, ]), ndarray(shape=[正样本数, ]) )   即正样本在labels中的下标，因为labels是2维的，所以一个正样本有2个下标。
        reg_targets_pos = reg_targets[pos_ind[0], :]  # [正样本数, 4]   正样本格子需要学习 的 lrtb
        split_sections = []  # 每一个感受野 最后一个格子 在reg_targets中的位置（第一维的位置）
        beg = 0
        for lvl in range(len(num_points_each_level)):
            end = beg + num_points_each_level[lvl]
            split_sections.append(end)
            beg = end
        if no_gt:  # 如果没有gt，labels里全部置为0（背景的类别id是0）即表示所有格子都是负样本
            labels[:, :] = 0
        labels_by_level = np.split(labels, split_sections, axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        reg_targets_by_level = np.split(reg_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。
        ctn_targets_by_level = np.split(ctn_targets, split_sections,
                                        axis=0)  # 一个list，根据split_sections切分，各个感受野的target切分开来。

        # 最后一步是reshape，和格子的位置对应上。
        for lvl in range(len(self.downsample_ratios)):
            grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))  # 格子列数
            grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))  # 格子行数
            if self.norm_reg_targets:  # 是否将reg目标归一化，配置里是True
                sample['reg_target{}'.format(lvl)] = \
                    np.reshape(
                        reg_targets_by_level[lvl] / \
                        self.downsample_ratios[lvl],  # 归一化方式是除以格子边长（即下采样倍率）
                        newshape=[grid_h, grid_w, 4])  # reshape成[grid_h, grid_w, 4]
            else:
                sample['reg_target{}'.format(lvl)] = np.reshape(
                    reg_targets_by_level[lvl],
                    newshape=[grid_h, grid_w, 4])
            sample['labels{}'.format(lvl)] = np.reshape(
                labels_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
            sample['centerness{}'.format(lvl)] = np.reshape(
                ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])  # reshape成[grid_h, grid_w, 1]
        return sample



class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def __call__(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


class Resize(BaseOperator):
    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly).astype('float32')
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warning(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                    format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


class RandomResize(BaseOperator):
    def __init__(self,
                 target_size,
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_size=True,
                 random_interp=False):
        """
        Resize image to target size randomly. random target_size and interpolation method
        Args:
            target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
            keep_ratio (bool): whether keep_raio or not, default true
            interp (int): the interpolation method
            random_size (bool): whether random select target size of image
            random_interp (bool): whether random select interpolation method
        """
        super(RandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        assert isinstance(target_size, (
            Integral, Sequence)), "target_size must be Integer, List or Tuple"
        if random_size and not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List or Tuple, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, sample, context=None):
        """ Resize the image numpy.
        """
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp)
        return resizer(sample, context=context)


def cal_line_length(point1, point2):
    import math
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combinate[force_flag]).reshape(8)



class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [get_best_begin_point_single(e) for e in bbox]
        return bbox

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                    width)
            if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
                sample['gt_keypoint'] = self.apply_keypoint(
                    sample['gt_keypoint'], width)

            if 'semantic' in sample and sample['semantic']:
                sample['semantic'] = sample['semantic'][:, ::-1]

            if 'gt_segm' in sample and sample['gt_segm'].any():
                sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

            if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
                sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                         width)

            sample['flipped'] = True
            sample['image'] = im
        return sample


def get_sample_transforms(cfg):
    # sample_transforms
    sample_transforms = []
    for preprocess_name in cfg.sample_transforms_seq:
        if preprocess_name == 'colorDistort':
            preprocess = ColorDistort(**cfg.colorDistort)  # 颜色扰动
        elif preprocess_name == 'randomExpand':
            preprocess = RandomExpand(**cfg.randomExpand)  # 随机填充
        elif preprocess_name == 'randomCrop':
            preprocess = RandomCrop(**cfg.randomCrop)        # 随机裁剪
        elif preprocess_name == 'randomFlipImage':
            preprocess = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
        elif preprocess_name == 'normalizeImage':
            preprocess = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。
        elif preprocess_name == 'permute':
            preprocess = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
        elif preprocess_name == 'randomShape':
            resize_box = False
            if 'resize_box' in cfg.randomShape.keys():
                resize_box = cfg.randomShape['resize_box']
            preprocess = RandomShapeSingle(random_inter=cfg.randomShape['random_inter'], resize_box=resize_box)  # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
        elif preprocess_name == 'padGT':
            preprocess = PadGTSingle(**cfg.padGT)   #
        else:
            raise NotImplementedError("Transform \'{}\' is not implemented.".format(preprocess_name))
        sample_transforms.append(preprocess)
    return sample_transforms
