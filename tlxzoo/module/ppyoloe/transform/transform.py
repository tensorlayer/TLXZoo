import random

import cv2
import numpy as np
from PIL import Image


class LabelFormatConvert(object):
    def __init__(self, class_ids):
        self.class_ids = class_ids

    def __call__(self, target):
        gt_bbox = []
        gt_class = []
        for ann in target['annotations']:
            gt_bbox.append(ann['bbox'])
            gt_class.append([self.class_ids.index(ann['category_id'])])

        target = {
            'image': target['image'],
            'gt_bbox': np.array(gt_bbox, dtype=np.float32),
            'gt_class': np.array(gt_class, dtype=np.int32)
        }
        return target


class ResizeImage(object):
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
        self.max_size = int(max_size)
        self.interp = int(interp)
        self.use_cv2 = use_cv2
        self.resize_box = resize_box
        if not (isinstance(target_size, int) or isinstance(target_size, list)):
            raise TypeError("Type of target_size is invalid. Must be Integer or List, now is {}".
                            format(type(target_size)))
        self.target_size = target_size
        if not (isinstance(self.max_size, int) and isinstance(self.interp, int)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ValueError('{}: image is not 3-dimensional.'.format(self))
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


class NormalizeImage(object):
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

    def __call__(self, sample):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']
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
        sample['image'] = im
        return sample


class Permute(object):
    def __init__(self, to_bgr=True, channel_first=True):
        """
        Change the channel.
        Args:
            to_bgr (bool): confirm whether to convert RGB to BGR
            channel_first (bool): confirm whether to change channel
        """
        self.to_bgr = to_bgr
        self.channel_first = channel_first
        if not (isinstance(self.to_bgr, bool) and isinstance(self.channel_first, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample):
        im = sample['image']
        if self.channel_first:
            im = np.swapaxes(im, 1, 2)
            im = np.swapaxes(im, 1, 0)
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        sample['image'] = im
        return sample


class PadGTSingle(object):
    def __init__(self, num_max_boxes=200, return_gt_mask=True):
        self.num_max_boxes = num_max_boxes
        self.return_gt_mask = return_gt_mask

    def __call__(self, sample):
        im = sample['image']
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
        sample['image'] = im
        return sample
