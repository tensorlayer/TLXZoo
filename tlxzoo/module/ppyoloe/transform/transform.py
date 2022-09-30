import numpy as np

from .data_augment import *


class PPYOLOETrainTransform:
    def __init__(self, channel_first=True):
        self.aug_list = [
            ResizeImage(
                target_size=640
            ),
            NormalizeImage(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                is_scale=True,
                is_channel_first=False
            ),
            Permute(
                to_bgr=False,
                channel_first=channel_first
            ),
            PadGTSingle(
                num_max_boxes=200,
            )
        ]

    def __call__(self, image, targets):
        for aug in self.aug_list:
            image, targets = aug(image, targets)

        return image, targets

def preproc(img, input_size):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img

class PPYOLOEValTransform:
    def __init__(self, channel_first=True, input_size=(640, 640)):
        self.channel_first = channel_first
        self.input_size = input_size

    def __call__(self, img, target):
        img = preproc(img, self.input_size)
        img = img.astype(np.float32)
        img /= 255.0
        img -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        img /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        return img, target


class PPYOLOETransform:
    class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    def __init__(self, channel_first=True):
        self.channel_first = channel_first
        self.transform = PPYOLOETrainTransform(channel_first=self.channel_first)
        self.is_train = True

    def set_eval(self):
        self.transform = PPYOLOEValTransform(channel_first=self.channel_first)
        self.is_train = False

    def set_train(self):
        self.transform = PPYOLOETrainTransform(channel_first=self.channel_first)
        self.is_train = True
        
    def __call__(self, image, target):
        if self.is_train:
            gt_bbox = []
            gt_class = []
            for ann in target['annotations']:
                gt_bbox.append(ann['bbox'])
                gt_class.append([self.class_ids.index(ann['category_id'])])
            
            target = {
                'gt_bbox': np.array(gt_bbox, dtype=np.float32),
                'gt_class': np.array(gt_class, dtype=np.int32)
            }
            image, target = self.transform(image, target)
            
            target = {
                'gt_bbox': target['gt_bbox'],
                'gt_class': target['gt_class'],
                'pad_gt_mask': target['pad_gt_mask']
            }
            return image, target
        else:
            return self.transform(image, target)