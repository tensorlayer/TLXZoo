import tensorlayerx as tlx
import numpy as np


def resize(image, size):
    if isinstance(size, int):
        size = (size, size)
    elif isinstance(size, list):
        size = tuple(size)
    img = tlx.prepro.imresize(image, size)
    return img


def normalize(image, mean, std):
    if isinstance(mean, list):
        mean = np.array(mean, dtype=np.float32).reshape([1, 1, 3])

    if isinstance(std, list):
        std = np.array(std, dtype=np.float32).reshape([1, 1, 3])

    if mean is not None:
        image = image - mean

    if std is not None:
        image = image / std

    return image


class BaseVisionTransform(object):
    def __init__(self, do_resize=False,
                 do_normalize=False,
                 mean=None,
                 std=None,
                 resize_size=None
                 ):
        self.is_train = True
        self.RandomRotation = tlx.vision.transforms.RandomRotation(degrees=15, interpolation='bilinear', expand=False,
                                                                   center=None, fill=0)
        self.RandomShift = tlx.vision.transforms.RandomShift(shift=(0.1, 0.1), interpolation='bilinear', fill=0)
        self.RandomFlipHorizontal = tlx.vision.transforms.RandomFlipHorizontal(prob=0.5)
        self.RandomCrop = tlx.vision.transforms.RandomCrop(32, 4)

        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.resize_size = resize_size

    def set_eval(self):
        self.is_train = False

    def set_train(self):
        self.is_train = True

    def __call__(self, image, label, *args, **kwargs):

        image = image.astype('float32')

        if self.do_resize:
            image = resize(image=image, size=self.resize_size)

        if self.do_normalize:
            image = normalize(image=image, mean=self.mean, std=self.std)

        if self.is_train:
            image = self.RandomRotation(image)
            image = self.RandomShift(image)
            image = self.RandomFlipHorizontal(image)
            image = self.RandomCrop(image)

        image = image.astype('float32')

        return image, label

