import tensorlayerx as tlx
import numpy as np


class ImageFeatureMixin:
    def _image_type(self, image):
        if isinstance(image, np.ndarray):
            raise

    def resize(self, image, size):
        # self._image_type(image)
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        img = tlx.prepro.imresize(image, size)
        return img

    def normalize(self, image, mean, std):
        # self._image_type(image)
        if isinstance(mean, list):
            mean = np.array(mean, dtype=np.float32).reshape([1, 1, 3])
        if std:
            return (image - mean) / std
        return image - mean
