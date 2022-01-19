import tensorlayerx as tlx
import numpy as np


class ImageFeaturePreTrainedMixin:
    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, **kwargs
    ):
        ...


class ImageFeatureMixin:
    def _image_type(self, image):
        if isinstance(image, np.ndarray):
            raise

    def resize(self, image, size):
        self._image_type(image)
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        img = tlx.prepro.imresize(image, size)[:, :, ::-1]
        return img

    def normalize(self, image, mean, std):
        self._image_type(image)
        if std:
            return (image - mean) / std
        return image - mean
