from ...feature.image_feature import ImageFeatureMixin
from ...feature.feature import FeaturePreTrainedMixin
from ...config.config import BaseImageFeatureConfig
import numpy as np


class VGGFeature(FeaturePreTrainedMixin, ImageFeatureMixin):
    config_class = BaseImageFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config

        resize_size = kwargs.pop("resize_size", None)
        if resize_size is not None:
            self.config.resize_size = resize_size

        mean = kwargs.pop("mean", None)
        if mean is not None:
            self.config.mean = mean

        std = kwargs.pop("std", None)
        if std is not None:
            self.config.image_std = std

        self.resize_size = self.config.resize_size
        self.mean = self.config.mean
        self.std = self.config.std

    def __call__(self, images, *args, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]

        if self.config.do_resize:
            images = [self.resize(image=image, size=self.resize_size) for image in images]

        if self.config.do_normalize:
            images = [self.normalize(image=image, mean=self.mean, std=self.std) for image in images]

        return np.array(images)



