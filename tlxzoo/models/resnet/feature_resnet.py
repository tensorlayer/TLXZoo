from ..image_feature import ImageFeaturePreTrainedMixin, ImageFeatureMixin
import numpy as np
import tensorlayerx as tlx


class FeatureResNet(ImageFeaturePreTrainedMixin, ImageFeatureMixin):
    def __init__(
            self,
            **kwargs
    ):
        self.resize_size = kwargs.pop("resize_size", None)
        self.image_mean = kwargs.pop("image_mean", None)
        self.image_std = kwargs.pop("image_std", None)

    def __call__(self, images, *args, **kwargs):
        if self.resize_size:
            images = [self.resize(image=image, size=self.resize_size) for image in images]

        if self.image_mean:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        return images


