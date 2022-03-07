from ...feature.feature import BaseImageFeature
from ...config.config import BaseImageFeatureConfig
import numpy as np
from ...utils.registry import Registers
import cv2


@Registers.features.register
class YOLOv4Feature(BaseImageFeature):
    config_class = BaseImageFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config

        super(YOLOv4Feature, self).__init__(config, **kwargs)

    def __call__(self, images, *args, **kwargs):
        if not isinstance(images, (list, tuple)):
            images = [images]

        return np.array(images)
