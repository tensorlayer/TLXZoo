from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os
import numpy as np
from ...utils.registry import Registers

weights_url = {'link': 'https://pan.baidu.com/s/1MC1dmEwpxsdgHO1MZ8fYRQ', 'password': 'idsz'}


class YOLOv4ModelConfig(BaseModelConfig):
    model_type = "yolov4"

    def __init__(
            self,
            conv1_1_filters_shape=(1, 1, 512, 256),
            conv1_2_filters_shape=(1, 1, 512, 256),
            conv2_1_filters_shape=(1, 1, 512, 256),
            conv2_2_filters_shape=(3, 3, 256, 512),
            conv2_3_filters_shape=(1, 1, 512, 256),
            conv2_4_filters_shape=(3, 3, 256, 512),
            conv2_5_filters_shape=(1, 1, 512, 256),
            conv3_1_filters_shape=(1, 1, 256, 128),
            conv3_2_filters_shape=(1, 1, 256, 128),
            conv4_1_filters_shape=(1, 1, 256, 128),
            conv4_2_filters_shape=(3, 3, 128, 256),
            conv4_3_filters_shape=(1, 1, 256, 128),
            conv4_4_filters_shape=(3, 3, 128, 256),
            conv4_5_filters_shape=(1, 1, 256, 128),
            conv5_1_filters_shape=(3, 3, 128, 256),
            conv6_1_filters_shape=(3, 3, 128, 256),
            conv6_2_filters_shape=(1, 1, 512, 256),
            conv6_3_filters_shape=(3, 3, 256, 512),
            conv6_4_filters_shape=(1, 1, 512, 256),
            conv6_5_filters_shape=(3, 3, 256, 512),
            conv6_6_filters_shape=(1, 1, 512, 256),
            conv7_1_filters_shape=(3, 3, 256, 512),
            conv7_3_filters_shape=(3, 3, 256, 512),
            conv8_1_filters_shape=(1, 1, 1024, 512),
            conv8_2_filters_shape=(3, 3, 512, 1024),
            conv8_3_filters_shape=(1, 1, 1024, 512),
            conv8_4_filters_shape=(3, 3, 512, 1024),
            conv8_5_filters_shape=(1, 1, 1024, 512),
            conv9_1_filters_shape=(3, 3, 512, 1024),
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.conv1_1_filters_shape = conv1_1_filters_shape
        self.conv1_2_filters_shape = conv1_2_filters_shape
        self.conv2_1_filters_shape = conv2_1_filters_shape
        self.conv2_2_filters_shape = conv2_2_filters_shape
        self.conv2_3_filters_shape = conv2_3_filters_shape
        self.conv2_4_filters_shape = conv2_4_filters_shape
        self.conv2_5_filters_shape = conv2_5_filters_shape
        self.conv3_1_filters_shape = conv3_1_filters_shape
        self.conv3_2_filters_shape = conv3_2_filters_shape
        self.conv4_1_filters_shape = conv4_1_filters_shape
        self.conv4_2_filters_shape = conv4_2_filters_shape
        self.conv4_3_filters_shape = conv4_3_filters_shape
        self.conv4_4_filters_shape = conv4_4_filters_shape
        self.conv4_5_filters_shape = conv4_5_filters_shape
        self.conv5_1_filters_shape = conv5_1_filters_shape
        self.conv6_1_filters_shape = conv6_1_filters_shape
        self.conv6_2_filters_shape = conv6_2_filters_shape
        self.conv6_3_filters_shape = conv6_3_filters_shape
        self.conv6_4_filters_shape = conv6_4_filters_shape
        self.conv6_5_filters_shape = conv6_5_filters_shape
        self.conv6_6_filters_shape = conv6_6_filters_shape
        self.conv7_1_filters_shape = conv7_1_filters_shape
        self.conv7_3_filters_shape = conv7_3_filters_shape
        self.conv8_1_filters_shape = conv8_1_filters_shape
        self.conv8_2_filters_shape = conv8_2_filters_shape
        self.conv8_3_filters_shape = conv8_3_filters_shape
        self.conv8_4_filters_shape = conv8_4_filters_shape
        self.conv8_5_filters_shape = conv8_5_filters_shape
        self.conv9_1_filters_shape = conv9_1_filters_shape

        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class YOLOv4ForObjectDetectionTaskConfig(BaseTaskConfig):
    task_type = "vgg_for_object_detection"
    model_config_type = YOLOv4ModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 num_labels=80,
                 sconv_filters_shape=(1, 1, 256),
                 mconv_filters_shape=(1, 1, 512),
                 lconv_filters_shape=(1, 1, 1024),
                 strides=np.array([8, 16, 32]),
                 anchors=np.array(
                     [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]).reshape(3, 3, 2),
                 xyscale=(1.2, 1.1, 1.05),
                 iou_loss_thresh=0.5,
                 train_input_size=416,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        self.sconv_filters_shape = tuple(list(sconv_filters_shape) + [3 * (num_labels + 5)])
        self.mconv_filters_shape = tuple(list(mconv_filters_shape) + [3 * (num_labels + 5)])
        self.lconv_filters_shape = tuple(list(lconv_filters_shape) + [3 * (num_labels + 5)])
        self.strides = strides
        self.anchors = anchors
        self.xyscale = xyscale
        self.train_input_size = train_input_size
        self.iou_loss_thresh = iou_loss_thresh
        if model_config is None:
            model_config = self.model_config_type()
        self.num_labels = num_labels
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(YOLOv4ForObjectDetectionTaskConfig, self).__init__(model_config, **kwargs)