from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os
from ...utils.registry import Registers


ResNet_PRETRAINED_CONFIG = {
    "resnet50": ("resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                 "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/")
}


@Registers.model_configs.register
class ResNetModelConfig(BaseModelConfig):
    model_type = "resnet"

    def __init__(
            self,
            end_with='avg_pool',
            n_classes=1000,
            conv1_n_filter=64,
            conv1_filter_size=(7, 7),
            conv1_in_channels=3,
            conv1_strides=(2, 3),
            bn_conv1_num_features=64,
            max_pool1_filter_size=(3, 3),
            max_pool1_strides=(2, 2),
            conv_block_kernel_size=3,
            identity_block_kernel_size=3,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.end_with = end_with
        self.n_classes = n_classes
        self.conv1_n_filter = conv1_n_filter
        self.conv1_filter_size = conv1_filter_size
        self.conv1_in_channels = conv1_in_channels
        self.conv1_strides = conv1_strides
        self.bn_conv1_num_features = bn_conv1_num_features
        self.max_pool1_filter_size = max_pool1_filter_size
        self.max_pool1_strides = max_pool1_strides
        self.conv_block_kernel_size = conv_block_kernel_size
        self.identity_block_kernel_size = identity_block_kernel_size

        if weights_path is None:
            weights_path = os.path.join(ResNet_PRETRAINED_CONFIG["resnet50"][1],
                                        ResNet_PRETRAINED_CONFIG["resnet50"][0])
        else:
            weights_path = weights_path

        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class ResNetForImageClassificationTaskConfig(BaseTaskConfig):
    task_type = "resnet_for_image_classification"
    model_config_type = ResNetModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 num_labels=1000,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()
        self.num_labels = num_labels
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(ResNetForImageClassificationTaskConfig, self).__init__(model_config, **kwargs)