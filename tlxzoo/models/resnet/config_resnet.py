from ...config.config import BaseModelConfig

ResNet_PRETRAINED_CONFIG = {
    "resnet50": ("resnet50_weights_tf_dim_ordering_tf_kernels.h5",
                 "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/")
}


class ResNetConfig(BaseModelConfig):
    model_type = "resnet"

    def __init__(
            self,
            end_with='fc1000',
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

        pretrained_path = ResNet_PRETRAINED_CONFIG["resnet50"]

        super().__init__(
            pretrained_path=pretrained_path,
            **kwargs,
        )
