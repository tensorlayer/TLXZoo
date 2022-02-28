from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os
from ...utils.registry import Registers


ResNet_PRETRAINED_CONFIG = {
    "resnet50": ("",
                 "")
}


@Registers.model_configs.register
class ResNetModelConfig(BaseModelConfig):
    model_type = "resnet"

    def __init__(
            self,
            num_layers=44,
            shortcut_connection=True,
            weight_decay=1e-4,
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            drop_rate=0.1,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.num_layers = num_layers
        self.shortcut_connection = shortcut_connection
        self.weight_decay = weight_decay
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.drop_rate = drop_rate

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
                 num_labels=10,
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