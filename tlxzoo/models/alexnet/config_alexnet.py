from tkinter import N
from ...config.config import BaseModelConfig, BaseTaskConfig, BaseImageFeatureConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os


model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
}



class AlexNetModelConfig(BaseModelConfig):
    model_type = "vgg"

    def __init__(self, fc_units=4096, weights_path=MODEL_WEIGHT_NAME, **kwargs):
        self.fc_unints = 4096
        if weights_path is None:
            self.weights_path = model_urls['alexnet'])
        else:
            self.weights_path = weights_path

        super().__init__(**kwargs, )

    def _get_last_output_size(self):
        return None, self.fc_unints


class AlexNetForImageClassificationTaskConfig(BaseTaskConfig):
    task_type = "vgg_for_image_classification"
    model_config_type = AlexNetModelConfig

    def __init__(self,
                 model_config,
                 num_classes=1000,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()
        self.num_classes = num_classes
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(AlexNetForImageClassificationTaskConfig,
              self).__init__(model_config, **kwargs)
