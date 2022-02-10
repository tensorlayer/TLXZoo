from tensorlayerx import nn, logging
import os
from ..utils import TASK_WEIGHT_FORMAT, TASK_WEIGHT_NAME
from ..config import BaseTaskConfig
from ..utils.from_pretrained import ModuleFromPretrainedMixin


class BaseTask(nn.Module, ModuleFromPretrainedMixin):
    config_class = BaseTaskConfig

    def __init__(self, config: BaseTaskConfig, *args, **kwargs):

        super(BaseTask, self).__init__(*args, **kwargs)
        self.config = config

    def save_pretrained(self, save_directory):
        self._save_pretrained(save_directory, TASK_WEIGHT_NAME, TASK_WEIGHT_FORMAT)


class BaseForImageClassification(BaseTask):
    task_type = "image_classification"

    def __call__(self, *args, **kwargs):
        return super(BaseForImageClassification, self).__call__(*args, **kwargs).logits


class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"

