from tensorlayerx import nn, logging
import os
from ..utils import TASK_WEIGHT_FORMAT, TASK_WEIGHT_NAME
from ..config import BaseTaskConfig
from ..utils.from_pretrained import ModuleFromPretrainedMixin
from ..utils.registry import Registers


class BaseTask(nn.Module, ModuleFromPretrainedMixin):
    config_class = BaseTaskConfig

    def __init__(self, config: BaseTaskConfig, *args, **kwargs):

        super(BaseTask, self).__init__(*args, **kwargs)
        self.config = config
        self.config.task_class = self.__class__.__name__

    @classmethod
    def config_from_pretrained(cls, pretrained_base_path, **kwargs):
        return BaseTaskConfig.from_pretrained(pretrained_base_path, **kwargs)

    def save_pretrained(self, save_directory):
        self._save_pretrained(save_directory, TASK_WEIGHT_NAME, TASK_WEIGHT_FORMAT)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        return Registers.tasks[config.task_class](config, *args, **kwargs)


@Registers.tasks.register
class BaseForImageClassification(BaseTask):
    task_type = "image_classification"

    def __call__(self, *args, return_output=False, **kwargs):
        if return_output:
            return super(BaseForImageClassification, self).__call__(*args, **kwargs)
        else:
            return super(BaseForImageClassification, self).__call__(*args, **kwargs).logits


@Registers.tasks.register
class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"

