from tensorlayerx import nn, logging
from ..config.config import BaseTaskConfig


class BaseTask(nn.Module):
    def __init__(self, config: BaseTaskConfig, *args, **kwargs):

        super(BaseTask, self).__init__(*args, **kwargs)
        self.config = config


class BaseForImageClassification(BaseTask):
    task_type = "image_classification"


class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"

