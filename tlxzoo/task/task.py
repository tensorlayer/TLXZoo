from tensorlayerx import nn, logging
from ..config.config import BaseTaskConfig
import os


class BaseTask(nn.Module):
    def __init__(self, config: BaseTaskConfig, *args, **kwargs):

        super(BaseTask, self).__init__(*args, **kwargs)
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_base_path, *task_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_path and config are both None.")
            config = cls.config_class.from_pretrained(pretrained_base_path, **kwargs)

        task = cls(config, *task_args, **kwargs)

        if pretrained_base_path is None:
            logging.warning("Don't load weight.")
            return task

        weights_path = config.weights_path
        task.load_weights(weights_path)
        return task

    def save_pretrained(self, save_directory, weights_with_model=True):
        if os.path.isfile(save_directory):
            logging.error(f"Save directory ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # save weight
        self.save_weights(save_directory)

        # save config
        self.config.save_pretrained(save_directory)


class BaseForImageClassification(BaseTask):
    task_type = "image_classification"


class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"

