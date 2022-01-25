from tensorlayerx import nn, logging
import os
from ..utils import TASK_WEIGHT_FORMAT, TASK_WEIGHT_NAME
from ..config.config import BaseTaskConfig
from tensorlayerx.files import maybe_download_and_extract


class BaseTask(nn.Module):
    config_class = BaseTaskConfig

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

        weights_path = config.weights_path

        if not weights_path.startswith("http"):
            if pretrained_base_path is None:
                logging.warning("Don't load weight.")
                return task

            weights_path = os.path.join(pretrained_base_path, weights_path)
        else:
            if pretrained_base_path is None:
                pretrained_base_path = ".cache"
            url, name = weights_path.rsplit("/", 1)
            maybe_download_and_extract(name, pretrained_base_path, url + "/")
            weights_path = os.path.join(pretrained_base_path, name)
        task.load_weights(weights_path)
        return task

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            logging.error(f"Save directory ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # save weight
        self.save_weights(os.path.join(save_directory, TASK_WEIGHT_NAME), TASK_WEIGHT_FORMAT)

        # save config
        self.config.save_pretrained(save_directory)


class BaseForImageClassification(BaseTask):
    task_type = "image_classification"


class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"

