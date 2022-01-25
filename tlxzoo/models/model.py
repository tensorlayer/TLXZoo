from tensorlayerx import nn, logging
import os
from ..utils import MODEL_WEIGHT_FORMAT, MODEL_WEIGHT_NAME
from ..config.config import BaseModelConfig
from tensorlayerx.files import maybe_download_and_extract


class BaseModule(nn.Module):
    config_class = BaseModelConfig

    def __init__(self, config, *args, **kwargs):
        """Initialize BaseModule, inherited from `tensorlayerx.nn.Module`"""

        super(BaseModule, self).__init__(*args, **kwargs)
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_base_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_path and config are both None.")
            config = cls.config_class.from_pretrained(pretrained_base_path, **kwargs)

        model = cls(config, *model_args, **kwargs)

        weights_path = config.weights_path
        if not weights_path.startswith("http"):
            if pretrained_base_path is None:
                logging.warning("Don't load weight.")
                return model

            weights_path = os.path.join(pretrained_base_path, weights_path)
        else:
            if pretrained_base_path is None:
                pretrained_base_path = ".cache"
            url, name = weights_path.rsplit("/", 1)
            maybe_download_and_extract(name, pretrained_base_path, url + "/")
            weights_path = os.path.join(pretrained_base_path, name)

        model.load_weights(weights_path)
        return model

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            logging.error(f"Save directory ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # save weight
        self.save_weights(os.path.join(save_directory, MODEL_WEIGHT_NAME), MODEL_WEIGHT_FORMAT)

        # save config
        self.config.save_pretrained(save_directory)