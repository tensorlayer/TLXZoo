from tensorlayerx import nn, logging
import os
from ..utils import MODEL_WEIGHT_FORMAT, MODEL_WEIGHT_NAME
from ..config.config import BaseModelConfig
from ..utils.from_pretrained import ModuleFromPretrainedMixin


class BaseModule(nn.Module, ModuleFromPretrainedMixin):
    config_class = BaseModelConfig

    def __init__(self, config, *args, **kwargs):
        """Initialize BaseModule, inherited from `tensorlayerx.nn.Module`"""

        super(BaseModule, self).__init__(*args, **kwargs)
        self.config = config

    def save_pretrained(self, save_directory):
        self._save_pretrained(save_directory, MODEL_WEIGHT_NAME, MODEL_WEIGHT_FORMAT)