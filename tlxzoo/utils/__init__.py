MODEL_CONFIG = "model_config.json"
TASK_CONFIG = "task_config.json"
IMAGE_FEATURE_CONFIG = "image_feature_config.json"

MODEL_WEIGHT_FORMAT = "npz"
MODEL_WEIGHT_NAME = "model.npz"
TASK_WEIGHT_FORMAT = "npz"
TASK_WEIGHT_NAME = "task.npz"

from .from_pretrained import ModuleFromPretrainedMixin
from .output import (BaseModelOutput,
                    BaseForImageClassificationTaskOutput)

from .registry import Registers