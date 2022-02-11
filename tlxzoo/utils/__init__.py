MODEL_CONFIG = "model_config.json"
TASK_CONFIG = "task_config.json"
FEATURE_CONFIG = "feature_config.json"
RUNNER_CONFIG = "runner_config.json"
DATA_CONFIG = "data_config.json"
INFER_CONFIG = "infer_config.json"
APP_CONFIG = "app_config.json"

MODEL_WEIGHT_FORMAT = "npz"
MODEL_WEIGHT_NAME = "model.npz"
TASK_WEIGHT_FORMAT = "npz"
TASK_WEIGHT_NAME = "task.npz"

from .from_pretrained import ModuleFromPretrainedMixin
from .output import (BaseModelOutput,
                     BaseForImageClassificationTaskOutput)

from .registry import Registers
