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
        if not hasattr(config, "task_class") and cls is not BaseTask:
            return cls(config, *args, **kwargs)
        return Registers.tasks[config.task_class](config, *args, **kwargs)

    def __call__(self, *args, return_output=False, **kwargs):
        if return_output:
            return super(BaseTask, self).__call__(*args, **kwargs)
        else:
            return super(BaseTask, self).__call__(*args, **kwargs).logits


@Registers.tasks.register
class BaseForImageClassification(BaseTask):
    task_type = "image_classification"


@Registers.tasks.register
class BaseForObjectDetection(BaseTask):
    task_type = "object_detection"


@Registers.tasks.register
class BaseForConditionalGeneration(BaseTask):
    task_type = "conditional_generation"


@Registers.tasks.register
class BaseForTextClassification(BaseTask):
    task_type = "text_classification"


@Registers.tasks.register
class BaseForPairTextClassification(BaseTask):
    task_type = "pair_text_classification"


@Registers.tasks.register
class BaseForTokenClassification(BaseTask):
    task_type = "token_classification"


@Registers.tasks.register
class BaseForAutomaticSpeechRecognition(BaseTask):
    task_type = "automatic_speech_recognition"


@Registers.tasks.register
class BaseForOpticalCharacterRecognition(BaseTask):
    task_type = "optical_character_recognition"


@Registers.tasks.register
class BaseForHumanPoseEstimation(BaseTask):
    task_type = "human_pose_estimation"


@Registers.tasks.register
class BaseForFaceRecognition(BaseTask):
    task_type = "face_recognition"

    def __call__(self, *args, return_output=True, **kwargs):
        if return_output:
            return super(BaseTask, self).__call__(*args, **kwargs)
        else:
            return super(BaseTask, self).__call__(*args, **kwargs).logits


@Registers.tasks.register
class BaseForFaceEmbedding(BaseForFaceRecognition):
    task_type = "face_embedding"

