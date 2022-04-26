from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
from ...utils.registry import Registers


@Registers.model_configs.register
class RetinaFaceModelConfig(BaseModelConfig):
    model_type = "retina_face"

    def __init__(
            self,
            input_size=640,
            weights_decay=5e-4,
            out_channel=256,
            min_sizes=None,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.input_size = input_size
        self.weights_decay = weights_decay
        self.out_channel = out_channel
        self.min_sizes = min_sizes if min_sizes else [[16, 32], [64, 128], [256, 512]]
        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class RetinaFaceForFaceRecognitionTaskConfig(BaseTaskConfig):
    task_type = "retina_face_for_face_recognition"
    model_config_type = RetinaFaceModelConfig

    def __init__(self,
                 model_config: model_config_type = None,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        if model_config is None:
            model_config = self.model_config_type()

        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(RetinaFaceForFaceRecognitionTaskConfig, self).__init__(model_config, **kwargs)
