from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
from ...utils.registry import Registers


@Registers.model_configs.register
class ArcFaceModelConfig(BaseModelConfig):
    model_type = "arc_face"

    def __init__(
            self,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        super().__init__(
            weights_path=weights_path,
            **kwargs,
        )


@Registers.task_configs.register
class ArcFaceForFaceEmbeddingTaskConfig(BaseTaskConfig):
    task_type = "arc_face_for_face_embedding"
    model_config_type = ArcFaceModelConfig

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
        super(ArcFaceForFaceEmbeddingTaskConfig, self).__init__(model_config, **kwargs)
