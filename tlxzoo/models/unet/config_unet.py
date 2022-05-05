from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils.registry import Registers
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME


@Registers.model_configs.register
class UnetModelConfig(BaseModelConfig):
    model_type = "unet"

    def __init__(
            self,
            nx=172,
            ny=172,
            channels=1,
            num_classes=2,
            layer_depth=3,
            filters_root=64,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.num_classes = num_classes
        self.layer_depth = layer_depth
        self.filters_root = filters_root
        super(UnetModelConfig, self).__init__(weights_path=weights_path,
                                              **kwargs, )


@Registers.task_configs.register
class UnetForImageSegmentationTaskConfig(BaseTaskConfig):
    task_type = "unet_for_image_segmentation"
    model_config_type = UnetModelConfig

    def __init__(self,
                 model_config: UnetModelConfig = None,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(UnetForImageSegmentationTaskConfig, self).__init__(model_config, **kwargs)
