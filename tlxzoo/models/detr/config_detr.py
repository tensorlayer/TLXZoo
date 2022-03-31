from ...config.config import BaseModelConfig, BaseTaskConfig
from ...utils.registry import Registers
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME


@Registers.model_configs.register
class DetrModelConfig(BaseModelConfig):
    model_type = "detr"

    def __init__(
            self,
            num_queries=100,
            num_encoder_layers=6,
            num_decoder_layers=6,
            model_dim=256,
            backbone_bn_shape=64,
            backbone_layer1_bn_shape=((64, 64, 256, 256), (64, 64, 256), (64, 64, 256)),
            backbone_layer2_bn_shape=((128, 128, 512, 512), (128, 128, 512), (128, 128, 512), (128, 128, 512)),
            backbone_layer3_bn_shape=((256, 256, 1024, 1024), (256, 256, 1024), (256, 256, 1024), (256, 256, 1024),
                                      (256, 256, 1024), (256, 256, 1024)),
            backbone_layer4_bn_shape=((512, 512, 2048, 2048), (512, 512, 2048), (512, 512, 2048)),
            return_intermediate_dec=True,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model_dim = model_dim
        self.backbone_bn_shape = backbone_bn_shape
        self.return_intermediate_dec = return_intermediate_dec
        self.backbone_layer1_bn_shape = backbone_layer1_bn_shape
        self.backbone_layer2_bn_shape = backbone_layer2_bn_shape
        self.backbone_layer3_bn_shape = backbone_layer3_bn_shape
        self.backbone_layer4_bn_shape = backbone_layer4_bn_shape
        super(DetrModelConfig, self).__init__(weights_path=weights_path,
                                              **kwargs, )


@Registers.task_configs.register
class DetrForObjectDetectionTaskConfig(BaseTaskConfig):
    task_type = "detr_for_object_detection"
    model_config_type = DetrModelConfig

    def __init__(self,
                 model_config: DetrModelConfig = None,
                 weights_path=TASK_WEIGHT_NAME,
                 num_classes=92,
                 class_cost=1,
                 bbox_cost=5,
                 giou_cost=2,
                 num_labels=91,
                 dice_loss_coefficient=1,
                 bbox_loss_coefficient=5,
                 giou_loss_coefficient=2,
                 eos_coefficient=0.1,
                 num_queries=100,
                 auxiliary_loss=False,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        self.num_classes = num_classes
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.num_labels = num_labels
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.auxiliary_loss = auxiliary_loss
        self.num_queries = num_queries
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(DetrForObjectDetectionTaskConfig, self).__init__(model_config, **kwargs)


@Registers.task_configs.register
class DetrForSegmentationTaskConfig(BaseTaskConfig):
    task_type = "detr_for_segmentation"
    model_config_type = DetrModelConfig

    def __init__(self,
                 model_config: DetrModelConfig = None,
                 weights_path=TASK_WEIGHT_NAME,
                 num_classes=92,
                 class_cost=1,
                 bbox_cost=5,
                 giou_cost=2,
                 num_labels=91,
                 mask_loss_coefficient=1,
                 dice_loss_coefficient=1,
                 bbox_loss_coefficient=5,
                 giou_loss_coefficient=2,
                 eos_coefficient=0.1,
                 init_xavier_std=1.0,
                 auxiliary_loss=False,
                 num_queries=100,
                 **kwargs):

        if model_config is None:
            model_config = self.model_config_type()

        self.num_classes = num_classes
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.num_labels = num_labels
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.auxiliary_loss = auxiliary_loss
        self.init_xavier_std = init_xavier_std
        self.num_queries = num_queries
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(DetrForSegmentationTaskConfig, self).__init__(model_config, **kwargs)

