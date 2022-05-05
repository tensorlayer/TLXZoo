from ...utils.registry import Registers
from ...task.task import BaseForImageSegmentation
from .config_unet import *
from .unet import *


@Registers.tasks.register
class UnetForImageSegmentation(BaseForImageSegmentation):
    config_class = UnetForImageSegmentationTaskConfig

    def __init__(self, config=None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        super(UnetForImageSegmentation, self).__init__(config)

        if model is not None:
            self.model = model
        else:
            self.model = UnetModel(nx=config.model_config.nx, ny=config.model_config.ny,
                                   channels=config.model_config.channels, num_classes=config.model_config.num_classes,
                                   layer_depth=config.model_config.layer_depth,
                                   filters_root=config.model_config.filters_root)

    def forward(self, inputs):
        return self.model(inputs)

    def loss_fn(self, preds, labels):
        labels_argmax = tlx.argmax(labels, -1)

        return tlx.losses.softmax_cross_entropy_with_logits(preds, labels_argmax)


def mean_iou(y_true, y_pred):
    y_true = tlx.cast(y_true, tlx.float64)
    y_pred = tlx.cast(y_pred, tlx.float64)
    I = tlx.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tlx.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tlx.reduce_mean(I / U)


def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tlx.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tlx.reduce_sum(y_true, axis=[1, 2, 3]) + tlx.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tlx.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice



