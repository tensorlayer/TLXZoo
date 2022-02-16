from ...utils.registry import Registers
import tensorlayerx as tlx
from ...task.task import BaseForObjectDetection
from .config_yolo import YOLOv4ForObjectDetectionTaskConfig
from .yolo import YOLOv4, Convolutional
from ...utils.output import BaseForObjectDetectionTaskOutput, float_tensor, Optional
from dataclasses import dataclass


@dataclass
class YOLOv4ForObjectDetectionTaskOutput(BaseForObjectDetectionTaskOutput):
    sbbox: Optional[float_tensor] = None
    mbbox: Optional[float_tensor] = None
    lbbox: Optional[float_tensor] = None


@Registers.tasks.register
class YOLOv4ForImageClassification(BaseForObjectDetection):
    config_class = YOLOv4ForObjectDetectionTaskConfig

    def __init__(self, config: YOLOv4ForObjectDetectionTaskConfig = None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        super(YOLOv4ForImageClassification, self).__init__(config)

        if model is not None:
            self.yolo = model
        else:
            self.yolo = YOLOv4(self.config.model_config)

        self.sconv = Convolutional(self.config.sconv_filters_shape, activate=False, bn=False)
        self.mconv = Convolutional(self.config.mconv_filters_shape, activate=False, bn=False)
        self.lconv = Convolutional(self.config.lconv_filters_shape, activate=False, bn=False)

    def forward(self, pixels, labels=None):
        outs = self.yolo(pixels)

        sbbox = self.sconv(outs.soutput)
        mbbox = self.mconv(outs.moutput)
        lbbox = self.lconv(outs.loutput)

        return YOLOv4ForObjectDetectionTaskOutput(sbbox=sbbox, mbbox=mbbox, lbbox=lbbox)