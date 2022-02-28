from ...task.task import BaseForImageClassification
from .resnet import ResNet
from ...utils.registry import Registers
from ...utils.output import BaseForImageClassificationTaskOutput
from .config_resnet import ResNetForImageClassificationTaskConfig
from ...utils import glorot_uniform
import tensorlayerx as tlx


@Registers.tasks.register
class ResNetForImageClassification(BaseForImageClassification):
    config_class = ResNetForImageClassificationTaskConfig

    def __init__(self, config: ResNetForImageClassificationTaskConfig = None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        super(ResNetForImageClassification, self).__init__(config)

        if model is not None:
            self.resnet = model
        else:
            self.resnet = ResNet(self.config.model_config)

        self.num_labels = config.num_labels
        self._final_conv = tlx.nn.Conv2d(self.num_labels, filter_size=(1, 1), strides=(1, 1), W_init=glorot_uniform,
                                         b_init="zeros", in_channels=64, name="final_conv")

    def forward(self, pixels, labels=None):
        outs = self.resnet(pixels)

        last_out = outs.output

        net = self._final_conv(last_out)
        logits = tlx.squeeze(net, axis=[1, 2])
        if labels:
            loss = self.loss_fn(labels, logits)
        else:
            loss = None

        return BaseForImageClassificationTaskOutput(logits=logits, loss=loss)

    def loss_fn(self, output, target):
        loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)
        return loss


