from ...task.task import BaseForImageClassification
from .resnet import ResNet
from ...utils.registry import Registers
from ...utils.output import BaseForImageClassificationTaskOutput
from .config_resnet import ResNetForImageClassificationTaskConfig
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
        try:
            in_channels = self.config.model_config.get_last_output_size()[-1]
            self.classifier = tlx.nn.Dense(n_units=self.num_labels, in_channels=in_channels, name="classifier")
        except:
            self.classifier = tlx.nn.Dense(n_units=self.num_labels, name="classifier")

    def forward(self, pixels, labels=None):
        outs = self.resnet(pixels)

        last_out = outs.output

        logits = self.classifier(last_out)
        if labels:
            loss = tlx.losses.softmax_cross_entropy_with_logits(labels, logits)
        else:
            loss = None

        return BaseForImageClassificationTaskOutput(logits=logits, loss=loss)



