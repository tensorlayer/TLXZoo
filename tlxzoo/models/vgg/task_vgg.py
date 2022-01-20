from ...task.task import BaseForImageClassification
from ...config.config import BaseTaskConfig
from .vgg import VGG
from ...utils.registry import Registers
from ...utils.output import BaseForImageClassificationTaskOutput
import tensorlayerx as tlx


@Registers.tasks.register
class VGGForImageClassification(BaseForImageClassification):
    def __init__(self, config: BaseTaskConfig):
        super(VGGForImageClassification, self).__init__(config)

        self.vgg = VGG(self.config.model_config)

        self.num_labels = config.num_labels
        in_channels = self.config.model_config.get_last_output_size()[-1]
        self.classifier = tlx.nn.Dense(n_units=self.num_labels, in_channels=in_channels, name="classifier")

    def forward(self, pixels, labels=None):
        outs = self.vgg(pixels)

        last_out = outs.output

        logits = self.classifier(last_out)
        if labels:
            loss = tlx.losses.softmax_cross_entropy_with_logits(labels, logits)
        else:
            loss = None

        return BaseForImageClassificationTaskOutput(logits=logits, loss=loss)



