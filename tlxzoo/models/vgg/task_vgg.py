from ...task.task import BaseForImageClassification
from .vgg import VGG
from ...utils.registry import Registers
from ...utils.output import BaseForImageClassificationTaskOutput
from .config_vgg import VGGForImageClassificationTaskConfig
import tensorlayerx as tlx


@Registers.tasks.register
class VGGForImageClassification(BaseForImageClassification):
    config_class = VGGForImageClassificationTaskConfig

    def __init__(self, config: VGGForImageClassificationTaskConfig = None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        super(VGGForImageClassification, self).__init__(config)

        if model is not None:
            self.vgg = model
        else:
            self.vgg = VGG(self.config.model_config)

        self.num_labels = config.num_labels
        self.batch_norm = tlx.nn.BatchNorm()
        self.dropout = tlx.nn.Dropout(0.7)
        try:
            in_channels = self.config.model_config.get_last_output_size()[-1]
            self.classifier = tlx.nn.Dense(n_units=self.num_labels, in_channels=in_channels, name="classifier")
        except:
            self.classifier = tlx.nn.Dense(n_units=self.num_labels, name="classifier")

        self.train_weights = self.trainable_weights
        self.li_regularizer = tlx.losses.li_regularizer(0.00001)

        self.l2_weights = []
        for w in self.train_weights:
            if w.name.startswith("conv") and len(w.shape) >= 2:
                self.l2_weights.append(w)
            if w.name.startswith("fc1") and len(w.shape) >= 2:
                self.l2_weights.append(w)

    def loss_fn(self, output, target):
        loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)
        loss2 = 0
        for w in self.l2_weights:
            loss2 += self.li_regularizer(w)
        return loss + loss2

    def forward(self, pixels, labels=None):
        outs = self.vgg(pixels)

        last_out = outs.output
        last_out = self.batch_norm(last_out)
        last_out = self.dropout(last_out)

        logits = self.classifier(last_out)
        if labels:
            loss = self.loss_fn(labels, logits)
        else:
            loss = None

        return BaseForImageClassificationTaskOutput(logits=logits, loss=loss)



