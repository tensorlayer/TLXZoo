from tlxzoo.module import *
import tensorlayerx as tlx


class TextClassification(tlx.nn.Module):

    def __init__(self, backbone, **kwargs):
        super(TextClassification, self).__init__()

        if backbone == "t5":
            self.backbone = T5EncoderModel(**kwargs)
        elif backbone == "bert":
            self.backbone = Bert(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

        n_class = kwargs.pop("n_class", 2)
        self.method = kwargs.pop("method", "mean")

        initializer = tlx.initializers.RandomNormal(mean=0, stddev=1.0)
        self.classifier = tlx.layers.Linear(
            out_features=n_class, in_features=self.backbone.d_model, W_init=initializer
        )

    def loss_fn(self, logits, labels):
        loss = tlx.losses.softmax_cross_entropy_with_logits(logits, labels)
        return loss

    def forward(self, inputs=None, attention_mask=None, **kwargs):

        hidden_states = self.backbone(inputs=inputs, attention_mask=attention_mask, **kwargs)

        if self.method == "mean":
            hidden_state = tlx.reduce_mean(hidden_states, axis=1)
            logits = self.classifier(hidden_state)
        elif self.method == "first":
            hidden_state = hidden_states[:, 0]
            logits = self.classifier(hidden_state)
        else:
            hidden_state = tlx.reduce_mean(hidden_states, axis=1)
            logits = self.classifier(hidden_state)

        return logits
