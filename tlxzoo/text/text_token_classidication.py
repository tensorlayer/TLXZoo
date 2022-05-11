from tlxzoo.module import *
import tensorlayerx as tlx


class TextTokenClassification(tlx.nn.Module):

    def __init__(self, backbone, **kwargs):
        super(TextTokenClassification, self).__init__()

        if backbone in "t5":
            self.backbone = T5EncoderModel(**kwargs)
        elif backbone == "bert":
            self.backbone = Bert(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

        self.n_class = kwargs.pop("n_class", 9)

        initializer = tlx.initializers.RandomNormal(mean=0, stddev=1.0)
        self.classifier = tlx.layers.Linear(
            out_features=self.n_class, in_features=self.backbone.d_model, W_init=initializer
        )

    def loss_fn(self, logits, labels):
        loss = tlx.losses.cross_entropy_seq_with_mask

        mask = tlx.not_equal(labels, -100)
        logits = tlx.reshape(logits, shape=(-1, self.n_class))
        labels = tlx.where(mask, labels, 0)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def forward(self, inputs=None,
                attention_mask=None,
                **kwargs):

        hidden_states = self.backbone(inputs=inputs, attention_mask=attention_mask, **kwargs)

        logits = self.classifier(hidden_states)

        return logits
