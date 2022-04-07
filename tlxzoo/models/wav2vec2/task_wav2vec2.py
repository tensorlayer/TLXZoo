from ...task.task import BaseForAutomaticSpeechRecognition
from .wav2vec2 import *
from ...utils.output import BaseTaskOutput


class Wav2Vec2ForAutomaticSpeechRecognition(BaseForAutomaticSpeechRecognition):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.wav2vec2 = Wav2Vec2Model(config.model_config, name="wav2vec2")
        self.dropout = tlx.nn.Dropout(config.final_dropout)
        self.lm_head = tlx.nn.Linear(config.vocab_size, in_features=config.model_config.hidden_size, name="lm_head")

    def _load_huggingface_tf_weight(self, weight_path):
        import h5py
        file = h5py.File(weight_path, "r")

        names = []
        for w in self.all_weights:
            name = w.name
            coder = name.split("/")[0]
            huggingface_weight_name = f"{coder}/tf_wav2vec2_for_ctc/" + name
            huggingface_weight_name = huggingface_weight_name.replace("conv/filters:0", "conv/kernel:0")
            huggingface_weight_name = huggingface_weight_name.replace("biases:0", "bias:0")
            huggingface_weight_name = huggingface_weight_name.replace("weights:0", "kernel:0")

            if huggingface_weight_name not in file:
                print(f"{huggingface_weight_name} do not init.")
                continue
            names.append(huggingface_weight_name)
            w.assign(file[huggingface_weight_name])

        # self.wav2vec2.wav2vec2.feature_projection.layer_norm.gamma.assign(
        # file["wav2vec2/tf_wav2vec2_for_ctc/wav2vec2/feature_projection/layer_norm/gamma:0"])
        # self.wav2vec2.wav2vec2.feature_projection.layer_norm.beta.assign(
        #     file["wav2vec2/tf_wav2vec2_for_ctc/wav2vec2/feature_projection/layer_norm/beta:0"])
        for root_name, g in file.items():
            for _, weights_dirs in g.attrs.items():
                for i in weights_dirs:
                    name = root_name + "/" + str(i, encoding="utf-8")
                    if name not in names:
                        print(f"{name} do not use.")

        return self

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.model_config.conv_kernel, self.config.model_config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def loss_fn(self, logits, pixel_mask, labels):
        if tlx.reduce_max(labels) >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        attention_mask = pixel_mask
        input_lengths = self._get_feat_extract_output_lengths(tlx.reduce_sum(attention_mask, axis=-1))

        # assuming that padded tokens are filled with -100
        # when not being attended to
        labels_mask = tlx.cast(labels >= 0, tlx.int32)
        target_lengths = tlx.reduce_sum(labels_mask, axis=-1)
        backend = importlib.import_module(f'{tlx.BACKEND}')
        loss = backend.nn.ctc_loss(
            logits=logits,
            labels=labels,
            logit_length=input_lengths,
            label_length=target_lengths,
            blank_index=self.config.model_config.pad_token_id,
            logits_time_major=False,
        )

        if self.config.ctc_loss_reduction == "sum":
            loss = tlx.reduce_sum(loss)
        if self.config.ctc_loss_reduction == "mean":
            loss = tlx.reduce_mean(loss)
        return loss

    def forward(
        self,
        inputs,
        pixel_mask=None
    ):
        outputs = self.wav2vec2(
            inputs=inputs,
            pixel_mask=pixel_mask,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        return BaseTaskOutput(logits=logits)