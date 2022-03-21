from ...task.task import BaseForConditionalGeneration
from .t5 import T5Model
from ...utils.registry import Registers
from .config_t5 import T5ForConditionalGenerationTaskConfig
import tensorlayerx as tlx
from ...utils.output import BaseTaskOutput, dataclass, Optional, float_tensor
from .modeling import shape_list


@dataclass
class T5ForConditionalGenerationOutput(BaseTaskOutput):
    encoder_outputs: Optional[tuple] = None


@Registers.tasks.register
class T5ForConditionalGeneration(BaseForConditionalGeneration):
    config_class = T5ForConditionalGenerationTaskConfig

    def __init__(self, config: T5ForConditionalGenerationTaskConfig = None, model=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)
        self.model_dim = config.model_config.d_model
        super(T5ForConditionalGeneration, self).__init__(config)

        if model is not None:
            self.t5 = model
        else:
            self.t5 = T5Model(self.config.model_config)

        if not config.tie_word_embeddings:
            lm_head_initializer = tlx.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = tlx.layers.Dense(
                config.model_config.vocab_size, b_init=None, name="lm_head", W_init=lm_head_initializer
            )

    def _load_huggingface_tf_weight(self, weight_path):
        import h5py
        file = h5py.File(weight_path, "r")

        for w in self.all_weights:
            name = w.name.split("/", 1)[1]
            coder = name.split("/")[0]
            huggingface_weight_name = f"{coder}/tf_t5with_lm_head_model/" + name
            huggingface_weight_name = huggingface_weight_name.replace("shared/embeddings", "shared/weight")
            huggingface_weight_name = huggingface_weight_name.replace("q/weights", "q/kernel")
            huggingface_weight_name = huggingface_weight_name.replace("k/weights", "k/kernel")
            huggingface_weight_name = huggingface_weight_name.replace("v/weights", "v/kernel")
            huggingface_weight_name = huggingface_weight_name.replace("o/weights", "o/kernel")
            huggingface_weight_name = huggingface_weight_name.replace("wi/weights", "wi/kernel")
            huggingface_weight_name = huggingface_weight_name.replace("wo/weights", "wo/kernel")
            w.assign(file[huggingface_weight_name])

        return self

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.model_config.decoder_start_token_id
        pad_token_id = self.config.model_config.pad_token_id

        assert (
                decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. " \
           "In TF T5 it is usually set to the pad_token_id. See T5 docs for more information"

        start_tokens = decoder_start_token_id * tlx.ones((shape_list(input_ids)[0], 1))
        start_tokens = tlx.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
        shifted_input_ids = tlx.concat([start_tokens, input_ids[:, :-1]], -1)

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = tlx.where(
            shifted_input_ids == -100,
            tlx.cast(pad_token_id * tlx.ones(shape_list(shifted_input_ids)), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        shifted_input_ids = tlx.identity(shifted_input_ids)

        return shifted_input_ids

    def loss_fn(self, logits, labels):
        loss = tlx.losses.cross_entropy_seq_with_mask

        mask = tlx.not_equal(labels, -100)
        logits = tlx.reshape(logits, shape=(-1, self.config.model_config.vocab_size))
        labels = tlx.where(mask, labels, 0)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def generate(self, inputs=None,
                 attention_mask=None):
        ...

    def generate_one(self, inputs=None,
                     attention_mask=None):
        decoder_start_token_id = self.config.model_config.decoder_start_token_id
        start_tokens = decoder_start_token_id * tlx.ones((shape_list(inputs)[0], 1))
        start_tokens = tlx.cast(start_tokens, inputs.dtype)

        max_length = 512
        encoder_outputs = self.t5.encoder(
            inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
        )

        while int(start_tokens[0][-1]) != self.config.model_config.eos_token_id and \
                start_tokens.shape[1] < max_length:
            output = self.forward(inputs=inputs, attention_mask=attention_mask, decoder_input_ids=start_tokens,
                                  encoder_outputs=encoder_outputs)
            logits = output.logits
            last_tokens = tlx.argmax(logits, -1)[:, -1:]
            start_tokens = tlx.concat([start_tokens, last_tokens], axis=-1)
        return start_tokens

    def forward(self, inputs=None,
                attention_mask=None,
                labels=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):

        if (
                labels is not None
                and decoder_input_ids is None
                and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.t5(inputs=inputs, attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                                  decoder_head_mask=decoder_head_mask, encoder_outputs=encoder_outputs,
                                  past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                  decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache,
                                  output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                  return_dict=return_dict,
                                  **kwargs)
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            logits = self.t5.shared(sequence_output, mode="linear")
        else:
            logits = self.lm_head(sequence_output)

        logits = tlx.cast(logits, tlx.float32)

        return BaseTaskOutput(logits=logits)