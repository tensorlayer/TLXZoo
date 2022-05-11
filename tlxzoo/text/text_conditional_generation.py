from tlxzoo.module import *
import tensorlayerx as tlx


class TextForConditionalGeneration(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(TextForConditionalGeneration, self).__init__()
        if backbone in "t5":
            self.backbone = T5Model(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

        self.model_dim = self.backbone.model_dim
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)

        if not self.tie_word_embeddings:
            initializer_factor = kwargs.pop("initializer_factor", self.backbone.initializer_factor)
            vocab_size = kwargs.pop("vocab_size", self.backbone.vocab_size)
            lm_head_initializer = tlx.initializers.RandomNormal(mean=0, stddev=initializer_factor)
            self.lm_head = tlx.layers.Dense(
                vocab_size, b_init=None, name="lm_head", W_init=lm_head_initializer
            )

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.backbone.decoder_start_token_id
        pad_token_id = self.backbone.pad_token_id

        start_tokens = decoder_start_token_id * tlx.ones((shape_list(input_ids)[0], 1))
        start_tokens = tlx.cast(start_tokens, input_ids.dtype)  # Ensure compatible dtypes for concatenation
        shifted_input_ids = tlx.concat([start_tokens, input_ids[:, :-1]], -1)

        assert pad_token_id is not None, "pad_token_id has to be defined."
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
        logits = tlx.reshape(logits, shape=(-1, self.backbone.vocab_size))
        labels = tlx.where(mask, labels, 0)
        return loss(logits=logits, target_seqs=labels, input_mask=mask)

    def generate(self, inputs=None,
                 attention_mask=None):
        ...

    def generate_one(self, inputs=None,
                     attention_mask=None):
        decoder_start_token_id = self.backbone.decoder_start_token_id
        start_tokens = decoder_start_token_id * tlx.ones((shape_list(inputs)[0], 1))
        start_tokens = tlx.cast(start_tokens, inputs.dtype)

        max_length = 512
        encoder_outputs = self.backbone.encoder(
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

        while int(start_tokens[0][-1]) != self.backbone.eos_token_id and \
                start_tokens.shape[1] < max_length:
            logits = self.forward(inputs=inputs, attention_mask=attention_mask, decoder_input_ids=start_tokens,
                                  encoder_outputs=encoder_outputs)
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

        decoder_outputs = self.backbone(inputs=inputs, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                                        decoder_head_mask=decoder_head_mask, encoder_outputs=encoder_outputs,
                                        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                                        decoder_inputs_embeds=decoder_inputs_embeds, use_cache=use_cache,
                                        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                        return_dict=return_dict,
                                        **kwargs)
        sequence_output = decoder_outputs[0]

        if self.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            logits = self.backbone.shared(sequence_output, mode="linear")
        else:
            logits = self.lm_head(sequence_output)

        logits = tlx.cast(logits, tlx.float32)

        return logits
