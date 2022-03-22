import tensorlayerx as tlx
from ..model import BaseModule
import copy
from .modeling import *


def load_huggingface_tf_weight(mode, weight_path):
    import h5py
    file = h5py.File(weight_path, "r")

    for w in mode.all_weights:
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
        if huggingface_weight_name not in file:
            continue
        w.assign(file[huggingface_weight_name])

    return mode


class T5Model(BaseModule):
    def __init__(self, config, name="t5", **kwargs):
        super(T5Model, self).__init__(config, name=name)

        self.shared = T5Embedding(config.vocab_size, config.d_model, name=name+"/shared")

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_decoder = False
        self.encoder = T5MainLayer(encoder_config, self.shared, name=name+"/encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5MainLayer(decoder_config, self.shared, name=name+"/decoder")

    def _load_huggingface_tf_weight(self, weight_path):
        return load_huggingface_tf_weight(self, weight_path)

    def forward(
            self,
            inputs=None,
            attention_mask=None,
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
            **kwargs,
    ):
        if head_mask is not None and decoder_head_mask is None:
            decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                past_key_values=None,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        past = (encoder_outputs, decoder_outputs[1]) if use_cache else None
        if past is not None:
            decoder_outputs = decoder_outputs[:1] + (past,) + decoder_outputs[2:]
        return decoder_outputs + encoder_outputs


class T5EncoderModel(BaseModule):
    def __init__(self, config, name="t5_encoder", **kwargs):
        super(T5EncoderModel, self).__init__(config, name=name)

        self.shared = T5Embedding(config.vocab_size, config.d_model, name=name+"/shared")

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_decoder = False
        self.encoder = T5MainLayer(encoder_config, self.shared, name=name+"/encoder")

    def _load_huggingface_tf_weight(self, weight_path):
        return load_huggingface_tf_weight(self, weight_path)

    def forward(
            self,
            inputs=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):
        encoder_outputs = self.encoder(
            inputs,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = encoder_outputs[0]
        return hidden_states

