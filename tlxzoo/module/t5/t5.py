import tensorlayerx as tlx
import copy
from .modeling import *


class T5Model(tlx.nn.Module):
    def __init__(self, vocab_size=32128,
                 d_model=768,
                 d_kv=64,
                 d_ff=3072,
                 num_layers=12,
                 num_decoder_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 initializer_factor=1.0,
                 feed_forward_proj="relu",
                 is_encoder_decoder=True,
                 use_cache=True,
                 pad_token_id=0,
                 eos_token_id=1,
                 decoder_start_token_id=0,
                 name="t5", **kwargs):
        """
        :param vocab_size: (:obj:`int`, `optional`, defaults to 32128):
            Vocabulary size of the T5 model.
        :param d_model: (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        :param d_kv: (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head.
        :param d_ff: (:obj:`int`, `optional`, defaults to 2048):
            Size of the intermediate feed forward layer.
        :param num_layers: (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        :param num_decoder_layers: (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder.
        :param num_heads: (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        :param relative_attention_num_buckets: (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        :param dropout_rate: (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        :param layer_norm_epsilon: (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        :param initializer_factor: (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices .
        :param feed_forward_proj: (:obj:`string`, `optional`, defaults to :obj:`"relu"`):
            Type of feed forward layer to be used.
        :param use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions.
        """
        super(T5Model, self).__init__(name=name)

        self.vocab_size = vocab_size
        self.is_encoder_decoder = is_encoder_decoder
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.model_dim = d_model

        self.shared = T5Embedding(vocab_size, d_model, name=name + "/shared")

        self.encoder = T5MainLayer(False, False, relative_attention_num_buckets, d_model, d_kv, num_heads,
                                   initializer_factor, dropout_rate, layer_norm_epsilon, d_ff, feed_forward_proj,
                                   num_layers, self.shared, name=name + "/encoder")

        self.decoder = T5MainLayer(True, use_cache, relative_attention_num_buckets, d_model, d_kv, num_heads,
                                   initializer_factor, dropout_rate, layer_norm_epsilon, d_ff, feed_forward_proj,
                                   num_decoder_layers, self.shared, name=name + "/decoder")

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


class T5EncoderModel(tlx.nn.Module):
    def __init__(self, vocab_size=32128,
                 d_model=768,
                 d_kv=64,
                 d_ff=3072,
                 num_layers=12,
                 num_heads=12,
                 relative_attention_num_buckets=32,
                 dropout_rate=0.1,
                 layer_norm_epsilon=1e-6,
                 initializer_factor=1.0,
                 feed_forward_proj="relu",
                 name="t5_encoder", **kwargs):
        super(T5EncoderModel, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj

        self.shared = T5Embedding(vocab_size, d_model, name=name + "/shared")

        self.encoder = T5MainLayer(False, False, relative_attention_num_buckets, d_model,
                                   d_kv, num_heads, initializer_factor, dropout_rate, layer_norm_epsilon, d_ff,
                                   feed_forward_proj, num_layers, self.shared, name=name + "/encoder")

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
