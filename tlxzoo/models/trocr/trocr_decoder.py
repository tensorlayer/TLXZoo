
import copy
import math
import random
from .vit import *
from typing import Optional, Tuple

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

LARGE_NEGATIVE = -1e8


def _make_causal_mask(input_ids_shape, dtype, past_key_values_length=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = tlx.ones((tgt_len, tgt_len)) * LARGE_NEGATIVE
    mask_cond = tlx.arange(shape_list(mask)[-1])

    mask = tlx.where(mask_cond < tlx.reshape(mask_cond + 1, (shape_list(mask)[-1], 1)), 0.0, mask)

    if past_key_values_length > 0:
        mask = tlx.concat([tlx.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    return tlx.tile(mask[None, None, :, :], (bsz, 1, 1, 1))


def _expand_mask(mask, dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tlx.constant(1.0)
    mask = tlx.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tlx.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class TrOCREmbedding(tlx.nn.Embedding):
    def forward(self, inputs, mode="emb"):
        if mode == "emb":
            outputs = self.embedding_lookup(params=self.embeddings, ids=inputs)
            return outputs
        else:
            first_dims = shape_list(inputs)[:-1]
            x = tlx.reshape(inputs, [-1, self.embedding_size])
            logits = tlx.matmul(x, self.embeddings, transpose_b=True)

            return tlx.reshape(logits, first_dims + [self.vocabulary_size])


class TrOCRLearnedPositionalEmbedding(TrOCREmbedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings=num_embeddings + self.offset, embedding_dim=embedding_dim, **kwargs)

    def forward(self, input_shape, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_shape[:2]

        positions = tlx.arange(past_key_values_length, seq_len + past_key_values_length, delta=1)
        return super().forward(positions + self.offset)


class TrOCRSinusoidalPositionalEmbedding(Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(num_positions, embedding_dim, padding_idx)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tlx.exp(tlx.arange(half_dim, dtype=tlx.float32) * -emb)
        emb = tlx.arange(num_embeddings, dtype=tlx.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = tlx.concat([tlx.sin(emb), tlx.cos(emb)], axis=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = tlx.concat([emb, tlx.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input_ids, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # Create the position ids from the input token ids. Any padded tokens remain padded.
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # expand embeddings if needed
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

        return x

    def create_position_ids_from_input_ids(
        self, input_ids, padding_idx: int, past_key_values_length: Optional[int] = 0
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (tlx.cumsum(mask, axis=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class TrOCRAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = tlx.nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if not (self.head_dim * num_heads == self.embed_dim):
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = tlx.nn.Linear(in_features=self.kdim, out_features=embed_dim, b_init='constant' if bias else None)
        self.v_proj = tlx.nn.Linear(in_features=self.vdim, out_features=embed_dim, b_init='constant' if bias else None)
        self.q_proj = tlx.nn.Linear(in_features=embed_dim, out_features=embed_dim, b_init='constant' if bias else None)

        self.out_proj = tlx.nn.Linear(in_features=embed_dim, out_features=embed_dim, b_init='constant' if bias else None)

    def _shape(self, tensor, seq_len: int, bsz: int):
        return tlx.transpose(tlx.reshape(tensor, [bsz, seq_len, self.num_heads, self.head_dim]), perm=[0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = tlx.concat([past_key_value[0], key_states], axis=2)
            value_states = tlx.concat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tlx.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tlx.reshape(key_states, proj_shape)
        value_states = tlx.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        # attn_weights = tlx.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = tlx.matmul(query_states, key_states, transpose_b=True)

        if tuple(shape_list(attn_weights)) != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if tuple(shape_list(attention_mask)) != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = tlx.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tlx.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tlx.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            if tuple(shape_list(layer_head_mask)) != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = tlx.reshape(layer_head_mask, (1, -1, 1, 1)) * tlx.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))
            attn_weights = tlx.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights)

        # attn_output = tlx.bmm(attn_probs, value_states)
        attn_output = tlx.matmul(attn_probs, value_states)

        if tuple(shape_list(attn_output)) != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = tlx.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = tlx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tlx.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value


class TrOCRDecoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = TrOCRAttention(
            config,
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = tlx.nn.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = tlx.nn.LayerNorm(normalized_shape=self.embed_dim)
        self.self_attn_layer_norm.build([None, None, self.embed_dim])

        if config.is_decoder:
            self.encoder_attn = TrOCRAttention(
                config,
                embed_dim=self.embed_dim,
                num_heads=config.decoder_attention_heads,
                kdim=config.cross_attention_hidden_size,
                vdim=config.cross_attention_hidden_size,
                dropout=config.attention_dropout,
                is_decoder=True,
                is_cross_attention=True,
            )
            self.encoder_attn_layer_norm = tlx.nn.LayerNorm(self.embed_dim)
            self.encoder_attn_layer_norm.build([None, None, self.embed_dim])

        self.fc1 = tlx.nn.Linear(in_features=self.embed_dim, out_features=config.decoder_ffn_dim)
        self.fc2 = tlx.nn.Linear(in_features=config.decoder_ffn_dim, out_features=self.embed_dim)
        self.final_layer_norm = tlx.nn.LayerNorm(self.embed_dim)
        self.final_layer_norm.build([None, None, self.embed_dim])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        # hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.self_attn_layer_norm.layernorm,
                           self.self_attn_layer_norm.gamma, self.self_attn_layer_norm.beta)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None

        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
            )

            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = ln(hidden_states, self.encoder_attn_layer_norm.layernorm,
                               self.encoder_attn_layer_norm.gamma, self.encoder_attn_layer_norm.beta)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        # hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.final_layer_norm.layernorm,
                           self.final_layer_norm.gamma, self.final_layer_norm.beta)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TrOCRDecoder(Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`TrOCRDecoderLayer`

    Args:
        config: TrOCRConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = tlx.nn.Dropout(config.dropout)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = tlx.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)

        if config.use_learned_position_embeddings:
            self.embed_positions = TrOCRLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(
                config.max_position_embeddings + self.padding_idx + 1,
                config.hidden_size,
                self.padding_idx,
            )

        if config.layernorm_embedding:
            self.layernorm_embedding = tlx.nn.LayerNorm(normalized_shape=config.hidden_size)
            self.layernorm_embedding.build([None, None, config.hidden_size])
        else:
            self.layernorm_embedding = None

        self.decode_layers = tlx.nn.ModuleList([TrOCRDecoderLayer(config) for _ in range(config.decoder_layers)])

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tlx.get_tensor_shape(input_ids)
            input_ids = tlx.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = tlx.get_tensor_shape(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = shape_list(past_key_values[0][0])[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self.config.use_learned_position_embeddings:
            embed_pos = self.embed_positions(input_shape, past_key_values_length=past_key_values_length)
        else:
            embed_pos = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + embed_pos

        if self.layernorm_embedding is not None:
            hidden_states = ln(hidden_states, self.layernorm_embedding.layernorm,
                               self.layernorm_embedding.gamma, self.layernorm_embedding.beta)
            # hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = self.dropout(hidden_states)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # decoder layers
        all_hidden_states = ()
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.decode_layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.decode_layers)} layers, but it is for {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.decode_layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.is_train and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            # if self.is_train and (dropout_probability < self.layerdrop):
            #     continue

        # add hidden states from the last decoder layer
        all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states]
            if v is not None
        )


class TrOCRForCausalLM(Module):
    def __init__(self, config):
        super().__init__()
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        self.model = TrOCRDecoder(config)

        self.output_projection = tlx.nn.Linear(in_features=config.hidden_size,
                                               out_features=config.vocab_size, b_init=None)

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        logits = self.output_projection(outputs[0])

        return logits

