import collections.abc
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

logger = logging


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def get_initializer(initializer_range: float = 0.02):
    return tlx.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(x):
    return tlx.get_tensor_shape(x)


def ln(input, layernorm, gamma, beta):
    input_dtype = input.dtype
    if input_dtype in ('float16', 'bfloat16'):
        input = tlx.cast(input, tlx.float32)
    mean, var = tlx.moments(input, layernorm.axis, keepdims=True)
    scale, offset = layernorm._broadcast(gamma), layernorm._broadcast(beta)

    inv = tlx.rsqrt(var + layernorm.eps)
    if scale is not None:
        inv *= scale

    a = tlx.cast(inv, input.dtype)
    b = tlx.cast(offset - mean * inv if offset is not None else -mean * inv, input.dtype)

    output = a * input + b
    output = tlx.cast(output, input_dtype)
    return output


class ViTEmbeddings(Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.patch_embeddings = PatchEmbeddings(config, name=name+"/patch_embeddings")
        self.dropout = tlx.nn.Dropout(config.hidden_dropout_prob)
        self.config = config

        num_patches = self.patch_embeddings.num_patches

        self.cls_token = self._get_weights(
            shape=(1, 1, self.config.hidden_size), init=self.str_to_init("zeros"), trainable=True,
            var_name="cls_token"
        )
        self.position_embeddings = self._get_weights(
            shape=(1, num_patches + 1, self.config.hidden_size),
            init=self.str_to_init("zeros"),
            trainable=True,
            var_name="position_embeddings",
        )

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, seq_len, dim = shape_list(embeddings)
        npatch = seq_len - 1

        _, N, _ = shape_list(self.position_embeddings)
        N -= 1

        if npatch == N and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        patch_pos_embed = tlx.resize(
            tlx.reshape(patch_pos_embed, shape=(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)),
            output_size=(h0, w0),
            method="bicubic",
            antialias=False
        )

        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tlx.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tlx.concat([class_pos_embed, patch_pos_embed], axis=1)

    def forward(
            self, pixel_values, interpolate_pos_encoding=False
    ):
        batch_size, num_channels, height, width = shape_list(pixel_values)
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tlx.tile(self.cls_token, [batch_size, 1, 1])
        embeddings = tlx.concat([cls_tokens, embeddings], axis=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.config = config

        self.projection = tlx.nn.layers.Conv2d(out_channels=self.embed_dim,
                                               kernel_size=patch_size,
                                               stride=self.patch_size,
                                               padding="valid",
                                               data_format="channels_last",
                                               b_init="zeros",
                                               W_init=get_initializer(self.config.initializer_range),
                                               name=name+"/projection",
                                               in_channels=3,
                                               )

    def forward(
            self, pixel_values, interpolate_pos_encoding=False
    ):
        batch_size, num_channels, height, width = shape_list(pixel_values)
        if not interpolate_pos_encoding:
            if getattr(height, "numpy", None) and getattr(width, "numpy", None):
                if height != self.image_size[0] or width != self.image_size[1]:
                    raise ValueError(
                        f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                    )

        # shape = (batch_size, in_height, in_width, in_channels=num_channels)
        pixel_values = tlx.transpose(pixel_values, perm=(0, 2, 3, 1))

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (width // self.patch_size[1]) * (height // self.patch_size[0])
        x = tlx.reshape(tensor=projection, shape=(batch_size, num_patches, -1))

        return x


class ViTSelfAttention(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tlx.nn.Linear(
            out_features=self.all_head_size, W_init=get_initializer(config.initializer_range), name=name+"/query",
            in_features=config.hidden_size,
        )
        self.key = tlx.nn.Linear(
            out_features=self.all_head_size, W_init=get_initializer(config.initializer_range), name=name+"/key",
            in_features=config.hidden_size,
        )
        self.value = tlx.nn.Linear(
            out_features=self.all_head_size, W_init=get_initializer(config.initializer_range), name=name+"/value",
            in_features=config.hidden_size,
        )
        self.dropout = tlx.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor, batch_size):
        tensor = tlx.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        return tlx.transpose(tensor, perm=[0, 2, 1, 3])

    def forward(
            self,
            hidden_states,
            head_mask,
    ):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tlx.matmul(query_layer, key_layer, transpose_b=True)
        dk = tlx.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tlx.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tlx.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tlx.multiply(attention_probs, head_mask)

        attention_output = tlx.matmul(attention_probs, value_layer)
        attention_output = tlx.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tlx.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs)

        return outputs


class ViTSelfOutput(Module):

    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.nn.Linear(
            out_features=config.hidden_size, W_init=get_initializer(config.initializer_range), name=name+"/dense",
            in_features=config.hidden_size
        )
        self.dropout = tlx.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.self_attention = ViTSelfAttention(config, name=name+"/attention")
        self.dense_output = ViTSelfOutput(config, name=name+"/output")

    def forward(
            self,
            input_tensor,
            head_mask,
    ):
        self_outputs = self.self_attention(
            input_tensor, head_mask=head_mask
        )
        attention_output = self.dense_output(
            self_outputs[0], input_tensor=input_tensor
        )
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


def approximate_gelu_wrap(x):
    return tlx.gelu(x, approximate=True)


def mish(x):
    x = tlx.convert_to_tensor(x)

    return x * tlx.tanh(tlx.softplus(x))


def gelu_fast(x):
    x = tlx.convert_to_tensor(x)
    coeff1 = tlx.cast(0.044715, x.dtype)
    coeff2 = tlx.cast(0.7978845608, x.dtype)

    return 0.5 * x * (1.0 + tlx.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))


ACT2FN = {
    "gelu": tlx.gelu,
    "relu": tlx.relu,
    "gelu_new": approximate_gelu_wrap,
    "mish": mish,
    "tanh": tlx.tanh,
    "gelu_fast": gelu_fast,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


class ViTIntermediate(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.nn.Linear(
            out_features=config.intermediate_size, W_init=get_initializer(config.initializer_range), name=name+"/dense",
            in_features=config.hidden_size,
        )

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name,**kwargs)

        self.dense = tlx.nn.Linear(
            out_features=config.hidden_size, W_init=get_initializer(config.initializer_range), name=name+"/dense",
            in_features=config.intermediate_size
        )
        self.dropout = tlx.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTLayer(Module):

    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.attention = ViTAttention(config, name=name+"/attention")
        self.intermediate = ViTIntermediate(config, name=name+"/intermediate")
        self.vit_output = ViTOutput(config, name=name+"/output")

        self.layernorm_before = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                                 epsilon=config.layer_norm_eps, name=name+"/layernorm_before")
        self.layernorm_before.build([None, None, config.hidden_size])
        self.layernorm_after = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                                epsilon=config.layer_norm_eps, name=name+"/layernorm_after")
        self.layernorm_after.build([None, None, config.hidden_size])

    def forward(
            self,
            hidden_states,
            head_mask,
    ):
        input_tensor = ln(hidden_states, self.layernorm_before.layernorm,
                          self.layernorm_before.gamma, self.layernorm_before.beta)
        attention_outputs = self.attention(
            input_tensor,
            head_mask=head_mask,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = ln(hidden_states, self.layernorm_after.layernorm,
                          self.layernorm_after.gamma, self.layernorm_after.beta)

        intermediate_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(
            intermediate_output, input_tensor=hidden_states
        )
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them

        return outputs


class ViTEncoder(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.layer = tlx.nn.core.LayerList([ViTLayer(config, name=name+f"/layer_._{i}") for i in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            head_mask,
    ):
        all_hidden_states = ()

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                head_mask=head_mask[i],
            )
            hidden_states = layer_outputs[0]

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)


class ViTMainLayer(Module):

    def __init__(self, config, add_pooling_layer=True, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = config

        self.embeddings = ViTEmbeddings(config, name=name+"/embeddings")
        self.encoder = ViTEncoder(config, name=name+"/encoder")
        self.layernorm = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                          epsilon=config.layer_norm_eps, name=name+"/layernorm")
        self.layernorm.build([None, None, config.hidden_size])
        self.pooler = ViTPooler(config, name=name+"/pooler") if add_pooling_layer else None

    def forward(
            self,
            pixel_values,
            interpolate_pos_encoding: Optional[bool] = None,
            **kwargs,
    ):
        embedding_output = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = ln(sequence_output, self.layernorm.layernorm, self.layernorm.gamma, self.layernorm.beta)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class ViTModel(Module):
    def __init__(self, config, *inputs, add_pooling_layer=True, name="", **kwargs):
        super().__init__(name=name, *inputs, **kwargs)

        self.vit = ViTMainLayer(config, add_pooling_layer=add_pooling_layer, name="vit")

    def forward(
            self,
            pixel_values,
            interpolate_pos_encoding=None,
            **kwargs,
    ):
        outputs = self.vit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return outputs


class ViTPooler(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.nn.Linear(
            out_features=config.hidden_size,
            W_init=get_initializer(config.initializer_range),
            act="tanh",
            name=name+"/dense",
            in_features=config.hidden_size,
        )

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)

        return pooled_output
