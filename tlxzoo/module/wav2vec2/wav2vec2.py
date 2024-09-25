import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn.core import Module
import importlib
from tensorlayerx.nn import ModuleList

LARGE_NEGATIVE = -1e8


def shape_list(x):
    return tlx.get_tensor_shape(x)


def top_k_indices(tensor, num):
    index = tlx.argsort(tensor, descending=True)
    indices = index[..., :num]
    return indices


def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tlx.log(tlx.random_uniform(shape_list(distribution), 0, 1))
    distribution = distribution + z
    indices = top_k_indices(distribution, num_samples)
    return indices


def _scatter_values_on_batch_indices(values, batch_indices, output_shape):
    indices_shape = shape_list(batch_indices)
    # broadcast batch dim to indices_shape
    casted_batch_dims = tlx.expand_dims(tlx.arange(indices_shape[0]), axis=-1)
    broad_casted_batch_dims = tlx.reshape(
        tlx.tile(casted_batch_dims, [1] + indices_shape[1:]), [1, -1]
    )
    # transform batch_indices to pair_indices
    pair_indices = tlx.transpose(tlx.concat([broad_casted_batch_dims, tlx.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tlx.ops.scatter_nd(pair_indices, tlx.reshape(values, [-1]), output_shape)


def _compute_mask_indices(
        shape,
        mask_prob,
        mask_length,
        min_masks=0,
):
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )
    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + tlx.random_uniform((1,)))
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = tlx.zeros((batch_size, sequence_length), dtype=tlx.int32)

    # uniform distribution to sample from, make sure that offset samples are < sequence_length
    uniform_dist = tlx.ones((batch_size, sequence_length - (mask_length - 1)))

    # get random indices to mask
    spec_aug_mask_idxs = _sample_without_replacement(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = tlx.expand_dims(spec_aug_mask_idxs, -1)
    spec_aug_mask_idxs = tlx.tile(spec_aug_mask_idxs, (1, 1, mask_length))
    spec_aug_mask_idxs = tlx.reshape(spec_aug_mask_idxs, (batch_size, num_masked_spans * mask_length))

    offsets = tlx.arange(mask_length)[None, None, :]
    offsets = tlx.tile(offsets, (batch_size, num_masked_spans, 1))
    offsets = tlx.reshape(offsets, (batch_size, num_masked_spans * mask_length))

    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = _scatter_values_on_batch_indices(
        tlx.ones_like(spec_aug_mask_idxs), spec_aug_mask_idxs, spec_aug_mask.shape
    )

    return spec_aug_mask


def _expand_mask(mask, tgt_len=None, past_key_values_length=0):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    src_len = shape_list(mask)[1]
    tgt_len = tgt_len if tgt_len is not None else src_len
    one_cst = tlx.constant(1.0)
    mask = tlx.cast(mask, dtype=one_cst.dtype)
    expanded_mask = tlx.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

    return (one_cst - expanded_mask) * LARGE_NEGATIVE


class Wav2Vec2GroupNorm(Module):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
            self,
            input_shape,
            name="",
            groups=32,
            axis=-1,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = self.str_to_init(beta_initializer)
        self.gamma_initializer = self.str_to_init(gamma_initializer)
        self.beta_regularizer = self.str_to_init(beta_regularizer)
        self.gamma_regularizer = self.str_to_init(gamma_regularizer)
        self.beta_constraint = self.str_to_init(beta_constraint)
        self.gamma_constraint = self.str_to_init(gamma_constraint)

        self._set_number_of_groups_for_instance_norm(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)

    def forward(self, inputs):
        input_shape = shape_list(inputs)
        tensor_input_shape = tlx.get_tensor_shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tlx.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tlx.stack(group_shape)
            reshaped_inputs = tlx.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = shape_list(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tlx.moments(reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)

        def batch_normalization(x, mean, variance, offset, scale, variance_epsilon):
            inv = tlx.rsqrt(variance + variance_epsilon)
            if scale is not None:
                inv *= scale
            return x * tlx.cast(inv, x.dtype) + tlx.cast(
                offset - mean * inv if offset is not None else -mean * inv, x.dtype)

        normalized_inputs = batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )

        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tlx.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tlx.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self._get_weights(
                var_name="gamma",
                shape=shape,
                init=self.gamma_initializer,
                trainable=True,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self._get_weights(
                var_name="beta",
                shape=shape,
                init=self.beta_initializer,
                trainable=True,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class Wav2Vec2WeightNormConv1D(tlx.nn.Conv1d):
    def __init__(self, filters, kernel_size, groups, explicit_padding, in_channels, name="", **kwargs):
        # in_channels = in_channels + explicit_padding * 2
        in_channels = in_channels // groups
        super(Wav2Vec2WeightNormConv1D, self).__init__(
            out_channels=filters,
            kernel_size=kernel_size,
            padding='valid',
            b_init='he_normal',
            in_channels=in_channels,
            name=name,
            **kwargs
        )
        self.groups = groups
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.kernel_norm_axes = tlx.constant([0, 1], dtype=tlx.int32)

        self.weight_v = tlx.Variable(tlx.transpose(self.filters), name=name + "/weight_v", trainable=True)
        del self.filters
        # self.weight_v = self.filters

        kernel_norm = tlx.sqrt(tlx.reduce_sum(tlx.square(self.weight_v), axis=self.kernel_norm_axes))

        self.weight_g = self._get_weights(var_name="weight_g",
                                          shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1),
                                          init=lambda shape: kernel_norm[:, None, None],
                                          trainable=True,
                                          )

    def _normalize_kernel(self):
        """Generate normalized weights."""
        kernel = tlx.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tlx.transpose(self.weight_g)
        self.filters = tlx.transpose(kernel)

    def forward(self, inputs):
        self._normalize_kernel()

        padded_inputs = tlx.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().forward(padded_inputs)

        return output


class Wav2Vec2NoLayerNormConvLayer(Module):
    def __init__(self, config, layer_id=0, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tlx.nn.Conv1d(
            out_channels=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            padding="valid",
            name=name + "/conv",
            in_channels=self.in_conv_dim,
            b_init='constant' if config.conv_bias else None,
        )
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(Module):
    def __init__(self, config, layer_id=0, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tlx.nn.Conv1d(out_channels=self.out_conv_dim, kernel_size=config.conv_kernel[layer_id],
                                  stride=config.conv_stride[layer_id], padding="valid",
                                  b_init='constant' if config.conv_bias else None,
                                  in_channels=self.in_conv_dim,
                                  name=name + "/conv",
                                  )

        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=self.out_conv_dim,
                                           name=name + "/layer_norm", epsilon=config.layer_norm_eps)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)
        hidden_states = self.activation(hidden_states)
        return hidden_states


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


class Wav2Vec2GroupNormConvLayer(Module):
    def __init__(self, config, layer_id=0, name="", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.in_conv_dim = config.conv_dim[layer_id] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tlx.nn.Conv1d(
            out_channels=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            padding='valid',
            b_init='constant' if config.conv_bias else None,
            name=name + "/conv",
            in_channels=1,
        )
        self.activation = get_activation(config.feat_extract_activation)
        self.layer_norm = Wav2Vec2GroupNorm(input_shape=[self.out_conv_dim],
                                            groups=self.out_conv_dim, epsilon=config.layer_norm_eps,
                                            name=name + "/layer_norm"
                                            )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(**kwargs)
        self.conv = Wav2Vec2WeightNormConv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            groups=config.num_conv_pos_embedding_groups,
            explicit_padding=config.num_conv_pos_embeddings // 2,
            name=name + "/conv",
            in_channels=config.hidden_size,
        )
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_activation(config.feat_extract_activation)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2SamePadLayer(Module):
    def __init__(self, num_conv_pos_embeddings, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, : -self.num_pad_remove, :]
        return hidden_states


class Wav2Vec2FeatureExtractor(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0, name=name + f"/conv_layers.{0}")]
            other_layes = [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1, name=name + f"/conv_layers.{i + 1}")
                for i in range(config.num_feat_extract_layers - 1)
            ]
            conv_layers = conv_layers + other_layes
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i, name=name + f"/conv_layers.{i}")
                for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = tlx.nn.Sequential(conv_layers)

    def forward(self, input_values):
        hidden_states = tlx.expand_dims(input_values, -1)
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)
        return hidden_states


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


class Wav2Vec2FeatureProjection(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=config.conv_dim[-1],
                                           epsilon=config.layer_norm_eps, name=name + "/layer_norm")
        # import tensorflow as tf
        # self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        self.layer_norm.build([None, None, config.conv_dim[-1]])
        self.projection = tlx.nn.Linear(
            config.hidden_size,
            W_init=tlx.initializers.TruncatedNormal(stddev=config.initializer_range),
            b_init="zeros",
            in_features=config.conv_dim[-1],
            name=name + "/projection"
        )
        self.dropout = tlx.nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.bart.modeling_tf_bart.TFBartAttention with TFBart->TFWav2Vec2
class Wav2Vec2Attention(Module):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            is_decoder=False,
            bias=True,
            name="",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = tlx.nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = tlx.nn.Linear(embed_dim, in_features=embed_dim, b_init="constant" if bias else None,
                                    name=name + "/k_proj")
        self.q_proj = tlx.nn.Linear(embed_dim, in_features=embed_dim, b_init="constant" if bias else None,
                                    name=name + "/q_proj")
        self.v_proj = tlx.nn.Linear(embed_dim, in_features=embed_dim, b_init="constant" if bias else None,
                                    name=name + "/v_proj")
        self.out_proj = tlx.nn.Linear(embed_dim, in_features=embed_dim, b_init="constant" if bias else None,
                                      name=name + "/out_proj")

    def _shape(self, tensor, seq_len, bsz):
        return tlx.transpose(tlx.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

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
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = tlx.reshape(self._shape(query_states, tgt_len, bsz), proj_shape)
        key_states = tlx.reshape(key_states, proj_shape)
        value_states = tlx.reshape(value_states, proj_shape)

        src_len = shape_list(key_states)[1]
        attn_weights = tlx.matmul(query_states, key_states, transpose_b=True)

        if attention_mask is not None:
            attention_mask = tlx.cast(attention_mask, dtype=attn_weights.dtype)
            attn_weights = tlx.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = tlx.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_weights = tlx.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            attn_weights = tlx.reshape(layer_head_mask, (1, -1, 1, 1)) * tlx.reshape(
                attn_weights, (bsz, self.num_heads, tgt_len, src_len)
            )
            attn_weights = tlx.reshape(attn_weights, (bsz * self.num_heads, tgt_len, src_len))

        attn_probs = self.dropout(attn_weights)
        attn_output = tlx.matmul(attn_probs, value_states)

        attn_output = tlx.transpose(
            tlx.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim)), (0, 2, 1, 3)
        )
        attn_output = tlx.reshape(attn_output, (bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output)
        attn_weights = tlx.reshape(attn_weights, (bsz, self.num_heads, tgt_len, src_len))

        return attn_output, attn_weights, past_key_value


def get_initializer(initializer_range: float = 0.02):
    return tlx.initializers.TruncatedNormal(stddev=initializer_range)


class Wav2Vec2FeedForward(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.intermediate_dropout = tlx.nn.Dropout(config.activation_dropout)

        self.intermediate_dense = tlx.nn.Linear(config.intermediate_size,
                                                in_features=config.hidden_size,
                                                W_init=get_initializer(config.initializer_range),
                                                b_init="zeros",
                                                name=name + "/intermediate_dense")
        self.intermediate_act_fn = get_activation(config.hidden_act)

        self.output_dense = tlx.nn.Linear(config.hidden_size,
                                          in_features=config.intermediate_size,
                                          W_init=get_initializer(config.initializer_range),
                                          b_init="zeros",
                                          name=name + "/output_dense"
                                          )
        self.output_dropout = tlx.nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2EncoderLayer(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name=name + "/attention",
        )
        self.dropout = tlx.nn.Dropout(config.hidden_dropout)
        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                           epsilon=config.layer_norm_eps, name=name + "/layer_norm")
        self.layer_norm.build([None, None, config.hidden_size])
        self.feed_forward = Wav2Vec2FeedForward(config, name=name + "/feed_forward")
        self.final_layer_norm = tlx.nn.LayerNorm(
            normalized_shape=config.hidden_size,
            epsilon=config.layer_norm_eps, name=name + "/final_layer_norm"
        )
        self.final_layer_norm.build([None, None, config.hidden_size])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.final_layer_norm.layernorm, self.final_layer_norm.gamma,
                           self.final_layer_norm.beta)

        outputs = (hidden_states,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            name=name + "/attention",
        )
        self.dropout = tlx.nn.Dropout(config.hidden_dropout)
        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                           epsilon=config.layer_norm_eps, name=name + "/layer_norm")
        self.feed_forward = Wav2Vec2FeedForward(config, name=name + "/feed_forward")
        self.final_layer_norm = tlx.nn.LayerNorm(
            normalized_shape=config.hidden_size,
            epsilon=config.layer_norm_eps, name=name + "/final_layer_norm"
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        attn_residual = hidden_states
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = ln(hidden_states, self.final_layer_norm.layernorm, self.final_layer_norm.gamma,
                           self.final_layer_norm.beta)
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        outputs = (hidden_states,)

        return outputs


class Wav2Vec2Encoder(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config, name=name + "/pos_conv_embed")
        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                           epsilon=config.layer_norm_eps, name=name + "/layer_norm")
        self.layer_norm.build([None, None, config.hidden_size])
        self.dropout = tlx.nn.Dropout(config.hidden_dropout)
        self.layer = ModuleList(
            [Wav2Vec2EncoderLayer(config, name=name + f"/layers.{i}") for i in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        all_hidden_states = []

        if attention_mask is not None:
            hidden_states = hidden_states * tlx.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)
        hidden_states = self.dropout(hidden_states)

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + [hidden_states, ]

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if self.is_train and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)


class Wav2Vec2EncoderStableLayerNorm(Module):
    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config, name=name + "/pos_conv_embed")
        self.layer_norm = tlx.nn.LayerNorm(normalized_shape=config.hidden_size,
                                           epsilon=config.layer_norm_eps, name=name + "/layer_norm")
        self.layer_norm.build([None, None, config.hidden_size])
        self.dropout = tlx.nn.Dropout(config.hidden_dropout)
        self.layer = [
            Wav2Vec2EncoderLayerStableLayerNorm(config, name=name + f"/layers.{i}") for i in
            range(config.num_hidden_layers)
        ]

    def forward(
            self,
            hidden_states,
            attention_mask=None,
    ):
        all_hidden_states = None

        if attention_mask is not None:
            hidden_states = hidden_states * tlx.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for i, layer_module in enumerate(self.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)
            if self.is_train and (dropout_probability < self.config.layerdrop):  # skip the layer
                continue

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = ln(hidden_states, self.layer_norm.layernorm, self.layer_norm.gamma, self.layer_norm.beta)

        return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)


class Wav2Vec2MainLayer(Module):

    def __init__(self, config, name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureExtractor(config, name=name + "/feature_extractor")
        self.feature_projection = Wav2Vec2FeatureProjection(config, name=name + "/feature_projection")

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config, name=name + "/encoder")
        else:
            self.encoder = Wav2Vec2Encoder(config, name=name + "/encoder")

        self.masked_spec_embed = self._get_weights(var_name="masked_spec_embed", shape=(self.config.hidden_size,),
                                                   init=self.str_to_init("random_uniform"), trainable=True)

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _mask_hidden_states(self, hidden_states, mask_time_indices=None):
        """
        Masks extracted features along time axis and/or along feature axis according to `SpecAugment
        <https://arxiv.org/abs/1904.08779>`__ .
        """
        batch_size, sequence_length, hidden_size = shape_list(hidden_states)

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states = tlx.where(
                tlx.cast(mask_time_indices[:, :, None], tlx.bool),
                self.masked_spec_embed[None, None, :],
                hidden_states,
            )

        elif self.config.mask_time_prob > 0:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                min_masks=2,
            )
            hidden_states = tlx.where(
                tlx.cast(mask_time_indices[:, :, None], tlx.bool),
                self.masked_spec_embed[None, None, :],
                hidden_states,
            )

        # apply SpecAugment along feature axis
        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
            )
            hidden_states = tlx.where(mask_feature_indices[:, None, :], hidden_states, 0)

        return hidden_states

    def forward(
            self,
            input_values,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        hidden_states = self.feature_extractor(
            tlx.cast(input_values, tlx.float32)
        )

        if attention_mask is not None:
            # compute real output lengths according to convolution formula
            output_lengths = self._get_feat_extract_output_lengths(tlx.reduce_sum(attention_mask, -1))

            attention_masks = []
            maxlen = shape_list(hidden_states)[1]
            for length in output_lengths:
                attention_mask = tlx.concat([tlx.ones(length), tlx.zeros(maxlen - length)], axis=0)
                attention_masks.append(attention_mask)
            attention_mask = tlx.stack(attention_masks)
            attention_mask = tlx.cast(attention_mask, hidden_states.dtype)

        hidden_states = self.feature_projection(hidden_states)

        mask_time_indices = kwargs.get("mask_time_indices", None)
        if self.is_train:
            hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs[0]

        return hidden_states, encoder_outputs[1:]


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, item):
        return self.kwargs[item]


class Wav2Vec2(tlx.nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout=0.1,
                 activation_dropout=0.1,
                 attention_dropout=0.1,
                 feat_proj_dropout=0.1,
                 feat_quantizer_dropout=0.0,
                 layerdrop=0.1,
                 initializer_range=0.02,
                 layer_norm_eps=1e-5,
                 feat_extract_norm="group",
                 feat_extract_activation="gelu",
                 feat_extract_dropout=0.0,
                 hidden_dropout_prob=0.1,
                 conv_dim=(512, 512, 512, 512, 512, 512, 512),
                 conv_stride=(5, 2, 2, 2, 2, 2, 2),
                 conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                 conv_bias=False,
                 num_conv_pos_embeddings=128,
                 num_conv_pos_embedding_groups=16,
                 do_stable_layer_norm=False,
                 apply_spec_augment=True,
                 mask_time_prob=0.05,
                 mask_time_length=10,
                 mask_feature_prob=0.0,
                 mask_feature_length=10,
                 num_codevectors_per_group=320,
                 num_codevector_groups=2,
                 contrastive_logits_temperature=0.1,
                 num_negatives=100,
                 codevector_dim=256,
                 proj_codevector_dim=256,
                 use_weighted_layer_sum=False,
                 classifier_proj_size=256,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 vocab_size=32,
                 final_dropout=0.1,
                 diversity_loss_weight=0.1,
                 ctc_loss_reduction="sum",
                 ctc_zero_infinity=False,
                 name="wav2vec2", *inputs, **kwargs):
        """
        :param hidden_size: (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        :param num_hidden_layers: (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size: (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        :param hidden_act: (:obj:`str`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        :param hidden_dropout:  (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        :param activation_dropout: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the activation probabilities.
        :param attention_dropout: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        :param feat_proj_dropout: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the feature extractor probabilities.
        :param feat_quantizer_dropout: (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for quantized feature extractor states.
        :param initializer_range: (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        :param layer_norm_eps: (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        :param feat_extract_norm: str
            The norm to be applied to 1D convolutional layers in feature extractor.
        :param feat_extract_activation: (:obj:`str, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor.
        :param conv_dim: (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature extractor.
        :param conv_stride: (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature extractor. The length
            of `conv_stride` defines the number of convolutional layers and has to match the the length of `conv_dim`.
        :param conv_kernel: (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature extractor. The
            length of `conv_kernel` defines the number of convolutional layers and has to match the the length of
            `conv_dim`.
        :param conv_bias: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the 1D convolutional layers have a bias.
        :param num_conv_pos_embeddings: (:obj:`int`, `optional`, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        :param num_conv_pos_embedding_groups: (:obj:`int`, `optional`, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        :param do_stable_layer_norm: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to apply `stable` layer norm architecture of the Transformer encoder. ``do_stable_layer_norm is
            True`` corresponds to applying layer norm before the attention layer, whereas ``do_stable_layer_norm is
            False`` corresponds to applying layer norm after the attention layer.
        :param apply_spec_augment: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature extractor.
        :param mask_time_prob: (:obj:`float`, `optional`, defaults to 0.05):
            Propability of each feature vector along the time axis to be chosen as the start of the vector span to be
            masked.
        :param mask_time_length: (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the time axis.
        :param mask_feature_prob: (:obj:`float`, `optional`, defaults to 0.0):
            Propability of each feature vector along the feature axis to be chosen as the start of the vector span to
            be masked.
        :param mask_feature_length: (:obj:`int`, `optional`, defaults to 10):
            Length of vector span along the feature axis.
        :param num_codevectors_per_group: (:obj:`int`, `optional`, defaults to 320):
            Number of entries in each quantization codebook (group).
        :param num_codevector_groups: (:obj:`int`, `optional`, defaults to 2):
            Number of codevector groups for product codevector quantization.
        :param contrastive_logits_temperature: (:obj:`float`, `optional`, defaults to 0.1):
            The temperature `kappa` in the contrastive loss.
        :param num_negatives: (:obj:`int`, `optional`, defaults to 100):
            Number of negative samples for the contrastive loss.
        :param codevector_dim: (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the quantized feature vectors.
        :param proj_codevector_dim: (:obj:`int`, `optional`, defaults to 256):
            Dimensionality of the final projection of both the quantized and the transformer features.
        :param vocab_size: (:obj:`int`, `optional`, defaults to 32):
            Vocabulary size of the Wav2Vec2 model.
        :param diversity_loss_weight: (:obj:`int`, `optional`, defaults to 0.1):
            The weight of the codebook diversity loss component.
        :param ctc_loss_reduction: (:obj:`str`, `optional`, defaults to :obj:`"sum"`):
            Specifies the reduction to apply to the output of ctcloss.
        :param ctc_zero_infinity: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to zero infinite losses and the associated gradients of ctcloss.
        """
        super().__init__(name=name, *inputs, **kwargs)

        num_feat_extract_layers = len(conv_dim)

        config = Config(hidden_size=hidden_size,
                        num_feat_extract_layers=num_feat_extract_layers,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=intermediate_size,
                        hidden_act=hidden_act,
                        hidden_dropout=hidden_dropout,
                        activation_dropout=activation_dropout,
                        attention_dropout=attention_dropout,
                        feat_proj_dropout=feat_proj_dropout,
                        feat_quantizer_dropout=feat_quantizer_dropout,
                        layerdrop=layerdrop,
                        initializer_range=initializer_range,
                        layer_norm_eps=layer_norm_eps,
                        feat_extract_norm=feat_extract_norm,
                        feat_extract_activation=feat_extract_activation,
                        feat_extract_dropout=feat_extract_dropout,
                        hidden_dropout_prob=hidden_dropout_prob,
                        conv_dim=conv_dim,
                        conv_stride=conv_stride,
                        conv_kernel=conv_kernel,
                        conv_bias=conv_bias,
                        num_conv_pos_embeddings=num_conv_pos_embeddings,
                        num_conv_pos_embedding_groups=num_conv_pos_embedding_groups,
                        do_stable_layer_norm=do_stable_layer_norm,
                        apply_spec_augment=apply_spec_augment,
                        mask_time_prob=mask_time_prob,
                        mask_time_length=mask_time_length,
                        mask_feature_prob=mask_feature_prob,
                        mask_feature_length=mask_feature_length,
                        num_codevectors_per_group=num_codevectors_per_group,
                        num_codevector_groups=num_codevector_groups,
                        contrastive_logits_temperature=contrastive_logits_temperature,
                        num_negatives=num_negatives,
                        codevector_dim=codevector_dim,
                        proj_codevector_dim=proj_codevector_dim,
                        use_weighted_layer_sum=use_weighted_layer_sum,
                        classifier_proj_size=classifier_proj_size,
                        pad_token_id=pad_token_id,
                        bos_token_id=bos_token_id,
                        eos_token_id=eos_token_id)
        self.wav2vec2 = Wav2Vec2MainLayer(config, name=name)

        self.dropout = tlx.nn.Dropout(final_dropout)
        self.lm_head = tlx.nn.Linear(vocab_size, in_features=hidden_size, name="lm_head")

        self.diversity_loss_weight = diversity_loss_weight
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def forward(
            self,
            inputs,
            pixel_mask=None,
    ):
        outputs = self.wav2vec2(
            inputs,
            attention_mask=pixel_mask,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.conv_kernel, self.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def loss_fn(self, logits, labels, pixel_mask):
        if tlx.reduce_max(labels) >= self.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.vocab_size}")

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
            blank_index=self.pad_token_id,
            logits_time_major=False,
        )

        if self.ctc_loss_reduction == "sum":
            loss = tlx.reduce_sum(loss)
        if self.ctc_loss_reduction == "mean":
            loss = tlx.reduce_mean(loss)
        return loss

