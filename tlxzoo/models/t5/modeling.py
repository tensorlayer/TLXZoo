"""
# Reference:
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
"""
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module
from tensorlayerx.nn.core import LayerList
import numpy as np
import math


class T5Embedding(tlx.nn.Embedding):
    def forward(self, inputs, mode="emb"):
        if mode == "emb":
            outputs = self.embedding_lookup(params=self.embeddings, ids=inputs)
            return outputs
        else:
            first_dims = shape_list(inputs)[:-1]
            x = tlx.reshape(inputs, [-1, self.embedding_size])
            logits = tlx.matmul(x, self.embeddings, transpose_b=True)

            return tlx.reshape(logits, first_dims + [self.vocabulary_size])


class T5LayerNorm(Module):
    def __init__(self, d_model, epsilon=1e-6, name="layer_norm", **kwargs):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.variance_epsilon = epsilon
        self.weight = self._get_weights("weight", shape=(d_model,), init=tlx.initializers.ones())

    def __repr__(self):
        s = (
            '{classname}(d_model={d_model}, variance_epsilon={variance_epsilon}'
        )
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, hidden_states):
        variance = tlx.reduce_mean(tlx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * tlx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5DenseReluDense(Module):
    def __init__(self, config, name="DenseReluDense", **kwargs):
        super().__init__(name=name, **kwargs)
        wi_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model ** -0.5)
        )
        wo_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff ** -0.5)
        )
        self.d_ff = config.d_ff
        self.d_model = config.d_model
        self.wi = tlx.nn.layers.Dense(
            config.d_ff, in_channels=config.d_model, b_init=None, name=name+"/wi", W_init=wi_initializer
        )  # Update init weights as in flax
        self.wo = tlx.nn.layers.Dense(
            config.d_model, in_channels=config.d_ff, b_init=None, name=name+"/wo", W_init=wo_initializer
        )  # Update init weights as in flax
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name + "/dropout")
        self.act = tlx.relu
        self.dropout_rate = config.dropout_rate

    def __repr__(self):
        s = (
            '{classname}(d_ff={d_ff}, d_model={d_model}, dropout_rate={dropout_rate}'
        )
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5GatedGeluDense(Module):
    def __init__(self, config, name="GatedGeluDense", **kwargs):
        super().__init__(name=name, **kwargs)
        wi_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model ** -0.5)
        )
        wo_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff ** -0.5)
        )
        self.wi_0 = tlx.nn.layers.Dense(
            config.d_ff, in_channels=config.d_model, b_init=None, name=name+"/wi_0", W_init=wi_initializer
        )  # Update init weights as in flax
        self.wi_1 = tlx.nn.layers.Dense(
            config.d_ff, in_channels=config.d_model, b_init=None, name=name+"/wi_1", W_init=wi_initializer
        )  # Update init weights as in flax
        self.wo = tlx.nn.layers.Dense(
            config.d_model, in_channels=config.d_ff, b_init=None, name=name+"/wo", W_init=wo_initializer
        )  # Update init weights as in flax
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name + "/dropout")
        self.act = tlx.gelu

    def __repr__(self):
        s = (
            '{classname}(d_ff={d_ff}, d_model={d_model}, act={act}, dropout_rate={dropout_rate}'
        )
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(Module):
    def __init__(self, config, name="layer_", **kwargs):
        super().__init__(name=name, **kwargs)
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config, name=name+"/DenseReluDense")
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5GatedGeluDense(config, name=name+"/DenseReluDense")
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )
        self.layer_norm = T5LayerNorm(d_model=config.d_model, epsilon=config.layer_norm_epsilon, name=name+"/layer_norm")
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name+"/dropout")
        self.dropout_rate = config.dropout_rate
        self.d_model = config.d_model
        self.d_ff = config.d_ff

    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states)
        dense_output = self.DenseReluDense(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(dense_output)
        return hidden_states


def shape_list(tensor):
    return tensor.shape.as_list()


class T5Attention(Module):

    def __init__(self, config, has_relative_attention_bias=False, name="SelfAttention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.is_decoder = config.is_decoder
        self.use_cache = config.use_cache
        self.has_relative_attention_bias = has_relative_attention_bias
        # self.output_attentions = config.output_attentions

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        q_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        k_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim ** -0.5)
        )
        v_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim ** -0.5)
        )
        o_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim ** -0.5)
        )
        self.relative_attention_bias_initializer = tlx.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim ** -0.5)
        )

        self.q = tlx.nn.layers.Dense(
            self.inner_dim, in_channels=self.inner_dim, b_init=None, name=name+"/q", W_init=q_initializer
        )  # Update init weights as in flax
        self.k = tlx.nn.layers.Dense(
            self.inner_dim, in_channels=self.inner_dim, b_init=None, name=name+"/k", W_init=k_initializer
        )  # Update init weights as in flax
        self.v = tlx.nn.layers.Dense(
            self.inner_dim, in_channels=self.inner_dim, b_init=None, name=name+"/v", W_init=v_initializer
        )  # Update init weights as in flax
        self.o = tlx.nn.layers.Dense(
            self.d_model, in_channels=self.d_model, b_init=None, name=name+"/o", W_init=o_initializer
        )  # Update init weights as in flax
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name + "/dropout")
        if self.has_relative_attention_bias:
            self.relative_attention_bias = self._get_weights(
                "relative_attention_bias/embeddings", shape=[self.relative_attention_num_buckets, self.n_heads],
                init=self.relative_attention_bias_initializer
            )

        self.pruned_heads = set()

    def __repr__(self):
        s = (
            '{classname}(inner_dim={inner_dim}, n_heads={n_heads}, '
            'has_relative_attention_bias={has_relative_attention_bias}, is_decoder={is_decoder}'
        )
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def prune_heads(self, heads):
        raise NotImplementedError

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        import tensorflow as tf
        relative_buckets = 0
        #        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                    tlx.cast(tlx.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tlx.abs(relative_position)
        else:
            relative_position = -tlx.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = tlx.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tlx.cast(
            tlx.log(relative_position / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tlx.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tlx.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = tlx.range(query_length)[:, None]
        memory_position = tlx.range(key_length)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = tlx.gather(
            self.relative_attention_bias, relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = tlx.expand_dims(
            tlx.transpose(values, [2, 0, 1]), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, query_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = shape_list(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += shape_list(past_key_value[0])[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else shape_list(key_value_states)[1]

        def shape(hidden_states):
            """projection"""
            return tlx.transpose(
                tlx.reshape(hidden_states, (batch_size, -1, self.n_heads, self.key_value_proj_dim)), perm=(0, 2, 1, 3)
            )

        def unshape(hidden_states):
            """compute context"""
            return tlx.reshape(tlx.transpose(hidden_states, perm=(0, 2, 1, 3)), (batch_size, -1, self.inner_dim))

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = tlx.concat([past_key_value, hidden_states], axis=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, query_length, dim_per_head)

        # get key/value
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # to cope with keras serialization
        if self.is_decoder and use_cache:
            present_key_value_state = (key_states, value_states)
        else:
            present_key_value_state = None

        import tensorflow as tf
        scores = tf.einsum(
            "bnqd,bnkd->bnqk", query_states, key_states
        )  # (batch_size, n_heads, query_length, key_length)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tlx.zeros((1, self.n_heads, real_seq_length, key_length))
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                position_bias = tlx.cast(position_bias, dtype=mask.dtype)
                position_bias = position_bias + mask  # (batch_size, n_heads, query_length, key_length)

        scores += position_bias
        weights = tlx.softmax(scores, axis=-1)  # (batch_size, n_heads, query_length, key_length)
        weights = self.dropout(weights)  # (batch_size, n_heads, query_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            weights = tlx.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attn_output = tlx.matmul(weights, value_states)  # (batch_size, n_heads, query_length, dim_per_head)

        attn_output = self.o(unshape(attn_output))

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (weights,)

        return outputs


class T5LayerSelfAttention(Module):
    def __init__(self, config, has_relative_attention_bias=False, name="layer_", **kwargs):
        super().__init__(name=name, **kwargs)
        self.SelfAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name=name+"/SelfAttention",
        )
        self.layer_norm = T5LayerNorm(d_model=config.d_model, epsilon=config.layer_norm_epsilon, name=name+"/layer_norm")
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name+"/dropout")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(Module):
    def __init__(self, config, name="layer_", **kwargs):
        super().__init__(name=name, **kwargs)
        self.EncDecAttention = T5Attention(
            config,
            has_relative_attention_bias=False,
            name=name+"/EncDecAttention",
        )
        self.layer_norm = T5LayerNorm(d_model=config.d_model, epsilon=config.layer_norm_epsilon, name=name+"/layer_norm")
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name+"/dropout")

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(Module):
    def __init__(self, config, has_relative_attention_bias=False, name="block_", **kwargs):
        super().__init__(name=name, **kwargs)
        self.is_decoder = config.is_decoder

        self.self_attn = T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                name=name+"/layer_._0",
            )
        ffn_name = name+f"/layer_._1"

        if self.is_decoder:
            self.cross_attn = T5LayerCrossAttention(
                    config,
                    name=name+"/layer_._1",
                )
            ffn_name = name+f"/layer_._2"

        self.ffn = T5LayerFF(config, name=ffn_name)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention' if expected_num_past_key_values == 4 else ''}."
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = shape_list(present_key_value_state[0])[2]
            else:
                query_length = None

            cross_attention_outputs = self.cross_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.ffn(hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5MainLayer(Module):

    def __init__(self, config, embed_tokens=None, name="main_layer", **kwargs):
        super().__init__(name=name, **kwargs)

        self.config = config
        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        self.block = LayerList([
            T5Block(config, has_relative_attention_bias=bool(i == 0), name=name+f"/block_._{i}")
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(d_model=config.d_model,
                                            epsilon=config.layer_norm_epsilon, name=name+"/final_layer_norm")
        self.dropout = tlx.nn.layers.Dropout(float(1.0 - config.dropout_rate), name=name+"/dropout")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tlx.reshape(input_ids, (-1, input_shape[-1]))
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            shape_list(past_key_values[0][0])[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if attention_mask is None:
            attention_mask = tlx.ones((batch_size, mask_seq_length))
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = shape_list(encoder_hidden_states)[1]
            encoder_attention_mask = tlx.ones((batch_size, encoder_seq_length))

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        attention_mask = tlx.cast(attention_mask, dtype=inputs_embeds.dtype)
        num_dims_attention_mask = len(shape_list(attention_mask))
        if num_dims_attention_mask == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif num_dims_attention_mask == 2:
            # Provided a padding mask of dimensions [batch_size, mask_seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            if self.is_decoder:
                seq_ids = tlx.range(mask_seq_length)
                causal_mask = tlx.less_equal(
                    tlx.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                    seq_ids[None, :, None],
                )
                causal_mask = tlx.cast(causal_mask, dtype=attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                if past_key_values[0] is not None:
                    extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and  -1e9 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = tf.math.equal(extended_attention_mask,
        #                                         tf.transpose(extended_attention_mask, perm=(-1, -2)))

        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tlx.cast(
                encoder_attention_mask, dtype=extended_attention_mask.dtype
            )
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        present_key_value_states = () if use_cache and self.is_decoder else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for idx, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                encoder_layer_head_mask=encoder_head_mask[idx]
                if encoder_head_mask is not None
                else None,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, past_key_values, (self-attention weights),
            # (self-attention position bias), (cross-attention position bias), (cross-attention weights),
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if present_key_value_state is not None and use_cache and self.is_decoder:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # need to check if is decoder here as well for special cases when using keras compile
        if use_cache and self.is_decoder:
            outputs = outputs + (present_key_value_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)