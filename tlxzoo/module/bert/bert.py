import tensorlayerx as tlx
import numpy as np
import math

ACT2FN = {
    "gelu": tlx.gelu,
    "relu": tlx.relu,
    "swish": tlx.layers.Swish(),
    "silu": tlx.layers.Swish(),
    "mish": tlx.layers.Mish(),
    "tanh": tlx.tanh,
}


def get_initializer(initializer_range: float = 0.02):
    return tlx.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(tensor):
    return tlx.get_tensor_shape(tensor)


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


class BertEmbeddings(tlx.nn.Module):

    def __init__(self, vocab_size, type_vocab_size, hidden_size, max_position_embeddings, initializer_range,
                 layer_norm_eps, hidden_dropout_prob, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.LayerNorm = tlx.layers.LayerNorm(normalized_shape=hidden_size, epsilon=layer_norm_eps,
                                              name=name + "/LayerNorm")
        self.LayerNorm.build([None, None, hidden_size])
        self.dropout = tlx.layers.Dropout(hidden_dropout_prob)

        self.weight = self._get_weights(shape=[self.vocab_size, self.hidden_size], var_name="word_embeddings/weight",
                                        init=get_initializer(self.initializer_range))
        self.token_type_embeddings = self._get_weights(
            var_name="token_type_embeddings/embeddings",
            shape=[self.type_vocab_size, self.hidden_size],
            init=get_initializer(self.initializer_range),
        )
        self.position_embeddings = self._get_weights(
            var_name="position_embeddings/embeddings",
            shape=[self.max_position_embeddings, self.hidden_size],
            init=get_initializer(self.initializer_range),
        )

    def forward(
            self,
            inputs,
            position_ids,
            token_type_ids,
            inputs_embeds,
            past_key_values_length=0,
    ):
        if inputs is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if inputs is not None:
            inputs_embeds = tlx.gather(params=self.weight, indices=inputs)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tlx.zeros(input_shape)

        if position_ids is None:
            position_ids = tlx.expand_dims(
                tlx.arange(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        position_embeds = tlx.gather(params=self.position_embeddings, indices=position_ids)
        position_embeds = tlx.tile(input=position_embeds, multiples=(input_shape[0], 1, 1))
        token_type_embeds = tlx.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = tlx.add_n([inputs_embeds, position_embeds, token_type_embeds])
        # final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = ln(final_embeddings, self.LayerNorm.layernorm, self.LayerNorm.gamma, self.LayerNorm.beta)
        final_embeddings = self.dropout(inputs=final_embeddings)

        return final_embeddings


class BertSelfAttention(tlx.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob, is_decoder,
                 name="", **kwargs):
        super().__init__(name=name, **kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number "
                f"of attention heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tlx.layers.Linear(
            in_features=hidden_size, out_features=self.all_head_size, W_init=get_initializer(initializer_range),
            name=name + "/query"
        )
        self.key = tlx.layers.Linear(
            in_features=hidden_size, out_features=self.all_head_size, W_init=get_initializer(initializer_range),
            name=name + "/key"
        )
        self.value = tlx.layers.Linear(
            in_features=hidden_size, out_features=self.all_head_size, W_init=get_initializer(initializer_range),
            name=name + "/value"
        )
        self.dropout = tlx.layers.Dropout(attention_probs_dropout_prob)

        self.is_decoder = is_decoder

    def transpose_for_scores(self, tensor, batch_size):
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tlx.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tlx.transpose(tensor, perm=[0, 2, 1, 3])

    def forward(
            self,
            inputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
    ):
        hidden_states = inputs
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(inputs=encoder_hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=encoder_hidden_states), batch_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)
            key_layer = tlx.ops.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tlx.ops.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        if self.is_decoder:
            # if cross_attention save Tuple(tf.Tensor, tf.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(tf.Tensor, tf.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tlx.matmul(query_layer, key_layer, transpose_b=True)
        dk = tlx.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tlx.divide(attention_scores, dk)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TFBertModel call() function)
            attention_scores = tlx.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = tlx.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tlx.multiply(attention_probs, head_mask)

        attention_output = tlx.matmul(attention_probs, value_layer)
        attention_output = tlx.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tlx.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(tlx.nn.Module):
    def __init__(self, hidden_size, initializer_range, layer_norm_eps, hidden_dropout_prob, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.layers.Linear(
            in_features=hidden_size, out_features=hidden_size, W_init=get_initializer(initializer_range),
            name=name + "/dense"
        )
        self.LayerNorm = tlx.layers.LayerNorm(normalized_shape=hidden_size, epsilon=layer_norm_eps,
                                              name=name + "/LayerNorm")
        self.LayerNorm.build([None, None, hidden_size])
        self.dropout = tlx.layers.Dropout(hidden_dropout_prob)

    def forward(self, inputs, input_tensor):
        hidden_states = inputs
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states)
        # hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        hidden_states = ln(hidden_states + input_tensor, self.LayerNorm.layernorm, self.LayerNorm.gamma,
                           self.LayerNorm.beta)
        return hidden_states


class BertAttention(tlx.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob, is_decoder,
                 layer_norm_eps, hidden_dropout_prob, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.self_attention = BertSelfAttention(hidden_size, num_attention_heads, initializer_range,
                                                attention_probs_dropout_prob, is_decoder, name=name + "/self")
        self.dense_output = BertSelfOutput(hidden_size, initializer_range, layer_norm_eps, hidden_dropout_prob,
                                           name=name + "/output")

    def forward(
            self,
            inputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions: bool,
    ):
        input_tensor = inputs
        self_outputs = self.self_attention(
            inputs=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self.dense_output(
            inputs=self_outputs[0], input_tensor=input_tensor
        )
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class BertIntermediate(tlx.nn.Module):
    def __init__(self, hidden_size, intermediate_size, initializer_range, hidden_act, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.layers.Linear(
            in_features=hidden_size, out_features=intermediate_size,
            W_init=get_initializer(initializer_range), name=name + "/dense"
        )

        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, inputs):
        hidden_states = inputs
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(tlx.nn.Module):
    def __init__(self, hidden_size, intermediate_size, initializer_range, layer_norm_eps, hidden_dropout_prob,
                 name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.layers.Linear(
            in_features=intermediate_size, out_features=hidden_size, W_init=get_initializer(initializer_range),
            name=name + "/dense"
        )
        self.LayerNorm = tlx.layers.LayerNorm(hidden_size, epsilon=layer_norm_eps, name=name + "/LayerNorm")
        self.LayerNorm.build([None, None, hidden_size])
        self.dropout = tlx.layers.Dropout(hidden_dropout_prob)

    def forward(self, inputs, input_tensor):
        hidden_states = inputs
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states)
        # hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)
        hidden_states = ln(hidden_states + input_tensor, self.LayerNorm.layernorm, self.LayerNorm.gamma,
                           self.LayerNorm.beta)
        return hidden_states


class BertLayer(tlx.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob, is_decoder,
                 layer_norm_eps, hidden_dropout_prob, add_cross_attention, intermediate_size, hidden_act,
                 name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.attention = BertAttention(hidden_size, num_attention_heads, initializer_range,
                                       attention_probs_dropout_prob, is_decoder,
                                       layer_norm_eps, hidden_dropout_prob, name=name + "/attention")
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(hidden_size, num_attention_heads, initializer_range,
                                                attention_probs_dropout_prob, is_decoder,
                                                layer_norm_eps, hidden_dropout_prob, name=name + "/crossattention")
        self.intermediate = BertIntermediate(hidden_size, intermediate_size, initializer_range,
                                             hidden_act, name=name + "/intermediate")
        self.bert_output = BertOutput(hidden_size, intermediate_size, initializer_range, layer_norm_eps,
                                      hidden_dropout_prob, name=name + "/output")

    def forward(
            self,
            inputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions: bool,
    ):
        hidden_states = inputs
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            inputs=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers "
                    "by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(inputs=attention_output)
        layer_output = self.bert_output(
            inputs=intermediate_output, input_tensor=attention_output
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs


class BertEncoder(tlx.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob, is_decoder,
                 layer_norm_eps, hidden_dropout_prob, add_cross_attention, intermediate_size, hidden_act,
                 num_hidden_layers, name="", **kwargs):
        super().__init__(**kwargs)
        self.layer = tlx.nn.ModuleList([BertLayer(hidden_size, num_attention_heads, initializer_range,
                                                  attention_probs_dropout_prob,
                                                  is_decoder, layer_norm_eps, hidden_dropout_prob, add_cross_attention,
                                                  intermediate_size,
                                                  hidden_act, name=name + f"/layer_._{i}") for i in
                                        range(num_hidden_layers)])

    def forward(
            self,
            inputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions: bool,
            output_hidden_states: bool,
    ):
        hidden_states = inputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                inputs=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
        )


class BertPooler(tlx.nn.Module):
    def __init__(self, hidden_size, initializer_range, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = tlx.layers.Linear(
            out_features=hidden_size,
            W_init=get_initializer(initializer_range),
            act="tanh",
            name=name + "/dense",
            in_features=hidden_size,
        )

    def forward(self, inputs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states = inputs
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=inputs)

        return pooled_output


class BertMainLayer(tlx.nn.Module):

    def __init__(self, vocab_size, type_vocab_size, hidden_size, max_position_embeddings, initializer_range,
                 layer_norm_eps, hidden_dropout_prob, num_attention_heads, attention_probs_dropout_prob,
                 add_cross_attention, intermediate_size, hidden_act, num_hidden_layers, is_decoder, name="",
                 add_pooling_layer: bool = True, **kwargs):
        super().__init__(name=name, **kwargs)

        self.is_decoder = is_decoder
        self.num_hidden_layers = num_hidden_layers

        self.embeddings = BertEmbeddings(vocab_size, type_vocab_size, hidden_size, max_position_embeddings,
                                         initializer_range, layer_norm_eps, hidden_dropout_prob,
                                         name=name + "/embeddings")
        self.encoder = BertEncoder(hidden_size, num_attention_heads, initializer_range, attention_probs_dropout_prob,
                                   is_decoder, layer_norm_eps, hidden_dropout_prob, add_cross_attention,
                                   intermediate_size, hidden_act, num_hidden_layers, name=name + "/encoder")
        self.pooler = BertPooler(hidden_size, initializer_range, name=name + "/pooler") if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def forward(
            self,
            inputs=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):
        if not self.is_decoder:
            use_cache = False

        if inputs is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif inputs is not None:
            input_shape = shape_list(inputs)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]

        if attention_mask is None:
            attention_mask = tlx.ones(shape=(batch_size, seq_length + past_key_values_length))

        if token_type_ids is None:
            token_type_ids = tlx.zeros(shape=input_shape)

        embedding_output = self.embeddings(
            inputs=inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        # Copied from `modeling_tf_t5.py`
        # Provided a padding mask of dimensions [batch_size, mask_seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
        if self.is_decoder:
            seq_ids = tlx.arange(mask_seq_length)
            causal_mask = tlx.less_equal(
                tlx.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tlx.cast(causal_mask, dtype=attention_mask.dtype)
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tlx.reshape(
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
        else:
            extended_attention_mask = tlx.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tlx.cast(extended_attention_mask, dtype=embedding_output.dtype)
        one_cst = tlx.constant(value=1.0, dtype=embedding_output.dtype)
        ten_thousand_cst = tlx.constant(-10000.0, dtype=embedding_output.dtype)
        extended_attention_mask = tlx.multiply(tlx.subtract(one_cst, extended_attention_mask), ten_thousand_cst)

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
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

            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            inputs=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(inputs=sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output,) + encoder_outputs[1:]


class Bert(tlx.nn.Module):
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 add_cross_attention=False,
                 is_decoder=False,
                 add_pooling_layer=True,
                 *inputs, **kwargs):
        """
        :param vocab_size: (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model.
        :param hidden_size: (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        :param num_hidden_layers: (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        :param num_attention_heads: (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size: (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        :param hidden_act: (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function in the encoder and pooler.
        :param hidden_dropout_prob: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        :param attention_probs_dropout_prob: (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        :param max_position_embeddings: (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        :param initializer_range: (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        :param layer_norm_eps: (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        """
        super().__init__(*inputs, **kwargs)

        self.bert = BertMainLayer(vocab_size, type_vocab_size, hidden_size, max_position_embeddings, initializer_range,
                                  layer_norm_eps, hidden_dropout_prob, num_attention_heads,
                                  attention_probs_dropout_prob, add_cross_attention, intermediate_size, hidden_act,
                                  num_hidden_layers, is_decoder, add_pooling_layer=add_pooling_layer,
                                  name="bert")
        self.d_model = hidden_size
        # classifier_dropout = (
        #     classifier_dropout if classifier_dropout is not None else hidden_dropout_prob
        # )
        # self.dropout = tlx.layers.Dropout(classifier_dropout)

    def forward(
            self,
            inputs=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            **kwargs,
    ):
        outputs = self.bert(
            inputs=inputs,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs[1]
        return hidden_states
