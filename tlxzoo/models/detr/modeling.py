import tensorlayerx as tlx
from tensorlayerx.nn.layers import Conv2d, ZeroPad2d, MaxPool2d, ReLU
import numpy as np


class FrozenBatchNorm2D(tlx.nn.Module):
    def __init__(self, backbone_bn_shape, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

        self.weight = self._get_weights(var_name='weight', shape=[backbone_bn_shape],
                                        init=self.str_to_init("xavier_uniform"), trainable=False)
        self.bias = self._get_weights(var_name='bias', shape=[backbone_bn_shape],
                                      init=self.str_to_init("xavier_uniform"), trainable=False)
        self.running_mean = self._get_weights(var_name='running_mean', shape=[backbone_bn_shape],
                                              init=self.str_to_init("zeros"), trainable=False)
        self.running_var = self._get_weights(var_name='running_var', shape=[backbone_bn_shape],
                                             init=self.str_to_init("ones"), trainable=False)

    def forward(self, x):
        scale = self.weight * tlx.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift


class Linear(tlx.nn.Module):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

        self.kernel = self._get_weights(var_name='kernel',
                                        shape=[self.output_dim, input_dim],
                                        init=self.str_to_init("xavier_uniform"), trainable=True)
        self.bias = self._get_weights(var_name='bias',
                                      shape=[self.output_dim],
                                      init=self.str_to_init("xavier_uniform"), trainable=True)

    def forward(self, x):
        return tlx.matmul(x, self.kernel, transpose_b=True) + self.bias


class FixedEmbedding(tlx.nn.Module):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

        self.w = self._get_weights(var_name='kernel', shape=self.embed_shape,
                                   init=self.str_to_init("xavier_uniform"), trainable=True)

    def forward(self, x=None):
        return self.w


class ResNetBase(tlx.nn.Module):
    def __init__(self, config, name, **kwargs):
        super().__init__(name=name, **kwargs)

        self.pad1 = ZeroPad2d(3, name=name + '/pad1')
        self.conv1 = Conv2d(out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding='valid', in_channels=3,
                            b_init=None, name=name + '/conv1')
        self.bn1 = FrozenBatchNorm2D(config.backbone_bn_shape, name=name + '/bn1')
        self.relu = ReLU()
        self.pad2 = ZeroPad2d(1, name=name + '/pad2')
        self.maxpool = MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='valid')

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        xs = []
        x = self.layer1(x)
        xs.append(x)
        x = self.layer2(x)
        xs.append(x)
        x = self.layer3(x)
        xs.append(x)
        x = self.layer4(x)
        xs.append(x)
        return xs


class ResNet50Backbone(ResNetBase):
    def __init__(self, config, name, replace_stride_with_dilation=[False, False, False], **kwargs):
        super(ResNet50Backbone, self).__init__(config, name, **kwargs)

        self.layer1 = ResidualBlock(num_bottlenecks=3, dim1=64, dim2=256, strides=1, first_in_channels=64,
                                    bottlenecks_bn_shape=config.backbone_layer1_bn_shape,
                                    replace_stride_with_dilation=False, name=name + '/layer1')
        self.layer2 = ResidualBlock(num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                                    first_in_channels=config.backbone_layer1_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer2_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[0],
                                    name=name + '/layer2')
        self.layer3 = ResidualBlock(num_bottlenecks=6, dim1=256, dim2=1024, strides=2,
                                    first_in_channels=config.backbone_layer2_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer3_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[1],
                                    name=name + '/layer3')
        self.layer4 = ResidualBlock(num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                                    first_in_channels=config.backbone_layer3_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer4_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[2],
                                    name=name + '/layer4')


class ResNet101Backbone(ResNetBase):
    def __init__(self, config, name, replace_stride_with_dilation=[False, False, False], **kwargs):
        super(ResNet101Backbone, self).__init__(config, name, **kwargs)
        self.layer1 = ResidualBlock(num_bottlenecks=3, dim1=64, dim2=256, strides=1,
                                    bottlenecks_bn_shape=config.backbone_layer1_bn_shape,
                                    first_in_channels=config.backbone_layer1_bn_shape[-1][-1],
                                    replace_stride_with_dilation=False, name=name + '/layer1')
        self.layer2 = ResidualBlock(num_bottlenecks=4, dim1=128, dim2=512, strides=2,
                                    first_in_channels=config.backbone_layer2_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer2_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[0],
                                    name=name + '/layer2')
        self.layer3 = ResidualBlock(num_bottlenecks=23, dim1=256, dim2=1024, strides=2,
                                    first_in_channels=config.backbone_layer3_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer3_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[1],
                                    name=name + '/layer3')
        self.layer4 = ResidualBlock(num_bottlenecks=3, dim1=512, dim2=2048, strides=2,
                                    first_in_channels=config.backbone_layer4_bn_shape[-1][-1],
                                    bottlenecks_bn_shape=config.backbone_layer4_bn_shape,
                                    replace_stride_with_dilation=replace_stride_with_dilation[2],
                                    name=name + '/layer4')


class ResidualBlock(tlx.nn.Module):
    def __init__(self, num_bottlenecks, dim1, dim2, bottlenecks_bn_shape, name, first_in_channels, strides=1,
                 replace_stride_with_dilation=False, **kwargs):
        super(ResidualBlock, self).__init__(name=name, **kwargs)

        if replace_stride_with_dilation:
            strides = 1
            dilation = 2
        else:
            dilation = 1

        bottlenecks = [BottleNeck(dim1, dim2, bottlenecks_bn_shape[0], strides=strides,
                                  first_in_channels=first_in_channels,
                                  downsample=True, name=name + '/bottlenecks/0')]

        for idx in range(1, num_bottlenecks):
            bottlenecks.append(BottleNeck(dim1, dim2, bottlenecks_bn_shape[idx], name=name + "/bottlenecks/" + str(idx),
                                          dilation=dilation, first_in_channels=bottlenecks_bn_shape[idx - 1][-1]))
        self.bottlenecks = tlx.nn.SequentialLayer(bottlenecks)

    def forward(self, x):
        for btn in self.bottlenecks:
            x = btn(x)
        return x


class BottleNeck(tlx.nn.Module):
    def __init__(self, dim1, dim2, bn_shape, name, first_in_channels, strides=1, dilation=1, downsample=False,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.downsample = downsample
        self.pad = ZeroPad2d(dilation)
        self.relu = ReLU()

        self.conv1 = Conv2d(out_channels=dim1, kernel_size=(1, 1), in_channels=first_in_channels, padding="valid",
                            b_init=None, name=name + '/conv1')
        self.bn1 = FrozenBatchNorm2D(bn_shape[0], name=name + '/bn1')

        self.conv2 = Conv2d(out_channels=dim1, kernel_size=(3, 3), in_channels=dim1, stride=(strides, strides),
                            dilation=(dilation, dilation), b_init=None, name=name + '/conv2', padding="valid")
        self.bn2 = FrozenBatchNorm2D(bn_shape[1], name=name + '/bn2')

        self.conv3 = Conv2d(out_channels=dim2, kernel_size=(1, 1), in_channels=dim1, b_init=None, padding="valid",
                            name=name + '/conv3')
        self.bn3 = FrozenBatchNorm2D(bn_shape[2], name=name + '/bn3')

        if self.downsample:
            self.downsample_conv = Conv2d(out_channels=dim2, kernel_size=(1, 1), stride=(strides, strides),
                                          padding="valid",
                                          in_channels=first_in_channels, b_init=None, name=name + '/downsample_conv')
            self.downsample_bn = FrozenBatchNorm2D(bn_shape[3], name=name + '/downsample_bn')

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out


class Transformer(tlx.nn.Module):
    def __init__(self, name, model_dim=256, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        enc_norm = tlx.layers.LayerNorm(model_dim, epsilon=1e-5, name=name + '/norm_pre') if normalize_before else None
        self.encoder = TransformerEncoder(model_dim, num_heads, dim_feedforward,
                                          dropout, activation, normalize_before, enc_norm,
                                          num_encoder_layers, name=name + '/encoder')

        dec_norm = tlx.layers.LayerNorm(model_dim, epsilon=1e-5, name=name + '/decoder/norm')
        dec_norm.build([None, None, None])
        dec_norm._forward_state = True
        self.decoder = TransformerDecoder(model_dim, num_heads, dim_feedforward,
                                          dropout, activation, normalize_before, dec_norm,
                                          num_decoder_layers, name=name + '/decoder',
                                          return_intermediate=return_intermediate_dec)

    def forward(self, source, mask, query_encoding, pos_encoding):

        batch_size, rows, cols = [source.shape[i] for i in range(3)]
        source = tlx.reshape(source, [batch_size, -1, self.model_dim])
        source = tlx.transpose(source, [1, 0, 2])

        pos_encoding = tlx.reshape(pos_encoding, [batch_size, -1, self.model_dim])
        pos_encoding = tlx.transpose(pos_encoding, [1, 0, 2])

        query_encoding = tlx.expand_dims(query_encoding, axis=1)
        query_encoding = tlx.tile(query_encoding, [1, batch_size, 1])

        mask = tlx.reshape(mask, [batch_size, -1])

        target = tlx.zeros_like(query_encoding)

        memory = self.encoder(source, source_key_padding_mask=mask,
                              pos_encoding=pos_encoding)
        hs = self.decoder(target, memory, memory_key_padding_mask=mask,
                          pos_encoding=pos_encoding, query_encoding=query_encoding)

        hs = tlx.transpose(hs, [0, 2, 1, 3])
        # memory = tlx.transpose(memory, [1, 0, 2])
        # memory = tlx.reshape(memory, [batch_size, rows, cols, self.model_dim])

        return hs, memory


class TransformerEncoder(tlx.nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, norm=None,
                 num_encoder_layers=6, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)

        enc_layers = [EncoderLayer(model_dim, num_heads, dim_feedforward,
                                   dropout, activation, normalize_before,
                                   name=name + '/enc_layers/%d' % i)
                      for i in range(num_encoder_layers)]
        self.enc_layers = tlx.nn.SequentialLayer(enc_layers)

        self.norm = norm

    def forward(self, source, mask=None, source_key_padding_mask=None,
                pos_encoding=None):
        x = source

        for layer in self.enc_layers:
            x = layer(x, source_mask=mask, source_key_padding_mask=source_key_padding_mask,
                      pos_encoding=pos_encoding)

        if self.norm:
            x = self.norm(x)

        return x


class TransformerDecoder(tlx.nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, norm=None,
                 num_decoder_layers=6, return_intermediate=False, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)

        dec_layers = [DecoderLayer(model_dim, num_heads, dim_feedforward,
                                   dropout, activation, normalize_before,
                                   name=name+'/dec_layers/%d' % i)
                      for i in range(num_decoder_layers)]
        self.dec_layers = tlx.nn.SequentialLayer(dec_layers)

        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, target, memory, target_mask=None, memory_mask=None,
                target_key_padding_mask=None, memory_key_padding_mask=None,
                pos_encoding=None, query_encoding=None):

        x = target
        intermediate = []

        for layer in self.dec_layers:
            x = layer(x, memory,
                      target_mask=target_mask,
                      memory_mask=memory_mask,
                      target_key_padding_mask=target_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos_encoding=pos_encoding,
                      query_encoding=query_encoding)

            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(x))
                else:
                    intermediate.append(x)

        if self.return_intermediate:
            return tlx.stack(intermediate, axis=0)

        if self.norm:
            x = self.norm(x)

        return x


class EncoderLayer(tlx.nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, name="layer",
                 **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                            name=name + '/self_attn')

        self.dropout = tlx.nn.Dropout(dropout)
        if activation == "relu":
            self.activation = ReLU()

        self.linear1 = Linear(model_dim, dim_feedforward, name=name + '/linear1')
        self.linear2 = Linear(dim_feedforward, model_dim, name=name + '/linear2')

        self.norm1 = tlx.nn.layers.LayerNorm(model_dim, epsilon=1e-5, name=name + '/norm1')
        self.norm1.build([None, None, None])
        self.norm1._forward_state = True
        self.norm2 = tlx.nn.layers.LayerNorm(model_dim, epsilon=1e-5, name=name + '/norm2')
        self.norm2.build([None, None, None])
        self.norm2._forward_state = True

        self.normalize_before = normalize_before

    def forward(self, source, source_mask=None, source_key_padding_mask=None,
                pos_encoding=None):

        if pos_encoding is None:
            query = key = source
        else:
            query = key = source + pos_encoding
        attn_source = self.self_attn((query, key, source), attn_mask=source_mask,
                                     key_padding_mask=source_key_padding_mask,
                                     need_weights=False)
        source += self.dropout(attn_source)
        source = self.norm1(source)

        x = self.linear1(source)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        source += self.dropout(x)
        source = self.norm2(source)

        return source


class DecoderLayer(tlx.nn.Module):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, name="layer",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                            name=name+'/self_attn')
        self.multihead_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                                 name=name+'/multihead_attn')

        self.dropout = tlx.nn.Dropout(dropout)
        if activation == "relu":
            self.activation = ReLU()

        self.linear1 = Linear(model_dim, dim_feedforward, name=name+'/linear1')
        self.linear2 = Linear(dim_feedforward, model_dim, name=name+'/linear2')

        self.norm1 = tlx.nn.LayerNorm(model_dim, epsilon=1e-5, name=name+'/norm1')
        self.norm1.build([None, None, None])
        self.norm1._forward_state = True
        self.norm2 = tlx.nn.LayerNorm(model_dim, epsilon=1e-5, name=name+'/norm2')
        self.norm2.build([None, None, None])
        self.norm2._forward_state = True
        self.norm3 = tlx.nn.LayerNorm(model_dim, epsilon=1e-5, name=name+'/norm3')
        self.norm3.build([None, None, None])
        self.norm3._forward_state = True

        self.normalize_before = normalize_before

    def forward(self, target, memory, target_mask=None, memory_mask=None,
                target_key_padding_mask=None, memory_key_padding_mask=None,
                pos_encoding=None, query_encoding=None):
        query_tgt = key_tgt = target + query_encoding
        attn_target = self.self_attn((query_tgt, key_tgt, target), attn_mask=target_mask,
                                     key_padding_mask=target_key_padding_mask,
                                     need_weights=False)
        target += self.dropout(attn_target)
        target = self.norm1(target)

        query_tgt = target + query_encoding
        key_mem = memory + pos_encoding

        attn_target2 = self.multihead_attn((query_tgt, key_mem, memory), attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask,
                                           need_weights=False)
        target += self.dropout(attn_target2)
        target = self.norm2(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        target += self.dropout(x)
        target = self.norm3(target)

        return target


class MultiHeadAttention(tlx.nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0, name="multihead_attn", **kwargs):
        super().__init__(name=name, **kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = tlx.nn.Dropout(dropout)

        in_dim = self.model_dim * 3

        self.in_proj_weight = self._get_weights(
            var_name='in_proj_weight', shape=(in_dim, self.model_dim),
            init=self.str_to_init("xavier_uniform"), trainable=True
        )
        self.in_proj_bias = self._get_weights(
            var_name='in_proj_bias', shape=(in_dim,),
            init=self.str_to_init("xavier_uniform"), trainable=True
        )
        self.out_proj_weight = self._get_weights(
            var_name='out_proj_weight', shape=(self.model_dim, self.model_dim),
            init=self.str_to_init("xavier_uniform"), trainable=True
        )
        self.out_proj_bias = self._get_weights(
            var_name='out_proj_bias', shape=(self.model_dim,),
            init=self.str_to_init("xavier_uniform"), trainable=True
        )

    def forward(self, inputs, attn_mask=None, key_padding_mask=None,
                need_weights=True):

        query, key, value = inputs

        batch_size = query.shape[1]
        target_len = query.shape[0]
        source_len = key.shape[0]

        W = self.in_proj_weight[:self.model_dim, :]
        b = self.in_proj_bias[:self.model_dim]
        WQ = tlx.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.model_dim:2 * self.model_dim, :]
        b = self.in_proj_bias[self.model_dim:2 * self.model_dim]
        WK = tlx.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2 * self.model_dim:, :]
        b = self.in_proj_bias[2 * self.model_dim:]
        WV = tlx.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tlx.reshape(WQ, [target_len, batch_size * self.num_heads, self.head_dim])
        WQ = tlx.transpose(WQ, [1, 0, 2])

        WK = tlx.reshape(WK, [source_len, batch_size * self.num_heads, self.head_dim])
        WK = tlx.transpose(WK, [1, 0, 2])

        WV = tlx.reshape(WV, [source_len, batch_size * self.num_heads, self.head_dim])
        WV = tlx.transpose(WV, [1, 0, 2])

        attn_output_weights = tlx.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = tlx.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = tlx.matmul(attn_output_weights, WV)
        attn_output = tlx.transpose(attn_output, [1, 0, 2])
        attn_output = tlx.reshape(attn_output, [target_len, batch_size, self.model_dim])
        attn_output = tlx.matmul(attn_output, self.out_proj_weight,
                                 transpose_b=True) + self.out_proj_bias

        if need_weights:
            attn_output_weights = tlx.reshape(attn_output_weights,
                                              [batch_size, self.num_heads, target_len, source_len])
            # Retrun the average weight over the heads
            avg_weights = tlx.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights

        return attn_output


class PositionEmbeddingSine(tlx.nn.Module):

    def __init__(self, num_pos_features=64, temperature=10000,
                 normalize=False, scale=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps

    def forward(self, mask):
        not_mask = tlx.cast(mask, tlx.float32)
        y_embed = tlx.cumsum(not_mask, axis=1)
        x_embed = tlx.cumsum(not_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tlx.arange(self.num_pos_features, dtype=tlx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = tlx.stack([tlx.sin(pos_x[..., 0::2]),
                           tlx.cos(pos_x[..., 1::2])], axis=4)

        pos_y = tlx.stack([tlx.sin(pos_y[..., 0::2]),
                           tlx.cos(pos_y[..., 1::2])], axis=4)

        shape = [pos_x.shape[i] for i in range(3)] + [-1]
        pos_x = tlx.reshape(pos_x, shape)
        pos_y = tlx.reshape(pos_y, shape)

        pos_emb = tlx.concat([pos_y, pos_x], axis=3)
        return pos_emb
