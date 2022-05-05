import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.nn.core.common import str2act
import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, name="", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = nn.Conv2d(out_channels=filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  stride=(1, 1),
                                  padding=padding,
                                  W_init=_get_kernel_initializer(filters, kernel_size),
                                  name=name + "/conv2d_1"
                                  )
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.activation_1 = str2act(activation)()

        self.conv2d_2 = nn.Conv2d(out_channels=filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  stride=(1, 1),
                                  padding=padding,
                                  W_init=_get_kernel_initializer(filters, kernel_size),
                                  name=name + "/conv2d_2"
                                  )
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.activation_2 = str2act(activation)()

    def forward(self, inputs):
        x = inputs
        x = self.conv2d_1(x)

        x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        x = self.dropout_2(x)

        x = self.activation_2(x)
        return x


class UpconvBlock(nn.Module):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx + 1, filters_root)

        self.upconv = nn.ConvTranspose2d(out_channels=filters // 2,
                                         kernel_size=(pool_size, pool_size),
                                         stride=(pool_size, pool_size),
                                         padding=padding,
                                         W_init=_get_kernel_initializer(filters, kernel_size), )

        self.activation_1 = str2act(activation)()

    def forward(self, inputs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x


class CropConcatBlock(nn.Module):

    def forward(self, x, down_layer):
        x1_shape = tlx.get_tensor_shape(down_layer)
        x2_shape = tlx.get_tensor_shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: (x2_shape[1] + height_diff),
                             width_diff: (x2_shape[2] + width_diff),
                             :]

        x = tlx.concat([down_layer_cropped, x], axis=-1)
        return x


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return tlx.initializers.truncated_normal(stddev=stddev)


class UnetModel(nn.Module):
    def __init__(self, nx, ny, channels, num_classes, layer_depth, filters_root, kernel_size=3,
                 pool_size=2, dropout_rate=0.5, padding="valid", activation="relu", name="unet"):
        super(UnetModel, self).__init__(name=name)

        conv_params = dict(filters_root=filters_root,
                           kernel_size=kernel_size,
                           dropout_rate=dropout_rate,
                           padding=padding,
                           activation=activation)

        self.layer_depth = layer_depth

        conv_blocks = []
        max_pools = []
        for layer_idx in range(0, layer_depth - 1):
            conv_blocks.append(ConvBlock(layer_idx, **conv_params))
            max_pools.append(nn.MaxPool2d(kernel_size=(pool_size, pool_size)))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.max_pools = nn.ModuleList(max_pools)

        self.layer_idx = layer_depth - 2
        self.conv_block = ConvBlock(self.layer_idx + 1, **conv_params)

        upconv_blocks = []
        crop_concat_blocks = []
        conv_blocks2 = []
        for layer_idx in range(self.layer_idx, -1, -1):
            upconv_blocks.append(UpconvBlock(layer_idx,
                                             filters_root,
                                             kernel_size,
                                             pool_size,
                                             padding,
                                             activation))
            crop_concat_blocks.append(CropConcatBlock())
            conv_blocks2.append(ConvBlock(layer_idx, **conv_params))

        self.upconv_blocks = nn.ModuleList(upconv_blocks)
        self.crop_concat_blocks = nn.ModuleList(crop_concat_blocks)
        self.conv_blocks2 = nn.ModuleList(conv_blocks2)

        self.conv_2d = nn.Conv2d(out_channels=num_classes,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=padding,
                                 W_init=_get_kernel_initializer(filters_root, kernel_size))

        self.act = str2act(activation)()
        self.softmax = nn.Softmax()

        if nx is not None:
            self.build((2, nx, ny, channels))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, inputs):
        x = inputs
        contracting_layers = {}

        for layer_idx in range(0, self.layer_depth - 1):
            x = self.conv_blocks[layer_idx](x)
            contracting_layers[layer_idx] = x
            x = self.max_pools[layer_idx](x)

        x = self.conv_block(x)

        for layer_idx in range(self.layer_idx, -1, -1):
            x = self.upconv_blocks[layer_idx](x)
            x = self.crop_concat_blocks[layer_idx](x, contracting_layers[layer_idx])
            x = self.conv_blocks2[layer_idx](x)

        x = self.conv_2d(x)

        x = self.act(x)
        # outputs = self.softmax(x)
        return x
