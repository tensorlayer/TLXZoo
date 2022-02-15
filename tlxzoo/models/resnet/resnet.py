"""ResNet for ImageNet.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
"""
from ..model import BaseModule
from ...utils.registry import Registers
from tensorlayerx import logging

from tensorlayerx.nn import (BatchNorm, Conv2d, Dense, Elementwise, GlobalMeanPool2d, MaxPool2d)
from tensorlayerx.nn import Module, SequentialLayer
import os
from ...utils.output import BaseModelOutput
from dataclasses import dataclass
import tensorlayerx as tlx

__all__ = ["ResNet"]

block_names = ['2a', '2b', '2c', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '4e', '4f', '5a', '5b', '5c'] + [
    'avg_pool', 'fc1000']
block_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
in_channels_conv = [64, 256, 512, 1024]
in_channels_identity = [256, 512, 1024, 2048]
henorm = tlx.nn.initializers.he_normal()


@dataclass
class ResNetModelOutput(BaseModelOutput):
    ...


class IdentityBlock(Module):
    """The identity block where there is no conv layer at shortcut.
    Parameters
    ----------
    input : tf tensor
        Input tensor from above layer.
    kernel_size : int
        The kernel size of middle conv layer at main path.
    n_filters : list of integers
        The numbers of filters for 3 conv layer at main path.
    stage : int
        Current stage label.
    block : str
        Current block label.
    Returns
    -------
        Output tensor of this block.
    """

    def __init__(self, config, n_filters, stage, block):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = n_filters
        _in_channels = in_channels_identity[stage - 2]
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2d(filters1, (1, 1), W_init=henorm, name=conv_name_base + '2a', in_channels=_in_channels)
        self.bn1 = BatchNorm(name=bn_name_base + '2a', act='relu', num_features=filters1)

        ks = (config.identity_block_kernel_size, config.identity_block_kernel_size)
        self.conv2 = Conv2d(
            filters2, ks, padding='SAME', W_init=henorm, name=conv_name_base + '2b', in_channels=filters1
        )
        self.bn2 = BatchNorm(name=bn_name_base + '2b', act='relu', num_features=filters2)

        self.conv3 = Conv2d(filters3, (1, 1), W_init=henorm, name=conv_name_base + '2c', in_channels=filters2)
        self.bn3 = BatchNorm(name=bn_name_base + '2c', num_features=filters3)

        self.add = Elementwise(tlx.add, act='relu')

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        result = self.add([output, inputs])
        return result


class ConvBlock(Module):

    def __init__(self, config, n_filters, stage, block, strides=(2, 2)):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = n_filters
        _in_channels = in_channels_conv[stage - 2]
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.conv1 = Conv2d(
            filters1, (1, 1), strides=strides, W_init=henorm, name=conv_name_base + '2a', in_channels=_in_channels
        )
        self.bn1 = BatchNorm(name=bn_name_base + '2a', act='relu', num_features=filters1)

        ks = (config.conv_block_kernel_size, config.conv_block_kernel_size)
        self.conv2 = Conv2d(
            filters2, ks, padding='SAME', W_init=henorm, name=conv_name_base + '2b', in_channels=filters1
        )
        self.bn2 = BatchNorm(name=bn_name_base + '2b', act='relu', num_features=filters2)

        self.conv3 = Conv2d(filters3, (1, 1), W_init=henorm, name=conv_name_base + '2c', in_channels=filters2)
        self.bn3 = BatchNorm(name=bn_name_base + '2c', num_features=filters3)

        self.shortcut_conv = Conv2d(
            filters3, (1, 1), strides=strides, W_init=henorm, name=conv_name_base + '1', in_channels=_in_channels
        )
        self.shortcut_bn = BatchNorm(name=bn_name_base + '1', num_features=filters3)

        self.add = Elementwise(tlx.add, act='relu')

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.conv3(output)
        output = self.bn3(output)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut)

        result = self.add([output, shortcut])
        return result


@Registers.models.register
class ResNet(BaseModule):
    def __init__(self, config):
        super(ResNet, self).__init__(config)
        self.end_with = self.config.end_with
        self.n_classes = self.config.n_classes
        self.conv1 = Conv2d(self.config.conv1_n_filter, self.config.conv1_filter_size,
                            in_channels=self.config.conv1_in_channels, strides=self.config.conv1_strides,
                            padding='SAME', W_init=henorm, name='conv1')
        self.bn_conv1 = BatchNorm(name='bn_conv1', act="relu", num_features=self.config.bn_conv1_num_features)
        self.max_pool1 = MaxPool2d(self.config.max_pool1_filter_size, strides=self.config.max_pool1_strides,
                                   name='max_pool1')
        self.res_layer = self.make_layer()

    def forward(self, inputs):
        z = self.conv1(inputs)
        z = self.bn_conv1(z)
        z = self.max_pool1(z)
        z = self.res_layer(z)
        return ResNetModelOutput(output=z)

    def make_layer(self):
        layer_list = []
        for i, block_name in enumerate(block_names):
            if len(block_name) == 2:
                stage = int(block_name[0])
                block = block_name[1]
                if block == 'a':
                    strides = (1, 1) if stage == 2 else (2, 2)
                    layer_list.append(
                        ConvBlock(self.config, block_filters[stage - 2], stage=stage, block=block, strides=strides)
                    )
                else:
                    layer_list.append(IdentityBlock(self.config, block_filters[stage - 2], stage=stage, block=block))
            elif block_name == 'avg_pool':
                layer_list.append(GlobalMeanPool2d(name='avg_pool'))
            elif block_name == 'fc1000':
                layer_list.append(Dense(self.n_classes, name='fc1000', in_channels=2048))

            if block_name == self.end_with:
                break
        return SequentialLayer(layer_list)
