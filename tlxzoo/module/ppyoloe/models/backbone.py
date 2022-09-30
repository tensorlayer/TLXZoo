from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from .utils import get_act_fn


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None,
                 act_name=None,
                 data_format='channels_first'):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.GroupConv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=(filter_size, filter_size),
            stride=(stride, stride),
            padding='VALID' if padding==0 else 'SAME',
            n_group=groups,
            b_init=None,
            data_format=data_format
        )

        self.bn = nn.BatchNorm2d(
            num_features=ch_out,
            data_format=data_format
        )
        self.act_name = act_name
        if act is None or isinstance(act, (str, dict)):
            self.act = get_act_fn(act)
        else:
            self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu', data_format='channels_first'):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None, data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None, data_format=data_format)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu', shortcut=True, data_format='channels_first'):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act, act_name=act_name, data_format=data_format)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, act_name=act_name, data_format=data_format)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid', act_name='hardsigmoid', data_format='channels_first'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, 1),
            padding='VALID',
            data_format=data_format)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_first':
            x_se = tlx.reduce_mean(x, (2, 3), keepdims=True)
        else:
            x_se = tlx.reduce_mean(x, (1, 2), keepdims=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 act_name=None,
                 attn='eca',
                 data_format='channels_first'):
        super(CSPResStage, self).__init__()
        self.data_format = data_format

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act, act_name=act_name, data_format=data_format)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name, data_format=data_format)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name, data_format=data_format)
        self.blocks = nn.Sequential([
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, act_name=act_name, shortcut=True, data_format=data_format)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid', act_name='hardsigmoid', data_format=data_format)
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act, act_name=act_name, data_format=data_format)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        if self.data_format == 'channels_first':
            y = tlx.concat([y1, y2], 1)
        else:
            y = tlx.concat([y1, y2], -1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(nn.Module):
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 depth_wise=False,
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 data_format='channels_first'):
        super(CSPResNet, self).__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act_name = act
        act = get_act_fn(
            act) if act is None or isinstance(act,
                                                       (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential()
            self.stem.append(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name, data_format=data_format))
            self.stem.append(ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act, act_name=act_name, data_format=data_format))
            self.stem.append(ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name, data_format=data_format))
        else:
            self.stem = nn.Sequential()
            self.stem.append(ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name, data_format=data_format))
            self.stem.append(ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name, data_format=data_format))

        n = len(channels) - 1
        self.stages = nn.Sequential()
        for i in range(n):
            self.stages.append(CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, act_name=act_name, data_format=data_format))

        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs
