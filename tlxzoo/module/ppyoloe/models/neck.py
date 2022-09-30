import tensorlayerx as tlx
import tensorlayerx.nn as nn

from .backbone import BasicBlock, ConvBNLayer
from .utils import get_act_fn, numel


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='channels_first'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, "channels first" or "channels last"
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.is_train or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size ** 2)
            if self.data_format == 'channels_first':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = tlx.random_uniform(x.shape)
            matrix = tlx.cast(matrix < gamma, tlx.float32)
            mask_inv = tlx.ops.max_pool(
                matrix,
                (self.block_size, self.block_size),
                stride=(1, 1),
                padding='SAME')
            mask = 1. - mask_inv
            y = x * mask * (numel(mask) / tlx.reduce_sum(mask))
            return y


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 act_name='swish',
                 data_format='channels_first'):
        super(SPP, self).__init__()
        self.pool = nn.ModuleList()
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=(size, size),
                stride=(1, 1),
                padding='SAME',
                data_format=data_format)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k //
                                2, act=act, act_name=act_name, data_format=data_format)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'channels_first':
            y = tlx.concat(outs, 1)
        else:
            y = tlx.concat(outs, -1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', act_name='swish', spp=False, data_format='channels_first'):
        super(CSPStage, self).__init__()
        self.data_format = data_format

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(
            ch_in, ch_mid, 1, act=act, act_name=act_name, data_format=data_format)
        self.conv2 = ConvBNLayer(
            ch_in, ch_mid, 1, act=act, act_name=act_name, data_format=data_format)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.append(
                eval(block_fn)(next_ch_in, ch_mid, act=act, act_name=act_name, shortcut=False, data_format=data_format))
            if i == (n - 1) // 2 and spp:
                self.convs.append(
                    SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act, act_name=act_name, data_format=data_format))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(
            ch_mid * 2, ch_out, 1, act=act, act_name=act_name, data_format=data_format)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        if self.data_format == 'channels_first':
            y = tlx.concat([y1, y2], 1)
        else:
            y = tlx.concat([y1, y2], -1)
        y = self.conv3(y)
        return y


class CustomCSPPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format',
                  'width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 data_format='channels_first',
                 width_mult=1.0,
                 depth_mult=1.0):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act_name = act
        act = get_act_fn(act) if act is None or isinstance(
            act, (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.append(
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=(spp and i == 0),
                                   data_format=data_format))

            if drop_block:
                stage.append(DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        act_name=act_name,
                        data_format=data_format))

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name,
                    data_format=data_format))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.append(
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=False,
                                   data_format=data_format))
            if drop_block:
                stage.append(DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == 'channels_first':
                    block = tlx.concat([route, block], 1)
                else:
                    block = tlx.concat([route, block], -1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = tlx.Resize((2.0, 2.0), 'bilinear',
                                   data_format=self.data_format)(route)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            if self.data_format == 'channels_first':
                block = tlx.concat([route, block], 1)
            else:
                block = tlx.concat([route, block], -1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]
