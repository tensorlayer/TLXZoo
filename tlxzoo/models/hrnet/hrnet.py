from tensorlayerx import logging
import tensorlayerx as tlx
from tensorlayerx import nn

BN_MOMENTUM = 0.1
logger = logging


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.layers.Conv2d(out_channels=out_planes,
                            kernel_size=(3, 3),
                            stride=(stride, stride),
                            padding="same",
                            b_init=None,
                            in_channels=in_planes,
                            )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes, decay=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(num_features=planes, decay=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = tlx.nn.layers.Conv2d(out_channels=planes,
                                          kernel_size=(1, 1),
                                          stride=(stride, stride),
                                          padding="same",
                                          b_init=None,
                                          in_channels=inplanes,
                                          )
        self.bn1 = nn.BatchNorm2d(num_features=planes, decay=BN_MOMENTUM)
        self.conv2 = tlx.nn.layers.Conv2d(out_channels=planes,
                                          kernel_size=(3, 3),
                                          stride=(stride, stride),
                                          padding="same",
                                          b_init=None,
                                          in_channels=planes,
                                          )
        self.bn2 = nn.BatchNorm2d(num_features=planes, decay=BN_MOMENTUM)
        self.conv3 = tlx.nn.layers.Conv2d(out_channels=planes * self.expansion,
                                          kernel_size=(1, 1),
                                          padding="same",
                                          b_init=None,
                                          in_channels=planes,
                                          )
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion, decay=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class NoneModule(nn.Module):
    def forward(self, inputs):
        return inputs


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialLayer(
                [nn.layers.Conv2d(out_channels=num_channels[branch_index] * block.expansion,
                                  kernel_size=(1, 1),
                                  stride=(stride, stride),
                                  padding="same",
                                  b_init=None,
                                  in_channels=self.num_inchannels[branch_index],
                                  ),
                 nn.BatchNorm2d(num_features=num_channels[branch_index] * block.expansion, decay=BN_MOMENTUM)]
            )
        # print(downsample, stride, branch_index, self.num_inchannels, num_channels)

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.SequentialLayer(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.LayerList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialLayer([
                            nn.Conv2d(
                                in_channels=num_inchannels[j],
                                out_channels=num_inchannels[i],
                                kernel_size=(1, 1), stride=(1, 1), padding="same", b_init=None
                            ),
                            nn.BatchNorm2d(num_features=num_inchannels[i]),
                            tlx.nn.UpSampling2d(scale=(2 ** (j - i), 2 ** (j - i)))]
                        )
                    )
                elif j == i:
                    fuse_layer.append(NoneModule())
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.SequentialLayer(
                                    [nn.Conv2d(
                                        in_channels=num_inchannels[j],
                                        out_channels=num_outchannels_conv3x3,
                                        kernel_size=(3, 3), stride=(2, 2), padding="same", b_init=False
                                    ),
                                        nn.BatchNorm2d(num_features=num_outchannels_conv3x3)]
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.SequentialLayer(
                                    [nn.Conv2d(
                                        in_channels=num_inchannels[j],
                                        out_channels=num_outchannels_conv3x3,
                                        kernel_size=(3, 3), stride=(2, 2), padding="same", b_init=False
                                    ),
                                        nn.BatchNorm2d(num_features=num_outchannels_conv3x3),
                                        nn.ReLU()]
                                )
                            )
                    fuse_layer.append(nn.SequentialLayer(conv3x3s))
            fuse_layers.append(nn.LayerList(fuse_layer))

        return nn.LayerList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # for i in range(self.num_branches):
        #     print(i, x[i].shape)

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        self.config_params = config
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding="same",
                               b_init=None)
        self.bn1 = nn.BatchNorm2d(num_features=64, decay=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding="same",
                               b_init=None)
        self.bn2 = nn.BatchNorm2d(num_features=64, decay=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = self.config_params.stage_2
        num_channels = self.stage2_cfg.get_stage_channels()
        block = blocks_dict[self.stage2_cfg.get_block()]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = self.config_params.stage_3
        num_channels = self.stage3_cfg.get_stage_channels()
        block = blocks_dict[self.stage3_cfg.get_block()]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = self.config_params.stage_4
        num_channels = self.stage4_cfg.get_stage_channels()
        block = blocks_dict[self.stage4_cfg.get_block()]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=config.num_of_joints,
            kernel_size=(self.config_params.conv3_kernel, self.config_params.conv3_kernel),
            stride=(1, 1),
            padding="same" if self.config_params.conv3_kernel == 3 else "VALID"
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialLayer(
                            [nn.Conv2d(
                                in_channels=num_channels_pre_layer[i],
                                out_channels=num_channels_cur_layer[i],
                                kernel_size=(3, 3), stride=(1, 1), padding="SAME", b_init=None
                            ),
                                nn.BatchNorm2d(num_features=num_channels_cur_layer[i]),
                                nn.ReLU()]
                        )
                    )
                else:
                    transition_layers.append(NoneModule())
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.SequentialLayer(
                            [nn.Conv2d(
                                in_channels=inchannels, out_channels=outchannels, kernel_size=(3, 3),
                                stride=(2, 2), padding="SAME", b_init=None
                            ),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU()]
                        )
                    )
                transition_layers.append(nn.SequentialLayer(conv3x3s))

        return nn.LayerList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialLayer(
                [nn.Conv2d(
                    in_channels=self.inplanes, out_channels=planes * block.expansion,
                    kernel_size=(1, 1), stride=(stride, stride), b_init=None
                ),
                    nn.BatchNorm2d(num_features=planes * block.expansion, decay=BN_MOMENTUM)]
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialLayer(layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config.get_modules()
        num_branches = layer_config.get_branch_num()
        num_blocks = layer_config.get_num_blocks()
        num_channels = layer_config.get_stage_channels()
        block = blocks_dict[layer_config.get_block()]
        fuse_method = layer_config.get_fusion_method()

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.SequentialLayer(modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.get_branch_num()):
            # if self.transition1[i] is not None:
            if not isinstance(self.transition1[i], NoneModule):
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.get_branch_num()):
            if not isinstance(self.transition2[i], NoneModule):
            # if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.get_branch_num()):
            # if self.transition3[i] is not None:
            if not isinstance(self.transition3[i], NoneModule):
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x
