import tensorlayerx as tlx
from tensorlayerx.nn.core import Module


class BasicBlock(Module):
    def __init__(self, filter_num, stride=1, name=""):
        super(BasicBlock, self).__init__()
        self.conv1 = tlx.nn.layers.Conv2d(out_channels=filter_num,
                                          kernel_size=(3, 3),
                                          stride=(stride, stride),
                                          padding="same",
                                          name=name + "/conv1",
                                          in_channels=None,
                                          )
        self.bn1 = tlx.nn.BatchNorm(num_features=filter_num, decay=0.1, epsilon=1e-5)
        self.conv2 = tlx.nn.layers.Conv2d(out_channels=filter_num,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          padding="same",
                                          name=name + "/conv2",
                                          in_channels=None,
                                          )
        self.bn2 = tlx.nn.BatchNorm(num_features=filter_num, decay=0.1, epsilon=1e-5)

        if stride != 1:
            downsample = [tlx.nn.layers.Conv2d(out_channels=filter_num,
                                               kernel_size=(1, 1),
                                               stride=(stride, stride),
                                               padding="same",
                                               name=name + "/downsample",
                                               in_channels=None,
                                               ),
                          tlx.nn.BatchNorm(num_features=filter_num, decay=0.1, epsilon=1e-5)
                          ]
            self.downsample = tlx.nn.core.SequentialLayer(downsample)
        else:
            self.downsample = lambda x: x

        self.relu = tlx.nn.ReLU()
        self.filter_num = filter_num
        self.stride = stride

    def forward(self, inputs):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2)

        # print(inputs.shape, self.filter_num, self.stride, residual.shape, bn2.shape)
        output = self.relu(tlx.add(residual, bn2))

        return output


class BottleNeck(Module):
    def __init__(self, filter_num, stride=1, name=""):
        super(BottleNeck, self).__init__()
        self.conv1 = tlx.nn.layers.Conv2d(out_channels=filter_num,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding="same",
                                          name=name + "/conv1",
                                          b_init=None,
                                          in_channels=None,
                                          )
        self.bn1 = tlx.nn.BatchNorm(num_features=filter_num, decay=0.1, epsilon=1e-5)
        self.conv2 = tlx.nn.layers.Conv2d(out_channels=filter_num,
                                          kernel_size=(3, 3),
                                          stride=(stride, stride),
                                          padding="same",
                                          name=name + "/conv2",
                                          b_init=None,
                                          in_channels=None,
                                          )
        self.bn2 = tlx.nn.BatchNorm(num_features=filter_num, decay=0.1, epsilon=1e-5)
        self.conv3 = tlx.nn.layers.Conv2d(out_channels=filter_num * 4,
                                          kernel_size=(1, 1),
                                          stride=(1, 1),
                                          padding="same",
                                          name=name + "/conv3",
                                          b_init=None,
                                          in_channels=None,
                                          )
        self.bn3 = tlx.nn.BatchNorm(num_features=filter_num * 4, decay=0.1, epsilon=1e-5)

        downsample = [tlx.nn.layers.Conv2d(out_channels=filter_num * 4,
                                           kernel_size=(1, 1),
                                           stride=(stride, stride),
                                           padding="same",
                                           b_init=None,
                                           name=name + "/downsample",
                                           in_channels=None,
                                           ),
                      tlx.nn.BatchNorm(num_features=filter_num * 4, decay=0.1, epsilon=1e-5)
                      ]
        self.downsample = tlx.nn.core.SequentialLayer(downsample)
        self.relu = tlx.nn.ReLU()

    def forward(self, inputs):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)

        output = self.relu(tlx.add(residual, bn3))

        return output


def make_basic_layer(filter_num, blocks, stride=1):
    res_block = [BasicBlock(filter_num, stride=stride)]

    for _ in range(1, blocks):
        res_block.append(BasicBlock(filter_num, stride=1))

    return tlx.nn.SequentialLayer(res_block)


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = [BottleNeck(filter_num, stride=stride)]

    for _ in range(1, blocks):
        res_block.append(BottleNeck(filter_num, stride=1))

    return tlx.nn.SequentialLayer(res_block)


class NoneModule(Module):
    def forward(self, inputs):
        return inputs


class HighResolutionModule(Module):
    def __init__(self, num_branches, num_in_channels, num_channels, block, num_blocks, fusion_method, multi_scale_output=True, name=""):
        super(HighResolutionModule, self).__init__(name=name)
        self.num_branches = num_branches
        self.num_in_channels = num_in_channels
        self.fusion_method = fusion_method
        self.multi_scale_output = multi_scale_output
        self.branches = tlx.nn.LayerList(self.__make_branches(num_channels, block, num_blocks))
        self.fusion_layer = tlx.nn.LayerList(self.__make_fusion_layers())
        self.relu = tlx.nn.ReLU()

    def get_output_channels(self):
        return self.num_in_channels

    def __make_branches(self, num_channels, block, num_blocks):
        def __make_one_branch(block, num_blocks, num_channels, stride=1):
            if block == "BASIC":
                return make_basic_layer(filter_num=num_channels, blocks=num_blocks, stride=stride)
            elif block == "BOTTLENECK":
                return make_bottleneck_layer(filter_num=num_channels, blocks=num_blocks, stride=stride)

        branch_layers = []
        for i in range(self.num_branches):
            # print(12, i, num_channels[i], self.num_branches, num_blocks[i])
            branch_layers.append(__make_one_branch(block, num_blocks[i], num_channels[i]))
        return branch_layers

    def __make_fusion_layers(self):
        if self.num_branches == 1:
            return None

        fusion_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fusion_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fusion_layer.append(
                        tlx.nn.SequentialLayer([
                            tlx.nn.layers.Conv2d(out_channels=self.num_in_channels[i],
                                                 kernel_size=(1, 1),
                                                 stride=(1, 1),
                                                 padding="same",
                                                 b_init=None,
                                                 in_channels=None,
                                                 ),
                            tlx.nn.BatchNorm(num_features=self.num_in_channels[i], decay=0.1, epsilon=1e-5),
                            tlx.nn.UpSampling2d(scale=(2**(j-i), 2**(j-i))),
                            # tf.keras.layers.UpSampling2D(size=2**(j-i))
                        ])
                    )
                elif j == i:
                    fusion_layer.append(NoneModule())
                else:
                    down_sample = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            downsample_out_channels = self.num_in_channels[i]
                            down_sample.append(
                                tlx.nn.SequentialLayer([
                                    tlx.nn.layers.Conv2d(out_channels=downsample_out_channels,
                                                         kernel_size=(3, 3),
                                                         stride=(2, 2),
                                                         padding="same",
                                                         b_init=None,
                                                         in_channels=None,
                                                         ),
                                    tlx.nn.BatchNorm(num_features=downsample_out_channels, decay=0.1, epsilon=1e-5),
                                ])
                            )
                        else:
                            downsample_out_channels = self.num_in_channels[j]
                            down_sample.append(
                                tlx.nn.SequentialLayer([
                                    tlx.nn.layers.Conv2d(out_channels=downsample_out_channels,
                                                         kernel_size=(3, 3),
                                                         stride=(2, 2),
                                                         padding="same",
                                                         b_init=None,
                                                         in_channels=None,
                                                         ),
                                    tlx.nn.BatchNorm(num_features=downsample_out_channels, decay=0.1, epsilon=1e-5),
                                    tlx.nn.ReLU()
                                ])
                            )
                    fusion_layer.append(tlx.nn.SequentialLayer(down_sample))
            fusion_layers.append(tlx.nn.LayerList(fusion_layer))
        return fusion_layers

    def forward(self, inputs):
        if self.num_branches == 1:
            return [self.branches[0](inputs[0])]

        for i in range(self.num_branches):
            inputs[i] = self.branches[i](inputs[i])
        x = inputs
        x_fusion = []

        for i in range(len(self.fusion_layer)):
            y = x[0] if i == 0 else self.fusion_layer[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fusion_layer[i][j](x[j])
            x_fusion.append(self.relu(y))
        return x_fusion


class StackLayers(Module):
    def __init__(self, layers):
        super(StackLayers, self).__init__()
        self.layers_list = tlx.nn.LayerList(layers)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class PoseHighResolutionNet(Module):
    def __init__(self, config, name=""):
        super(PoseHighResolutionNet, self).__init__()
        self.config_params = config
        self.conv1 = tlx.nn.layers.Conv2d(out_channels=64,
                                          kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding="same",
                                          b_init=None,
                                          name=name + "/conv1",
                                          in_channels=3,
                                          )
        self.bn1 = tlx.nn.BatchNorm(num_features=64, decay=0.1, epsilon=1e-5)

        self.conv2 = tlx.nn.layers.Conv2d(out_channels=64,
                                          kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding="same",
                                          b_init=None,
                                          name=name + "/conv2",
                                          in_channels=64,
                                          )
        self.bn2 = tlx.nn.BatchNorm(num_features=64, decay=0.1, epsilon=1e-5)
        self.layer1 = make_bottleneck_layer(filter_num=64, blocks=4)
        self.transition1 = self.__make_transition_layer(previous_branches_num=1,
                                                        previous_channels=[256],
                                                        current_branches_num=self.config_params.get_stage("s2")[1],
                                                        current_channels=self.config_params.get_stage("s2")[0])
        self.stage2 = self.__make_stages("s2", self.config_params.get_stage("s2")[0])
        self.transition2 = self.__make_transition_layer(previous_branches_num=self.config_params.get_stage("s2")[1],
                                                        previous_channels=self.config_params.get_stage("s2")[0],
                                                        current_branches_num=self.config_params.get_stage("s3")[1],
                                                        current_channels=self.config_params.get_stage("s3")[0])
        self.stage3 = self.__make_stages("s3", self.config_params.get_stage("s3")[0])
        self.transition3 = self.__make_transition_layer(previous_branches_num=self.config_params.get_stage("s3")[1],
                                                        previous_channels=self.config_params.get_stage("s3")[0],
                                                        current_branches_num=self.config_params.get_stage("s4")[1],
                                                        current_channels=self.config_params.get_stage("s4")[0])
        self.stage4 = self.__make_stages("s4", self.config_params.get_stage("s4")[0], False)
        self.conv3 = tlx.nn.layers.Conv2d(out_channels=self.config_params.num_of_joints,
                                          kernel_size=(self.config_params.conv3_kernel, self.config_params.conv3_kernel),
                                          stride=(1, 1),
                                          padding="same",
                                          name=name + "/conv3",
                                          in_channels=None,
                                          )
        self.relu = tlx.nn.ReLU()

    def __make_stages(self, stage_name, in_channels, multi_scale_output=True):
        stage_info = self.config_params.get_stage(stage_name)
        channels, num_branches, num_modules, block, num_blocks, fusion_method = stage_info
        module_list = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            module_list.append(HighResolutionModule(num_branches=num_branches,
                                                    num_in_channels=in_channels,
                                                    num_channels=channels,
                                                    block=block,
                                                    num_blocks=num_blocks,
                                                    fusion_method=fusion_method,
                                                    multi_scale_output=reset_multi_scale_output))
        return StackLayers(layers=module_list)

    @staticmethod
    def __make_transition_layer(previous_branches_num, previous_channels, current_branches_num, current_channels):
        transition_layers = []
        for i in range(current_branches_num):
            if i < previous_branches_num:
                if current_channels[i] != previous_channels[i]:
                    transition_layers.append(
                        tlx.nn.SequentialLayer([
                            tlx.nn.layers.Conv2d(out_channels=current_channels[i],
                                                 kernel_size=(3, 3),
                                                 stride=(1, 1),
                                                 padding="same",
                                                 b_init=None,
                                                 in_channels=None,
                                                 ),
                            tlx.nn.BatchNorm(num_features=current_channels[i], decay=0.1, epsilon=1e-5),
                            tlx.nn.ReLU()
                        ])
                    )
                else:
                    transition_layers.append(NoneModule())
            else:
                down_sampling_layers = []
                for j in range(i + 1 - previous_branches_num):
                    in_channels = previous_channels[-1],
                    out_channels = current_channels[i] if j == i - previous_branches_num else in_channels
                    down_sampling_layers.append(
                        tlx.nn.SequentialLayer([
                            tlx.nn.layers.Conv2d(out_channels=out_channels,
                                                 kernel_size=(3, 3),
                                                 stride=(2, 2),
                                                 padding="same",
                                                 b_init=None,
                                                 in_channels=None,
                                                 ),
                            tlx.nn.BatchNorm(num_features=out_channels, decay=0.1, epsilon=1e-5),
                            tlx.nn.ReLU()
                        ])
                    )
                transition_layers.append(tlx.nn.SequentialLayer(down_sampling_layers))
        return tlx.nn.LayerList(transition_layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        feature_list = []
        for i in range(self.config_params.get_stage("s2")[1]):
            if not isinstance(self.transition1[i], NoneModule):
                feature_list.append(self.transition1[i](x))
            else:
                feature_list.append(x)
        y_list = self.stage2(feature_list)

        feature_list = []
        for i in range(self.config_params.get_stage("s3")[1]):
            if not isinstance(self.transition2[i], NoneModule):
                feature_list.append(self.transition2[i](y_list[-1]))
            else:
                feature_list.append(y_list[i])
        y_list = self.stage3(feature_list)

        feature_list = []
        for i in range(self.config_params.get_stage("s4")[1]):
            if not isinstance(self.transition3[i], NoneModule):
                feature_list.append(self.transition3[i](y_list[-1]))
            else:
                feature_list.append(y_list[i])

        y_list = self.stage4(feature_list)

        outputs = self.conv3(y_list[0])

        return outputs

