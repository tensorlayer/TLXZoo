import tensorlayerx as tlx
from tensorlayerx import nn
import numpy as np


class Block(nn.Module):
    def __init__(self, filters, kernel_size=3, stride=1, conv_shortcut=True, name=""):
        super(Block, self).__init__(name=name)

        self.conv_shortcut = conv_shortcut

        if conv_shortcut:
            self.conv_0 = nn.Conv2d(out_channels=4 * filters,
                                    kernel_size=(1, 1), stride=(stride, stride),
                                    padding="valid",
                                    name=name + '_0_conv')
            self.bn_0 = nn.BatchNorm(epsilon=1.001e-5, name=name + '_0_bn')

        self.conv_1 = nn.Conv2d(out_channels=filters,
                                kernel_size=(1, 1), stride=(stride, stride),
                                padding="valid",
                                name=name + '_1_conv')
        self.bn_1 = nn.BatchNorm(epsilon=1.001e-5, name=name + '_1_bn')

        self.relu = tlx.ReLU()

        self.conv_2 = nn.Conv2d(out_channels=filters,
                                kernel_size=(kernel_size, kernel_size),
                                padding='SAME',
                                name=name + '_2_conv')
        self.bn_2 = nn.BatchNorm(epsilon=1.001e-5, name=name + '_2_bn')

        self.conv_3 = nn.Conv2d(out_channels=4 * filters,
                                kernel_size=(1, 1),
                                padding="valid",
                                name=name + '_3_conv')
        self.bn_3 = nn.BatchNorm(epsilon=1.001e-5, name=name + '_3_bn')

    def forward(self, inputs, extra_return_tensors_index=None, index=0, extra_return_tensors=None):
        if extra_return_tensors_index is not None and extra_return_tensors is None:
            extra_return_tensors = []

        if self.conv_shortcut:
            shortcut = self.conv_0(inputs)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index,
                                                           shortcut, extra_return_tensors)
            shortcut = self.bn_0(shortcut)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index,
                                                           shortcut, extra_return_tensors)
        else:
            shortcut = inputs

        x = self.conv_1(inputs)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.bn_1(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.relu(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        x = self.conv_2(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.bn_2(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.relu(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        x = self.conv_3(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.bn_3(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        x = tlx.add(shortcut, x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.relu(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        return x, index, extra_return_tensors


class Stack(nn.Module):
    def __init__(self, filters, blocks_num, stride1=2, name="stack"):
        super(Stack, self).__init__(name=name)

        blocks = [Block(filters, stride=stride1, name=name + '_block1')]

        for i in range(2, blocks_num + 1):
            blocks.append(Block(filters, conv_shortcut=False, name=name + '_block' + str(i)))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs, extra_return_tensors_index=None, index=0, extra_return_tensors=None):
        if extra_return_tensors_index is not None and extra_return_tensors is None:
            extra_return_tensors = []

        for block in self.blocks:
            inputs, index, extra_return_tensors = block(inputs, extra_return_tensors_index=extra_return_tensors_index,
                                                        index=index, extra_return_tensors=extra_return_tensors)
        return inputs, index, extra_return_tensors


def add_extra_tensor(index, extra_return_tensors_index, tensor, extra_return_tensors):
    if extra_return_tensors_index is None:
        return index + 1, extra_return_tensors

    if index in extra_return_tensors_index:
        return index + 1, extra_return_tensors + [tensor]

    return index + 1, extra_return_tensors


class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x, data_format="channels_last"):
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if len(tlx.get_tensor_shape(x)) == 3:
                x = x[::-1, ...]
            else:
                x = x[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        std = None

        mean_tensor = tlx.constant(-np.array(mean))

        if data_format == "channels_last":
            data_format = 'NHWC'
        if data_format == 'channels_first':
            data_format = 'NCHW'
        # Zero-center by mean pixel
        if x.dtype != mean_tensor.dtype:
            x = tlx.bias_add(x, tlx.cast(mean_tensor, x.dtype), data_format=data_format)
        else:
            x = tlx.bias_add(x, mean_tensor, data_format)
        if std is not None:
            x /= std
        return x


class ResNet50(nn.Module):
    def __init__(self, input_shape, preact=False, use_bias=True, use_preprocess=True, include_top=False,
                 pooling=None, num_labels=1000, name="resnet50"):
        """
        :param input_shape: optional shape tuple, E.g. `(None, 200, 200, 3)` would be one valid value.
        :param preact: whether to use pre-activation or not (True for ResNetV2, False for ResNet and ResNeXt).
        :param use_bias: whether to use biases for convolutional layers or not (True for ResNet and ResNetV2, False for ResNeXt).
        :param use_preprocess: whether use data preprocess in backbone.
        :param include_top: whether to include the fully-connected layer at the top of the network.
        :param pooling: optional pooling mode for feature extraction
        :param num_labels: optional number of classes to classify images
        :param name: module name
        """
        super(ResNet50, self).__init__(name=name)
        self.preact = preact

        if use_preprocess:
            self.preprocess = Preprocess()
        else:
            self.preprocess = None

        self.conv1_pad = tlx.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv1_conv = nn.Conv2d(out_channels=64, kernel_size=(7, 7),
                                    stride=(2, 2),
                                    b_init="constant" if use_bias else None,
                                    padding="valid",
                                    name="conv1_conv",
                                    )
        self.relu = tlx.ReLU()

        if not preact:
            self.conv1_bn = nn.BatchNorm(0.99, epsilon=1.001e-5, name='conv1_bn')

        self.pool1_pad = tlx.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.pool1_pool = tlx.nn.MaxPool2d(kernel_size=(3, 3), padding="valid",
                                           stride=(2, 2), name='pool1_pool')

        stacks = [Stack(64, 3, stride1=1, name="conv2"),
                  Stack(128, 4, name="conv3"),
                  Stack(256, 6, name="conv4"),
                  Stack(512, 3, name="conv5"),
                  ]
        self.stacks = nn.ModuleList(stacks)

        if preact:
            self.post_bn = nn.BatchNorm(epsilon=1.001e-5, name='post_bn')

        self.include_top = include_top
        self.pooling = pooling
        if include_top:
            self.pool = tlx.nn.GlobalMeanPool2d(name='avg_pool')
            self.include_top_dense = tlx.nn.Linear(out_features=num_labels, act=None, name='predictions')
        else:
            if pooling == 'avg':
                self.pool = tlx.nn.GlobalMeanPool2d(name='avg_pool')
            elif pooling == 'max':
                self.pool = tlx.nn.GlobalMaxPool2d(name='max_pool')

        if input_shape is not None:
            self.build(input_shape)

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, inputs, extra_return_tensors_index=None, index=0, extra_return_tensors=None):
        if extra_return_tensors_index is not None and extra_return_tensors is None:
            extra_return_tensors = []
        if self.preprocess:
            inputs = self.preprocess(inputs)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, inputs, extra_return_tensors)
        x = self.conv1_pad(inputs)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.conv1_conv(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        if not self.preact:
            x = self.conv1_bn(x)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
            x = self.relu(x)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        x = self.pool1_pad(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        x = self.pool1_pool(x)
        index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
        for stack in self.stacks:
            x, index, extra_return_tensors = stack(x, extra_return_tensors_index=extra_return_tensors_index,
                                                   index=index, extra_return_tensors=extra_return_tensors)

        if self.preact:
            x = self.post_bn(x)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)
            x = self.relu(x)
            index, extra_return_tensors = add_extra_tensor(index, extra_return_tensors_index, x, extra_return_tensors)

        if self.include_top:
            x = self.pool(x)
            x = self.include_top_dense(x)
        else:
            if self.pooling == 'avg':
                x = self.pool(x)
            elif self.pooling == 'max':
                x = self.pool(x)
        if extra_return_tensors is None:
            return x
        return x, index, extra_return_tensors
