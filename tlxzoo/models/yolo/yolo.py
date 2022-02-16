"""YOLOv4 for MS-COCO.
# Reference:
- [tensorflow-yolov4-tflite](
    https://github.com/hunglc007/tensorflow-yolov4-tflite)
"""
from ..model import BaseModule
from ...utils.registry import Registers
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.nn import Mish
from tensorlayerx.nn import Conv2d, MaxPool2d, BatchNorm2d, ZeroPad2d, UpSampling2d, Concat, Elementwise
from tensorlayerx.nn import Module, SequentialLayer
from tensorlayerx import logging
from dataclasses import dataclass
from ...utils.output import BaseModelOutput, float_tensor

__all__ = ['YOLOv4']


@dataclass
class YOLOModelOutput(BaseModelOutput):
    soutput: float_tensor = None
    moutput: float_tensor = None
    loutput: float_tensor = None


class Convolutional(Module):
    """
    Create Convolution layer
    Because it is only a stack of reference layers, there is no build, so self._built=True
    """

    def __init__(self, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky', name=None):
        super(Convolutional, self).__init__()
        self.act = activate
        self.act_type = activate_type
        self.downsample = downsample
        self.bn = bn
        self._built = True
        if downsample:
            padding = 'VALID'
            strides = 2
        else:
            strides = 1
            padding = 'SAME'

        if bn:
            b_init = None
        else:
            b_init = tlx.nn.initializers.constant(value=0.0)

        self.zeropad = ZeroPad2d(((1, 0), (1, 0)))
        self.conv = Conv2d(
            n_filter=filters_shape[-1], in_channels=filters_shape[2], filter_size=(filters_shape[0], filters_shape[1]),
            strides=(strides, strides), padding=padding, b_init=b_init, name=name
        )

        if bn:
            if activate == True:
                if activate_type == 'leaky':
                    self.batchnorm2d = BatchNorm2d(act='leaky_relu0.1', num_features=filters_shape[-1])
                elif activate_type == 'mish':
                    self.batchnorm2d = BatchNorm2d(act=Mish, num_features=filters_shape[-1])
            else:
                self.batchnorm2d = BatchNorm2d(act=None, num_features=filters_shape[-1])

    def forward(self, input):
        if self.downsample:
            input = self.zeropad(input)

        output = self.conv(input)

        if self.bn:
            output = self.batchnorm2d(output)
        return output


class ResidualBlock(Module):

    def __init__(self, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        super(ResidualBlock, self).__init__()
        self.conv1 = Convolutional(filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        self.conv2 = Convolutional(filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)
        self.add = Elementwise(tlx.add)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.add([inputs, output])
        return output


def residual_block_num(num, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    residual_list = []
    for i in range(num):
        residual_list.append(ResidualBlock(input_channel, filter_num1, filter_num2, activate_type=activate_type))
    return SequentialLayer(residual_list)


class CspdarkNet53(Module):

    def __init__(self):
        super(CspdarkNet53, self).__init__()
        self._built = True
        self.conv1_1 = Convolutional((3, 3, 3, 32), activate_type='mish')
        self.conv1_2 = Convolutional((3, 3, 32, 64), downsample=True, activate_type='mish')
        self.conv1_3 = Convolutional((1, 1, 64, 64), activate_type='mish', name='conv_rote_block_1')
        self.conv1_4 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.residual_1 = residual_block_num(1, 64, 32, 64, activate_type="mish")

        self.conv2_1 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.concat = Concat()
        self.conv2_2 = Convolutional((1, 1, 128, 64), activate_type='mish')
        self.conv2_3 = Convolutional((3, 3, 64, 128), downsample=True, activate_type='mish')
        self.conv2_4 = Convolutional((1, 1, 128, 64), activate_type='mish', name='conv_rote_block_2')
        self.conv2_5 = Convolutional((1, 1, 128, 64), activate_type='mish')
        self.residual_2 = residual_block_num(2, 64, 64, 64, activate_type='mish')

        self.conv3_1 = Convolutional((1, 1, 64, 64), activate_type='mish')
        self.conv3_2 = Convolutional((1, 1, 128, 128), activate_type='mish')
        self.conv3_3 = Convolutional((3, 3, 128, 256), downsample=True, activate_type='mish')
        self.conv3_4 = Convolutional((1, 1, 256, 128), activate_type='mish', name='conv_rote_block_3')
        self.conv3_5 = Convolutional((1, 1, 256, 128), activate_type='mish')
        self.residual_3 = residual_block_num(8, 128, 128, 128, activate_type="mish")

        self.conv4_1 = Convolutional((1, 1, 128, 128), activate_type='mish')
        self.conv4_2 = Convolutional((1, 1, 256, 256), activate_type='mish')
        self.conv4_3 = Convolutional((3, 3, 256, 512), downsample=True, activate_type='mish')
        self.conv4_4 = Convolutional((1, 1, 512, 256), activate_type='mish', name='conv_rote_block_4')
        self.conv4_5 = Convolutional((1, 1, 512, 256), activate_type='mish')
        self.residual_4 = residual_block_num(8, 256, 256, 256, activate_type="mish")

        self.conv5_1 = Convolutional((1, 1, 256, 256), activate_type='mish')
        self.conv5_2 = Convolutional((1, 1, 512, 512), activate_type='mish')
        self.conv5_3 = Convolutional((3, 3, 512, 1024), downsample=True, activate_type='mish')
        self.conv5_4 = Convolutional((1, 1, 1024, 512), activate_type='mish', name='conv_rote_block_5')
        self.conv5_5 = Convolutional((1, 1, 1024, 512), activate_type='mish')
        self.residual_5 = residual_block_num(4, 512, 512, 512, activate_type="mish")

        self.conv6_1 = Convolutional((1, 1, 512, 512), activate_type='mish')
        self.conv6_2 = Convolutional((1, 1, 1024, 1024), activate_type='mish')
        self.conv6_3 = Convolutional((1, 1, 1024, 512))
        self.conv6_4 = Convolutional((3, 3, 512, 1024))
        self.conv6_5 = Convolutional((1, 1, 1024, 512))

        self.maxpool1 = MaxPool2d(filter_size=(13, 13), strides=(1, 1))
        self.maxpool2 = MaxPool2d(filter_size=(9, 9), strides=(1, 1))
        self.maxpool3 = MaxPool2d(filter_size=(5, 5), strides=(1, 1))

        self.conv7_1 = Convolutional((1, 1, 2048, 512))
        self.conv7_2 = Convolutional((3, 3, 512, 1024))
        self.conv7_3 = Convolutional((1, 1, 1024, 512))

    def forward(self, input_data):
        input_data = self.conv1_1(input_data)
        input_data = self.conv1_2(input_data)
        route = input_data
        route = self.conv1_3(route)
        input_data = self.conv1_4(input_data)
        input_data = self.residual_1(input_data)

        input_data = self.conv2_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv2_2(input_data)
        input_data = self.conv2_3(input_data)
        route = input_data
        route = self.conv2_4(route)
        input_data = self.conv2_5(input_data)
        input_data = self.residual_2(input_data)

        input_data = self.conv3_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv3_2(input_data)
        input_data = self.conv3_3(input_data)
        route = input_data
        route = self.conv3_4(route)
        input_data = self.conv3_5(input_data)
        input_data = self.residual_3(input_data)

        input_data = self.conv4_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv4_2(input_data)
        route_1 = input_data
        input_data = self.conv4_3(input_data)
        route = input_data
        route = self.conv4_4(route)
        input_data = self.conv4_5(input_data)
        input_data = self.residual_4(input_data)

        input_data = self.conv5_1(input_data)
        input_data = self.concat([input_data, route])
        input_data = self.conv5_2(input_data)
        route_2 = input_data
        input_data = self.conv5_3(input_data)
        route = input_data
        route = self.conv5_4(route)
        input_data = self.conv5_5(input_data)
        input_data = self.residual_5(input_data)

        input_data = self.conv6_1(input_data)
        input_data = self.concat([input_data, route])

        input_data = self.conv6_2(input_data)
        input_data = self.conv6_3(input_data)
        input_data = self.conv6_4(input_data)
        input_data = self.conv6_5(input_data)

        maxpool1 = self.maxpool1(input_data)
        maxpool2 = self.maxpool2(input_data)
        maxpool3 = self.maxpool3(input_data)
        input_data = self.concat([maxpool1, maxpool2, maxpool3, input_data])

        input_data = self.conv7_1(input_data)
        input_data = self.conv7_2(input_data)
        input_data = self.conv7_3(input_data)

        return route_1, route_2, input_data


@Registers.models.register
class YOLOv4(BaseModule):

    def __init__(self, config):
        super(YOLOv4, self).__init__(config)
        self.config = config
        # self.num_class = self.config.num_class
        self.cspdarnnet = CspdarkNet53()

        self.conv1_1 = Convolutional(self.config.conv1_1_filters_shape)
        self.upsamle = UpSampling2d(scale=2)
        self.conv1_2 = Convolutional(self.config.conv1_2_filters_shape, name='conv_yolo_1')
        self.concat = Concat()

        self.conv2_1 = Convolutional(self.config.conv2_1_filters_shape)
        self.conv2_2 = Convolutional(self.config.conv2_2_filters_shape)
        self.conv2_3 = Convolutional(self.config.conv2_3_filters_shape)
        self.conv2_4 = Convolutional(self.config.conv2_4_filters_shape)
        self.conv2_5 = Convolutional(self.config.conv2_5_filters_shape)

        self.conv3_1 = Convolutional(self.config.conv3_1_filters_shape)
        self.conv3_2 = Convolutional(self.config.conv3_2_filters_shape, name='conv_yolo_2')

        self.conv4_1 = Convolutional(self.config.conv4_1_filters_shape)
        self.conv4_2 = Convolutional(self.config.conv4_2_filters_shape)
        self.conv4_3 = Convolutional(self.config.conv4_3_filters_shape)
        self.conv4_4 = Convolutional(self.config.conv4_4_filters_shape)
        self.conv4_5 = Convolutional(self.config.conv4_5_filters_shape)

        self.conv5_1 = Convolutional(self.config.conv5_1_filters_shape, name='conv_route_1')
        # self.conv5_2 = Convolutional((1, 1, 256, 3 * (self.num_class + 5)), activate=False, bn=False)

        self.conv6_1 = Convolutional(self.config.conv6_1_filters_shape, downsample=True, name='conv_route_2')
        self.conv6_2 = Convolutional(self.config.conv6_2_filters_shape)
        self.conv6_3 = Convolutional(self.config.conv6_3_filters_shape)
        self.conv6_4 = Convolutional(self.config.conv6_4_filters_shape)
        self.conv6_5 = Convolutional(self.config.conv6_5_filters_shape)
        self.conv6_6 = Convolutional(self.config.conv6_6_filters_shape)

        self.conv7_1 = Convolutional(self.config.conv7_1_filters_shape, name='conv_route_3')
        # self.conv7_2 = Convolutional((1, 1, 512, 3 * (self.num_class + 5)), activate=False, bn=False)
        self.conv7_3 = Convolutional(self.config.conv7_3_filters_shape, downsample=True, name='conv_route_4')

        self.conv8_1 = Convolutional(self.config.conv8_1_filters_shape)
        self.conv8_2 = Convolutional(self.config.conv8_2_filters_shape)
        self.conv8_3 = Convolutional(self.config.conv8_3_filters_shape)
        self.conv8_4 = Convolutional(self.config.conv8_4_filters_shape)
        self.conv8_5 = Convolutional(self.config.conv8_5_filters_shape)

        self.conv9_1 = Convolutional(self.config.conv9_1_filters_shape)
        # self.conv9_2 = Convolutional((1, 1, 1024, 3 * (self.num_class + 5)), activate=False, bn=False)

    def forward(self, inputs):
        route_1, route_2, conv = self.cspdarnnet(inputs)

        route = conv
        conv = self.conv1_1(conv)
        conv = self.upsamle(conv)
        route_2 = self.conv1_2(route_2)
        conv = self.concat([route_2, conv])

        conv = self.conv2_1(conv)
        conv = self.conv2_2(conv)
        conv = self.conv2_3(conv)
        conv = self.conv2_4(conv)
        conv = self.conv2_5(conv)

        route_2 = conv
        conv = self.conv3_1(conv)
        conv = self.upsamle(conv)
        route_1 = self.conv3_2(route_1)
        conv = self.concat([route_1, conv])

        conv = self.conv4_1(conv)
        conv = self.conv4_2(conv)
        conv = self.conv4_3(conv)
        conv = self.conv4_4(conv)
        conv = self.conv4_5(conv)

        route_1 = conv
        sconv = self.conv5_1(conv)
        # conv_sbbox = self.conv5_2(conv)

        conv = self.conv6_1(route_1)
        conv = self.concat([conv, route_2])

        conv = self.conv6_2(conv)
        conv = self.conv6_3(conv)
        conv = self.conv6_4(conv)
        conv = self.conv6_5(conv)
        conv = self.conv6_6(conv)

        route_2 = conv
        mconv = self.conv7_1(conv)
        # conv_mbbox = self.conv7_2(conv)
        conv = self.conv7_3(route_2)
        conv = self.concat([conv, route])

        conv = self.conv8_1(conv)
        conv = self.conv8_2(conv)
        conv = self.conv8_3(conv)
        conv = self.conv8_4(conv)
        conv = self.conv8_5(conv)

        lconv = self.conv9_1(conv)
        # conv_lbbox = self.conv9_2(conv)

        return YOLOModelOutput(soutput=sconv, moutput=mconv, loutput=lconv)
