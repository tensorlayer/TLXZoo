"""YOLOv4 for MS-COCO.
# Reference:
- [tensorflow-yolov4-tflite](
    https://github.com/hunglc007/tensorflow-yolov4-tflite)
"""
from ...utils.registry import Registers
import tensorlayerx as tlx
from tensorlayerx.nn import Mish
from tensorlayerx.nn import Conv2d, MaxPool2d, BatchNorm2d, ZeroPad2d, UpSampling2d, Concat, Elementwise
from tensorlayerx.nn import Module, SequentialLayer
from tensorlayerx import logging
import numpy as np

__all__ = ['YOLOv4']

random_normal_initializer = tlx.initializers.RandomNormal(stddev=0.01)


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
            out_channels=filters_shape[-1], in_channels=filters_shape[2],
            kernel_size=(filters_shape[0], filters_shape[1]),
            stride=(strides, strides), padding=padding, b_init=b_init, name=name, W_init=random_normal_initializer
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

        self.maxpool1 = MaxPool2d(kernel_size=(13, 13), stride=(1, 1))
        self.maxpool2 = MaxPool2d(kernel_size=(9, 9), stride=(1, 1))
        self.maxpool3 = MaxPool2d(kernel_size=(5, 5), stride=(1, 1))

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


class YOLOv4(tlx.nn.Module):

    def __init__(self, conv1_1_filters_shape=(1, 1, 512, 256),
                 conv1_2_filters_shape=(1, 1, 512, 256),
                 conv2_1_filters_shape=(1, 1, 512, 256),
                 conv2_2_filters_shape=(3, 3, 256, 512),
                 conv2_3_filters_shape=(1, 1, 512, 256),
                 conv2_4_filters_shape=(3, 3, 256, 512),
                 conv2_5_filters_shape=(1, 1, 512, 256),
                 conv3_1_filters_shape=(1, 1, 256, 128),
                 conv3_2_filters_shape=(1, 1, 256, 128),
                 conv4_1_filters_shape=(1, 1, 256, 128),
                 conv4_2_filters_shape=(3, 3, 128, 256),
                 conv4_3_filters_shape=(1, 1, 256, 128),
                 conv4_4_filters_shape=(3, 3, 128, 256),
                 conv4_5_filters_shape=(1, 1, 256, 128),
                 conv5_1_filters_shape=(3, 3, 128, 256),
                 conv6_1_filters_shape=(3, 3, 128, 256),
                 conv6_2_filters_shape=(1, 1, 512, 256),
                 conv6_3_filters_shape=(3, 3, 256, 512),
                 conv6_4_filters_shape=(1, 1, 512, 256),
                 conv6_5_filters_shape=(3, 3, 256, 512),
                 conv6_6_filters_shape=(1, 1, 512, 256),
                 conv7_1_filters_shape=(3, 3, 256, 512),
                 conv7_3_filters_shape=(3, 3, 256, 512),
                 conv8_1_filters_shape=(1, 1, 1024, 512),
                 conv8_2_filters_shape=(3, 3, 512, 1024),
                 conv8_3_filters_shape=(1, 1, 1024, 512),
                 conv8_4_filters_shape=(3, 3, 512, 1024),
                 conv8_5_filters_shape=(1, 1, 1024, 512),
                 conv9_1_filters_shape=(3, 3, 512, 1024),
                 num_labels=80,
                 sconv_filters_shape=(1, 1, 256),
                 mconv_filters_shape=(1, 1, 512),
                 lconv_filters_shape=(1, 1, 1024),
                 strides=(8, 16, 32),
                 anchors=(12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401),
                 xyscale=(1.2, 1.1, 1.05),
                 iou_loss_thresh=0.5,
                 train_input_size=416,
                 ):
        """
        :param num_labels: (:obj:`int`, `optional`, defaults to 80):
            Num of labels.
        :param strides: (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(8, 16, 32)`):
            A tuple of integers defining the stride of each convolutional layer in the feature extractor.
        :param anchors: (:obj:`Tuple[int]`, `optional`, defaults to :obj:`(12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401)`):
            A tuple of integers defining the anchor.
        :param xyscale: (:obj:`Tuple[float]`, `optional`, defaults to :obj:`(1.2, 1.1, 1.05)`):
            A tuple of integers defining the xy scale.
        :param iou_loss_thresh: (:obj:`float`, `optional`, defaults to 0.5):
            Thresh for computing iou loss.
        """
        super(YOLOv4, self).__init__()
        self.conv1_1_filters_shape = conv1_1_filters_shape
        self.conv1_2_filters_shape = conv1_2_filters_shape
        self.conv2_1_filters_shape = conv2_1_filters_shape
        self.conv2_2_filters_shape = conv2_2_filters_shape
        self.conv2_3_filters_shape = conv2_3_filters_shape
        self.conv2_4_filters_shape = conv2_4_filters_shape
        self.conv2_5_filters_shape = conv2_5_filters_shape
        self.conv3_1_filters_shape = conv3_1_filters_shape
        self.conv3_2_filters_shape = conv3_2_filters_shape
        self.conv4_1_filters_shape = conv4_1_filters_shape
        self.conv4_2_filters_shape = conv4_2_filters_shape
        self.conv4_3_filters_shape = conv4_3_filters_shape
        self.conv4_4_filters_shape = conv4_4_filters_shape
        self.conv4_5_filters_shape = conv4_5_filters_shape
        self.conv5_1_filters_shape = conv5_1_filters_shape
        self.conv6_1_filters_shape = conv6_1_filters_shape
        self.conv6_2_filters_shape = conv6_2_filters_shape
        self.conv6_3_filters_shape = conv6_3_filters_shape
        self.conv6_4_filters_shape = conv6_4_filters_shape
        self.conv6_5_filters_shape = conv6_5_filters_shape
        self.conv6_6_filters_shape = conv6_6_filters_shape
        self.conv7_1_filters_shape = conv7_1_filters_shape
        self.conv7_3_filters_shape = conv7_3_filters_shape
        self.conv8_1_filters_shape = conv8_1_filters_shape
        self.conv8_2_filters_shape = conv8_2_filters_shape
        self.conv8_3_filters_shape = conv8_3_filters_shape
        self.conv8_4_filters_shape = conv8_4_filters_shape
        self.conv8_5_filters_shape = conv8_5_filters_shape
        self.conv9_1_filters_shape = conv9_1_filters_shape
        self.sconv_filters_shape = tuple(list(sconv_filters_shape) + [3 * (num_labels + 5)])
        self.mconv_filters_shape = tuple(list(mconv_filters_shape) + [3 * (num_labels + 5)])
        self.lconv_filters_shape = tuple(list(lconv_filters_shape) + [3 * (num_labels + 5)])
        self.strides = np.array(strides)
        self.anchors = np.array(anchors).reshape(3, 3, 2)
        self.xyscale = xyscale
        self.train_input_size = train_input_size
        self.iou_loss_thresh = iou_loss_thresh
        self.num_labels = num_labels

        self.cspdarnnet = CspdarkNet53()

        self.conv1_1 = Convolutional(self.conv1_1_filters_shape)
        self.upsamle = UpSampling2d(scale=2)
        self.conv1_2 = Convolutional(self.conv1_2_filters_shape, name='conv_yolo_1')
        self.concat = Concat()

        self.conv2_1 = Convolutional(self.conv2_1_filters_shape)
        self.conv2_2 = Convolutional(self.conv2_2_filters_shape)
        self.conv2_3 = Convolutional(self.conv2_3_filters_shape)
        self.conv2_4 = Convolutional(self.conv2_4_filters_shape)
        self.conv2_5 = Convolutional(self.conv2_5_filters_shape)

        self.conv3_1 = Convolutional(self.conv3_1_filters_shape)
        self.conv3_2 = Convolutional(self.conv3_2_filters_shape, name='conv_yolo_2')

        self.conv4_1 = Convolutional(self.conv4_1_filters_shape)
        self.conv4_2 = Convolutional(self.conv4_2_filters_shape)
        self.conv4_3 = Convolutional(self.conv4_3_filters_shape)
        self.conv4_4 = Convolutional(self.conv4_4_filters_shape)
        self.conv4_5 = Convolutional(self.conv4_5_filters_shape)

        self.conv5_1 = Convolutional(self.conv5_1_filters_shape, name='conv_route_1')
        # self.conv5_2 = Convolutional((1, 1, 256, 3 * (self.num_class + 5)), activate=False, bn=False)

        self.conv6_1 = Convolutional(self.conv6_1_filters_shape, downsample=True, name='conv_route_2')
        self.conv6_2 = Convolutional(self.conv6_2_filters_shape)
        self.conv6_3 = Convolutional(self.conv6_3_filters_shape)
        self.conv6_4 = Convolutional(self.conv6_4_filters_shape)
        self.conv6_5 = Convolutional(self.conv6_5_filters_shape)
        self.conv6_6 = Convolutional(self.conv6_6_filters_shape)

        self.conv7_1 = Convolutional(self.conv7_1_filters_shape, name='conv_route_3')
        # self.conv7_2 = Convolutional((1, 1, 512, 3 * (self.num_class + 5)), activate=False, bn=False)
        self.conv7_3 = Convolutional(self.conv7_3_filters_shape, downsample=True, name='conv_route_4')

        self.conv8_1 = Convolutional(self.conv8_1_filters_shape)
        self.conv8_2 = Convolutional(self.conv8_2_filters_shape)
        self.conv8_3 = Convolutional(self.conv8_3_filters_shape)
        self.conv8_4 = Convolutional(self.conv8_4_filters_shape)
        self.conv8_5 = Convolutional(self.conv8_5_filters_shape)

        self.conv9_1 = Convolutional(self.conv9_1_filters_shape)
        # self.conv9_2 = Convolutional((1, 1, 1024, 3 * (self.num_class + 5)), activate=False, bn=False)

        self.sconv = Convolutional(self.sconv_filters_shape, activate=False, bn=False)
        self.mconv = Convolutional(self.mconv_filters_shape, activate=False, bn=False)
        self.lconv = Convolutional(self.lconv_filters_shape, activate=False, bn=False)

    def loss_fn(self, pred_result, target):
        giou_loss = conf_loss = prob_loss = 0

        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            target_0 = tlx.cast(target[i][0], tlx.float32)
            target_1 = tlx.cast(target[i][1], tlx.float32)
            loss_items = compute_loss(pred, conv, target_0, target_1, STRIDES=self.strides,
                                      NUM_CLASS=self.num_labels,
                                      IOU_LOSS_THRESH=self.iou_loss_thresh, i=i)
            if tlx.is_nan(loss_items[0]):
                giou_loss += 0
            else:
                giou_loss += loss_items[0]

            if tlx.is_nan(loss_items[1]):
                conf_loss += 0
            else:
                conf_loss += loss_items[1]

            if tlx.is_nan(loss_items[2]):
                prob_loss += 0
            else:
                prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        return total_loss

    def forward(self, inputs):
        inputs = tlx.cast(inputs, tlx.float32)
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

        sbbox = self.sconv(sconv)
        mbbox = self.mconv(mconv)
        lbbox = self.lconv(lconv)

        feature_maps = [sbbox, mbbox, lbbox]

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, self.train_input_size // 8, self.num_labels,
                                           self.strides, self.anchors, i, self.xyscale)
            elif i == 1:
                bbox_tensor = decode_train(fm, self.train_input_size // 16, self.num_labels,
                                           self.strides, self.anchors, i, self.xyscale)
            else:
                bbox_tensor = decode_train(fm, self.train_input_size // 32, self.num_labels,
                                           self.strides, self.anchors, i, self.xyscale)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

        return bbox_tensors


def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE):
    conv_output = tlx.reshape(conv_output, (conv_output.shape[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tlx.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tlx.meshgrid(tlx.range(output_size), tlx.range(output_size))
    xy_grid = tlx.expand_dims(tlx.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tlx.tile(tlx.expand_dims(xy_grid, axis=0), [conv_output.shape[0], 1, 1, 3, 1])

    xy_grid = tlx.cast(xy_grid, tlx.float32)

    pred_xy = ((tlx.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
    pred_wh = (tlx.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tlx.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tlx.sigmoid(conv_raw_conf)
    pred_prob = tlx.sigmoid(conv_raw_prob)

    return tlx.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tlx.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tlx.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tlx.cast(input_size, tlx.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tlx.expand_dims(tlx.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tlx.cast(max_iou < IOU_LOSS_THRESH, tlx.float32)

    conf_focal = tlx.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                  logits=conv_raw_conf)
            +
            respond_bgd * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                 logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tlx.ops.load_backend.sigmoid_cross_entropy_with_logits(labels=label_prob,
                                                                                      logits=conv_raw_prob)

    giou_loss = tlx.reduce_mean(tlx.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tlx.reduce_mean(tlx.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tlx.reduce_mean(tlx.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tlx.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tlx.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tlx.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tlx.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tlx.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tlx.divide(inter_area, union_area)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tlx.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tlx.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tlx.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tlx.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tlx.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tlx.divide(inter_area, union_area)

    enclose_left_up = tlx.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tlx.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - tlx.divide(enclose_area - union_area, enclose_area)

    return giou
