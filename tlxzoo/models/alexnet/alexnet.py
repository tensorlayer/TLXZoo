'''
Author: jianzhnie
Date: 2022-01-28 10:39:08
LastEditTime: 2022-01-28 10:56:46
LastEditors: jianzhnie
Description:

'''

from dataclasses import dataclass

from tensorlayerx.nn import AdaptiveMeanPool2d, Conv2d, Dense, Dropout, MaxPool2d, ReLU, SequentialLayer

from ...utils.output import BaseModelOutput
from ...utils.registry import Registers
from ..model import BaseModule
from .config_alexnet import AlexNetModelConfig


@dataclass
class AlexNetModelOutput(BaseModelOutput):
    ...


@Registers.models.register
class AlexNet(BaseModule):
    config_class = AlexNetModelConfig

    def __init__(self, config, **kwargs):
        super(AlexNet, self).__init__(config)
        num_classes = config.num_classes
        dropout = config.dropout

        layer_list = []
        layer_list.append(
            Conv2d(
                in_channels=3,
                n_filter=64,
                filter_size=(11, 11),
                strides=(4, 4),
                padding=2))
        layer_list.append(ReLU())
        layer_list.append(MaxPool2d(filter_size=(3, 3), strides=(2, 2)))
        layer_list.append(
            Conv2d(
                in_channels=64, n_filter=192, filter_size=(5, 5), padding=2))
        layer_list.append(ReLU())
        layer_list.append(MaxPool2d(filter_size=(3, 3), strides=(2, 2)))
        layer_list.append(
            Conv2d(
                in_channels=192, n_filter=384, filter_size=(3, 3), padding=1))
        layer_list.append(ReLU())
        layer_list.append(
            Conv2d(
                in_channels=384, n_filter=256, filter_size=(3, 3), padding=1))
        layer_list.append(ReLU())
        layer_list.append(
            Conv2d(
                in_channels=256, n_filter=256, filter_size=(3, 3), padding=1))
        layer_list.append(ReLU())
        layer_list.append(MaxPool2d(filter_size=(3, 3), strides=(2, 2)))
        self.features = SequentialLayer(layer_list)
        self.avgpool = AdaptiveMeanPool2d((6, 6))

        output_list = []
        output_list.append(Dropout(keep=1 - dropout))
        output_list.append(Dense(in_channels=256 * 6 * 6, n_units=4096))
        output_list.append(ReLU())
        output_list.append(Dropout(keep=1 - dropout))
        output_list.append(Dense(in_channels=4096, n_units=4096))
        output_list.append(ReLU())
        output_list.append(Dense(in_channels=4096, n_units=num_classes))

        self.classifier = SequentialLayer(layer_list)

    def forward(self, pixels):
        """
        pixel : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        # pixels = pixels * 255 - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
        out = self.features(pixels)
        out = self.avgpool(out)
        return AlexNetModelOutput(output=out)
