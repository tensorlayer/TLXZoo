from ..model import BaseModule
from ...utils.registry import Registers

import os
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn import (BatchNorm, Conv2d, Dense, Flatten, ReLU, Dropout,
                             SequentialLayer, MaxPool2d, AdaptiveMeanPool2d)
from ...utils.output import BaseModelOutput
from dataclasses import dataclass
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
            Conv2d(in_channels=3,
                   n_filter=64,
                   filter_size=(11, 11),
                   strides=(4, 4),
                   padding=2))
        layer_list.append(ReLU(inplace=True))
        layer_list.append(MaxPool2d(filter_size=(3, 3), stride=(2, 2)))
        layer_list.append(
            Conv2d(in_channels=64, n_filter=192, filter_size=(5, 5),
                   padding=2))
        layer_list.append(ReLU(inplace=True))
        layer_list.append(MaxPool2d(filter_size=(3, 3), stride=(2, 2)))
        layer_list.append(
            Conv2d(in_channels=192,
                   n_filter=384,
                   filter_size=(3, 3),
                   padding=1))
        layer_list.append(ReLU(inplace=True))
        layer_list.append(
            Conv2d(in_channels=384,
                   n_filter=256,
                   filter_size=(3, 3),
                   padding=1))
        layer_list.append(ReLU(inplace=True))
        layer_list.append(
            Conv2d(in_channels=256,
                   n_filter=256,
                   filter_size=(3, 3),
                   padding=1))
        layer_list.append(ReLU(inplace=True))
        layer_list.append(MaxPool2d(filter_size=(3, 3), stride=(2, 2)))
        self.features = SequentialLayer(layer_list)
        self.avgpool = AdaptiveMeanPool2d((6, 6))

        output_list = []
        output_list.append(Dropout(p=dropout))
        output_list.append(Dense(inchannles=256 * 6 * 6, n_units=4096))
        output_list.append(ReLU(inplace=True))
        output_list.append(Dropout(p=dropout))
        output_list.append(Dense(inchannles=4096, n_units=4096))
        output_list.append(ReLU(inplace=True))
        output_list.append(Dense(inchannles=4096, n_units=num_classes))

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