"""
VGG for ImageNet.
Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper "Very Deep Convolutional Networks for
Large-Scale Image Recognition"  . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.
Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Model weights in this example - vgg19.npy : https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe model to TensorFlow's : https://github.com/ethereon/caffe-tensorflow
Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""
from ..model import BaseModule
from ...utils.registry import Registers

import os
import numpy as np
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn import (BatchNorm, Conv2d, Dense, Flatten, SequentialLayer, MaxPool2d, Dropout)
from ...utils.output import BaseModelOutput
from dataclasses import dataclass
from .config_vgg import VGGModelConfig

__all__ = ['VGG']

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]


@dataclass
class VGGModelOutput(BaseModelOutput):
    ...


def make_layers(layer_config, config):
    layer_list = []
    is_end = False
    for layer_group_idx, layer_group in enumerate(layer_config):
        if isinstance(layer_group, list):
            for idx, layer in enumerate(layer_group):
                layer_name = layer_names[layer_group_idx][idx]
                n_filter = layer
                if idx == 0:
                    if layer_group_idx > 0:
                        in_channels = layer_config[layer_group_idx - 2][-1]
                    else:
                        in_channels = 3
                else:
                    in_channels = layer_group[idx - 1]
                layer_list.append(
                    Conv2d(
                        n_filter=n_filter, filter_size=(3, 3), strides=(1, 1), act=tlx.ReLU, padding='SAME',
                        in_channels=in_channels, name=layer_name
                    )
                )
                if config.batch_norm:
                    layer_list.append(BatchNorm(num_features=n_filter))
                if idx < (len(layer_group) - 1):
                    layer_list.append(Dropout(0.7))
                if layer_name == config.end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx]
            if layer_group == 'M':
                layer_list.append(MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name=layer_name))
            elif layer_group == 'O':
                layer_list.append(Dense(n_units=1000, in_channels=config.fc2_units, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Dropout(0.7))
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(
                    Dense(n_units=config.fc1_units, act=tlx.ReLU, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(
                    Dense(n_units=config.fc2_units, act=tlx.ReLU, in_channels=config.fc1_units, name=layer_name))
            if layer_name == config.end_with:
                is_end = True
        if is_end:
            break
    return SequentialLayer(layer_list)


@Registers.models.register
class VGG(BaseModule):
    config_class = VGGModelConfig

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)

        self.end_with = config.end_with = kwargs.pop("end_with", config.end_with)
        self.batch_norm = config.batch_norm = kwargs.pop("batch_norm", config.batch_norm)

        super(VGG, self).__init__(config, **kwargs)

        self.make_layer = make_layers(config.layers, config)

    def forward(self, pixels):
        """
        pixel : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        # pixels = pixels * 255 - np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape([1, 1, 1, 3])
        out = self.make_layer(pixels)
        return VGGModelOutput(output=out)

