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


import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn import (BatchNorm, Conv2d, Linear, Flatten, Sequential, MaxPool2d, Dropout, Module)

__all__ = [
    'VGG',
]

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F',
            'fc1', 'fc2', 'O'
        ],
    'E':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ],
}

mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

model_urls = {
    'vgg16': 'http://www.cs.toronto.edu/~frossard/vgg16/',
    'vgg19': 'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/'
}

model_saved_name = {'vgg16': 'vgg16_weights.npz', 'vgg19': 'vgg19.npy'}


class VGG(Module):

    def __init__(self, layer_type, batch_norm=True, end_with='outputs', num_labels=1000, name=None):
        """
        VGG19 model
        :param layer_type: str
            One of vgg11,vgg13,vgg16,vgg19
        :param batch_norm: boolean
            Whether use batch norm
        :param end_with: str
            The end point of the model. Default ``fc3_relu`` i.e. the whole model.
        :param num_labels: str
            Number of classes to classify images
        :param name: str
            Module name
        """
        super(VGG, self).__init__(name=name)
        self.end_with = end_with

        config = cfg[mapped_cfg[layer_type]]
        self.dropout = tlx.nn.Dropout(0.3)
        self.make_layer = make_layers(config, batch_norm, end_with)
        self.batch_norm = tlx.nn.BatchNorm(num_features=512)
        self.classifier = tlx.nn.Linear(out_features=num_labels, in_features=512, name="classifier")

    def forward(self, inputs):
        out = self.make_layer(inputs)
        last_out = self.batch_norm(out)
        last_out = self.dropout(last_out)

        logits = self.classifier(last_out)
        return logits


def make_layers(layer_config, batch_norm=False, end_with='outputs', fc1_units=512, fc2_units=512):
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
                        out_channels=n_filter, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME',
                        in_channels=in_channels, name=layer_name
                    )
                )
                if batch_norm:
                    layer_list.append(BatchNorm(num_features=n_filter, gamma_init="ones", moving_var_init="ones"))
                if idx < (len(layer_group) - 1):
                    layer_list.append(Dropout(0.3))
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[layer_group_idx]
            if layer_group == 'M':
                # padding="valid", strides=None
                layer_list.append(MaxPool2d(kernel_size=(2, 2), padding='valid', name=layer_name))
                # layer_list.append(MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name=layer_name))
            elif layer_group == 'O':
                layer_list.append(Linear(out_features=1000, in_features=fc2_units, name=layer_name))
            elif layer_group == 'F':
                layer_list.append(Dropout(0.3))
                layer_list.append(Flatten(name='flatten'))
            elif layer_group == 'fc1':
                layer_list.append(
                    Linear(out_features=fc1_units, act=tlx.ReLU, in_features=512, name=layer_name))
            elif layer_group == 'fc2':
                layer_list.append(
                    Linear(out_features=fc2_units, act=tlx.ReLU, in_features=fc1_units, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return Sequential(layer_list)


