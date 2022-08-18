import tensorlayerx as tlx
from tensorlayerx import nn
from tensorlayerx.nn.core.common import str2act
import numpy as np
from typing import Tuple


class ConvBlock(nn.Module):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, name="", **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = nn.Conv2d(out_channels=filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  stride=(1, 1),
                                  padding=padding,
                                  W_init=_get_kernel_initializer(filters, kernel_size),
                                  )
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.activation_1 = str2act(activation)()

        self.conv2d_2 = nn.Conv2d(out_channels=filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  stride=(1, 1),
                                  padding=padding,
                                  W_init=_get_kernel_initializer(filters, kernel_size),
                                  )
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.activation_2 = str2act(activation)()

    def forward(self, inputs):
        x = inputs
        x = self.conv2d_1(x)

        x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        x = self.dropout_2(x)

        x = self.activation_2(x)
        return x


class UpconvBlock(nn.Module):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx = layer_idx
        self.filters_root = filters_root
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

        filters = _get_filter_count(layer_idx + 1, filters_root)

        self.upconv = nn.ConvTranspose2d(out_channels=filters // 2,
                                         kernel_size=(pool_size, pool_size),
                                         stride=(pool_size, pool_size),
                                         padding=padding,
                                         W_init=_get_kernel_initializer(filters, kernel_size), )

        self.activation_1 = str2act(activation)()

    def forward(self, inputs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x


class CropConcatBlock(nn.Module):

    def forward(self, x, down_layer):
        x1_shape = tlx.get_tensor_shape(down_layer)
        x2_shape = tlx.get_tensor_shape(x)

        height_diff = (x1_shape[1] - x2_shape[1]) // 2
        width_diff = (x1_shape[2] - x2_shape[2]) // 2

        down_layer_cropped = down_layer[:, height_diff: (x2_shape[1] + height_diff),
                             width_diff: (x2_shape[2] + width_diff),
                             :]

        x = tlx.concat([down_layer_cropped, x], axis=-1)
        return x


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return tlx.initializers.truncated_normal(stddev=stddev)


class Unet(nn.Module):
    def __init__(self, nx=172, ny=172, channels=1, num_classes=2, layer_depth=3, filters_root=64, kernel_size=3,
                 pool_size=2, dropout_rate=0.5, padding="valid", activation="relu", name="unet"):
        """
        :param nx: (:obj:`int`, defaults to 172):
            (Optional) image size on x-axis
        :param ny: (:obj:`int`, defaults to 172):
            (Optional) image size on y-axis
        :param channels: (:obj:`int`, defaults to 1):
            number of channels of the input tensors
        :param num_classes: (:obj:`int`, defaults to 2):
            number of classes
        :param layer_depth: (:obj:`int`, defaults to 3):
            total depth of unet
        :param filters_root: (:obj:`int`, defaults to 64):
            number of filters in top unet layer
        :param kernel_size: (:obj:`int`, defaults to 3):
            size of convolutional layers
        :param pool_size: (:obj:`int`, defaults to 2):
            size of maxplool layers
        :param dropout_rate: (:obj:`float`, `optional`, defaults to 0.5):
            The dropout ratio for the layer.
        :param padding: (:obj:`str`, `optional`, defaults to valid):
            padding to be used in convolutions
        :param activation:(:obj:`str`, `optional`, defaults to :obj:`"relu"`):
            The non-linear activation function.
        """
        super(Unet, self).__init__(name=name)

        conv_params = dict(filters_root=filters_root,
                           kernel_size=kernel_size,
                           dropout_rate=dropout_rate,
                           padding=padding,
                           activation=activation)

        self.layer_depth = layer_depth

        conv_blocks = []
        max_pools = []
        for layer_idx in range(0, layer_depth - 1):
            conv_blocks.append(ConvBlock(layer_idx, **conv_params))
            max_pools.append(nn.MaxPool2d(kernel_size=(pool_size, pool_size)))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.max_pools = nn.ModuleList(max_pools)

        self.layer_idx = layer_depth - 2
        self.conv_block = ConvBlock(self.layer_idx + 1, **conv_params)

        upconv_blocks = []
        crop_concat_blocks = []
        conv_blocks2 = []
        for layer_idx in range(self.layer_idx, -1, -1):
            upconv_blocks.append(UpconvBlock(layer_idx,
                                             filters_root,
                                             kernel_size,
                                             pool_size,
                                             padding,
                                             activation))
            crop_concat_blocks.append(CropConcatBlock())
            conv_blocks2.append(ConvBlock(layer_idx, **conv_params))

        self.upconv_blocks = nn.ModuleList(upconv_blocks)
        self.crop_concat_blocks = nn.ModuleList(crop_concat_blocks)
        self.conv_blocks2 = nn.ModuleList(conv_blocks2)

        self.conv_2d = nn.Conv2d(out_channels=num_classes,
                                 kernel_size=(1, 1),
                                 stride=(1, 1),
                                 padding=padding,
                                 W_init=_get_kernel_initializer(filters_root, kernel_size))

        self.act = str2act(activation)()
        self.softmax = nn.Softmax()

        if nx is not None:
            self.build((2, nx, ny, channels))

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)

    def forward(self, inputs):
        x = inputs
        contracting_layers = {}

        for layer_idx in range(0, self.layer_depth - 1):
            x = self.conv_blocks[layer_idx](x)
            contracting_layers[layer_idx] = x
            x = self.max_pools[layer_idx](x)

        x = self.conv_block(x)

        for layer_idx in range(self.layer_idx, -1, -1):
            x = self.upconv_blocks[layer_idx](x)
            x = self.crop_concat_blocks[layer_idx](x, contracting_layers[layer_idx])
            x = self.conv_blocks2[layer_idx](x)

        x = self.conv_2d(x)

        x = self.act(x)
        # outputs = self.softmax(x)
        return x


def crop_to_shape(data, shape: Tuple[int, int, int]):
    """
    Crops the array to the given image shape by removing the border

    :param data: the array to crop, expects a tensor of shape [batches, nx, ny, channels]
    :param shape: the target shape [batches, nx, ny, channels]
    """
    diff_nx = (data.shape[0] - shape[0])
    diff_ny = (data.shape[1] - shape[1])

    if diff_nx == 0 and diff_ny == 0:
        return data

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[0] == shape[0]
    assert cropped.shape[1] == shape[1]
    return cropped


def crop_labels_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return image, crop_to_shape(label, shape)
    return crop


def crop_image_and_label_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return crop_to_shape(image, shape), \
               crop_to_shape(label, shape)
    return crop


def to_rgb(img: np.array):
    """
    Converts the given array into a RGB image and normalizes the values to [0, 1).
    If the number of channels is less than 3, the array is tiled such that it has 3 channels.
    If the number of channels is greater than 3, only the first 3 channels are used

    :param img: the array to convert [bs, nx, ny, channels]

    :returns img: the rgb image [bs, nx, ny, 3]
    """
    img = img.astype(np.float32)
    img = np.atleast_3d(img)

    channels = img.shape[-1]
    if channels == 1:
        img = np.tile(img, 3)

    elif channels == 2:
        img = np.concatenate((img, img[..., :1]), axis=-1)

    elif channels > 3:
        img = img[..., :3]

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)

    return img


class UnetTransform(object):
    def __init__(
            self,
            image_size=(172, 172),
            label_size=(132, 132, 2),
            **kwargs
    ):
        super(UnetTransform, self).__init__()

        self.label_size = label_size
        self.image_size = image_size
        self.crop = crop_labels_to_shape(label_size)
        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_eval(self):
        self.is_train = False

    def __call__(self, image, label):
        image, label = self.crop(image, label)
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        return image, label
