import copy
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * tlx.sigmoid(x)


def stochastic_depth(input, p: float, mode: str, training: bool = True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(
            f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(
            f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = np.random.binomial(1, survival_rate, size).astype('float32')
    noise = tlx.convert_to_tensor(noise)
    if survival_rate > 0.0:
        noise = noise / survival_rate
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.is_train)


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., tlx.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``tlx.nn.BatchNorm2d``
        activation_layer (Callable[..., tlx.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``tlx.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.GroupConv2d(
                out_channels,
                (kernel_size, kernel_size),
                (stride, stride),
                groups,
                None,
                'SAME',
                dilation=(dilation, dilation),
                W_init='he_normal',
                b_init='zeros'
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer())
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., tlx.nn.Module], optional): ``delta`` activation. Default: ``tlx.nn.ReLU``
        scale_activation (Callable[..., tlx.nn.Module]): ``sigma`` activation. Default: ``tlx.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(squeeze_channels, (1, 1), padding='VALID',
                             W_init='he_normal', b_init='zeros')
        self.fc2 = nn.Conv2d(input_channels, (1, 1), padding='VALID',
                             W_init='he_normal', b_init='zeros')
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = SiLU

        # expand
        expanded_channels = cnf.adjust_channels(
            cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels,
                      squeeze_channels, activation=SiLU))

        # project
        layers.append(
            ConvNormActivation(
                cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_labels: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_labels (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError(
                "The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(
            cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * \
                    float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                num_labels,
                W_init=nn.initializers.random_uniform(
                    -1.0 / math.sqrt(num_labels), 1.0 / math.sqrt(num_labels)),
                b_init='zeros'
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def build(self, inputs_shape):
        ones = tlx.ones(inputs_shape)
        _ = self(ones)


def _efficientnet(width_mult, depth_mult, dropout, input_shape, **kwargs):
    inverted_residual_setting = [
        MBConvConfig(1, 3, 1, 32, 16, 1, width_mult, depth_mult),
        MBConvConfig(6, 3, 2, 16, 24, 2, width_mult, depth_mult),
        MBConvConfig(6, 5, 2, 24, 40, 2, width_mult, depth_mult),
        MBConvConfig(6, 3, 2, 40, 80, 3, width_mult, depth_mult),
        MBConvConfig(6, 5, 1, 80, 112, 3, width_mult, depth_mult),
        MBConvConfig(6, 5, 2, 112, 192, 4, width_mult, depth_mult),
        MBConvConfig(6, 3, 1, 192, 320, 1, width_mult, depth_mult),
    ]
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    model.build((1, input_shape, input_shape, 3))
    return model


def efficientnet(arch, **kwargs):
    if arch == 'efficientnet_b0':
        return _efficientnet(1.0, 1.0, 0.2, 224, norm_layer=partial(nn.BatchNorm2d, epsilon=1e-5, momentum=0.1), **kwargs)
    elif arch == 'efficientnet_b1':
        return _efficientnet(1.0, 1.1, 0.2, 240, norm_layer=partial(nn.BatchNorm2d, epsilon=1e-5, momentum=0.1), **kwargs)
    elif arch == 'efficientnet_b2':
        return _efficientnet(1.1, 1.2, 0.3, 260, norm_layer=partial(nn.BatchNorm2d, epsilon=1e-5, momentum=0.1), **kwargs)
    elif arch == 'efficientnet_b3':
        return _efficientnet(1.2, 1.4, 0.3, 300, norm_layer=partial(nn.BatchNorm2d, epsilon=1e-5, momentum=0.1), **kwargs)
    elif arch == 'efficientnet_b4':
        return _efficientnet(1.4, 1.8, 0.4, 380, norm_layer=partial(nn.BatchNorm2d, epsilon=1e-5, momentum=0.1), **kwargs)
    elif arch == 'efficientnet_b5':
        return _efficientnet(1.6, 2.2, 0.4, 456, norm_layer=partial(nn.BatchNorm2d, epsilon=0.001, momentum=0.01), **kwargs)
    elif arch == 'efficientnet_b6':
        return _efficientnet(1.8, 2.6, 0.5, 528, norm_layer=partial(nn.BatchNorm2d, epsilon=0.001, momentum=0.01), **kwargs)
    elif arch == 'efficientnet_b7':
        return _efficientnet(2.0, 3.1, 0.5, 600, norm_layer=partial(nn.BatchNorm2d, epsilon=0.001, momentum=0.01), **kwargs)
    else:
        raise ValueError(f"tlxzoo don`t support {arch}")
