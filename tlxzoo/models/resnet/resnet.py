import tensorlayerx as tlx
from ...utils.output import BaseModelOutput
from ..model import BaseModule
from ...utils.registry import Registers
from .config_resnet import ResNetModelConfig
from ...utils import glorot_uniform


@Registers.models.register
class ResNet(BaseModule):
    config_class = ResNetModelConfig

    def __init__(self,
                 config, **kwargs):
        if config is None:
            config = self.config_class(**kwargs)
        super(ResNet, self).__init__(config, **kwargs)

        num_layers = config.num_layers
        shortcut_connection = config.shortcut_connection
        weight_decay = config.weight_decay
        batch_norm_momentum = config.batch_norm_momentum
        batch_norm_epsilon = config.batch_norm_epsilon
        drop_rate = config.drop_rate

        if num_layers not in (20, 32, 44, 56, 110):
            raise ValueError('num_layers must be one of 20, 32, 44, 56 or 110.')

        self._num_layers = num_layers
        self._shortcut_connection = shortcut_connection
        self._weight_decay = weight_decay
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon

        self._num_units = (num_layers - 2) // 6

        self._init_conv = tlx.nn.Conv2d(16, in_channels=3, W_init=glorot_uniform, b_init=None, name="init_conv")
        self.dropout_1 = tlx.nn.Dropout(drop_rate)

        self._block1 = tlx.nn.SequentialLayer([ResNetUnit(
            16,
            1,
            shortcut_connection,
            True if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            'res_net_unit_1_%d' % (i + 1), drop_rate=drop_rate) for i in range(self._num_units)])

        self._block2 = tlx.nn.SequentialLayer([ResNetUnit(
            32,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            'res_net_unit_2_%d' % (i + 1), drop_rate=drop_rate) for i in range(self._num_units)])

        self._block3 = tlx.nn.SequentialLayer([ResNetUnit(
            64,
            2 if i == 0 else 1,
            shortcut_connection,
            False if i == 0 else False,
            weight_decay,
            batch_norm_momentum,
            batch_norm_epsilon,
            'res_net_unit_3_%d' % (i + 1), drop_rate=drop_rate) for i in range(self._num_units)])

        self._final_bn = tlx.nn.BatchNorm(decay=batch_norm_momentum, epsilon=batch_norm_epsilon, num_features=64,
                                          gamma_init="ones", moving_var_init="ones", name="final_bn")
        self.dropout_2 = tlx.nn.Dropout(drop_rate)

        # self._final_conv = tlx.nn.Conv2d(10, filter_size=(1, 1), strides=(1, 1), W_init=glorot_uniform,
        #                                  b_init="zeros", in_channels=64, name="final_conv")

    def forward(self, inputs):
        net = inputs
        net = self._init_conv(net)
        net = self.dropout_1(net)

        net = self._block1(net)
        net = self._block2(net)
        net = self._block3(net)

        net = self._final_bn(net)
        net = tlx.relu(net)
        net = self.dropout_2(net)
        net = tlx.reduce_mean(net, [1, 2], keepdims=True)
        # net = self._final_conv(net)
        # net = tlx.squeeze(net, axis=[1, 2])

        return BaseModelOutput(output=net)


class ResNetUnit(tlx.nn.Module):
    def __init__(self,
                 depth,
                 stride,
                 shortcut_connection,
                 shortcut_from_preact,
                 weight_decay,
                 batch_norm_momentum,
                 batch_norm_epsilon,
                 name, drop_rate=0.05):
        super(ResNetUnit, self).__init__(name=name)
        self._depth = depth
        self._stride = stride
        self._shortcut_connection = shortcut_connection
        self._shortcut_from_preact = shortcut_from_preact
        self._weight_decay = weight_decay

        self._bn1 = tlx.nn.BatchNorm(decay=batch_norm_momentum, epsilon=batch_norm_epsilon,
                                     num_features=depth if stride == 1 else int(depth / 2),
                                     gamma_init="ones", moving_var_init="ones", name="batchnorm_1")

        self.dropout_1 = tlx.nn.Dropout(drop_rate)

        self._conv1 = tlx.nn.Conv2d(depth, (3, 3), (stride, stride),
                                    in_channels=depth if stride == 1 else int(depth / 2),
                                    W_init=glorot_uniform, b_init=None, name="conv1")

        self._bn2 = tlx.nn.BatchNorm(decay=batch_norm_momentum, epsilon=batch_norm_epsilon,
                                     num_features=depth,
                                     gamma_init="ones", moving_var_init="ones", name="batchnorm_2")
        self.dropout_2 = tlx.nn.Dropout(drop_rate)

        self._conv2 = tlx.nn.Conv2d(depth, (3, 3), (1, 1), in_channels=depth if stride == 1 else int(depth / 2),
                                    W_init=glorot_uniform, b_init=None, name="conv2")

    def forward(self, inputs):
        depth_in = tlx.get_tensor_shape(inputs)[3]
        depth = self._depth
        preact = tlx.relu(self._bn1(inputs))
        preact = self.dropout_1(preact)

        shortcut = preact if self._shortcut_from_preact else inputs

        if depth != depth_in:
            shortcut = tlx.backend.ops.avg_pool(shortcut, (2, 2), strides=(1, 2, 2, 1), padding='SAME')
            shortcut = tlx.pad(shortcut, [[0, 0], [0, 0], [0, 0], [(depth - depth_in) // 2] * 2])

        net = self._conv1(preact)
        residual = tlx.relu(self._bn2(net))
        residual = self.dropout_2(residual)
        residual = self._conv2(residual)

        output = residual + shortcut if self._shortcut_connection else residual

        return output
