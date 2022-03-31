import tensorlayerx as tlx


def cdist(box_a, box_b):
    A = tlx.get_tensor_shape(box_a)[0]  # Number of bbox in box_a
    B = tlx.get_tensor_shape(box_b)[0]  # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tlx.tile(tlx.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tlx.tile(tlx.expand_dims(box_b, axis=0), [A, 1, 1])
    return tlx.reduce_sum(tlx.abs(tiled_box_a - tiled_box_b), axis=-1)


class GroupNorm(tlx.nn.Module):
    def __init__(self, c, name=None):
        super(GroupNorm, self).__init__(name=name)
        self.c = c

        self.gamma = self._get_weights(var_name="gamma", shape=[1, 1, 1, c], init=tlx.initializers.ones())
        self.beta = self._get_weights(var_name="beta", shape=[1, 1, 1, c], init=tlx.initializers.zeros())

    def forward(self, x, g=8, eps=1e-5):
        n, h, w, c = tlx.get_tensor_shape(x)
        g = tlx.minimum(g, c)

        x = tlx.reshape(x, [n, h, w, g, c // g])
        mean, var = tlx.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tlx.sqrt(var + eps)

        x = tlx.reshape(x, [n, h, w, c]) * self.gamma + self.beta
        return x


if tlx.ops.BACKEND == "tensorflow":

    import tensorflow as tf

    einsum = tf.einsum
elif tlx.ops.BACKEND == "pytorch":
    import torch
    einsum = torch.einsum
else:
    def einsum(equation, *inputs, **kwargs):
        raise ValueError(f"einsum do not support {tlx.ops.BACKEND}")