import tensorlayerx as tlx
from tensorlayerx import nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = self._get_weights(
            "weight",
            (in_features, out_features),
            tlx.initializers.random_uniform()
        )
        if bias:
            self.bias = self._get_weights(
                "bias",
                (out_features,),
                tlx.initializers.random_uniform()
            )

    def forward(self, input, adj):
        support = tlx.matmul(input, self.weight)
        output = tlx.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = tlx.nn.Dropout(dropout)

    def forward(self, inputs):
        x, adj, mask = inputs
        x = tlx.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        shape = tlx.get_tensor_shape(x)
        shape[1] = -1
        x = tlx.mask_select(x, tlx.equal(mask, 1))
        x = tlx.reshape(x, shape)
        return x
