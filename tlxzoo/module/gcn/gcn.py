import tensorlayerx as tlx
from tensorlayerx import nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = tlx.random_uniform((in_features, out_features))
        if bias:
            self.bias = tlx.random_uniform((out_features,))

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

    def forward(self, x, adj):
        x = tlx.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return tlx.logsoftmax(x, dim=1)
