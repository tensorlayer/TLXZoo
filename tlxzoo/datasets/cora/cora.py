import os
import numpy as np
import scipy.sparse as sp
from tensorlayerx.dataflow import Dataset


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class CoraDataset(Dataset):
    def __init__(self, root_path, split="train"):
        idx_features_labels = np.genfromtxt(
            os.path.join(root_path, "cora.content"), dtype=np.dtype(str)
        )
        features = sp.csr_matrix(
            idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(
            os.path.join(root_path, "cora.cites"), dtype=np.int32
        )
        edges = np.array(
            list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
        ).reshape(edges_unordered.shape)
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = normalize(features)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        if split == "train":
            idx = slice(140)
        elif split == "val":
            idx = slice(200, 500)
        else:
            idx = slice(500, 1500)

        self.features = np.array(features.todense())
        self.labels = np.where(labels)[1][idx]
        self.adj = np.array(adj.todense()).astype(np.float32)
        self.mask = np.zeros_like(labels, dtype=np.int)
        self.mask[idx, :] = 1

    def __getitem__(self, index):
        return (self.features, self.adj, self.mask), self.labels

    def __len__(self):
        return 1
