import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets import CoraDataset
from tlxzoo.graph.node_classification import NodeClassification


if __name__ == "__main__":
    model = NodeClassification(backbone="gcn", nfeat=1433, nclass=7)
    model.load_weights("./demo/graph/node_classification/gcn/model.npz")
    model.set_eval()

    test_dataset = CoraDataset(root_path="./data/cora", split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    inputs, labels = next(iter(test_dataloader))

    outputs = model.predict(inputs)
    print(tlx.count_nonzero(outputs == labels) /
          tlx.get_tensor_shape(labels)[1])
