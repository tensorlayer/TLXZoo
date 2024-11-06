import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets import CoraDataset
from tlxzoo.graph.node_classification import NodeClassification
import os


def device_info():
    found = False
    if not found and os.system("npu-smi info > /dev/null 2>&1") == 0:
        cmd = "npu-smi info"
        found = True
    elif not found and os.system("nvidia-smi > /dev/null 2>&1") == 0:
        cmd = "nvidia-smi"
        found = True
    elif not found and os.system("ixsmi > /dev/null 2>&1") == 0:
        cmd = "ixsmi"
        found = True
    elif not found and os.system("cnmon > /dev/null 2>&1") == 0:
        cmd = "cnmon"
        found = True
    
    os.system(cmd)
    cmd = "lscpu"
    os.system(cmd)
    
if __name__ == "__main__":
    device_info()
    model = NodeClassification(backbone="gcn", nfeat=1433, nclass=7)
    model.load_weights("./demo/graph/node_classification/gcn/model.npz")
    model.set_eval()

    test_dataset = CoraDataset(root_path="./data/cora", split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    inputs, labels = next(iter(test_dataloader))

    outputs = model.predict(inputs)
    print(tlx.count_nonzero(outputs == labels) /
          tlx.get_tensor_shape(labels)[1])
