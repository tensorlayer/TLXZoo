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
    train_dataset = CoraDataset(root_path="./data/cora", split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataset = CoraDataset(root_path="./data/cora", split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    model = NodeClassification(backbone="gcn", nfeat=1433, nclass=7)

    optimizer = tlx.optimizers.Adam(0.01, weight_decay=5e-4)
    metric = tlx.metrics.Accuracy()
    n_epoch = 200

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric
    )
    trainer.train(
        n_epoch=n_epoch,
        train_dataset=train_dataloader,
        test_dataset=val_dataloader,
        print_freq=1,
        print_train_batch=False,
    )

    model.save_weights("./demo/graph/node_classification/gcn/model.npz")
