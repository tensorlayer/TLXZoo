import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets import CoraDataset
from tlxzoo.graph.node_classification import NodeClassification


if __name__ == "__main__":
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
