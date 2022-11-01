import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose, Normalize, ToTensor
from tlxzoo.datasets import Cifar10Dataset
from tlxzoo.vision.image_classification import ImageClassification

if __name__ == '__main__':
    transform = Compose([
        Normalize(mean=(125.31, 122.95, 113.86), std=(62.99, 62.09, 66.70)),
        ToTensor()
    ])
    train_dataset = Cifar10Dataset(
        root_path='./data', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    test_dataset = Cifar10Dataset(
        root_path='./data', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    model = ImageClassification(
        backbone="vgg16", l2_weights=True, num_labels=10)

    optimizer = tlx.optimizers.Adam(0.0001)
    metric = tlx.metrics.Accuracy()

    n_epoch = 500

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=False)

    model.save_weights("./demo/vision/image_classification/vgg/model.npz")
