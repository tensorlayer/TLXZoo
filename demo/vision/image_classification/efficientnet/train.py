import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor
from tlxzoo.datasets import ImagenetDataset
from tlxzoo.vision.image_classification import ImageClassification

if __name__ == '__main__':
    input_shapes = {
        'efficientnet_b0': (224, 224),
        'efficientnet_b1': (240, 240),
        'efficientnet_b2': (260, 260),
        'efficientnet_b3': (300, 300),
        'efficientnet_b4': (380, 380),
        'efficientnet_b5': (456, 456),
        'efficientnet_b6': (528, 528),
        'efficientnet_b7': (600, 600),
    }
    transform = Compose([
        Resize(input_shapes['efficientnet_b0']),
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        ToTensor()
    ])
    train_dataset = ImagenetDataset(
        root_path='data/imagenet', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    test_dataset = ImagenetDataset(
        root_path='data/imagenet', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128)

    model = ImageClassification(backbone='efficientnet_b0', num_labels=1000)

    optimizer = tlx.optimizers.RMSprop(0.0001, momentum=0.9, weight_decay=1e-5)
    metric = tlx.metrics.Accuracy()

    n_epoch = 300

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=False)

    model.save_weights(
        './demo/vision/image_classification/efficientnet/model.npz')
