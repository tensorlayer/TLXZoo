import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets.circles import CirclesDataset
from tlxzoo.module.unet import UnetTransform
from tlxzoo.vision.image_segmentation import Accuracy, ImageSegmentation, val


if __name__ == '__main__':
    transform = UnetTransform()
    train_dataset = CirclesDataset(1000, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataset = CirclesDataset(100, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    model = ImageSegmentation(backbone="unet")

    optimizer = tlx.optimizers.Adam(1e-3)
    metrics = Accuracy()
    n_epoch = 15

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics)
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=False)

    val(model, test_dataloader)

    model.save_weights("./demo/vision/image_segmentation/unet/model.npz")



