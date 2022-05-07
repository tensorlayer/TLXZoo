from tlxzoo.datasets import DataLoaders
from tlxzoo.module.unet import UnetTransform
from tlxzoo.vision.image_segmentation import ImageSegmentation, Accuracy, val
import tensorlayerx as tlx


if __name__ == '__main__':
    circles = DataLoaders("Circles", per_device_train_batch_size=2, per_device_eval_batch_size=2)
    transform = UnetTransform()

    circles.register_transform_hook(transform)

    model = ImageSegmentation(backbone="unet")

    optimizer = tlx.optimizers.Adam(1e-3)
    metrics = Accuracy()
    n_epoch = 15

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics)
    trainer.train(n_epoch=n_epoch, train_dataset=circles.train, test_dataset=circles.test, print_freq=1,
                  print_train_batch=False)

    val(model, circles.test)

    model.save_weights("./demo/vision/image_segmentation/unet/model.npz")



