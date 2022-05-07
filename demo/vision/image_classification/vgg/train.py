from tlxzoo.datasets import DataLoaders
from tlxzoo.vision.transforms import BaseVisionTransform
from tlxzoo.vision.image_classification import ImageClassification
import tensorlayerx as tlx


if __name__ == '__main__':
    cifar10 = DataLoaders("Cifar10", per_device_train_batch_size=128, per_device_eval_batch_size=128)
    transform = BaseVisionTransform(do_resize=False, do_normalize=True, mean=(125.31, 122.95, 113.86),
                                    std=(62.99, 62.09, 66.70))

    cifar10.register_transform_hook(transform)
    print(cifar10.dataset_dict["train"].transforms[-1].is_train)
    print(cifar10.dataset_dict["test"].transforms[-1].is_train)

    model = ImageClassification(backbone="vgg16", l2_weights=True, num_labels=10)

    optimizer = tlx.optimizers.Adam(0.0001)
    metric = tlx.metrics.Accuracy()

    n_epoch = 500

    trainer = tlx.model.Model(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metric=metric)
    trainer.train(n_epoch=n_epoch, train_dataset=cifar10.train, test_dataset=cifar10.test, print_freq=1,
                  print_train_batch=False)

    model.save_weights("./demo/vision/image_classification/vgg/model.npz")

