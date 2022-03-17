from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset
import time


def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images


def load_images():
    train_images, train_labels, test_images, test_labels = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3),
                                                                                          plotable=False)

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)

    return train_images, train_labels, test_images, test_labels


class Trainer(tlx.model.Model):
    def tf_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    # _loss_ce = tf.reduce_mean(loss_fn(_logits, y_batch))
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)
                if 0 < epoch < 80:
                    grad = optimizer._clip_gradients(grad)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

                if X_batch.shape[0] != 128:
                    break

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.set_eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        # y_batch = tf.cast(y_batch, dtype=tf.float32)
                        # val_loss += tf.reduce_mean(loss_fn(_logits, y_batch))
                        val_loss += loss_fn(_logits, y_batch)
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))

if __name__ == '__main__':
    print(tf.__version__)
    print(keras.__version__)

    # use_gpu()

    training_epochs = 800
    batch_size = 128
    learning_rate = 0.001
    tf.random.set_seed(777)

    train_images, train_labels, test_images, test_labels = load_images()

    data_generator = ImageDataGenerator(
        # brightness_range=[.2, .2],
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False  # randomly flip images
    )
    data_generator.fit(train_images)


    class mnistdataset(Dataset):

        def __init__(self, data=None, label=None, if_train=False):
            self.data = data
            self.label = label
            self.if_train = if_train

        def __getitem__(self, index):
            data = self.data[index].astype('float32')
            if self.if_train:
                data = data_generator.random_transform(data)
                data = data_generator.standardize(data)
            label = self.label[index]
            return data, label

        def __len__(self):
            return len(self.data)


    from tlxzoo.models.resnet import *

    resnet_model_config = ResNetModelConfig()
    resnet_task_config = ResNetForImageClassificationTaskConfig(resnet_model_config)
    resnet_task = ResNetForImageClassification(resnet_task_config)
    print(resnet_task)
    optimizer = tlx.optimizers.Adam(learning_rate, clipvalue=0.001)
    from tensorflow.python.keras.engine.training import *

    loss_fn = tlx.losses.softmax_cross_entropy_with_logits
    metric = tlx.metrics.Accuracy()

    test_dataset = mnistdataset(data=test_images, label=test_labels)
    # test_dataset = tlx.dataflow.FromGenerator(
    #     test_dataset, output_types=[tlx.float32, tlx.int64], column_names=['data', 'label']
    # )
    test_loader = tlx.dataflow.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = mnistdataset(data=train_images, label=train_labels, if_train=True)
    # train_dataset2 = tlx.dataflow.FromGenerator(
    #     train_dataset, output_types=[tlx.float32, tlx.int64], column_names=['data', 'label']
    # )
    train_loader = tlx.dataflow.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=12, prefetch_factor=batch_size)
    # train_loader = train_loader.prefetch(128)

    model = Trainer(network=resnet_task, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    model.train(n_epoch=training_epochs, train_dataset=train_loader, test_dataset=test_loader, print_freq=1,
                print_train_batch=False)
