import time
import tensorlayerx as tlx
from tlxzoo.datasets import DataLoaders
from tlxzoo.module.ppyoloe import PPYOLOETransform
from tlxzoo.vision.object_detection import ObjectDetection


class Trainer(tlx.model.Model):
    def th_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                y_batch['epoch_id'] = epoch
                network.set_train()
                output = network(X_batch)
                loss = loss_fn(output, y_batch)
                grads = optimizer.gradient(loss, train_weights)
                optimizer.apply_gradients(zip(grads, train_weights))

                train_loss += loss.item()
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))


if __name__ == '__main__':
    transform = PPYOLOETransform(channel_first=True)
    coco = DataLoaders("Coco", per_device_train_batch_size=128, per_device_eval_batch_size=32,
                       root_path="./data/coco2017",
                       train_ann_path="./data/coco2017/annotations/instances_train2017.json",
                       val_ann_path="./data/coco2017/annotations/instances_val2017.json",
                       num_workers=0,
                       image_format='opencv'
                       )
    coco.register_transform_hook(transform)

    model = ObjectDetection(backbone="ppyoloe_s", num_classes=80, data_format='channels_first')

    optimizer = tlx.optimizers.SGD(lr=1e-3, momentum=0.9, weight_decay=5e-4)
    n_epoch = 300

    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=None)
    trainer.train(n_epoch=n_epoch, train_dataset=coco.train, test_dataset=coco.test, print_freq=1,
                  print_train_batch=False)

    model.save_weights('model.npz')