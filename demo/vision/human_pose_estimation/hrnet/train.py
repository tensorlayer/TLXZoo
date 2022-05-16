from tlxzoo.datasets import DataLoaders
import tensorlayerx as tlx
from tlxzoo.vision.human_pose_estimation import HumanPoseEstimation
from tlxzoo.module.hrnet import HRNetTransform, PCK
from tqdm import tqdm
from tensorlayerx.optimizers.lr import LRScheduler
pck = PCK()


def valid(model, test_data):
    model.set_eval()
    accs = []
    for idx, batch in enumerate(tqdm(test_data)):
        images = batch[0]
        target, target_weight = batch[1]["target"]
        y_pred = model(images)
        _, avg_accuracy, _, _ = pck(network_output=y_pred, target=target)
        if avg_accuracy == 0:
            continue
        accs.append(avg_accuracy)

    print((sum(accs) * 1.0) / len(accs))


class Trainer(tlx.model.Model):
    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import tensorflow as tf
        import time
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    _loss_ce = loss_fn(_logits, target=y_batch["target"][0], target_weight=y_batch["target"][1])

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                _, avg_accuracy, _, _ = pck(network_output=_logits, target=y_batch["target"][0])
                train_acc += avg_accuracy

                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc: {}".format(train_acc / n_iter))
                    print("   learning rate: ", optimizer.lr())

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc: {}".format(train_acc / n_iter))

            if (epoch + 1) % 5 == 0:
                valid(network, test_dataset)
                model.save_weights("./demo/vision/human_pose_estimation/hrnet/model.npz")

            optimizer.lr.step()


class EpochDecay(LRScheduler):
    def __init__(self, learning_rate, last_epoch=0, verbose=False):
        super(EpochDecay, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):

        if int(self.last_epoch) >= 65:
            return self.base_lr * 0.01

        if int(self.last_epoch) >= 40:
            return self.base_lr * 0.1

        return self.base_lr


if __name__ == '__main__':
    datasets = DataLoaders(root_path="./",
                           per_device_eval_batch_size=14,
                           per_device_train_batch_size=14,
                           data_name="Coco",
                           train_ann_path="./annotations/person_keypoints_train2017.json",
                           val_ann_path="./annotations/person_keypoints_val2017.json",
                           num_workers=4)

    transform = HRNetTransform()
    datasets.register_transform_hook(transform)

    model = HumanPoseEstimation("hrnet")

    scheduler = EpochDecay(1e-3)
    optimizer = tlx.optimizers.Adam(lr=scheduler)
    # optimizer = tlx.optimizers.SGD(lr=scheduler)

    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=None)
    trainer.train(n_epoch=80, train_dataset=datasets.train, test_dataset=datasets.test, print_freq=1,
                  print_train_batch=True)

    model.save_weights("./demo/vision/human_pose_estimation/hrnet/model.npz")