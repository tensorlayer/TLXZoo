import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
from tensorlayerx.optimizers.lr import LRScheduler
from tensorlayerx.vision.transforms import Compose
from tqdm import tqdm

from tlxzoo.datasets import CocoHumanPoseEstimationDataset
from tlxzoo.module.hrnet import *
from tlxzoo.vision.human_pose_estimation import HumanPoseEstimation


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
            for batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(batch['image'])
                    _loss_ce = loss_fn(_logits, target=batch['target'], target_weight=batch['target_weight'])

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                _, avg_accuracy, _, _ = pck(network_output=_logits, target=batch['target'])
                train_acc += avg_accuracy

                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc: {}".format(train_acc / n_iter))
                    print("   learning rate: ", optimizer.lr().numpy())

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc: {}".format(train_acc / n_iter))

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
    transform = Compose([
        Gather(),
        Crop(),
        Resize((256, 256)),
        Normalize(),
        GenerateTarget()
    ])
    train_dataset = CocoHumanPoseEstimationDataset(root='./data/coco2017', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataset = CocoHumanPoseEstimationDataset(root='./data/coco2017', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    model = HumanPoseEstimation("hrnet")

    scheduler = EpochDecay(1e-3)
    optimizer = tlx.optimizers.Adam(lr=scheduler)
    # optimizer = tlx.optimizers.SGD(lr=scheduler)

    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=None)
    trainer.train(n_epoch=80, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=False)

    model.save_weights("./demo/vision/human_pose_estimation/hrnet/model.npz")