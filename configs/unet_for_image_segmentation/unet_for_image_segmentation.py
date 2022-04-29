from tqdm import tqdm
from tlxzoo.dataset import DataLoaders, ImageSegmentationDataConfig
from tlxzoo.models.unet import *
from tlxzoo.models.unet.task_unet import mean_iou, dice_coefficient
import tensorlayerx as tlx

image_detection_config = ImageSegmentationDataConfig(num_workers=0)

data_loaders = DataLoaders(image_detection_config)

feat_config = UnetFeatureConfig()
feat = UnetFeature(feat_config)

data_loaders.register_transform_hook(feat)

# for i, j in data_loaders.train:
#     print(i)
#     print(j)
#     raise

model_config = UnetModelConfig()
task_config = UnetForImageSegmentationTaskConfig(model_config)

model = UnetForImageSegmentation(task_config)
metrics_acc = tlx.metrics.Accuracy()


def val(model, test_data):
    model.set_eval()
    auc_sum = 0
    mean_iou_sum = 0
    dice_coefficient_sum = 0
    num = 0
    for x, labels in tqdm(test_data):
        _logits = model(x)
        _logits = tlx.softmax(_logits)
        mean_iou_sum += mean_iou(labels, _logits)
        dice_coefficient_sum += dice_coefficient(labels, _logits)
        y_batch_argmax = tlx.argmax(labels, -1)
        y_batch_argmax = tlx.reshape(y_batch_argmax, [-1])
        _logits = tlx.reshape(_logits, [-1, tlx.get_tensor_shape(_logits)[-1]])
        metrics_acc.update(_logits, y_batch_argmax)
        auc_sum += metrics_acc.result()
        metrics_acc.reset()
        num += 1

    print(f"val_auc: {auc_sum / num}")
    print(f"val_mean_iou: {mean_iou_sum / num}")
    print(f"val_dice_coefficient: {dice_coefficient_sum / num}")


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
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                if metrics:
                    y_batch_argmax = tlx.argmax(y_batch, -1)
                    y_batch_argmax = tlx.reshape(y_batch_argmax, [-1])
                    _logits = tlx.reshape(_logits, [-1, tlx.get_tensor_shape(_logits)[-1]])
                    metrics.update(_logits, y_batch_argmax)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, -1), np.argmax(y_batch, -1)))

                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            val(network, test_dataset)


optimizer = tlx.optimizers.Adam(1e-3)
trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics_acc)
trainer.train(n_epoch=25, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
              print_train_batch=False)
