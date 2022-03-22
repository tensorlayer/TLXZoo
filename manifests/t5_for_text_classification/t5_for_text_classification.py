from tlxzoo.dataset import DataLoaders, TextClassificationDataConfig
import tensorlayerx as tlx
import time
import tensorflow as tf
from tlxzoo.models.t5.feature_t5 import T5FeatureConfig, T5Feature

t5_feat_config = T5FeatureConfig(vocab_file="./spiece.model", prefix="ss2 sentence: ", source_max_length=128,
                                 label_max_length=128)
t5_feat = T5Feature(t5_feat_config)

data_config = TextClassificationDataConfig(per_device_train_batch_size=24,
                                           per_device_eval_batch_size=24,
                                           path="./data",
                                           num_workers=8)
data_loaders = DataLoaders(data_config)
data_loaders.register_transform_hook(t5_feat)

from tlxzoo.models.t5 import *

t5_model_config = T5Config()
t5_task_config = T5ForTextClassificationTaskConfig(t5_model_config)

t5 = T5ForTextClassification.from_pretrained(config=t5_task_config,
                                             pretrained_base_path="./tf_model.h5",
                                             weight_from="huggingface")
optimizer = tlx.optimizers.Adam(learning_rate=0.0001)

loss_fn = t5.loss_fn
metric = tlx.metrics.Accuracy()


def valid(model, test_dataset):
    model.set_eval()
    metrics = tlx.metrics.Accuracy()
    val_acc = 0
    n_iter = 0
    for index, (X_batch, y_batch) in enumerate(test_dataset):
        _logits = model(inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"])
        metrics.update(_logits, y_batch["labels"])
        val_acc += metrics.result()
        metrics.reset()
        n_iter += 1

    print(val_acc / n_iter)


# valid(t5, data_loaders.test)


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
                    _logits = network(**X_batch)
                    # _loss_ce = tf.reduce_mean(loss_fn(_logits, y_batch))
                    _loss_ce = loss_fn(_logits, **y_batch)

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch["labels"])
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            valid(network, test_dataset)


model = Trainer(network=t5, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=3, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
            print_train_batch=False)
