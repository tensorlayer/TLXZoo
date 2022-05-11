from tlxzoo.datasets import DataLoaders
from tlxzoo.module.t5 import T5Transform
import tensorlayerx as tlx
from tlxzoo.text.text_token_classidication import TextTokenClassification


def valid(model, test_dataset):
    model.set_eval()
    metrics = tlx.metrics.Accuracy()
    val_acc = 0
    n_iter = 0
    for index, (X_batch, y_batch) in enumerate(test_dataset):
        _logits = model(inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"])
        for logit, y in zip(_logits, y_batch["labels"]):
            mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
            mask_sum = int(tlx.reduce_sum(mask))
            y = y[:mask_sum]
            logit = logit[:mask_sum]
            metrics.update(logit, y)
        val_acc += metrics.result()
        metrics.reset()
        n_iter += 1

    print(val_acc / n_iter)


class Trainer(tlx.model.Model):
    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import time
        import tensorflow as tf
        import numpy as np
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
                    for logit, y in zip(_logits, y_batch["labels"]):
                        mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
                        mask_sum = int(tlx.reduce_sum(mask))
                        y = y[:mask_sum]
                        logit = logit[:mask_sum]
                        metrics.update(logit, y)
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


if __name__ == '__main__':
    datasets = DataLoaders(per_device_train_batch_size=12,
                           per_device_eval_batch_size=12,
                           data_name="Conll2003",
                           path="./data",
                           task="ner",
                           num_workers=0)

    transform = T5Transform(vocab_file="./demo/text/nmt/t5/spiece.model", prefix="", task="token",
                            source_max_length=256, label_max_length=256)

    datasets.register_transform_hook(transform)
    print("train data:", len(datasets.train))

    model = TextTokenClassification("t5", n_class=9)
    model.load_weights("./demo/text/token_classification/t5/model.npz")

    optimizer = tlx.optimizers.Adam(lr=0.0001)

    loss_fn = model.loss_fn
    metric = tlx.metrics.Accuracy()

    trainer = Trainer(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    # 0.9720519
    trainer.train(n_epoch=5, train_dataset=datasets.train, test_dataset=datasets.test, print_freq=1,
                  print_train_batch=True)
    model.save_weights("./demo/text/token_classification/t5/model.npz")
