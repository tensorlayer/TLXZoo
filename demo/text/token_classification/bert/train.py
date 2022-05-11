from tlxzoo.datasets import DataLoaders
from tlxzoo.module.bert.transform import BertTransform
import tensorlayerx as tlx
from tlxzoo.text.text_token_classidication import TextTokenClassification


def valid(model, test_dataset):
    model.set_eval()
    metrics = tlx.metrics.Accuracy()
    val_acc = 0
    n_iter = 0
    for index, (X_batch, y_batch) in enumerate(test_dataset):
        _logits = model(**X_batch)
        for logit, y in zip(_logits, y_batch["labels"]):
            mask = tlx.cast(tlx.not_equal(y, -100), dtype=tlx.int32)
            mask_sum = int(tlx.reduce_sum(mask))
            y = y[1:mask_sum+1]
            logit = logit[1:mask_sum+1]
            metrics.update(logit, y)
        val_acc += metrics.result()
        metrics.reset()
        n_iter += 1

    print(val_acc / n_iter)


def load_huggingface_tf_weight(mode, weight_path):
    import h5py
    file = h5py.File(weight_path, "r")

    for w in mode.all_weights:
        name = w.name
        coder = name.split("/")[0]
        huggingface_weight_name = f"{coder}/tf_bert_for_pre_training/" + name
        huggingface_weight_name = huggingface_weight_name.replace("query/weights", "query/kernel")
        huggingface_weight_name = huggingface_weight_name.replace("key/weights", "key/kernel")
        huggingface_weight_name = huggingface_weight_name.replace("value/weights", "value/kernel")
        huggingface_weight_name = huggingface_weight_name.replace("dense/weights", "dense/kernel")
        huggingface_weight_name = huggingface_weight_name.replace("biases:0", "bias:0")
        if huggingface_weight_name not in file:
            print(huggingface_weight_name)
            continue
        w.assign(file[huggingface_weight_name])

    return mode


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
                        y = y[1:mask_sum+1]
                        logit = logit[1:mask_sum+1]
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
            model.save_weights("./demo/text/token_classification/bert/model.npz")


if __name__ == '__main__':
    datasets = DataLoaders(per_device_train_batch_size=16,
                           per_device_eval_batch_size=16,
                           data_name="Conll2003",
                           path="./data",
                           task="ner",
                           num_workers=0)

    transform = BertTransform(vocab_file="./demo/text/text_classification/bert/vocab.txt", task="token", max_length=128)

    datasets.register_transform_hook(transform)
    print("train data:", len(datasets.train))

    model = TextTokenClassification("bert", n_class=9)
    model.load_weights("./demo/text/token_classification/bert/model.npz")

    optimizer = tlx.optimizers.Adam(lr=0.00001)

    loss_fn = model.loss_fn
    metric = tlx.metrics.Accuracy()

    trainer = Trainer(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=3, train_dataset=datasets.train, test_dataset=datasets.test, print_freq=1,
                  print_train_batch=True)
    model.save_weights("./demo/text/token_classification/bert/model.npz")
