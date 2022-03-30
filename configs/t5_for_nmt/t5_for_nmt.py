from tlxzoo.dataset import DataLoaders, ConditionalGeneration
import tensorlayerx as tlx
import time
import tensorflow as tf
from tlxzoo.utils.metrics import bleu
from tlxzoo.models.t5.feature_t5 import T5FeatureConfig, T5Feature

t5_feat_config = T5FeatureConfig(vocab_file="./spiece.model", source_max_length=128, label_max_length=128)
t5_feat = T5Feature(t5_feat_config)

data_config = ConditionalGeneration(per_device_train_batch_size=8,
                                    per_device_eval_batch_size=1,
                                    data_name="Text2Text",
                                    source_train_path="./giga-fren.release2.fixed.en",
                                    target_train_path="./giga-fren.release2.fixed.fr",
                                    source_dev_path="newstest2014-fren-en.txt",
                                    target_dev_path="newstest2014-fren-fr.txt",
                                    num_workers=8)
data_loaders = DataLoaders(data_config, train_limit=8)

data_loaders.register_transform_hook(t5_feat)

from tlxzoo.models.t5 import *

t5_model_config = T5Config()
t5_task_config = T5ForConditionalGenerationTaskConfig(t5_model_config)

t5 = T5ForConditionalGeneration.from_pretrained(config=t5_task_config,
                                                pretrained_base_path="./tf_model.h5",
                                                weight_from="huggingface")
optimizer = tlx.optimizers.Adam(learning_rate=0.0001)

loss_fn = t5.loss_fn
metric = None


def valid_bleu(model, test_dataset):
    model.set_eval()
    targets = []
    predictions = []
    for index, (X_batch, y_batch) in enumerate(test_dataset):
        decode_id = model.generate_one(inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"])
        decode_str = t5_feat.ids_to_string(decode_id[0])
        label_str = t5_feat.ids_to_string(y_batch["labels"][0])
        targets.append(label_str)
        predictions.append(decode_str)

    print(bleu(targets, predictions))

# 41.06
valid_bleu(t5, data_loaders.test)


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
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                else:
                    train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))

        valid_bleu(network, test_dataset)


model = Trainer(network=t5, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=1, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
            print_train_batch=True)
