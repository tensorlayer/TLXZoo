from tlxzoo.datasets import DataLoaders
from tlxzoo.module.t5 import T5Transform
import tensorlayerx as tlx
from tlxzoo.text.text_conditional_generation import TextForConditionalGeneration
from tlxzoo.text.metrics import bleu


def valid_bleu(model, test_dataset, transform):
    from tqdm import tqdm
    model.set_eval()
    targets = []
    predictions = []
    for index, (X_batch, y_batch) in enumerate(tqdm(test_dataset)):
        decode_id = model.generate_one(inputs=X_batch["inputs"], attention_mask=X_batch["attention_mask"])
        decode_str = transform.ids_to_string(decode_id[0])
        label_str = transform.ids_to_string(y_batch["labels"][0])
        targets.append(label_str)
        predictions.append(decode_str)

    print(bleu(targets, predictions))


class Trainer(tlx.model.Model):
    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import time
        import tensorflow as tf
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(**X_batch)
                    _loss_ce = loss_fn(_logits, **y_batch)

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce
                if metrics:
                    metrics.update(_logits, y_batch)
                    train_acc += metrics.result()
                    metrics.reset()
                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))


if __name__ == '__main__':
    datasets = DataLoaders(per_device_train_batch_size=8,
                           per_device_eval_batch_size=1,
                           data_name="Text2Text",
                           source_train_path="./demo/text/nmt/t5/giga-fren.release2.fixed.en",
                           target_train_path="./demo/text/nmt/t5/giga-fren.release2.fixed.fr",
                           source_dev_path="./demo/text/nmt/t5/newstest2014-fren-en.txt",
                           target_dev_path="./demo/text/nmt/t5/newstest2014-fren-fr.txt",
                           num_workers=0, train_limit=16)
    transform = T5Transform(vocab_file="./demo/text/nmt/t5/spiece.model", source_max_length=128, label_max_length=128)
    datasets.register_transform_hook(transform)

    model = TextForConditionalGeneration("t5")

    model.load_weights("./demo/text/nmt/t5/model.npz")
    # valid_bleu(model, datasets.test, transform)

    optimizer = tlx.optimizers.Adam(lr=0.0001)
    loss_fn = model.loss_fn
    metric = None

    trainer = Trainer(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=1, train_dataset=datasets.train, test_dataset=datasets.test, print_freq=1,
                  print_train_batch=True)
    valid_bleu(model, datasets.test, transform)


