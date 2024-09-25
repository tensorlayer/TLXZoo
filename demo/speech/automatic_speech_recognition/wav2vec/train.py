from tlxzoo.datasets import DataLoaders
from tlxzoo.module.wav2vec2 import Wav2Vec2Transform
from tlxzoo.speech.automatic_speech_recognition import AutomaticSpeechRecognition
import tensorlayerx as tlx


def valid(model, test_dataset, transform):
    from jiwer import wer
    from tqdm import tqdm
    model.set_eval()
    targets = []
    predictions = []
    print(f"length test_dataset: {len(test_dataset)}")
    for index, (X_batch, y_batch) in enumerate(tqdm(test_dataset)):
        logits = model(**X_batch)
        predicted_ids = tlx.argmax(logits, axis=-1)
        for predicted_id, text in zip(predicted_ids, y_batch["texts"]):
            transcription = transform.ids_to_string(predicted_id)
            predictions.append(transcription)
            targets.append(text)
    error = wer(targets, predictions)
    print(error)


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
                    _logits = network(**X_batch)
                    # _loss_ce = tf.reduce_mean(loss_fn(_logits, y_batch))
                    _loss_ce = loss_fn(_logits, y_batch["labels"], pixel_mask=X_batch["pixel_mask"])

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))


if __name__ == '__main__':
    transform = Wav2Vec2Transform(vocab_file="./demo/speech/automatic_speech_recognition/wav2vec/vocab.json")
    # download dataset from https://www.openslr.org/12
    libri_speech = DataLoaders("LibriSpeech",
                               train_path="./LibriSpeech/train-clean-100/",
                               test_path="./LibriSpeech/dev-clean/",
                               per_device_train_batch_size=1, per_device_eval_batch_size=1, num_workers=0,
                               collate_fn=transform.collate_fn)

    libri_speech.register_transform_hook(transform)

    model = AutomaticSpeechRecognition(backbone="wav2vec")

    optimizer = tlx.optimizers.Adam(lr=0.0001)
    metric = None
    loss_fn = model.loss_fn

    trainer = Trainer(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=1, train_dataset=libri_speech.train, test_dataset=libri_speech.test, print_freq=1,
                  print_train_batch=True)

    valid(model, libri_speech.test, transform)


