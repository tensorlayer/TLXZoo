from tlxzoo.dataset import DataLoaders, AutomaticSpeechRecognitionDataConfig
from tlxzoo.models.wav2vec2 import *
from jiwer import wer
from tqdm import tqdm

feat_config = Wav2Vec2FeatureConfig(vocab_file="./vocab.json")
feat = Wav2Vec2Feature(feat_config)

data_config = AutomaticSpeechRecognitionDataConfig(per_device_train_batch_size=1,
                                                   per_device_eval_batch_size=1,
                                                   train_path="./LibriSpeech/train-clean-100/",
                                                   test_path="./LibriSpeech/dev-clean/",
                                                   num_workers=0)

data_loaders = DataLoaders(data_config, train_limit=2, collate_fn=feat.collate_fn)
data_loaders.register_transform_hook(feat)

model_config = Wav2Vec2ModelConfig()
task_config = Wav2Vec2ForAutomaticSpeechRecognitionTaskConfig(model_config)

# wav2vec2 = Wav2Vec2ForAutomaticSpeechRecognition(config=task_config)
wav2vec2 = Wav2Vec2ForAutomaticSpeechRecognition.from_pretrained(config=task_config,
                                                                 pretrained_base_path="./tf_model.h5",
                                                                 weight_from="huggingface")

optimizer = tlx.optimizers.Adam(learning_rate=0.0001)
metric = None
loss_fn = wav2vec2.loss_fn


def valid(model, test_dataset):
    model.set_eval()
    targets = []
    predictions = []
    print(f"length test_dataset: {len(test_dataset)}")
    for index, (X_batch, y_batch) in enumerate(tqdm(test_dataset)):
        logits = model(**X_batch)
        predicted_ids = tlx.argmax(logits, axis=-1)
        for predicted_id, text in zip(predicted_ids, y_batch["texts"]):
            transcription = feat.ids_to_string(predicted_id)
            predictions.append(transcription)
            targets.append(text)
    error = wer(targets, predictions)
    print(error)


# valid(wav2vec2, data_loaders.test)


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
                    _loss_ce = loss_fn(_logits, X_batch["pixel_mask"], y_batch["labels"])

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

        valid(wav2vec2, data_loaders.test)


model = Trainer(network=wav2vec2, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=1, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
            print_train_batch=True)

