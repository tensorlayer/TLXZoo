from tlxzoo.datasets import Synth90kDataset
from tlxzoo.module.trocr.transform import TrOCRTransform
import tensorlayerx as tlx
from tlxzoo.vision.ocr import OpticalCharacterRecognition
from tqdm import tqdm
from jiwer import cer
from tensorlayerx.dataflow import DataLoader


def valid(model, test_dataset, limit=None):
    transform = TrOCRTransform(merges_file="./demo/vision/ocr/trocr/merges.txt",
                               vocab_file="./demo/vision/ocr/trocr/vocab.json", max_length=12)
    model.set_eval()
    targets = []
    predictions = []

    print(f"length test_dataset: {len(test_dataset)}")
    for index, (X_batch, y_batch) in enumerate(tqdm(test_dataset)):
        predicted_ids = model.generate_one(inputs=X_batch["inputs"], max_length=24)

        for predicted_id, text, input_id in zip(predicted_ids, y_batch[1], y_batch[0]["inputs"]):
            transcription = transform.ids_to_string(predicted_id)
            predictions.append(transcription)
            targets.append(text)
            print(transcription)
            print(text)
        if limit is not None and index >= limit:
            break
    error = cer(targets, predictions)
    print(f"cer:{error}")


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
                    attention_mask = y_batch[0]["attention_mask"]
                    length = tlx.reduce_max(tlx.reduce_sum(attention_mask, axis=-1))
                    length = int(length)
                    input_ids = y_batch[0]["inputs"][:, :length]
                    attention_mask = y_batch[0]["attention_mask"][:, :length]
                    # compute outputs
                    _logits = network(X_batch["inputs"], input_ids=input_ids,
                                      attention_mask=attention_mask)
                    # _loss_ce = tf.reduce_mean(loss_fn(_logits, y_batch))
                    _loss_ce = loss_fn(_logits, input_ids=input_ids,
                                       attention_mask=attention_mask)

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce
                n_iter += 1

                if print_train_batch:
                    if isinstance(print_train_batch, int):
                        if n_iter % print_train_batch == 0:
                            print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter,
                                                                     time.time() - start_time))
                            print("   train loss: {}".format(train_loss / n_iter))
                    else:
                        print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                        print("   train loss: {}".format(train_loss / n_iter))

                if n_iter >= 1 and n_iter % 100000 == 0:
                    valid(network, test_dataset, 5000)
                    model.save_weights("./demo/vision/ocr/trocr/model.npz")

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))

        model.save_weights("./demo/vision/ocr/trocr/model.npz")
        valid(network, test_dataset)


if __name__ == '__main__':
    transform = TrOCRTransform(merges_file="./demo/vision/ocr/trocr/merges.txt",
                               vocab_file="./demo/vision/ocr/trocr/vocab.json", max_length=12)
    train_dataset = Synth90kDataset(archive_path='./data/mjsynth/mnt/ramdisk/max/90kDICT32px/', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=8)
    test_dataset = Synth90kDataset(archive_path='./data/mjsynth/mnt/ramdisk/max/90kDICT32px/', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    model = OpticalCharacterRecognition("trocr")

    optimizer = tlx.optimizers.Adam(lr=0.00001)
    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=None)
    trainer.train(n_epoch=1, train_dataset=train_dataloader, test_dataset=test_dataloader, print_freq=1,
                  print_train_batch=False)
