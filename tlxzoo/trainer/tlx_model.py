import tensorlayerx as tlx
import time
import numpy as np


class TLXModel(tlx.model.Model):
    def tf_train(
        self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
        print_freq, test_dataset
    ):
        import tensorflow as tf
        clip_epochs = self.clip_epochs
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    # compute loss and update model
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)
                if clip_epochs:
                    if clip_epochs[0] < epoch < clip_epochs[1]:
                        grad = optimizer._clip_gradients(grad)
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
                    print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))
                    print("   train acc:  {}".format(train_acc / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))
                print("   train acc:  {}".format(train_acc / n_iter))

            if test_dataset:
                # use training and evaluation sets to evaluate the model every print_freq epoch
                if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                    network.set_eval()
                    val_loss, val_acc, n_iter = 0, 0, 0
                    for X_batch, y_batch in test_dataset:
                        _logits = network(X_batch)  # is_train=False, disable dropout
                        val_loss += loss_fn(_logits, y_batch)
                        if metrics:
                            metrics.update(_logits, y_batch)
                            val_acc += metrics.result()
                            metrics.reset()
                        else:
                            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
                        n_iter += 1
                    print("   val loss: {}".format(val_loss / n_iter))
                    print("   val acc:  {}".format(val_acc / n_iter))