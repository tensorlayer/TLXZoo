import tensorlayerx as tlx
import numpy as np
import random
from .tlx_model import TLXModel


def get_loss_from_config(config):
    loss_fn = getattr(tlx.losses, config.loss)
    return loss_fn


def get_optimizer_from_config(config):
    optimizer = getattr(tlx.optimizers, config.optimizers)(**config.lr)
    return optimizer


def get_metric_from_config(config):
    metric = getattr(tlx.metrics, config.metric)()
    return metric


class Trainer(object):
    def __init__(self,
                 task,
                 data_loader,
                 config,
                 auto_config=False,
                 ):
        self.config = config
        self.task = task
        self.data_loader = data_loader
        self.auto_config = auto_config

        self.trainer = TLXModel(network=self.task)
        self.trainer.clip_epochs = self.config.clip_epochs
        self.register_hooks_from_config(self.config)

        self.feature = None

        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def register_logger_hook(self, log_hook):
        ...

    def register_evaluator_hook(self, eval_hook):
        self.trainer.metrics = eval_hook

    def register_optimizer_hook(self, optimizer_hook):
        self.trainer.optimizer = optimizer_hook

    def register_loss_hook(self, loss_hook):
        self.trainer.loss_fn = loss_hook

    def register_feature_transform_hook(self, feature_transform_hook):
        self.feature = feature_transform_hook
        self.data_loader.register_feature_transform_hook(feature_transform_hook, index=0)

    def register_hooks_from_config(self, config):
        loss_fn = get_loss_from_config(config)
        self.register_loss_hook(loss_fn)

        optimizer = get_optimizer_from_config(config)
        self.register_optimizer_hook(optimizer)

        metric = get_metric_from_config(config)
        self.register_evaluator_hook(metric)

    def train(self, n_epoch, print_freq, print_train_batch):
        if hasattr(self.task, "loss_fn"):
            self.register_loss_hook(self.task.loss_fn)

        if self.data_loader.test:
            self.trainer.train(n_epoch=n_epoch, train_dataset=self.data_loader.train,
                               test_dataset=self.data_loader.test,
                               print_freq=print_freq,
                               print_train_batch=print_train_batch)
        else:
            self.trainer.train(n_epoch=n_epoch, train_dataset=self.data_loader.train, print_freq=print_freq,
                               print_train_batch=print_train_batch)

    def save_task(self, path):
        self.task.save_pretrained(path)

    def save_config(self, path):
        self.config.save_pretrained(path)

    def save_feature(self, path):
        self.feature.save_pretrained(path)

    def save(self, path):
        self.save_task(path)
        self.save_config(path)
        self.save_feature(path)


