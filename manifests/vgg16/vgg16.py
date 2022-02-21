#! /usr/bin/python
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
os.environ['TL_BACKEND'] = 'tensorflow'
import tensorlayerx as tlx
import tensorflow as tf
from tensorlayerx.dataflow import Dataset
from tlxzoo.models.vgg.task_vgg import VGGForImageClassification
from tlxzoo.models.vgg.config_vgg import *

from tlxzoo.config.config import BaseImageFeatureConfig
from tlxzoo.models.vgg.feature_vgg import VGGFeature
import numpy as np
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

image_feat_config = BaseImageFeatureConfig(do_resize=False, do_normalize=True, mean=(125.31, 122.95, 113.86), std=(62.99, 62.09, 66.70))
vgg_feature = VGGFeature(image_feat_config)

X_train, y_train, X_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# def normalization(train_images, test_images):
#     mean = np.mean(train_images, axis=(0, 1, 2, 3))
#     std = np.std(train_images, axis=(0, 1, 2, 3))
#     train_images = (train_images - mean) / (std + 1e-7)
#     test_images = (test_images - mean) / (std + 1e-7)
#     return train_images, test_images
#
#
# X_train = X_train.astype(np.float32)
# X_test = X_test.astype(np.float32)
#
# (X_train, X_test) = normalization(X_train, X_test)


class mnistdataset(Dataset):

    def __init__(self, data=X_train, label=y_train, if_train=False, feature_transform=None):
        self.data = data
        self.label = label
        self.feature_transform = feature_transform
        self.if_train = if_train
        self.RandomRotation = tlx.vision.transforms.RandomRotation(degrees=15, interpolation='bilinear', expand=False, center=None, fill=0)
        self.RandomShift = tlx.vision.transforms.RandomShift(shift=(0.1, 0.1), interpolation='bilinear', fill=0)
        self.RandomFlipHorizontal = tlx.vision.transforms.RandomFlipHorizontal(prob=0.5)
        self.RandomCrop = tlx.vision.transforms.RandomCrop(32, 4)

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        data = self.feature_transform([data])[0]
        if self.if_train:
            data = self.RandomRotation(data)
            data = self.RandomShift(data)
            data = self.RandomFlipHorizontal(data)
            data = self.RandomCrop(data)
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)

    def register_feature_transform_hook(self, feature_transform_hook):
        self.feature_transform = feature_transform_hook


vgg16_model_config = VGGModelConfig()
vgg16_task_config = VGGForImageClassificationTaskConfig(vgg16_model_config, num_labels=10)
vgg16_task = VGGForImageClassification(vgg16_task_config)
print(vgg16_task)
n_epoch = 250 * 2
batch_size = 128
print_freq = 1

train_weights = vgg16_task.trainable_weights
li_regularizer = tlx.losses.li_regularizer(0.00001)

l2_weights = []
l2_names = []
for w in train_weights:
    print(w.name,w.shape)
    if w.name.startswith("conv") and len(w.shape) >= 2:
        l2_weights.append(w)
        l2_names.append(w.name)
    if w.name.startswith("fc1") and len(w.shape) >= 2:
        l2_weights.append(w)
        l2_names.append(w.name)
print("l2:",len(l2_weights))
print(l2_names)
optimizer = tlx.optimizers.Adam(0.0001)
metric = tlx.metrics.Accuracy()

def loss_function(output, target):
    loss = tlx.losses.softmax_cross_entropy_with_logits(output, target)
    loss2 = 0
    for w in l2_weights:
        loss2 += li_regularizer(w)
    return loss + loss2
loss_fn = loss_function

train_dataset = mnistdataset(data=X_train, label=y_train, if_train=True)
train_dataset.register_feature_transform_hook(vgg_feature)
train_dataset2 = tlx.dataflow.FromGenerator(
    train_dataset, output_types=[tlx.float32, tlx.int64], column_names=['data', 'label']
)
train_loader = tlx.dataflow.Dataloader(train_dataset2, batch_size=batch_size, shuffle=True)


test_dataset = mnistdataset(data=X_test, label=y_test)
test_dataset.register_feature_transform_hook(vgg_feature)

test_dataset = tlx.dataflow.FromGenerator(
    test_dataset, output_types=[tlx.float32, tlx.int64], column_names=['data', 'label']
)
test_loader = tlx.dataflow.Dataloader(test_dataset, batch_size=batch_size, shuffle=True)

vgg16_task.set_train()
model = tlx.model.Model(network=vgg16_task, loss_fn=loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=n_epoch, train_dataset=train_loader, test_dataset=test_loader, print_freq=print_freq, print_train_batch=False)

