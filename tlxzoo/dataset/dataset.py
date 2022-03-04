from tensorlayerx import logging
from tensorlayerx.dataflow import Dataset
import tensorlayerx as tlx
import numpy as np
import random
from ..utils.registry import Registers


class BaseDataSetInfoMixin:
    """get label class, data description and so on"""
    ...


class BaseDataSet(Dataset, BaseDataSetInfoMixin):
    def __init__(self, data, label, feature_transform=None):
        self.data = data
        self.label = label
        self.feature_transform = feature_transform
        self.random_transform_hook = None
        super(BaseDataSet, self).__init__()

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        if self.feature_transform:
            data = self.feature_transform([data])[0]
        if self.random_transform_hook:
            data = self.random_transform_hook(data)
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)

    def register_feature_transform_hook(self, feature_transform_hook):
        self.feature_transform = feature_transform_hook

    def register_random_transform_hook(self, random_transform_hook):
        self.random_transform_hook = random_transform_hook

    def validate(self, schema):
        index = random.randint(0, len(self) - 1)
        data = self[index]
        data = {j: i for i, j in zip(data, schema.schema_type)}
        assert schema.validate(data)
        logging.info(f"{data} pass validation")


class BaseDataSetDict(dict):
    @classmethod
    def load(cls):
        ...


class ImageNetDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls):
        ...


class CoCoDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls):
        ...


@Registers.datasets.register("Mnist")
class MnistDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None):
        x_train, y_train, x_val, y_val, x_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        return cls({"train": BaseDataSet(x_train, y_train), "val": BaseDataSet(x_val, y_val),
                    "test": BaseDataSet(x_test, y_test)})

    def get_image_classification_schema_dataset(self, dataset_type):
        if dataset_type == "train":
            dataset = self["train"]
        elif dataset_type == "eval":
            dataset = self["eval"]
        else:
            dataset = self["test"]

        return dataset


@Registers.datasets.register("Cifar10")
class Cifar10DataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None):
        x_train, y_train, x_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        return cls({"train": BaseDataSet(x_train, y_train),
                    "test": BaseDataSet(x_test, y_test)})

    def get_image_classification_schema_dataset(self, dataset_type):
        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset


@Registers.datasets.register("Coco")
class CocoDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, image_path, train_ann_path, val_ann_path):
        def load_ann(annot_path):
            with open(annot_path, "r") as f:
                txt = f.readlines()
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            np.random.shuffle(annotations)
            return annotations

        train_ann = load_ann(train_ann_path)


