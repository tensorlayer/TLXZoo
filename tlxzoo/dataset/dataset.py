from tensorlayerx import logging
from tensorlayerx.dataflow import Dataset, IterableDataset
import tensorlayerx as tlx
import random
import os
from ..utils.registry import Registers
from .data_random import DataRandom


class BaseDataSetMixin:

    def register_transform_hook(self, transform_hook, index=None):
        if index is None:
            self.transforms.append(transform_hook)
        if not isinstance(index, int):
            raise ValueError("{index} is not int.")
        self.transforms.insert(index, transform_hook)

    def transform(self, data, label):
        for transform in self.feature_transforms:
            data, label = transform(data, label)
        return data, label

    def validate(self, schema):
        index = random.randint(0, len(self) - 1)
        data = self[index]
        data = {j: i for i, j in zip(data, schema.schema_type)}
        assert schema.validate(data)
        logging.info(f"{data} pass validation")


class BaseDataSet(Dataset, BaseDataSetMixin):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        super(BaseDataSet, self).__init__()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        return self.transform(data, label)

    def __len__(self):
        return len(self.data)


class FileDataSet(IterableDataset, BaseDataSetMixin):
    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        super(FileDataSet, self).__init__()

    def __iter__(self):

        for data, label in zip(open(self.data), open(self.label)):
            data = data.strip()
            label = label.strip()
            data, label = self.transform(data, label)
            yield data, label


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


def classify_label_transform(data, label):
    return data, label.astype('int64')


@Registers.datasets.register("Mnist")
class MnistDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        x_train, y_train, x_val, y_val, x_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        label_transform = classify_label_transform

        return cls({"train": BaseDataSet(x_train, y_train, transforms=[label_transform]),
                    "val": BaseDataSet(x_val, y_val, transforms=[label_transform]),
                    "test": BaseDataSet(x_test, y_test, transforms=[label_transform])})

    def get_image_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
            if config is not None:
                data_random_hook = DataRandom(random_rotation_degrees=config.random_rotation_degrees,
                                              random_shift=config.random_shift,
                                              random_flip_horizontal_prop=config.random_flip_horizontal_prop,
                                              random_crop_size=config.random_crop_size)
                dataset.register_transform_hook(data_random_hook, index=-1)
        elif dataset_type == "eval":
            dataset = self["eval"]
        else:
            dataset = self["test"]

        return dataset


@Registers.datasets.register("Cifar10")
class Cifar10DataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        x_train, y_train, x_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        label_transform = classify_label_transform

        return cls({"train": BaseDataSet(x_train, y_train, transforms=[label_transform]),
                    "test": BaseDataSet(x_test, y_test, transforms=[label_transform])})

    def get_image_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
            if config is not None:
                data_random_hook = DataRandom(random_rotation_degrees=config.random_rotation_degrees,
                                              random_shift=config.random_shift,
                                              random_flip_horizontal_prop=config.random_flip_horizontal_prop,
                                              random_crop_size=config.random_crop_size)
                dataset.register_transform_hook(data_random_hook, index=-1)
        else:
            dataset = self["test"]

        return dataset


@Registers.datasets.register("WmtEnfr")
class WmtEnfrDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        from tensorlayerx.files.dataset_loaders.wmt_en_fr_dataset import load_wmt_en_fr_dataset
        train_path, dev_path = load_wmt_en_fr_dataset()

        return cls({"train": FileDataSet(train_path + ".en", train_path + ".fr"),
                    "test": FileDataSet(dev_path + ".en", dev_path + ".fr")})

    def get_conditional_generation_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset
