from tensorlayerx import logging
from tensorlayerx.dataflow import Dataset
import tensorlayerx as tlx
import random
from ..utils.registry import Registers
from .data_random import DataRandom


class BaseDataSetInfoMixin:
    """get label class, data description and so on"""
    ...


class BaseDataSet(Dataset, BaseDataSetInfoMixin):
    def __init__(self, data, label, feature_transforms=None, label_transform=None):
        self.data = data
        self.label = label
        if feature_transforms is not None:
            self.feature_transforms = feature_transforms
        else:
            self.feature_transforms = []
        self.label_transform = label_transform
        self.random_transform_hook = None
        super(BaseDataSet, self).__init__()

    def __getitem__(self, index):
        data = self.data[index]
        if self.feature_transforms:
            for feature_transform in self.feature_transforms:
                data = feature_transform([data])[0]

        label = self.label[index]

        if self.random_transform_hook:
            data, label = self.random_transform_hook(data, label)

        if self.label_transform:
            label = self.label_transform(label)

        return data, label

    def __len__(self):
        return len(self.data)

    def register_feature_transform_hook(self, feature_transform_hook):
        # self.feature_transform = feature_transform_hook
        self.feature_transforms.append(feature_transform_hook)

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


def classify_label_transform(label):
    return label.astype('int64')


@Registers.datasets.register("Mnist")
class MnistDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        x_train, y_train, x_val, y_val, x_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        label_transform = classify_label_transform

        return cls({"train": BaseDataSet(x_train, y_train, label_transform=label_transform),
                    "val": BaseDataSet(x_val, y_val, label_transform=label_transform),
                    "test": BaseDataSet(x_test, y_test, label_transform=label_transform)})

    def get_image_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
            if config is not None:
                data_random_hook = DataRandom(random_rotation_degrees=config.random_rotation_degrees,
                                              random_shift=config.random_shift,
                                              random_flip_horizontal_prop=config.random_flip_horizontal_prop,
                                              random_crop_size=config.random_crop_size)
                dataset.register_random_transform_hook(data_random_hook)
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

        return cls({"train": BaseDataSet(x_train, y_train, label_transform=label_transform),
                    "test": BaseDataSet(x_test, y_test, label_transform=label_transform)})

    def get_image_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
            if config is not None:
                data_random_hook = DataRandom(random_rotation_degrees=config.random_rotation_degrees,
                                              random_shift=config.random_shift,
                                              random_flip_horizontal_prop=config.random_flip_horizontal_prop,
                                              random_crop_size=config.random_crop_size)
                dataset.register_random_transform_hook(data_random_hook)
        else:
            dataset = self["test"]

        return dataset
