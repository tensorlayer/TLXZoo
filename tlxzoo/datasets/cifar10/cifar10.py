from ...utils.registry import Registers
from ..dataset import *
import tensorlayerx as tlx


def classify_label_transform(data, label):
    return data, label.astype('int64')


@Registers.datasets.register("Cifar10")
class Cifar10DataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, **kwargs):
        x_train, y_train, x_test, y_test = tlx.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
        if train_limit is not None:
            x_train = x_train[:train_limit]
            y_train = y_train[:train_limit]

        label_transform = classify_label_transform

        return cls({"train": BaseDataSet(x_train, y_train, transforms=[label_transform]),
                    "test": BaseDataSet(x_test, y_test, transforms=[label_transform])})
