from tensorlayerx import logging
from tensorlayerx.dataflow import Dataset
import tensorlayerx as tlx
import random
from ..utils.registry import Registers


class BaseDataSetInfoMixin:
    """get label class, data description and so on"""
    ...


class BaseDataSet(Dataset, BaseDataSetInfoMixin):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        super(BaseDataSet, self).__init__()

    def __getitem__(self, index):
        data = self.data[index].astype('float32')
        label = self.label[index].astype('int64')
        return data, label

    def __len__(self):
        return len(self.data)

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
    def load(cls):
        x_train, y_train, x_val, y_val, x_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

        return cls({"train": BaseDataSet(x_train, y_train), "val": BaseDataSet(x_val, y_val),
                    "test": BaseDataSet(x_test, y_test)})

    def get_image_classification_schema_dataset(self, dataset_type):
        if dataset_type == "train":
            dataset = self["train"]
        elif dataset_type == "eval":
            dataset = self["eval"]
        else:
            dataset = self["test"]

        # dataset.validate(image_classification_task_data_set_schema)
        return dataset



