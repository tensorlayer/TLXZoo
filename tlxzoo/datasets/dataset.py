from tensorlayerx.dataflow import IterableDataset
import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Normalize, Compose


class BaseDataSetInfoMixin:
    ...


class BaseDataSetDictLoadMixin:
    @classmethod
    def load(cls):
        ...


class BaseDataSet(IterableDataset, BaseDataSetInfoMixin):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform
        super(BaseDataSet, self).__init__()

    def __iter__(self):
        for i in range(len(self.data)):
            data = self.data[i].astype('float32')
            data = self.transform(data)
            label = self.label[i].astype('int64')
            yield data, label


class BaseDataSetDict(dict, BaseDataSetDictLoadMixin):
    ...


class ImageNetDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls):
        ...


class CoCoNetDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls):
        ...


class MnistNetDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls):
        x_train, y_train, x_val, y_val, x_test, y_test = tlx.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
        # transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='HWC')])
        transform = lambda x: x

        return cls({"train": BaseDataSet(x_train, y_train, transform), "val": BaseDataSet(x_val, y_val, transform),
                    "test": BaseDataSet(x_test, y_test, transform)})




