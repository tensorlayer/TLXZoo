from tensorlayerx import logging
from tensorlayerx.dataflow import Dataset, IterableDataset
import tensorlayerx as tlx


class BaseDataSetMixin:
    def register_transform_hook(self, transform_hook, index=None):
        if index is None:
            self.transforms.append(transform_hook)
        else:
            if not isinstance(index, int):
                raise ValueError(f"{index} is not int.")
            self.transforms.insert(index, transform_hook)

    def transform(self, data, label):
        for transform in self.transforms:
            data, label = transform(data, label)
        return data, label


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
    def __init__(self, data, label, transforms=None, limit=None):
        self.data = data
        self.label = label
        self.limit = limit
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        super(FileDataSet, self).__init__()

    def __iter__(self):
        index = 0
        for data, label in zip(open(self.data), open(self.label)):
            if self.limit and index >= self.limit:
                break
            index += 1
            data = data.strip()
            label = label.strip()
            data, label = self.transform(data, label)
            yield data, label


class BaseDataSetDict(dict):
    @classmethod
    def load(cls):
        ...