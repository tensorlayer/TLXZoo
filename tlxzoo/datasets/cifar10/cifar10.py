from tensorlayerx.dataflow import Dataset
from tensorlayerx.files import load_cifar10_dataset


class Cifar10Dataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        x_train, y_train, x_test, y_test = load_cifar10_dataset(path=root_path)
        if split == 'train':
            self.data = x_train
            self.label = y_train
        else:
            self.data = x_test
            self.label = y_test
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)
