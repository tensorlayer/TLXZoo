import glob
import os

import cv2
from scipy.io import loadmat

from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, BaseDataSetMixin, Dataset


class Imagenet(Dataset, BaseDataSetMixin):
    def __init__(self, image_dir, meta_filename, val_gt_filename, transforms=None, limit=None):
        if meta_filename:
            meta = loadmat(meta_filename)['synsets']
            indexes = {}
            for i in range(1000):
                indexes[meta[i][0][1][0]] = i

            self.filenames = []
            self.labels = []
            for category in os.listdir(image_dir):
                label = indexes[category]
                for filename in glob.glob(os.path.join(image_dir, category, '*')):
                    self.filenames.append(filename)
                    self.labels.append(label)
        else:
            self.filenames = glob.glob(os.path.join(image_dir, '*'))
            self.filenames.sort()
            with open(val_gt_filename) as f:
                labels = f.read().split('\n')[:-1]
            self.labels = [int(label)-1 for label in labels]

        if limit:
            self.filenames = self.filenames[:limit]
            self.labels = self.labels[:limit]

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = []

    def __getitem__(self, index):
        image = cv2.imread(self.filenames[index])
        label = self.labels[index]
        return self.transform(image, label)

    def __len__(self):
        return len(self.filenames)


@Registers.datasets.register("Imagenet")
class ImagenetDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, root_path, train_limit=None):
        return cls({
            "train": Imagenet(os.path.join(root_path, 'ILSVRC2012_img_train'),
                              os.path.join(
                                  root_path, 'ILSVRC2012_devkit_t12/data/meta.mat'),
                              None, limit=train_limit),
            "test": Imagenet(os.path.join(root_path, 'ILSVRC2012_img_val'), None,
                             os.path.join(root_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'))
        })
