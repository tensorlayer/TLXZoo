import glob
import os

from scipy.io import loadmat
from tensorlayerx.dataflow import Dataset
from tensorlayerx.vision.utils import load_image


class ImagenetDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        if split == 'train':
            image_dir = os.path.join(root_path, 'ILSVRC2012_img_train')
            meta_filename = os.path.join(
                root_path, 'ILSVRC2012_devkit_t12/data/meta.mat')

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
            image_dir = os.path.join(root_path, 'ILSVRC2012_img_val')
            val_gt_filename = os.path.join(
                root_path, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt')

            self.filenames = glob.glob(os.path.join(image_dir, '*'))
            self.filenames.sort()
            with open(val_gt_filename) as f:
                labels = f.read().split('\n')[:-1]
            self.labels = [int(label)-1 for label in labels]

        self.transform = transform

    def __getitem__(self, index):
        image = load_image(self.filenames[index])
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.filenames)
