from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
import numpy as np


class Circles(Dataset, BaseDataSetMixin):
    def __init__(
            self, num, nx=172, ny=172, transforms=None, limit=None,
    ):
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

        super(Circles, self).__init__()

        if limit:
            num = limit

        self.nx = nx
        self.ny = ny
        self.num = num

    def __getitem__(self, index: int):

        image, mask = _create_image_and_mask(self.nx, self.ny)
        label = np.empty((self.nx, self.ny, 2))
        label[..., 0] = ~mask
        label[..., 1] = mask

        image, label = self.transform(image, label)

        return image, label

    def __len__(self) -> int:
        return self.num


def _build_samples(sample_count: int, nx: int, ny: int, **kwargs):
    images = np.empty((sample_count, nx, ny, 1))
    labels = np.empty((sample_count, nx, ny, 2))
    for i in range(sample_count):
        image, mask = _create_image_and_mask(nx, ny, **kwargs)
        images[i] = image
        labels[i, ..., 0] = ~mask
        labels[i, ..., 1] = mask
    return images, labels


def _create_image_and_mask(nx, ny, cnt=10, r_min=3, r_max=10, border=32, sigma=20):
    image = np.ones((nx, ny, 1))
    mask = np.zeros((nx, ny), dtype=np.bool)

    for _ in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1, 255)

        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)

        image[m] = h

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    return image, mask


@Registers.datasets.register("Circles")
class CirclesDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        return cls({"train": Circles(1000, limit=train_limit),
                    "test": Circles(100)})

    def get_image_segmentation_schema_dataset(self, dataset_type, config=None):

        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset
