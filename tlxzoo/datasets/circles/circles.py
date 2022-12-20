import numpy as np
from tensorlayerx.dataflow import Dataset


class CirclesDataset(Dataset):
    def __init__(
            self, num, nx=172, ny=172, transform=None
    ):
        self.nx = nx
        self.ny = ny
        self.num = num
        self.transform = transform

    def __getitem__(self, index: int):
        image, mask = _create_image_and_mask(self.nx, self.ny)
        label = np.empty((self.nx, self.ny, 2))
        label[..., 0] = ~mask
        label[..., 1] = mask
        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def __len__(self) -> int:
        return self.num


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