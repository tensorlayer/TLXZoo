import matplotlib.pyplot as plt
import numpy as np
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets.circles import CirclesDataset
from tlxzoo.module.unet import UnetTransform, crop_image_and_label_to_shape
from tlxzoo.vision.image_segmentation import ImageSegmentation


if __name__ == '__main__':
    transform = UnetTransform()
    test_dataset = CirclesDataset(100, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    model = ImageSegmentation(backbone="unet")
    model.load_weights("./demo/vision/image_segmentation/unet/model.npz")
    crop = crop_image_and_label_to_shape(transform.label_size)

    for i, j in test_dataloader:
        prediction = model.predict(i)
        fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
        for i, (image, label) in enumerate(zip(i, j)):
            image, label = crop(image, label)
            ax[i][0].matshow(image[..., -1])
            ax[i][0].set_title('Original Image')
            ax[i][0].axis('off')
            ax[i][1].matshow(np.argmax(label, axis=-1), cmap=plt.cm.gray)
            ax[i][1].set_title('Original Mask')
            ax[i][1].axis('off')
            ax[i][2].matshow(np.argmax(prediction[i, ...], axis=-1), cmap=plt.cm.gray)
            ax[i][2].set_title('Predicted Mask')
            ax[i][2].axis('off')
        plt.savefig("./demo/vision/image_segmentation/unet/circle.png")
        break