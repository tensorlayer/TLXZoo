from tlxzoo.datasets import DataLoaders
from tlxzoo.module.unet import UnetTransform, crop_image_and_label_to_shape
from tlxzoo.vision.image_segmentation import ImageSegmentation, Accuracy, val
import tensorlayerx as tlx
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    circles = DataLoaders("Circles", per_device_train_batch_size=2, per_device_eval_batch_size=3)
    transform = UnetTransform()
    circles.register_transform_hook(transform)

    model = ImageSegmentation(backbone="unet")
    model.load_weights("./demo/vision/image_segmentation/unet/model.npz")
    crop = crop_image_and_label_to_shape(transform.label_size)

    for i, j in circles.test:
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
        plt.savefig("circle.png")
        break