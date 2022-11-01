import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor
from tensorlayerx.vision.utils import load_image
from tlxzoo.vision.image_classification import ImageClassification


if __name__ == '__main__':
    input_shapes = {
        'efficientnet_b0': (224, 224),
        'efficientnet_b1': (240, 240),
        'efficientnet_b2': (260, 260),
        'efficientnet_b3': (300, 300),
        'efficientnet_b4': (380, 380),
        'efficientnet_b5': (456, 456),
        'efficientnet_b6': (528, 528),
        'efficientnet_b7': (600, 600),
    }

    model = ImageClassification(backbone='efficientnet_b0', num_labels=1000)
    model.load_weights(
        "./demo/vision/image_classification/efficientnet/model.npz")
    model.set_eval()

    image = load_image(
        "./demo/vision/image_classification/efficientnet/dog.png")
    transforms = Compose([
        Resize(input_shapes['efficientnet_b0']),
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        ToTensor()
    ])
    image = tlx.expand_dims(transforms(image), 0)

    print(model.predict(image))
