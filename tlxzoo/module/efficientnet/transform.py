import cv2
import numpy as np


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


class EfficientnetTransform(object):
    def __init__(self, backbone, **kwargs):
        super(EfficientnetTransform, self).__init__(**kwargs)

        self.size = input_shapes[backbone]
        self.mean = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.std = np.array([0.485, 0.456, 0.406], dtype=np.float32)

        self.is_train = True

    def set_train(self):
        self.is_train = True

    def set_eval(self):
        self.is_train = False

    def __call__(self, image, label):
        image = cv2.resize(image, self.size)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, label
