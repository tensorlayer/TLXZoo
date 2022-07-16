import cv2
import tensorlayerx as tlx
from tlxzoo.module.efficientnet import EfficientnetTransform
from tlxzoo.vision.image_classification import ImageClassification


if __name__ == '__main__':
    transform = EfficientnetTransform('efficientnet_b0')
    transform.set_eval()

    model = ImageClassification(backbone='efficientnet_b0', num_labels=1000)
    model.load_weights("./model.npz")
    model.set_eval()

    image = cv2.imread("dog.png")
    image, _ = transform(image, None)
    image = tlx.convert_to_tensor([image])

    print(model.predict(image))
