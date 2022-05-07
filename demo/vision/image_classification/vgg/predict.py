from tlxzoo.datasets import DataLoaders
from tlxzoo.vision.transforms import BaseVisionTransform
from tlxzoo.vision.image_classification import ImageClassification
import tensorlayerx as tlx
import cv2

if __name__ == '__main__':
    # 0: airplane
    # 1: automobile
    # 2: bird
    # 3: cat
    # 4: deer
    # 5: dog
    # 6: frog
    # 7: horse
    # 8: ship
    # 9: truck
    transform = BaseVisionTransform(do_resize=False, do_normalize=True, mean=(125.31, 122.95, 113.86),
                                    std=(62.99, 62.09, 66.70))

    model = ImageClassification(backbone="vgg16", l2_weights=True, num_labels=10)
    model.load_weights("./demo/vision/image_classification/vgg/model.npz")
    model.set_eval()

    image = cv2.imread("dog.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))

    transform.set_eval()
    image, _ = transform(image, None)
    image = tlx.convert_to_tensor([image])

    print(model.predict(image))



