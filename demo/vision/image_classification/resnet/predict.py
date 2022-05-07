from tlxzoo.datasets import DataLoaders
from tlxzoo.vision.transforms import BaseVisionTransform
from tlxzoo.vision.image_classification import ImageClassification
import tensorlayerx as tlx
import cv2

if __name__ == '__main__':
    transform = BaseVisionTransform(do_resize=False, do_normalize=True, mean=(120.70748, 120.70748, 120.70748),
                                    std=(64.150024, 64.150024, 64.150024))

    model = ImageClassification(backbone="resnet50", num_labels=10, input_shape=(2, 32, 32, 3))
    model.load_weights("./model.npz")
    model.set_eval()

    image = cv2.imread("dog.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))

    transform.set_eval()
    image, _ = transform(image, None)
    image = tlx.convert_to_tensor([image])

    print(model.predict(image))



