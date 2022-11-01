import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor
from tensorlayerx.vision.utils import load_image
from tlxzoo.vision.image_classification import ImageClassification

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
    model = ImageClassification(
        backbone="vgg16", l2_weights=True, num_labels=10)
    model.load_weights("./demo/vision/image_classification/vgg/model.npz")
    model.set_eval()

    image = load_image("./demo/vision/image_classification/vgg/dog.png")
    transform = Compose([
        Resize((32, 32)),
        Normalize(mean=(125.31, 122.95, 113.86), std=(62.99, 62.09, 66.70)),
        ToTensor()
    ])
    image = tlx.expand_dims(transform(image), 0)

    print(model.predict(image))
