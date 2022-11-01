import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose, Normalize, Resize, ToTensor
from tensorlayerx.vision.utils import load_image
from tlxzoo.vision.image_classification import ImageClassification

if __name__ == '__main__':
    model = ImageClassification(
        backbone="resnet50", num_labels=10, input_shape=(2, 32, 32, 3))
    model.load_weights("./demo/vision/image_classification/resnet/model.npz")
    model.set_eval()

    image = load_image("./demo/vision/image_classification/resnet/dog.png")
    transform = Compose([
        Resize((32, 32)),
        Normalize(mean=(120.70748, 120.70748, 120.70748),
                  std=(64.150024, 64.150024, 64.150024)),
        ToTensor()
    ])
    image = tlx.expand_dims(transform(image), 0)

    print(model.predict(image))
