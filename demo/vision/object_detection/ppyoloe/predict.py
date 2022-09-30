import cv2
import tensorlayerx as tlx
from tlxzoo.module.ppyoloe import PPYOLOETransform
from tlxzoo.vision.object_detection import ObjectDetection

if __name__ == '__main__':
    transform = PPYOLOETransform()
    transform.set_eval()

    model = ObjectDetection(backbone="ppyoloe_s", num_classes=80, data_format='channels_first')
    model.load_weights("./model.npz")
    model.set_eval()
    model.backbone.set_eval()

    image_path = "./image.png"
    image = cv2.imread(image_path)

    image, _ = transform(image, None)

    inputs = tlx.convert_to_tensor([image])

    scale_factor = tlx.convert_to_tensor([[1, 1]], dtype=tlx.float32)
    outputs = model(inputs=inputs, scale_factor=scale_factor)

    print(outputs)
