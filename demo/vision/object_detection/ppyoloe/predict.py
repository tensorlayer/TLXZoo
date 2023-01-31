import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose
from tensorlayerx.vision.utils import load_image

from tlxzoo.module.ppyoloe import *
from tlxzoo.vision.object_detection import ObjectDetection


if __name__ == '__main__':
    model = ObjectDetection(backbone="ppyoloe_s", num_classes=80, data_format='channels_first')
    model.load_weights("demo/vision/object_detection/ppyoloe/model.npz")
    model.set_eval()
    model.backbone.set_eval()

    image_path = "demo/vision/object_detection/detr/000000039769.jpeg"
    image = load_image(image_path)
    transform = Compose([
        ResizeImage(
            target_size=640
        ),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False
        ),
        Permute(
            to_bgr=False,
            channel_first=True
        )
    ])
    image = transform({'image': image})['image']

    inputs = tlx.convert_to_tensor([image])

    scale_factor = tlx.convert_to_tensor([[1, 1]], dtype=tlx.float32)
    outputs = model(inputs=inputs, scale_factor=scale_factor)

    print(outputs)
